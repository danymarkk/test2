"""ETH-only parameter sweep for ATR multiplier, RR ratio, and slip/comm.

Phase 2 diagnostics wiring:
- Exports per-config raw per-trade diagnostics CSV/JSON under results/sweeps/raw/
- Aggregates per-config metrics into results/sweeps/eth_param_sweep.csv and .md
"""

from __future__ import annotations

from itertools import product
import argparse
import os
import time
import logging
from typing import List, Dict, Tuple

import pandas as pd

from params import get_config, get_strategy_params, get_logger
from data_loader import load_data
from indicators_core import compute_vwap, compute_atr, compute_ema, generate_bias
from fair_value_gap import detect_fvgs
from liquidity_sweep import detect_liquidity_sweeps
from rejection_confirmation import confirm_rejection
from signal_engine import generate_signals
from execution_engine import simulate_trades
from diagnostics import reset as diag_reset, finalize_and_export as diag_export


SESSIONS = {
    "Asia": (0, 8),
    "London": (8, 16),
    "New York": (16, 24),
}


def _filter_session(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    hours = pd.to_datetime(df["timestamp"]).dt.hour
    return df.loc[(hours >= start) & (hours < end)].reset_index(drop=True)


def _parse_timeframe_to_minutes(tf: str) -> int:
    tf = str(tf).lower().strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Unsupported timeframe: {tf}")


def _load_last_hours_chunked(
    symbol: str, hours_lookback: int, timeframe: str, base_offset_hours: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tf_min = _parse_timeframe_to_minutes(timeframe)
    max_bars = 1000
    hours_per_call = int((max_bars * tf_min) / 60)
    if hours_per_call <= 0:
        hours_per_call = 1

    offsets: List[int] = list(
        range(base_offset_hours, base_offset_hours + hours_lookback, hours_per_call)
    )

    ltf_parts: List[pd.DataFrame] = []
    htf_parts: List[pd.DataFrame] = []

    for offs in offsets:
        remaining_hours = max(0, (base_offset_hours + hours_lookback) - offs)
        bars_needed = int((remaining_hours * 60 + tf_min - 1) // tf_min)
        lookback_bars = min(max_bars, max(1, bars_needed))
        df_ltf, df_htf = load_data(
            symbol,
            lookback=lookback_bars,
            timeframe=timeframe,
            time_offset_hours=offs,
        )
        ltf_parts.append(df_ltf)
        htf_parts.append(df_htf)

    if ltf_parts:
        ltf_all = pd.concat(ltf_parts, ignore_index=True)
        ltf_all = (
            ltf_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        )
    else:
        ltf_all = pd.DataFrame()

    if htf_parts:
        htf_all = pd.concat(htf_parts, ignore_index=True)
        htf_all = (
            htf_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        )
    else:
        htf_all = pd.DataFrame()

    return ltf_all.reset_index(drop=True), htf_all.reset_index(drop=True)


def _compute_window_metrics(sim_df: pd.DataFrame) -> Dict[str, float]:
    trades = int(len(sim_df))
    wr = float((sim_df["pnl"] > 0).mean() * 100.0) if trades else 0.0
    total_r = float(sim_df["pnl"].sum()) if trades else 0.0
    avg_r = float(sim_df["pnl"].mean()) if trades else 0.0
    # Max drawdown in R units on cumulative pnl
    if trades:
        cum = sim_df["pnl"].cumsum()
        dd_series = cum - cum.cummax()
        max_dd = float(dd_series.min()) if not dd_series.empty else 0.0
    else:
        max_dd = 0.0
    return {
        "trades": trades,
        "wr_pct": wr,
        "total_r": total_r,
        "avg_r": avg_r,
        "max_dd": max_dd,
    }


def run_sweep(
    sessions: List[str],
    lookback_hours: int,
    offset_hours: int,
    min_trades: int,
    grid: str,
    oos_offsets: List[int] | None = None,
    atr_list: List[float] | None = None,
    rr_list: List[float] | None = None,
    slip_override: float | None = None,
    comm_list: List[float] | None = None,
    window_trades_min: int = 50,
    window_dd_min: float = -20.0,
    window_avg_min: float = 0.30,
    across_std_max: float = 0.10,
    across_min_avg_min: float = 0.25,
) -> pd.DataFrame:
    cfg = get_config()
    params = get_strategy_params(cfg)
    log = get_logger("param_sweep")
    # Reduce noisy inner-engine logs for sweep runs
    try:
        logging.getLogger("execution_engine").setLevel(logging.WARNING)
    except Exception:
        pass

    # Guards
    log.info("ETH-only mode active: %s", cfg.get("pairs"))
    assert cfg.get("pairs") == ["ETH/USDT"], "ETH-only mode expected"
    log.info("use_dynamic_slippage=%s", cfg.get("use_dynamic_slippage"))
    assert not cfg.get("use_dynamic_slippage", False), "Dynamic slippage must be False for parity phase"

    symbol = "ETH/USDT"
    # Grid selection with CLI overrides
    if atr_list is not None and len(atr_list) > 0:
        atr_grid = atr_list
    elif grid in ("tight", "tight_oos"):
        atr_grid = [1.8, 2.0]
    elif grid == "small":
        atr_grid = [1.8, 2.0]
    else:
        atr_grid = [1.5, 1.8, 2.0, 2.2, 2.5]

    if rr_list is not None and len(rr_list) > 0:
        rr_grid = rr_list
    elif grid in ("tight", "tight_oos"):
        rr_grid = [1.5, 2.0]
    elif grid == "small":
        rr_grid = [1.5, 2.0]
    else:
        rr_grid = [1.2, 1.5, 1.8, 2.0]

    slip_base = params["slippage_pct"]
    if slip_override is not None:
        slip_grid = [float(slip_override)]
    elif grid in ("tight", "tight_oos"):
        slip_grid = [0.03]
    else:
        slip_grid = [slip_base * 0.5, slip_base, slip_base * 1.5]

    comm_base = params["commission_pct"]
    if comm_list is not None and len(comm_list) > 0:
        comm_grid = comm_list
    elif grid in ("tight", "tight_oos"):
        comm_grid = [0.10, 0.05]
    else:
        comm_grid = [comm_base * 0.5, comm_base, comm_base * 1.5]

    results: List[Dict] = []
    run_id = str(int(time.time()))

    # Derive session list
    session_names = []
    for s in sessions:
        normalized = "New York" if s.replace(" ", "").lower() == "newyork" else s
        if normalized not in SESSIONS:
            raise ValueError(f"Unknown session: {s}")
        session_names.append(normalized)

    # Per-session avgR thresholds
    session_avg_threshold = {"Asia": 0.30, "London": 0.30, "New York": 0.20}

    for session_name in session_names:
        start_h, end_h = SESSIONS[session_name]
        # Prepare bases for each offset window
        offsets_to_run = oos_offsets if oos_offsets else [offset_hours]

        # Precompute features per offset window
        window_bases: Dict[int, pd.DataFrame] = {}
        for offs in offsets_to_run:
            df_ltf, df_htf = _load_last_hours_chunked(
                symbol,
                hours_lookback=lookback_hours,
                timeframe=params["timeframe"],
                base_offset_hours=offs,
            )
            df_ltf = _filter_session(df_ltf, start_h, end_h)
            if df_ltf.empty:
                continue
            df_ltf = df_ltf.copy()
            df_htf = df_htf.copy()
            # Seam telemetry: log first/last timestamps
            try:
                log.info(
                    "Seam post-chunk: LTF %s->%s HTF %s->%s offs=%d",
                    str(pd.to_datetime(df_ltf["timestamp"]).min()),
                    str(pd.to_datetime(df_ltf["timestamp"]).max()),
                    str(pd.to_datetime(df_htf["timestamp"]).min()),
                    str(pd.to_datetime(df_htf["timestamp"]).max()),
                    offs,
                )
            except Exception:
                pass
            df_ltf["vwap"] = compute_vwap(df_ltf)
            df_ltf["atr"] = compute_atr(df_ltf)
            df_ltf["ema_21"] = compute_ema(df_ltf)
            df_ltf["bias"] = generate_bias(df_ltf, df_htf)
            if not df_ltf["bias"].notna().all():
                log.error("Bias alignment produced nulls; skipping %s offs=%d", session_name, offs)
                continue
            df_ltf = detect_fvgs(df_ltf)
            df_ltf = detect_liquidity_sweeps(df_ltf)
            df_ltf["reject_low"] = confirm_rejection(df_ltf, sweep_col="sweep_low")
            df_ltf["reject_high"] = confirm_rejection(df_ltf, sweep_col="sweep_high")
            df_base = generate_signals(
                df_ltf,
                long_threshold=params["long_threshold"],
                short_threshold=params["short_threshold"],
            )
            window_bases[offs] = df_base
        if not window_bases:
            continue

        combos = list(product(atr_grid, rr_grid, slip_grid, comm_grid))
        total = len(combos)
        for idx, (atr_mult, rr, slip, comm) in enumerate(combos, start=1):
            log.info(
                "[%s] Config %d/%d: ATR=%.2f RR=%.2f slip=%.2f comm=%.2f",
                session_name,
                idx,
                total,
                atr_mult,
                rr,
                slip,
                comm,
            )

            window_metrics: List[Dict[str, float]] = []
            valid_all_windows = True
            worst_dd = 0.0
            avg_values: List[float] = []
            min_trades_any = 10**9

            for offs in offsets_to_run:
                df_base = window_bases.get(offs)
                if df_base is None or df_base.empty:
                    valid_all_windows = False
                    break
                diag_reset()
                sim = simulate_trades(
                    df_base,
                    atr_multiplier=atr_mult,
                    rr_ratio=rr,
                    max_hold_bars=cfg.get("max_hold_bars", 8),
                    slippage_pct=slip,
                    commission_pct=comm,
                    use_dynamic_slippage=False,
                    asset=symbol,
                    session=session_name,
                    lookback_hours=lookback_hours,
                    offset_hours=offs,
                    run_id=run_id,
                )
                if len(sim) == 0:
                    valid_all_windows = False
                    break
                # Export raw diagnostics per window
                raw_dir = "results/sweeps/raw"
                os.makedirs(raw_dir, exist_ok=True)
                csv_path = (
                    f"{raw_dir}/eth_trades_{session_name.replace(' ', '')}_atr{atr_mult}_rr{rr}_slip{slip}_comm{comm}_lb{lookback_hours}_off{offs}_{run_id}.csv"
                )
                json_path = csv_path.replace(".csv", ".json")
                diag_export(csv_path, json_path)

                m = _compute_window_metrics(sim)
                window_metrics.append(m)
                avg_values.append(m["avg_r"])
                worst_dd = min(worst_dd, m["max_dd"]) if window_metrics else m["max_dd"]
                min_trades_any = min(min_trades_any, m["trades"])

            if not valid_all_windows or not window_metrics:
                continue

            mean_avg = float(sum(avg_values) / len(avg_values))
            std_avg = float(pd.Series(avg_values).std(ddof=0)) if len(avg_values) > 1 else 0.0
            min_avg = float(min(avg_values))

            # Gating per-window and across windows (CLI-driven thresholds)
            per_window_ok = all(
                (wm["trades"] >= window_trades_min and wm["max_dd"] >= window_dd_min and wm["avg_r"] >= window_avg_min)
                for wm in window_metrics
            )
            across_ok = (std_avg <= across_std_max) and (min_avg >= across_min_avg_min)

            score = mean_avg - 0.5 * std_avg

            row: Dict[str, float | int | str] = {
                "Asset": symbol,
                "Session": session_name,
                "ATR Mult": atr_mult,
                "RR Ratio": rr,
                "Slip%": slip,
                "Comm%": comm,
                "lookback_hours": lookback_hours,
                "oos_offsets": ",".join(str(x) for x in offsets_to_run),
                "mean_avgR": mean_avg,
                "std_avgR": std_avg,
                "min_avgR": min_avg,
                "min_trades": int(min_trades_any),
                "maxDD_worst": float(worst_dd),
                "score": score,
                "passes": bool(per_window_ok and across_ok),
            }
            # Add per-window metrics columns
            for j, offs in enumerate(offsets_to_run, start=1):
                wm = window_metrics[j - 1]
                row[f"trades_w{j}"] = wm["trades"]
                row[f"avgR_w{j}"] = wm["avg_r"]
                row[f"wr_w{j}"] = wm["wr_pct"]
                row[f"maxDD_w{j}"] = wm["max_dd"]

            results.append(row)

    df_out = pd.DataFrame(results)
    if df_out.empty:
        return df_out
    # Rank/filter
    if "Trades" in df_out.columns:
        df_out = df_out[df_out["Trades"] >= min_trades].copy()
    elif "min_trades" in df_out.columns:
        df_out = df_out[df_out["min_trades"] >= min_trades].copy()
    # For multi-window runs, sorting by score; otherwise keep previous sort
    if "score" in df_out.columns:
        df_out = df_out.sort_values(["Session", "passes", "score"], ascending=[True, False, False])
    else:
        df_out = df_out.sort_values(["Session", "Avg R/Trade"], ascending=[True, False])
    return df_out


def main():
    parser = argparse.ArgumentParser(description="ETH-only parameter sweep with diagnostics export")
    parser.add_argument("--sessions", nargs="+", default=["Asia", "London", "NewYork"], help="List of sessions: Asia London NewYork")
    parser.add_argument("--lookback_hours", type=int, default=1500)
    parser.add_argument("--offset_hours", type=int, default=0)
    parser.add_argument("--oos_offsets", type=str, default=None, help="Comma-separated offsets for multi-window OOS, e.g. '3000,4500'")
    parser.add_argument("--min_trades", type=int, default=30)
    parser.add_argument("--grid", choices=["tight", "small", "full"], default="tight")
    # Custom grid overrides
    parser.add_argument("--atr_list", type=str, default=None, help="Comma list of ATR multipliers, e.g. '1.6,1.8,2.0,2.2'")
    parser.add_argument("--rr_list", type=str, default=None, help="Comma list of RR ratios, e.g. '1.5,1.8,2.0'")
    parser.add_argument("--slip", type=float, default=None, help="Override fixed slippage pct, e.g. 0.03")
    parser.add_argument("--comm_list", type=str, default=None, help="Comma list of commissions pct, e.g. '0.05,0.10'")
    parser.add_argument("--raw_dir", type=str, default="results/sweeps/raw", help="Directory for per-trade raw dumps")
    # Gating thresholds (optional overrides)
    parser.add_argument("--window_trades_min", type=int, default=50)
    parser.add_argument("--window_dd_min", type=float, default=-25.0)
    parser.add_argument("--window_avg_min", type=float, default=0.30)
    parser.add_argument("--across_std_max", type=float, default=0.10)
    parser.add_argument("--across_min_avg_min", type=float, default=0.25)
    args = parser.parse_args()

    offsets = None
    if args.oos_offsets:
        try:
            offsets = [int(x.strip()) for x in args.oos_offsets.split(",") if x.strip()]
        except Exception:
            offsets = None
    # Override grids if provided
    if args.atr_list:
        try:
            custom_atr = [float(x.strip()) for x in args.atr_list.split(",") if x.strip()]
        except Exception:
            custom_atr = None
    else:
        custom_atr = None
    if args.rr_list:
        try:
            custom_rr = [float(x.strip()) for x in args.rr_list.split(",") if x.strip()]
        except Exception:
            custom_rr = None
    else:
        custom_rr = None
    if args.comm_list:
        try:
            custom_comm = [float(x.strip()) for x in args.comm_list.split(",") if x.strip()]
        except Exception:
            custom_comm = None
    else:
        custom_comm = None

    # Monkey-patch globals for grids if provided
    # We'll pass through run_sweep using global grids inferred from args inside run_sweep
    global SESSIONS  # no-op, just clarity

    df = run_sweep(
        args.sessions,
        args.lookback_hours,
        args.offset_hours,
        args.min_trades,
        ("tight" if (args.atr_list or args.rr_list or args.comm_list or args.slip is not None) else args.grid),
        offsets,
        atr_list=custom_atr,
        rr_list=custom_rr,
        slip_override=args.slip,
        comm_list=custom_comm,
        window_trades_min=args.window_trades_min,
        window_dd_min=args.window_dd_min,
        window_avg_min=args.window_avg_min,
        across_std_max=args.across_std_max,
        across_min_avg_min=args.across_min_avg_min,
    )
    out_dir = "results/sweeps"
    os.makedirs(out_dir, exist_ok=True)
    if df.empty:
        print("No results")
        return
    # CSV aggregate
    # If single session London with OOS offsets, write to results/oos
    single_london = (len(args.sessions) == 1 and (args.sessions[0] in ("London", "NewYork", "Asia")) and offsets)
    if single_london and args.sessions[0] == "London":
        out_dir = "results/oos"
        os.makedirs(out_dir, exist_ok=True)
    agg_csv = os.path.join(out_dir, "eth_param_sweep.csv") if not offsets else os.path.join(out_dir, ("eth_london_oos_stability.csv" if single_london and args.sessions[0] == "London" else "eth_oos_stability.csv"))
    df.to_csv(agg_csv, index=False)
    # Markdown aggregate
    md_path = os.path.join(out_dir, "eth_param_sweep.md") if not offsets else os.path.join(out_dir, ("eth_london_oos_stability.md" if single_london and args.sessions[0] == "London" else "eth_oos_stability.md"))
    with open(md_path, "w", encoding="utf-8") as f:
        if not offsets:
            f.write("| Asset | Session | Trades | Win Rate (%) | Total R | Avg R/Trade | ATR Mult | RR Ratio | Slip% | Comm% |\n")
            f.write("|-------|---------|--------|--------------|---------|-------------|----------|---------|-------|-------|\n")
            for _, r in df.iterrows():
                f.write(
                    f"| {r['Asset']} | {r['Session']} | {int(r['Trades']):>6} | "
                    f"{r['Win Rate (%)']:>12.1f} | {r['Total R']:>7.2f} | {r['Avg R/Trade']:>11.3f} | "
                    f"{r['ATR Mult']:>8.2f} | {r['RR Ratio']:>7.2f} | {r['Slip%']:>5.2f} | {r['Comm%']:>5.2f} |\n"
                )
        else:
            f.write("| Asset | Session | ATR | RR | Slip% | Comm% | Offsets | mean_avgR | std_avgR | min_avgR | min_trades | maxDD_worst | passes | score |\n")
            f.write("|-------|---------|-----|----|-------|--------|---------|-----------|----------|----------|------------|-------------|--------|-------|\n")
            for _, r in df.iterrows():
                f.write(
                    f"| {r['Asset']} | {r['Session']} | {r['ATR Mult']:.2f} | {r['RR Ratio']:.2f} | {r['Slip%']:.2f} | {r['Comm%']:.2f} | {r['oos_offsets']} | "
                    f"{r['mean_avgR']:.3f} | {r['std_avgR']:.3f} | {r['min_avgR']:.3f} | {int(r['min_trades'])} | {r['maxDD_worst']:.2f} | {str(bool(r['passes']))} | {r['score']:.3f} |\n"
                )
            # If London-only, add top-5 and decision
            if single_london and args.sessions[0] == "London":
                f.write("\n\nTop 5 by score (passes only):\n")
                f.write("| ATR | RR | Slip% | Comm% | mean_avgR | std_avgR | min_avgR | min_trades | maxDD_worst | score |\n")
                f.write("|-----|----|-------|--------|-----------|----------|----------|------------|-------------|-------|\n")
                df_pass = df[df["passes"] == True].copy() if "passes" in df.columns else df.head(0)
                df_pass = df_pass.sort_values("score", ascending=False).head(5)
                for _, r in df_pass.iterrows():
                    f.write(
                        f"| {r['ATR Mult']:.2f} | {r['RR Ratio']:.2f} | {r['Slip%']:.2f} | {r['Comm%']:.2f} | {r['mean_avgR']:.3f} | {r['std_avgR']:.3f} | {r['min_avgR']:.3f} | {int(r['min_trades'])} | {r['maxDD_worst']:.2f} | {r['score']:.3f} |\n"
                    )
                if len(df_pass) == 0:
                    f.write("\nDecision: FAIL — No stable London config under two-window gates.\n")
                else:
                    top = df_pass.iloc[0]
                    f.write(
                        f"\nDecision: PASS — Winner ATR={top['ATR Mult']:.2f}, RR={top['RR Ratio']:.2f}, slip={top['Slip%']:.2f}, comm={top['Comm%']:.2f}.\n"
                    )
    # Also echo markdown to stdout
    with open(md_path, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()


