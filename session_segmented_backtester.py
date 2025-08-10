"""Session-Segmented Backtester

Runs the unified ICT pipeline segmented by UTC sessions for multiple assets.

Outputs a Markdown table with per-asset, per-session metrics.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple
import argparse
import os
import json
import math

import pandas as pd

from params import get_config, get_strategy_params, get_logger
from data_loader import load_data
from indicators_core import (
    compute_vwap,
    compute_atr,
    compute_ema,
)
from fair_value_gap import detect_fvgs
from liquidity_sweep import detect_liquidity_sweeps
from rejection_confirmation import confirm_rejection
from signal_engine import generate_signals
from execution_engine import simulate_trades
from sessions import filter_session
from htf_merge import merge_htf
from diagnostics import finalize_and_export as diag_export, reset as diag_reset
from sessions import SESSIONS


ASSETS: List[str] = ["ETH/USDT"]


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
    """Fetch up to hours_lookback using multiple calls.

    Honors per-call bar limits. Uses data_loader.load_data with offsets so
    we don't reimplement fetch logic.
    """
    _ = get_config()
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
        bars_needed = math.ceil((remaining_hours * 60) / tf_min)
        lookback_bars = min(max_bars, bars_needed)
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
        ltf_all = ltf_all.drop_duplicates(subset=["timestamp"]).sort_values(
            "timestamp"
        )
    else:
        ltf_all = pd.DataFrame()

    if htf_parts:
        htf_all = pd.concat(htf_parts, ignore_index=True)
        htf_all = htf_all.drop_duplicates(subset=["timestamp"]).sort_values(
            "timestamp"
        )
    else:
        htf_all = pd.DataFrame()

    # Telemetry: seam ranges
    try:
        if not ltf_all.empty and not htf_all.empty:
            get_logger("session_backtest").info(
                "Seam post-chunk: LTF %s->%s HTF %s->%s",
                str(pd.to_datetime(ltf_all["timestamp"]).min()),
                str(pd.to_datetime(ltf_all["timestamp"]).max()),
                str(pd.to_datetime(htf_all["timestamp"]).min()),
                str(pd.to_datetime(htf_all["timestamp"]).max()),
            )
    except Exception:
        pass
    return ltf_all.reset_index(drop=True), htf_all.reset_index(drop=True)


def _run_single_segment(
    symbol: str,
    session_name: str,
    hours_lookback: int,
    base_offset_hours: int,
    export_trades_path: str | None = None,
    atr_override: float | None = None,
    rr_override: float | None = None,
    slip_override: float | None = None,
    comm_override: float | None = None,
) -> Dict[str, float]:
    cfg = get_config()
    params = get_strategy_params(cfg)
    log = get_logger("session_backtest")

    # Fetch hours_lookback with chunking to respect per-call limits
    try:
        df_ltf, df_htf = _load_last_hours_chunked(
            symbol,
            hours_lookback,
            timeframe=params["timeframe"],
            base_offset_hours=base_offset_hours,
        )
    except Exception as e:
        log.error("Data load failed for %s (%s): %s", symbol, session_name, e)
        return {
            "Asset": symbol,
            "Session": session_name,
            "Trades": 0,
            "Win Rate (%)": 0.0,
            "Total R": 0.0,
            "Avg R/Trade": 0.0,
        }

    # Session filter on LTF candles; align HTF to LTF after filter
    df_ltf = filter_session(df_ltf, session_name)
    if df_ltf.empty:
        return {
            "Asset": symbol,
            "Session": session_name,
            "Trades": 0,
            "Win Rate (%)": 0.0,
            "Total R": 0.0,
            "Avg R/Trade": 0.0,
        }

    # Compute indicators and features (fresh copies per run to avoid any bleed)
    df_ltf = df_ltf.copy()
    df_htf = df_htf.copy()
    # Quick logs of ranges
    try:
        log.info(
            "LTF range: %s -> %s | HTF range: %s -> %s | session=%s",
            str(pd.to_datetime(df_ltf["timestamp"]).min()),
            str(pd.to_datetime(df_ltf["timestamp"]).max()),
            str(pd.to_datetime(df_htf["timestamp"]).min()),
            str(pd.to_datetime(df_htf["timestamp"]).max()),
            session_name,
        )
    except Exception:
        pass

    df_ltf["vwap"] = compute_vwap(df_ltf)
    df_ltf["atr"] = compute_atr(df_ltf)
    df_ltf["ema_21"] = compute_ema(df_ltf)
    # Shared HTF merge for bias input
    merged = merge_htf(df_ltf, df_htf, ltf_timeframe=params["timeframe"]) 
    # Use merged HTF close as proxy for bias reference (actual bias computed in indicators_core)
    # Maintain legacy API by calling generate_bias through indicators_core via merged frames
    from indicators_core import generate_bias as _gen_bias
    df_ltf["bias"] = _gen_bias(merged, df_htf)
    if not df_ltf["bias"].notna().all():
        # Skip invalid periods
        return {
            "Asset": symbol,
            "Session": session_name,
            "Trades": 0,
            "Win Rate (%)": 0.0,
            "Total R": 0.0,
            "Avg R/Trade": 0.0,
        }

    df_ltf = detect_fvgs(df_ltf)
    df_ltf = detect_liquidity_sweeps(df_ltf)
    df_ltf["reject_low"] = confirm_rejection(df_ltf, sweep_col="sweep_low")
    df_ltf["reject_high"] = confirm_rejection(df_ltf, sweep_col="sweep_high")

    # Signals
    df_signals = generate_signals(
        df_ltf,
        long_threshold=params["long_threshold"],
        short_threshold=params["short_threshold"],
    )

    # Resolve per-session overrides and quality gates
    per_session_cfg = params.get("per_session", {})
    sess_over = per_session_cfg.get(symbol, {}).get(session_name, {})
    atr_mult_eff = float(atr_override) if atr_override is not None else float(sess_over.get("atr_multiplier", params["atr_multiplier"]))
    rr_eff = float(rr_override) if rr_override is not None else float(sess_over.get("rr_ratio", params["rr_ratio"]))
    risk_weight = float(sess_over.get("risk_weight", 1.0))

    # Quality gate: ATR percent
    min_atr_pct = float(params.get("quality_gates", {}).get("min_atr_percent", 0.0))
    if min_atr_pct > 0:
        df_signals = df_signals.copy()
        df_signals["atr_percent"] = (df_signals["atr"] / df_signals["close"]) * 100.0
        df_signals = df_signals.loc[df_signals["atr_percent"] >= min_atr_pct].copy()
        if df_signals.empty:
            return {
                "Asset": symbol,
                "Session": session_name,
                "Trades": 0,
                "Win Rate (%)": 0.0,
                "Total R": 0.0,
                "Avg R/Trade": 0.0,
            }

    # Simulate trades - isolate with params
    # Reset diagnostics accumulator to isolate this run
    diag_reset()
    try:
        trades = simulate_trades(
            df_signals,
            atr_multiplier=atr_mult_eff,
            rr_ratio=rr_eff,
            max_hold_bars=cfg.get("max_hold_bars", 8),
            slippage_pct=float(slip_override) if slip_override is not None else params["slippage_pct"],
            commission_pct=float(comm_override) if comm_override is not None else params["commission_pct"],
            use_dynamic_slippage=False,
            asset=symbol,
            session=session_name,
            lookback_hours=hours_lookback,
            offset_hours=base_offset_hours,
            run_id=str(int(pd.Timestamp.utcnow().timestamp())),
            risk_weight=risk_weight,
        )
    except Exception:
        trades = pd.DataFrame(columns=["pnl"])  # Fail-safe empty

    if len(trades) == 0:
        return {
            "Asset": symbol,
            "Session": session_name,
            "Trades": 0,
            "Win Rate (%)": 0.0,
            "Total R": 0.0,
            "Avg R/Trade": 0.0,
        }

    trades = trades.copy()
    # Optional export path for parity harness (using diagnostics accumulator)
    if export_trades_path:
        json_path = export_trades_path.replace(".csv", ".json")
        os.makedirs(os.path.dirname(export_trades_path) or ".", exist_ok=True)
        diag_export(export_trades_path, json_path)
    win_rate = float((trades["pnl"] > 0).mean() * 100.0)
    total_r = float(trades["pnl"].sum())
    avg_r = float(trades["pnl"].mean())

    return {
        "Asset": symbol,
        "Session": session_name,
        "Trades": int(len(trades)),
        "Win Rate (%)": win_rate,
        "Total R": total_r,
        "Avg R/Trade": avg_r,
    }


def main():
    log = get_logger("session_backtest")
    cfg = get_config()
    params = get_strategy_params(cfg)

    parser = argparse.ArgumentParser(description="Session-segmented backtester (ETH-only parity)")
    parser.add_argument("--session", default="London", help="Asia|London|NewYork|All")
    parser.add_argument("--lookback_hours", type=int, default=1500)
    parser.add_argument("--offset_hours", type=int, default=0)
    parser.add_argument("--export-trades", dest="export_trades", default=None)
    # Optional overrides for quick OOS validation
    parser.add_argument("--atr_override", type=float, default=None)
    parser.add_argument("--rr_override", type=float, default=None)
    parser.add_argument("--slip_override", type=float, default=None)
    parser.add_argument("--comm_override", type=float, default=None)
    parser.add_argument("--dump-config", dest="dump_config", default=None)
    args = parser.parse_args()

    # Guards and logs
    log.info("ETH-only mode active: %s", cfg.get("pairs"))
    assert cfg.get("pairs") == ["ETH/USDT"], "ETH-only mode expected"
    log.info("use_dynamic_slippage=%s", cfg.get("use_dynamic_slippage"))
    assert not cfg.get("use_dynamic_slippage", False), "Dynamic slippage must be False for parity phase"

    sessions_to_run: List[str]
    if args.session.lower() in ("all", "*"):
        sessions_to_run = ["Asia", "London", "New York"]
    else:
        sessions_to_run = ["New York" if args.session.replace(" ", "").lower() == "newyork" else args.session]

    results: List[Dict[str, float]] = []
    for session_name in sessions_to_run:
        # Log first/last bars after load
        res = _run_single_segment(
            "ETH/USDT",
            session_name,
            hours_lookback=args.lookback_hours,
            base_offset_hours=args.offset_hours,
            export_trades_path=args.export_trades,
            atr_override=args.atr_override,
            rr_override=args.rr_override,
            slip_override=args.slip_override,
            comm_override=args.comm_override,
        )
        results.append(res)

    # Optional: export per-trade diagnostics already handled by execution_engine via diagnostics hooks
    # Export config dump if requested
    if args.dump_config:
        os.makedirs(os.path.dirname(args.dump_config) or ".", exist_ok=True)
        eff = dict(cfg)
        eff["sessions"] = ["Asia", "London", "New York"]
        with open(args.dump_config, "w", encoding="utf-8") as f:
            json.dump(eff, f, ensure_ascii=False, indent=2)

    # Output markdown summary for visibility
    header = "| Asset | Session | Trades | Win Rate (%) | Total R | Avg R/Trade |"
    sep = "|-------|---------|--------|--------------|---------|-------------|"
    print(header)
    print(sep)
    for r in results:
        print(
            f"| {r['Asset']} | {r['Session']} | {r['Trades']:>6} | {r['Win Rate (%)']:>12.1f} | {r['Total R']:>7.2f} | {r['Avg R/Trade']:>11.3f} |"
        )


if __name__ == "__main__":
    sys.exit(main())
