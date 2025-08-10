"""Canonical speed tester runner.

Unified, config-driven multi-period speed testing for any symbol.

This module centralizes the pipeline used by the legacy asset-specific
speed testers and should be treated as the single source of truth for
backtest-style multi-period validation.
"""

from __future__ import annotations

import time
import sys
import argparse
from typing import Dict, List, Optional, Tuple

import pandas as pd

from params import get_config, get_strategy_params, get_logger
from data_loader import load_data
from diagnostics import finalize_and_export as diag_export, reset as diag_reset
from indicators_core import (
    compute_vwap,
    compute_atr,
    compute_ema,
    generate_bias,
)
from fair_value_gap import detect_fvgs
from liquidity_sweep import detect_liquidity_sweeps
from rejection_confirmation import confirm_rejection
from signal_engine import generate_signals
from execution_engine import simulate_trades
from sessions import SESSIONS, filter_session
from typing import Optional as _Optional
from htf_merge import merge_htf


def _default_test_periods() -> List[Dict[str, int]]:
    return [
        {"name": "Recent 500H", "lookback": 500, "offset_hours": 0},
        {"name": "Recent 750H", "lookback": 750, "offset_hours": 0},
        {"name": "Recent 1000H", "lookback": 1000, "offset_hours": 0},
        {"name": "1000H ago", "lookback": 1000, "offset_hours": 500},
        {"name": "1500H ago", "lookback": 1000, "offset_hours": 1000},
    ]


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
    """Fetch up to hours_lookback using multiple paged calls starting at base_offset.

    Uses data_loader.load_data with offsets to avoid duplicating fetch logic.
    """
    tf_min = _parse_timeframe_to_minutes(timeframe)
    max_bars = 1000
    hours_per_call = int((max_bars * tf_min) / 60)
    if hours_per_call <= 0:
        hours_per_call = 1

    offsets: List[int] = list(range(base_offset_hours, base_offset_hours + hours_lookback, hours_per_call))

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
        # Seam guard: drop dups/sort/reset, and log ranges
        ltf_all = ltf_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    else:
        ltf_all = pd.DataFrame()

    if htf_parts:
        htf_all = pd.concat(htf_parts, ignore_index=True)
        htf_all = htf_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    else:
        htf_all = pd.DataFrame()

    # Telemetry: basic seam logging
    try:
        if not ltf_all.empty and not htf_all.empty:
            get_logger("speed_tester").info(
                "Seam post-chunk: LTF %s->%s HTF %s->%s",
                str(pd.to_datetime(ltf_all["timestamp"]).min()),
                str(pd.to_datetime(ltf_all["timestamp"]).max()),
                str(pd.to_datetime(htf_all["timestamp"]).min()),
                str(pd.to_datetime(htf_all["timestamp"]).max()),
            )
    except Exception:
        pass
    return ltf_all.reset_index(drop=True), htf_all.reset_index(drop=True)


def run_speed_test(
    asset: Optional[str] = None,
    symbol: Optional[str] = None,
    test_periods: Optional[List[Dict[str, int]]] = None,
) -> List[Dict[str, float]]:
    """Run the unified multi-period speed test for a given symbol.

    Returns a list of period result dictionaries with summary metrics.
    """
    cfg = get_config()
    params = get_strategy_params(cfg)
    log = get_logger("speed_tester")

    chosen_symbol = symbol or asset or cfg.get("primary_pair", "ETH/USDT")
    periods = test_periods or _default_test_periods()

    log.info("SPEED TESTING - %s - MULTIPLE PERIOD VALIDATION", chosen_symbol)

    all_results: List[Dict[str, float]] = []
    total_trades_overall = 0

    for period in periods:
        period_name = period["name"]
        lookback = int(period["lookback"])
        offset_hours = int(period.get("offset_hours", 0))

        log.info("Testing %s...", period_name)

        # Load data with a single quick retry
        try:
            df_ltf, df_htf = load_data(
                chosen_symbol,
                lookback=lookback,
                timeframe=params["timeframe"],
                time_offset_hours=offset_hours,
            )
            time.sleep(0.5)
        except Exception as e:
            log.warning("Failed to load data for %s: %s", period_name, e)
            log.info("Waiting 3 seconds before retrying...")
            time.sleep(3)
            try:
                df_ltf, df_htf = load_data(
                    chosen_symbol,
                    lookback=lookback,
                    timeframe=params["timeframe"],
                    time_offset_hours=offset_hours,
                )
            except Exception as e2:
                log.error(
                    "Retry failed for %s: %s. Skipping period.",
                    period_name,
                    e2,
                )
                continue

        # Compute indicators and features
        df_ltf["vwap"] = compute_vwap(df_ltf)
        df_ltf["atr"] = compute_atr(df_ltf)
        df_ltf["ema_21"] = compute_ema(df_ltf)
        df_ltf["bias"] = generate_bias(df_ltf, df_htf)
        # Bias non-null assertion: skip this period if invalid
        if not df_ltf["bias"].notna().all():
            log.error(
                "Bias alignment produced nulls; skipping period: %s",
                period_name,
            )
            continue
        df_ltf = detect_fvgs(df_ltf)
        df_ltf = detect_liquidity_sweeps(df_ltf)
        df_ltf["reject_low"] = confirm_rejection(df_ltf, sweep_col="sweep_low")
        df_ltf["reject_high"] = confirm_rejection(
            df_ltf,
            sweep_col="sweep_high",
        )

        # Generate signals
        df_signals = generate_signals(
            df_ltf,
            long_threshold=params["long_threshold"],
            short_threshold=params["short_threshold"],
        )

        # Simulate trades
        try:
            results = simulate_trades(
                df_signals,
                atr_multiplier=params["atr_multiplier"],
                rr_ratio=params["rr_ratio"],
                max_hold_bars=cfg.get("max_hold_bars", 8),
                slippage_pct=params["slippage_pct"],
                commission_pct=params["commission_pct"],
                use_dynamic_slippage=bool(params.get("use_dynamic_slippage", False)),
                asset=chosen_symbol,
                session="MULTI",
            )
        except Exception as e:
            log.error("Trade simulation failed for %s: %s", period_name, e)
            continue

        if len(results) == 0:
            log.warning("No trades in this period")
            continue

        # Diagnostics sanity guard
        required_cols = [
            "pnl",
            "exit_reason",
            "slippage_r",
        ]
        for col in required_cols:
            if col not in results.columns:
                log.error("Diagnostics missing required column: %s", col)
                raise RuntimeError(f"Missing diagnostics column: {col}")
        if results[required_cols].isnull().any().any():
            log.error("Diagnostics contain nulls in required columns")
            raise RuntimeError("Diagnostics contain nulls")

        # Summarize metrics
        win_rate = (results["pnl"] > 0).mean() * 100.0
        total_pnl = float(results["pnl"].sum())
        avg_pnl = float(results["pnl"].mean())
        max_drawdown = float(results["pnl"].cumsum().min())

        period_summary = {
            "period": period_name,
            "trades": int(len(results)),
            "win_rate": float(win_rate),
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "max_drawdown": max_drawdown,
        }
        all_results.append(period_summary)
        total_trades_overall += int(len(results))

        log.info(
            "✅ %d trades, %.1f%% WR, %.2fR P&L",
            len(results),
            win_rate,
            total_pnl,
        )

    # Final summary
    log.info("SPEED TEST SUMMARY:")
    log.info("TOTAL TRADES TESTED: %d", total_trades_overall)
    log.info(
        "TIME TO GET %d TRADES: ~minutes vs days live!",
        total_trades_overall,
    )

    if all_results:
        avg_win_rate = (
            sum(r["win_rate"] for r in all_results) / len(all_results)
        )
        total_pnl_all = sum(r["total_pnl"] for r in all_results)
        avg_pnl_all = (
            sum(r["avg_pnl"] for r in all_results) / len(all_results)
        )

        log.info("AGGREGATE RESULTS:")
        log.info("Average Win Rate: %.1f%%", avg_win_rate)
        log.info("Total P&L (all periods): %.2fR", total_pnl_all)
        log.info(
            "Average P&L per trade: %.3fR",
            avg_pnl_all,
        )

        win_rates = [r["win_rate"] for r in all_results]
        avg_wr = sum(win_rates) / len(win_rates)
        wr_std = (
            sum((wr - avg_wr) ** 2 for wr in win_rates) / len(win_rates)
        ) ** 0.5
        log.info(
            "Win Rate Consistency: %.1f%% std dev",
            wr_std,
        )

        for result in all_results:
            log.info(
                "%s: %d trades, %.1f%% WR, %.2fR",
                result["period"],
                result["trades"],
                result["win_rate"],
                result["total_pnl"],
            )

    return all_results


def run_speed_test_sessions(
    symbol: str,
    session: str = "All",
    lookback_hours: int = 1500,
    offset_hours: int = 0,
    export_trades_path: Optional[str] = None,
) -> List[Dict[str, float]]:
    cfg = get_config()
    params = get_strategy_params(cfg)
    log = get_logger("speed_tester")

    # Sanity guard: ETH-only + dynamic slippage disabled
    log.info("ETH-only mode active: %s", cfg.get("pairs"))
    assert cfg.get("pairs") == ["ETH/USDT"], "ETH-only mode expected"
    log.info("use_dynamic_slippage=%s", cfg.get("use_dynamic_slippage"))
    assert not cfg.get("use_dynamic_slippage", False), "Dynamic slippage must be False for parity phase"

    chosen_symbol = symbol or cfg.get("primary_pair", "ETH/USDT")
    if cfg.get("pairs") == ["ETH/USDT"] and chosen_symbol != "ETH/USDT":
        log.info("ETH-only mode: skipping %s", chosen_symbol)
        return []

    sessions_to_run: List[str]
    if session.lower() in ("all", "*"):
        sessions_to_run = ["Asia", "London", "New York"]
    else:
        # Normalize 'NewYork' -> 'New York'
        normalized = "New York" if session.replace(" ", "").lower() == "newyork" else session
        if normalized not in SESSIONS:
            raise ValueError(f"Unknown session: {session}")
        sessions_to_run = [normalized]

    all_results: List[Dict[str, float]] = []
    run_id = str(int(time.time()))

    # Pull per-session overrides and quality gates
    per_session_cfg = params.get("per_session", {})
    enabled_sessions = set(params.get("enabled_sessions", ["Asia", "London", "New York"]))
    quality_gates = params.get("quality_gates", {"min_atr_percent": 0.0})

    for sess_name in sessions_to_run:
        if sess_name not in enabled_sessions:
            log.info("Session %s disabled via enabled_sessions; skipping", sess_name)
            continue
        log.info(
            "Running session block: symbol=%s session=%s lookback=%dh offset=%dh",
            chosen_symbol,
            sess_name,
            lookback_hours,
            offset_hours,
        )

        # Load data via chunked paging
        df_ltf, df_htf = _load_last_hours_chunked(
            chosen_symbol, lookback_hours, params["timeframe"], base_offset_hours=offset_hours
        )

        # Session gating BEFORE signal generation (shared helper)
        df_ltf = filter_session(df_ltf, sess_name)
        if df_ltf.empty:
            log.warning("No candles after session filter; skipping session %s", sess_name)
            continue

        # Indicators and features
        df_ltf = df_ltf.copy()
        df_htf = df_htf.copy()
        # Quick logs of first/last bars
        try:
            log.info(
                "LTF range: %s -> %s | HTF range: %s -> %s | session=%s",
                str(pd.to_datetime(df_ltf["timestamp"]).min()),
                str(pd.to_datetime(df_ltf["timestamp"]).max()),
                str(pd.to_datetime(df_htf["timestamp"]).min()),
                str(pd.to_datetime(df_htf["timestamp"]).max()),
                sess_name,
            )
        except Exception:
            pass
        df_ltf["vwap"] = compute_vwap(df_ltf)
        df_ltf["atr"] = compute_atr(df_ltf)
        df_ltf["ema_21"] = compute_ema(df_ltf)
        # Shared HTF merge for consistent bias inputs
        merged = merge_htf(df_ltf, df_htf, ltf_timeframe=params["timeframe"]) 
        from indicators_core import generate_bias as _gen_bias
        df_ltf["bias"] = _gen_bias(merged, df_htf)
        if not df_ltf["bias"].notna().all():
            log.error("Bias alignment produced nulls; skipping session %s", sess_name)
            continue
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

        # Quality gate: ATR percent filter
        min_atr_pct = float(quality_gates.get("min_atr_percent", 0.0))
        if min_atr_pct > 0:
            close = df_signals["close"].astype(float)
            atr = df_signals["atr"].astype(float)
            atr_percent = (atr / close).replace([pd.NA, pd.NaT], 0).fillna(0.0) * 100.0
            df_signals = df_signals.assign(atr_percent=atr_percent)
            pre = len(df_signals)
            df_signals = df_signals.loc[df_signals["atr_percent"] >= min_atr_pct].copy()
            log.info(
                "Quality gate: kept %d/%d rows with ATR%% >= %.2f in session %s",
                len(df_signals), pre, min_atr_pct, sess_name,
            )
            if df_signals.empty:
                log.warning("All rows filtered by quality gate; skipping %s", sess_name)
                continue

        # Isolate diagnostics per session
        diag_reset()

        # Resolve per-session overrides
        sess_over = per_session_cfg.get(chosen_symbol, {}).get(sess_name, {})
        atr_mult = float(sess_over.get("atr_multiplier", params["atr_multiplier"]))
        rr_ratio = float(sess_over.get("rr_ratio", params["rr_ratio"]))
        risk_weight = float(sess_over.get("risk_weight", 1.0))

        log.info(
            "Params for %s: ATR=%.2f RR=%.2f risk_weight=%.2f (override=%s)",
            sess_name,
            atr_mult,
            rr_ratio,
            risk_weight,
            "yes" if sess_over else "no",
        )

        # Simulate trades
        results = simulate_trades(
            df_signals,
            atr_multiplier=atr_mult,
            rr_ratio=rr_ratio,
            max_hold_bars=cfg.get("max_hold_bars", 8),
            slippage_pct=params["slippage_pct"],
            commission_pct=params["commission_pct"],
            use_dynamic_slippage=bool(params.get("use_dynamic_slippage", False)),
            asset=chosen_symbol,
            session=sess_name,
            lookback_hours=lookback_hours,
            offset_hours=offset_hours,
            run_id=run_id,
            risk_weight=risk_weight,
        )

        if len(results) == 0:
            log.warning("No trades in session %s", sess_name)
            continue

        # Export per-trade diagnostics for this session
        if export_trades_path and len(sessions_to_run) == 1:
            csv_path = export_trades_path
        else:
            csv_path = (
                f"results/diagnostics/eth_trades_{sess_name.replace(' ', '')}_{lookback_hours}_{offset_hours}_{run_id}.csv"
            )
        json_path = csv_path.replace(".csv", ".json")
        diag_export(csv_path, json_path)

        # Summarize
        win_rate = (results["pnl"] > 0).mean() * 100.0
        total_pnl = float(results["pnl"].sum())
        avg_pnl = float(results["pnl"].mean())
        summary = {
            "session": sess_name,
            "trades": int(len(results)),
            "win_rate": float(win_rate),
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
        }
        all_results.append(summary)
        log.info(
            "✅ %s: %d trades, %.1f%% WR, %.2fR P&L",
            sess_name,
            len(results),
            win_rate,
            total_pnl,
        )

    return all_results


if __name__ == "__main__":
    cfg = get_config()
    parser = argparse.ArgumentParser(description="ETH speed tester with session parity")
    parser.add_argument("symbol", nargs="?", default=cfg.get("primary_pair", "ETH/USDT"))
    parser.add_argument("--session", default="All", help="Asia|London|NewYork|All")
    parser.add_argument("--lookback_hours", type=int, default=1500)
    parser.add_argument("--offset_hours", type=int, default=0)
    parser.add_argument("--export-trades", "--export_trades", dest="export_trades", default=None)
    parser.add_argument("--dump-config", "--dump_config", dest="dump_config", default=None)
    args = parser.parse_args()

    # ETH-only guard
    if cfg.get("pairs") == ["ETH/USDT"] and args.symbol != "ETH/USDT":
        get_logger("speed_tester").info("ETH-only mode: skipping %s", args.symbol)
        sys.exit(0)

    # Default to session-based parity runner
    summaries = run_speed_test_sessions(
        symbol=args.symbol,
        session=args.session,
        lookback_hours=args.lookback_hours,
        offset_hours=args.offset_hours,
        export_trades_path=args.export_trades,
    )

    # Optional dump of effective config
    if args.dump_config:
        import json
        import os
        eff = dict(cfg)
        eff["sessions"] = ["Asia", "London", "New York"]
        eff_path = args.dump_config
        os.makedirs(os.path.dirname(eff_path) or ".", exist_ok=True)
        with open(eff_path, "w", encoding="utf-8") as f:
            json.dump(eff, f, ensure_ascii=False, indent=2)
