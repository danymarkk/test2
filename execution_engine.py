import pandas as pd
import numpy as np

# Optional diagnostics hook (no-op if module not available)
try:  # pragma: no cover
    from diagnostics import record_trade as _diag_record_trade
except Exception:  # pragma: no cover
    def _diag_record_trade(_trade):  # type: ignore
        return


def find_smart_exit(
    df, entry_idx, signal_type, entry_price, atr_value, atr_multiplier, max_hold
):
    """
    Smart exit logic v1.5 - proven v1 momentum logic + eliminated time_exit
    Returns: (exit_idx, exit_price, exit_reason)
    """
    best_exit_idx = entry_idx + max_hold  # Default fallback
    best_exit_price = df.at[min(best_exit_idx, len(df) - 1), "close"]
    best_exit_reason = "timeout"

    # Look for momentum-based exits (back to proven v1 logic)
    for j in range(
        4, min(max_hold + 1, len(df) - entry_idx)
    ):  # Start checking after 4 hours
        current_idx = entry_idx + j
        if current_idx >= len(df):
            break

        current_price = df.at[current_idx, "close"]

        # Check if we should take profit based on momentum
        if signal_type == "long":
            # For longs: exit if momentum stalls and we have decent profit
            price_progress = (current_price - entry_price) / (
                atr_value * atr_multiplier
            )

            if price_progress > 1.0:  # If we're in good profit (1R+) - RESTORED from v1
                # Check if momentum is dying (price not making higher highs) - SIMPLE v1 logic
                if j >= 8:  # After 8 hours minimum
                    # Add bounds protection to prevent negative indices
                    start_recent = max(0, current_idx - 3)
                    start_prev = max(0, current_idx - 6)
                    end_prev = max(2, current_idx - 2)

                    recent_high = df.iloc[start_recent : current_idx + 1]["high"].max()
                    prev_high = df.iloc[start_prev:end_prev]["high"].max()

                    if recent_high <= prev_high:  # Momentum dying - SIMPLE check
                        best_exit_idx = current_idx
                        best_exit_price = current_price
                        best_exit_reason = "momentum_exit"
                        break

            # ELIMINATED time_exit - only timeout for truly dead trades
            # Only timeout if trade never moved significantly (< 0.5R after 12h)
            elif j >= 12 and abs(price_progress) < 0.5:  # Dead trade
                best_exit_idx = current_idx
                best_exit_price = current_price
                best_exit_reason = "timeout"
                break

        else:  # short
            # For shorts: similar logic but inverted
            price_progress = (entry_price - current_price) / (
                atr_value * atr_multiplier
            )

            if price_progress > 1.0:  # If we're in good profit (1R+) - RESTORED from v1
                if j >= 8:  # After 8 hours minimum
                    # Add bounds protection to prevent negative indices
                    start_recent = max(0, current_idx - 3)
                    start_prev = max(0, current_idx - 6)
                    end_prev = max(2, current_idx - 2)

                    recent_low = df.iloc[start_recent : current_idx + 1]["low"].min()
                    prev_low = df.iloc[start_prev:end_prev]["low"].min()

                    if recent_low >= prev_low:  # Momentum dying - SIMPLE check
                        best_exit_idx = current_idx
                        best_exit_price = current_price
                        best_exit_reason = "momentum_exit"
                        break

            # Only timeout if dead trade
            elif j >= 12 and abs(price_progress) < 0.5:
                best_exit_idx = current_idx
                best_exit_price = current_price
                best_exit_reason = "timeout"
                break

    # Ensure we don't go beyond data
    if best_exit_idx >= len(df):
        best_exit_idx = len(df) - 1
        best_exit_price = df.at[best_exit_idx, "close"]

    return best_exit_idx, best_exit_price, best_exit_reason


def simulate_trades(
    df,
    atr_multiplier=0.75,
    rr_ratio=2.0,
    max_hold=15,
    slippage_pct=0.05,
    commission_pct=0.1,
    max_hold_bars=8,
    use_dynamic_tp=False,
    use_breakeven=False,
    use_dynamic_slippage: bool = False,
    asset: str | None = None,
    session: str | None = "ALL",
    lookback_hours: int | None = None,
    offset_hours: int | None = None,
    run_id: str | None = None,
    risk_weight: float = 1.0,
):
    """
    Clean trade simulation engine with realism factors
    Args:
        slippage_pct: Slippage as % of entry price (0.05 = 0.05%)
        commission_pct: Commission as % per trade (0.1 = 0.1%)
    """
    df = df.copy()
    # Use a local logger to avoid noisy prints
    try:
        from params import get_logger as _get_logger

        _log = _get_logger("execution_engine")
        _log.info("Simulating trades on %d candles...", len(df))
    except Exception:
        pass

    # Initialize trade tracking columns
    df["in_trade"] = False
    df["entry_price"] = np.nan
    df["stop_loss"] = np.nan
    df["take_profit"] = np.nan
    df["exit_price"] = np.nan
    df["exit_reason"] = pd.Series([None] * len(df), dtype="object")
    df["pnl"] = np.nan
    df["exit_index"] = np.nan
    df["side"] = pd.Series([None] * len(df), dtype="object")
    df["slippage_r"] = 0.0
    df["mae_r"] = np.nan
    df["mfe_r"] = np.nan
    if asset is not None:
        df["asset"] = asset
    if session is not None:
        df["session"] = session
    # Optional: carry atr_percent if present for diagnostics
    if "atr" in df.columns and "close" in df.columns:
        with pd.option_context("mode.use_inf_as_na", True):
            df["atr_percent"] = (df["atr"] / df["close"]) * 100.0

    trades_processed = 0
    i = 0

    while i < len(df):  # Process all signals
        row = df.iloc[i]

        # Check for valid signal with required data
        atr_val = row.get("atr", np.nan)
        has_long_signal = (
            row.get("long_signal", False) and not pd.isna(atr_val) and atr_val > 0
        )
        has_short_signal = (
            row.get("short_signal", False) and not pd.isna(atr_val) and atr_val > 0
        )

        if has_long_signal or has_short_signal:
            signal_type = "long" if has_long_signal else "short"
            # Apply parameterized slippage at entry (fixed default)
            entry_slippage = (
                (1 + slippage_pct / 100.0)
                if signal_type == "long"
                else (1 - slippage_pct / 100.0)
            )
            entry_price = row["close"] * entry_slippage
            atr_value = row["atr"]

            # CRITICAL: Record entry price immediately
            df.at[i, "entry_price"] = entry_price
            df.at[i, "side"] = signal_type

            if atr_value <= 0:  # Skip invalid ATR
                i += 1
                continue

            # DYNAMIC TP: Get confluence score and set dynamic R:R ratio
            confluence_score = 0
            if signal_type == "long":
                confluence_score = row.get(
                    "confluence_long", 3
                )  # Default to 3 if missing
            else:
                confluence_score = row.get("confluence_short", 3)

            # Score-based TP logic
            if use_dynamic_tp:
                if confluence_score >= 4:
                    dynamic_rr = 2.0  # Best setups get 2R
                elif confluence_score >= 3:
                    dynamic_rr = 1.5  # Good setups get 1.5R
                else:  # score 2
                    dynamic_rr = 1.0  # Minimum setups get 1R
            else:
                dynamic_rr = rr_ratio  # Use default

            # Calculate stop and target
            if signal_type == "long":
                stop_loss = entry_price - (atr_value * atr_multiplier)
                take_profit = entry_price + (atr_value * atr_multiplier * dynamic_rr)
            else:  # short
                stop_loss = entry_price + (atr_value * atr_multiplier)
                take_profit = entry_price - (atr_value * atr_multiplier * dynamic_rr)

            # Store the actual R:R used for this trade
            actual_rr_ratio = dynamic_rr

            # Record trade entry
            df.at[i, "in_trade"] = True
            df.at[i, "entry_price"] = entry_price
            df.at[i, "stop_loss"] = stop_loss
            df.at[i, "take_profit"] = take_profit

            # Track MFE/MAE in R-units over the hold
            mfe_r = -np.inf
            mae_r = np.inf

            # Simulate forward to find exit
            exit_found = False
            breakeven_triggered = False

            for j in range(1, min(max_hold_bars + 1, len(df) - i)):
                if i + j >= len(df):
                    break

                future_candle = df.iloc[i + j]
                candle_high = future_candle["high"]
                candle_low = future_candle["low"]
                current_price = future_candle["close"]
                exit_idx = i + j

                # BREAKEVEN LOGIC: Move SL to entry when +0.5R profit
                if use_breakeven and not breakeven_triggered:
                    if signal_type == "long":
                        breakeven_threshold = entry_price + (
                            atr_value * atr_multiplier * 0.5
                        )
                        if candle_high >= breakeven_threshold:
                            stop_loss = entry_price  # Move SL to breakeven
                            breakeven_triggered = True
                    else:  # short
                        breakeven_threshold = entry_price - (
                            atr_value * atr_multiplier * 0.5
                        )
                        if candle_low <= breakeven_threshold:
                            stop_loss = entry_price  # Move SL to breakeven
                            breakeven_triggered = True

                # Update MFE/MAE trackers
                if signal_type == "long":
                    favorable = (candle_high - entry_price) / (atr_value * atr_multiplier)
                    adverse = (candle_low - entry_price) / (atr_value * atr_multiplier)
                else:
                    favorable = (entry_price - candle_low) / (atr_value * atr_multiplier)
                    adverse = (entry_price - candle_high) / (atr_value * atr_multiplier)
                mfe_r = max(mfe_r, favorable)
                mae_r = min(mae_r, adverse)

                # TIMEOUT CHECK: Exit if trade held too long
                if j >= max_hold_bars:
                    df.at[i, "exit_price"] = current_price
                    df.at[i, "exit_reason"] = "timeout"
                    df.at[i, "exit_index"] = i + j  # Should be entry index + duration

                    # Calculate timeout P&L based on actual price movement
                    if signal_type == "long":
                        pnl = (current_price - entry_price) / (
                            atr_value * atr_multiplier
                        )
                    else:  # short
                        pnl = (entry_price - current_price) / (
                            atr_value * atr_multiplier
                        )

                    df.at[i, "pnl"] = pnl
                    df.at[i, "slippage_r"] = 0.0
                    df.at[i, "mfe_r"] = float(mfe_r if mfe_r != -np.inf else 0.0)
                    df.at[i, "mae_r"] = float(mae_r if mae_r != np.inf else 0.0)
                    # Diagnostics record (timeout)
                    try:
                        entry_ts = (
                            pd.to_datetime(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else pd.NaT
                        )
                        exit_ts = (
                            pd.to_datetime(df.iloc[i + j]["timestamp"]) if "timestamp" in df.columns else pd.NaT
                        )
                        commission_r = (2 * (commission_pct / 100.0)) * (
                            entry_price / (atr_value * atr_multiplier)
                        )
                    _diag_record_trade(
                            {
                                "timestamp_entry": entry_ts,
                                "timestamp_exit": exit_ts,
                                "asset": asset or "ETH/USDT",
                                "session": session or "ALL",
                                "side": signal_type,
                                "entry_price": float(entry_price),
                                "exit_price": float(current_price),
                                "r": float(pnl - commission_r),
                                "mae_r": float(mae_r if mae_r != np.inf else 0.0),
                                "mfe_r": float(mfe_r if mfe_r != -np.inf else 0.0),
                                "exit_reason": "timeout",
                                "slippage_r": 0.0,
                                "fees_r": float(commission_r),
                                "atr_multiplier": float(atr_multiplier),
                                "rr_ratio": float(dynamic_rr),
                                "slippage_pct": float(slippage_pct),
                                "commission_pct": float(commission_pct),
                                "lookback_hours": int(lookback_hours or 0),
                                "offset_hours": int(offset_hours or 0),
                                "run_id": str(run_id or ""),
                            "risk_weight": float(risk_weight),
                            "atr_percent": float(df.iloc[i].get("atr_percent", 0.0)),
                            }
                        )
                    except Exception:
                        pass
                    exit_found = True
                    break

                # Check exit conditions (apply slippage and commission to realized prices)
                if signal_type == "long":
                    if candle_low <= stop_loss:
                        # Stop loss hit
                        realized_exit = stop_loss * (1 - slippage_pct / 100.0)
                        df.at[i, "exit_price"] = realized_exit
                        df.at[i, "exit_reason"] = (
                            "breakeven" if breakeven_triggered else "stop"
                        )
                        # Calculate actual P&L based on exit price
                        if breakeven_triggered:
                            df.at[i, "pnl"] = 0.0
                            df.at[i, "slippage_r"] = 0.0
                        else:
                            # R multiple based on effective entry/exit
                            df.at[i, "pnl"] = (realized_exit - entry_price) / (
                                atr_value * atr_multiplier
                            )
                            # Slippage in R
                            base_exit = stop_loss
                            slip_r = abs(base_exit - realized_exit) / (
                                atr_value * atr_multiplier
                            )
                            df.at[i, "slippage_r"] = float(slip_r)
                        df.at[i, "exit_index"] = exit_idx
                        df.at[i, "mfe_r"] = float(mfe_r if mfe_r != -np.inf else 0.0)
                        df.at[i, "mae_r"] = float(mae_r if mae_r != np.inf else 0.0)
                        # Diagnostics record (long stop)
                        try:
                            entry_ts = (
                                pd.to_datetime(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else pd.NaT
                            )
                            exit_ts = (
                                pd.to_datetime(df.iloc[exit_idx]["timestamp"]) if "timestamp" in df.columns else pd.NaT
                            )
                            commission_r = (2 * (commission_pct / 100.0)) * (
                                entry_price / (atr_value * atr_multiplier)
                            )
                            _diag_record_trade(
                                {
                                    "timestamp_entry": entry_ts,
                                    "timestamp_exit": exit_ts,
                                    "asset": asset or "ETH/USDT",
                                    "session": session or "ALL",
                                    "side": signal_type,
                                    "entry_price": float(entry_price),
                                    "exit_price": float(realized_exit),
                                    "r": float(df.at[i, "pnl"] - commission_r),
                                    "mae_r": float(mae_r if mae_r != np.inf else 0.0),
                                    "mfe_r": float(mfe_r if mfe_r != -np.inf else 0.0),
                                    "exit_reason": df.at[i, "exit_reason"],
                                    "slippage_r": float(df.at[i, "slippage_r"]),
                                    "fees_r": float(commission_r),
                                    "atr_multiplier": float(atr_multiplier),
                                    "rr_ratio": float(dynamic_rr),
                                    "slippage_pct": float(slippage_pct),
                                    "commission_pct": float(commission_pct),
                                    "lookback_hours": int(lookback_hours or 0),
                                    "offset_hours": int(offset_hours or 0),
                                    "run_id": str(run_id or ""),
                                    "risk_weight": float(risk_weight),
                                    "atr_percent": float(df.iloc[i].get("atr_percent", 0.0)),
                                }
                            )
                        except Exception:
                            pass
                        exit_found = True
                        break
                    elif candle_high >= take_profit:
                        realized_exit = take_profit * (1 - slippage_pct / 100.0)
                        df.at[i, "exit_price"] = realized_exit
                        df.at[i, "exit_reason"] = "target"
                        df.at[i, "pnl"] = (realized_exit - entry_price) / (
                            atr_value * atr_multiplier
                        )
                        base_exit = take_profit
                        slip_r = abs(base_exit - realized_exit) / (
                            atr_value * atr_multiplier
                        )
                        df.at[i, "slippage_r"] = float(slip_r)
                        df.at[i, "exit_index"] = exit_idx
                        df.at[i, "mfe_r"] = float(mfe_r if mfe_r != -np.inf else 0.0)
                        df.at[i, "mae_r"] = float(mae_r if mae_r != np.inf else 0.0)
                        # Diagnostics record (long target)
                        try:
                            entry_ts = (
                                pd.to_datetime(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else pd.NaT
                            )
                            exit_ts = (
                                pd.to_datetime(df.iloc[exit_idx]["timestamp"]) if "timestamp" in df.columns else pd.NaT
                            )
                            commission_r = (2 * (commission_pct / 100.0)) * (
                                entry_price / (atr_value * atr_multiplier)
                            )
                            _diag_record_trade(
                                {
                                    "timestamp_entry": entry_ts,
                                    "timestamp_exit": exit_ts,
                                    "asset": asset or "ETH/USDT",
                                    "session": session or "ALL",
                                    "side": signal_type,
                                    "entry_price": float(entry_price),
                                    "exit_price": float(realized_exit),
                                    "r": float(df.at[i, "pnl"] - commission_r),
                                    "mae_r": float(mae_r if mae_r != np.inf else 0.0),
                                    "mfe_r": float(mfe_r if mfe_r != -np.inf else 0.0),
                                    "exit_reason": df.at[i, "exit_reason"],
                                    "slippage_r": float(df.at[i, "slippage_r"]),
                                    "fees_r": float(commission_r),
                                    "atr_multiplier": float(atr_multiplier),
                                    "rr_ratio": float(dynamic_rr),
                                    "slippage_pct": float(slippage_pct),
                                    "commission_pct": float(commission_pct),
                                    "lookback_hours": int(lookback_hours or 0),
                                    "offset_hours": int(offset_hours or 0),
                                    "run_id": str(run_id or ""),
                                    "risk_weight": float(risk_weight),
                                    "atr_percent": float(df.iloc[i].get("atr_percent", 0.0)),
                                }
                            )
                        except Exception:
                            pass
                        exit_found = True
                        break
                else:  # short
                    if candle_high >= stop_loss:
                        realized_exit = stop_loss * (1 + slippage_pct / 100.0)
                        df.at[i, "exit_price"] = realized_exit
                        df.at[i, "exit_reason"] = (
                            "breakeven" if breakeven_triggered else "stop"
                        )
                        # Calculate actual P&L based on exit price
                        if breakeven_triggered:
                            df.at[i, "pnl"] = 0.0
                            df.at[i, "slippage_r"] = 0.0
                        else:
                            df.at[i, "pnl"] = (entry_price - realized_exit) / (
                                atr_value * atr_multiplier
                            )
                            base_exit = stop_loss
                            slip_r = abs(realized_exit - base_exit) / (
                                atr_value * atr_multiplier
                            )
                            df.at[i, "slippage_r"] = float(slip_r)
                        df.at[i, "exit_index"] = exit_idx
                        df.at[i, "mfe_r"] = float(mfe_r if mfe_r != -np.inf else 0.0)
                        df.at[i, "mae_r"] = float(mae_r if mae_r != np.inf else 0.0)
                        # Diagnostics record (short stop)
                        try:
                            entry_ts = (
                                pd.to_datetime(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else pd.NaT
                            )
                            exit_ts = (
                                pd.to_datetime(df.iloc[exit_idx]["timestamp"]) if "timestamp" in df.columns else pd.NaT
                            )
                            commission_r = (2 * (commission_pct / 100.0)) * (
                                entry_price / (atr_value * atr_multiplier)
                            )
                            _diag_record_trade(
                                {
                                    "timestamp_entry": entry_ts,
                                    "timestamp_exit": exit_ts,
                                    "asset": asset or "ETH/USDT",
                                    "session": session or "ALL",
                                    "side": signal_type,
                                    "entry_price": float(entry_price),
                                    "exit_price": float(realized_exit),
                                    "r": float(df.at[i, "pnl"] - commission_r),
                                    "mae_r": float(mae_r if mae_r != np.inf else 0.0),
                                    "mfe_r": float(mfe_r if mfe_r != -np.inf else 0.0),
                                    "exit_reason": df.at[i, "exit_reason"],
                                    "slippage_r": float(df.at[i, "slippage_r"]),
                                    "fees_r": float(commission_r),
                                    "atr_multiplier": float(atr_multiplier),
                                    "rr_ratio": float(dynamic_rr),
                                    "slippage_pct": float(slippage_pct),
                                    "commission_pct": float(commission_pct),
                                    "lookback_hours": int(lookback_hours or 0),
                                    "offset_hours": int(offset_hours or 0),
                                    "run_id": str(run_id or ""),
                                    "risk_weight": float(risk_weight),
                                    "atr_percent": float(df.iloc[i].get("atr_percent", 0.0)),
                                }
                            )
                        except Exception:
                            pass
                        exit_found = True
                        break
                    elif candle_low <= take_profit:
                        realized_exit = take_profit * (1 + slippage_pct / 100.0)
                        df.at[i, "exit_price"] = realized_exit
                        df.at[i, "exit_reason"] = "target"
                        df.at[i, "pnl"] = (entry_price - realized_exit) / (
                            atr_value * atr_multiplier
                        )
                        base_exit = take_profit
                        slip_r = abs(realized_exit - base_exit) / (
                            atr_value * atr_multiplier
                        )
                        df.at[i, "slippage_r"] = float(slip_r)
                        df.at[i, "exit_index"] = exit_idx
                        df.at[i, "mfe_r"] = float(mfe_r if mfe_r != -np.inf else 0.0)
                        df.at[i, "mae_r"] = float(mae_r if mae_r != np.inf else 0.0)
                        # Diagnostics record (short target)
                        try:
                            entry_ts = (
                                pd.to_datetime(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else pd.NaT
                            )
                            exit_ts = (
                                pd.to_datetime(df.iloc[exit_idx]["timestamp"]) if "timestamp" in df.columns else pd.NaT
                            )
                            commission_r = (2 * (commission_pct / 100.0)) * (
                                entry_price / (atr_value * atr_multiplier)
                            )
                            _diag_record_trade(
                                {
                                    "timestamp_entry": entry_ts,
                                    "timestamp_exit": exit_ts,
                                    "asset": asset or "ETH/USDT",
                                    "session": session or "ALL",
                                    "side": signal_type,
                                    "entry_price": float(entry_price),
                                    "exit_price": float(realized_exit),
                                    "r": float(df.at[i, "pnl"] - commission_r),
                                    "mae_r": float(mae_r if mae_r != np.inf else 0.0),
                                    "mfe_r": float(mfe_r if mfe_r != -np.inf else 0.0),
                                    "exit_reason": df.at[i, "exit_reason"],
                                    "slippage_r": float(df.at[i, "slippage_r"]),
                                    "fees_r": float(commission_r),
                                    "atr_multiplier": float(atr_multiplier),
                                    "rr_ratio": float(dynamic_rr),
                                    "slippage_pct": float(slippage_pct),
                                    "commission_pct": float(commission_pct),
                                    "lookback_hours": int(lookback_hours or 0),
                                    "offset_hours": int(offset_hours or 0),
                                    "run_id": str(run_id or ""),
                                }
                            )
                        except Exception:
                            pass
                        exit_found = True
                        break

            # Handle smart exit if no SL/TP hit - record on ENTRY row
            if not exit_found:
                # Find best exit using smart logic
                smart_exit_idx, smart_exit_price, smart_exit_reason = find_smart_exit(
                    df, i, signal_type, entry_price, atr_value, atr_multiplier, max_hold
                )

                df.at[i, "exit_price"] = smart_exit_price
                df.at[i, "exit_reason"] = smart_exit_reason

                # Calculate smart exit PnL with slippage
                if signal_type == "long":
                    realized_exit = smart_exit_price * (1 - slippage_pct / 100.0)
                    pnl = (realized_exit - entry_price) / (atr_value * atr_multiplier)
                    base_exit = smart_exit_price
                    slip_r = abs(base_exit - realized_exit) / (
                        atr_value * atr_multiplier
                    )
                else:
                    realized_exit = smart_exit_price * (1 + slippage_pct / 100.0)
                    pnl = (entry_price - realized_exit) / (atr_value * atr_multiplier)
                    base_exit = smart_exit_price
                    slip_r = abs(realized_exit - base_exit) / (
                        atr_value * atr_multiplier
                    )
                df.at[i, "pnl"] = pnl
                df.at[i, "exit_index"] = smart_exit_idx
                df.at[i, "slippage_r"] = float(slip_r)
                df.at[i, "mfe_r"] = float(mfe_r if mfe_r != -np.inf else 0.0)
                df.at[i, "mae_r"] = float(mae_r if mae_r != np.inf else 0.0)
                # Diagnostics record (smart exit)
                try:
                    entry_ts = (
                        pd.to_datetime(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else pd.NaT
                    )
                    exit_ts = (
                        pd.to_datetime(df.iloc[smart_exit_idx]["timestamp"]) if "timestamp" in df.columns else pd.NaT
                    )
                    commission_r = (2 * (commission_pct / 100.0)) * (
                        entry_price / (atr_value * atr_multiplier)
                    )
                    _diag_record_trade(
                        {
                            "timestamp_entry": entry_ts,
                            "timestamp_exit": exit_ts,
                            "asset": asset or "ETH/USDT",
                            "session": session or "ALL",
                            "side": signal_type,
                            "entry_price": float(entry_price),
                            "exit_price": float(realized_exit),
                            "r": float(df.at[i, "pnl"] - commission_r),
                            "mae_r": float(mae_r if mae_r != np.inf else 0.0),
                            "mfe_r": float(mfe_r if mfe_r != -np.inf else 0.0),
                            "exit_reason": df.at[i, "exit_reason"],
                            "slippage_r": float(df.at[i, "slippage_r"]),
                            "fees_r": float(commission_r),
                            "atr_multiplier": float(atr_multiplier),
                            "rr_ratio": float(dynamic_rr),
                            "slippage_pct": float(slippage_pct),
                            "commission_pct": float(commission_pct),
                            "lookback_hours": int(lookback_hours or 0),
                            "offset_hours": int(offset_hours or 0),
                            "run_id": str(run_id or ""),
                            "risk_weight": float(risk_weight),
                            "atr_percent": float(df.iloc[i].get("atr_percent", 0.0)),
                        }
                    )
                except Exception:
                    pass

            trades_processed += 1
            # Process next bar (avoid skipping potential signals). To prevent overlapping entries,
            # the next iteration will see in_trade=False because exits are recorded on the entry index only.
            i += 1

            if trades_processed % 10 == 0:
                try:
                    _log.info("Processed %d trades...", trades_processed)
                except Exception:
                    pass
        else:
            i += 1

    # Filter to only completed trades
    completed_trades = df[df["exit_reason"].notna()].copy()

    # Apply commission to pnl in R units (approximate): commission is proportional to notional.
    # Since pnl is expressed in R (= price_move / (ATR*mult)), applying commission precisely
    # requires per-trade notional. For a consistent approximation, we adjust pnl by a factor
    # representing commission on entry and exit relative to the R denominator.
    if len(completed_trades) > 0 and commission_pct:
        # Approximate commission in R terms by translating price commission into R using entry ATR scale.
        # This remains an approximation but avoids subtracting a flat constant irrespective of ATR size.
        # We compute per-trade commissionR = 2 * commission_pct% * (entry_price / (atr*mult)).
        denom = (
            completed_trades["entry_price"] / (completed_trades["atr"] * atr_multiplier)
        ).replace(0, pd.NA)
        commission_r = (2 * (commission_pct / 100.0)) * denom
        commission_r = commission_r.fillna(0.0)
        completed_trades["fees_r"] = commission_r
        completed_trades["pnl"] = completed_trades["pnl"] - commission_r

    try:
        _log.info(
            "Trade simulation complete: %d trades executed", len(completed_trades)
        )
    except Exception:
        pass

    return completed_trades
