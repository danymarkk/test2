"""Multi-Asset Paper Trader - BTC, ETH, XRP Live Trading (SPOT).

Adds bar-close validation and retry/backoff on exchange calls. Pulls defaults from config when available.
"""

import ccxt
import pandas as pd
import csv
import time
import json

# logging configured centrally via params.get_logger
from threading import Thread

# Import our strategy modules
from indicators_core import compute_vwap, compute_atr, compute_ema, generate_bias
from fair_value_gap import detect_fvgs
from liquidity_sweep import detect_liquidity_sweeps
from rejection_confirmation import confirm_rejection
from signal_engine import generate_signals

from params import get_config, get_logger


class MultiAssetPaperTrader:
    def __init__(
        self,
        exchange_id="binance",
        symbols=["BTC/USDT", "ETH/USDT", "XRP/USDT"],
        timeframe="30m",
        initial_balance=1000,
        risk_per_trade=0.02,
        atr_multiplier=0.75,
        rr_ratio=2.0,
        max_hold_hours=4,
    ):

        # Initialize exchange
        # Load defaults from centralized params config
        cfg = get_config()
        if cfg:
            symbols = cfg.get("pairs", symbols)
            timeframe = cfg.get("timeframe", timeframe)
            risk_per_trade = cfg.get("risk_per_trade", risk_per_trade)
            atr_multiplier = cfg.get("atr_multiplier", atr_multiplier)
            rr_ratio = cfg.get("rr_ratio", rr_ratio)
            # Costs
            self.slippage_pct = float(cfg.get("slippage_pct", 0.05))
            self.commission_pct = float(cfg.get("commission_pct", 0.1))
        else:
            self.slippage_pct = 0.05
            self.commission_pct = 0.1

        ex_id = cfg.get("exchange_id", exchange_id)
        ex_options = {
            "defaultType": cfg.get("default_type", cfg.get("defaultType", "spot"))
        }
        self.exchange = getattr(ccxt, ex_id)(
            {
                "enableRateLimit": True,
                "options": ex_options,
            }
        )
        try:
            self.exchange.set_sandbox_mode(bool(cfg.get("testnet", False)))
        except Exception:
            # Fallback if not supported on this exchange/ccxt version
            pass
        try:
            self.exchange.load_markets()
        except Exception:
            pass

        # Trading parameters
        self.symbols = symbols
        self.timeframe = timeframe
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.rr_ratio = rr_ratio
        self.max_hold_hours = max_hold_hours

        # Trading state - separate for each asset
        self.open_trades = {}  # symbol: trade_dict
        self.trade_history = []
        self.last_candle_timestamps = {}  # symbol: timestamp

        # Initialize tracking for each symbol
        for symbol in self.symbols:
            self.open_trades[symbol] = None
            self.last_candle_timestamps[symbol] = None

        # Setup logging via centralized params logger
        self.logger = get_logger(__name__)

        self.logger.info("Multi-Asset Paper Trader initialized:")
        self.logger.info(f"   Symbols: {', '.join(symbols)}")
        self.logger.info(f"   Initial Balance: ${initial_balance:,.2f}")
        self.logger.info(f"   Risk per Trade: {risk_per_trade*100}%")
        self.logger.info(f"   ATR Multiplier: {atr_multiplier}")
        self.logger.info(f"   R/R Ratio: {rr_ratio}:1")

    def fetch_market_data(self, symbol, lookback=500):
        """Fetch recent market data from Binance for a specific symbol"""
        try:
            # Fetch OHLCV data with simple retry/backoff
            for attempt in range(3):
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol, self.timeframe, limit=lookback
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    self.logger.warning(
                        f"{symbol}: fetch_ohlcv failed ({attempt+1}/3): {e}"
                    )
                    time.sleep(1 + attempt)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Also fetch 2H data for bias calculation (with retry)
            for attempt in range(3):
                try:
                    ohlcv_2h = self.exchange.fetch_ohlcv(
                        symbol, "2h", limit=max(1, lookback // 4)
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    self.logger.warning(
                        f"{symbol}: fetch_ohlcv 2h failed ({attempt+1}/3): {e}"
                    )
                    time.sleep(1 + attempt)
            df_2h = pd.DataFrame(
                ohlcv_2h,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df_2h["timestamp"] = pd.to_datetime(df_2h["timestamp"], unit="ms")

            return df, df_2h

        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")
            return None, None

    def calculate_indicators(self, df_1h, df_2h):
        """Calculate all technical indicators"""
        try:
            df = df_1h.copy()

            # Core indicators
            df["vwap"] = compute_vwap(df)
            df["atr"] = compute_atr(df)
            df["ema_21"] = compute_ema(df)
            df["bias"] = generate_bias(df, df_2h)

            # ICT patterns
            df = detect_fvgs(df)
            df = detect_liquidity_sweeps(df)
            df["reject_low"] = confirm_rejection(df, sweep_col="sweep_low")
            df["reject_high"] = confirm_rejection(df, sweep_col="sweep_high")

            # Runtime bias assertion: ensure bias is present
            if not df["bias"].notna().all():
                self.logger.error("Bias column contains nulls; skipping this iteration")
                return None

            return df

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return None

    def check_for_signals(self, df, symbol):
        """Check for trading signals using our strategy"""
        try:
            # Ensure we operate on the last CLOSED candle only
            if len(df) < 2:
                return None, None
            df = df.iloc[:-1].copy()

            df_signals = generate_signals(df, long_threshold=3, short_threshold=3)

            # CRITICAL: Check if signal generation failed
            if df_signals.empty:
                self.logger.warning(
                    f"No signals generated for {symbol}. Check strategy logic."
                )
                return None, None

            # Log signal generation stats for debugging
            long_count = df_signals["long_signal"].sum()
            short_count = df_signals["short_signal"].sum()
            total_signals = long_count + short_count

            if total_signals == 0:
                # Get confluence stats for debugging
                if "confluence_long" in df_signals.columns:
                    max_long_conf = df_signals["confluence_long"].max()
                    max_short_conf = df_signals["confluence_short"].max()
                    self.logger.debug(
                        f"{symbol}: No qualifying signals. Max confluence: "
                        f"Long={max_long_conf}, Short={max_short_conf} (need >=3)"
                    )
                else:
                    self.logger.debug(
                        f"{symbol}: No signals - confluence columns missing"
                    )
            else:
                self.logger.info(
                    f"{symbol}: Generated {long_count} long, {short_count} short signals"
                )

            # Get the last timestamp we checked for this symbol
            last_checked = self.last_candle_timestamps[symbol]

            # Find new candles since last check
            if last_checked is not None:
                new_signals = df_signals[df_signals["timestamp"] > last_checked]
            else:
                # First run - check only the last fully closed candle
                new_signals = df_signals.iloc[-1:].copy()

            if new_signals.empty:
                return None, None

            # Update last checked timestamp to the most recent candle
            self.last_candle_timestamps[symbol] = df_signals.iloc[-1]["timestamp"]

            # Look for signals in new candles (prioritize most recent)
            for idx in range(len(new_signals) - 1, -1, -1):  # Check newest first
                candle = new_signals.iloc[idx]

                if candle.get("long_signal", False) and not pd.isna(candle.get("atr")):
                    self.logger.info(
                        f"{symbol}: 🔵 LONG signal detected at {candle['timestamp']}"
                    )
                    return "long", candle
                elif candle.get("short_signal", False) and not pd.isna(
                    candle.get("atr")
                ):
                    self.logger.info(
                        f"{symbol}: 🔴 SHORT signal detected at {candle['timestamp']}"
                    )
                    return "short", candle

            # No signals found in new candles
            return None, None

        except Exception as e:
            self.logger.error(f"Error checking signals for {symbol}: {e}")
            return None, None

    def calculate_dynamic_slippage(self, atr_value, symbol):
        """Calculate dynamic slippage based on ATR and asset type"""
        base_slippage = 0.001  # 0.1% base

        # Major pairs get lower slippage
        if "BTC" in symbol or "ETH" in symbol:
            atr_multiplier = 0.0001  # Lower for majors
        else:  # XRP and other alts
            atr_multiplier = 0.0002  # Higher for alts

        dynamic_slippage = base_slippage + (atr_value * atr_multiplier)

        # Cap slippage between 0.05% and 0.5%
        return max(0.0005, min(0.005, dynamic_slippage))

    def calculate_position_size(self, entry_price, stop_loss, atr_value, symbol):
        """Calculate position size based on risk management with dynamic slippage"""
        risk_amount = self.balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)

        if price_risk <= 0:
            return 0

        # Position size in base currency
        position_size = risk_amount / price_risk

        # Add slippage/fees buffer (fixed or dynamic per flag)
        use_dyn = self.use_dynamic_slippage
        if use_dyn:
            slippage_buffer = max(
                self.slippage_pct / 100.0,
                self.calculate_dynamic_slippage(atr_value, symbol),
            )
        else:
            slippage_buffer = self.slippage_pct / 100.0
        commission_buffer = self.commission_pct / 100.0
        total_cost_buffer = max(0.0, min(0.99, slippage_buffer + commission_buffer))

        position_size *= 1 - total_cost_buffer

        self.logger.debug(
            (
                f"{symbol}: Dynamic slippage {slippage_buffer:.4f} + commission "
                f"{commission_buffer:.4f} = {total_cost_buffer:.4f}"
            )
        )

        return position_size

    def open_position(self, symbol, signal_type, candle_data):
        """Open a new paper trading position"""
        try:
            entry_price = candle_data["close"]
            atr_value = candle_data["atr"]

            if atr_value <= 0:
                self.logger.warning(
                    f"Invalid ATR for {symbol}: {atr_value} at {candle_data['timestamp']} "
                    f"(Close: ${entry_price:.4f}, High: ${candle_data['high']:.4f}, "
                    f"Low: ${candle_data['low']:.4f}, Volume: {candle_data.get('volume', 'N/A')})"
                )
                return False

            # Calculate stop loss and take profit
            if signal_type == "long":
                stop_loss = entry_price - (atr_value * self.atr_multiplier)
                take_profit = entry_price + (
                    atr_value * self.atr_multiplier * self.rr_ratio
                )
            else:  # short
                stop_loss = entry_price + (atr_value * self.atr_multiplier)
                take_profit = entry_price - (
                    atr_value * self.atr_multiplier * self.rr_ratio
                )

            # Calculate position size with dynamic slippage
            position_size = self.calculate_position_size(
                entry_price, stop_loss, atr_value, symbol
            )

            if position_size <= 0:
                self.logger.warning(
                    f"Invalid position size for {symbol}, skipping trade"
                )
                return False

            # Create trade record
            trade_id = (
                len(self.trade_history)
                + sum(1 for t in self.open_trades.values() if t is not None)
                + 1
            )

            self.open_trades[symbol] = {
                "id": trade_id,
                "symbol": symbol,
                "signal_type": signal_type,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": position_size,
                "entry_time": candle_data["timestamp"],
                "atr": atr_value,
                "risk_amount": self.balance * self.risk_per_trade,
            }

            # Terminal alert for signal trigger
            self.logger.info(
                "SIGNAL: %s %s at $%.4f (TP: $%.4f / SL: $%.4f)",
                symbol,
                signal_type.upper(),
                entry_price,
                take_profit,
                stop_loss,
            )

            self.logger.info(f"{symbol} {signal_type.upper()} position opened:")
            self.logger.info(f"   Entry: ${entry_price:.4f}")
            self.logger.info(f"   Stop Loss: ${stop_loss:.4f}")
            self.logger.info(f"   Take Profit: ${take_profit:.4f}")
            self.logger.info(f"   Position Size: {position_size:.6f}")
            self.logger.info(f"   Risk: ${self.open_trades[symbol]['risk_amount']:.2f}")

            # Save open trade to CSV
            self.save_open_trades_csv()

            return True

        except Exception as e:
            self.logger.error(f"Error opening position for {symbol}: {e}")
            return False

    def check_exit_conditions(self, symbol, current_candle):
        """Check if open position should be closed"""
        if not self.open_trades[symbol]:
            return False

        try:
            current_price = current_candle["close"]
            current_high = current_candle["high"]
            current_low = current_candle["low"]
            current_time = current_candle["timestamp"]

            trade = self.open_trades[symbol]

            # Get trade parameters for exit checks
            # Check time-based exit (max hold)
            time_diff = current_time - trade["entry_time"]
            hours_held = time_diff.total_seconds() / 3600

            if hours_held >= self.max_hold_hours:
                return self.close_position(
                    symbol, "timeout", current_price, current_time
                )

            # Check price-based exits (using potentially updated stop loss)
            if trade["signal_type"] == "long":
                if current_low <= trade["stop_loss"]:
                    return self.close_position(
                        symbol, "stop_loss", trade["stop_loss"], current_time
                    )
                elif current_high >= trade["take_profit"]:
                    return self.close_position(
                        symbol, "take_profit", trade["take_profit"], current_time
                    )
            else:  # short
                if current_high >= trade["stop_loss"]:
                    return self.close_position(
                        symbol, "stop_loss", trade["stop_loss"], current_time
                    )
                elif current_low <= trade["take_profit"]:
                    return self.close_position(
                        symbol, "take_profit", trade["take_profit"], current_time
                    )

            return False

        except Exception as e:
            self.logger.error(f"Error checking exit conditions for {symbol}: {e}")
            return False

    def close_position(self, symbol, exit_reason, exit_price, exit_time):
        """Close the current position and calculate P&L"""
        try:
            if not self.open_trades[symbol]:
                return False

            trade = self.open_trades[symbol]

            # Calculate P&L in R units (risk-adjusted)
            entry_price = trade["entry_price"]
            atr_value = trade["atr"]
            signal_type = trade["signal_type"]

            # Apply exit slippage (use max of base and dynamic slippage)
            use_dyn = self.use_dynamic_slippage
            if use_dyn:
                slip_frac = max(
                    self.slippage_pct / 100.0,
                    self.calculate_dynamic_slippage(atr_value, symbol),
                )
            else:
                slip_frac = self.slippage_pct / 100.0
            if signal_type == "long":
                realized_exit = exit_price * (1 - slip_frac)
            else:  # short
                realized_exit = exit_price * (1 + slip_frac)

            denom_r = (atr_value * self.atr_multiplier)
            if denom_r <= 0:
                return False

            # Base R multiple from effective realized exit
            if signal_type == "long":
                pnl_r_base = (realized_exit - entry_price) / denom_r
            else:
                pnl_r_base = (entry_price - realized_exit) / denom_r

            # Commission in R-units (entry+exit) approximated like backtester
            commission_r = 2 * (self.commission_pct / 100.0) * (entry_price / denom_r)
            pnl_r = pnl_r_base - commission_r

            # Calculate dollar P&L
            pnl_dollars = pnl_r * trade["risk_amount"]

            # Update balance
            self.balance += pnl_dollars

            # Create trade record
            # Slippage impact in R-units (approximate)
            if signal_type == "long":
                slippage_r = (exit_price - realized_exit) / denom_r
            else:
                slippage_r = (realized_exit - exit_price) / denom_r

            trade_record = {
                **trade,
                "exit_reason": exit_reason,
                "exit_price": realized_exit,
                "exit_time": exit_time,
                "pnl_r": pnl_r,
                "pnl_dollars": pnl_dollars,
                "new_balance": self.balance,
                "slippage_r": slippage_r,
                "fees_r": float(commission_r),
            }

            self.trade_history.append(trade_record)

            # Terminal alert for trade closure
            pnl_sign = "+" if pnl_r >= 0 else ""
            self.logger.info(
                "TRADE CLOSED: %s %s%.2fR ($%.2f) | Reason: %s",
                symbol,
                pnl_sign,
                pnl_r,
                pnl_dollars,
                exit_reason,
            )

            # Log results
            self.logger.info(f"{symbol} Position closed ({exit_reason}):")
            self.logger.info(
                f"   Exit Price (realized): ${realized_exit:.4f}"
            )
            self.logger.info(f"   P&L: {pnl_r:.3f}R (${pnl_dollars:.2f})")
            self.logger.info(f"   New Balance: ${self.balance:.2f}")

            # Clear open position
            self.open_trades[symbol] = None

            # Save trade to both JSON and CSV files
            self.save_trade_history()
            self.save_trade_history_csv()

            # Update open trades CSV
            self.save_open_trades_csv()

            return True

        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return False

    def save_trade_history(self):
        """Save trade history to JSON file"""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_history = []
            for trade in self.trade_history:
                trade_copy = trade.copy()
                trade_copy["entry_time"] = trade_copy["entry_time"].isoformat()
                trade_copy["exit_time"] = trade_copy["exit_time"].isoformat()
                serializable_history.append(trade_copy)

            with open("multi_asset_paper_trading_history.json", "w") as f:
                json.dump(serializable_history, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving trade history: {e}")

    def save_trade_history_csv(self):
        """Save trade history to CSV file"""
        try:
            if not self.trade_history:
                return

            # Define CSV fieldnames
            fieldnames = [
                "id",
                "symbol",
                "timestamp",
                "entry_price",
                "exit_price",
                "pnl_r",
                "pnl_dollars",
                "signal_type",
                "exit_reason",
                "balance",
                "slippage_r",
                "fees_r",
            ]

            with open("multi_asset_paper_trades.csv", "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for trade in self.trade_history:
                    # Convert trade data to CSV format
                    csv_row = {
                        "id": trade["id"],
                        "symbol": trade["symbol"],
                        "timestamp": trade["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": round(trade["entry_price"], 6),
                        "exit_price": round(trade["exit_price"], 6),
                        "pnl_r": round(trade["pnl_r"], 3),
                        "pnl_dollars": round(trade["pnl_dollars"], 2),
                        "signal_type": trade["signal_type"],
                        "exit_reason": trade["exit_reason"],
                        "balance": round(trade["new_balance"], 2),
                        "slippage_r": round(trade.get("slippage_r", 0.0), 4),
                        "fees_r": round((2 * (self.commission_pct / 100.0)) * (trade["entry_price"] / (trade["atr"] * self.atr_multiplier)), 4),
                    }
                    writer.writerow(csv_row)

        except Exception as e:
            self.logger.error(f"Error saving trade history CSV: {e}")

    def save_open_trades_csv(self):
        """Save current open trades to CSV file (overwrites each time)"""
        try:
            fieldnames = [
                "id",
                "symbol",
                "signal_type",
                "entry_price",
                "stop_loss",
                "take_profit",
                "entry_time",
                "position_size",
            ]

            with open("multi_asset_open_trades.csv", "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for symbol, trade in self.open_trades.items():
                    if trade:  # Only write if there's an open trade
                        csv_row = {
                            "id": trade["id"],
                            "symbol": trade["symbol"],
                            "signal_type": trade["signal_type"],
                            "entry_price": round(trade["entry_price"], 6),
                            "stop_loss": round(trade["stop_loss"], 6),
                            "take_profit": round(trade["take_profit"], 6),
                            "entry_time": trade["entry_time"].strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "position_size": round(trade["position_size"], 8),
                        }
                        writer.writerow(csv_row)

        except Exception as e:
            self.logger.error(f"Error saving open trades CSV: {e}")

    def print_performance_summary(self):
        """Print current performance statistics"""
        if not self.trade_history:
            self.logger.info("No completed trades yet")
            return

        # Calculate metrics
        df_trades = pd.DataFrame(self.trade_history)
        total_trades = len(df_trades)
        wins = (df_trades["pnl_r"] > 0).sum()
        win_rate = (wins / total_trades) * 100
        total_pnl = df_trades["pnl_r"].sum()
        avg_pnl = df_trades["pnl_r"].mean()

        # Asset breakdown (logged per symbol below)

        roi = ((self.balance - self.initial_balance) / self.initial_balance) * 100

        self.logger.info("PERFORMANCE SUMMARY:")
        self.logger.info(f"   Total Trades: {total_trades}")
        self.logger.info(f"   Win Rate: {win_rate:.1f}%")
        self.logger.info(f"   Total P&L: {total_pnl:.2f}R")
        self.logger.info(f"   Average P&L: {avg_pnl:.3f}R")
        self.logger.info(f"   ROI: {roi:.2f}%")
        self.logger.info(f"   Current Balance: ${self.balance:.2f}")

        self.logger.info("ASSET BREAKDOWN:")
        for symbol in self.symbols:
            symbol_trades = df_trades[df_trades["symbol"] == symbol]
            if not symbol_trades.empty:
                count = len(symbol_trades)
                wr = (symbol_trades["pnl_r"] > 0).mean() * 100
                pnl = symbol_trades["pnl_r"].sum()
                self.logger.info(
                    "%s: %d trades, %.1f%% WR, %.2fR", symbol, count, wr, pnl
                )

    def monitor_asset(self, symbol, check_interval_minutes=5):
        """Monitor a single asset (run in separate thread)"""
        self.logger.info(f"Starting monitoring for {symbol}")

        while True:
            try:
                # Fetch latest market data
                df_1h, df_2h = self.fetch_market_data(symbol)

                if df_1h is None or df_2h is None:
                    self.logger.warning(
                        f"Failed to fetch market data for {symbol}, retrying in 5 minutes..."
                    )
                    time.sleep(300)
                    continue

                # Validate that the most recent candle is still open; use only closed bars for decisions
                if len(df_1h) < 2:
                    time.sleep(check_interval_minutes * 60)
                    continue

                # Calculate indicators
                df_processed = self.calculate_indicators(df_1h, df_2h)

                if df_processed is None:
                    self.logger.warning(
                        f"Failed to process indicators for {symbol}, retrying..."
                    )
                    time.sleep(300)
                    continue

                # Check for exit conditions if we have an open trade (on closed bar)
                if self.open_trades[symbol]:
                    latest_candle = df_processed.iloc[-2]
                    self.check_exit_conditions(symbol, latest_candle)

                # Check for new signals if no open trade
                if not self.open_trades[symbol]:
                    signal_type, candle_data = self.check_for_signals(
                        df_processed, symbol
                    )

                    if signal_type and candle_data is not None:
                        self.open_position(symbol, signal_type, candle_data)

                # Wait for next check
                time.sleep(check_interval_minutes * 60)

            except Exception as e:
                self.logger.error(f"Error monitoring {symbol}: {e}")
                time.sleep(300)  # Wait 5 minutes before retry

    def run_live_trading(self, check_interval_minutes=5):
        """Run the paper trading system in real-time for all assets"""
        self.logger.info("Starting multi-asset live paper trading...")
        self.logger.info(f"   Check interval: {check_interval_minutes} minutes")
        self.logger.info(f"   Monitoring: {', '.join(self.symbols)}")

        try:
            # Start monitoring each asset in separate threads
            threads = []
            for symbol in self.symbols:
                thread = Thread(
                    target=self.monitor_asset, args=(symbol, check_interval_minutes)
                )
                thread.daemon = True
                thread.start()
                threads.append(thread)
                self.logger.info(f"   Started monitoring thread for {symbol}")

            # Main loop for performance reporting
            while True:
                time.sleep(600)  # Print summary every 10 minutes

                # Print performance summary if we have trades
                if len(self.trade_history) > 0:
                    self.print_performance_summary()

        except KeyboardInterrupt:
            self.logger.info("Multi-asset paper trading stopped by user")
            self.print_performance_summary()
        except Exception as e:
            self.logger.error(f"Critical error in trading loop: {e}")
            self.print_performance_summary()


if __name__ == "__main__":
    # Initialize multi-asset paper trader from centralized config
    _cfg = get_config()
    trader = MultiAssetPaperTrader(
        symbols=_cfg.get("pairs", ["BTC/USDT", "ETH/USDT", "XRP/USDT"]),
        timeframe=_cfg.get("timeframe", "30m"),
        initial_balance=1000,
        risk_per_trade=_cfg.get("risk_per_trade", 0.02),
        atr_multiplier=_cfg.get("atr_multiplier", 0.75),
        rr_ratio=_cfg.get("rr_ratio", 2.0),
        max_hold_hours=4,
    )

    # Start live multi-asset paper trading (checks every 30 minutes)
    trader.run_live_trading(check_interval_minutes=30)
