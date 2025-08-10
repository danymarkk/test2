from data_loader import load_data
from indicators_core import (
    compute_vwap,
    compute_atr,
    compute_ema,
    generate_bias,
)
from fair_value_gap import detect_fvgs
from liquidity_sweep import detect_liquidity_sweeps
from signal_engine import generate_signals
from execution_engine import simulate_trades
from rejection_confirmation import confirm_rejection
from params import get_config, get_strategy_params, get_logger


def run_test():
    log = get_logger("main")
    log.info("Starting ICT Trading Strategy Backtest...")

    cfg = get_config()
    params = get_strategy_params(cfg)
    # logger already created above
    pair = cfg.get("primary_pair", "ETH/USDT")
    timeframe = params["timeframe"]
    lookback = cfg.get("lookback", 2000)

    df_30m, df_2h = load_data(pair, lookback=lookback, timeframe=timeframe)
    log.info(
        "Loaded %d 30m candles and %d 2H candles",
        len(df_30m),
        len(df_2h),
    )

    # Indicators
    df_30m["vwap"] = compute_vwap(df_30m)
    df_30m["atr"] = compute_atr(df_30m)
    df_30m["ema_21"] = compute_ema(df_30m)
    df_30m["bias"] = generate_bias(df_30m, df_2h)
    log.info("Technical indicators calculated")

    # ICT patterns
    df_30m = detect_fvgs(df_30m)
    df_30m = detect_liquidity_sweeps(df_30m)
    df_30m["reject_low"] = confirm_rejection(df_30m, sweep_col="sweep_low")
    df_30m["reject_high"] = confirm_rejection(df_30m, sweep_col="sweep_high")
    log.info("ICT patterns detected")

    # Pattern counts
    log.info("Pattern Counts:")
    log.info("   FVG Bullish: %d", int(df_30m["fvg_bullish"].sum()))
    log.info("   FVG Bearish: %d", int(df_30m["fvg_bearish"].sum()))
    log.info("   Sweep Low: %d", int(df_30m["sweep_low"].sum()))
    log.info("   Sweep High: %d", int(df_30m["sweep_high"].sum()))
    log.info("   Reject Low: %d", int(df_30m["reject_low"].sum()))
    log.info("   Reject High: %d", int(df_30m["reject_high"].sum()))
    log.info("   Bias Bullish: %d", int((df_30m["bias"] == "bullish").sum()))
    log.info("   Bias Bearish: %d", int((df_30m["bias"] == "bearish").sum()))

    # Signals
    long_th = params["long_threshold"]
    short_th = params["short_threshold"]
    df_signals = generate_signals(
        df_30m, long_threshold=long_th, short_threshold=short_th
    )
    log.info("Signals generated")

    # Signal counts
    long_count = int(df_signals["long_signal"].sum())
    short_count = int(df_signals["short_signal"].sum())
    log.info("Signal Counts:")
    log.info("   Long Signals: %d", long_count)
    log.info("   Short Signals: %d", short_count)
    log.info("   Total Signals: %d", long_count + short_count)

    if long_count == 0 and short_count == 0:
        log.warning("No signals generated. Check pattern detection logic.")
        return

    log.info("Running trade simulation...")
    results = simulate_trades(
        df_signals,
        atr_multiplier=cfg.get("atr_multiplier", 0.75),
        rr_ratio=cfg.get("rr_ratio", 2.0),
        max_hold_bars=cfg.get("max_hold_bars", 8),
        slippage_pct=cfg.get("slippage_pct", 0.05),
        commission_pct=cfg.get("commission_pct", 0.1),
    )

    if len(results) > 0:
        log.info("Trade simulation complete")
        log.info("Results Summary:")
        log.info("   Total Trades: %d", len(results))
        log.info(
            "   Successful Exits: %d",
            int(results["exit_reason"].notna().sum()),
        )
        log.info(
            "%s",
            results[
                [
                    "timestamp",
                    "entry_price",
                    "exit_price",
                    "exit_reason",
                    "pnl",
                ]
            ].tail(),
        )
        results.to_csv("signals.csv", index=False)
        log.info("Results saved to signals.csv")
    else:
        log.warning("No trades executed. Check execution engine logic.")


if __name__ == "__main__":
    run_test()
