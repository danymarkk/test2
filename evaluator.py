import pandas as pd

import matplotlib.pyplot as plt


def evaluate_signals(file_path="signals.csv"):
    # Load data
    df = pd.read_csv(file_path, parse_dates=["timestamp"])

    if df.empty:
        try:
            from params import get_logger as _get_logger

            _get_logger("evaluator").warning("No signals found in the CSV.")
        except Exception:
            pass
        return

    # Drop rows with missing PnL or Exit Reason
    df = df.dropna(subset=["pnl", "exit_reason"])

    # Compute basic stats
    total_trades = len(df)
    winning_trades = df[df["pnl"] > 0]
    losing_trades = df[df["pnl"] <= 0]
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

    avg_r = df["pnl"].mean()
    avg_win = winning_trades["pnl"].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades["pnl"].mean() if not losing_trades.empty else 0

    # Equity curve
    df["cum_return"] = df["pnl"].cumsum()
    df["drawdown"] = df["cum_return"] - df["cum_return"].cummax()
    max_drawdown = df["drawdown"].min()

    # Print metrics
    try:
        _log = _get_logger("evaluator")
        _log.info("Strategy Evaluation Metrics")
        _log.info("Total Trades      : %d", total_trades)
        _log.info("Win Rate          : %.2f%%", win_rate * 100)
        _log.info("Average R         : %.4f", avg_r)
        _log.info("Average Win       : %.4f", avg_win)
        _log.info("Average Loss      : %.4f", avg_loss)
        _log.info("Max Drawdown (R)  : %.4f", max_drawdown)
    except Exception:
        pass

    # Optional: Plot equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["cum_return"], label="Equity Curve", color="blue")
    plt.fill_between(
        df["timestamp"],
        df["cum_return"],
        df["cum_return"].cummax(),
        color="red",
        alpha=0.3,
        label="Drawdown",
    )
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return (R)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_signals()
