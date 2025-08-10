import pandas as pd


def confirm_rejection(
    df: pd.DataFrame,
    sweep_col: str = "sweep_low",
    lookahead: int = 3,
    buffer_pct: float = 0.0015,
) -> pd.Series:
    """
    Confirm rejection after a liquidity sweep.

    A rejection is confirmed if:
    - The close after the sweep (within lookahead bars) moves back above (or below) the sweep candle's open
    - There is no break below (or above) the sweep wick

    Parameters:
    - df: OHLCV DataFrame
    - sweep_col: 'sweep_low' or 'sweep_high'
    - lookahead: number of candles to check for rejection
    - buffer_pct: small price buffer for wick invalidation

    Returns:
    - Boolean Series marking rejection confirmation
    """
    rejections = []
    for i in range(len(df)):
        if not df[sweep_col].iloc[i]:
            rejections.append(False)
            continue

        sweep_candle = df.iloc[i]
        confirm = False

        for j in range(1, lookahead + 1):
            if i + j >= len(df):
                break
            future = df.iloc[i + j]

            if sweep_col == "sweep_low":
                below_wick = future["low"] < sweep_candle["low"] * (1 - buffer_pct)
                close_above_open = future["close"] > sweep_candle["open"]
                if not below_wick and close_above_open:
                    confirm = True
                    break

            elif sweep_col == "sweep_high":
                above_wick = future["high"] > sweep_candle["high"] * (1 + buffer_pct)
                close_below_open = future["close"] < sweep_candle["open"]
                if not above_wick and close_below_open:
                    confirm = True
                    break

        rejections.append(confirm)

    return pd.Series(rejections, index=df.index)


# Usage:
# df['reject_low'] = confirm_rejection(df, sweep_col='sweep_low')
# df['reject_high'] = confirm_rejection(df, sweep_col='sweep_high')
