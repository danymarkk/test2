import pandas as pd


def detect_fvgs(
    df: pd.DataFrame, min_gap: float = 0.0002, max_gap: float = 0.02
) -> pd.DataFrame:
    """
    Detect 3-candle Fair Value Gaps (FVGs):
    - Bullish: Candle 1 high < Candle 3 low
    - Bearish: Candle 1 low > Candle 3 high
    """
    if df.empty or len(df) < 3 or df[["high", "low", "close"]].isnull().any().any():
        raise ValueError("Invalid data for FVG detection")
    df = df.copy()

    # Corrected 3-candle logic
    bullish_fvg = df["high"].shift(2) < df["low"]  # C1 high < C3 low
    bearish_fvg = df["low"].shift(2) > df["high"]  # C1 low > C3 high

    # Size of the gap from C1 high to C3 low, relative to current close
    gap_bull_size = abs(df["low"] - df["high"].shift(2)) / df["close"]
    gap_bear_size = abs(df["high"] - df["low"].shift(2)) / df["close"]

    # Use unified size filter
    valid_bull = (gap_bull_size >= min_gap) & (gap_bull_size <= max_gap)
    valid_bear = (gap_bear_size >= min_gap) & (gap_bear_size <= max_gap)

    df["fvg_bullish"] = bullish_fvg & valid_bull
    df["fvg_bearish"] = bearish_fvg & valid_bear

    return df
