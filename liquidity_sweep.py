# liquidity_sweep.py

import pandas as pd
import numpy as np


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Detect liquidity sweeps using local highs/lows relative to past N candles.
    - sweep_high: current high exceeds all previous N highs
    - sweep_low: current low breaks all previous N lows
    """
    df = df.copy()
    df["sweep_high"] = False
    df["sweep_low"] = False
    df["sweep_low_price"] = np.nan
    df["sweep_high_price"] = np.nan

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback : i]
        highest = window["high"].max()
        lowest = window["low"].min()
        if df.iloc[i]["high"] > highest:
            df.at[i, "sweep_high"] = True
            df.at[i, "sweep_high_price"] = df.iloc[i]["high"]
        if df.iloc[i]["low"] < lowest:
            df.at[i, "sweep_low"] = True
            df.at[i, "sweep_low_price"] = df.iloc[i]["low"]

    return df
