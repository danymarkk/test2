"""Core technical indicators for ICT strategy"""
import pandas as pd
import numpy as np
import logging


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    if df.empty or df[["high", "low", "close", "volume"]].isnull().any().any():
        raise ValueError("Invalid data for VWAP calculation")
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()
    return cumulative_tp_vol / cumulative_vol


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty or df[["high", "low", "close"]].isnull().any().any():
        raise ValueError("Invalid data for ATR calculation")
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_ema(df: pd.DataFrame, period: int = 21) -> pd.Series:
    return df["close"].ewm(span=period, adjust=False).mean()


def generate_bias(df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> pd.Series:
    """
    Time-aligned bias calculation using HTF VWAP.

    For each LTF candle timestamp, look back to the most recent HTF candle's
    VWAP and set bias bullish if LTF close > HTF VWAP, else bearish.
    """
    if df_ltf.empty or df_htf.empty:
        return pd.Series(["bullish"] * len(df_ltf), index=df_ltf.index)

    htf = df_htf[["timestamp", "high", "low", "close", "volume"]].copy()
    htf = htf.sort_values("timestamp")
    htf["vwap_htf"] = compute_vwap(htf)

    ltf = df_ltf[["timestamp", "close"]].copy().sort_values("timestamp")

    # Merge as-of to align latest HTF vwap at or before each LTF timestamp
    aligned = pd.merge_asof(
        ltf,
        htf[["timestamp", "vwap_htf"]],
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("12H"),
    )

    # If initial rows have NaN vwap_htf (before first HTF bar), forward/back fill
    aligned["vwap_htf"] = aligned["vwap_htf"].ffill().bfill()
    # Telemetry: warn if any remaining NaNs (should be none after fill)
    try:
        if aligned["vwap_htf"].isna().any():
            nulls = int(aligned["vwap_htf"].isna().sum())
            logging.getLogger("indicators_core").warning(
                "generate_bias: %d NaNs in aligned HTF VWAP after merge_asof",
                nulls,
            )
    except Exception:
        pass

    bias_bool = aligned["close"] > aligned["vwap_htf"]
    return bias_bool.replace({True: "bullish", False: "bearish"}).reindex(df_ltf.index)


# Example usage in pipeline:
# df_1h['vwap'] = compute_vwap(df_1h)
# df_1h['atr'] = compute_atr(df_1h)
# df_1h['ema_21'] = compute_ema(df_1h)
# df_1h['bias'] = generate_bias(df_1h, df_4h)
