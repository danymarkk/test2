"""Shared HTF merge utility.

merge_htf aligns higher-timeframe (HTF) candles to LTF rows so each LTF row
uses the most recent closed HTF bar. Uses pandas.merge_asof with backward
direction, with optional tolerance based on LTF timeframe.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


def _ltf_tolerance_minutes(timeframe: str) -> int:
    tf = str(timeframe).lower().strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    return 60  # default 1h


def merge_htf(
    ltf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    ltf_timeframe: str = "30m",
    method: Literal["backward", "nearest"] = "backward",
) -> pd.DataFrame:
    if ltf_df.empty or htf_df.empty:
        return ltf_df.copy()
    ltf = ltf_df.copy()
    htf = htf_df.copy()
    ltf["timestamp"] = pd.to_datetime(ltf["timestamp"])  # ensure ts dtype
    htf["timestamp"] = pd.to_datetime(htf["timestamp"])  # ensure ts dtype

    tol = pd.Timedelta(minutes=_ltf_tolerance_minutes(ltf_timeframe))
    merged = pd.merge_asof(
        ltf.sort_values("timestamp"),
        htf.sort_values("timestamp")[
            ["timestamp", "open", "high", "low", "close", "volume"]
        ].rename(columns={
            "open": "htf_open",
            "high": "htf_high",
            "low": "htf_low",
            "close": "htf_close",
            "volume": "htf_volume",
        }),
        on="timestamp",
        direction=method,
        tolerance=tol,
    )
    return merged


