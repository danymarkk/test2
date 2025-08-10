"""Shared session helpers for UTC session gating.

Definitions (half-open intervals):
- Asia:    [00:00, 08:00)
- London:  [08:00, 16:00)
- New York:[16:00, 24:00)
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


SESSIONS: Dict[str, Tuple[int, int]] = {
    "Asia": (0, 8),
    "London": (8, 16),
    "New York": (16, 24),
}


def in_session(ts: pd.Timestamp) -> str:
    hour = int(pd.to_datetime(ts).hour)
    if 0 <= hour < 8:
        return "Asia"
    if 8 <= hour < 16:
        return "London"
    return "New York"


def filter_session(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    normalized = (
        "New York" if name.replace(" ", "").lower() == "newyork" else name
    )
    if normalized not in SESSIONS:
        raise ValueError(f"Unknown session: {name}")
    start, end = SESSIONS[normalized]
    hours = pd.to_datetime(df["timestamp"]).dt.hour
    mask = (hours >= start) & (hours < end)
    df_out = df.loc[mask].copy()
    # Seam guard: drop dupes, sort, reset index
    if not df_out.empty:
        df_out = (
            df_out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        )
    return df_out

