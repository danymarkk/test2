"""Diagnostics aggregation and export utilities for per-trade metrics.

Provides a small, process-local accumulator API so any caller (backtester,
paper trader, sweep tool) can record per-trade diagnostics and export them
later in a consistent schema.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional

import pandas as pd


# Required schema for diagnostics exports
REQUIRED_FIELDS: List[str] = [
    "timestamp_entry",
    "timestamp_exit",
    "asset",
    "session",
    "side",
    "entry_price",
    "exit_price",
    "r",
    "mae_r",
    "mfe_r",
    "exit_reason",
    "slippage_r",
    "fees_r",
    "atr_multiplier",
    "rr_ratio",
    "slippage_pct",
    "commission_pct",
    "lookback_hours",
    "offset_hours",
    "run_id",
]


_REC: List[Dict] = []


def reset() -> None:
    """Clear any accumulated diagnostics in-memory."""
    _REC.clear()


def record_trade(trade: Dict) -> None:
    """Append a single trade diagnostics record.

    The provided dict MUST contain all REQUIRED_FIELDS. This keeps this small
    and predictable. Callers should construct the dict explicitly.
    """
    # Shallow copy to avoid caller mutation
    _REC.append(dict(trade))


def finalize_and_export(path_csv: str, path_json: Optional[str] = None) -> None:
    """Validate accumulated diagnostics and write CSV/JSON.

    Creates parent directories as needed. Resets accumulator after export.
    """
    df = pd.DataFrame(_REC)

    # Sanity: required fields present and non-null
    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing diagnostics fields: {missing}")
    if df[REQUIRED_FIELDS].isnull().any().any():
        raise RuntimeError("Diagnostics contain nulls in required fields")

    # Normalize datetime-like columns to ISO strings for JSON safety
    for col in ("timestamp_entry", "timestamp_exit"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(path_csv) or ".", exist_ok=True)

    # Write CSV
    df.to_csv(path_csv, index=False)

    # Write JSON if requested
    if path_json:
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    # Reset after successful export
    reset()


# Backward-compatible helper if older callers still use it
def export_per_trade(df: pd.DataFrame, csv_path: str, json_path: str) -> None:
    """Deprecated: prefer record_trade()/finalize_and_export().

    Attempts to coerce common columns and export minimal fields.
    """
    # Best-effort coercion
    compat = pd.DataFrame()
    compat["timestamp_entry"] = pd.to_datetime(
        df.get("timestamp") or df.get("entry_time")
    )
    compat["timestamp_exit"] = pd.to_datetime(
        df.get("exit_time") or df.get("timestamp")
    )
    compat["asset"] = df.get("asset") or df.get("symbol") or "ETH/USDT"
    compat["session"] = df.get("session") or "MULTI"
    compat["side"] = df.get("side") or df.get("signal_type") or ""
    compat["entry_price"] = df.get("entry_price")
    compat["exit_price"] = df.get("exit_price")
    compat["r"] = df.get("pnl") if "pnl" in df.columns else df.get("pnl_r")
    compat["mae_r"] = df.get("mae_r", 0.0)
    compat["mfe_r"] = df.get("mfe_r", 0.0)
    compat["exit_reason"] = df.get("exit_reason") or ""
    compat["slippage_r"] = df.get("slippage_r", 0.0)
    compat["fees_r"] = df.get("fees_r", 0.0)
    # Fill remaining required fields with placeholders
    for field in [
        "atr_multiplier",
        "rr_ratio",
        "slippage_pct",
        "commission_pct",
        "lookback_hours",
        "offset_hours",
        "run_id",
    ]:
        compat[field] = compat.get(field, None)

    # Validate and export
    missing = [c for c in REQUIRED_FIELDS if c not in compat.columns]
    if missing:
        raise RuntimeError(f"Missing diagnostics fields: {missing}")
    if compat[REQUIRED_FIELDS].isnull().any().any():
        raise RuntimeError("Diagnostics contain nulls in required fields")

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    compat.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(compat.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

