"""Centralized parameters and logging configuration.

Provides:
- get_config(): load and validate config/settings.json with sane defaults
- get_strategy_params(): slippage/commission/ATR/thresholds/timeframes
- get_logger(name): initialize and return a module logger with consistent formatting
"""

from __future__ import annotations

import logging
from typing import Any, Dict

try:
    from config import load_config, get_default_config, validate_config
except Exception:  # pragma: no cover
    load_config = None
    get_default_config = None
    validate_config = None


_LOGGING_CONFIGURED = False


def get_config() -> Dict[str, Any]:
    """Load config from config/settings.json, falling back to defaults and validating bounds.

    Returns a merged dictionary of settings.
    """
    cfg: Dict[str, Any] = {}
    if load_config is not None:
        try:
            cfg = load_config("config/settings.json")
        except Exception:
            # fallback to defaults if file missing or invalid
            if get_default_config is not None:
                cfg = get_default_config()
            else:
                cfg = {}

    # If the config package is not available, provide minimal sane defaults
    if not cfg:
        cfg = {
            "risk_per_trade": 0.02,
            "rr_ratio": 2.0,
            "atr_multiplier": 0.75,
            "timeframe": "30m",
            "htf": "2h",
            "confluence_threshold": 3,
            "max_hold_bars": 8,
            "slippage_pct": 0.05,
            "commission_pct": 0.1,
            "use_dynamic_slippage": False,
            "pairs": ["ETH/USDT", "BTC/USDT", "XRP/USDT"],
            "primary_pair": "ETH/USDT",
            "lookback": 2000,
            "exchange_id": "binance",
            "default_type": "spot",
            "testnet": False,
        }

    # Validate bounds if available
    if validate_config is not None:
        try:
            cfg = validate_config(cfg)
        except Exception:
            # keep cfg as-is if validation raises; callers will handle
            pass

    # Assert presence and types for critical keys
    errors: list[str] = []

    # timeframe
    timeframe = cfg.get("timeframe")
    if not isinstance(timeframe, str) or timeframe.strip() == "":
        errors.append("Missing or invalid 'timeframe' (expected non-empty string)")

    # thresholds: accept either confluence_threshold OR both long/short
    has_confluence = isinstance(cfg.get("confluence_threshold"), (int, float))
    has_long_short = isinstance(cfg.get("long_threshold"), (int, float)) and isinstance(
        cfg.get("short_threshold"), (int, float)
    )
    if not (has_confluence or has_long_short):
        errors.append(
            "Missing thresholds: provide 'confluence_threshold' or both 'long_threshold' and 'short_threshold'"
        )

    # rr_ratio > 0
    try:
        rr_ratio = float(cfg.get("rr_ratio"))
        if rr_ratio <= 0:
            raise ValueError
    except Exception:
        errors.append("Missing or invalid 'rr_ratio' (expected float > 0)")

    # atr_multiplier > 0
    try:
        atr_multiplier = float(cfg.get("atr_multiplier"))
        if atr_multiplier <= 0:
            raise ValueError
    except Exception:
        errors.append("Missing or invalid 'atr_multiplier' (expected float > 0)")

    # slippage_pct >= 0
    try:
        slippage_pct = float(cfg.get("slippage_pct"))
        if slippage_pct < 0:
            raise ValueError
    except Exception:
        errors.append("Missing or invalid 'slippage_pct' (expected float >= 0)")

    # commission_pct >= 0
    try:
        commission_pct = float(cfg.get("commission_pct"))
        if commission_pct < 0:
            raise ValueError
    except Exception:
        errors.append("Missing or invalid 'commission_pct' (expected float >= 0)")

    # symbols list
    pairs = cfg.get("pairs")
    if (
        not isinstance(pairs, list)
        or len(pairs) == 0
        or not all(isinstance(s, str) and s for s in pairs)
    ):
        errors.append("Missing or invalid 'pairs' (expected non-empty list of symbols)")

    if errors:
        raise ValueError("Config assertions failed: " + "; ".join(errors))
    return cfg


def get_strategy_params(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a dictionary of commonly used strategy parameters from config."""
    if cfg is None:
        cfg = get_config()
    long_threshold = cfg.get("long_threshold", cfg.get("confluence_threshold", 3))
    short_threshold = cfg.get("short_threshold", cfg.get("confluence_threshold", 3))
    return {
        "slippage_pct": float(cfg.get("slippage_pct", 0.05)),
        "commission_pct": float(cfg.get("commission_pct", 0.1)),
        "use_dynamic_slippage": bool(cfg.get("use_dynamic_slippage", False)),
        "atr_multiplier": float(cfg.get("atr_multiplier", 0.75)),
        "rr_ratio": float(cfg.get("rr_ratio", 2.0)),
        "long_threshold": int(long_threshold),
        "short_threshold": int(short_threshold),
        "timeframe": str(cfg.get("timeframe", "30m")),
        "htf_timeframe": str(cfg.get("htf", "2h")),
        # 24/7 session routing and overrides
        "enabled_sessions": list(cfg.get("enabled_sessions", ["Asia", "London", "New York"])),
        "per_session": dict(cfg.get("per_session", {})),
        "quality_gates": dict(cfg.get("quality_gates", {"min_atr_percent": 0.0})),
    }


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger with a consistent format.

    Idempotent: configures root logging only once.
    """
    global _LOGGING_CONFIGURED
    if not _LOGGING_CONFIGURED:
        cfg = get_config()
        level = logging.DEBUG if bool(cfg.get("debug_mode", False)) else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        _LOGGING_CONFIGURED = True
    return logging.getLogger(name)
