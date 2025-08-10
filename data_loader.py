"""Data loading module for fetching OHLCV data from Binance (SPOT by default).

Supports configurable exchange/type via config when available.
"""

import ccxt
import pandas as pd
import time
from typing import Tuple
from params import get_logger

try:
    # Optional: centralized configuration
    from config import load_config
except Exception:  # pragma: no cover - config is optional for this module
    load_config = None


def safe_exchange_call(func, *args, **kwargs):
    """Simple retry wrapper for exchange calls"""
    logger = get_logger("data_loader")
    for attempt in range(3):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == 2:
                raise e
            logger.warning(
                "Exchange call failed (attempt %d/3): %s",
                attempt + 1,
                e,
            )
            time.sleep(1 + attempt)  # Exponential backoff


def _create_exchange():
    """Create a ccxt exchange instance based on config, defaulting to SPOT binance.

    Returns a configured ccxt client with rate limiting enabled.
    """
    config = load_config() if load_config else {}
    exchange_id = config.get("exchange_id", "binance")
    default_type = config.get("default_type", config.get("defaultType", "spot"))

    options = {"defaultType": default_type}
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True, "options": options})
    # Unified testnet toggle for consistency with futures scaffold
    try:
        exchange.set_sandbox_mode(bool(config.get("testnet", False)))
    except Exception:
        pass
    try:
        exchange.load_markets()
    except Exception:
        # non-fatal for data pulls
        pass
    return exchange


def load_data(
    pair: str = "ETH/USDT",
    lookback: int = 2000,
    timeframe: str = "30m",
    time_offset_hours: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    exchange = _create_exchange()

    import time as time_module

    # Calculate time range for the request
    # If time_offset_hours is provided, go back further in time
    current_time = time_module.time() * 1000  # Current time in milliseconds
    offset_ms = time_offset_hours * 60 * 60 * 1000  # Convert hours to ms
    end_time = int(current_time - offset_ms)

    # Binance limits: 1000 for most timeframes, 1500 for some
    # Cap the request to avoid getting default data
    max_limit = 1000 if lookback > 1000 else lookback

    # Fetch data with time constraints if offset is provided
    if time_offset_hours > 0:
        ohlcv_30m = safe_exchange_call(
            exchange.fetch_ohlcv,
            pair,
            timeframe=timeframe,
            limit=max_limit,
            params={"endTime": end_time},
        )
    else:
        ohlcv_30m = safe_exchange_call(
            exchange.fetch_ohlcv,
            pair,
            timeframe=timeframe,
            limit=max_limit,
        )

    df_30m = pd.DataFrame(
        ohlcv_30m, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df_30m["timestamp"] = pd.to_datetime(df_30m["timestamp"], unit="ms")

    get_logger("data_loader").info(
        "Loaded %d %s candles for %s (requested: %d, offset: %dh)",
        len(df_30m),
        timeframe,
        pair,
        lookback,
        time_offset_hours,
    )

    # For HTF bias, use 2H candles (4x 30m) instead of 4H
    # Guard against zero-limit which would return empty data
    htf_limit = max(1, max_limit // 4)

    if time_offset_hours > 0:
        ohlcv_2h = safe_exchange_call(
            exchange.fetch_ohlcv,
            pair,
            timeframe="2h",
            limit=htf_limit,
            params={"endTime": end_time},
        )
    else:
        ohlcv_2h = safe_exchange_call(
            exchange.fetch_ohlcv,
            pair,
            timeframe="2h",
            limit=htf_limit,
        )

    df_2h = pd.DataFrame(
        ohlcv_2h, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df_2h["timestamp"] = pd.to_datetime(df_2h["timestamp"], unit="ms")

    get_logger("data_loader").info(
        "Loaded %d 2h candles for %s (htf_limit: %d, offset: %dh)",
        len(df_2h),
        pair,
        htf_limit,
        time_offset_hours,
    )

    return df_30m, df_2h
