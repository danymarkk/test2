"""BTC/USDT Speed Tester - Thin wrapper calling canonical speed_tester CLI."""

import sys
from speed_tester import get_config, get_logger, run_speed_test


if __name__ == "__main__":
    cfg = get_config()
    if cfg.get("pairs") == ["ETH/USDT"]:
        get_logger("speed_tester").info("ETH-only mode: skipping BTC/USDT")
        sys.exit(0)
    run_speed_test(symbol="BTC/USDT")
