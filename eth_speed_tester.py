"""ETH/USDT Speed Tester - Thin wrapper over canonical speed_tester"""

from speed_tester import run_speed_test


if __name__ == "__main__":
    run_speed_test(symbol="ETH/USDT")
