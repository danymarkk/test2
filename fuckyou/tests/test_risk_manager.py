import sys
import os
import pytest

# Add project directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from risk_management.risk_manager import RiskManager


def test_calculate_stop_loss():
    risk_manager = RiskManager()
    entry_price = 100
    atr_value = 2
    stop_loss_long = risk_manager.calculate_stop_loss(entry_price, atr_value, 'long')
    stop_loss_short = risk_manager.calculate_stop_loss(entry_price, atr_value, 'short')
    assert stop_loss_long == 97, "Expected stop loss for long position to be 97"
    assert stop_loss_short == 103, "Expected stop loss for short position to be 103"


def test_calculate_take_profit():
    risk_manager = RiskManager()
    entry_price = 100
    stop_loss = 97
    take_profit_long = risk_manager.calculate_take_profit(entry_price, stop_loss, 'long')
    take_profit_short = risk_manager.calculate_take_profit(entry_price, stop_loss, 'short')
    assert take_profit_long == 106, "Expected take profit for long position to be 106"
    assert take_profit_short == 94, "Expected take profit for short position to be 94"


def test_calculate_position_size():
    risk_manager = RiskManager()
    account_balance = 1000
    entry_price = 100
    stop_loss = 97
    position_size = risk_manager.calculate_position_size(account_balance, entry_price, stop_loss)
    assert round(position_size, 2) == 333.33, "Expected position size to be approximately 333.33"

if __name__ == "__main__":
    pytest.main()