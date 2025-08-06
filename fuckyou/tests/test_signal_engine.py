import sys
import os
import pytest
import pandas as pd

# Add project directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from signal_engine import generate_signals


def test_generate_signals():
    # Sample data for testing
    data = {
        'timestamp': ['2025-08-04 12:00:00', '2025-08-04 12:05:00'],
        'fvg_bullish': [True, True],  # Ensure confluence
        'sweep_low': [True, True],    # Ensure confluence
        'reject_low': [True, True],   # Ensure confluence
        'bias': ['bullish', 'bullish'],  # Ensure confluence
        'atr': [0.01, 0.02],
    }
    df = pd.DataFrame(data)

    # Generate signals
    df_signals = generate_signals(df, long_threshold=3, short_threshold=3)

    # Assertions
    assert df_signals['long_signal'].iloc[0] is True, "Expected long signal for first row"
    assert df_signals['long_signal'].iloc[1] is False, "Expected no long signal for second row"

if __name__ == "__main__":
    pytest.main()