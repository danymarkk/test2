import pandas as pd


def compute_adaptive_confluence_long(row):
    """
    Compute confluence score for long signals (0-4)
    """
    score = 0
    if row.get("bias") == "bullish":
        score += 1
    if row.get("sweep_low", False):
        score += 1
    if row.get("fvg_bullish", False):
        score += 1
    if row.get("reject_low", False):
        score += 1
    return score


def compute_adaptive_confluence_short(row):
    """
    Compute confluence score for short signals (0-4)
    """
    score = 0
    if row.get("bias") == "bearish":
        score += 1
    if row.get("sweep_high", False):
        score += 1
    if row.get("fvg_bearish", False):
        score += 1
    if row.get("reject_high", False):
        score += 1
    return score


def generate_signals(
    df: pd.DataFrame, long_threshold: int = 3, short_threshold: int = 3
):
    """
    ADAPTIVE CONFLUENCE ICT signal generation with configurable thresholds
    """
    df = df.copy()

    # 24/7 TRADING: Remove session filter for global crypto markets
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["trading_session"] = True  # Trade all hours for crypto

    # Ensure all required columns exist with proper defaults
    required_cols = {
        "bias": df.get("bias", "neutral"),
        "sweep_low": df.get("sweep_low", False),
        "sweep_high": df.get("sweep_high", False),
        "reject_low": df.get("reject_low", False),
        "reject_high": df.get("reject_high", False),
        "fvg_bullish": df.get("fvg_bullish", False),
        "fvg_bearish": df.get("fvg_bearish", False),
    }

    for col, default in required_cols.items():
        if col not in df.columns:
            if isinstance(default, bool):
                df[col] = False
            else:
                df[col] = default

    # ADAPTIVE CONFLUENCE SCORING SYSTEM
    # Compute confluence scores for each row
    df["confluence_long"] = df.apply(compute_adaptive_confluence_long, axis=1)
    df["confluence_short"] = df.apply(compute_adaptive_confluence_short, axis=1)

    # Create filter columns based on thresholds
    df["strong_confluence_long"] = df["confluence_long"] >= long_threshold
    df["strong_confluence_short"] = df["confluence_short"] >= short_threshold

    # REVERTED: Volatility filter removed - session + smart exits was optimal
    # df['atr_median_20'] = df['atr'].rolling(window=20).median()
    # df['high_volatility'] = df['atr'] > (df['atr_median_20'] * 0.8)

    # REVERTED: 2+ signals destroyed win rate (46.6% → 36.8%) - back to 3+
    df["long_signal"] = df["strong_confluence_long"] & df["trading_session"]
    df["short_signal"] = df["strong_confluence_short"] & df["trading_session"]

    # (Logging responsibility moved to callers; keep this function pure.)

    return df
