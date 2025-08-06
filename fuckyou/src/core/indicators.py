"""
PRODUCTION-GRADE TECHNICAL INDICATORS
Bulletproof implementations with institutional-level error handling
"""

import pandas as pd
import numpy as np
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class IndicatorCalculationError(Exception):
    """Raised when indicator calculation fails"""
    pass


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range with robust error handling
    
    Args:
        df: OHLCV DataFrame
        period: ATR calculation period
        
    Returns:
        ATR Series with guaranteed positive values
        
    Raises:
        IndicatorCalculationError: If calculation fails
    """
    try:
        if len(df) < period:
            raise IndicatorCalculationError(f"Insufficient data: {len(df)} < {period}")
            
        # Validate required columns
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise IndicatorCalculationError(f"Missing columns: {missing_cols}")
        
        # Calculate True Range components
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        # True Range is the maximum of the three
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR using rolling mean
        atr = true_range.rolling(window=period, min_periods=1).mean()
        
        # Ensure ATR is never zero (minimum 0.01% of current price)
        min_atr = df['close'] * 0.0001
        atr = np.maximum(atr, min_atr)
        
        # Validate output
        if atr.isna().any() or (atr <= 0).any():
            logger.warning("ATR contains invalid values, applying safety corrections")
            atr = atr.fillna(method='ffill').fillna(min_atr)
            atr = np.maximum(atr, min_atr)
            
        return atr
        
    except Exception as e:
        raise IndicatorCalculationError(f"ATR calculation failed: {str(e)}")


def compute_ema(df: pd.DataFrame, period: int = 21, price_col: str = 'close') -> pd.Series:
    """
    Calculate Exponential Moving Average with validation
    
    Args:
        df: DataFrame with price data
        period: EMA period
        price_col: Column to calculate EMA on
        
    Returns:
        EMA Series
        
    Raises:
        IndicatorCalculationError: If calculation fails
    """
    try:
        if price_col not in df.columns:
            raise IndicatorCalculationError(f"Column '{price_col}' not found")
            
        if len(df) == 0:
            raise IndicatorCalculationError("Empty DataFrame")
            
        ema = df[price_col].ewm(span=period, adjust=False, min_periods=1).mean()
        
        # Validate output
        if ema.isna().any():
            logger.warning(f"EMA-{period} contains NaN values, forward filling")
            ema = ema.fillna(method='ffill')
            
        return ema
        
    except Exception as e:
        raise IndicatorCalculationError(f"EMA-{period} calculation failed: {str(e)}")


def compute_vwap(df: pd.DataFrame, reset_daily: bool = False) -> pd.Series:
    """
    Calculate Volume Weighted Average Price
    
    Args:
        df: OHLCV DataFrame
        reset_daily: Whether to reset VWAP daily (for long-running calculations)
        
    Returns:
        VWAP Series
        
    Raises:
        IndicatorCalculationError: If calculation fails
    """
    try:
        required_cols = ['high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise IndicatorCalculationError(f"Missing columns for VWAP: {missing_cols}")
            
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Handle zero volume
        volume = df['volume'].copy()
        volume = np.where(volume <= 0, 1, volume)  # Replace zero/negative volume with 1
        
        # Calculate VWAP
        if reset_daily and 'timestamp' in df.columns:
            # Reset VWAP at start of each day
            df_copy = df.copy()
            # Ensure timestamp is datetime type before using .dt accessor
            if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            df_copy['date'] = df_copy['timestamp'].dt.date
            
            vwap_list = []
            for date in df_copy['date'].unique():
                day_data = df_copy[df_copy['date'] == date]
                day_typical = (day_data['high'] + day_data['low'] + day_data['close']) / 3
                day_volume = np.where(day_data['volume'] <= 0, 1, day_data['volume'])
                
                cumulative_tp_vol = (day_typical * day_volume).cumsum()
                cumulative_vol = day_volume.cumsum()
                day_vwap = cumulative_tp_vol / cumulative_vol
                
                vwap_list.append(day_vwap)
                
            vwap = pd.concat(vwap_list)
        else:
            # Standard cumulative VWAP
            cumulative_tp_vol = (typical_price * volume).cumsum()
            cumulative_vol = volume.cumsum()
            vwap = cumulative_tp_vol / cumulative_vol
            
        # Validate output
        if vwap.isna().any() or (vwap <= 0).any():
            logger.warning("VWAP contains invalid values, applying corrections")
            vwap = vwap.fillna(method='ffill').fillna(typical_price)
            vwap = np.where(vwap <= 0, typical_price, vwap)
            
        return vwap
        
    except Exception as e:
        raise IndicatorCalculationError(f"VWAP calculation failed: {str(e)}")


def generate_market_bias(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, 
                        ema_period: int = 21) -> pd.Series:
    """
    Generate market bias using Higher Time Frame (HTF) analysis
    
    Args:
        df_ltf: Lower timeframe DataFrame
        df_htf: Higher timeframe DataFrame 
        ema_period: EMA period for bias calculation
        
    Returns:
        Bias series ('bullish', 'bearish', 'neutral')
        
    Raises:
        IndicatorCalculationError: If calculation fails
    """
    try:
        if len(df_htf) < ema_period:
            logger.warning(f"Insufficient HTF data for bias: {len(df_htf)} < {ema_period}")
            return pd.Series(['neutral'] * len(df_ltf), index=df_ltf.index)
            
        # Calculate HTF indicators
        htf_ema = compute_ema(df_htf, ema_period)
        htf_vwap = compute_vwap(df_htf)
        
        # Get latest HTF values
        if len(htf_ema) == 0 or len(htf_vwap) == 0:
            return pd.Series(['neutral'] * len(df_ltf), index=df_ltf.index)
            
        latest_price = df_htf['close'].iloc[-1]
        latest_ema = htf_ema.iloc[-1]
        latest_vwap = htf_vwap.iloc[-1]
        
        # Bias determination logic
        price_above_ema = latest_price > latest_ema
        price_above_vwap = latest_price > latest_vwap
        
        # Check EMA slope (using last 3 periods)
        if len(htf_ema) >= 3:
            ema_slope_bullish = htf_ema.iloc[-1] > htf_ema.iloc[-3]
        else:
            ema_slope_bullish = True  # Default assumption
            
        # Determine bias
        bullish_signals = sum([price_above_ema, price_above_vwap, ema_slope_bullish])
        
        if bullish_signals >= 2:
            bias = 'bullish'
        elif bullish_signals <= 1:
            bias = 'bearish'
        else:
            bias = 'neutral'
            
        # Apply bias to all LTF candles (in live trading, this would be updated periodically)
        return pd.Series([bias] * len(df_ltf), index=df_ltf.index)
        
    except Exception as e:
        logger.error(f"Bias calculation failed: {str(e)}")
        return pd.Series(['neutral'] * len(df_ltf), index=df_ltf.index)


def compute_all_indicators(df: pd.DataFrame, df_htf: Optional[pd.DataFrame] = None,
                          atr_period: int = 14, ema_fast: int = 9, ema_slow: int = 21) -> pd.DataFrame:
    """
    Compute all required indicators in one function with error handling
    
    Args:
        df: Main DataFrame (OHLCV)
        df_htf: Higher timeframe DataFrame for bias
        atr_period: ATR calculation period
        ema_fast: Fast EMA period
        ema_slow: Slow EMA period
        
    Returns:
        DataFrame with all indicators added
        
    Raises:
        IndicatorCalculationError: If critical indicators fail
    """
    try:
        df = df.copy()
        
        # Core indicators (these must succeed)
        df['atr'] = compute_atr(df, atr_period)
        df['ema_fast'] = compute_ema(df, ema_fast)
        df['ema_slow'] = compute_ema(df, ema_slow)
        
        # Volume indicators for proper filtering
        df['volume_sma_20'] = df['volume'].rolling(20, min_periods=1).mean()
        df['volume_avg_20'] = df['volume_sma_20']  # Alias for consistency
        
        # VWAP (with fallback)
        try:
            df['vwap'] = compute_vwap(df)
        except IndicatorCalculationError as e:
            logger.warning(f"VWAP calculation failed, using close price: {e}")
            df['vwap'] = df['close']
            
        # Market bias (with fallback)
        try:
            if df_htf is not None:
                df['bias'] = generate_market_bias(df, df_htf, ema_slow)
            else:
                # Simple bias using fast vs slow EMA
                df['bias'] = np.where(df['ema_fast'] > df['ema_slow'], 'bullish', 'bearish')
        except Exception as e:
            logger.warning(f"Bias calculation failed, using neutral: {e}")
            df['bias'] = 'neutral'
            
        logger.info(f"Successfully computed indicators for {len(df)} candles")
        return df
        
    except IndicatorCalculationError:
        # Re-raise indicator errors as they're already properly formatted
        raise
    except Exception as e:
        raise IndicatorCalculationError(f"Indicator computation failed: {str(e)}")


# Validation functions
def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """
    Validate OHLCV data integrity
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Check required columns
    if not all(col in df.columns for col in required_cols):
        return False
        
    # Check for reasonable OHLC relationships
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    
    if invalid_ohlc.any():
        logger.warning(f"Found {invalid_ohlc.sum()} candles with invalid OHLC relationships")
        return False
        
    # Check for negative or zero prices
    price_cols = ['open', 'high', 'low', 'close']
    invalid_prices = (df[price_cols] <= 0).any(axis=1)
    
    if invalid_prices.any():
        logger.warning(f"Found {invalid_prices.sum()} candles with invalid prices")
        return False
        
    return True