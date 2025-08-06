"""
INSTITUTIONAL-GRADE ICT PATTERN DETECTION
Professional implementation of Fair Value Gaps, Liquidity Sweeps, and Rejections
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PatternDetectionError(Exception):
    """Raised when pattern detection fails"""
    pass


def detect_fair_value_gaps(df: pd.DataFrame, min_gap_pct: float = 0.0003, 
                          max_gap_pct: float = 0.015) -> pd.DataFrame:
    """
    Detect 3-candle Fair Value Gaps with institutional-grade validation
    
    FVG Definition:
    - Bullish FVG: Candle 1 high < Candle 3 low (gap up)
    - Bearish FVG: Candle 1 low > Candle 3 high (gap down)
    
    Args:
        df: OHLCV DataFrame
        min_gap_pct: Minimum gap size as % of current price (filters noise)
        max_gap_pct: Maximum gap size as % of current price (filters extreme moves)
        
    Returns:
        DataFrame with fvg_bullish and fvg_bearish columns added
        
    Raises:
        PatternDetectionError: If detection fails
    """
    try:
        df = df.copy()
        
        # Validate input data
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise PatternDetectionError(f"Missing columns: {missing_cols}")
            
        if len(df) < 3:
            logger.warning("Insufficient data for FVG detection, need at least 3 candles")
            df['fvg_bullish'] = False
            df['fvg_bearish'] = False
            return df
            
        # Initialize FVG columns
        df['fvg_bullish'] = False
        df['fvg_bearish'] = False
        
        # 3-candle FVG logic: C1 vs C3 (skip middle candle C2)
        # Bullish FVG: C1.high < C3.low
        bullish_condition = df['high'].shift(2) < df['low']
        
        # Bearish FVG: C1.low > C3.high  
        bearish_condition = df['low'].shift(2) > df['high']
        
        # Calculate gap sizes as percentage of current price
        bullish_gap_size = (df['low'] - df['high'].shift(2)) / df['close']
        bearish_gap_size = (df['low'].shift(2) - df['high']) / df['close']
        
        # Apply size filters to remove noise and extreme moves
        valid_bullish_size = (bullish_gap_size >= min_gap_pct) & (bullish_gap_size <= max_gap_pct)
        valid_bearish_size = (bearish_gap_size >= min_gap_pct) & (bearish_gap_size <= max_gap_pct)
        
        # Final FVG detection
        df['fvg_bullish'] = bullish_condition & valid_bullish_size
        df['fvg_bearish'] = bearish_condition & valid_bearish_size
        
        # Additional validation: ensure price action makes sense
        # Bullish FVG should occur during upward movement
        bullish_price_context = df['close'] > df['close'].shift(3)  # Higher than 3 candles ago
        df.loc[df['fvg_bullish'], 'fvg_bullish'] = (
            df.loc[df['fvg_bullish'], 'fvg_bullish'] & 
            bullish_price_context.loc[df['fvg_bullish']]
        )
        
        # Bearish FVG should occur during downward movement
        bearish_price_context = df['close'] < df['close'].shift(3)  # Lower than 3 candles ago
        df.loc[df['fvg_bearish'], 'fvg_bearish'] = (
            df.loc[df['fvg_bearish'], 'fvg_bearish'] & 
            bearish_price_context.loc[df['fvg_bearish']]
        )
        
        fvg_bull_count = df['fvg_bullish'].sum()
        fvg_bear_count = df['fvg_bearish'].sum()
        logger.debug(f"FVG Detection: {fvg_bull_count} bullish, {fvg_bear_count} bearish")
        
        return df
        
    except Exception as e:
        raise PatternDetectionError(f"FVG detection failed: {str(e)}")


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 5, 
                           min_break_pct: float = 0.0002) -> pd.DataFrame:
    """
    Detect liquidity sweeps with professional validation
    
    Sweep Definition:
    - Bullish Sweep: Break below recent lows, then close higher (stop hunt + reversal)
    - Bearish Sweep: Break above recent highs, then close lower (stop hunt + reversal)
    
    Args:
        df: OHLCV DataFrame
        lookback: Number of candles to look back for highs/lows
        min_break_pct: Minimum break distance as % of price (filters noise)
        
    Returns:
        DataFrame with sweep_low and sweep_high columns added
        
    Raises:
        PatternDetectionError: If detection fails
    """
    try:
        df = df.copy()
        
        # Validate input
        required_cols = ['high', 'low', 'open', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise PatternDetectionError(f"Missing columns: {missing_cols}")
            
        if len(df) < lookback + 1:
            logger.warning(f"Insufficient data for sweep detection: {len(df)} < {lookback + 1}")
            df['sweep_low'] = False
            df['sweep_high'] = False
            return df
            
        # Initialize sweep columns
        df['sweep_low'] = False
        df['sweep_high'] = False
        
        # Detect sweeps using vectorized operations where possible
        for i in range(lookback, len(df)):
            current = df.iloc[i]
            recent_window = df.iloc[i-lookback:i]  # Last N candles (excluding current)
            
            if len(recent_window) == 0:
                continue
                
            recent_low = recent_window['low'].min()
            recent_high = recent_window['high'].max()
            
            # Calculate minimum break distances
            min_break_distance_low = recent_low * min_break_pct
            min_break_distance_high = recent_high * min_break_pct
            
            # Bullish Sweep: Break below recent low + close above open (reversal candle)
            breaks_low = current['low'] <= (recent_low - min_break_distance_low)
            reversal_bullish = current['close'] > current['open']  # Green candle
            
            if breaks_low and reversal_bullish:
                # Additional validation: ensure it's actually a meaningful sweep
                wick_size = current['close'] - current['low']  # Size of lower wick
                body_size = abs(current['close'] - current['open'])
                
                # Valid sweep should have decent wick showing rejection
                if wick_size >= body_size * 0.5:  # Wick at least 50% of body
                    df.iloc[i, df.columns.get_loc('sweep_low')] = True
                    
            # Bearish Sweep: Break above recent high + close below open (reversal candle)
            breaks_high = current['high'] >= (recent_high + min_break_distance_high)
            reversal_bearish = current['close'] < current['open']  # Red candle
            
            if breaks_high and reversal_bearish:
                # Additional validation: ensure it's actually a meaningful sweep
                wick_size = current['high'] - current['close']  # Size of upper wick
                body_size = abs(current['close'] - current['open'])
                
                # Valid sweep should have decent wick showing rejection
                if wick_size >= body_size * 0.5:  # Wick at least 50% of body
                    df.iloc[i, df.columns.get_loc('sweep_high')] = True
                    
        sweep_low_count = df['sweep_low'].sum()
        sweep_high_count = df['sweep_high'].sum()
        logger.debug(f"Sweep Detection: {sweep_low_count} low sweeps, {sweep_high_count} high sweeps")
        
        return df
        
    except Exception as e:
        raise PatternDetectionError(f"Liquidity sweep detection failed: {str(e)}")


def detect_rejection_patterns(df: pd.DataFrame, min_wick_ratio: float = 0.6,
                             min_body_ratio: float = 0.2) -> pd.DataFrame:
    """
    Detect rejection patterns (hammer/shooting star type candles)
    
    Rejection Definition:
    - Bullish Rejection: Long lower wick + close in upper part of range
    - Bearish Rejection: Long upper wick + close in lower part of range
    
    Args:
        df: OHLCV DataFrame
        min_wick_ratio: Minimum wick size as ratio of total candle range
        min_body_ratio: Minimum body size as ratio of total candle range (filters dojis)
        
    Returns:
        DataFrame with reject_low and reject_high columns added
        
    Raises:
        PatternDetectionError: If detection fails
    """
    try:
        df = df.copy()
        
        # Validate input
        required_cols = ['high', 'low', 'open', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise PatternDetectionError(f"Missing columns: {missing_cols}")
            
        # Initialize rejection columns
        df['reject_low'] = False
        df['reject_high'] = False
        
        # Calculate candle components
        candle_range = df['high'] - df['low']
        body_size = abs(df['close'] - df['open'])
        upper_wick = df['high'] - np.maximum(df['open'], df['close'])
        lower_wick = np.minimum(df['open'], df['close']) - df['low']
        
        # Filter out candles with zero range (avoid division by zero)
        valid_candles = candle_range > 0
        
        # Calculate ratios only for valid candles
        upper_wick_ratio = np.where(valid_candles, upper_wick / candle_range, 0)
        lower_wick_ratio = np.where(valid_candles, lower_wick / candle_range, 0)
        body_ratio = np.where(valid_candles, body_size / candle_range, 0)
        
        # Bullish Rejection Criteria:
        # 1. Long lower wick (shows rejection of lower prices)
        # 2. Decent body size (not a doji)
        # 3. Close in upper half of candle (bullish bias)
        close_position = np.where(valid_candles, 
                                 (df['close'] - df['low']) / candle_range, 0.5)
        
        bullish_rejection = (
            valid_candles &
            (lower_wick_ratio >= min_wick_ratio) &
            (body_ratio >= min_body_ratio) &
            (close_position >= 0.6) &  # Close in upper 40% of range
            (df['close'] >= df['open'])  # Green or equal close
        )
        
        # Bearish Rejection Criteria:
        # 1. Long upper wick (shows rejection of higher prices)
        # 2. Decent body size (not a doji)  
        # 3. Close in lower half of candle (bearish bias)
        bearish_rejection = (
            valid_candles &
            (upper_wick_ratio >= min_wick_ratio) &
            (body_ratio >= min_body_ratio) &
            (close_position <= 0.4) &  # Close in lower 40% of range
            (df['close'] <= df['open'])  # Red or equal close
        )
        
        df['reject_low'] = bullish_rejection
        df['reject_high'] = bearish_rejection
        
        reject_low_count = df['reject_low'].sum()
        reject_high_count = df['reject_high'].sum()
        logger.debug(f"Rejection Detection: {reject_low_count} low rejections, {reject_high_count} high rejections")
        
        return df
        
    except Exception as e:
        raise PatternDetectionError(f"Rejection pattern detection failed: {str(e)}")


def confirm_sweep_rejection(df: pd.DataFrame, lookahead: int = 3, 
                           confirmation_threshold: float = 0.002) -> pd.DataFrame:
    """
    Confirm sweep rejections with forward-looking price action
    
    This function validates that a sweep was actually rejected by checking
    if price moves in the opposite direction after the sweep.
    
    Args:
        df: DataFrame with sweep columns
        lookahead: Number of candles to look ahead for confirmation
        confirmation_threshold: Minimum price movement for confirmation (as % of price)
        
    Returns:
        DataFrame with confirmed_sweep_low and confirmed_sweep_high columns
        
    Raises:
        PatternDetectionError: If confirmation fails
    """
    try:
        df = df.copy()
        
        # Validate input
        if 'sweep_low' not in df.columns or 'sweep_high' not in df.columns:
            raise PatternDetectionError("Sweep columns not found. Run detect_liquidity_sweeps first.")
            
        # Initialize confirmation columns
        df['confirmed_sweep_low'] = False
        df['confirmed_sweep_high'] = False
        
        # Check each sweep for confirmation
        sweep_low_indices = df[df['sweep_low']].index
        sweep_high_indices = df[df['sweep_high']].index
        
        # Confirm bullish sweeps (sweep low followed by upward movement)
        for idx in sweep_low_indices:
            if idx >= len(df) - lookahead:
                continue  # Not enough future data
                
            sweep_price = df.loc[idx, 'low']
            future_highs = df.loc[idx+1:idx+lookahead+1, 'high']
            
            if len(future_highs) > 0:
                max_future_high = future_highs.max()
                price_recovery = (max_future_high - sweep_price) / sweep_price
                
                if price_recovery >= confirmation_threshold:
                    df.loc[idx, 'confirmed_sweep_low'] = True
                    
        # Confirm bearish sweeps (sweep high followed by downward movement)
        for idx in sweep_high_indices:
            if idx >= len(df) - lookahead:
                continue  # Not enough future data
                
            sweep_price = df.loc[idx, 'high']
            future_lows = df.loc[idx+1:idx+lookahead+1, 'low']
            
            if len(future_lows) > 0:
                min_future_low = future_lows.min()
                price_decline = (sweep_price - min_future_low) / sweep_price
                
                if price_decline >= confirmation_threshold:
                    df.loc[idx, 'confirmed_sweep_high'] = True
                    
        conf_low_count = df['confirmed_sweep_low'].sum()
        conf_high_count = df['confirmed_sweep_high'].sum()
        logger.debug(f"Sweep Confirmation: {conf_low_count} confirmed low sweeps, {conf_high_count} confirmed high sweeps")
        
        return df
        
    except Exception as e:
        raise PatternDetectionError(f"Sweep confirmation failed: {str(e)}")


def detect_all_patterns(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Detect all ICT patterns in one function with comprehensive error handling
    
    Args:
        df: OHLCV DataFrame
        **kwargs: Optional parameters for individual pattern detection functions
        
    Returns:
        DataFrame with all pattern columns added
        
    Raises:
        PatternDetectionError: If critical pattern detection fails
    """
    try:
        df = df.copy()
        
        # Validate OHLCV data first
        from .indicators import validate_ohlcv_data
        if not validate_ohlcv_data(df):
            raise PatternDetectionError("Invalid OHLCV data provided")
            
        logger.info(f"Starting pattern detection on {len(df)} candles")
        
        # Detect Fair Value Gaps
        df = detect_fair_value_gaps(df, **{k: v for k, v in kwargs.items() 
                                          if k in ['min_gap_pct', 'max_gap_pct']})
        
        # Detect Liquidity Sweeps  
        df = detect_liquidity_sweeps(df, **{k: v for k, v in kwargs.items()
                                           if k in ['lookback', 'min_break_pct']})
        
        # Detect Rejection Patterns
        df = detect_rejection_patterns(df, **{k: v for k, v in kwargs.items()
                                             if k in ['min_wick_ratio', 'min_body_ratio']})
        
        # Confirm sweep rejections (optional - requires future data)
        try:
            df = confirm_sweep_rejection(df, **{k: v for k, v in kwargs.items()
                                               if k in ['lookahead', 'confirmation_threshold']})
        except Exception as e:
            logger.warning(f"Sweep confirmation failed, skipping: {e}")
            
        # Pattern summary
        patterns_found = {
            'fvg_bullish': df['fvg_bullish'].sum(),
            'fvg_bearish': df['fvg_bearish'].sum(),
            'sweep_low': df['sweep_low'].sum(),
            'sweep_high': df['sweep_high'].sum(),
            'reject_low': df['reject_low'].sum(),
            'reject_high': df['reject_high'].sum()
        }
        
        logger.info(f"Pattern detection complete: {patterns_found}")
        return df
        
    except PatternDetectionError:
        raise
    except Exception as e:
        raise PatternDetectionError(f"Pattern detection failed: {str(e)}")


def get_pattern_summary(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get summary of detected patterns
    
    Args:
        df: DataFrame with pattern columns
        
    Returns:
        Dictionary with pattern counts
    """
    pattern_cols = ['fvg_bullish', 'fvg_bearish', 'sweep_low', 'sweep_high', 
                   'reject_low', 'reject_high']
    
    summary = {}
    for col in pattern_cols:
        if col in df.columns:
            summary[col] = df[col].sum()
        else:
            summary[col] = 0
            
    return summary