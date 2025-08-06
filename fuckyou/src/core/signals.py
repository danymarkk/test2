"""
INSTITUTIONAL-GRADE SIGNAL GENERATION
Professional ICT confluence scoring and signal validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Import performance tracking
try:
    from ..utils.performance_tracker import get_performance_tracker
except ImportError:
    def get_performance_tracker():
        return None


class SignalGenerationError(Exception):
    """Raised when signal generation fails"""
    pass


class ICTSignalEngine:
    """
    Professional ICT Signal Engine with institutional-grade confluence scoring
    """
    
    def __init__(self, demand_threshold: int = 3, supply_threshold: int = 3,
                 volume_filter: bool = True, momentum_filter: bool = True,
                 signal_deduplication: bool = True, max_signals_per_minute: int = 1):
        """
        Initialize ICT Signal Engine for Spot Trading
        
        Args:
            demand_threshold: Minimum confluence score for demand signals (BUY)
            supply_threshold: Minimum confluence score for supply signals (SELL if holding)
            volume_filter: Enable volume-based signal filtering
            momentum_filter: Enable momentum-based signal filtering
            signal_deduplication: Enable signal deduplication (ChatGPT recommendation)
            max_signals_per_minute: Max signals per symbol per minute (ChatGPT: 1)
        """
        self.demand_threshold = demand_threshold
        self.supply_threshold = supply_threshold
        self.volume_filter = volume_filter
        self.momentum_filter = momentum_filter
        self.signal_deduplication = signal_deduplication
        self.max_signals_per_minute = max_signals_per_minute
        
        # GLOBAL deduplication: track last signal time per symbol
        self.last_signal_time = {}  # symbol -> timestamp
        
        logger.info(f"ICT Signal Engine initialized: Demand={demand_threshold}, Supply={supply_threshold}, Dedup={signal_deduplication}")
    
    def calculate_confluence_score(self, row: pd.Series, direction: str) -> Tuple[int, Dict[str, bool]]:
        """
        Calculate confluence score for ICT spot trading signals
        
        Args:
            row: DataFrame row with pattern and indicator data
            direction: 'demand' (bullish patterns for BUY) or 'supply' (bearish patterns for SELL)
            
        Returns:
            Tuple of (confluence_score, pattern_breakdown)
        """
        try:
            score = 0
            breakdown = {}
            
            if direction == 'demand':
                # Bullish ICT patterns indicating institutional demand (BUY signal)
                patterns = {
                    'fvg_bullish': row.get('fvg_bullish', False),
                    'sweep_low': row.get('sweep_low', False),
                    'reject_low': row.get('reject_low', False),
                    'bias_bullish': row.get('bias', 'neutral') == 'bullish',
                    'price_above_vwap': row.get('close', 0) > row.get('vwap', 0),
                    'ema_alignment': row.get('ema_fast', 0) > row.get('ema_slow', 0)
                }
                
                # Enhanced pattern weighting for demand zones
                weights = {
                    'fvg_bullish': 1.5,      # FVGs show institutional demand
                    'sweep_low': 2.0,        # Sweeps grab liquidity before demand
                    'reject_low': 1.5,       # Price rejection shows strong demand
                    'bias_bullish': 1.0,     # HTF bias confirmation
                    'price_above_vwap': 0.5, # Above institutional VWAP
                    'ema_alignment': 0.5     # Trend confluence
                }
                
            else:  # supply
                # Bearish ICT patterns indicating institutional supply (SELL signal if holding)
                patterns = {
                    'fvg_bearish': row.get('fvg_bearish', False),
                    'sweep_high': row.get('sweep_high', False),
                    'reject_high': row.get('reject_high', False),
                    'bias_bearish': row.get('bias', 'neutral') == 'bearish',
                    'price_below_vwap': row.get('close', 0) < row.get('vwap', 0),
                    'ema_alignment': row.get('ema_fast', 0) < row.get('ema_slow', 0)
                }
                
                weights = {
                    'fvg_bearish': 1.5,      # FVGs show institutional supply
                    'sweep_high': 2.0,       # Sweeps grab liquidity before supply
                    'reject_high': 1.5,      # Price rejection shows strong supply
                    'bias_bearish': 1.0,     # HTF bias confirmation
                    'price_below_vwap': 0.5, # Below institutional VWAP
                    'ema_alignment': 0.5     # Trend confluence
                }
            
            # Calculate weighted score
            for pattern, active in patterns.items():
                breakdown[pattern] = active
                if active:
                    score += weights.get(pattern, 1.0)
            
            # Round to reasonable precision
            score = round(score, 1)
            
            return score, breakdown
            
        except Exception as e:
            logger.error(f"Confluence calculation failed for {direction}: {e}")
            return 0, {}
    
    def calculate_signal_quality(self, row: pd.Series, direction: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculate signal quality score (0-5) based on market conditions
        
        Args:
            row: DataFrame row with market data
            direction: 'demand' (bullish for BUY) or 'supply' (bearish for SELL)
            
        Returns:
            Tuple of (quality_score, quality_breakdown)
        """
        try:
            quality = 0.0
            breakdown = {}
            
            # ATR-based volatility score (0-1.5 points)
            atr = row.get('atr', 0)
            close = row.get('close', 1)
            if close > 0 and atr > 0:
                atr_pct = (atr / close) * 100
                if atr_pct >= 3.0:
                    volatility_score = 1.5
                elif atr_pct >= 2.0:
                    volatility_score = 1.0
                elif atr_pct >= 1.0:
                    volatility_score = 0.5
                else:
                    volatility_score = 0.0
            else:
                volatility_score = 0.0
            
            quality += volatility_score
            breakdown['volatility'] = volatility_score
            
            # Volume score (0-1.5 points) - requires volume history
            volume_score = 0.0
            try:
                current_vol = row.get('volume', 0)
                avg_vol = row.get('volume_avg_20', current_vol)  # Assume this is pre-calculated
                
                if avg_vol > 0 and current_vol > 0:
                    vol_ratio = current_vol / avg_vol
                    if vol_ratio >= 2.5:
                        volume_score = 1.5
                    elif vol_ratio >= 1.8:
                        volume_score = 1.0
                    elif vol_ratio >= 1.3:
                        volume_score = 0.5
                    elif vol_ratio < 0.7:
                        volume_score = -0.5  # Penalty for low volume
            except:
                pass
                
            quality += volume_score
            breakdown['volume'] = volume_score
            
            # Candle strength score (0-1 points)
            candle_strength = 0.0
            try:
                open_price = row.get('open', 0)
                high = row.get('high', 0)
                low = row.get('low', 0)
                close_price = row.get('close', 0)
                
                if high > low > 0:
                    body_size = abs(close_price - open_price)
                    total_range = high - low
                    body_ratio = body_size / total_range if total_range > 0 else 0
                    
                    # Strong directional candle for spot trading
                    # FIXED: Use consistent direction handling
                    bullish_direction = direction in ('demand', 'long', 'buy')
                    bearish_direction = direction in ('supply', 'short', 'sell')
                    
                    if bullish_direction and close_price > open_price and body_ratio > 0.6:
                        candle_strength = 1.0  # Strong bullish candle for BUY signal
                    elif bearish_direction and close_price < open_price and body_ratio > 0.6:
                        candle_strength = 1.0  # Strong bearish candle for SELL signal
                    elif body_ratio > 0.4:
                        candle_strength = 0.5
            except:
                pass
                
            quality += candle_strength
            breakdown['candle_strength'] = candle_strength
            
            # Pattern confluence bonus (0-1 points)
            confluence_bonus = 0.0
            try:
                # FIXED: Handle both demand/supply AND long/short directions
                bullish = direction in ('demand', 'long', 'buy')
                bearish = direction in ('supply', 'short', 'sell')
                
                # Count number of active patterns
                if bullish:
                    patterns = [
                        row.get('fvg_bullish', False),
                        row.get('sweep_low', False),
                        row.get('reject_low', False)
                    ]
                else:
                    patterns = [
                        row.get('fvg_bearish', False),
                        row.get('sweep_high', False),
                        row.get('reject_high', False)
                    ]
                
                active_patterns = sum(patterns)
                if active_patterns >= 3:
                    confluence_bonus = 1.0
                elif active_patterns >= 2:
                    confluence_bonus = 0.5
            except:
                pass
                
            quality += confluence_bonus
            breakdown['confluence_bonus'] = confluence_bonus
            
            # Market structure score (0-1 points)
            structure_score = 0.0
            try:
                bias = row.get('bias', 'neutral')
                # FIXED: Use bullish/bearish variables for consistency
                if bullish and bias == 'bullish':
                    structure_score = 1.0
                elif bearish and bias == 'bearish':
                    structure_score = 1.0
                elif bias == 'neutral':
                    structure_score = 0.3  # Neutral is acceptable
            except:
                pass
                
            quality += structure_score
            breakdown['market_structure'] = structure_score
            
            # Cap quality at 5.0
            quality = min(quality, 5.0)
            
            return quality, breakdown
            
        except Exception as e:
            logger.error(f"Quality calculation failed for {direction}: {e}")
            return 0.0, {}
    
    def apply_volume_filter(self, row: pd.Series) -> bool:
        """
        Apply volume-based filtering
        
        Args:
            row: DataFrame row with volume data
            
        Returns:
            True if signal passes volume filter
        """
        if not self.volume_filter:
            return True
            
        try:
            current_vol = row.get('volume', 0)
            avg_vol = row.get('volume_avg_20', current_vol)
            
            if avg_vol <= 0:
                return False
                
            vol_ratio = current_vol / avg_vol
            
            # Require at least 120% of average volume
            return vol_ratio >= 1.2
            
        except Exception as e:
            logger.warning(f"Volume filter error: {e}")
            return False
    
    def apply_momentum_filter(self, row: pd.Series, direction: str) -> bool:
        """
        Apply momentum-based filtering for spot trading
        
        Args:
            row: DataFrame row with price data
            direction: 'demand' (bullish momentum for BUY) or 'supply' (bearish momentum for SELL)
            
        Returns:
            True if signal passes momentum filter
        """
        if not self.momentum_filter:
            return True
            
        try:
            # Simple momentum check using EMA relationship
            ema_fast = row.get('ema_fast', 0)
            ema_slow = row.get('ema_slow', 0)
            
            if direction == 'demand':
                return ema_fast > ema_slow  # Bullish momentum for BUY
            else:  # supply
                return ema_fast < ema_slow  # Bearish momentum for SELL
                
        except Exception as e:
            logger.warning(f"Momentum filter error: {e}")
            return False
    
    def generate_signals(self, df: pd.DataFrame, symbol: str = "UNKNOWN/USDT") -> pd.DataFrame:
        """
        Generate ICT signals with comprehensive filtering
        
        Args:
            df: DataFrame with indicators and patterns
            
        Returns:
            DataFrame with signal columns added
            
        Raises:
            SignalGenerationError: If signal generation fails
        """
        try:
            df = df.copy()
            
            # Validate required columns
            required_cols = ['close', 'atr']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise SignalGenerationError(f"Missing required columns: {missing_cols}")
            
            # Initialize signal columns for spot trading
            df['demand_signal'] = False   # BUY signals (bullish ICT patterns)
            df['supply_signal'] = False   # SELL signals (bearish ICT patterns)
            df['confluence_demand'] = 0.0
            df['confluence_supply'] = 0.0
            df['quality_demand'] = 0.0
            df['quality_supply'] = 0.0
            df['signal_reason'] = None
            
            # Volume average now computed in indicators module
            
            signals_generated = 0
            signals_filtered = 0
            
            # Generate signals for each row
            for idx in range(len(df)):
                row = df.iloc[idx]
                
                # Calculate confluence scores
                conf_demand, breakdown_demand = self.calculate_confluence_score(row, 'demand')
                conf_supply, breakdown_supply = self.calculate_confluence_score(row, 'supply')
                
                df.iloc[idx, df.columns.get_loc('confluence_demand')] = conf_demand
                df.iloc[idx, df.columns.get_loc('confluence_supply')] = conf_supply
                
                # Calculate quality scores
                qual_demand, qual_breakdown_demand = self.calculate_signal_quality(row, 'demand')
                qual_supply, qual_breakdown_supply = self.calculate_signal_quality(row, 'supply')
                
                df.iloc[idx, df.columns.get_loc('quality_demand')] = qual_demand
                df.iloc[idx, df.columns.get_loc('quality_supply')] = qual_supply
                
                # Check for demand signals (BUY opportunities) with ChatGPT quality filters
                if conf_demand >= self.demand_threshold:
                    # Calculate volume ratio and signal strength for additional filtering
                    volume_ratio = row.get('volume', 1) / row.get('volume_sma_20', 1) if row.get('volume_sma_20', 0) > 0 else 1.0
                    signal_strength = 'strong' if conf_demand >= 4 else 'medium' if conf_demand >= 3 else 'weak'
                    
                    # Apply all quality filters: basic + volume + momentum + ChatGPT recommendations
                    if (self.apply_volume_filter(row) and 
                        self.apply_momentum_filter(row, 'demand') and
                        volume_ratio >= 1.2 and  # ChatGPT: 70th percentile volume
                        signal_strength in ['medium', 'strong']):  # ChatGPT: drop weak signals
                        
                        df.iloc[idx, df.columns.get_loc('demand_signal')] = True
                        df.iloc[idx, df.columns.get_loc('signal_reason')] = (
                            f"DEMAND: conf={conf_demand}, qual={qual_demand:.1f}")
                        signals_generated += 1
                        
                        # Log signal data for performance tracking
                        tracker = get_performance_tracker()
                        if tracker:
                            signal_data = {
                                'quality': qual_demand,
                                'confluence': conf_demand,
                                'htf_bias': row.get('bias', 'neutral'),
                                'volume_ratio': row.get('volume', 1) / row.get('volume_sma_20', 1) if row.get('volume_sma_20', 0) > 0 else 1.0,
                                'current_price': row.get('close', 0),
                                'atr': row.get('atr', 0),
                                'patterns': breakdown_demand,
                                'filters_passed': [f for f, passed in [
                                    ('volume', self.apply_volume_filter(row)),
                                    ('momentum', self.apply_momentum_filter(row, 'demand'))
                                ] if passed],
                                'filters_failed': [f for f, passed in [
                                    ('volume', self.apply_volume_filter(row)),
                                    ('momentum', self.apply_momentum_filter(row, 'demand'))
                                ] if not passed],
                                'entry_ready': True,
                                'strength': 'strong' if conf_demand >= 4 else 'medium' if conf_demand >= 3 else 'weak'
                            }
                            
                            # Don't log here - will log after deduplication
                            # tracker.log_signal(symbol, 'demand', signal_data)
                        
                        logger.debug(f"Demand signal at index {idx}: confluence={conf_demand}, quality={qual_demand:.1f}")
                    else:
                        signals_filtered += 1
                        
                        # Log filtered demand signal for analysis (with ChatGPT quality filters)
                        tracker = get_performance_tracker()
                        if tracker:
                            volume_ratio = row.get('volume', 1) / row.get('volume_sma_20', 1) if row.get('volume_sma_20', 0) > 0 else 1.0
                            signal_strength = 'strong' if conf_demand >= 4 else 'medium' if conf_demand >= 3 else 'weak'
                            
                            filter_reasons = []
                            if not self.apply_volume_filter(row):
                                filter_reasons.append('volume')
                            if not self.apply_momentum_filter(row, 'demand'):
                                filter_reasons.append('momentum')
                            if volume_ratio < 1.2:
                                filter_reasons.append('volume_ratio')
                            if signal_strength == 'weak':
                                filter_reasons.append('weak_strength')
                            
                            signal_data = {
                                'quality': qual_demand,
                                'confluence': conf_demand,
                                'htf_bias': row.get('bias', 'neutral'),
                                'volume_ratio': row.get('volume', 1) / row.get('volume_sma_20', 1) if row.get('volume_sma_20', 0) > 0 else 1.0,
                                'current_price': row.get('close', 0),
                                'atr': row.get('atr', 0),
                                'patterns': breakdown_demand,
                                'filters_passed': [],
                                'filters_failed': filter_reasons,
                                'entry_ready': False,
                                'strength': 'filtered',
                                'filter_reason': f"Failed: {', '.join(filter_reasons)}"
                            }
                            
                            # Don't spam signal_analysis.jsonl with filtered signals
                            # tracker.log_signal(symbol, 'demand_filtered', signal_data)
                        
                        logger.debug(f"Demand signal filtered at index {idx}")
                
                # SPOT TRADING: Supply signals disabled (only demand/BUY signals for spot)
                # ChatGPT Analysis: Supply signals are 50% of signal flood - removing them
                logger.debug(f"Supply signal calculation skipped for spot trading at index {idx}")
            
            logger.info(f"Signal generation complete: {signals_generated} signals generated, {signals_filtered} filtered")
            
            # Always call deduplication to ensure signal logging works
            # Even if deduplication is disabled, we still need logging
            df = self._deduplicate_signals(df, symbol)
            
            return df
            
        except Exception as e:
            raise SignalGenerationError(f"Signal generation failed: {str(e)}")
    
    def _deduplicate_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Deduplicate signals per ChatGPT recommendation: 1 signal per symbol per minute
        Keep the signal with highest confluence score within each minute
        
        Args:
            df: DataFrame with signals
            symbol: Symbol being processed
            
        Returns:
            DataFrame with deduplicated signals
        """
        try:
            if 'demand_signal' not in df.columns:
                return df
                
            # Find all demand signals
            demand_signals = df[df['demand_signal'] == True].copy()
            
            if len(demand_signals) == 0:
                return df
                
            # Simple deduplication: keep only the last signal (most recent) with highest confluence
            # This avoids datetime parsing issues while still reducing signal flood
            if len(demand_signals) > self.max_signals_per_minute:
                # Sort by confluence_demand and keep only the best ones
                demand_signals_sorted = demand_signals.sort_values('confluence_demand', ascending=False)
                best_signals = demand_signals_sorted.head(self.max_signals_per_minute).index.tolist()
                
                # Reset all demand signals, then set only the best ones
                df['demand_signal'] = False
                df.loc[best_signals, 'demand_signal'] = True
                
                original_count = len(demand_signals)
                final_count = len(best_signals)
                
                logger.info(f"Signal deduplication: {original_count} → {final_count} demand signals for {symbol} (kept highest confluence)")
            
            # Log only the final deduplicated signals to signal_analysis.jsonl
            self._log_final_signals(df, symbol)
            
            return df
            
        except Exception as e:
            logger.error(f"Signal deduplication failed: {e}")
            logger.error(f"  DataFrame shape: {df.shape if hasattr(df, 'shape') else 'Unknown'}")
            logger.error(f"  DataFrame columns: {df.columns.tolist() if hasattr(df, 'columns') else 'Unknown'}")
            # Return original DataFrame on error to prevent complete failure
            return df  # Return original if deduplication fails
    
    def _log_final_signals(self, df: pd.DataFrame, symbol: str):
        """
        Log only the final deduplicated signals to signal_analysis.jsonl
        WITH GLOBAL TIME-BASED DEDUPLICATION
        
        Args:
            df: DataFrame with final signals after deduplication
            symbol: Symbol being processed
        """
        try:
            import time
            current_time = time.time()
            
            # GLOBAL deduplication: check if we've logged this symbol recently
            # Reduce from 60s to 10s to allow more frequent logging
            if symbol in self.last_signal_time:
                time_since_last = current_time - self.last_signal_time[symbol]
                if time_since_last < 10:  # Less than 10 seconds
                    logger.debug(f"GLOBAL DEDUP: Skipping {symbol} signal (last logged {time_since_last:.1f}s ago)")
                    return
            
            tracker = get_performance_tracker()
            if not tracker:
                return
                
            # Find final demand signals that survived deduplication
            final_signals = df[df['demand_signal'] == True]
            
            if len(final_signals) > 0:
                # Log only the FIRST signal (highest confluence after dedup)
                row = final_signals.iloc[0]
                
                # Get signal data for logging with FULL pattern data flow
                signal_data = {
                    'quality': row.get('quality_demand', 0),
                    'confluence': row.get('confluence_demand', 0),
                    'htf_bias': row.get('bias', 'neutral'),
                    'volume_ratio': row.get('volume', 1) / row.get('volume_sma_20', 1) if row.get('volume_sma_20', 0) > 0 else 1.0,
                    'current_price': row.get('close', 0),
                    'atr': row.get('atr', 0),
                    # CRITICAL FIX: Extract actual pattern data from DataFrame
                    'patterns': {
                        'fvg_bullish': bool(row.get('fvg_bullish', False)),
                        'fvg_bearish': bool(row.get('fvg_bearish', False)),
                        'sweep_low': bool(row.get('sweep_low', False)),
                        'sweep_high': bool(row.get('sweep_high', False)),
                        'reject_low': bool(row.get('reject_low', False)),
                        'reject_high': bool(row.get('reject_high', False))
                    },
                    # CRITICAL FIX: Add missing risk calculations
                    'risk_reward_ratio': row.get('risk_reward_ratio', 0),
                    'stop_distance': row.get('stop_distance', 0),
                    'take_profit_distance': row.get('take_profit_distance', 0),
                    'filters_passed': ['volume', 'momentum'],  # Simplified
                    'filters_failed': [],
                    'entry_ready': True,
                    'strength': 'strong' if row.get('confluence_demand', 0) >= 4 else 'medium' if row.get('confluence_demand', 0) >= 3 else 'weak'
                }
                
                # Log only if passes global time check
                tracker.log_signal(symbol, 'demand', signal_data)
                
                # Update last signal time for this symbol
                self.last_signal_time[symbol] = current_time
                logger.debug(f"GLOBAL DEDUP: Logged {symbol} signal, next allowed in 60s")
                
        except Exception as e:
            logger.error(f"Final signal logging failed: {e}")
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get summary of generated spot trading signals
        
        Args:
            df: DataFrame with signal columns
            
        Returns:
            Dictionary with signal counts and statistics
        """
        try:
            summary = {
                'total_demand_signals': df['demand_signal'].sum() if 'demand_signal' in df.columns else 0,
                'total_supply_signals': df['supply_signal'].sum() if 'supply_signal' in df.columns else 0,
                'avg_confluence_demand': df['confluence_demand'].mean() if 'confluence_demand' in df.columns else 0,
                'avg_confluence_supply': df['confluence_supply'].mean() if 'confluence_supply' in df.columns else 0,
                'avg_quality_demand': df['quality_demand'].mean() if 'quality_demand' in df.columns else 0,
                'avg_quality_supply': df['quality_supply'].mean() if 'quality_supply' in df.columns else 0,
                'total_candles': len(df)
            }
            
            # Safe division to avoid division by zero
            total_signals = summary['total_demand_signals'] + summary['total_supply_signals']
            summary['signal_frequency'] = (total_signals / len(df) * 100) if len(df) > 0 else 0.0
            
            return summary
            
        except Exception as e:
            logger.error(f"Signal summary calculation failed: {e}")
            # Return safe default summary instead of empty dict
            return {
                'total_demand_signals': 0,
                'total_supply_signals': 0,
                'avg_confluence_demand': 0.0,
                'avg_confluence_supply': 0.0,
                'avg_quality_demand': 0.0,
                'avg_quality_supply': 0.0,
                'total_candles': 0,
                'signal_frequency': 0.0
            }


# REMOVED: Unused factory function - signal engine created directly in strategy


# REMOVED: Legacy backwards compatibility function - dead code removed