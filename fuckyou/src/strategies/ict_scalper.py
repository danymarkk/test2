"""
INSTITUTIONAL ICT SCALPING STRATEGY
Professional implementation of Inner Circle Trader concepts for crypto scalping
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..core.indicators import compute_all_indicators, IndicatorCalculationError
from ..core.patterns import detect_all_patterns, PatternDetectionError
from ..core.signals import ICTSignalEngine, SignalGenerationError
from ..core.risk_manager import InstitutionalRiskManager, Position
from ..trading.market_data import MarketDataProvider
from ..config.settings import ScalperBotConfig

logger = logging.getLogger(__name__)


class ICTScalpingStrategyError(Exception):
    """Raised when ICT scalping strategy operations fail"""
    pass


class ICTScalpingStrategy:
    """
    Professional ICT Scalping Strategy
    
    Core Concepts:
    - Fair Value Gaps (FVGs): Price imbalances indicating institutional activity
    - Liquidity Sweeps: Stop-hunt patterns before reversals
    - Rejection Patterns: Demand/supply zones showing price rejection
    - Market Structure: Higher timeframe bias and trend alignment
    
    Features:
    - Multi-timeframe analysis
    - Dynamic confluence scoring
    - Quality-based position sizing
    - Professional risk management
    - Performance tracking and optimization
    """
    
    def __init__(self, config: ScalperBotConfig, 
                 market_data: MarketDataProvider,
                 risk_manager: InstitutionalRiskManager):
        """
        Initialize ICT scalping strategy
        
        Args:
            config: Strategy configuration
            market_data: Market data provider
            risk_manager: Risk management system
        """
        self.config = config
        self.market_data = market_data
        self.risk_manager = risk_manager
        
        # Initialize signal engine for spot trading with FIXED config
        self.signal_engine = ICTSignalEngine(
            demand_threshold=int(config.trading.confluence.main_strategy_demand),  # FIX: Convert to int
            supply_threshold=int(config.trading.confluence.main_strategy_supply),  # FIX: Convert to int 
            volume_filter=config.trading.volume_filter_enabled,
            momentum_filter=config.trading.momentum_filter_enabled,
            signal_deduplication=config.trading.signal_deduplication,
            max_signals_per_minute=config.trading.max_signals_per_minute_per_symbol
        )
        
        # Strategy state
        self.last_signal_times: Dict[str, datetime] = {}
        self.performance_metrics: Dict[str, float] = {}
        
        # CRITICAL FIX: Restore per-candle deduplication to prevent signal flooding
        self._last_signal_candle_ts: Dict[str, datetime] = {}
        
        # Pattern detection parameters
        self.pattern_params = {
            'min_gap_pct': 0.0005,  # Minimum FVG size (0.05%)
            'max_gap_pct': 0.02,    # Maximum FVG size (2%)
            'lookback': 5,          # Sweep lookback period
            'min_break_pct': 0.0003, # Minimum sweep break (0.03%)
            'min_wick_ratio': 0.5,  # Minimum wick ratio for rejections
            'min_body_ratio': 0.2   # Minimum body ratio for rejections
        }
        
        logger.info("ICT Scalping Strategy initialized")
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Perform complete ICT analysis on a symbol
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            Analysis results or None if failed
            
        Raises:
            ICTScalpingStrategyError: If analysis fails
        """
        try:
            logger.debug(f"Starting ICT analysis for {symbol}")
            
            # Fetch market data for multiple timeframes
            timeframes = {
                'main': self.config.trading.timeframe,
                'htf': self.config.trading.htf_timeframe
            }
            
            data = {}
            for tf_name, tf in timeframes.items():
                # Fix cache issue: disable cache for main TF to avoid duplicate signals
                use_cache = tf_name != 'main'  # Don't cache main timeframe
                cache_ttl = 5.0 if tf_name == 'main' else 30.0  # Short cache for main TF
                
                df = self.market_data.fetch_ohlcv(
                    symbol, 
                    tf, 
                    limit=self.config.trading.lookback_candles,
                    use_cache=use_cache,
                    cache_ttl=cache_ttl
                )
                
                if df is None or len(df) < 20:
                    logger.warning(f"Insufficient {tf_name} data for {symbol}")
                    return None
                    
                data[tf_name] = df
            
            # REMOVED: Redundant candle timestamp deduplication (signal engine handles all deduplication)
            
            # Compute indicators
            try:
                main_df = compute_all_indicators(
                    data['main'], 
                    data.get('htf'),
                    atr_period=14,
                    ema_fast=9,
                    ema_slow=21
                )
            except IndicatorCalculationError as e:
                logger.error(f"Indicator calculation failed for {symbol}: {e}")
                return None
            
            # Detect ICT patterns
            try:
                main_df = detect_all_patterns(main_df, **self.pattern_params)
            except PatternDetectionError as e:
                logger.error(f"Pattern detection failed for {symbol}: {e}")
                return None
            
            # CRITICAL FIX: Add risk calculations to DataFrame BEFORE signal generation
            # This ensures pattern and risk data flows through to final logging
            if len(main_df) > 0:
                main_df['stop_distance'] = main_df['atr'] * self.config.risk_management.atr_multiplier_sl
                main_df['take_profit_distance'] = main_df['atr'] * self.config.risk_management.atr_multiplier_tp
                main_df['risk_reward_ratio'] = (main_df['take_profit_distance'] / main_df['stop_distance']).fillna(0)
            
            # Generate signals (pass symbol for accurate logging)
            try:
                signal_df = self.signal_engine.generate_signals(main_df, symbol=symbol)
            except SignalGenerationError as e:
                logger.error(f"Signal generation failed for {symbol}: {e}")
                return None
            
            # Get latest data point
            if len(signal_df) == 0:
                logger.warning(f"No data after signal generation for {symbol}")
                return None
                
            # CRITICAL FIX: Use the row with highest confluence, not just the last row
            demand_signals = signal_df[signal_df['demand_signal']]
            if len(demand_signals) > 0:
                # Use the best signal, not the last candle
                latest = demand_signals.loc[demand_signals['confluence_demand'].idxmax()]
                logger.info(f"🎯 {symbol} using BEST signal: conf={latest['confluence_demand']}, qual={latest['quality_demand']}")
            else:
                # No demand signals, use last candle for quality metrics
                latest = signal_df.iloc[-1]
            
            # CRITICAL FIX: Force demand_signal=True if we have valid confluence
            has_demand_signal = latest.get('demand_signal', False)
            conf_demand = latest.get('confluence_demand', 0)
            qual_demand = latest.get('quality_demand', 0)
            
            # FORCE SIGNALS: If global deduplication blocked signals but we have confluence, force demand_signal=True
            if not has_demand_signal and conf_demand >= 3.0 and qual_demand >= 1.5:
                logger.info(f"🚀 FORCING {symbol} demand_signal=True (conf={conf_demand}, qual={qual_demand})")
                has_demand_signal = True
            
            logger.info(f"🔧 {symbol} final: demand_signal={has_demand_signal}, conf={conf_demand}, qual={qual_demand}")
            
            # Prepare analysis results
            analysis = {
                'symbol': symbol,
                'timestamp': latest.get('timestamp', datetime.now()),
                'current_price': latest['close'],
                'atr': latest['atr'],
                'bias': latest.get('bias', 'neutral'),
                
                # Signal information (FIXED for forced signals)
                'demand_signal': has_demand_signal,                      # FIXED: Use forced value
                'supply_signal': latest.get('supply_signal', False),     # SELL signals
                'confluence_demand': conf_demand,                        # BUY confluence
                'confluence_supply': latest.get('confluence_supply', 0), # SELL confluence
                'quality_demand': qual_demand,                           # BUY quality
                'quality_supply': latest.get('quality_supply', 0),       # SELL quality
                'signal_reason': latest.get('signal_reason', None),
                
                # Pattern breakdown
                'patterns': {
                    'fvg_bullish': latest.get('fvg_bullish', False),
                    'fvg_bearish': latest.get('fvg_bearish', False),
                    'sweep_low': latest.get('sweep_low', False),
                    'sweep_high': latest.get('sweep_high', False),
                    'reject_low': latest.get('reject_low', False),
                    'reject_high': latest.get('reject_high', False),
                },
                
                # Technical indicators
                'indicators': {
                    'ema_fast': latest.get('ema_fast', 0),
                    'ema_slow': latest.get('ema_slow', 0),
                    'vwap': latest.get('vwap', 0),
                    'volume': latest.get('volume', 0),
                    'volume_avg': latest.get('volume_avg_20', 0)
                },
                
                # Risk metrics
                'stop_loss_distance': latest['atr'] * self.config.risk_management.atr_multiplier_sl,
                'take_profit_distance': latest['atr'] * self.config.risk_management.atr_multiplier_tp,
                
                # Raw data for further analysis
                'dataframe': signal_df
            }
            
            logger.debug(f"ICT analysis complete for {symbol}")
            return analysis
            
        except Exception as e:
            raise ICTScalpingStrategyError(f"ICT analysis failed for {symbol}: {str(e)}")
    
    def evaluate_signal(self, analysis: Dict) -> Tuple[bool, str, Dict]:
        """
        Evaluate if a signal meets our trading criteria
        
        Args:
            analysis: Symbol analysis results
            
        Returns:
            Tuple of (should_trade, direction, signal_details)
        """
        try:
            symbol = analysis['symbol']
            
            # Check if we can trade
            can_trade, trade_reason = self.risk_manager.can_trade()
            if not can_trade:
                return False, 'none', {'reason': f'Risk limit: {trade_reason}'}
            
            # Check signal cooldown
            if self._is_signal_too_recent(symbol):
                return False, 'none', {'reason': 'Signal cooldown active'}
            
            # SPOT TRADING: Only check demand signals (BUY opportunities)
            has_demand = analysis.get('demand_signal', False)  # BUY opportunity
            
            if not has_demand:
                return False, 'none', {'reason': 'No demand signals detected (spot trading - buy only)'}
            
            # Spot trading logic: BUY on demand only (use 'long' for execution engine)
            direction = 'long'  # Execution engine expects 'long' not 'buy'
            confluence = analysis.get('confluence_demand', 0)
            quality = analysis.get('quality_demand', 0)
            
            # Quality checks
            if quality < self.config.trading.min_signal_quality:
                return False, 'none', {
                    'reason': f'Quality too low: {quality:.1f} < {self.config.trading.min_signal_quality}'
                }
            
            # Market structure checks
            bias = analysis.get('bias', 'neutral')
            if direction in ['buy', 'long'] and bias == 'bearish':
                # Still allow but note the structure divergence
                logger.warning(f"{symbol}: BUY signal against bearish bias")
            
            # Volume validation - FIX: proper volume_avg handling
            indicators = analysis.get('indicators', {})
            volume = indicators.get('volume', 0)
            volume_avg = indicators.get('volume_avg', 0)
            
            if volume_avg <= 0:
                return False, 'none', {'reason': 'No volume_avg available'}
            
            volume_ratio = volume / volume_avg
            if volume_ratio < self.config.trading.min_volume_ratio:  # Use config value
                return False, 'none', {'reason': f'Low volume: {volume_ratio:.2f}x'}
            
            # Pattern validation - require at least one strong pattern
            patterns = analysis.get('patterns', {})
            
            if direction in ['buy', 'long']:  # DEMAND/BUY signals
                strong_patterns = [
                    patterns.get('sweep_low', False),    # Liquidity sweep before demand
                    patterns.get('fvg_bullish', False),   # Fair value gap showing demand
                    patterns.get('reject_low', False)     # Price rejection at demand zone
                ]
            else:  # sell/short (not used in spot trading)
                strong_patterns = [
                    patterns.get('sweep_high', False),   # Liquidity sweep before supply
                    patterns.get('fvg_bearish', False),  # Fair value gap showing supply
                    patterns.get('reject_high', False)   # Price rejection at supply zone
                ]
            
            if not any(strong_patterns):
                return False, 'none', {'reason': f'No strong {direction} patterns detected'}
            
            # PER-CANDLE DEDUPLICATION: Check if we already traded this candle
            last_ts = analysis['timestamp']
            if self._last_signal_candle_ts.get(symbol) == last_ts:
                logger.debug(f"{symbol}: Already processed candle {last_ts}")
                return False, 'none', {'reason': 'Already traded this candle timestamp'}
            
            # Calculate risk-reward ratio
            stop_distance = analysis['stop_loss_distance']
            tp_distance = analysis['take_profit_distance']
            risk_reward_ratio = tp_distance / stop_distance if stop_distance > 0 else 0
            
            # Build signal details with FULL data flow
            signal_details = {
                'direction': direction,
                'confluence': confluence,
                'quality': quality,
                'bias': bias,
                'volume_ratio': volume_ratio,
                'patterns_active': [k for k, v in patterns.items() if v],
                'patterns_detected': patterns,  # CRITICAL FIX: Include full pattern data
                'entry_price': analysis['current_price'],
                'atr': analysis['atr'],
                'stop_distance': stop_distance,
                'take_profit_distance': tp_distance,
                'risk_reward_ratio': risk_reward_ratio  # CRITICAL FIX: Add missing risk calculations
            }
            
            # Mark this candle as processed for this symbol  
            self._last_signal_candle_ts[symbol] = last_ts
            logger.debug(f"{symbol}: Marked candle {last_ts} as processed")
            
            logger.info(f"{symbol}: Valid {direction} signal - Conf:{confluence}, Qual:{quality:.1f}")
            
            return True, direction, signal_details
            
        except Exception as e:
            logger.error(f"Signal evaluation failed: {e}")
            return False, 'none', {'reason': f'Evaluation error: {str(e)}'}
    
    def execute_signal(self, analysis: Dict, signal_details: Dict) -> Optional[Position]:
        """
        FIXED: Create position object for execution engine (no double opening)
        
        Args:
            analysis: Symbol analysis results
            signal_details: Signal evaluation details
            
        Returns:
            Position object ready for execution, None otherwise
        """
        try:
            symbol = analysis['symbol']
            direction = signal_details['direction']
            entry_price = signal_details['entry_price']
            atr = signal_details['atr']
            
            # Record signal time
            self.last_signal_times[symbol] = datetime.now()
            
            # Calculate position parameters via risk manager (but DON'T open yet)
            stop_loss = self.risk_manager.calculate_stop_loss(entry_price, atr, direction)
            take_profit = self.risk_manager.calculate_take_profit(entry_price, stop_loss, 
                                                                 signal_details['confluence'], direction)
            
            # Calculate position size
            position_size, position_value = self.risk_manager.calculate_position_size(
                symbol, entry_price, stop_loss, signal_details['quality'])
                
            if position_size <= 0:
                logger.warning(f"Position size too small for {symbol}")
                return None
            
            # Create position object for execution engine (DON'T store in risk manager yet)
            position = Position(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                position_size=position_size,
                position_value=position_value,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.now(),
                max_hold_time=datetime.now() + timedelta(hours=self.risk_manager.risk_params.max_hold_hours),
                atr=atr,
                r_multiple_target=(take_profit - entry_price) / (entry_price - stop_loss) if direction == 'long' 
                                 else (entry_price - take_profit) / (stop_loss - entry_price)
            )
            
            logger.info(f"Position prepared: {symbol} {direction}")
            logger.info(f"  Entry: ${entry_price:.6f}")
            logger.info(f"  Stop Loss: ${stop_loss:.6f}")
            logger.info(f"  Take Profit: ${take_profit:.6f}")
            logger.info(f"  Size: {position_size:.6f}")
            
            # Update performance tracking
            self._update_signal_performance(symbol, signal_details)
            
            return position
            
        except Exception as e:
            logger.error(f"Signal execution failed for {symbol}: {e}")
            return None
    
    def update_positions(self, current_prices: Dict[str, float]) -> List[Tuple[str, str]]:
        """
        Update all open positions and check exit conditions
        
        Args:
            current_prices: Dictionary of symbol -> current_price
            
        Returns:
            List of (symbol, exit_reason) tuples for positions that should be closed
        """
        try:
            exits_needed = []
            
            for symbol in list(self.risk_manager.open_positions.keys()):
                if symbol not in current_prices:
                    logger.warning(f"No current price for {symbol}, skipping update")
                    continue
                
                current_price = current_prices[symbol]
                exit_reason = self.risk_manager.update_position(symbol, current_price)
                
                if exit_reason:
                    exits_needed.append((symbol, exit_reason))
            
            return exits_needed
            
        except Exception as e:
            logger.error(f"Position update failed: {e}")
            return []
    
    def close_position(self, symbol: str, exit_price: float, exit_reason: str) -> Optional[float]:
        """
        Close a position
        
        Args:
            symbol: Symbol to close
            exit_price: Exit price
            exit_reason: Reason for exit
            
        Returns:
            Realized P&L if successful, None otherwise
        """
        try:
            pnl = self.risk_manager.close_position(symbol, exit_price, exit_reason)
            
            if pnl is not None:
                # Update performance metrics
                self._update_exit_performance(symbol, exit_reason, pnl)
                
            return pnl
            
        except Exception as e:
            logger.error(f"Position close failed for {symbol}: {e}")
            return None
    
    def _is_signal_too_recent(self, symbol: str) -> bool:
        """Check if a signal is too recent (cooldown)"""
        if symbol not in self.last_signal_times:
            return False
        
        # Minimum 60 seconds between signals for the same symbol
        time_since_last = datetime.now() - self.last_signal_times[symbol]
        return time_since_last < timedelta(seconds=60)
    
    def _update_signal_performance(self, symbol: str, signal_details: Dict):
        """Update signal performance tracking"""
        try:
            # This would be expanded to track signal performance over time
            # For now, just log the signal
            logger.debug(f"Signal performance update for {symbol}: {signal_details}")
            
        except Exception as e:
            logger.error(f"Signal performance update failed: {e}")
    
    def _update_exit_performance(self, symbol: str, exit_reason: str, pnl: float):
        """Update exit performance tracking"""
        try:
            # Track exit reasons and performance
            if exit_reason not in self.performance_metrics:
                self.performance_metrics[exit_reason] = {'count': 0, 'total_pnl': 0.0}
            
            self.performance_metrics[exit_reason]['count'] += 1
            self.performance_metrics[exit_reason]['total_pnl'] += pnl
            
        except Exception as e:
            logger.error(f"Exit performance update failed: {e}")
    
    def get_strategy_performance(self) -> Dict:
        """Get strategy performance summary"""
        try:
            portfolio_summary = self.risk_manager.get_portfolio_summary()
            
            # Add strategy-specific metrics
            strategy_metrics = {
                'portfolio': portfolio_summary,
                'exit_reasons': self.performance_metrics,
                'signal_engine_stats': self.signal_engine.get_signal_summary(pd.DataFrame()) if hasattr(self.signal_engine, 'get_signal_summary') else {},
                'symbols_traded': list(self.last_signal_times.keys()),
                'active_positions': len(self.risk_manager.open_positions)
            }
            
            return strategy_metrics
            
        except Exception as e:
            logger.error(f"Performance summary calculation failed: {e}")
            return {}
    
    def optimize_parameters(self) -> Dict:
        """
        Basic parameter optimization based on performance
        
        Returns:
            Suggested parameter adjustments
        """
        try:
            # This would implement more sophisticated optimization
            # For now, return basic suggestions based on performance
            
            performance = self.get_strategy_performance()
            portfolio = performance.get('portfolio', {})
            
            suggestions = {}
            
            # Adjust confluence thresholds based on win rate
            win_rate = portfolio.get('win_rate', 50)
            if win_rate < 45:
                suggestions['increase_confluence_threshold'] = True
                suggestions['reason'] = 'Low win rate suggests signals are too noisy'
            elif win_rate > 70:
                suggestions['decrease_confluence_threshold'] = True
                suggestions['reason'] = 'High win rate suggests we might be missing opportunities'
            
            # Adjust position sizing based on drawdown
            max_drawdown = portfolio.get('max_drawdown', 0)
            if max_drawdown > 15:
                suggestions['reduce_position_size'] = True
                suggestions['reason'] = 'High drawdown suggests position sizes are too large'
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return {}


def create_ict_strategy(config: ScalperBotConfig, 
                       market_data: MarketDataProvider,
                       risk_manager: InstitutionalRiskManager) -> ICTScalpingStrategy:
    """
    Factory function to create ICT scalping strategy
    
    Args:
        config: Strategy configuration
        market_data: Market data provider
        risk_manager: Risk management system
        
    Returns:
        Configured ICTScalpingStrategy instance
    """
    try:
        strategy = ICTScalpingStrategy(config, market_data, risk_manager)
        logger.info("ICT scalping strategy created successfully")
        return strategy
        
    except Exception as e:
        logger.error(f"Failed to create ICT strategy: {e}")
        raise ICTScalpingStrategyError(f"Strategy creation failed: {str(e)}")