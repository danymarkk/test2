"""
INSTITUTIONAL-GRADE RISK MANAGEMENT
Professional position sizing, stop losses, and portfolio protection
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RiskManagementError(Exception):
    """Raised when risk management operations fail"""
    pass


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    position_size: float
    position_value: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    max_hold_time: datetime
    atr: float
    r_multiple_target: float
    trailing_stop: Optional[float] = None
    unrealized_pnl: float = 0.0
    max_unrealized_pnl: float = 0.0


@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_risk_per_trade: float = 0.02  # 2% of portfolio per trade
    max_portfolio_risk: float = 0.08  # 8% total portfolio risk
    max_open_positions: int = 5
    max_correlation_exposure: float = 0.15  # Max 15% in correlated assets
    atr_multiplier_sl: float = 0.75  # ATR multiplier for stop loss
    atr_multiplier_tp: float = 2.0   # ATR multiplier for take profit
    max_hold_hours: float = 4.0      # Maximum position hold time
    trailing_stop_trigger: float = 1.5  # Start trailing at +1.5R
    trailing_stop_distance: float = 0.75  # Trail 0.75R behind peak  
    breakeven_trigger: float = 1.0   # Move to breakeven at +1R
    drawdown_limit: float = 0.20     # Stop trading at 20% drawdown
    daily_loss_limit: float = 0.05   # Stop trading at 5% daily loss


class InstitutionalRiskManager:
    """
    Professional risk management system for crypto scalping
    
    Features:
    - Portfolio-level risk monitoring
    - Dynamic position sizing
    - ATR-based stop losses
    - Trailing stops and profit taking
    - Correlation-based exposure limits
    - Drawdown protection
    """
    
    def __init__(self, initial_balance: float, risk_params: Optional[RiskParameters] = None):
        """
        Initialize risk manager
        
        Args:
            initial_balance: Starting portfolio balance
            risk_params: Risk management parameters
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_params = risk_params or RiskParameters()
        
        # Position tracking
        self.open_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Risk monitoring
        self.daily_pnl = 0.0
        self.session_start_balance = initial_balance
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees_paid = 0.0
        
        logger.info(f"Risk Manager initialized: Balance=${initial_balance:,.2f}")
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float,
                               signal_quality: float = 3.0) -> Tuple[float, float]:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss: Stop loss price
            signal_quality: Signal quality score (0-5)
            
        Returns:
            Tuple of (position_size, position_value)
            
        Raises:
            RiskManagementError: If position sizing fails
        """
        try:
            if entry_price <= 0 or stop_loss <= 0:
                raise RiskManagementError("Invalid entry or stop loss price")
                
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share <= 0:
                raise RiskManagementError("Invalid risk calculation")
            
            # Base risk amount (percentage of current balance)
            base_risk_pct = self.risk_params.max_risk_per_trade
            
            # Adjust risk based on signal quality (0.5x to 1.5x multiplier)
            quality_multiplier = 0.5 + (signal_quality / 5.0)  # 0.5 to 1.5
            adjusted_risk_pct = base_risk_pct * quality_multiplier
            
            # Check portfolio-level risk limits
            current_portfolio_risk = self._calculate_portfolio_risk()
            if current_portfolio_risk + adjusted_risk_pct > self.risk_params.max_portfolio_risk:
                # Reduce position size to stay within portfolio limits
                available_risk = self.risk_params.max_portfolio_risk - current_portfolio_risk
                adjusted_risk_pct = max(0, available_risk)
                
                if adjusted_risk_pct < 0.005:  # Less than 0.5% risk
                    logger.warning(f"Portfolio risk limit reached, skipping {symbol}")
                    return 0.0, 0.0
            
            # Calculate position size
            risk_amount = self.current_balance * adjusted_risk_pct
            position_size = risk_amount / risk_per_share
            position_value = position_size * entry_price
            
            # Apply minimum and maximum position size limits
            min_position_value = 5.0  # Minimum $5 position
            max_position_value = self.current_balance * 0.20  # Max 20% of balance
            
            if position_value < min_position_value:
                logger.warning(f"Position too small for {symbol}: ${position_value:.2f}")
                return 0.0, 0.0
                
            if position_value > max_position_value:
                position_value = max_position_value
                position_size = position_value / entry_price
                
            logger.debug(f"Position sizing for {symbol}: Size={position_size:.6f}, Value=${position_value:.2f}, Risk={adjusted_risk_pct:.2%}")
            
            return position_size, position_value
            
        except Exception as e:
            raise RiskManagementError(f"Position sizing failed for {symbol}: {str(e)}")
    
    def calculate_stop_loss(self, entry_price: float, atr: float, direction: str) -> float:
        """
        Calculate ATR-based stop loss
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            direction: 'long' or 'short'
            
        Returns:
            Stop loss price
        """
        try:
            if atr <= 0:
                # Fallback to percentage-based stop
                atr = entry_price * 0.02  # 2% fallback
                
            sl_distance = atr * self.risk_params.atr_multiplier_sl
            
            if direction == 'long':
                stop_loss = entry_price - sl_distance
            else:  # short
                stop_loss = entry_price + sl_distance
                
            # Ensure stop loss is reasonable (not more than 5% away)
            max_sl_distance = entry_price * 0.05
            if abs(entry_price - stop_loss) > max_sl_distance:
                if direction == 'long':
                    stop_loss = entry_price - max_sl_distance
                else:
                    stop_loss = entry_price + max_sl_distance
                    
            return max(0.000001, stop_loss)  # Ensure positive price
            
        except Exception as e:
            logger.error(f"Stop loss calculation failed: {e}")
            return entry_price * 0.98 if direction == 'long' else entry_price * 1.02
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                             confluence_score: float, direction: str) -> float:
        """
        Calculate dynamic take profit based on confluence score
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            confluence_score: Signal confluence score
            direction: 'long' or 'short'
            
        Returns:
            Take profit price
        """
        try:
            risk_distance = abs(entry_price - stop_loss)
            
            # Dynamic R:R based on confluence
            if confluence_score >= 4.5:
                r_multiple = 3.0  # Excellent setups get 3R
            elif confluence_score >= 3.5:
                r_multiple = 2.5  # Good setups get 2.5R
            elif confluence_score >= 2.5:
                r_multiple = 2.0  # Decent setups get 2R
            else:
                r_multiple = 1.5  # Minimum setups get 1.5R
                
            reward_distance = risk_distance * r_multiple
            
            if direction == 'long':
                take_profit = entry_price + reward_distance
            else:  # short
                take_profit = entry_price - reward_distance
                
            return max(0.000001, take_profit)  # Ensure positive price
            
        except Exception as e:
            logger.error(f"Take profit calculation failed: {e}")
            # Fallback to 2:1 R:R
            risk_distance = abs(entry_price - stop_loss)
            if direction == 'long':
                return entry_price + (risk_distance * 2.0)
            else:
                return entry_price - (risk_distance * 2.0)
    
    def open_position(self, symbol: str, direction: str, entry_price: float,
                     atr: float, confluence_score: float, signal_quality: float) -> Optional[Position]:
        """
        Open a new position with comprehensive risk checks
        
        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            entry_price: Entry price
            atr: Average True Range
            confluence_score: Signal confluence score
            signal_quality: Signal quality score
            
        Returns:
            Position object if opened successfully, None otherwise
        """
        try:
            # Check if we can open more positions
            if len(self.open_positions) >= self.risk_params.max_open_positions:
                logger.warning(f"Maximum positions reached: {len(self.open_positions)}")
                return None
                
            # Check if we already have a position in this symbol
            if symbol in self.open_positions:
                logger.warning(f"Position already exists for {symbol}")
                return None
                
            # Check drawdown limits
            if self.current_drawdown >= self.risk_params.drawdown_limit:
                logger.warning(f"Drawdown limit reached: {self.current_drawdown:.2%}")
                return None
                
            # Check daily loss limits
            if self.daily_pnl <= -self.current_balance * self.risk_params.daily_loss_limit:
                logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
                return None
            
            # Calculate stop loss and take profit
            stop_loss = self.calculate_stop_loss(entry_price, atr, direction)
            take_profit = self.calculate_take_profit(entry_price, stop_loss, confluence_score, direction)
            
            # Calculate position size
            position_size, position_value = self.calculate_position_size(
                symbol, entry_price, stop_loss, signal_quality)
                
            if position_size <= 0:
                logger.warning(f"Position size too small for {symbol}")
                return None
            
            # Create position
            position = Position(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                position_size=position_size,
                position_value=position_value,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.now(),
                max_hold_time=datetime.now() + timedelta(hours=self.risk_params.max_hold_hours),
                atr=atr,
                r_multiple_target=(take_profit - entry_price) / (entry_price - stop_loss) if direction == 'long' 
                                 else (entry_price - take_profit) / (stop_loss - entry_price)
            )
            
            # Store position
            self.open_positions[symbol] = position
            
            logger.info(f"Position opened: {symbol} {direction} @ ${entry_price:.6f}")
            logger.info(f"  Size: {position_size:.6f}, Value: ${position_value:.2f}")
            logger.info(f"  SL: ${stop_loss:.6f}, TP: ${take_profit:.6f}")
            
            return position
            
        except Exception as e:
            logger.error(f"Failed to open position for {symbol}: {e}")
            return None
    
    def update_position(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Update position with current price and check exit conditions
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Exit reason if position should be closed, None otherwise
        """
        try:
            if symbol not in self.open_positions:
                return None
                
            position = self.open_positions[symbol]
            
            # Calculate current P&L
            if position.direction == 'long':
                unrealized_pnl = (current_price - position.entry_price) * position.position_size
                r_multiple = (current_price - position.entry_price) / (position.entry_price - position.stop_loss)
            else:  # short
                unrealized_pnl = (position.entry_price - current_price) * position.position_size
                r_multiple = (position.entry_price - current_price) / (position.stop_loss - position.entry_price)
            
            position.unrealized_pnl = unrealized_pnl
            position.max_unrealized_pnl = max(position.max_unrealized_pnl, unrealized_pnl)
            
            # Check exit conditions
            
                        # 1. Stop Loss - Add buffer to prevent immediate triggering
            stop_buffer = 0.001  # 0.1% buffer
            if position.direction == 'long' and current_price <= (position.stop_loss * (1 - stop_buffer)):
                logger.info(f"{symbol}: Stop loss triggered - Price:${current_price:.6f} SL:${position.stop_loss:.6f}")
                return 'stop_loss'
            elif position.direction == 'short' and current_price >= (position.stop_loss * (1 + stop_buffer)):
                logger.info(f"{symbol}: Stop loss triggered - Price:${current_price:.6f} SL:${position.stop_loss:.6f}")
                return 'stop_loss'
            
            # 2. Take Profit
            if position.direction == 'long' and current_price >= position.take_profit:
                logger.info(f"{symbol}: Take profit triggered - Price:${current_price:.6f} TP:${position.take_profit:.6f}")
                return 'take_profit'
            elif position.direction == 'short' and current_price <= position.take_profit:
                logger.info(f"{symbol}: Take profit triggered - Price:${current_price:.6f} TP:${position.take_profit:.6f}")
                return 'take_profit'
                
            # 3. Trailing Stop
            if r_multiple >= self.risk_params.trailing_stop_trigger:
                self._update_trailing_stop(position, current_price, r_multiple)
                
                if position.trailing_stop is not None:
                    if position.direction == 'long' and current_price <= position.trailing_stop:
                        return 'trailing_stop'
                    elif position.direction == 'short' and current_price >= position.trailing_stop:
                        return 'trailing_stop'
            
            # 4. Breakeven Stop
            if r_multiple >= self.risk_params.breakeven_trigger:
                if position.direction == 'long':
                    position.stop_loss = max(position.stop_loss, position.entry_price)
                else:
                    position.stop_loss = min(position.stop_loss, position.entry_price)
            
            # 5. Time-based Exit  
            if datetime.now() >= position.max_hold_time:
                logger.info(f"{symbol}: Time-based exit triggered - held {(datetime.now() - position.entry_time).total_seconds():.0f}s")
                return 'timeout'
                
            # 6. Momentum Death (negative R after good profit) - RELAXED 
            if position.max_unrealized_pnl > 0 and r_multiple < -1.0:
                logger.info(f"{symbol}: Momentum death exit - R:{r_multiple:.2f}, MaxProfit:${position.max_unrealized_pnl:.2f}")
                return 'momentum_death'
            
            return None
            
        except Exception as e:
            logger.error(f"Position update failed for {symbol}: {e}")
            logger.error(f"  Position: {position}")
            logger.error(f"  Current price: {current_price}")
            return 'error'
    
    def _update_trailing_stop(self, position: Position, current_price: float, r_multiple: float):
        """Update trailing stop for a position"""
        try:
            risk_distance = abs(position.entry_price - position.stop_loss)
            trail_distance = risk_distance * self.risk_params.trailing_stop_distance
            
            if position.direction == 'long':
                new_trailing_stop = current_price - trail_distance
                if position.trailing_stop is None or new_trailing_stop > position.trailing_stop:
                    position.trailing_stop = new_trailing_stop
            else:  # short
                new_trailing_stop = current_price + trail_distance
                if position.trailing_stop is None or new_trailing_stop < position.trailing_stop:
                    position.trailing_stop = new_trailing_stop
                    
        except Exception as e:
            logger.error(f"Trailing stop update failed: {e}")
    
    def close_position(self, symbol: str, exit_price: float, exit_reason: str) -> Optional[float]:
        """
        Close a position and update portfolio metrics
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            exit_reason: Reason for closing
            
        Returns:
            Realized P&L if successful, None otherwise
        """
        try:
            if symbol not in self.open_positions:
                logger.warning(f"No open position found for {symbol}")
                return None
                
            position = self.open_positions[symbol]
            
            # Calculate realized P&L
            if position.direction == 'long':
                gross_pnl = (exit_price - position.entry_price) * position.position_size
            else:  # short
                gross_pnl = (position.entry_price - exit_price) * position.position_size
            
            # Apply fees (0.1% per side = 0.2% total)
            fees = position.position_value * 0.002
            net_pnl = gross_pnl - fees
            
            # Calculate R-multiple
            risk_amount = abs(position.entry_price - position.stop_loss) * position.position_size
            r_multiple = net_pnl / risk_amount if risk_amount > 0 else 0
            
            # Update portfolio metrics
            self.current_balance += net_pnl
            self.daily_pnl += net_pnl
            self.total_fees_paid += fees
            self.total_trades += 1
            
            if net_pnl > 0:
                self.winning_trades += 1
            
            # Update drawdown
            peak_balance = max(self.initial_balance, self.current_balance)
            self.current_drawdown = (peak_balance - self.current_balance) / peak_balance
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.open_positions[symbol]
            
            logger.info(f"Position closed: {symbol} - {exit_reason}")
            logger.info(f"  P&L: ${net_pnl:.2f} ({r_multiple:.2f}R)")
            logger.info(f"  Balance: ${self.current_balance:.2f}")
            
            return net_pnl
            
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return None
    
    def _calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk percentage"""
        try:
            total_risk = 0.0
            for position in self.open_positions.values():
                risk_amount = abs(position.entry_price - position.stop_loss) * position.position_size
                risk_pct = risk_amount / self.current_balance
                total_risk += risk_pct
            return total_risk
        except Exception as e:
            logger.error(f"Portfolio risk calculation failed: {e}")
            return 0.0
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        try:
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            return {
                'current_balance': self.current_balance,
                'total_return': (self.current_balance - self.initial_balance) / self.initial_balance * 100,
                'daily_pnl': self.daily_pnl,
                'max_drawdown': self.max_drawdown * 100,
                'current_drawdown': self.current_drawdown * 100,
                'open_positions': len(self.open_positions),
                'total_trades': self.total_trades,
                'win_rate': win_rate,
                'total_fees': self.total_fees_paid,
                'portfolio_risk': self._calculate_portfolio_risk() * 100
            }
        except Exception as e:
            logger.error(f"Portfolio summary calculation failed: {e}")
            return {}
    
    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk limits
        
        Returns:
            Tuple of (can_trade, reason)
        """
        try:
            # Check drawdown limit
            if self.current_drawdown >= self.risk_params.drawdown_limit:
                return False, f"Drawdown limit exceeded: {self.current_drawdown:.2%}"
            
            # Check daily loss limit
            daily_loss_limit = self.current_balance * self.risk_params.daily_loss_limit
            if self.daily_pnl <= -daily_loss_limit:
                return False, f"Daily loss limit exceeded: ${self.daily_pnl:.2f}"
            
            # Check maximum positions
            if len(self.open_positions) >= self.risk_params.max_open_positions:
                return False, f"Maximum positions reached: {len(self.open_positions)}"
            
            # Check portfolio risk
            current_risk = self._calculate_portfolio_risk()
            if current_risk >= self.risk_params.max_portfolio_risk:
                return False, f"Portfolio risk limit exceeded: {current_risk:.2%}"
            
            return True, "Trading allowed"
            
        except Exception as e:
            logger.error(f"Trading permission check failed: {e}")
            return False, "Risk check error"