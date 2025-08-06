"""
INSTITUTIONAL-GRADE PERFORMANCE TRACKING SYSTEM
Comprehensive trade analytics with JSON logging for scalping bot analysis
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from enum import Enum

from ..core.risk_manager import Position


class TradeResult(Enum):
    """Trade outcome enumeration"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


class ExitReason(Enum):
    """Trade exit reason enumeration"""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIMEOUT = "timeout"
    MANUAL = "manual"
    DRAWDOWN_LIMIT = "drawdown_limit"
    RISK_MANAGEMENT = "risk_management"
    SIGNAL_REVERSAL = "signal_reversal"


@dataclass
class TradePerformanceData:
    """Comprehensive trade performance data structure"""
    
    # Basic Trade Info
    trade_id: str
    timestamp_entry: str
    timestamp_exit: str
    symbol: str
    direction: str
    
    # Pricing Data
    entry_price: float
    exit_price: float
    sl_price: Optional[float]
    tp_price: Optional[float]
    
    # Performance Metrics
    pnl_usdt: float
    pnl_percentage: float
    r_multiple: Optional[float]  # Actual R vs risk
    position_size_usdt: float
    fees_total: float
    slippage_impact: float
    
    # Trade Duration
    hold_time_seconds: int
    hold_time_bars: int
    
    # Signal Quality
    signal_quality: float
    confluence_score: int
    htf_bias: str
    volume_ratio: float
    
    # Market Conditions
    atr_entry: float
    volatility_percentile: float
    market_trend: str
    spread_bps: Optional[float]
    
    # Exit Analysis
    exit_reason: str
    result: str  # win/loss/breakeven
    max_favorable_excursion: float  # MFE - best unrealized profit
    max_adverse_excursion: float    # MAE - worst unrealized loss
    
    # Risk Metrics
    risk_per_trade: float
    account_balance_pre: float
    account_balance_post: float
    drawdown_impact: float
    
    # Execution Quality
    execution_latency_ms: Optional[int]
    order_fill_quality: float  # how well we got filled vs expected
    
    # Additional Context
    session_pnl_before: float
    daily_pnl_before: float
    consecutive_wins: int
    consecutive_losses: int
    trade_number_daily: int
    trade_number_session: int


class PerformanceTracker:
    """
    Institutional-grade performance tracking system
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize performance tracker
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Performance log files
        self.trades_log = self.log_dir / "trade_performance.jsonl"
        self.signals_log = self.log_dir / "signal_analysis.jsonl" 
        self.daily_summary_log = self.log_dir / "daily_summary.jsonl"
        
        # Session tracking
        self.session_start = datetime.now(timezone.utc)
        self.active_trades: Dict[str, Dict] = {}
        self.completed_trades: List[TradePerformanceData] = []
        
        # Performance metrics
        self.daily_pnl = 0.0
        self.session_pnl = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.trade_count_daily = 0
        self.trade_count_session = 0
        self.max_drawdown_session = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        
        # Setup JSON loggers for each file
        self._setup_json_loggers()
        
        self.logger.info("Performance tracking system initialized")
        self.logger.info(f"Trade logs: {self.trades_log}")
        self.logger.info(f"Signal logs: {self.signals_log}")
    
    def _setup_json_loggers(self):
        """Setup dedicated JSON loggers for each type of data"""
        
        # Trade performance logger
        self.trade_logger = logging.getLogger('trade_performance')
        trade_handler = logging.FileHandler(self.trades_log, mode='a')
        trade_handler.setFormatter(logging.Formatter('%(message)s'))
        self.trade_logger.addHandler(trade_handler)
        self.trade_logger.setLevel(logging.INFO)
        self.trade_logger.propagate = False
        
        # Signal analysis logger  
        self.signal_logger = logging.getLogger('signal_analysis')
        signal_handler = logging.FileHandler(self.signals_log, mode='a')
        signal_handler.setFormatter(logging.Formatter('%(message)s'))
        self.signal_logger.addHandler(signal_handler)
        self.signal_logger.setLevel(logging.INFO)
        self.signal_logger.propagate = False
        
        # Daily summary logger
        self.summary_logger = logging.getLogger('daily_summary')
        summary_handler = logging.FileHandler(self.daily_summary_log, mode='a')
        summary_handler.setFormatter(logging.Formatter('%(message)s'))
        self.summary_logger.addHandler(summary_handler)
        self.summary_logger.setLevel(logging.INFO)
        self.summary_logger.propagate = False
    
    def log_signal(self, symbol: str, direction: str, signal_data: Dict[str, Any]):
        """
        Log signal generation with comprehensive analysis
        
        Args:
            symbol: Trading symbol
            direction: Signal direction (long/short)
            signal_data: Signal analysis data
        """
        try:
            signal_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'event_type': 'signal',
                'symbol': symbol,
                'direction': direction,
                'signal_quality': signal_data.get('quality', 0),
                'confluence_score': signal_data.get('confluence', 0),
                'htf_bias': signal_data.get('htf_bias', 'neutral'),
                'volume_ratio': signal_data.get('volume_ratio', 1.0),
                'atr_current': signal_data.get('atr', 0),
                'price_current': signal_data.get('current_price', 0),
                'patterns_detected': signal_data.get('patterns', {}),
                'risk_reward_ratio': signal_data.get('risk_reward_ratio', signal_data.get('risk_reward', 0)),
                'filters_passed': signal_data.get('filters_passed', []),
                'filters_failed': signal_data.get('filters_failed', []),
                'market_conditions': signal_data.get('market_conditions', {}),
                'signal_strength': signal_data.get('strength', 'weak'),
                'entry_conditions_met': signal_data.get('entry_ready', False)
            }
            
            # Log as JSON with proper boolean handling
            self.signal_logger.info(json.dumps(signal_entry, default=self._json_serializer))
            
        except Exception as e:
            self.logger.error(f"Signal logging failed: {e}")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer that handles numpy types properly"""
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif hasattr(obj, '__str__'):
            return str(obj)
        else:
            return obj
    
    def start_trade_tracking(self, position: Position, signal_data: Dict[str, Any],
                           market_conditions: Dict[str, Any]):
        """
        Start tracking a new trade
        
        Args:
            position: Position object
            signal_data: Original signal data
            market_conditions: Current market conditions
        """
        try:
            with self._lock:
                trade_id = f"{position.symbol}_{position.direction}_{int(datetime.now().timestamp())}"
                
                trade_entry = {
                    'trade_id': trade_id,
                    'position': position,
                    'timestamp_entry': datetime.now(timezone.utc).isoformat(),
                    'signal_data': signal_data,
                    'market_conditions': market_conditions,
                    'session_pnl_before': self.session_pnl,
                    'daily_pnl_before': self.daily_pnl,
                    'consecutive_wins': self.consecutive_wins,
                    'consecutive_losses': self.consecutive_losses,
                    'trade_number_daily': self.trade_count_daily + 1,
                    'trade_number_session': self.trade_count_session + 1,
                    'mfe': 0.0,  # Max Favorable Excursion
                    'mae': 0.0   # Max Adverse Excursion
                }
                
                self.active_trades[trade_id] = trade_entry
                
                self.logger.info(f"Started tracking trade: {trade_id}")
                
                return trade_id
                
        except Exception as e:
            self.logger.error(f"Trade tracking start failed: {e}")
            return None
    
    def update_trade_metrics(self, trade_id: str, current_price: float,
                           unrealized_pnl: float):
        """
        Update running trade metrics (MFE/MAE)
        
        Args:
            trade_id: Trade identifier
            current_price: Current market price
            unrealized_pnl: Current unrealized P&L
        """
        try:
            with self._lock:
                if trade_id not in self.active_trades:
                    return
                
                trade = self.active_trades[trade_id]
                
                # Update MFE (best profit)
                if unrealized_pnl > trade['mfe']:
                    trade['mfe'] = unrealized_pnl
                
                # Update MAE (worst loss)  
                if unrealized_pnl < trade['mae']:
                    trade['mae'] = unrealized_pnl
                
                trade['current_price'] = current_price
                trade['unrealized_pnl'] = unrealized_pnl
                
        except Exception as e:
            self.logger.error(f"Trade metrics update failed: {e}")
    
    def complete_trade(self, trade_id: str, exit_price: float, exit_reason: ExitReason,
                      pnl_usdt: float, fees: float, slippage: float = 0.0,
                      account_balance: float = 0.0) -> Optional[TradePerformanceData]:
        """
        Complete trade tracking and log comprehensive performance data
        
        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_reason: Reason for exit
            pnl_usdt: Realized P&L in USDT
            fees: Total fees paid
            slippage: Slippage impact
            account_balance: Current account balance
            
        Returns:
            TradePerformanceData if successful, None otherwise
        """
        try:
            with self._lock:
                if trade_id not in self.active_trades:
                    self.logger.warning(f"Trade not found for completion: {trade_id}")
                    return None
                
                trade_data = self.active_trades.pop(trade_id)
                position = trade_data['position']
                signal_data = trade_data['signal_data']
                market_conditions = trade_data['market_conditions']
                
                # Calculate comprehensive metrics
                timestamp_exit = datetime.now(timezone.utc)
                timestamp_entry = datetime.fromisoformat(trade_data['timestamp_entry'])
                hold_duration = timestamp_exit - timestamp_entry
                
                # Performance calculations
                entry_price = position.entry_price
                pnl_percentage = (pnl_usdt / position.position_value) * 100 if position.position_value > 0 else 0
                
                # R-multiple calculation (if we have SL/TP)
                r_multiple = None
                if position.stop_loss and entry_price:
                    risk_per_share = abs(entry_price - position.stop_loss)
                    if risk_per_share > 0:
                        reward_per_share = abs(exit_price - entry_price)
                        r_multiple = reward_per_share / risk_per_share
                        if pnl_usdt < 0:
                            r_multiple = -r_multiple
                
                # Trade result
                if pnl_usdt > 0.01:
                    result = TradeResult.WIN
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                elif pnl_usdt < -0.01:
                    result = TradeResult.LOSS
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                else:
                    result = TradeResult.BREAKEVEN
                
                # Update session metrics
                self.session_pnl += pnl_usdt
                self.daily_pnl += pnl_usdt
                self.trade_count_session += 1
                self.trade_count_daily += 1
                
                # Create comprehensive performance record
                perf_data = TradePerformanceData(
                    # Basic Trade Info
                    trade_id=trade_id,
                    timestamp_entry=trade_data['timestamp_entry'],
                    timestamp_exit=timestamp_exit.isoformat(),
                    symbol=position.symbol,
                    direction=position.direction,
                    
                    # Pricing Data
                    entry_price=entry_price,
                    exit_price=exit_price,
                    sl_price=position.stop_loss,
                    tp_price=position.take_profit,
                    
                    # Performance Metrics
                    pnl_usdt=pnl_usdt,
                    pnl_percentage=pnl_percentage,
                    r_multiple=r_multiple,
                    position_size_usdt=position.position_value,
                    fees_total=fees,
                    slippage_impact=slippage,
                    
                    # Trade Duration
                    hold_time_seconds=int(hold_duration.total_seconds()),
                    hold_time_bars=max(1, int(hold_duration.total_seconds() / 60)),  # Assuming 1m bars
                    
                    # Signal Quality
                    signal_quality=signal_data.get('quality', 0),
                    confluence_score=signal_data.get('confluence', 0),
                    htf_bias=signal_data.get('htf_bias', 'neutral'),
                    volume_ratio=signal_data.get('volume_ratio', 1.0),
                    
                    # Market Conditions
                    atr_entry=market_conditions.get('atr', 0),
                    volatility_percentile=market_conditions.get('volatility_percentile', 50),
                    market_trend=market_conditions.get('trend', 'sideways'),
                    spread_bps=market_conditions.get('spread_bps'),
                    
                    # Exit Analysis
                    exit_reason=exit_reason.value,
                    result=result.value,
                    max_favorable_excursion=trade_data.get('mfe', 0),
                    max_adverse_excursion=trade_data.get('mae', 0),
                    
                    # Risk Metrics
                    risk_per_trade=abs(pnl_usdt) if result == TradeResult.LOSS else position.position_value * 0.01,
                    account_balance_pre=account_balance - pnl_usdt,
                    account_balance_post=account_balance,
                    drawdown_impact=min(0, self.session_pnl),
                    
                    # Execution Quality  
                    execution_latency_ms=None,  # TODO: Add execution timing
                    order_fill_quality=1.0 - abs(slippage),  # Simple quality metric
                    
                    # Additional Context
                    session_pnl_before=trade_data['session_pnl_before'],
                    daily_pnl_before=trade_data['daily_pnl_before'],
                    consecutive_wins=self.consecutive_wins,
                    consecutive_losses=self.consecutive_losses,
                    trade_number_daily=trade_data['trade_number_daily'],
                    trade_number_session=trade_data['trade_number_session']
                )
                
                # Log as JSON for Data Preview Extension
                trade_json = asdict(perf_data)
                self.trade_logger.info(json.dumps(trade_json, default=self._json_serializer))
                
                # Store in memory
                self.completed_trades.append(perf_data)
                
                # Log summary to main logger
                r_display = f"{r_multiple:.2f}" if r_multiple else "N/A"
                self.logger.info(
                    f"TRADE_COMPLETED: {position.symbol} {position.direction.upper()} "
                    f"Entry:${entry_price:.6f} Exit:${exit_price:.6f} "
                    f"P&L:${pnl_usdt:.2f} ({pnl_percentage:.1f}%) "
                    f"R:{r_display} "
                    f"Hold:{hold_duration.total_seconds():.0f}s "
                    f"Reason:{exit_reason.value} Result:{result.value.upper()}"
                )
                
                return perf_data
                
        except Exception as e:
            self.logger.error(f"Trade completion failed: {e}")
            return None
    
    def log_daily_summary(self):
        """Log comprehensive daily performance summary"""
        try:
            if not self.completed_trades:
                return
            
            # Calculate daily statistics
            daily_trades = [t for t in self.completed_trades 
                          if t.timestamp_entry.startswith(datetime.now().strftime('%Y-%m-%d'))]
            
            if not daily_trades:
                return
            
            total_pnl = sum(t.pnl_usdt for t in daily_trades)
            total_fees = sum(t.fees_total for t in daily_trades)
            
            wins = [t for t in daily_trades if t.result == 'win']
            losses = [t for t in daily_trades if t.result == 'loss']
            
            win_rate = len(wins) / len(daily_trades) * 100 if daily_trades else 0
            avg_win = sum(t.pnl_usdt for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t.pnl_usdt for t in losses) / len(losses) if losses else 0
            
            profit_factor = abs(sum(t.pnl_usdt for t in wins) / sum(t.pnl_usdt for t in losses)) if losses and sum(t.pnl_usdt for t in losses) != 0 else float('inf')
            
            avg_hold_time = sum(t.hold_time_seconds for t in daily_trades) / len(daily_trades) if daily_trades else 0
            avg_r_multiple = sum(t.r_multiple for t in daily_trades if t.r_multiple) / len([t for t in daily_trades if t.r_multiple]) if any(t.r_multiple for t in daily_trades) else 0
            
            daily_summary = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'event_type': 'daily_summary',
                
                # Trade Count
                'total_trades': len(daily_trades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'breakeven_trades': len(daily_trades) - len(wins) - len(losses),
                
                # Performance Metrics
                'total_pnl_usdt': total_pnl,
                'total_fees_usdt': total_fees,
                'net_pnl_usdt': total_pnl - total_fees,
                'win_rate_percent': win_rate,
                'profit_factor': profit_factor,
                'avg_win_usdt': avg_win,
                'avg_loss_usdt': avg_loss,
                'avg_r_multiple': avg_r_multiple,
                
                # Risk Metrics
                'max_drawdown_usdt': min(0, min(t.drawdown_impact for t in daily_trades)),
                'largest_win_usdt': max((t.pnl_usdt for t in wins), default=0),
                'largest_loss_usdt': min((t.pnl_usdt for t in losses), default=0),
                'max_consecutive_wins': max((t.consecutive_wins for t in daily_trades), default=0),
                'max_consecutive_losses': max((t.consecutive_losses for t in daily_trades), default=0),
                
                # Execution Quality
                'avg_hold_time_seconds': avg_hold_time,
                'avg_hold_time_minutes': avg_hold_time / 60,
                'avg_signal_quality': sum(t.signal_quality for t in daily_trades) / len(daily_trades),
                'avg_confluence_score': sum(t.confluence_score for t in daily_trades) / len(daily_trades),
                
                # Symbol Performance
                'symbols_traded': list(set(t.symbol for t in daily_trades)),
                'best_symbol': max(((symbol, sum(t.pnl_usdt for t in daily_trades if t.symbol == symbol)) 
                                 for symbol in set(t.symbol for t in daily_trades)), 
                                 key=lambda x: x[1], default=('None', 0))[0],
                
                # Exit Reasons
                'exit_reasons': {reason: len([t for t in daily_trades if t.exit_reason == reason]) 
                               for reason in set(t.exit_reason for t in daily_trades)}
            }
            
            # Log daily summary as JSON
            self.summary_logger.info(json.dumps(daily_summary, default=self._json_serializer))
            
            # Log summary to main logger
            self.logger.info(
                f"DAILY_SUMMARY: {len(daily_trades)} trades, "
                f"P&L: ${total_pnl:.2f}, Win Rate: {win_rate:.1f}%, "
                f"PF: {profit_factor:.2f}, Avg R: {avg_r_multiple:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Daily summary logging failed: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        try:
            return {
                'session_pnl': self.session_pnl,
                'daily_pnl': self.daily_pnl,
                'trades_session': self.trade_count_session,
                'trades_daily': self.trade_count_daily,
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses,
                'active_trades': len(self.active_trades),
                'session_duration_hours': (datetime.now(timezone.utc) - self.session_start).total_seconds() / 3600
            }
        except Exception as e:
            self.logger.error(f"Session stats retrieval failed: {e}")
            return {}


# Global performance tracker instance
performance_tracker: Optional[PerformanceTracker] = None


def initialize_performance_tracking(log_dir: str = "logs") -> PerformanceTracker:
    """Initialize global performance tracker"""
    global performance_tracker
    performance_tracker = PerformanceTracker(log_dir)
    return performance_tracker


def get_performance_tracker() -> Optional[PerformanceTracker]:
    """Get global performance tracker instance"""
    return performance_tracker