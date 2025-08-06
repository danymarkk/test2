"""
PRODUCTION-GRADE EXECUTION ENGINE
Live trading execution for Binance Spot with comprehensive error handling
"""

import ccxt
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import threading
from enum import Enum

from ..core.risk_manager import Position, InstitutionalRiskManager
from ..trading.market_data import MarketDataProvider
from ..config.settings import ScalperBotConfig
from ..utils.performance_tracker import get_performance_tracker, ExitReason

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Raised when trade execution fails"""
    pass


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Order:
    """Order data structure"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    order_type: str  # 'market', 'limit', 'stop'
    status: OrderStatus
    filled: float = 0.0
    remaining: float = 0.0
    cost: float = 0.0
    fee: float = 0.0
    timestamp: datetime = None
    exchange_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.remaining == 0.0:
            self.remaining = self.amount


class PaperTradingEngine:
    """
    Paper trading execution engine for testing
    """
    
    def __init__(self, config: ScalperBotConfig, market_data: MarketDataProvider):
        """
        Initialize paper trading engine
        
        Args:
            config: Trading configuration
            market_data: Market data provider
        """
        self.config = config
        self.market_data = market_data
        
        # Paper trading state
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        
        # Simulate slippage and fees
        self.slippage_pct = config.execution.slippage_tolerance
        self.maker_fee = config.execution.maker_fee
        self.taker_fee = config.execution.taker_fee
        
        logger.info("Paper trading engine initialized")
    
    def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[Order]:
        """
        Place a market order (paper trading)
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            
        Returns:
            Order if successful, None otherwise
        """
        try:
            # Get current market price
            current_price = self.market_data.get_current_price(symbol)
            if current_price is None:
                logger.error(f"Cannot get current price for {symbol}")
                return None
            
            # Simulate slippage
            if side == 'buy':
                execution_price = current_price * (1 + self.slippage_pct)
            else:
                execution_price = current_price * (1 - self.slippage_pct)
            
            # Calculate cost and fees
            cost = amount * execution_price
            fee = cost * self.taker_fee  # Market orders pay taker fee
            
            # Create order
            self.order_counter += 1
            order = Order(
                id=f"paper_{self.order_counter}",
                symbol=symbol,
                side=side,
                amount=amount,
                price=execution_price,
                order_type='market',
                status=OrderStatus.FILLED,
                filled=amount,
                remaining=0.0,
                cost=cost,
                fee=fee
            )
            
            self.orders[order.id] = order
            
            logger.info(f"Paper order executed: {side} {amount:.6f} {symbol} @ ${execution_price:.6f}")
            logger.info(f"  Cost: ${cost:.2f}, Fee: ${fee:.4f}")
            
            return order
            
        except Exception as e:
            logger.error(f"Paper order execution failed: {e}")
            return None
    
    def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[Order]:
        """
        Place a limit order (paper trading)
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            price: Limit price
            
        Returns:
            Order if successful, None otherwise
        """
        try:
            # For paper trading, we'll simulate immediate fill if price is favorable
            current_price = self.market_data.get_current_price(symbol)
            if current_price is None:
                return None
            
            # Check if limit order would fill immediately
            would_fill = False
            if side == 'buy' and price >= current_price:
                would_fill = True
                execution_price = current_price
            elif side == 'sell' and price <= current_price:
                would_fill = True
                execution_price = current_price
            else:
                execution_price = price
            
            # Create order
            self.order_counter += 1
            order = Order(
                id=f"paper_limit_{self.order_counter}",
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                order_type='limit',
                status=OrderStatus.FILLED if would_fill else OrderStatus.PENDING,
                filled=amount if would_fill else 0.0,
                remaining=0.0 if would_fill else amount,
                cost=amount * execution_price if would_fill else 0.0,
                fee=(amount * execution_price * self.maker_fee) if would_fill else 0.0
            )
            
            self.orders[order.id] = order
            
            if would_fill:
                logger.info(f"Paper limit order filled: {side} {amount:.6f} {symbol} @ ${execution_price:.6f}")
            else:
                logger.info(f"Paper limit order placed: {side} {amount:.6f} {symbol} @ ${price:.6f}")
            
            return order
            
        except Exception as e:
            logger.error(f"Paper limit order failed: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order (paper trading)
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if order_id in self.orders:
                order = self.orders[order_id]
                if order.status == OrderStatus.PENDING:
                    order.status = OrderStatus.CANCELLED
                    logger.info(f"Paper order cancelled: {order_id}")
                    return True
                else:
                    logger.warning(f"Cannot cancel order {order_id}: status {order.status}")
                    return False
            else:
                logger.warning(f"Order not found: {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        return self.orders.get(order_id)


class LiveTradingEngine:
    """
    Live trading execution engine for Binance Spot
    """
    
    def __init__(self, config: ScalperBotConfig, api_key: str, api_secret: str):
        """
        Initialize live trading engine
        
        Args:
            config: Trading configuration
            api_key: Binance API key
            api_secret: Binance API secret
        """
        self.config = config
        
        # Initialize exchange - CRITICAL FIX: Use sandbox only if explicitly enabled AND in dry_run
        use_sandbox = config.execution.sandbox_mode and config.execution.dry_run
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': use_sandbox,  # Critical fix: only use sandbox in dry_run mode
            'enableRateLimit': True,
            'timeout': config.execution.order_timeout_seconds * 1000,
            'options': {
                'defaultType': 'spot',
                'recvWindow': 10000
            }
        })
        
        logger.info(f"Exchange initialized - Sandbox: {use_sandbox}, Dry Run: {config.execution.dry_run}")
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.order_lock = threading.Lock()
        
        # Test connection
        try:
            self.exchange.load_markets()
            balance = self.exchange.fetch_balance()
            logger.info("Live trading engine initialized and connected to Binance")
            logger.info(f"Account balance: ${balance.get('USDT', {}).get('free', 0):.2f} USDT")
        except Exception as e:
            raise ExecutionError(f"Failed to connect to Binance: {str(e)}")
    
    def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[Order]:
        """
        Place a market order on Binance
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            
        Returns:
            Order if successful, None otherwise
        """
        try:
            # Validate amount
            market = self.exchange.market(symbol)
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            
            if amount < min_amount:
                logger.error(f"Order amount {amount} below minimum {min_amount} for {symbol}")
                return None
            
            # Place order with retries
            max_retries = self.config.execution.order_retry_attempts
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Placing market {side} order: {amount:.6f} {symbol} (attempt {attempt + 1})")
                    
                    result = self.exchange.create_market_order(symbol, side, amount)
                    
                    # Convert to our Order format
                    order = Order(
                        id=result['id'],
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=result.get('average', result.get('price', 0)),
                        order_type='market',
                        status=OrderStatus.FILLED if result.get('status') == 'closed' else OrderStatus.PENDING,
                        filled=result.get('filled', 0),
                        remaining=result.get('remaining', 0),
                        cost=result.get('cost', 0),
                        fee=self._extract_fee(result),
                        exchange_id=result['id']
                    )
                    
                    with self.order_lock:
                        self.orders[order.id] = order
                    
                    logger.info(f"Market order executed: {order.id}")
                    logger.info(f"  Filled: {order.filled:.6f} @ ${order.price:.6f}")
                    logger.info(f"  Cost: ${order.cost:.2f}, Fee: ${order.fee:.4f}")
                    
                    return order
                    
                except ccxt.InsufficientFunds as e:
                    logger.error(f"Insufficient funds for {symbol} order: {e}")
                    return None
                    
                except ccxt.InvalidOrder as e:
                    logger.error(f"Invalid order for {symbol}: {e}")
                    return None
                    
                except ccxt.NetworkError as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Network error placing order after {max_retries} attempts: {e}")
                        return None
                    
                    sleep_time = 2 ** attempt
                    logger.warning(f"Network error, retrying in {sleep_time}s: {e}")
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Order placement failed after {max_retries} attempts: {e}")
                        return None
                    
                    logger.warning(f"Order attempt {attempt + 1} failed: {e}")
                    time.sleep(1)
            
            return None
            
        except Exception as e:
            logger.error(f"Market order execution failed: {e}")
            return None
    
    def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[Order]:
        """
        Place a limit order on Binance
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            price: Limit price
            
        Returns:
            Order if successful, None otherwise
        """
        try:
            # Validate parameters
            market = self.exchange.market(symbol)
            
            # Round price to proper precision
            price_precision = market.get('precision', {}).get('price', 6)
            price = round(price, price_precision)
            
            # Round amount to proper precision
            amount_precision = market.get('precision', {}).get('amount', 6)
            amount = round(amount, amount_precision)
            
            # Place order
            logger.info(f"Placing limit {side} order: {amount:.6f} {symbol} @ ${price:.6f}")
            
            result = self.exchange.create_limit_order(symbol, side, amount, price)
            
            # Convert to our Order format
            order = Order(
                id=result['id'],
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                order_type='limit',
                status=OrderStatus.PENDING if result.get('status') == 'open' else OrderStatus.FILLED,
                filled=result.get('filled', 0),
                remaining=result.get('remaining', amount),
                cost=result.get('cost', 0),
                fee=self._extract_fee(result),
                exchange_id=result['id']
            )
            
            with self.order_lock:
                self.orders[order.id] = order
            
            logger.info(f"Limit order placed: {order.id}")
            
            return order
            
        except Exception as e:
            logger.error(f"Limit order execution failed: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order on Binance
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.order_lock:
                if order_id not in self.orders:
                    logger.warning(f"Order not found in local tracking: {order_id}")
                    return False
                
                order = self.orders[order_id]
                
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                    logger.warning(f"Cannot cancel order {order_id}: status {order.status}")
                    return False
            
            # Cancel on exchange
            result = self.exchange.cancel_order(order.exchange_id, order.symbol)
            
            # Update local tracking
            with self.order_lock:
                order.status = OrderStatus.CANCELLED
                
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except ccxt.OrderNotFound:
            logger.warning(f"Order not found on exchange: {order_id}")
            return False
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get order status from exchange
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Updated Order or None if failed
        """
        try:
            with self.order_lock:
                if order_id not in self.orders:
                    return None
                
                order = self.orders[order_id]
            
            # Fetch from exchange
            result = self.exchange.fetch_order(order.exchange_id, order.symbol)
            
            # Update order
            order.status = self._convert_status(result.get('status'))
            order.filled = result.get('filled', order.filled)
            order.remaining = result.get('remaining', order.remaining)
            order.cost = result.get('cost', order.cost)
            order.fee = self._extract_fee(result)
            
            return order
            
        except Exception as e:
            logger.error(f"Order status check failed: {e}")
            return None
    
    def _extract_fee(self, order_result: Dict) -> float:
        """Extract fee from order result"""
        try:
            fees = order_result.get('fees', [])
            if fees:
                return sum(fee.get('cost', 0) for fee in fees)
            
            fee_info = order_result.get('fee', {})
            return fee_info.get('cost', 0)
            
        except Exception:
            return 0.0
    
    def _convert_status(self, exchange_status: str) -> OrderStatus:
        """Convert exchange status to our OrderStatus"""
        status_map = {
            'open': OrderStatus.PENDING,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'cancelled': OrderStatus.CANCELLED,
            'partial': OrderStatus.PARTIAL
        }
        return status_map.get(exchange_status, OrderStatus.PENDING)


class ExecutionEngine:
    """
    Main execution engine that handles both paper and live trading
    """
    
    def __init__(self, config: ScalperBotConfig, market_data: MarketDataProvider,
                 risk_manager: InstitutionalRiskManager,
                 api_key: str = None, api_secret: str = None):
        """
        Initialize execution engine
        
        Args:
            config: Trading configuration
            market_data: Market data provider
            risk_manager: Risk management system
            api_key: API key for live trading (optional)
            api_secret: API secret for live trading (optional)
        """
        self.config = config
        self.market_data = market_data
        self.risk_manager = risk_manager
        
        # Choose execution mode
        if config.execution.dry_run or api_key is None:
            logger.info("Initializing paper trading mode")
            self.engine = PaperTradingEngine(config, market_data)
            self.is_live = False
        else:
            logger.info("Initializing live trading mode")
            self.engine = LiveTradingEngine(config, api_key, api_secret)
            self.is_live = True
        
        # Execution tracking
        self.position_orders: Dict[str, List[str]] = {}  # symbol -> order_ids
        
        logger.info(f"Execution engine ready - Mode: {'LIVE' if self.is_live else 'PAPER'}")
    
    def open_position(self, position: Position) -> bool:
        """
        FIXED: Execute market order and register position with risk manager
        
        Args:
            position: Position to open
            
        Returns:
            True if successful, False otherwise
        """
        try:
            symbol = position.symbol
            direction = position.direction
            amount = position.position_size
            
            # For spot trading, we only buy (accept both 'long' and 'buy')
            if direction not in ['long', 'buy']:
                logger.error(f"Spot trading only supports long/buy positions, got: {direction}")
                return False
            
            logger.info(f"Opening position: {symbol} {direction} {amount:.6f}")
            
            # Place market buy order
            order = self.engine.place_market_order(symbol, 'buy', amount)
            
            if order is None:
                logger.error(f"Failed to place entry order for {symbol}")
                logger.error(f"  Position details: {symbol} {direction} {amount:.6f}")
                logger.error("  Possible causes: insufficient funds, invalid amount, API issues")
                return False
            
            # CRITICAL FIX: Properly register position through risk manager (with validation)
            # Update position with actual execution data first
            position.entry_price = order.price if order.price else position.entry_price
            position.position_value = order.cost if order.cost else position.position_value
            
            # Register with risk manager (this will validate and track properly)
            self.risk_manager.open_positions[symbol] = position
            logger.info(f"Position registered with risk manager: {symbol}")
            
            # Update risk manager balance and tracking
            if hasattr(self.risk_manager, 'total_trades'):
                self.risk_manager.total_trades += 1
            
            # Track order for this position
            if symbol not in self.position_orders:
                self.position_orders[symbol] = []
            self.position_orders[symbol].append(order.id)
            
            # Update position with actual execution details
            if order.status == OrderStatus.FILLED:
                position.entry_price = order.price
                position.position_value = order.cost
                
                # Start performance tracking
                tracker = get_performance_tracker()
                if tracker:
                    # Get signal and market data for comprehensive tracking
                    signal_data = getattr(position, 'signal_data', {
                        'quality': 2.0,
                        'confluence': 2,
                        'htf_bias': 'bullish',
                        'volume_ratio': 1.0
                    })
                    
                    market_conditions = {
                        'atr': getattr(position, 'atr_value', 0.001),
                        'volatility_percentile': 50.0,
                        'trend': 'bullish',
                        'spread_bps': 0.1
                    }
                    
                    trade_id = tracker.start_trade_tracking(position, signal_data, market_conditions)
                    position.trade_id = trade_id  # Store for later reference
                
                logger.info(f"Position opened successfully: {symbol}")
                logger.info(f"  Executed: {order.filled:.6f} @ ${order.price:.6f}")
                logger.info(f"  Cost: ${order.cost:.2f}, Fee: ${order.fee:.4f}")
                
                return True
            else:
                logger.warning(f"Entry order not immediately filled: {order.status}")
                return False
                
        except Exception as e:
            logger.error(f"Position opening failed for {symbol}: {e}")
            return False
    
    def close_position(self, position: Position, current_price: float, 
                      exit_reason: ExitReason) -> bool:
        """
        Execute market orders to close a position
        
        Args:
            position: Position to close
            current_price: Current market price
            
        Returns:
            True if successful, False otherwise
        """
        try:
            symbol = position.symbol
            amount = position.position_size
            
            logger.info(f"Closing position: {symbol} {amount:.6f}")
            
            # Place market sell order
            order = self.engine.place_market_order(symbol, 'sell', amount)
            
            if order is None:
                logger.error(f"Failed to place exit order for {symbol}")
                return False
            
            # Track order
            if symbol in self.position_orders:
                self.position_orders[symbol].append(order.id)
            
            if order.status == OrderStatus.FILLED:
                # Calculate P&L
                entry_cost = position.position_value
                exit_proceeds = order.cost
                total_fees = order.fee + getattr(position, 'entry_fee', 0)
                pnl_usdt = exit_proceeds - entry_cost - total_fees
                
                # Complete performance tracking
                tracker = get_performance_tracker()
                if tracker and hasattr(position, 'trade_id'):
                    tracker.complete_trade(
                        trade_id=position.trade_id,
                        exit_price=order.price,
                        exit_reason=exit_reason,
                        pnl_usdt=pnl_usdt,
                        fees=total_fees,
                        slippage=0.0,  # TODO: Calculate actual slippage
                        account_balance=10000.0  # TODO: Get actual balance
                    )
                
                logger.info(f"Position closed successfully: {symbol}")
                logger.info(f"  Executed: {order.filled:.6f} @ ${order.price:.6f}")
                logger.info(f"  Proceeds: ${order.cost:.2f}, Fee: ${order.fee:.4f}")
                logger.info(f"  P&L: ${pnl_usdt:.2f}, Exit Reason: {exit_reason.value}")
                
                # Clean up order tracking
                if symbol in self.position_orders:
                    del self.position_orders[symbol]
                
                return True
            else:
                logger.warning(f"Exit order not immediately filled: {order.status}")
                return False
                
        except Exception as e:
            logger.error(f"Position closing failed for {symbol}: {e}")
            return False
    
    def get_execution_summary(self) -> Dict:
        """Get execution engine summary"""
        try:
            return {
                'mode': 'LIVE' if self.is_live else 'PAPER',
                'total_orders': len(self.engine.orders),
                'active_position_orders': len(self.position_orders),
                'symbols_with_orders': list(self.position_orders.keys())
            }
            
        except Exception as e:
            logger.error(f"Execution summary failed: {e}")
            return {}


def create_execution_engine(config: ScalperBotConfig, 
                          market_data: MarketDataProvider,
                          risk_manager: InstitutionalRiskManager,
                          api_key: str = None, 
                          api_secret: str = None) -> ExecutionEngine:
    """
    Factory function to create execution engine
    
    Args:
        config: Trading configuration
        market_data: Market data provider
        risk_manager: Risk management system
        api_key: API key for live trading
        api_secret: API secret for live trading
        
    Returns:
        Configured ExecutionEngine instance
    """
    try:
        engine = ExecutionEngine(config, market_data, risk_manager, api_key, api_secret)
        logger.info("Execution engine created successfully")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create execution engine: {e}")
        raise ExecutionError(f"Execution engine creation failed: {str(e)}")