"""
PROFESSIONAL ICT SCALPING BOT
Production-grade crypto scalping system with institutional-level architecture
"""

import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Optional
import threading
import logging

# Core imports
from .config.settings import load_config, ScalperBotConfig
from .utils.performance_tracker import initialize_performance_tracking
from .core.risk_manager import InstitutionalRiskManager, RiskParameters
from .trading.market_data import MarketDataProvider
from .trading.execution_engine import ExecutionEngine
from .strategies.ict_scalper import ICTScalpingStrategy
from .utils.logger import setup_logging, log_signal, log_trade_open, log_trade_close

logger = logging.getLogger(__name__)


class ScalperBotError(Exception):
    """Raised when scalper bot operations fail"""
    pass


class ProfessionalScalperBot:
    """
    Professional ICT Scalping Bot
    
    Features:
    - Institutional-grade ICT pattern detection
    - Multi-symbol trading with correlation management
    - Professional risk management and position sizing
    - Real-time execution with paper/live trading modes
    - Comprehensive performance tracking and monitoring
    - Graceful shutdown and error recovery
    """
    
    def __init__(self, config_file: str = "scalper_config.json",
                 api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None):
        """
        Initialize the professional scalper bot
        
        Args:
            config_file: Configuration file name
            api_key: Binance API key for live trading (optional)
            api_secret: Binance API secret for live trading (optional)
        """
        self.config_file = config_file
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Core components (initialized in setup)
        self.config: Optional[ScalperBotConfig] = None
        self.market_data: Optional[MarketDataProvider] = None
        self.risk_manager: Optional[InstitutionalRiskManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.strategy: Optional[ICTScalpingStrategy] = None
        
        # Bot state (thread-safe)
        self._state_lock = threading.Lock()
        self._is_running = False
        self._shutdown_requested = False
        self.last_health_check = datetime.now()
        self.last_performance_report = datetime.now()
        
        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.session_start_time = datetime.now()
        self.total_signals_processed = 0
        self.total_trades_executed = 0
        
        logger.info("Professional Scalper Bot initialized")
    
    @property 
    def is_running(self) -> bool:
        """Thread-safe is_running property"""
        with self._state_lock:
            return self._is_running
    
    @is_running.setter
    def is_running(self, value: bool):
        """Thread-safe is_running setter"""
        with self._state_lock:
            self._is_running = value
    
    @property
    def shutdown_requested(self) -> bool:
        """Thread-safe shutdown_requested property"""
        with self._state_lock:
            return self._shutdown_requested
    
    @shutdown_requested.setter  
    def shutdown_requested(self, value: bool):
        """Thread-safe shutdown_requested setter"""
        with self._state_lock:
            self._shutdown_requested = value
    
    def setup(self):
        """Initialize all bot components"""
        try:
            logger.info("Setting up bot components...")
            
            # Load configuration
            self.config = load_config(self.config_file)
            logger.info(f"Configuration loaded: {self.config.bot_name} v{self.config.version}")
            
            # Setup logging based on config
            setup_logging(
                log_level=self.config.monitoring.log_level,
                log_file=self.config.monitoring.log_file,
                max_size_mb=self.config.monitoring.log_max_size_mb,
                backup_count=self.config.monitoring.log_backup_count,
                log_dir="logs",
                enable_json=self.config.monitoring.enable_json_logging
            )
            
            # Initialize performance tracking with JSON logging
            if self.config.monitoring.log_trades_to_json:
                self.performance_tracker = initialize_performance_tracking("logs")
                logger.info("Performance tracking system initialized with comprehensive JSON logging")
                logger.info("Trade data will be logged to: logs/trade_performance.jsonl")
                logger.info("Signal data will be logged to: logs/signal_analysis.jsonl")
                logger.info("Daily summaries will be logged to: logs/daily_summary.jsonl")
            
            # Initialize market data provider (always mainnet for public data)
            self.market_data = MarketDataProvider(
                exchange_id=self.config.execution.exchange,
                sandbox=False  # FIXED: Always use mainnet for market data
            )
            
            # 🚨 STALE DATA FIX: Clear cache to force fresh data
            self.market_data.clear_cache()
            
            # Initialize risk manager
            risk_params = RiskParameters(
                max_risk_per_trade=self.config.risk_management.max_risk_per_trade,
                max_portfolio_risk=self.config.risk_management.max_portfolio_risk,
                max_open_positions=self.config.risk_management.max_open_positions,
                atr_multiplier_sl=self.config.risk_management.atr_multiplier_sl,
                atr_multiplier_tp=self.config.risk_management.atr_multiplier_tp,
                max_hold_hours=self.config.risk_management.max_hold_hours,
                trailing_stop_trigger=self.config.risk_management.trailing_stop_trigger,
                trailing_stop_distance=self.config.risk_management.trailing_stop_distance,
                breakeven_trigger=self.config.risk_management.breakeven_trigger,
                drawdown_limit=self.config.risk_management.drawdown_limit,
                daily_loss_limit=self.config.risk_management.daily_loss_limit
            )
            
            self.risk_manager = InstitutionalRiskManager(
                initial_balance=self.config.initial_balance,
                risk_params=risk_params
            )
            
            # Initialize execution engine
            self.execution_engine = ExecutionEngine(
                config=self.config,
                market_data=self.market_data,
                risk_manager=self.risk_manager,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            # Initialize ICT strategy
            self.strategy = ICTScalpingStrategy(
                config=self.config,
                market_data=self.market_data,
                risk_manager=self.risk_manager
            )
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("Bot setup completed successfully")
            logger.info(f"Trading symbols: {', '.join(self.config.trading.symbols)}")
            logger.info(f"Initial balance: ${self.config.initial_balance:,.2f}")
            logger.info(f"Execution mode: {'LIVE' if not self.config.execution.dry_run else 'PAPER'}")
            
        except Exception as e:
            logger.error(f"Bot setup failed: {e}", exc_info=True)
            raise ScalperBotError(f"Setup failed: {str(e)}")
    
    def start(self):
        """Start the scalper bot"""
        try:
            if self.is_running:
                logger.warning("Bot is already running")
                return
            
            logger.info("🚀 Starting Professional ICT Scalper Bot")
            logger.info("=" * 60)
            
            self.is_running = True
            self.shutdown_requested = False
            
            # Start main trading loop in separate thread
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            logger.info("Bot started successfully")
            logger.info("Press Ctrl+C to stop")
            
            # Keep main thread alive
            try:
                while self.is_running and not self.shutdown_requested:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.stop()
            
        except Exception as e:
            logger.error(f"Bot start failed: {e}", exc_info=True)
            self.stop()
            raise ScalperBotError(f"Start failed: {str(e)}")
    
    def stop(self):
        """Stop the scalper bot gracefully"""
        try:
            if not self.is_running:
                logger.info("Bot is not running")
                return
            
            logger.info("🛑 Stopping Professional ICT Scalper Bot...")
            
            self.shutdown_requested = True
            
            # Close all open positions
            self._close_all_positions()
            
            # Wait for threads to finish
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=30)
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
            
            self.is_running = False
            
            # Final performance report
            self._generate_final_report()
            
            logger.info("✅ Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Bot stop failed: {e}", exc_info=True)
    
    def _main_loop(self):
        """Main trading loop"""
        try:
            logger.info("Main trading loop started")
            
            # CRITICAL FIX: Wait for market to settle before trading
            startup_delay = 60  # 1 minute startup delay
            logger.info(f"⏳ Startup delay: waiting {startup_delay}s for fresh market data...")
            time.sleep(startup_delay)
            
            while self.is_running and not self.shutdown_requested:
                try:
                    loop_start = time.time()
                    
                    # Check if we can trade
                    can_trade, reason = self.risk_manager.can_trade()
                    if not can_trade:
                        logger.warning(f"Trading suspended: {reason}")
                        time.sleep(60)  # Wait 1 minute before checking again
                        continue
                    
                    # Update existing positions
                    self._update_positions()
                    
                    # Scan for new signals
                    open_positions_count = len(self.risk_manager.open_positions)
                    max_positions = self.config.risk_management.max_open_positions
                    
                    logger.debug(f"Position check: {open_positions_count}/{max_positions} positions")
                    
                    if open_positions_count < max_positions:
                        logger.debug("Starting signal scan...")
                        self._scan_for_signals()
                    else:
                        logger.debug(f"Skipping signal scan: max positions reached ({open_positions_count}/{max_positions})")
                    
                    # Calculate sleep time to maintain scan interval
                    loop_duration = time.time() - loop_start
                    sleep_time = max(0, self.config.trading.scan_interval - loop_duration)
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    time.sleep(5)  # Short delay before retrying
            
            logger.info("Main trading loop ended")
            
        except Exception as e:
            logger.error(f"Main loop failed: {e}", exc_info=True)
    
    def _monitor_loop(self):
        """Monitoring and maintenance loop"""
        try:
            logger.info("Monitor loop started")
            
            while self.is_running and not self.shutdown_requested:
                try:
                    # Health check every 5 minutes
                    if datetime.now() - self.last_health_check > timedelta(minutes=5):
                        self._perform_health_check()
                        self.last_health_check = datetime.now()
                    
                    # Performance report based on config interval
                    report_interval = timedelta(seconds=self.config.monitoring.performance_report_interval)
                    if datetime.now() - self.last_performance_report > report_interval:
                        self._generate_performance_report()
                        self.last_performance_report = datetime.now()
                    
                    # Cleanup tasks
                    self._cleanup_tasks()
                    
                    time.sleep(60)  # Monitor loop runs every minute
                    
                except Exception as e:
                    logger.error(f"Error in monitor loop: {e}", exc_info=True)
                    time.sleep(60)
            
            logger.info("Monitor loop ended")
            
        except Exception as e:
            logger.error(f"Monitor loop failed: {e}", exc_info=True)
    
    def _scan_for_signals(self):
        """Scan all symbols for trading signals - ONLY ON FRESH DATA"""
        try:
            # CRITICAL FIX: Don't analyze stale data on startup
            if not self._is_market_data_fresh():
                logger.info("⏳ Waiting for fresh market data before signal analysis...")
                return
                
            logger.info(f"🔍 Scanning {len(self.config.trading.symbols)} symbols for signals...")
            logger.info(f"Symbols to scan: {self.config.trading.symbols}")
            logger.info(f"Open positions: {list(self.risk_manager.open_positions.keys())}")
            
            for symbol in self.config.trading.symbols:
                try:
                    # Skip if we already have a position in this symbol
                    if symbol in self.risk_manager.open_positions:
                        logger.info(f"⏭️ Skipping {symbol} - already have position")
                        continue
                    
                    # Analyze symbol
                    logger.info(f"📊 Analyzing {symbol}...")
                    analysis = self.strategy.analyze_symbol(symbol)
                    if analysis is None:
                        logger.debug(f"Analysis failed for {symbol}")
                        continue
                    
                    # Log analysis results
                    demand_sig = analysis.get('demand_signal', False)
                    supply_sig = analysis.get('supply_signal', False)
                    conf_demand = analysis.get('confluence_demand', 0)
                    conf_supply = analysis.get('confluence_supply', 0)
                    qual_demand = analysis.get('quality_demand', 0)
                    qual_supply = analysis.get('quality_supply', 0)
                    
                    logger.info(f"🔍 {symbol} analysis: demand={demand_sig}({conf_demand:.1f},{qual_demand:.1f}) supply={supply_sig}({conf_supply:.1f},{qual_supply:.1f})")
                    
                    # Evaluate signal
                    should_trade, direction, signal_details = self.strategy.evaluate_signal(analysis)
                    
                    logger.info(f"🔍 {symbol} evaluation: should_trade={should_trade}, direction={direction}")
                    if not should_trade:
                        logger.info(f"❌ {symbol} rejection reason: {signal_details.get('reason', 'Unknown')}")
                    
                    if should_trade:
                        self.total_signals_processed += 1
                        
                        logger.info(f"✅ {symbol}: VALID {direction.upper()} SIGNAL - Starting execution...")
                        
                        # Log signal
                        log_signal(
                            logger, symbol, direction,
                            signal_details['confluence'],
                            signal_details['quality'],
                            patterns=signal_details['patterns_active'],
                            bias=signal_details['bias']
                        )
                        
                        # Execute signal
                        position = self.strategy.execute_signal(analysis, signal_details)
                        
                        if position:
                            # Open position through execution engine
                            success = self.execution_engine.open_position(position)
                            
                            if success:
                                self.total_trades_executed += 1
                                
                                log_trade_open(
                                    logger, symbol, direction,
                                    position.entry_price,
                                    position.position_size,
                                    stop_loss=position.stop_loss,
                                    take_profit=position.take_profit
                                )
                            else:
                                logger.error(f"Failed to execute position for {symbol}")
                                # Position was never added to risk manager since execution failed
                                # No cleanup needed - position object just gets discarded
                    
                except Exception as e:
                    logger.error(f"Signal scan failed for {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Signal scanning failed: {e}")
    
    def _is_market_data_fresh(self) -> bool:
        """Check if market data is fresh enough for live trading"""
        try:
            import time
            current_time = time.time()
            
            # Check if we have any recent data for major symbols
            test_symbols = ['DOGE/USDT', 'PEPE/USDT']
            fresh_count = 0
            
            for symbol in test_symbols:
                try:
                    # Get 1m data to check freshness
                    df = self.market_data.fetch_ohlcv(symbol, '1m', limit=5, use_cache=False)
                    if df is not None and len(df) > 0:
                        # Handle both pandas Timestamp and Unix timestamp
                        ts_value = df.iloc[-1]['timestamp']
                        if hasattr(ts_value, 'timestamp'):
                            # pandas Timestamp object
                            latest_timestamp = ts_value.timestamp()
                        else:
                            # Unix timestamp in milliseconds
                            latest_timestamp = ts_value / 1000
                        time_diff = current_time - latest_timestamp
                        
                        # Data should be less than 2 minutes old for live trading
                        if time_diff < 120:  # 2 minutes
                            fresh_count += 1
                            logger.debug(f"{symbol}: Fresh data ({time_diff:.0f}s old)")
                        else:
                            logger.warning(f"{symbol}: Stale data ({time_diff:.0f}s old)")
                
                except Exception as e:
                    logger.warning(f"Could not check data freshness for {symbol}: {e}")
            
            is_fresh = fresh_count >= len(test_symbols) // 2  # At least half should be fresh
            
            if not is_fresh:
                logger.info("💤 Market data not fresh enough - waiting for real-time data...")
            
            return is_fresh
            
        except Exception as e:
            logger.error(f"Data freshness check failed: {e}")
            return False  # Conservative: don't trade if we can't verify freshness
    
    def _update_positions(self):
        """Update all open positions"""
        try:
            if not self.risk_manager.open_positions:
                return
            
            # Get current prices for all symbols with open positions
            symbols = list(self.risk_manager.open_positions.keys())
            current_prices = {}
            
            for symbol in symbols:
                price = self.market_data.get_current_price(symbol)
                if price is not None:
                    current_prices[symbol] = price
            
            # Update positions and check for exits
            exits_needed = self.strategy.update_positions(current_prices)
            
            # Execute exits
            for symbol, exit_reason in exits_needed:
                if symbol in current_prices:
                    exit_price = current_prices[symbol]
                    logger.info(f"🚪 Executing exit for {symbol}: reason='{exit_reason}', price=${exit_price:.6f}")
                    
                    # Close through execution engine
                    position = self.risk_manager.open_positions.get(symbol)
                    if position:
                        # Convert exit_reason string to ExitReason enum
                        from src.utils.performance_tracker import ExitReason
                        
                        # Complete mapping with error handling
                        exit_reason_map = {
                            'stop_loss': ExitReason.STOP_LOSS,
                            'take_profit': ExitReason.TAKE_PROFIT,
                            'timeout': ExitReason.TIMEOUT,
                            'trailing_stop': ExitReason.TRAILING_STOP,
                            'momentum_death': ExitReason.MOMENTUM_DEATH,
                            'error': ExitReason.MANUAL,  # Error fallback
                            'manual': ExitReason.MANUAL   # Explicit manual
                        }
                        
                        exit_reason_enum = exit_reason_map.get(exit_reason)
                        if exit_reason_enum is None:
                            logger.error(f"UNKNOWN EXIT REASON: '{exit_reason}' for {symbol} - defaulting to MANUAL")
                            logger.error(f"Expected reasons: {list(exit_reason_map.keys())}")
                            exit_reason_enum = ExitReason.MANUAL
                        
                        success = self.execution_engine.close_position(position, exit_price, exit_reason_enum)
                        
                        if success:
                            # Close in risk manager
                            pnl = self.strategy.close_position(symbol, exit_price, exit_reason)
                            logger.info(f"✅ {symbol} position closed: {exit_reason_enum.value}, P&L: ${pnl:.2f}")
                            
                            if pnl is not None:
                                log_trade_close(
                                    logger, symbol, exit_price, pnl, exit_reason,
                                    hold_time=(datetime.now() - position.entry_time).total_seconds() / 60
                                )
                        else:
                            logger.error(f"Failed to execute position close for {symbol}")
                            
        except Exception as e:
            logger.error(f"Position update failed: {e}")
    
    def _close_all_positions(self):
        """Close all open positions during shutdown"""
        try:
            if not self.risk_manager.open_positions:
                return
            
            logger.info("Closing all open positions...")
            
            for symbol in list(self.risk_manager.open_positions.keys()):
                try:
                    position = self.risk_manager.open_positions[symbol]
                    current_price = self.market_data.get_current_price(symbol)
                    
                    if current_price is not None:
                        from src.utils.performance_tracker import ExitReason
                        success = self.execution_engine.close_position(position, current_price, ExitReason.MANUAL)
                        
                        if success:
                            pnl = self.strategy.close_position(symbol, current_price, "shutdown")
                            logger.info(f"Closed {symbol} position on shutdown: P&L ${pnl:.2f}")
                        else:
                            logger.error(f"Failed to close {symbol} position")
                    
                except Exception as e:
                    logger.error(f"Error closing position {symbol}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
    
    def _perform_health_check(self):
        """Perform system health check"""
        try:
            logger.debug("Performing health check...")
            
            # Check market data connection
            test_symbol = self.config.trading.primary_symbol
            test_price = self.market_data.get_current_price(test_symbol)
            
            if test_price is None:
                logger.warning("Market data connection issue detected")
            
            # Check risk manager state
            portfolio_summary = self.risk_manager.get_portfolio_summary()
            current_drawdown = portfolio_summary.get('current_drawdown', 0)
            
            if current_drawdown > 15:
                logger.warning(f"High drawdown detected: {current_drawdown:.1f}%")
            
            # Check execution engine
            exec_summary = self.execution_engine.get_execution_summary()
            logger.debug(f"Execution engine status: {exec_summary}")
            
            logger.debug("Health check completed")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _generate_performance_report(self):
        """Generate periodic performance report"""
        try:
            logger.info("=== PERFORMANCE REPORT ===")
            
            # Strategy performance
            strategy_perf = self.strategy.get_strategy_performance()
            portfolio = strategy_perf.get('portfolio', {})
            
            # Session statistics
            session_duration = (datetime.now() - self.session_start_time).total_seconds() / 3600
            
            logger.info(f"Session Duration: {session_duration:.1f} hours")
            logger.info(f"Signals Processed: {self.total_signals_processed}")
            logger.info(f"Trades Executed: {self.total_trades_executed}")
            logger.info(f"Current Balance: ${portfolio.get('current_balance', 0):,.2f}")
            logger.info(f"Total Return: {portfolio.get('total_return', 0):.2f}%")
            logger.info(f"Win Rate: {portfolio.get('win_rate', 0):.1f}%")
            logger.info(f"Max Drawdown: {portfolio.get('max_drawdown', 0):.2f}%")
            logger.info(f"Open Positions: {portfolio.get('open_positions', 0)}")
            
            logger.info("=== END REPORT ===")
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
    
    def _generate_final_report(self):
        """Generate final session report"""
        try:
            logger.info("\n" + "=" * 60)
            logger.info("FINAL SESSION REPORT")
            logger.info("=" * 60)
            
            session_duration = datetime.now() - self.session_start_time
            hours = session_duration.total_seconds() / 3600
            
            strategy_perf = self.strategy.get_strategy_performance()
            portfolio = strategy_perf.get('portfolio', {})
            
            logger.info(f"Session Duration: {hours:.2f} hours")
            logger.info(f"Total Signals: {self.total_signals_processed}")
            logger.info(f"Total Trades: {self.total_trades_executed}")
            logger.info(f"Final Balance: ${portfolio.get('current_balance', 0):,.2f}")
            logger.info(f"Total Return: {portfolio.get('total_return', 0):.2f}%")
            logger.info(f"Win Rate: {portfolio.get('win_rate', 0):.1f}%")
            logger.info(f"Max Drawdown: {portfolio.get('max_drawdown', 0):.2f}%")
            logger.info(f"Total Fees: ${portfolio.get('total_fees', 0):.2f}")
            
            if self.total_trades_executed > 0:
                trades_per_hour = self.total_trades_executed / hours
                logger.info(f"Trade Frequency: {trades_per_hour:.2f} trades/hour")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Final report generation failed: {e}")
    
    def _cleanup_tasks(self):
        """Perform periodic cleanup tasks"""
        try:
            # Cleanup market data cache
            if self.market_data:
                self.market_data.cleanup_cache()
                
        except Exception as e:
            logger.error(f"Cleanup tasks failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
        sys.exit(0)


def main():
    """Main entry point for the scalper bot"""
    try:
        # Create bot instance
        bot = ProfessionalScalperBot()
        
        # Setup and start
        bot.setup()
        bot.start()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Bot execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()