"""
PRODUCTION-GRADE LOGGING SYSTEM
Structured logging with rotation, filtering, and performance monitoring
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import threading


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        try:
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 
                              'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                    log_entry[key] = value
            
            return json.dumps(log_entry, default=str)
            
        except Exception as e:
            # Fallback to standard formatting if JSON fails
            return f"LOG_FORMAT_ERROR: {str(e)} | Original: {record.getMessage()}"


class TradingLogFilter(logging.Filter):
    """Custom filter for trading-specific log levels"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records based on trading context"""
        # Add trading-specific context
        if hasattr(record, 'symbol'):
            record.trading_symbol = record.symbol
        
        if hasattr(record, 'pnl'):
            record.trading_pnl = record.pnl
            
        # Filter out excessive debug messages in production
        if record.levelno == logging.DEBUG:
            # Only allow debug from core trading modules
            allowed_modules = ['signals', 'patterns', 'risk_manager', 'execution_engine']
            if not any(module in record.module for module in allowed_modules):
                return False
        
        return True


class PerformanceLogger:
    """Performance monitoring logger"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._timers: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        with self._lock:
            self._timers[operation] = datetime.now().timestamp()
    
    def end_timer(self, operation: str, log_level: int = logging.INFO):
        """End timing and log the duration"""
        with self._lock:
            if operation in self._timers:
                duration = datetime.now().timestamp() - self._timers[operation]
                self.logger.log(
                    log_level, 
                    f"Performance: {operation} completed in {duration:.3f}s",
                    extra={'operation': operation, 'duration': duration}
                )
                del self._timers[operation]
            else:
                self.logger.warning(f"Timer not found for operation: {operation}")


def setup_logging(log_level: str = "INFO", 
                 log_file: str = "scalper_bot.log",
                 max_size_mb: int = 100,
                 backup_count: int = 5,
                 log_dir: str = "logs",
                 enable_json: bool = False,
                 enable_console: bool = True) -> logging.Logger:
    """
    Setup production-grade logging system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name
        max_size_mb: Maximum log file size in MB
        backup_count: Number of backup files to keep
        log_dir: Directory for log files
        enable_json: Use JSON formatting
        enable_console: Enable console output
        
    Returns:
        Configured logger instance
    """
    try:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Convert log level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create root logger
        logger = logging.getLogger()
        logger.setLevel(numeric_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatters
        if enable_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(TradingLogFilter())
        logger.addHandler(file_handler)
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            
            # Use simpler format for console
            console_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)-5s | %(name)-15s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            console_handler.addFilter(TradingLogFilter())
            logger.addHandler(console_handler)
        
        # Error handler (separate file for errors)
        error_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / 'errors.log',
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        # Set third-party library log levels
        logging.getLogger('ccxt').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        
        logger.info("Logging system initialized")
        logger.info(f"Log level: {log_level}")
        logger.info(f"Log file: {log_path / log_file}")
        logger.info(f"JSON formatting: {enable_json}")
        
        return logger
        
    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger()
        logger.error(f"Failed to setup advanced logging: {e}")
        return logger


def create_performance_logger(name: str) -> PerformanceLogger:
    """Create a performance logger for a specific component"""
    logger = logging.getLogger(name)
    return PerformanceLogger(logger)


# Trading-specific logging helpers
def log_signal(logger: logging.Logger, symbol: str, direction: str, 
               confluence: float, quality: float, **kwargs):
    """Log a trading signal with structured data"""
    logger.info(
        f"SIGNAL: {symbol} {direction.upper()} - Conf:{confluence} Qual:{quality:.1f}",
        extra={
            'event_type': 'signal',
            'symbol': symbol,
            'direction': direction,
            'confluence': confluence,
            'quality': quality,
            **kwargs
        }
    )


def log_trade_open(logger: logging.Logger, symbol: str, direction: str,
                   entry_price: float, position_size: float, **kwargs):
    """Log trade opening with structured data"""
    logger.info(
        f"TRADE_OPEN: {symbol} {direction.upper()} @ ${entry_price:.6f} Size:{position_size:.6f}",
        extra={
            'event_type': 'trade_open',
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'position_size': position_size,
            **kwargs
        }
    )


def log_trade_close(logger: logging.Logger, symbol: str, exit_price: float,
                    pnl: float, exit_reason: str, **kwargs):
    """Log trade closing with structured data"""
    logger.info(
        f"TRADE_CLOSE: {symbol} @ ${exit_price:.6f} P&L:${pnl:.2f} ({exit_reason})",
        extra={
            'event_type': 'trade_close',
            'symbol': symbol,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason,
            **kwargs
        }
    )


def log_error_with_context(logger: logging.Logger, error: Exception, 
                          context: Dict[str, Any] = None):
    """Log error with additional context"""
    logger.error(
        f"ERROR: {str(error)}",
        extra={
            'event_type': 'error',
            'error_type': type(error).__name__,
            'context': context or {},
        },
        exc_info=True
    )


# Configure logging for specific modules
def configure_module_logging():
    """Configure logging for specific modules with appropriate levels"""
    
    # Core modules - detailed logging
    logging.getLogger('src.core').setLevel(logging.DEBUG)
    logging.getLogger('src.strategies').setLevel(logging.INFO)
    logging.getLogger('src.trading').setLevel(logging.INFO)
    
    # Reduce verbosity for less critical modules
    logging.getLogger('src.utils').setLevel(logging.WARNING)
    logging.getLogger('src.config').setLevel(logging.INFO)