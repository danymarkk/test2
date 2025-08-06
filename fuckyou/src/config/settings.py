"""
UNIFIED CONFIGURATION SYSTEM
Centralized, validated configuration management for production scalping bot
"""

import json
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration operations fail"""
    pass


@dataclass
class PositionSizingConfig:
    """Position sizing configuration"""
    strategy: str = "fixed_usdt"  # "fixed_usdt", "risk_percent", "dynamic"
    main_strategy_usdt: float = 15.0
    fast_scalp_usdt: float = 7.5
    risk_percent: Optional[float] = None  # e.g., 2.0 for 2% risk per trade
    max_position_percent: float = 20.0  # Max 20% of balance per position


@dataclass
class RiskManagementConfig:
    """Risk management configuration"""
    atr_multiplier_sl: float = 0.75  # Stop loss ATR multiplier
    atr_multiplier_tp: float = 2.0   # Take profit ATR multiplier
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    max_portfolio_risk: float = 0.08  # 8% max total portfolio risk
    max_open_positions: int = 5
    max_hold_hours: float = 4.0
    trailing_stop_trigger: float = 1.0  # Start trailing at +1R
    trailing_stop_distance: float = 0.5  # Trail 0.5R behind peak
    breakeven_trigger: float = 0.5  # Move to breakeven at +0.5R
    drawdown_limit: float = 0.20  # Stop trading at 20% drawdown
    daily_loss_limit: float = 0.05  # Stop trading at 5% daily loss


@dataclass
class ConfluenceConfig:
    """Signal confluence thresholds for spot trading"""
    main_strategy_demand: int = 3   # BUY signal threshold (bullish ICT patterns)
    main_strategy_supply: int = 3   # SELL signal threshold (bearish ICT patterns)
    fast_scalp_demand: int = 2      # Lower threshold for aggressive demand scalping
    fast_scalp_supply: int = 2      # Lower threshold for aggressive supply scalping
    test_mode_threshold: int = 1    # Lower threshold for testing


@dataclass
class TradingConfig:
    """Core trading configuration"""
    # Required fields (no defaults) - must come first
    symbols: List[str]
    primary_symbol: str
    confluence: ConfluenceConfig
    
    # Optional fields (with defaults)
    timeframe: str = "1m"
    htf_timeframe: str = "5m"
    min_signal_quality: float = 2.0
    min_volume_ratio: float = 1.2
    min_signal_strength: str = "medium"
    max_hold_bars: int = 5
    scan_interval: int = 15  # seconds
    lookback_candles: int = 100
    volume_filter_enabled: bool = True
    momentum_filter_enabled: bool = True
    session_filter_enabled: bool = False
    signal_deduplication: bool = True
    max_signals_per_minute_per_symbol: int = 1
    session_start_utc: int = 0
    session_end_utc: int = 24


@dataclass
class ExecutionConfig:
    """Trade execution configuration"""
    exchange: str = "binance"
    sandbox_mode: bool = False
    dry_run: bool = True  # Paper trading by default
    
    # Fees and slippage
    maker_fee: float = 0.00075  # Binance Spot maker fee
    taker_fee: float = 0.001    # Binance Spot taker fee
    slippage_tolerance: float = 0.005  # 0.5% max slippage
    
    # Order management
    order_retry_attempts: int = 3
    order_timeout_seconds: int = 30
    price_precision: Optional[Dict[str, int]] = None


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_level: str = "INFO"
    log_file: str = "scalper_bot.log"
    log_max_size_mb: int = 100
    log_backup_count: int = 5
    
    # JSON Logging for Performance Analysis
    enable_json_logging: bool = True  # Enable JSON format for Data Preview Extension
    log_trades_to_json: bool = True   # Log detailed trade performance data
    log_signals_to_json: bool = True  # Log signal analysis data
    log_daily_summary: bool = True    # Log daily performance summaries
    
    # Performance tracking
    save_trade_history: bool = True
    save_signal_history: bool = True
    performance_report_interval: int = 1800  # 30 minutes for scalping
    
    # Alerts and notifications
    enable_alerts: bool = True
    alert_on_large_loss: bool = True
    alert_threshold_pct: float = 3.0  # Alert on 3%+ loss (tighter for scalping)


@dataclass
class ScalperBotConfig:
    """Complete scalper bot configuration"""
    # Core configurations
    position_sizing: PositionSizingConfig
    risk_management: RiskManagementConfig
    trading: TradingConfig
    execution: ExecutionConfig
    monitoring: MonitoringConfig
    
    # Global settings
    initial_balance: float = 1000.0
    bot_name: str = "ICT_Scalper_Bot"
    version: str = "2.0.0"


class ConfigManager:
    """
    Professional configuration manager with validation and environment support
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self._config: Optional[ScalperBotConfig] = None
        self._config_file: Optional[Path] = None
        
        logger.info(f"Config manager initialized: {self.config_dir}")
    
    def load_config(self, config_file: str = "scalper_config.json") -> ScalperBotConfig:
        """
        Load configuration from file with validation
        
        Args:
            config_file: Configuration file name
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            ConfigurationError: If loading or validation fails
        """
        try:
            config_path = self.config_dir / config_file
            self._config_file = config_path
            
            if not config_path.exists():
                logger.info(f"Config file not found, creating default: {config_path}")
                default_config = self._create_default_config()
                self.save_config(default_config, config_file)
                return default_config
            
            logger.info(f"Loading configuration from: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Parse configuration with validation
            config = self._parse_config(config_data)
            self._validate_config(config)
            
            self._config = config
            logger.info("Configuration loaded and validated successfully")
            
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def save_config(self, config: ScalperBotConfig, config_file: str = "scalper_config.json"):
        """
        Save configuration to file
        
        Args:
            config: Configuration to save
            config_file: Target file name
            
        Raises:
            ConfigurationError: If saving fails
        """
        try:
            config_path = self.config_dir / config_file
            
            # Convert to dictionary
            config_dict = asdict(config)
            
            # Save with pretty formatting
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    def _create_default_config(self) -> ScalperBotConfig:
        """Create default configuration for production scalping"""
        try:
            # Optimized symbols for 1m scalping - high volatility meme coins
            default_symbols = [
                "DOGE/USDT", "PEPE/USDT", "TRUMP/USDT", "PENGU/USDT", 
                "BONK/USDT", "WIF/USDT", "SPX/USDT"
            ]
            
            # Precise price formatting for tiny meme coin prices
            default_precision = {
                "DOGE/USDT": 6,
                "PEPE/USDT": 10,
                "TRUMP/USDT": 4,
                "PENGU/USDT": 4,
                "BONK/USDT": 8,
                "WIF/USDT": 4,
                "SPX/USDT": 4
            }
            
            config = ScalperBotConfig(
                position_sizing=PositionSizingConfig(
                    main_strategy_usdt=15.0,
                    fast_scalp_usdt=7.5
                ),
                risk_management=RiskManagementConfig(
                    atr_multiplier_sl=0.75,
                    atr_multiplier_tp=2.0,
                    max_open_positions=8,
                    max_hold_hours=0.083,  # 5 minutes for scalping
                    drawdown_limit=0.15,   # 15% max drawdown
                    daily_loss_limit=0.03  # 3% daily loss limit
                ),
                trading=TradingConfig(
                    symbols=default_symbols,
                    primary_symbol="DOGE/USDT",
                    timeframe="1m",
                    htf_timeframe="5m",
                    confluence=ConfluenceConfig(
                        main_strategy_demand=3.5,   # ChatGPT Analysis: raise from 2→3.5
                        main_strategy_supply=3.5,   # ChatGPT Analysis: raise from 2→3.5
                        fast_scalp_demand=3.0,      # Higher quality fast scalping
                        fast_scalp_supply=3.0,      # Higher quality fast scalping
                        test_mode_threshold=2.0     # Even test mode needs quality
                    ),
                    max_hold_bars=5,           # 5 minute max hold
                    scan_interval=8,           # 8 second scanning
                    min_signal_quality=2.0,   # Base quality threshold
                    min_volume_ratio=1.2,      # ChatGPT Analysis: 70th percentile
                    min_signal_strength="medium",  # Drop weak signals entirely
                    signal_deduplication=True, # 1 signal per symbol per minute
                    max_signals_per_minute_per_symbol=1
                ),
                execution=ExecutionConfig(
                    dry_run=True,  # Start in paper trading
                    price_precision=default_precision,
                    slippage_tolerance=0.005
                ),
                monitoring=MonitoringConfig(
                    performance_report_interval=1800  # 30 minute reports
                ),
                initial_balance=1000.0,
                bot_name="ICT_Scalper_Bot_v2"
            )
            
            logger.info("Created default configuration")
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create default config: {str(e)}")
    
    def _parse_config(self, config_data: Dict[str, Any]) -> ScalperBotConfig:
        """Parse configuration dictionary into structured config"""
        try:
            # Parse each section with defaults
            position_sizing = PositionSizingConfig(
                **config_data.get("position_sizing", {})
            )
            
            risk_management = RiskManagementConfig(
                **config_data.get("risk_management", {})
            )
            
            confluence_data = config_data.get("trading", {}).get("confluence", {})
            confluence = ConfluenceConfig(**confluence_data)
            
            trading_data = config_data.get("trading", {})
            trading_data["confluence"] = confluence
            trading = TradingConfig(**trading_data)
            
            execution = ExecutionConfig(
                **config_data.get("execution", {})
            )
            
            monitoring = MonitoringConfig(
                **config_data.get("monitoring", {})
            )
            
            # Global settings
            global_settings = config_data.get("global", {})
            
            config = ScalperBotConfig(
                position_sizing=position_sizing,
                risk_management=risk_management,
                trading=trading,
                execution=execution,
                monitoring=monitoring,
                initial_balance=global_settings.get("initial_balance", 1000.0),
                bot_name=global_settings.get("bot_name", "ICT_Scalper_Bot"),
                version=global_settings.get("version", "2.0.0")
            )
            
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to parse configuration: {str(e)}")
    
    def _validate_config(self, config: ScalperBotConfig):
        """Validate configuration parameters"""
        try:
            # Validate position sizing
            if config.position_sizing.main_strategy_usdt <= 0:
                raise ValueError("main_strategy_usdt must be positive")
            
            if config.position_sizing.risk_percent and (
                config.position_sizing.risk_percent <= 0 or 
                config.position_sizing.risk_percent > 10
            ):
                raise ValueError("risk_percent must be between 0 and 10")
            
            # Validate risk management
            if config.risk_management.max_risk_per_trade <= 0 or config.risk_management.max_risk_per_trade > 0.1:
                raise ValueError("max_risk_per_trade must be between 0 and 0.1 (10%)")
            
            if config.risk_management.atr_multiplier_sl <= 0:
                raise ValueError("atr_multiplier_sl must be positive")
            
            # Validate trading config
            if not config.trading.symbols:
                raise ValueError("symbols list cannot be empty")
            
            if config.trading.primary_symbol not in config.trading.symbols:
                raise ValueError("primary_symbol must be in symbols list")
            
            valid_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"]
            if config.trading.timeframe not in valid_timeframes:
                raise ValueError(f"timeframe must be one of: {valid_timeframes}")
            
            # Validate confluence thresholds
            if config.trading.confluence.main_strategy_demand < 1:
                raise ValueError("main_strategy_demand threshold must be >= 1")
            if config.trading.confluence.main_strategy_supply < 1:
                raise ValueError("main_strategy_supply threshold must be >= 1")
            
            # Validate execution config
            if config.execution.slippage_tolerance < 0 or config.execution.slippage_tolerance > 0.1:
                raise ValueError("slippage_tolerance must be between 0 and 0.1 (10%)")
            
            # Validate fees
            if config.execution.maker_fee < 0 or config.execution.taker_fee < 0:
                raise ValueError("fees cannot be negative")
            
            logger.info("Configuration validation passed")
            
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")
    
    def get_symbol_precision(self, symbol: str) -> int:
        """Get price precision for a symbol"""
        if self._config and self._config.execution.price_precision:
            return self._config.execution.price_precision.get(symbol, 6)
        return 6  # Default precision
    
    def update_config(self, **kwargs):
        """Update configuration parameters dynamically"""
        if not self._config:
            raise ConfigurationError("No configuration loaded")
        
        # Implementation for dynamic config updates would go here
        # For now, just log the request
        logger.info(f"Config update requested: {kwargs}")
    
    def reload_config(self):
        """Reload configuration from file"""
        if self._config_file:
            return self.load_config(self._config_file.name)
        else:
            raise ConfigurationError("No config file to reload")
    
    @property
    def config(self) -> Optional[ScalperBotConfig]:
        """Get current configuration"""
        return self._config


# Thread-safe global configuration manager instance
_config_manager: Optional[ConfigManager] = None
_config_lock = threading.Lock()


def get_config_manager(config_dir: str = "config") -> ConfigManager:
    """
    Get global configuration manager instance (thread-safe singleton pattern)
    
    Args:
        config_dir: Configuration directory
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    # Thread-safe double-checked locking pattern
    if _config_manager is None:
        with _config_lock:
            if _config_manager is None:  # Double check inside lock
                _config_manager = ConfigManager(config_dir)
    
    return _config_manager


def load_config(config_file: str = "scalper_config.json") -> ScalperBotConfig:
    """
    Convenience function to load configuration
    
    Args:
        config_file: Configuration file name
        
    Returns:
        Loaded configuration
    """
    manager = get_config_manager()
    return manager.load_config(config_file)