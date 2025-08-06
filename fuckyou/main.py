#!/usr/bin/env python3
"""
PROFESSIONAL ICT SCALPING BOT - MAIN ENTRY POINT
Production-grade crypto scalping system

Usage:
    python main.py [--config CONFIG_FILE] [--live] [--api-key KEY] [--api-secret SECRET]

Examples:
    # Paper trading (default)
    python main.py
    
    # Live trading with API credentials
    python main.py --live --api-key YOUR_API_KEY --api-secret YOUR_API_SECRET
    
    # Custom configuration
    python main.py --config my_config.json
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.scalper_bot import ProfessionalScalperBot
from src.utils.logger import setup_logging


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Professional ICT Scalping Bot for Crypto Markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Paper trading mode
  %(prog)s --live --api-key KEY --api-secret SECRET  # Live trading
  %(prog)s --config custom.json              # Custom configuration
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        default="scalper_config.json",
        help="Configuration file path (default: scalper_config.json)"
    )
    
    parser.add_argument(
        "--live", "-l",
        action="store_true",
        help="Enable live trading (requires API credentials)"
    )
    
    parser.add_argument(
        "--api-key", "-k",
        help="Binance API key for live trading"
    )
    
    parser.add_argument(
        "--api-secret", "-s",
        help="Binance API secret for live trading"
    )
    
    parser.add_argument(
        "--log-level", "-v",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Log directory (default: logs)"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments"""
    errors = []
    
    # Validate live trading requirements
    if args.live:
        if not args.api_key:
            errors.append("--api-key is required for live trading")
        if not args.api_secret:
            errors.append("--api-secret is required for live trading")
    
    # Configuration validation is handled by the config manager
    # It will auto-create default config if not found
    
    if errors:
        print("Error: Invalid arguments:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


def print_banner():
    """Print startup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║               PROFESSIONAL ICT SCALPING BOT v2.0                 ║
    ║                                                                   ║
    ║  🎯 Inner Circle Trader Concepts for Crypto Scalping            ║
    ║  ⚡ Institutional-Grade Risk Management                          ║
    ║  🔒 Production-Ready Architecture                                ║
    ║  📊 Real-Time Performance Monitoring                            ║
    ║                                                                   ║
    ║  ⚠️  WARNING: Trading involves significant financial risk        ║
    ║      Only trade with capital you can afford to lose             ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point"""
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)
        
        # Print banner
        print_banner()
        
        # Setup initial logging
        setup_logging(
            log_level=args.log_level,
            log_dir=args.log_dir,
            enable_console=True
        )
        
        # Display configuration
        print(f"📁 Configuration: {args.config}")
        print(f"📊 Trading Mode: {'LIVE' if args.live else 'PAPER TRADING'}")
        print(f"📝 Log Level: {args.log_level}")
        print(f"📂 Log Directory: {args.log_dir}")
        
        if args.live:
            print("🔑 API Credentials: Provided")
            print("\n⚠️  LIVE TRADING MODE ENABLED")
            print("   Real money will be used for trading!")
            
            # Confirmation for live trading
            response = input("\nContinue with live trading? (yes/no): ").lower().strip()
            if response != "yes":
                print("Live trading cancelled by user.")
                sys.exit(0)
        else:
            print("\n✅ PAPER TRADING MODE")
            print("   No real money will be used.")
        
        print("\n" + "="*60)
        
        # Create and run bot
        bot = ProfessionalScalperBot(
            config_file=args.config,
            api_key=args.api_key if args.live else None,
            api_secret=args.api_secret if args.live else None
        )
        
        # Setup and start
        bot.setup()
        bot.start()
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()