"""
IntelCLaw - Autonomous AI Agent for Windows

Main entry point for the IntelCLaw application.
"""

import asyncio
import sys
from pathlib import Path

from loguru import logger


def setup_logging():
    """Configure logging."""
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    
    # File handler
    logger.add(
        log_path / "intelclaw.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )


async def main():
    """Main entry point."""
    setup_logging()
    
    logger.info("=" * 50)
    logger.info("IntelCLaw - Autonomous AI Agent")
    logger.info("=" * 50)
    
    try:
        from intelclaw.core.app import IntelCLawApp
        
        app = IntelCLawApp()
        await app.startup()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested via keyboard")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


def cli():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="IntelCLaw - Autonomous AI Agent for Windows"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Start in safe mode with minimal features"
    )
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Clear all memory on startup"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="IntelCLaw 0.1.0"
    )
    
    args = parser.parse_args()
    
    # Store args for app to use
    import os
    if args.safe_mode:
        os.environ["INTELCLAW_SAFE_MODE"] = "true"
    if args.reset_memory:
        os.environ["INTELCLAW_RESET_MEMORY"] = "true"
    if args.debug:
        os.environ["INTELCLAW_DEBUG"] = "true"
    if args.config:
        os.environ["INTELCLAW_CONFIG"] = args.config
    
    # Run the application
    asyncio.run(main())


if __name__ == "__main__":
    cli()
