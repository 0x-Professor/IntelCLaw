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
        
        # Run the main event loop
        await app.run()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested via keyboard")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


def run_with_qt():
    """Run with Qt event loop integration using qasync."""
    try:
        from PyQt6.QtWidgets import QApplication
        import qasync
        
        # Create Qt application
        qt_app = QApplication(sys.argv)
        
        # Create qasync event loop
        loop = qasync.QEventLoop(qt_app)
        asyncio.set_event_loop(loop)
        
        # Run the async main
        with loop:
            loop.run_until_complete(main())
            
    except ImportError:
        # Fallback to regular asyncio
        asyncio.run(main())


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
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Authenticate with GitHub Copilot"
    )
    parser.add_argument(
        "--logout",
        action="store_true",
        help="Clear saved GitHub authentication"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start web interface instead of desktop overlay"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for web server (default: 8765)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for web server (default: 127.0.0.1)"
    )
    
    args = parser.parse_args()
    
    # Handle auth commands
    if args.auth:
        from intelclaw.integrations.llm_provider import GitHubAuth
        asyncio.run(GitHubAuth.authenticate())
        return
    
    if args.logout:
        from intelclaw.integrations.llm_provider import GitHubAuth
        GitHubAuth.clear_token()
        return
    
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
    
    # Run web interface or desktop app
    if args.web:
        asyncio.run(run_web_server(args.host, args.port))
    else:
        run_with_qt()


async def run_web_server(host: str, port: int):
    """Run the web server interface."""
    setup_logging()
    
    logger.info("=" * 50)
    logger.info("IntelCLaw - Web Interface")
    logger.info("=" * 50)
    
    try:
        from intelclaw.core.app import IntelCLawApp
        from intelclaw.web.server import WebServer
        import webbrowser
        
        # Initialize the app (without Qt UI)
        app = IntelCLawApp()
        
        # Skip Qt UI components for web mode
        os.environ["INTELCLAW_NO_QT"] = "true"
        
        await app.startup()
        
        # Create and start web server
        web_server = WebServer(app=app, host=host, port=port)
        
        # Open browser
        url = f"http://{host}:{port}"
        logger.info(f"Opening browser at {url}")
        webbrowser.open(url)
        
        # Run web server
        await web_server.start()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
