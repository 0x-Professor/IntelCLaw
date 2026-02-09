"""
IntelCLaw - Autonomous AI Agent for Windows

Main entry point for the IntelCLaw application.
Inspired by OpenClaw (https://docs.openclaw.ai)
"""

import asyncio
import os
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
        description="IntelCLaw - Autonomous AI Agent for Windows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  intelclaw onboard              Run the onboarding wizard
  intelclaw --web                Start web interface
  intelclaw --auth               Authenticate with GitHub
  intelclaw models status        Check auth status
  intelclaw models auth login    Login to a provider

Examples:
  intelclaw onboard              # First-time setup
  intelclaw --web --port 8765    # Start web chat
  intelclaw models status        # Check authentication
"""
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # onboard command
    onboard_parser = subparsers.add_parser(
        "onboard",
        help="Run the onboarding wizard"
    )
    onboard_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without prompts"
    )
    onboard_parser.add_argument(
        "--auth-choice",
        choices=["github-copilot", "github-models", "openai", "anthropic", "skip"],
        help="Pre-select authentication method"
    )
    onboard_parser.add_argument(
        "--workspace",
        type=str,
        help="Workspace directory"
    )
    onboard_parser.add_argument(
        "--gateway-port",
        type=int,
        default=8765,
        help="Gateway port"
    )
    onboard_parser.add_argument(
        "--skip-skills",
        action="store_true",
        help="Skip skill/MCP setup steps"
    )
    onboard_parser.add_argument(
        "--install-windows-mcp",
        action="store_true",
        default=None,
        help="Install windows-mcp during onboarding"
    )
    onboard_parser.add_argument(
        "--install-whatsapp-mcp",
        action="store_true",
        default=None,
        help="Install whatsapp-mcp during onboarding"
    )
    
    # models command
    models_parser = subparsers.add_parser(
        "models",
        help="Model and auth management"
    )
    models_subparsers = models_parser.add_subparsers(dest="models_command")
    
    # models status
    models_subparsers.add_parser("status", help="Show auth status")
    
    # models list
    models_subparsers.add_parser("list", help="List available models")
    
    # models auth
    auth_parser = models_subparsers.add_parser("auth", help="Authentication")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command")
    
    # models auth login
    login_parser = auth_subparsers.add_parser("login", help="Login to a provider")
    login_parser.add_argument(
        "--provider",
        choices=["github-copilot", "github-models", "openai", "anthropic"],
        required=True,
        help="Provider to authenticate with"
    )
    
    # models auth logout
    logout_parser = auth_subparsers.add_parser("logout", help="Logout from a provider")
    logout_parser.add_argument(
        "--provider",
        required=True,
        help="Provider to logout from"
    )
    
    # configure command
    subparsers.add_parser("configure", help="Reconfigure settings")
    
    # gateway command (future)
    gateway_parser = subparsers.add_parser("gateway", help="Run gateway server")
    gateway_parser.add_argument("--port", type=int, default=8765)
    gateway_parser.add_argument("--host", default="127.0.0.1")
    
    # Main parser arguments
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
    
    # Handle subcommands
    if args.command == "onboard":
        from intelclaw.cli.onboard import run_onboard
        asyncio.run(run_onboard(
            non_interactive=args.non_interactive,
            auth_choice=args.auth_choice,
            workspace=args.workspace,
            gateway_port=args.gateway_port,
            skip_skills=args.skip_skills,
            install_windows_mcp=args.install_windows_mcp,
            install_whatsapp_mcp=args.install_whatsapp_mcp,
        ))
        return
    
    elif args.command == "models":
        from intelclaw.cli.auth import AuthManager
        auth_manager = AuthManager()
        
        if args.models_command == "status":
            auth_manager.print_status()
            return
        
        elif args.models_command == "list":
            print("\n📋 Available Models")
            print("=" * 50)
            for provider, info in auth_manager.PROVIDERS.items():
                print(f"\n{info['name']} ({provider})")
                for model in info["models"]:
                    print(f"  • {provider}/{model}")
            return
        
        elif args.models_command == "auth":
            if args.auth_command == "login":
                if args.provider == "github-copilot":
                    asyncio.run(auth_manager.login_github_copilot())
                elif args.provider == "github-models":
                    asyncio.run(auth_manager.login_github_models())
                elif args.provider == "openai":
                    api_key = input("Enter your OpenAI API key: ").strip()
                    if api_key:
                        auth_manager.save_api_key("openai", api_key)
                        print("✅ OpenAI API key saved!")
                elif args.provider == "anthropic":
                    api_key = input("Enter your Anthropic API key: ").strip()
                    if api_key:
                        auth_manager.save_api_key("anthropic", api_key)
                        print("✅ Anthropic API key saved!")
                return
            
            elif args.auth_command == "logout":
                profile_id = f"{args.provider}:default"
                if auth_manager.delete_profile(profile_id):
                    print(f"✅ Logged out from {args.provider}")
                else:
                    print(f"❌ No profile found for {args.provider}")
                return
        
        # Default: show status
        auth_manager.print_status()
        return
    
    elif args.command == "configure":
        # Re-run onboard in modify mode
        from intelclaw.cli.onboard import run_onboard
        asyncio.run(run_onboard())
        return
    
    elif args.command == "gateway":
        # Run as gateway (web server only)
        asyncio.run(run_web_server(args.host, args.port))
        return
    
    # Handle legacy auth commands
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
