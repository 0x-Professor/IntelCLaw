"""
Onboarding Wizard - OpenClaw-style setup flow.

Guides users through:
1. Model/auth selection (GitHub Copilot, OpenAI, Anthropic, GitHub Models)
2. Workspace configuration
3. Gateway settings
4. Health check
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from .auth import AuthManager


# Default directories
STATE_DIR = Path.home() / ".intelclaw"
CONFIG_FILE = STATE_DIR / "intelclaw.json"
WORKSPACE_DIR = STATE_DIR / "workspace"

# Project .env file location
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def save_to_env(key: str, value: str, env_file: Path = ENV_FILE) -> bool:
    """
    Save a key-value pair to the .env file.
    
    If the key exists, update it. If not, append it.
    
    Args:
        key: Environment variable name
        value: Value to set
        env_file: Path to .env file
        
    Returns:
        True if successful
    """
    try:
        content = ""
        key_pattern = re.compile(rf'^{re.escape(key)}=.*$', re.MULTILINE)
        
        if env_file.exists():
            content = env_file.read_text(encoding="utf-8")
        
        new_line = f"{key}={value}"
        
        if key_pattern.search(content):
            # Update existing key
            content = key_pattern.sub(new_line, content)
            logger.debug(f"Updated {key} in .env")
        else:
            # Append new key
            if content and not content.endswith("\n"):
                content += "\n"
            content += f"\n{new_line}\n"
            logger.debug(f"Added {key} to .env")
        
        env_file.write_text(content, encoding="utf-8")
        
        # Also set in current environment
        os.environ[key] = value
        
        return True
    except Exception as e:
        logger.error(f"Failed to save {key} to .env: {e}")
        return False


def load_from_env(key: str, env_file: Path = ENV_FILE) -> Optional[str]:
    """
    Load a value from the .env file.
    
    Args:
        key: Environment variable name
        env_file: Path to .env file
        
    Returns:
        Value if found, None otherwise
    """
    # First check current environment
    if key in os.environ:
        return os.environ[key]
    
    # Then check .env file
    if env_file.exists():
        try:
            content = env_file.read_text(encoding="utf-8")
            pattern = re.compile(rf'^{re.escape(key)}=(.*)$', re.MULTILINE)
            match = pattern.search(content)
            if match:
                return match.group(1).strip()
        except Exception as e:
            logger.debug(f"Failed to read {key} from .env: {e}")
    
    return None


def print_banner():
    """Print the IntelCLaw banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🦞 IntelCLaw - Autonomous AI Agent                         ║
║                                                              ║
║   Onboarding Wizard                                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_step(step: int, total: int, title: str):
    """Print a step header."""
    print(f"\n{'━' * 60}")
    print(f"  Step {step}/{total}: {title}")
    print(f"{'━' * 60}\n")


def prompt_choice(prompt: str, choices: list, default: Optional[int] = None) -> int:
    """Prompt user to select from a list of choices."""
    print(prompt)
    for i, choice in enumerate(choices, 1):
        print(f"  [{i}] {choice}")
    
    default_str = f" (default: {default})" if default else ""
    while True:
        try:
            response = input(f"\nEnter choice{default_str}: ").strip()
            if not response and default:
                return default
            idx = int(response)
            if 1 <= idx <= len(choices):
                return idx
        except ValueError:
            pass
        print("Invalid choice, please try again.")


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt for yes/no answer."""
    default_str = "Y/n" if default else "y/N"
    while True:
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not response:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'")


def prompt_string(prompt: str, default: Optional[str] = None, required: bool = True) -> str:
    """Prompt for a string value."""
    default_str = f" (default: {default})" if default else ""
    while True:
        response = input(f"{prompt}{default_str}: ").strip()
        if not response and default:
            return default
        if response or not required:
            return response
        print("This field is required.")


def load_config() -> Dict[str, Any]:
    """Load existing configuration."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    return {}


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


async def run_onboard(
    non_interactive: bool = False,
    auth_choice: Optional[str] = None,
    workspace: Optional[str] = None,
    gateway_port: int = 8765,
    skip_auth: bool = False,
    skip_skills: bool = False,
    install_windows_mcp: Optional[bool] = None,
    install_whatsapp_mcp: Optional[bool] = None,
) -> bool:
    """
    Run the onboarding wizard.
    
    Args:
        non_interactive: Skip prompts and use defaults/provided values
        auth_choice: Pre-selected auth choice
        workspace: Pre-selected workspace path
        gateway_port: Gateway port
        skip_auth: Skip authentication step
        skip_skills: Skip skill/MCP setup steps
        install_windows_mcp: Force install windows-mcp when set (defaults to enabled in non-interactive)
        install_whatsapp_mcp: Force install whatsapp-mcp when set (defaults to disabled in non-interactive)
    
    Returns:
        True if onboarding completed successfully
    """
    if not non_interactive:
        print_banner()
    
    config = load_config()
    auth_manager = AuthManager()
    
    # Step 1: Check existing config
    if CONFIG_FILE.exists() and not non_interactive:
        print_step(0, 6, "Existing Configuration")
        print("An existing configuration was found.")
        choice = prompt_choice(
            "What would you like to do?",
            ["Keep existing and modify", "Start fresh (reset)", "Exit"],
            default=1
        )
        if choice == 3:
            print("\n👋 Goodbye!")
            return False
        if choice == 2:
            config = {}
    
    # Step 2: Model/Auth
    if not skip_auth:
        if not non_interactive:
            print_step(1, 6, "Model & Authentication")
            
            print("Select your LLM provider:")
            print()
            
            auth_choices = [
                "GitHub Copilot (requires subscription) - Use your existing Copilot access",
                "GitHub Models API (FREE) - Works with any GitHub account",
                "OpenAI API Key - Requires OPENAI_API_KEY",
                "Anthropic API Key - Requires ANTHROPIC_API_KEY",
                "Skip authentication for now",
            ]
            
            choice = prompt_choice("", auth_choices, default=2)
            
            if choice == 1:
                # GitHub Copilot - Device OAuth flow
                print("\n🔐 Authenticating with GitHub Copilot...")
                print("   This will open a browser for GitHub device authorization.")
                print()
                
                profile = await auth_manager.login_github_copilot()
                if profile:
                    # Save the GitHub token to .env for reuse
                    if profile.extra and profile.extra.get("github_token"):
                        github_token = profile.extra["github_token"]
                        save_to_env("COPILOT_GITHUB_TOKEN", github_token)
                        print("\n✅ GitHub token saved to .env file")
                    
                    # Save Copilot base URL if available
                    if profile.extra and profile.extra.get("copilot_base_url"):
                        save_to_env("COPILOT_API_BASE_URL", profile.extra["copilot_base_url"])
                    
                    config.setdefault("agents", {}).setdefault("defaults", {})["model"] = {
                        "primary": "github-copilot/gpt-4o"
                    }
                    
                    # Set default model in .env
                    save_to_env("INTELCLAW_DEFAULT_MODEL", "gpt-4o")
                    save_to_env("INTELCLAW_PROVIDER", "github-copilot")
            
            elif choice == 2:
                # GitHub Models (FREE) - Token-based
                print("\n🔐 Authenticating with GitHub Models API...")
                
                # Check if we already have a token
                existing_token = load_from_env("GITHUB_TOKEN")
                if existing_token:
                    print(f"   Found existing GITHUB_TOKEN in .env")
                    use_existing = prompt_yes_no("Use existing token?", default=True)
                    if use_existing:
                        print("\n✅ Using existing GitHub token from .env")
                        config.setdefault("agents", {}).setdefault("defaults", {})["model"] = {
                            "primary": "github-models/gpt-4o-mini"
                        }
                        save_to_env("INTELCLAW_DEFAULT_MODEL", "gpt-4o-mini")
                        save_to_env("INTELCLAW_PROVIDER", "github-models")
                    else:
                        profile = await auth_manager.login_github_models()
                        if profile and profile.access_token:
                            save_to_env("GITHUB_TOKEN", profile.access_token)
                            print("\n✅ GitHub token saved to .env file")
                            config.setdefault("agents", {}).setdefault("defaults", {})["model"] = {
                                "primary": "github-models/gpt-4o-mini"
                            }
                            save_to_env("INTELCLAW_DEFAULT_MODEL", "gpt-4o-mini")
                            save_to_env("INTELCLAW_PROVIDER", "github-models")
                else:
                    profile = await auth_manager.login_github_models()
                    if profile and profile.access_token:
                        save_to_env("GITHUB_TOKEN", profile.access_token)
                        print("\n✅ GitHub token saved to .env file")
                        config.setdefault("agents", {}).setdefault("defaults", {})["model"] = {
                            "primary": "github-models/gpt-4o-mini"
                        }
                        save_to_env("INTELCLAW_DEFAULT_MODEL", "gpt-4o-mini")
                        save_to_env("INTELCLAW_PROVIDER", "github-models")
            
            elif choice == 3:
                # OpenAI
                print("\n🔑 OpenAI API Key")
                api_key = load_from_env("OPENAI_API_KEY")
                if api_key:
                    print(f"   Found OPENAI_API_KEY in .env")
                    use_env = prompt_yes_no("Use this key?", default=True)
                    if not use_env:
                        api_key = prompt_string("Enter your OpenAI API key", required=True)
                else:
                    api_key = prompt_string("Enter your OpenAI API key", required=True)
                
                if api_key:
                    save_to_env("OPENAI_API_KEY", api_key)
                    auth_manager.save_api_key("openai", api_key)
                    config.setdefault("agents", {}).setdefault("defaults", {})["model"] = {
                        "primary": "openai/gpt-4o"
                    }
                    save_to_env("INTELCLAW_DEFAULT_MODEL", "gpt-4o")
                    save_to_env("INTELCLAW_PROVIDER", "openai")
                    print("\n✅ OpenAI API key saved to .env file!")
            
            elif choice == 4:
                # Anthropic
                print("\n🔑 Anthropic API Key")
                api_key = load_from_env("ANTHROPIC_API_KEY")
                if api_key:
                    print(f"   Found ANTHROPIC_API_KEY in .env")
                    use_env = prompt_yes_no("Use this key?", default=True)
                    if not use_env:
                        api_key = prompt_string("Enter your Anthropic API key", required=True)
                else:
                    api_key = prompt_string("Enter your Anthropic API key", required=True)
                
                if api_key:
                    save_to_env("ANTHROPIC_API_KEY", api_key)
                    auth_manager.save_api_key("anthropic", api_key)
                    config.setdefault("agents", {}).setdefault("defaults", {})["model"] = {
                        "primary": "anthropic/claude-3.5-sonnet"
                    }
                    save_to_env("INTELCLAW_DEFAULT_MODEL", "claude-3.5-sonnet")
                    save_to_env("INTELCLAW_PROVIDER", "anthropic")
                    print("\n✅ Anthropic API key saved to .env file!")
            
            else:
                print("\n⏭️  Skipping authentication. You can set it up later with:")
                print("   intelclaw models auth login --provider <provider>")
        
        else:
            # Non-interactive auth
            if auth_choice == "github-copilot":
                profile = await auth_manager.login_github_copilot()
                if profile:
                    if profile.extra and profile.extra.get("github_token"):
                        save_to_env("COPILOT_GITHUB_TOKEN", profile.extra["github_token"])
                    config.setdefault("agents", {}).setdefault("defaults", {})["model"] = {
                        "primary": "github-copilot/gpt-4o"
                    }
            elif auth_choice == "github-models":
                profile = await auth_manager.login_github_models()
                if profile and profile.access_token:
                    save_to_env("GITHUB_TOKEN", profile.access_token)
                    config.setdefault("agents", {}).setdefault("defaults", {})["model"] = {
                        "primary": "github-models/gpt-4o-mini"
                    }
    
    # Step 3: Workspace
    if not non_interactive:
        print_step(2, 6, "Workspace")
        
        default_workspace = str(WORKSPACE_DIR)
        print(f"The workspace is where IntelCLaw stores files and memories.")
        workspace_path = prompt_string(
            "Workspace directory",
            default=default_workspace
        )
    else:
        workspace_path = workspace or str(WORKSPACE_DIR)
    
    workspace_dir = Path(workspace_path).expanduser()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    config.setdefault("agents", {}).setdefault("defaults", {})["workspace"] = str(workspace_dir)
    
    # Create bootstrap files
    await _create_bootstrap_files(workspace_dir)
    
    if not non_interactive:
        print(f"\n✅ Workspace configured at: {workspace_dir}")
    
    # Step 4: Gateway Settings
    if not non_interactive:
        print_step(3, 6, "Gateway Settings")
        
        print("The gateway provides the web interface and API.")
        
        port = prompt_string(
            "Gateway port",
            default=str(gateway_port)
        )
        try:
            port = int(port)
        except:
            port = gateway_port
        
        bind = prompt_choice(
            "Bind address:",
            [
                "Loopback only (127.0.0.1) - Most secure",
                "All interfaces (0.0.0.0) - For network access",
            ],
            default=1
        )
        
        host = "127.0.0.1" if bind == 1 else "0.0.0.0"
    else:
        port = gateway_port
        host = "127.0.0.1"
    
    config.setdefault("gateway", {}).update({
        "port": port,
        "host": host,
    })
    
    if not non_interactive:
        print(f"\n✅ Gateway will run at http://{host}:{port}")
    
    # Step 5: Save Config
    if not non_interactive:
        print_step(4, 6, "Saving Configuration")
    
    config["wizard"] = {
        "lastRunAt": __import__("datetime").datetime.now().isoformat(),
        "lastRunVersion": "0.1.0",
    }
    
    save_config(config)
    
    if not non_interactive:
        print(f"✅ Configuration saved to: {CONFIG_FILE}")
    
    # Step 6: Skills & MCP Servers
    if not skip_skills:
        if not non_interactive:
            print_step(5, 6, "Skills & MCP Servers")
        await _setup_skills_and_mcp(
            non_interactive=non_interactive,
            force_install_windows_mcp=install_windows_mcp,
            force_install_whatsapp_mcp=install_whatsapp_mcp,
        )

    # Step 7: Health Check
    if not non_interactive:
        print_step(6, 6, "Health Check")
        
        # Check auth status
        auth_manager.print_status()
        
        print("\n" + "=" * 60)
        print("\n🎉 Onboarding complete!")
        print("\nNext steps:")
        print(f"  1. Start the gateway: intelclaw gateway --port {port}")
        print(f"  2. Or use web mode: intelclaw --web --port {port}")
        print(f"  3. Open dashboard: http://{host}:{port}")
        print("\nUseful commands:")
        print("  intelclaw models status      - Check auth status")
        print("  intelclaw models list        - List available models")
        print("  intelclaw configure          - Reconfigure settings")
        print()
    
    return True


def _run_cmd(cmd: list[str], *, cwd: Optional[Path] = None) -> int:
    try:
        res = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
        return int(res.returncode)
    except FileNotFoundError:
        return 127
    except Exception:
        return 1


def _print_missing_exe(name: str, *, hint: Optional[str] = None) -> None:
    msg = f"âš ï¸  Missing dependency: '{name}' is not on PATH."
    if hint:
        msg += f"\n   {hint}"
    print(msg)


def _install_windows_mcp() -> bool:
    print("\nðŸªŸ Installing Windows-MCP (windows-mcp)...")
    rc = _run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "windows-mcp"])
    if rc != 0:
        print("âš ï¸  Failed to install windows-mcp via pip.")
        return False
    rc2 = _run_cmd(["windows-mcp", "--help"])
    if rc2 != 0:
        print("âš ï¸  windows-mcp installed but could not be executed from PATH.")
        return False
    print("âœ… Windows-MCP installed.")
    return True


def _ensure_uv_available() -> bool:
    if shutil.which("uv"):
        return True
    print("\nðŸ”§ Installing uv...")
    rc = _run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "uv"])
    return rc == 0 and bool(shutil.which("uv"))


def _install_whatsapp_mcp(*, launch_bridge: bool) -> bool:
    print("\nðŸ’¬ Installing WhatsApp-MCP (lharries/whatsapp-mcp)...")
    if not shutil.which("git"):
        _print_missing_exe("git", hint="Install Git for Windows, then re-run onboarding.")
        return False

    vendor_dir = PROJECT_ROOT / "data" / "vendor" / "whatsapp-mcp"
    vendor_dir.parent.mkdir(parents=True, exist_ok=True)

    if (vendor_dir / ".git").exists():
        print(f"ðŸ”„ Updating repo: {vendor_dir}")
        _run_cmd(["git", "-C", str(vendor_dir), "pull", "--ff-only"])
    else:
        print(f"ðŸ“¦ Cloning repo to: {vendor_dir}")
        rc = _run_cmd(["git", "clone", "https://github.com/lharries/whatsapp-mcp", str(vendor_dir)])
        if rc != 0:
            print("âš ï¸  Failed to clone whatsapp-mcp repo.")
            return False

    if not _ensure_uv_available():
        _print_missing_exe("uv", hint="Install uv, then re-run onboarding.")
        return False

    server_dir = vendor_dir / "whatsapp-mcp-server"
    if server_dir.exists():
        print("ðŸ§ª Warming up Python dependencies (uv sync)...")
        _run_cmd(["uv", "--directory", str(server_dir), "sync"])
    else:
        print("âš ï¸  Expected server directory not found:", server_dir)

    if launch_bridge:
        bridge_dir = vendor_dir / "whatsapp-bridge"
        if not bridge_dir.exists():
            print("âš ï¸  Expected bridge directory not found:", bridge_dir)
        elif not shutil.which("go"):
            _print_missing_exe(
                "go",
                hint=(
                    "Install Go (and a C compiler like gcc for go-sqlite3), then run the bridge:\n"
                    f"   powershell -NoProfile -ExecutionPolicy Bypass -File \"{(PROJECT_ROOT / 'scripts' / 'run_whatsapp_bridge.ps1')}\" -RepoRoot \"{PROJECT_ROOT}\"\n"
                    "   (or manually)\n"
                    f"   cd \"{bridge_dir}\"\n"
                    "   go env -w CGO_ENABLED=1\n"
                    "   go mod download\n"
                    "   go run ."
                ),
            )
        else:
            print("ðŸš€ Launching WhatsApp bridge in a new console (scan QR code there)...")
            creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            ps1 = PROJECT_ROOT / "scripts" / "run_whatsapp_bridge.ps1"
            if ps1.exists():
                subprocess.Popen(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(ps1),
                        "-RepoRoot",
                        str(PROJECT_ROOT),
                    ],
                    cwd=str(PROJECT_ROOT),
                    creationflags=creationflags,
                )
            else:
                subprocess.Popen(
                    ["cmd.exe", "/k", "go env -w CGO_ENABLED=1 && go mod download && go run ."],
                    cwd=str(bridge_dir),
                    creationflags=creationflags,
                )

    print("âœ… WhatsApp-MCP setup complete (repo present).")
    return True


async def _mcp_smoke_check_windows(*, timeout_seconds: float = 10.0) -> None:
    try:
        from intelclaw.mcp.connection import MCPServerConnection, MCPServerSpec
    except Exception as e:
        print(f"âš ï¸  MCP smoke check skipped (mcp client unavailable): {e}")
        return

    spec = MCPServerSpec(
        skill_id="windows",
        server_id="windows_mcp",
        transport="stdio",
        command="windows-mcp",
        args=["--transport", "stdio"],
        env={"ANONYMIZED_TELEMETRY": "false"},
        cwd=None,
        tool_namespace="windows",
        tool_allowlist=[],
        tool_denylist=[],
    )
    conn = MCPServerConnection(spec)
    try:
        tools = await asyncio.wait_for(conn.list_tools(), timeout=float(timeout_seconds))
        print(f"âœ… Windows MCP smoke check: {len(list(tools or []))} tools detected.")
    except Exception as e:
        print(f"âš ï¸  Windows MCP smoke check failed: {e}")
    finally:
        try:
            await asyncio.wait_for(conn.shutdown(), timeout=3.0)
        except Exception:
            pass


async def _mcp_smoke_check_whatsapp(*, timeout_seconds: float = 10.0) -> None:
    vendor_dir = PROJECT_ROOT / "data" / "vendor" / "whatsapp-mcp"
    server_dir = vendor_dir / "whatsapp-mcp-server"
    if not server_dir.exists():
        return
    if not shutil.which("uv"):
        return
    try:
        from intelclaw.mcp.connection import MCPServerConnection, MCPServerSpec
    except Exception:
        return

    spec = MCPServerSpec(
        skill_id="whatsapp",
        server_id="whatsapp_mcp",
        transport="stdio",
        command="uv",
        args=["run", "main.py"],
        env={},
        cwd=server_dir,
        tool_namespace="whatsapp",
        tool_allowlist=[],
        tool_denylist=[],
    )
    conn = MCPServerConnection(spec)
    try:
        tools = await asyncio.wait_for(conn.list_tools(), timeout=float(timeout_seconds))
        print(f"âœ… WhatsApp MCP smoke check: {len(list(tools or []))} tools detected.")
    except Exception as e:
        print(f"âš ï¸  WhatsApp MCP smoke check failed: {e}")
    finally:
        try:
            await asyncio.wait_for(conn.shutdown(), timeout=3.0)
        except Exception:
            pass


async def _setup_skills_and_mcp(
    *,
    non_interactive: bool,
    force_install_windows_mcp: Optional[bool],
    force_install_whatsapp_mcp: Optional[bool],
) -> None:
    # Decide actions
    if non_interactive:
        do_windows = True if force_install_windows_mcp is None else bool(force_install_windows_mcp)
        do_whatsapp = bool(force_install_whatsapp_mcp) if force_install_whatsapp_mcp is not None else False
        launch_bridge = False
    else:
        do_windows = True if force_install_windows_mcp else prompt_yes_no("Install Windows-MCP (windows-mcp) now?", default=True)
        do_whatsapp = True if force_install_whatsapp_mcp else prompt_yes_no(
            "Install WhatsApp-MCP (lharries/whatsapp-mcp) now?",
            default=True,
        )
        launch_bridge = False
        if do_whatsapp:
            launch_bridge = prompt_yes_no(
                "Launch WhatsApp bridge in a new console now (QR login)?",
                default=True,
            )

    # Execute (best-effort)
    try:
        if do_windows:
            _install_windows_mcp()
    except Exception as e:
        print(f"âš ï¸  Windows-MCP install failed: {e}")

    try:
        if do_whatsapp:
            _install_whatsapp_mcp(launch_bridge=launch_bridge)
    except Exception as e:
        print(f"âš ï¸  WhatsApp-MCP install failed: {e}")

    # Best-effort MCP smoke checks (non-fatal)
    try:
        if do_windows:
            await _mcp_smoke_check_windows()
    except Exception:
        pass
    try:
        if do_whatsapp:
            await _mcp_smoke_check_whatsapp()
    except Exception:
        pass


async def _create_bootstrap_files(workspace_dir: Path) -> None:
    """Create bootstrap files in the workspace (OpenClaw-style)."""
    
    # AGENTS.md - Agent configuration
    agents_file = workspace_dir / "AGENTS.md"
    if not agents_file.exists():
        agents_file.write_text("""# Agent Configuration

This file configures the IntelCLaw agent behavior.

## Capabilities

- Autonomous task execution
- Code generation and editing
- File management
- Web search and research
- Memory and context management

## Guidelines

1. Always confirm destructive actions
2. Prefer safe operations
3. Keep the user informed of progress
""", encoding="utf-8")
    
    # SOUL.md - Agent personality
    soul_file = workspace_dir / "SOUL.md"
    if not soul_file.exists():
        soul_file.write_text("""# Soul - Agent Personality

You are IntelCLaw, an autonomous AI assistant running on Windows.

## Core Traits

- Helpful and proactive
- Clear and concise communication
- Safety-conscious
- Efficient problem solver

## Communication Style

- Be direct but friendly
- Explain your reasoning
- Ask for clarification when needed
- Celebrate successes with the user
""", encoding="utf-8")
    
    # USER.md - User preferences
    user_file = workspace_dir / "USER.md"
    if not user_file.exists():
        user_file.write_text("""# User Preferences

Add your preferences here and IntelCLaw will remember them.

## Examples

- Preferred coding style
- Common tasks you perform
- Projects you're working on
- Tools you prefer
""", encoding="utf-8")
    
    # Create memory directory
    memory_dir = workspace_dir / "memory"
    memory_dir.mkdir(exist_ok=True)
    
    # Create sessions directory
    sessions_dir = workspace_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)


def cli_onboard():
    """CLI entry point for onboard command."""
    import argparse
    
    parser = argparse.ArgumentParser(description="IntelCLaw Onboarding Wizard")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without prompts"
    )
    parser.add_argument(
        "--auth-choice",
        choices=["github-copilot", "github-models", "openai", "anthropic", "skip"],
        help="Pre-select authentication method"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="Workspace directory"
    )
    parser.add_argument(
        "--gateway-port",
        type=int,
        default=8765,
        help="Gateway port"
    )
    parser.add_argument(
        "--skip-auth",
        action="store_true",
        help="Skip authentication step"
    )
    parser.add_argument(
        "--skip-skills",
        action="store_true",
        help="Skip skill/MCP setup steps"
    )
    parser.add_argument(
        "--install-windows-mcp",
        action="store_true",
        default=None,
        help="Install windows-mcp during onboarding"
    )
    parser.add_argument(
        "--install-whatsapp-mcp",
        action="store_true",
        default=None,
        help="Install whatsapp-mcp during onboarding"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_onboard(
        non_interactive=args.non_interactive,
        auth_choice=args.auth_choice,
        workspace=args.workspace,
        gateway_port=args.gateway_port,
        skip_auth=args.skip_auth,
        skip_skills=args.skip_skills,
        install_windows_mcp=args.install_windows_mcp,
        install_whatsapp_mcp=args.install_whatsapp_mcp,
    ))


if __name__ == "__main__":
    cli_onboard()
