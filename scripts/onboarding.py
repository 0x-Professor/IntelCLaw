"""
IntelCLaw Onboarding Script

Interactive setup wizard for first-time configuration.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich for better UI...")
    os.system("uv pip install rich")
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.markdown import Markdown

console = Console()


def print_banner():
    """Print welcome banner."""
    banner = """
    ██╗███╗   ██╗████████╗███████╗██╗      ██████╗██╗      █████╗ ██╗    ██╗
    ██║████╗  ██║╚══██╔══╝██╔════╝██║     ██╔════╝██║     ██╔══██╗██║    ██║
    ██║██╔██╗ ██║   ██║   █████╗  ██║     ██║     ██║     ███████║██║ █╗ ██║
    ██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██║     ██║     ██╔══██║██║███╗██║
    ██║██║ ╚████║   ██║   ███████╗███████╗╚██████╗███████╗██║  ██║╚███╔███╔╝
    ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚══════╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝
    
                    🦅 Autonomous AI Agent for Windows 🦅
    """
    console.print(Panel(banner, style="bold blue"))


def setup_api_keys() -> dict:
    """Configure API keys."""
    console.print("\n[bold cyan]Step 1: LLM Configuration[/bold cyan]\n")
    
    keys = {}
    
    console.print("""
[bold]IntelCLaw can use your LLM in two ways:[/bold]

[green]1. GitHub Copilot (Recommended)[/green]
   - Uses your existing Copilot subscription
   - No additional API keys needed
   - Automatically detected from VS Code

[yellow]2. Direct API Keys[/yellow]
   - Use OpenAI, Anthropic, etc. directly
   - Requires separate API keys
    """)
    
    use_copilot = Confirm.ask("Use GitHub Copilot as primary LLM?", default=True)
    
    if use_copilot:
        console.print("\n[green]✓ Will use GitHub Copilot[/green]")
        console.print("  Make sure you're logged into GitHub Copilot in VS Code")
        keys["USE_COPILOT"] = "true"
        
        # Optional: GitHub token for enhanced features
        console.print("\n[yellow]GitHub Token[/yellow] (Optional - for repository access)")
        if Confirm.ask("Add GitHub token for repo access?", default=False):
            github_token = Prompt.ask("Enter GitHub Personal Access Token", password=True)
            if github_token:
                keys["GITHUB_TOKEN"] = github_token
    else:
        # OpenAI (Required if not using Copilot)
        console.print("\n[yellow]OpenAI API Key[/yellow] (Required)")
        console.print("Get your key at: https://platform.openai.com/api-keys")
        openai_key = Prompt.ask("Enter OpenAI API Key", password=True)
        if openai_key:
            keys["OPENAI_API_KEY"] = openai_key
        
        # Anthropic (Optional)
        console.print("\n[yellow]Anthropic API Key[/yellow] (Optional fallback)")
        if Confirm.ask("Configure Anthropic?", default=False):
            anthropic_key = Prompt.ask("Enter Anthropic API Key", password=True)
            if anthropic_key:
                keys["ANTHROPIC_API_KEY"] = anthropic_key
    
    # Tavily (Optional - for web search)
    console.print("\n[yellow]Tavily API Key[/yellow] (Optional - for web search)")
    console.print("Get your free key at: https://tavily.com")
    if Confirm.ask("Configure Tavily for web search?", default=False):
        tavily_key = Prompt.ask("Enter Tavily API Key", password=True)
        if tavily_key:
            keys["TAVILY_API_KEY"] = tavily_key
    
    return keys


def setup_github_copilot() -> dict:
    """Configure GitHub Copilot integration."""
    console.print("\n[bold cyan]Step 2: GitHub Copilot Integration[/bold cyan]\n")
    
    copilot_config = {
        "enabled": True,
        "auto_suggestions": True,
        "model": "gpt-4o",
        "context_length": 8000,
    }
    
    console.print("""
[bold]GitHub Copilot Integration Options:[/bold]

IntelCLaw can integrate with GitHub Copilot in several ways:
1. Use Copilot's suggestions for code completion
2. Leverage Copilot Chat for coding assistance
3. Sync agent skills with Copilot context
    """)
    
    if Confirm.ask("Enable GitHub Copilot integration?", default=True):
        copilot_config["enabled"] = True
        
        # Model selection
        console.print("\n[yellow]Select preferred model:[/yellow]")
        models = ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "o1-preview", "o1-mini"]
        for i, model in enumerate(models, 1):
            console.print(f"  {i}. {model}")
        
        model_choice = Prompt.ask("Choose model", choices=[str(i) for i in range(1, len(models)+1)], default="1")
        copilot_config["model"] = models[int(model_choice) - 1]
        
        # Auto-suggestions
        copilot_config["auto_suggestions"] = Confirm.ask("Enable auto-suggestions?", default=True)
        
        # Context length
        context = Prompt.ask("Context window size", default="8000")
        copilot_config["context_length"] = int(context)
    
    return copilot_config


def setup_user_profile() -> dict:
    """Collect user information for personalization."""
    console.print("\n[bold cyan]Step 3: User Profile Setup[/bold cyan]\n")
    
    console.print("""
[bold]Personal Information[/bold]
This helps IntelCLaw personalize your experience and remember important details.
All data is stored locally on your machine.
    """)
    
    profile = {}
    
    # Basic info
    profile["name"] = Prompt.ask("Your name")
    profile["email"] = Prompt.ask("Email address", default="")
    profile["occupation"] = Prompt.ask("Occupation/Role", default="Developer")
    profile["timezone"] = Prompt.ask("Timezone", default="UTC")
    
    # Preferences
    console.print("\n[yellow]Work Preferences:[/yellow]")
    profile["preferred_language"] = Prompt.ask("Primary programming language", default="Python")
    profile["work_hours"] = Prompt.ask("Typical work hours (e.g., 9-17)", default="9-17")
    
    # Contacts
    if Confirm.ask("\nAdd emergency/important contacts?", default=False):
        profile["contacts"] = []
        while True:
            contact = {
                "name": Prompt.ask("Contact name"),
                "email": Prompt.ask("Contact email", default=""),
                "phone": Prompt.ask("Contact phone", default=""),
                "relationship": Prompt.ask("Relationship (e.g., colleague, manager)", default=""),
            }
            profile["contacts"].append(contact)
            if not Confirm.ask("Add another contact?", default=False):
                break
    
    return profile


def setup_autonomy_level() -> dict:
    """Configure agent autonomy settings."""
    console.print("\n[bold cyan]Step 4: Autonomy Configuration[/bold cyan]\n")
    
    console.print("""
[bold]Autonomy Levels:[/bold]

IntelCLaw can operate at different autonomy levels:

[green]1. Full Autonomy[/green] - Agent can modify configs, create skills, fix issues automatically
[yellow]2. Supervised[/yellow] - Agent asks before major changes
[red]3. Restricted[/red] - Agent only executes explicit commands

[bold]Note:[/bold] Destructive operations (delete, format, etc.) always require confirmation.
    """)
    
    level = Prompt.ask("Select autonomy level", choices=["1", "2", "3"], default="1")
    
    autonomy = {
        "level": ["full", "supervised", "restricted"][int(level) - 1],
        "can_modify_config": level == "1",
        "can_create_skills": level in ["1", "2"],
        "can_fix_issues": level == "1",
        "can_change_models": level == "1",
        "can_install_packages": level in ["1", "2"],
        "can_access_network": True,
        "can_store_personal_data": True,
        "require_confirmation_destructive": True,  # Always true for safety
    }
    
    if level == "1":
        console.print("\n[green]✓ Full autonomy enabled[/green]")
        console.print("  - Agent can modify its own configuration")
        console.print("  - Agent can create and improve skills")
        console.print("  - Agent can diagnose and fix issues")
        console.print("  - Agent can change AI models as needed")
    
    return autonomy


def setup_data_storage() -> dict:
    """Configure data storage preferences."""
    console.print("\n[bold cyan]Step 5: Data Storage Settings[/bold cyan]\n")
    
    storage = {
        "store_conversations": True,
        "store_contacts": True,
        "store_files": True,
        "store_browsing_history": True,
        "store_app_usage": True,
        "store_code_snippets": True,
        "store_credentials": True,  # Encrypted
        "retention_days": 365,
    }
    
    console.print("""
[bold]Data Storage Options:[/bold]

IntelCLaw stores data locally to provide intelligent assistance.
Configure what data should be remembered:
    """)
    
    storage["store_conversations"] = Confirm.ask("Store conversation history?", default=True)
    storage["store_contacts"] = Confirm.ask("Store contact information?", default=True)
    storage["store_code_snippets"] = Confirm.ask("Store code snippets?", default=True)
    storage["store_app_usage"] = Confirm.ask("Track application usage?", default=True)
    storage["store_browsing_history"] = Confirm.ask("Store relevant browsing context?", default=True)
    
    retention = Prompt.ask("Data retention period (days, 0=forever)", default="365")
    storage["retention_days"] = int(retention)
    
    return storage


def generate_config(
    api_keys: dict,
    copilot: dict,
    profile: dict,
    autonomy: dict,
    storage: dict
) -> str:
    """Generate the configuration file."""
    
    config = f"""# IntelCLaw Configuration
# Generated by onboarding wizard

app:
  name: IntelCLaw
  version: 0.1.0
  debug: false

# User Profile
user:
  name: "{profile.get('name', '')}"
  email: "{profile.get('email', '')}"
  occupation: "{profile.get('occupation', 'Developer')}"
  timezone: "{profile.get('timezone', 'UTC')}"
  preferred_language: "{profile.get('preferred_language', 'Python')}"
  work_hours: "{profile.get('work_hours', '9-17')}"

# Hotkeys
hotkeys:
  summon: ctrl+shift+space
  quick_action: ctrl+shift+q
  dismiss: escape

# UI Settings
ui:
  theme: dark
  transparency: 0.95
  position: center
  width: 600
  height: 500

# AI Models
models:
  primary: {copilot.get('model', 'gpt-4o')}
  fallback: gpt-4o-mini
  coding: {copilot.get('model', 'gpt-4o')}
  temperature: 0.1
  context_length: {copilot.get('context_length', 8000)}

# GitHub Copilot Integration
copilot:
  enabled: {str(copilot.get('enabled', True)).lower()}
  auto_suggestions: {str(copilot.get('auto_suggestions', True)).lower()}
  model: {copilot.get('model', 'gpt-4o')}
  sync_context: true

# Autonomy Settings
autonomy:
  level: {autonomy.get('level', 'full')}
  can_modify_config: {str(autonomy.get('can_modify_config', True)).lower()}
  can_create_skills: {str(autonomy.get('can_create_skills', True)).lower()}
  can_fix_issues: {str(autonomy.get('can_fix_issues', True)).lower()}
  can_change_models: {str(autonomy.get('can_change_models', True)).lower()}
  can_install_packages: {str(autonomy.get('can_install_packages', True)).lower()}
  can_access_network: true
  can_store_personal_data: true
  require_confirmation_destructive: true

# Memory Settings
memory:
  max_conversation_history: 100
  working_db_path: data/working_memory.db
  vector_store:
    collection: intelclaw
    path: data/vector_db
  retention_days: {storage.get('retention_days', 365)}
  auto_cleanup: true

# Data Storage
storage:
  store_conversations: {str(storage.get('store_conversations', True)).lower()}
  store_contacts: {str(storage.get('store_contacts', True)).lower()}
  store_code_snippets: {str(storage.get('store_code_snippets', True)).lower()}
  store_app_usage: {str(storage.get('store_app_usage', True)).lower()}
  store_browsing_history: {str(storage.get('store_browsing_history', True)).lower()}

# Screen Perception
perception:
  enabled: true
  capture_interval: 5.0
  multi_monitor: true
  ocr_language: eng

# Privacy (Minimal - Only protect truly sensitive data)
privacy:
  screen_capture: true
  activity_monitoring: true
  track_keyboard: true
  track_mouse: true
  track_clipboard: true
  excluded_windows: []  # No exclusions - full access
  privacy_filter: false  # Disabled for full intelligence

# Tools
tools:
  enabled_categories:
    - system
    - search
    - productivity
    - coding
    - files
    - web
    - automation

# MCP Servers
mcp:
  enabled: true
  servers: []

# Security (Only destructive protection)
security:
  require_confirmation_for_sensitive: false
  require_confirmation_for_destructive: true
  max_file_size_mb: 100
  allowed_directories: []  # All directories allowed
  audit_log_path: data/audit.log
"""
    return config


def generate_env_file(api_keys: dict) -> str:
    """Generate .env file content."""
    lines = ["# IntelCLaw Environment Variables", "# Generated by onboarding wizard", ""]
    
    for key, value in api_keys.items():
        lines.append(f"{key}={value}")
    
    return "\n".join(lines)


async def test_configuration(api_keys: dict):
    """Test the configuration."""
    console.print("\n[bold cyan]Testing Configuration...[/bold cyan]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Test OpenAI
        task = progress.add_task("Testing OpenAI connection...", total=None)
        try:
            import openai
            client = openai.OpenAI(api_key=api_keys.get("OPENAI_API_KEY", ""))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'IntelCLaw ready!' in 3 words or less"}],
                max_tokens=10
            )
            progress.update(task, description="[green]✓ OpenAI connected[/green]")
        except Exception as e:
            progress.update(task, description=f"[red]✗ OpenAI error: {e}[/red]")
        
        # Test Tavily
        if api_keys.get("TAVILY_API_KEY"):
            task = progress.add_task("Testing Tavily connection...", total=None)
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=api_keys.get("TAVILY_API_KEY", ""))
                progress.update(task, description="[green]✓ Tavily configured[/green]")
            except Exception as e:
                progress.update(task, description=f"[yellow]⚠ Tavily: {e}[/yellow]")


def save_configuration(config_content: str, env_content: str, profile: dict):
    """Save configuration files."""
    console.print("\n[bold cyan]Saving Configuration...[/bold cyan]\n")
    
    # Save config.yaml
    config_path = Path("config.yaml")
    config_path.write_text(config_content, encoding="utf-8")
    console.print(f"[green]✓ Saved {config_path}[/green]")
    
    # Save .env
    env_path = Path(".env")
    env_path.write_text(env_content, encoding="utf-8")
    console.print(f"[green]✓ Saved {env_path}[/green]")
    
    # Save user profile
    import json
    profile_path = Path("data/user_profile.json")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    console.print(f"[green]✓ Saved {profile_path}[/green]")
    
    # Create data directories
    for dir_name in ["data", "data/vector_db", "logs", "skills"]:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    console.print("[green]✓ Created data directories[/green]")


def print_completion():
    """Print completion message."""
    console.print("\n")
    console.print(Panel("""
[bold green]🎉 IntelCLaw Setup Complete! 🎉[/bold green]

Your autonomous AI agent is now configured and ready to run.

[bold]To start IntelCLaw:[/bold]
    uv run python main.py

[bold]To summon the agent:[/bold]
    Press Ctrl+Shift+Space

[bold]What IntelCLaw can now do:[/bold]
    ✓ Modify its own configuration
    ✓ Create and improve skills autonomously
    ✓ Diagnose and fix issues automatically
    ✓ Change AI models as needed
    ✓ Store contacts and personal information
    ✓ Full screen and activity access
    ✓ Everything except destructive operations

[bold yellow]Safety Note:[/bold yellow]
Destructive operations (delete, format, etc.) will always ask for confirmation.
    """, title="Setup Complete", border_style="green"))


async def main():
    """Main onboarding flow."""
    print_banner()
    
    console.print("\n[bold]Welcome to IntelCLaw Setup Wizard![/bold]")
    console.print("This wizard will configure your autonomous AI agent.\n")
    
    if not Confirm.ask("Ready to begin setup?", default=True):
        console.print("[yellow]Setup cancelled.[/yellow]")
        return
    
    # Collect configuration
    api_keys = setup_api_keys()
    copilot = setup_github_copilot()
    profile = setup_user_profile()
    autonomy = setup_autonomy_level()
    storage = setup_data_storage()
    
    # Generate configuration
    config_content = generate_config(api_keys, copilot, profile, autonomy, storage)
    env_content = generate_env_file(api_keys)
    
    # Show summary
    console.print("\n[bold cyan]Configuration Summary:[/bold cyan]")
    table = Table()
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("User", profile.get("name", "Not set"))
    table.add_row("Model", copilot.get("model", "gpt-4o"))
    table.add_row("Autonomy", autonomy.get("level", "full"))
    table.add_row("Copilot Integration", "Enabled" if copilot.get("enabled") else "Disabled")
    table.add_row("Data Retention", f"{storage.get('retention_days', 365)} days")
    
    console.print(table)
    
    if Confirm.ask("\nSave this configuration?", default=True):
        # Test configuration
        await test_configuration(api_keys)
        
        # Save files
        save_configuration(config_content, env_content, profile)
        
        # Print completion
        print_completion()
    else:
        console.print("[yellow]Configuration not saved.[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
