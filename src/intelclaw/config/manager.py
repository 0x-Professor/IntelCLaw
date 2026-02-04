"""
Configuration Manager - Settings and preferences.

Handles YAML/JSON configuration with hot-reload support.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigManager:
    """
    Configuration manager for IntelCLaw.
    
    Features:
    - YAML/JSON configuration files
    - Hot-reload support
    - Environment variable overrides
    - Profile switching
    - Default values
    """
    
    DEFAULT_CONFIG = {
        "app": {
            "name": "IntelCLaw",
            "version": "0.1.0",
            "debug": False,
        },
        "hotkeys": {
            "summon": "ctrl+shift+space",
            "quick_action": "ctrl+shift+q",
            "dismiss": "escape",
        },
        "ui": {
            "theme": "dark",
            "transparency": 0.95,
            "position": "center",
            "width": 600,
            "height": 500,
        },
        "models": {
            "primary": "gpt-4o",
            "fallback": "gpt-4o-mini",
            "coding": "gpt-4o",
            "temperature": 0.1,
        },
        "memory": {
            "max_conversation_history": 50,
            "working_db_path": "data/working_memory.db",
            "vector_store": {
                "collection": "intelclaw",
                "path": "data/vector_db",
            },
            "retention_days": 365,
            "auto_cleanup": True,
        },
        "perception": {
            "capture_interval": 5.0,
            "multi_monitor": True,
            "ocr_language": "eng",
        },
        "privacy": {
            "screen_capture": True,
            "activity_monitoring": True,
            "track_keyboard": False,
            "track_mouse": True,
            "track_clipboard": False,
            "excluded_windows": ["*password*", "*bank*", "*1password*"],
            "privacy_filter": True,
        },
        "tools": {
            "enabled_categories": ["system", "search", "productivity"],
        },
        "mcp": {
            "enabled": True,
            "servers": [],
        },
        "security": {
            "require_confirmation_for_sensitive": True,
            "max_file_size_mb": 10,
            "allowed_directories": [],
        },
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self._config_path = Path(config_path) if config_path else Path("config.yaml")
        self._config: Dict[str, Any] = {}
        self._watchers: list = []
        self._loaded = False
    
    async def load(self) -> None:
        """Load configuration from file."""
        # Start with defaults
        self._config = self._deep_copy(self.DEFAULT_CONFIG)
        
        # Load from file if exists
        if self._config_path.exists():
            try:
                content = self._config_path.read_text(encoding="utf-8")
                
                if self._config_path.suffix in [".yaml", ".yml"] and YAML_AVAILABLE:
                    file_config = yaml.safe_load(content) or {}
                else:
                    file_config = json.loads(content)
                
                # Merge with defaults
                self._deep_merge(self._config, file_config)
                logger.info(f"Configuration loaded from {self._config_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        else:
            # Create default config file
            await self.save()
            logger.info("Created default configuration file")
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        self._loaded = True
    
    async def save(self) -> None:
        """Save configuration to file."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self._config_path.suffix in [".yaml", ".yml"] and YAML_AVAILABLE:
                content = yaml.dump(self._config, default_flow_style=False, sort_keys=False)
            else:
                content = json.dumps(self._config, indent=2)
            
            self._config_path.write_text(content, encoding="utf-8")
            logger.debug(f"Configuration saved to {self._config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Dot-notation key (e.g., "models.primary")
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        parts = key.split(".")
        value = self._config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Dot-notation key
            value: Value to set
        """
        parts = key.split(".")
        config = self._config
        
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        config[parts[-1]] = value
        
        # Notify watchers
        for watcher in self._watchers:
            try:
                watcher(key, value)
            except Exception as e:
                logger.warning(f"Config watcher error: {e}")
    
    def watch(self, callback) -> None:
        """Register a configuration change watcher."""
        self._watchers.append(callback)
    
    def unwatch(self, callback) -> None:
        """Unregister a watcher."""
        if callback in self._watchers:
            self._watchers.remove(callback)
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        import os
        
        # Map environment variables to config keys
        env_mappings = {
            "INTELCLAW_DEBUG": ("app.debug", lambda x: x.lower() == "true"),
            "OPENAI_API_KEY": ("openai_api_key", str),
            "TAVILY_API_KEY": ("tools.tavily_api_key", str),
            "GITHUB_TOKEN": ("tools.github_token", str),
        }
        
        for env_var, (config_key, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                try:
                    self.set(config_key, converter(value))
                    logger.debug(f"Applied env override: {env_var}")
                except Exception as e:
                    logger.warning(f"Failed to apply {env_var}: {e}")
    
    def _deep_merge(self, base: dict, override: dict) -> None:
        """Deep merge override into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy a dictionary."""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        return obj
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._deep_copy(self._config)
