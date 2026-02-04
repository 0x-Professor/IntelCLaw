"""
GitHub Copilot Integration - Model switching and context sync.

Enables IntelCLaw to work alongside GitHub Copilot.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from intelclaw.config.manager import ConfigManager


class CopilotIntegration:
    """
    GitHub Copilot integration for IntelCLaw.
    
    Features:
    - Model switching
    - Context synchronization
    - Suggestion handling
    - Settings management
    """
    
    # VS Code Copilot settings path
    VSCODE_SETTINGS_PATH = Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "Code" / "User" / "settings.json"
    
    # Available models
    AVAILABLE_MODELS = {
        "gpt-4o": "GPT-4o - Best overall performance",
        "gpt-4o-mini": "GPT-4o Mini - Fast and efficient",
        "claude-3.5-sonnet": "Claude 3.5 Sonnet - Excellent for coding",
        "o1-preview": "O1 Preview - Advanced reasoning",
        "o1-mini": "O1 Mini - Fast reasoning",
    }
    
    def __init__(self, config: "ConfigManager"):
        """
        Initialize Copilot integration.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self._enabled = config.get("copilot.enabled", True)
        self._current_model = config.get("copilot.model", "gpt-4o-mini")
    
    async def initialize(self) -> None:
        """Initialize Copilot integration."""
        if not self._enabled:
            logger.info("Copilot integration disabled")
            return
        
        # Check if VS Code is available
        if not self.VSCODE_SETTINGS_PATH.exists():
            logger.warning("VS Code settings not found - Copilot integration limited")
        
        logger.info(f"Copilot integration initialized (model: {self._current_model})")
    
    async def get_current_model(self) -> str:
        """Get the current Copilot model."""
        return self._current_model
    
    async def set_model(self, model: str) -> Dict[str, Any]:
        """
        Change the Copilot model.
        
        Args:
            model: Model name
            
        Returns:
            Result with old and new model
        """
        if model not in self.AVAILABLE_MODELS:
            return {
                "success": False,
                "error": f"Unknown model: {model}",
                "available": list(self.AVAILABLE_MODELS.keys())
            }
        
        old_model = self._current_model
        self._current_model = model
        
        # Update config
        self.config.set("copilot.model", model)
        self.config.set("models.primary", model)
        await self.config.save()
        
        # Try to update VS Code settings
        await self._update_vscode_settings(model)
        
        logger.info(f"Copilot model changed: {old_model} -> {model}")
        
        return {
            "success": True,
            "old_model": old_model,
            "new_model": model,
            "description": self.AVAILABLE_MODELS[model]
        }
    
    async def _update_vscode_settings(self, model: str) -> bool:
        """Update VS Code settings for Copilot."""
        if not self.VSCODE_SETTINGS_PATH.exists():
            return False
        
        try:
            # Read current settings
            settings = json.loads(self.VSCODE_SETTINGS_PATH.read_text(encoding="utf-8"))
            
            # Update Copilot settings
            settings["github.copilot.chat.models.default"] = model
            
            # Map to provider-specific settings
            if "gpt" in model or "o1" in model:
                settings["github.copilot.chat.models.provider"] = "openai"
            elif "claude" in model:
                settings["github.copilot.chat.models.provider"] = "anthropic"
            
            # Write back
            self.VSCODE_SETTINGS_PATH.write_text(
                json.dumps(settings, indent=4),
                encoding="utf-8"
            )
            
            logger.info("VS Code Copilot settings updated")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to update VS Code settings: {e}")
            return False
    
    async def get_available_models(self) -> Dict[str, str]:
        """Get list of available models."""
        return self.AVAILABLE_MODELS.copy()
    
    async def enable(self) -> None:
        """Enable Copilot integration."""
        self._enabled = True
        self.config.set("copilot.enabled", True)
        await self.config.save()
        logger.info("Copilot integration enabled")
    
    async def disable(self) -> None:
        """Disable Copilot integration."""
        self._enabled = False
        self.config.set("copilot.enabled", False)
        await self.config.save()
        logger.info("Copilot integration disabled")
    
    async def sync_context(self, context: Dict[str, Any]) -> None:
        """
        Sync context with Copilot.
        
        Args:
            context: Context data to sync
        """
        if not self._enabled:
            return
        
        # Store context for Copilot to access
        context_path = Path("data/copilot_context.json")
        context_path.parent.mkdir(parents=True, exist_ok=True)
        context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
        
        logger.debug("Context synced with Copilot")
    
    async def get_settings(self) -> Dict[str, Any]:
        """Get current Copilot settings."""
        return {
            "enabled": self._enabled,
            "model": self._current_model,
            "auto_suggestions": self.config.get("copilot.auto_suggestions", True),
            "context_length": self.config.get("copilot.context_length", 8000),
            "sync_context": self.config.get("copilot.sync_context", True),
        }
    
    async def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update Copilot settings.
        
        Args:
            settings: Settings to update
            
        Returns:
            Updated settings
        """
        for key, value in settings.items():
            if key == "model":
                await self.set_model(value)
            elif key == "enabled":
                if value:
                    await self.enable()
                else:
                    await self.disable()
            else:
                self.config.set(f"copilot.{key}", value)
        
        await self.config.save()
        return await self.get_settings()


class ModelManager:
    """
    Manages AI models across the application.
    
    Handles model selection, fallback chains, and optimization.
    """
    
    def __init__(self, config: "ConfigManager"):
        """Initialize model manager."""
        self.config = config
        self._models: Dict[str, Any] = {}
        self._usage_stats: Dict[str, Dict[str, int]] = {}
    
    async def initialize(self) -> None:
        """Initialize model manager."""
        # Load model configurations
        self._models = {
            "primary": self.config.get("models.primary", "gpt-4o"),
            "fallback": self.config.get("models.fallback", "gpt-4o-mini"),
            "coding": self.config.get("models.coding", "gpt-4o"),
        }
        
        logger.info(f"Model manager initialized: {self._models}")
    
    async def get_model(self, purpose: str = "primary") -> str:
        """
        Get the model for a specific purpose.
        
        Args:
            purpose: "primary", "fallback", "coding"
            
        Returns:
            Model name
        """
        return self._models.get(purpose, self._models.get("primary", "gpt-4o"))
    
    async def set_model(self, purpose: str, model: str) -> Dict[str, Any]:
        """
        Set a model for a specific purpose.
        
        Args:
            purpose: Model purpose
            model: Model name
            
        Returns:
            Result
        """
        old_model = self._models.get(purpose)
        self._models[purpose] = model
        
        self.config.set(f"models.{purpose}", model)
        await self.config.save()
        
        logger.info(f"Model for {purpose} changed: {old_model} -> {model}")
        
        return {
            "success": True,
            "purpose": purpose,
            "old_model": old_model,
            "new_model": model
        }
    
    async def auto_select_model(self, task_type: str) -> str:
        """
        Automatically select the best model for a task.
        
        Args:
            task_type: Type of task
            
        Returns:
            Recommended model
        """
        # Task-specific model selection
        task_models = {
            "coding": self._models.get("coding", "gpt-4o"),
            "reasoning": "o1-preview" if "o1-preview" in str(self._models.values()) else "gpt-4o",
            "simple": self._models.get("fallback", "gpt-4o-mini"),
            "research": self._models.get("primary", "gpt-4o"),
            "creative": "claude-3.5-sonnet" if "claude" in str(self._models.values()) else "gpt-4o",
        }
        
        return task_models.get(task_type, self._models.get("primary", "gpt-4o"))
    
    def log_usage(self, model: str, tokens: int = 0, success: bool = True) -> None:
        """Log model usage for optimization."""
        if model not in self._usage_stats:
            self._usage_stats[model] = {"calls": 0, "tokens": 0, "errors": 0}
        
        self._usage_stats[model]["calls"] += 1
        self._usage_stats[model]["tokens"] += tokens
        if not success:
            self._usage_stats[model]["errors"] += 1
    
    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get usage statistics."""
        return self._usage_stats.copy()
