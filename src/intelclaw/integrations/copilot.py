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
    - Model switching (supports Copilot API models like gpt-4o, o1, etc.)
    - Context synchronization
    - Suggestion handling
    - Settings management
    
    Two modes:
    1. Copilot API - Uses Copilot subscription models (gpt-4o, o1, etc.)
    2. GitHub Models API - Free tier with rate limits (gpt-4o-mini, llama, etc.)
    """
    
    # VS Code Copilot settings path
    VSCODE_SETTINGS_PATH = Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "Code" / "User" / "settings.json"
    
    # GitHub Copilot models (available via subscription)
    # Based on OpenClaw's supported Copilot models
    COPILOT_MODELS = {
        "gpt-4o": "GPT-4o - Best overall performance (Copilot)",
        "gpt-4.1": "GPT-4.1 - Latest GPT-4 (Copilot)",
        "gpt-4.1-mini": "GPT-4.1 Mini - Fast and efficient (Copilot)",
        "gpt-4.1-nano": "GPT-4.1 Nano - Ultra fast (Copilot)",
        "o1": "O1 - Advanced reasoning (Copilot)",
        "o1-mini": "O1 Mini - Fast reasoning (Copilot)",
        "o3-mini": "O3 Mini - Smaller O3 model (Copilot)",
    }
    
    # GitHub Models API models (free, rate-limited)
    FREE_MODELS = {
        "gpt-4o-mini": "GPT-4o Mini - Free tier (rate-limited)",
        "gpt-4o": "GPT-4o - Free tier (rate-limited)",
        "llama-3.3-70b": "Llama 3.3 70B - Meta (free)",
        "deepseek-r1": "DeepSeek R1 - Reasoning (free)",
        "phi-4": "Phi-4 - Microsoft (free)",
    }
    
    @property
    def AVAILABLE_MODELS(self) -> Dict[str, str]:
        """Get all available models combining Copilot and free models."""
        return {**self.COPILOT_MODELS, **self.FREE_MODELS}
    
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
        
        Supports both Copilot API models (require subscription) and
        free GitHub Models API models.
        
        Args:
            model: Model name (e.g., 'gpt-4o', 'gpt-4o-mini', 'llama-3.3-70b')
            
        Returns:
            Result with old and new model
        """
        all_models = self.AVAILABLE_MODELS
        
        # Allow custom models not in the list
        if model not in all_models:
            logger.info(f"Using custom model: {model}")
        
        old_model = self._current_model
        self._current_model = model
        
        # Update config
        self.config.set("copilot.model", model)
        self.config.set("models.primary", model)
        await self.config.save()
        
        # Try to update VS Code settings
        await self._update_vscode_settings(model)
        
        logger.info(f"Copilot model changed: {old_model} -> {model}")
        
        # Get description for known models
        description = self.AVAILABLE_MODELS.get(model, f"Custom model: {model}")
        
        return {
            "success": True,
            "old_model": old_model,
            "new_model": model,
            "description": description,
            "is_copilot_model": model in self.COPILOT_MODELS,
            "is_free_model": model in self.FREE_MODELS,
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
    
    async def get_available_models(self) -> Dict[str, Any]:
        """
        Get list of available models with categories.
        
        Returns models organized by:
        - copilot: Models available via Copilot subscription
        - free: Models available via free GitHub Models API
        """
        return {
            "copilot_models": self.COPILOT_MODELS.copy(),
            "free_models": self.FREE_MODELS.copy(),
            "all": self.AVAILABLE_MODELS,
            "current": self._current_model,
        }
    
    async def get_copilot_models(self) -> Dict[str, str]:
        """Get models available via Copilot subscription."""
        return self.COPILOT_MODELS.copy()
    
    async def get_free_models(self) -> Dict[str, str]:
        """Get models available via free GitHub Models API."""
        return self.FREE_MODELS.copy()
    
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
