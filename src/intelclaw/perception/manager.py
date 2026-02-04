"""
Perception Manager - Coordinates all perception components.

Aggregates screen capture, OCR, UI automation, and activity monitoring.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from intelclaw.perception.screen_capture import ScreenCapture
from intelclaw.perception.ocr import OCRProcessor
from intelclaw.perception.ui_automation import UIAutomation
from intelclaw.perception.activity_monitor import ActivityMonitor
from intelclaw.perception.context_builder import ContextBuilder

if TYPE_CHECKING:
    from intelclaw.config.manager import ConfigManager
    from intelclaw.core.events import EventBus


@dataclass
class PerceptionContext:
    """Aggregated perception context."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    active_window: Optional[str] = None
    active_window_title: Optional[str] = None
    screen_text: Optional[str] = None
    ui_elements: List[Dict[str, Any]] = field(default_factory=list)
    recent_activity: List[Dict[str, Any]] = field(default_factory=list)
    clipboard_content: Optional[str] = None
    mouse_position: Optional[tuple] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "active_window": self.active_window,
            "active_window_title": self.active_window_title,
            "screen_text": self.screen_text,
            "ui_elements": self.ui_elements,
            "recent_activity": self.recent_activity,
            "clipboard_content": self.clipboard_content,
            "mouse_position": self.mouse_position,
        }


class PerceptionManager:
    """
    Manages all perception components and provides unified context.
    
    Components:
    - Screen Capture: Screenshots and image processing
    - OCR: Text extraction from screen
    - UI Automation: Windows UI element detection
    - Activity Monitor: Track user activity (privacy-aware)
    - Context Builder: Aggregate and filter context
    """
    
    def __init__(
        self,
        config: "ConfigManager",
        event_bus: "EventBus",
    ):
        """
        Initialize perception manager.
        
        Args:
            config: Configuration manager
            event_bus: Event bus for notifications
        """
        self.config = config
        self.event_bus = event_bus
        
        # Components (initialized later)
        self.screen_capture: Optional[ScreenCapture] = None
        self.ocr: Optional[OCRProcessor] = None
        self.ui_automation: Optional[UIAutomation] = None
        self.activity_monitor: Optional[ActivityMonitor] = None
        self.context_builder: Optional[ContextBuilder] = None
        
        # State
        self._is_active = False
        self._capture_interval = 5.0  # seconds
        self._last_context: Optional[PerceptionContext] = None
        
        logger.debug("PerceptionManager created")
    
    async def initialize(self) -> None:
        """Initialize all perception components."""
        logger.info("Initializing perception layer...")
        
        # Get config
        perception_config = self.config.get("perception", {})
        self._capture_interval = perception_config.get("capture_interval", 5.0)
        
        # Initialize components
        self.screen_capture = ScreenCapture(
            multi_monitor=perception_config.get("multi_monitor", True)
        )
        await self.screen_capture.initialize()
        
        self.ocr = OCRProcessor(
            language=perception_config.get("ocr_language", "eng")
        )
        await self.ocr.initialize()
        
        self.ui_automation = UIAutomation()
        await self.ui_automation.initialize()
        
        # Activity monitor (privacy settings)
        privacy_config = self.config.get("privacy", {})
        self.activity_monitor = ActivityMonitor(
            track_keyboard=privacy_config.get("track_keyboard", False),
            track_mouse=privacy_config.get("track_mouse", True),
            track_clipboard=privacy_config.get("track_clipboard", False),
            excluded_windows=privacy_config.get("excluded_windows", []),
        )
        await self.activity_monitor.initialize()
        
        self.context_builder = ContextBuilder(
            privacy_filter=privacy_config.get("privacy_filter", True)
        )
        
        self._is_active = True
        logger.success("Perception layer initialized")
    
    async def shutdown(self) -> None:
        """Shutdown all perception components."""
        logger.info("Shutting down perception layer...")
        self._is_active = False
        
        if self.activity_monitor:
            await self.activity_monitor.shutdown()
        
        if self.ui_automation:
            await self.ui_automation.shutdown()
        
        if self.screen_capture:
            await self.screen_capture.shutdown()
        
        logger.info("Perception layer shutdown complete")
    
    async def run(self) -> None:
        """Background run loop for continuous perception."""
        logger.info("Perception loop started")
        
        while self._is_active:
            try:
                # Update context
                context = await self._capture_context()
                self._last_context = context
                
                # Emit context update event
                await self.event_bus.emit("perception.context_updated", {
                    "active_window": context.active_window,
                    "has_text": bool(context.screen_text),
                })
                
                await asyncio.sleep(self._capture_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Perception error: {e}")
                await asyncio.sleep(1)
    
    async def _capture_context(self) -> PerceptionContext:
        """Capture current system context."""
        context = PerceptionContext()
        
        # Get active window info
        if self.ui_automation:
            window_info = await self.ui_automation.get_active_window()
            context.active_window = window_info.get("process_name")
            context.active_window_title = window_info.get("title")
            context.ui_elements = await self.ui_automation.get_elements()
        
        # Capture screen and OCR (if enabled)
        if self.screen_capture and self.ocr:
            screenshot = await self.screen_capture.capture_active_window()
            if screenshot:
                context.screen_text = await self.ocr.extract_text(screenshot)
        
        # Get activity
        if self.activity_monitor:
            context.recent_activity = self.activity_monitor.get_recent_activity()
            context.clipboard_content = await self.activity_monitor.get_clipboard()
            context.mouse_position = self.activity_monitor.get_mouse_position()
        
        # Apply privacy filter
        if self.context_builder:
            context = self.context_builder.filter_context(context)
        
        return context
    
    async def get_context(self) -> Dict[str, Any]:
        """Get current perception context."""
        if self._last_context:
            return self._last_context.to_dict()
        
        # Capture fresh context
        context = await self._capture_context()
        return context.to_dict()
    
    async def capture_screenshot(self, save_path: Optional[str] = None) -> Optional[bytes]:
        """Capture a screenshot of the current screen."""
        if not self.screen_capture:
            return None
        
        return await self.screen_capture.capture(save_path=save_path)
    
    async def get_screen_text(self) -> Optional[str]:
        """Get text visible on screen via OCR."""
        if not self.screen_capture or not self.ocr:
            return None
        
        screenshot = await self.screen_capture.capture_active_window()
        if screenshot:
            return await self.ocr.extract_text(screenshot)
        return None
    
    async def get_active_window(self) -> Dict[str, Any]:
        """Get information about the active window."""
        if not self.ui_automation:
            return {}
        
        return await self.ui_automation.get_active_window()
    
    @property
    def is_active(self) -> bool:
        """Check if perception is active."""
        return self._is_active
