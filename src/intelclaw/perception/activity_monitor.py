"""
Activity Monitor - Track user activity with privacy controls.

Monitors keyboard, mouse, clipboard, and application activity.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Set

from loguru import logger

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

try:
    import pyautogui
    MOUSE_AVAILABLE = True
except ImportError:
    MOUSE_AVAILABLE = False

try:
    import win32clipboard
    import win32con
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False


@dataclass
class ActivityEvent:
    """Represents a user activity event."""
    
    type: str  # "keyboard", "mouse", "clipboard", "window"
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)


class ActivityMonitor:
    """
    Privacy-aware user activity monitoring.
    
    Features:
    - Keyboard activity (not keystrokes for privacy)
    - Mouse position and clicks
    - Clipboard changes
    - Window focus changes
    
    Privacy:
    - No keystroke logging by default
    - Configurable exclusions
    - Data retention limits
    """
    
    MAX_HISTORY = 100
    
    def __init__(
        self,
        track_keyboard: bool = False,
        track_mouse: bool = True,
        track_clipboard: bool = False,
        excluded_windows: Optional[List[str]] = None,
    ):
        """
        Initialize activity monitor.
        
        Args:
            track_keyboard: Track keyboard activity (not keystrokes)
            track_mouse: Track mouse position
            track_clipboard: Track clipboard changes
            excluded_windows: Window patterns to exclude
        """
        self._track_keyboard = track_keyboard
        self._track_mouse = track_mouse
        self._track_clipboard = track_clipboard
        self._excluded_windows = set(excluded_windows or [])
        
        self._activity_history: Deque[ActivityEvent] = deque(maxlen=self.MAX_HISTORY)
        self._last_mouse_pos: Optional[tuple] = None
        self._last_clipboard: Optional[str] = None
        self._is_active = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize activity monitoring."""
        logger.info("Initializing activity monitor...")
        
        self._is_active = True
        
        # Start keyboard listener if enabled
        if self._track_keyboard and KEYBOARD_AVAILABLE:
            keyboard.on_press(self._on_key_press)
            logger.debug("Keyboard activity tracking enabled")
        
        # Start background monitor
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("Activity monitor initialized")
    
    async def shutdown(self) -> None:
        """Shutdown activity monitoring."""
        logger.info("Shutting down activity monitor...")
        self._is_active = False
        
        if self._track_keyboard and KEYBOARD_AVAILABLE:
            keyboard.unhook_all()
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Activity monitor shutdown complete")
    
    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._is_active:
            try:
                # Track mouse position
                if self._track_mouse and MOUSE_AVAILABLE:
                    try:
                        pos = pyautogui.position()
                        if pos != self._last_mouse_pos:
                            self._last_mouse_pos = pos
                            # Only log significant movements
                            if self._is_significant_movement(pos):
                                self._add_event(ActivityEvent(
                                    type="mouse",
                                    data={"x": pos[0], "y": pos[1]}
                                ))
                    except:
                        pass
                
                # Track clipboard
                if self._track_clipboard:
                    clipboard = await self.get_clipboard()
                    if clipboard and clipboard != self._last_clipboard:
                        self._last_clipboard = clipboard
                        self._add_event(ActivityEvent(
                            type="clipboard",
                            data={"length": len(clipboard)}  # Don't store content
                        ))
                
                await asyncio.sleep(0.5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Activity monitor error: {e}")
                await asyncio.sleep(1)
    
    def _on_key_press(self, event) -> None:
        """Handle key press (only logs activity, not keys)."""
        self._add_event(ActivityEvent(
            type="keyboard",
            data={"activity": True}  # Don't log actual keys
        ))
    
    def _is_significant_movement(self, pos: tuple) -> bool:
        """Check if mouse movement is significant."""
        if not self._last_mouse_pos:
            return True
        
        dx = abs(pos[0] - self._last_mouse_pos[0])
        dy = abs(pos[1] - self._last_mouse_pos[1])
        
        return dx > 50 or dy > 50
    
    def _add_event(self, event: ActivityEvent) -> None:
        """Add an activity event to history."""
        self._activity_history.append(event)
    
    async def get_clipboard(self) -> Optional[str]:
        """Get current clipboard text content."""
        if not CLIPBOARD_AVAILABLE:
            return None
        
        try:
            win32clipboard.OpenClipboard()
            try:
                if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                    data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                    return str(data)
            finally:
                win32clipboard.CloseClipboard()
        except:
            pass
        
        return None
    
    def get_recent_activity(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent activity events."""
        events = list(self._activity_history)[-limit:]
        return [
            {
                "type": e.type,
                "timestamp": e.timestamp.isoformat(),
                "data": e.data,
            }
            for e in events
        ]
    
    def get_mouse_position(self) -> Optional[tuple]:
        """Get last known mouse position."""
        return self._last_mouse_pos
    
    def is_window_excluded(self, window_title: str) -> bool:
        """Check if a window should be excluded from monitoring."""
        window_lower = window_title.lower()
        for pattern in self._excluded_windows:
            if pattern.lower() in window_lower:
                return True
        return False
    
    @property
    def is_active(self) -> bool:
        """Check if monitoring is active."""
        return self._is_active
