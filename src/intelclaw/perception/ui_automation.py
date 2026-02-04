"""
UI Automation - Windows UI element detection and interaction.

Uses pywinauto and Windows UI Automation for element detection.
"""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import win32gui
    import win32process
    import win32api
    import psutil
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    logger.warning("win32 modules not available")

try:
    from pywinauto import Desktop
    from pywinauto.application import Application
    PYWINAUTO_AVAILABLE = True
except ImportError:
    PYWINAUTO_AVAILABLE = False
    logger.warning("pywinauto not available")


class UIAutomation:
    """
    Windows UI Automation interface.
    
    Features:
    - Active window detection
    - UI element enumeration
    - Element interaction (click, type)
    - Window manipulation
    """
    
    def __init__(self):
        """Initialize UI automation."""
        self._desktop: Optional[Any] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize UI automation."""
        if not WIN32_AVAILABLE:
            logger.warning("UI Automation not available (win32 not installed)")
            return
        
        if PYWINAUTO_AVAILABLE:
            try:
                self._desktop = Desktop(backend="uia")
                self._initialized = True
                logger.info("UI Automation initialized with pywinauto")
            except Exception as e:
                logger.warning(f"pywinauto initialization failed: {e}")
        
        if not self._initialized:
            # Fallback to win32 only
            self._initialized = WIN32_AVAILABLE
            logger.info("UI Automation initialized with win32 only")
    
    async def shutdown(self) -> None:
        """Shutdown UI automation."""
        self._desktop = None
    
    async def get_active_window(self) -> Dict[str, Any]:
        """Get information about the active window."""
        if not WIN32_AVAILABLE:
            return {}
        
        try:
            hwnd = win32gui.GetForegroundWindow()
            
            if not hwnd:
                return {}
            
            # Get window title
            title = win32gui.GetWindowText(hwnd)
            
            # Get process info
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            process_name = ""
            try:
                process = psutil.Process(pid)
                process_name = process.name()
            except:
                pass
            
            # Get window rect
            rect = win32gui.GetWindowRect(hwnd)
            
            return {
                "hwnd": hwnd,
                "title": title,
                "process_id": pid,
                "process_name": process_name,
                "rect": {
                    "left": rect[0],
                    "top": rect[1],
                    "right": rect[2],
                    "bottom": rect[3],
                    "width": rect[2] - rect[0],
                    "height": rect[3] - rect[1],
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get active window: {e}")
            return {}
    
    async def get_elements(
        self,
        hwnd: Optional[int] = None,
        element_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get UI elements from a window.
        
        Args:
            hwnd: Window handle (None = active window)
            element_types: Filter by element types (Button, Edit, etc.)
            
        Returns:
            List of element dictionaries
        """
        if not self._initialized or not PYWINAUTO_AVAILABLE:
            return []
        
        try:
            if hwnd is None:
                hwnd = win32gui.GetForegroundWindow()
            
            if not hwnd:
                return []
            
            # Use pywinauto to enumerate elements
            elements = []
            
            app = Application(backend="uia").connect(handle=hwnd)
            window = app.window(handle=hwnd)
            
            # Get all descendants (limit depth for performance)
            for control in window.descendants()[:100]:
                try:
                    control_type = control.element_info.control_type
                    
                    # Filter by type if specified
                    if element_types and control_type not in element_types:
                        continue
                    
                    rect = control.rectangle()
                    
                    elements.append({
                        "type": control_type,
                        "name": control.element_info.name or "",
                        "automation_id": control.element_info.automation_id or "",
                        "rect": {
                            "left": rect.left,
                            "top": rect.top,
                            "right": rect.right,
                            "bottom": rect.bottom,
                        },
                        "is_enabled": control.is_enabled(),
                        "is_visible": control.is_visible(),
                    })
                except:
                    continue
            
            return elements
            
        except Exception as e:
            logger.error(f"Failed to get UI elements: {e}")
            return []
    
    async def click_element(
        self,
        hwnd: int,
        automation_id: Optional[str] = None,
        name: Optional[str] = None,
        coords: Optional[tuple] = None
    ) -> bool:
        """
        Click a UI element.
        
        Args:
            hwnd: Window handle
            automation_id: Element automation ID
            name: Element name
            coords: Direct coordinates (x, y)
            
        Returns:
            True if successful
        """
        if not PYWINAUTO_AVAILABLE:
            return False
        
        try:
            app = Application(backend="uia").connect(handle=hwnd)
            window = app.window(handle=hwnd)
            
            if coords:
                window.click_input(coords=coords)
                return True
            
            # Find element
            if automation_id:
                element = window.child_window(auto_id=automation_id)
            elif name:
                element = window.child_window(title=name)
            else:
                return False
            
            element.click_input()
            return True
            
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False
    
    async def type_text(
        self,
        hwnd: int,
        text: str,
        automation_id: Optional[str] = None
    ) -> bool:
        """Type text into an element or active window."""
        if not PYWINAUTO_AVAILABLE:
            return False
        
        try:
            app = Application(backend="uia").connect(handle=hwnd)
            window = app.window(handle=hwnd)
            
            if automation_id:
                element = window.child_window(auto_id=automation_id)
                element.type_keys(text, with_spaces=True)
            else:
                window.type_keys(text, with_spaces=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Type text failed: {e}")
            return False
    
    async def get_window_list(self) -> List[Dict[str, Any]]:
        """Get list of all visible windows."""
        if not WIN32_AVAILABLE:
            return []
        
        windows = []
        
        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    windows.append({
                        "hwnd": hwnd,
                        "title": title,
                    })
            return True
        
        win32gui.EnumWindows(callback, None)
        return windows
    
    async def focus_window(self, hwnd: int) -> bool:
        """Bring a window to foreground."""
        if not WIN32_AVAILABLE:
            return False
        
        try:
            win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception as e:
            logger.error(f"Focus window failed: {e}")
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if UI automation is available."""
        return self._initialized
