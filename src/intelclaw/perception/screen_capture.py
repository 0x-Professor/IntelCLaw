"""
Screen Capture - Fast screenshot capture for Windows.

Uses mss for high-performance multi-monitor screenshot capture.
"""

import asyncio
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    logger.warning("mss not available - screen capture disabled")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available - image processing disabled")


class ScreenCapture:
    """
    High-performance screen capture for Windows.
    
    Features:
    - Multi-monitor support
    - Active window capture
    - Region capture
    - Format conversion (PNG, JPEG)
    - Memory-efficient streaming
    """
    
    def __init__(self, multi_monitor: bool = True):
        """
        Initialize screen capture.
        
        Args:
            multi_monitor: Enable multi-monitor capture
        """
        self._multi_monitor = multi_monitor
        self._sct: Optional[Any] = None
        self._monitors: List[Dict[str, int]] = []
        
    async def initialize(self) -> None:
        """Initialize the screen capture system."""
        if not MSS_AVAILABLE:
            logger.warning("Screen capture not available (mss not installed)")
            return
        
        try:
            self._sct = mss.mss()
            self._monitors = list(self._sct.monitors)
            logger.info(f"Screen capture initialized with {len(self._monitors) - 1} monitors")
        except Exception as e:
            logger.error(f"Failed to initialize screen capture: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown screen capture."""
        if self._sct:
            self._sct.close()
            self._sct = None
    
    async def capture(
        self,
        monitor: int = 0,
        region: Optional[Dict[str, int]] = None,
        save_path: Optional[str] = None,
        format: str = "png"
    ) -> Optional[bytes]:
        """
        Capture a screenshot.
        
        Args:
            monitor: Monitor index (0 = all monitors, 1+ = specific monitor)
            region: Specific region {"left", "top", "width", "height"}
            save_path: Path to save the image
            format: Image format ("png" or "jpeg")
            
        Returns:
            Image bytes or None on failure
        """
        if not self._sct:
            return None
        
        try:
            # Determine capture area
            if region:
                capture_area = region
            elif monitor < len(self._monitors):
                capture_area = self._monitors[monitor]
            else:
                capture_area = self._monitors[0]  # All monitors
            
            # Capture
            screenshot = await asyncio.to_thread(
                self._sct.grab, capture_area
            )
            
            # Convert to bytes
            if PIL_AVAILABLE:
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                
                buffer = io.BytesIO()
                img.save(buffer, format=format.upper())
                image_bytes = buffer.getvalue()
                
                # Save if path provided
                if save_path:
                    path = Path(save_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(save_path)
                    logger.debug(f"Screenshot saved to {save_path}")
                
                return image_bytes
            else:
                # Use mss raw output
                return mss.tools.to_png(screenshot.rgb, screenshot.size)
            
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None
    
    async def capture_active_window(self) -> Optional[bytes]:
        """Capture only the active window."""
        try:
            import win32gui
            import win32con
            
            # Get active window handle
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return await self.capture(monitor=0)
            
            # Get window rectangle
            rect = win32gui.GetWindowRect(hwnd)
            region = {
                "left": rect[0],
                "top": rect[1],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1],
            }
            
            return await self.capture(region=region)
            
        except ImportError:
            logger.warning("win32gui not available, capturing full screen")
            return await self.capture(monitor=0)
        except Exception as e:
            logger.error(f"Active window capture failed: {e}")
            return await self.capture(monitor=0)
    
    async def capture_region(
        self,
        left: int,
        top: int,
        width: int,
        height: int
    ) -> Optional[bytes]:
        """Capture a specific region."""
        return await self.capture(region={
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        })
    
    def get_monitor_info(self) -> List[Dict[str, int]]:
        """Get information about available monitors."""
        return self._monitors.copy()
    
    @property
    def is_available(self) -> bool:
        """Check if screen capture is available."""
        return self._sct is not None
