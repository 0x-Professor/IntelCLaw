"""
System Tray - Background tray icon with menu.

Provides system tray integration for background operation.
"""

import threading
from typing import Any, Callable, Optional, TYPE_CHECKING

from loguru import logger

try:
    import pystray
    from PIL import Image, ImageDraw
    PYSTRAY_AVAILABLE = True
except ImportError:
    PYSTRAY_AVAILABLE = False
    logger.warning("pystray not available - system tray disabled")

if TYPE_CHECKING:
    from intelclaw.core.app import IntelCLawApp


class SystemTray:
    """
    System tray icon with context menu.
    
    Features:
    - Background icon
    - Status indicator
    - Quick actions menu
    - Notifications
    """
    
    def __init__(self, app: "IntelCLawApp"):
        """
        Initialize system tray.
        
        Args:
            app: Main application instance
        """
        self._app = app
        self._icon: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        if PYSTRAY_AVAILABLE:
            self._setup_tray()
    
    def _setup_tray(self) -> None:
        """Set up the system tray icon."""
        # Create icon image
        image = self._create_icon_image()
        
        # Create menu
        menu = pystray.Menu(
            pystray.MenuItem("Show IntelCLaw", self._on_show),
            pystray.MenuItem("Settings", self._on_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Pause Monitoring", self._on_pause),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", self._on_exit),
        )
        
        # Create icon
        self._icon = pystray.Icon(
            name="IntelCLaw",
            icon=image,
            title="IntelCLaw - AI Assistant",
            menu=menu,
        )
        
        # Start in thread
        self._thread = threading.Thread(target=self._run_icon, daemon=True)
        self._thread.start()
        self._running = True
        
        logger.info("System tray icon created")
    
    def _create_icon_image(self, size: int = 64) -> "Image.Image":
        """Create the tray icon image."""
        # Create a simple icon
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Draw a circle with gradient
        padding = 4
        draw.ellipse(
            [padding, padding, size - padding, size - padding],
            fill=(70, 130, 220, 255),
            outline=(100, 150, 255, 255),
            width=2
        )
        
        # Draw "AI" text
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", size // 3)
        except:
            font = ImageFont.load_default()
        
        text = "AI"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - 2
        
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
        
        return image
    
    def _run_icon(self) -> None:
        """Run the icon in a thread."""
        if self._icon:
            self._icon.run()
    
    def _on_show(self, icon, item) -> None:
        """Show the overlay window."""
        if self._app and self._app.overlay:
            self._app.toggle_overlay()
    
    def _on_settings(self, icon, item) -> None:
        """Open settings window."""
        logger.info("Settings requested")
        # TODO: Implement settings window
    
    def _on_pause(self, icon, item) -> None:
        """Pause/resume monitoring."""
        logger.info("Pause monitoring requested")
        # TODO: Implement pause functionality
    
    def _on_exit(self, icon, item) -> None:
        """Exit the application."""
        logger.info("Exit requested from tray")
        self.stop()
        
        import asyncio
        if self._app:
            asyncio.create_task(self._app.shutdown())
    
    def stop(self) -> None:
        """Stop the system tray icon."""
        self._running = False
        if self._icon:
            self._icon.stop()
            self._icon = None
        logger.info("System tray stopped")
    
    def update_icon(self, status: str = "normal") -> None:
        """
        Update the tray icon based on status.
        
        Args:
            status: "normal", "busy", "error", "paused"
        """
        if not self._icon:
            return
        
        # Create icon with different colors based on status
        colors = {
            "normal": (70, 130, 220),
            "busy": (220, 170, 50),
            "error": (220, 70, 70),
            "paused": (120, 120, 120),
        }
        
        color = colors.get(status, colors["normal"])
        
        # Update would require recreating the image
        # For simplicity, just log
        logger.debug(f"Tray status: {status}")
    
    def show_notification(
        self,
        title: str,
        message: str,
        timeout: int = 5
    ) -> None:
        """
        Show a notification.
        
        Args:
            title: Notification title
            message: Notification message
            timeout: Display duration in seconds
        """
        if self._icon:
            try:
                self._icon.notify(message, title)
            except Exception as e:
                logger.warning(f"Notification failed: {e}")
    
    @property
    def is_running(self) -> bool:
        """Check if tray is running."""
        return self._running
