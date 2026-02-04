"""
Overlay Window - Transparent chat UI.

PyQt6-based transparent overlay that appears above all windows.
"""

import asyncio
from typing import Any, Callable, Optional, TYPE_CHECKING

from loguru import logger

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLineEdit, QPushButton, QLabel, QScrollArea, QFrame
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QColor, QPalette, QKeySequence, QShortcut
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logger.warning("PyQt6 not available - overlay disabled")

if TYPE_CHECKING:
    from intelclaw.core.app import IntelCLawApp


class OverlayWindow:
    """
    Transparent overlay window for IntelCLaw.
    
    Features:
    - Always-on-top transparent window
    - Chat-style interface
    - Global hotkey activation
    - Smooth animations
    - Markdown rendering
    """
    
    def __init__(self, app: "IntelCLawApp"):
        """
        Initialize overlay window.
        
        Args:
            app: Main application instance
        """
        self._app = app
        self._window: Optional[Any] = None
        self._qapp: Optional[Any] = None
        self._visible = False
        self._hotkey_registered = False
        
        if PYQT_AVAILABLE:
            self._setup_window()
        else:
            logger.warning("Overlay not available (PyQt6 not installed)")
    
    def _setup_window(self) -> None:
        """Set up the PyQt window."""
        # Create QApplication if needed
        self._qapp = QApplication.instance()
        if not self._qapp:
            self._qapp = QApplication([])
        
        self._window = TransparentOverlay(self._app)
        
        # Register global hotkey
        self._register_hotkey()
        
        logger.info("Overlay window created")
    
    def _register_hotkey(self) -> None:
        """Register global hotkey for overlay toggle."""
        try:
            import keyboard
            
            hotkey = self._app.config.get("hotkeys.summon", "ctrl+shift+space") if self._app.config else "ctrl+shift+space"
            
            keyboard.add_hotkey(hotkey, self.toggle)
            self._hotkey_registered = True
            logger.info(f"Global hotkey registered: {hotkey}")
            
        except Exception as e:
            logger.warning(f"Failed to register hotkey: {e}")
    
    def show(self) -> None:
        """Show the overlay window."""
        if self._window:
            self._window.show()
            self._window.activateWindow()
            self._window.raise_()
            self._visible = True
    
    def hide(self) -> None:
        """Hide the overlay window."""
        if self._window:
            self._window.hide()
            self._visible = False
    
    def toggle(self) -> None:
        """Toggle overlay visibility."""
        if self._visible:
            self.hide()
        else:
            self.show()
    
    async def close(self) -> None:
        """Close the overlay window."""
        if self._hotkey_registered:
            try:
                import keyboard
                keyboard.unhook_all()
            except:
                pass
        
        if self._window:
            self._window.close()
            self._window = None
    
    @property
    def is_visible(self) -> bool:
        """Check if overlay is visible."""
        return self._visible


if PYQT_AVAILABLE:
    class TransparentOverlay(QMainWindow):
        """PyQt6 transparent overlay window."""
        
        response_received = pyqtSignal(str)
        
        def __init__(self, app: "IntelCLawApp"):
            super().__init__()
            self._app = app
            self._setup_ui()
            self._setup_style()
            
            # Connect signal
            self.response_received.connect(self._display_response)
        
        def _setup_ui(self) -> None:
            """Set up the UI components."""
            # Window properties
            self.setWindowTitle("IntelCLaw")
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.WindowStaysOnTopHint |
                Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            
            # Size and position
            self.setFixedSize(600, 500)
            self._center_on_screen()
            
            # Main widget
            main_widget = QWidget()
            main_widget.setObjectName("mainWidget")
            self.setCentralWidget(main_widget)
            
            layout = QVBoxLayout(main_widget)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(10)
            
            # Header
            header = QHBoxLayout()
            title = QLabel("🤖 IntelCLaw")
            title.setObjectName("titleLabel")
            header.addWidget(title)
            header.addStretch()
            
            close_btn = QPushButton("✕")
            close_btn.setObjectName("closeButton")
            close_btn.setFixedSize(30, 30)
            close_btn.clicked.connect(self.hide)
            header.addWidget(close_btn)
            
            layout.addLayout(header)
            
            # Chat area
            self.chat_display = QTextEdit()
            self.chat_display.setObjectName("chatDisplay")
            self.chat_display.setReadOnly(True)
            self.chat_display.setPlaceholderText("How can I help you today?")
            layout.addWidget(self.chat_display, stretch=1)
            
            # Input area
            input_layout = QHBoxLayout()
            
            self.input_field = QLineEdit()
            self.input_field.setObjectName("inputField")
            self.input_field.setPlaceholderText("Type your message...")
            self.input_field.returnPressed.connect(self._send_message)
            input_layout.addWidget(self.input_field, stretch=1)
            
            send_btn = QPushButton("Send")
            send_btn.setObjectName("sendButton")
            send_btn.clicked.connect(self._send_message)
            input_layout.addWidget(send_btn)
            
            layout.addLayout(input_layout)
            
            # Status bar
            self.status_label = QLabel("Ready")
            self.status_label.setObjectName("statusLabel")
            layout.addWidget(self.status_label)
        
        def _setup_style(self) -> None:
            """Apply styling."""
            self.setStyleSheet("""
                #mainWidget {
                    background-color: rgba(30, 30, 40, 240);
                    border-radius: 15px;
                    border: 1px solid rgba(100, 100, 120, 150);
                }
                
                #titleLabel {
                    color: white;
                    font-size: 18px;
                    font-weight: bold;
                    padding: 5px;
                }
                
                #closeButton {
                    background-color: rgba(255, 100, 100, 180);
                    border: none;
                    border-radius: 15px;
                    color: white;
                    font-weight: bold;
                }
                
                #closeButton:hover {
                    background-color: rgba(255, 50, 50, 220);
                }
                
                #chatDisplay {
                    background-color: rgba(40, 40, 50, 200);
                    border: 1px solid rgba(80, 80, 100, 150);
                    border-radius: 10px;
                    color: white;
                    font-size: 14px;
                    padding: 10px;
                }
                
                #inputField {
                    background-color: rgba(50, 50, 60, 220);
                    border: 1px solid rgba(100, 100, 120, 150);
                    border-radius: 8px;
                    color: white;
                    font-size: 14px;
                    padding: 10px;
                }
                
                #inputField:focus {
                    border: 1px solid rgba(100, 150, 255, 200);
                }
                
                #sendButton {
                    background-color: rgba(70, 130, 220, 220);
                    border: none;
                    border-radius: 8px;
                    color: white;
                    font-size: 14px;
                    padding: 10px 20px;
                }
                
                #sendButton:hover {
                    background-color: rgba(90, 150, 240, 240);
                }
                
                #statusLabel {
                    color: rgba(150, 150, 170, 200);
                    font-size: 12px;
                }
            """)
        
        def _center_on_screen(self) -> None:
            """Center window on screen."""
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)
        
        def _send_message(self) -> None:
            """Send user message to agent."""
            message = self.input_field.text().strip()
            if not message:
                return
            
            # Display user message
            self._append_message("You", message)
            self.input_field.clear()
            
            # Update status
            self.status_label.setText("Thinking...")
            
            # Process asynchronously
            asyncio.create_task(self._process_message(message))
        
        async def _process_message(self, message: str) -> None:
            """Process message through agent."""
            try:
                response = await self._app.process_user_input(message)
                self.response_received.emit(response)
            except Exception as e:
                self.response_received.emit(f"Error: {str(e)}")
        
        def _display_response(self, response: str) -> None:
            """Display agent response."""
            self._append_message("IntelCLaw", response)
            self.status_label.setText("Ready")
        
        def _append_message(self, sender: str, message: str) -> None:
            """Append a message to the chat display."""
            current = self.chat_display.toPlainText()
            if current:
                current += "\n\n"
            
            formatted = f"**{sender}:**\n{message}"
            self.chat_display.setPlainText(current + formatted)
            
            # Scroll to bottom
            scrollbar = self.chat_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        
        def keyPressEvent(self, event):
            """Handle key press events."""
            if event.key() == Qt.Key.Key_Escape:
                self.hide()
            else:
                super().keyPressEvent(event)
