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
        QTextEdit, QLineEdit, QPushButton, QLabel, QScrollArea, QFrame, QComboBox,
        QDialog, QCheckBox
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
    class SkillsDialog(QDialog):
        def __init__(self, app: "IntelCLawApp", parent: Optional[Any] = None):
            super().__init__(parent)
            self._app = app
            self._rows: dict[str, QCheckBox] = {}

            self.setWindowTitle("Skills")
            self.setModal(False)
            self.setMinimumSize(520, 420)

            root = QVBoxLayout(self)
            root.setContentsMargins(14, 14, 14, 14)
            root.setSpacing(10)

            header = QHBoxLayout()
            header.setSpacing(8)
            title = QLabel("Skills")
            title.setObjectName("skillsDialogTitle")
            header.addWidget(title)
            header.addStretch()

            refresh_btn = QPushButton("Refresh")
            refresh_btn.setObjectName("skillsRefreshButton")
            refresh_btn.clicked.connect(lambda: asyncio.create_task(self.refresh()))
            header.addWidget(refresh_btn)
            root.addLayout(header)

            self._scroll = QScrollArea()
            self._scroll.setWidgetResizable(True)
            self._scroll.setObjectName("skillsScroll")

            self._content = QWidget()
            self._content_layout = QVBoxLayout(self._content)
            self._content_layout.setContentsMargins(0, 0, 0, 0)
            self._content_layout.setSpacing(8)
            self._scroll.setWidget(self._content)
            root.addWidget(self._scroll, stretch=1)

            self._status = QLabel("")
            self._status.setObjectName("skillsStatusLabel")
            root.addWidget(self._status)

            self.setStyleSheet("""
                QDialog {
                    background-color: rgba(30, 30, 40, 250);
                    border: 1px solid rgba(100, 100, 120, 150);
                    border-radius: 12px;
                }
                #skillsDialogTitle {
                    color: white;
                    font-size: 14px;
                    font-weight: bold;
                }
                #skillsRefreshButton {
                    background-color: rgba(50, 50, 60, 220);
                    border: 1px solid rgba(100, 100, 120, 150);
                    border-radius: 8px;
                    color: white;
                    font-size: 12px;
                    padding: 6px 10px;
                }
                #skillsRefreshButton:hover {
                    border: 1px solid rgba(100, 150, 255, 200);
                }
                #skillsStatusLabel {
                    color: rgba(150, 150, 170, 200);
                    font-size: 11px;
                }
                QScrollArea {
                    border: none;
                    background: transparent;
                }
                QCheckBox {
                    color: white;
                    font-size: 12px;
                }
            """)

        async def refresh(self) -> None:
            skills = []
            try:
                if getattr(self._app, "skills", None):
                    skills = await self._app.skills.list_skills()
            except Exception:
                skills = []

            # Clear existing content
            for i in reversed(range(self._content_layout.count())):
                item = self._content_layout.itemAt(i)
                w = item.widget() if item else None
                if w is not None:
                    w.setParent(None)

            self._rows = {}

            if not skills:
                empty = QLabel("No skills found.")
                empty.setObjectName("skillsEmptyLabel")
                empty.setStyleSheet("color: rgba(150, 150, 170, 200); font-size: 12px;")
                self._content_layout.addWidget(empty)
                self._status.setText("")
                return

            for s in skills or []:
                sid = str(s.get("id") or "").strip()
                if not sid:
                    continue
                name = str(s.get("name") or sid).strip()
                enabled = bool(s.get("enabled", False))

                cb = QCheckBox(f"{name}  ({sid})")
                cb.setChecked(enabled)
                cb.toggled.connect(lambda checked, _sid=sid: asyncio.create_task(self._set_enabled(_sid, checked)))
                self._rows[sid] = cb
                self._content_layout.addWidget(cb)

            self._status.setText("")

        async def _set_enabled(self, skill_id: str, enabled: bool) -> None:
            skills_mgr = getattr(self._app, "skills", None)
            if not skills_mgr:
                self._status.setText("Skills subsystem unavailable.")
                return

            try:
                res = await (skills_mgr.enable(skill_id) if enabled else skills_mgr.disable(skill_id))
            except Exception as e:
                res = {"success": False, "error": str(e)}

            if not res or not res.get("success"):
                err = res.get("error") if isinstance(res, dict) else "Unknown error"
                self._status.setText(str(err))
                cb = self._rows.get(skill_id)
                if cb is not None:
                    cb.blockSignals(True)
                    cb.setChecked(not enabled)
                    cb.blockSignals(False)
                return

            # Sync MCP tools quickly (best-effort)
            if getattr(self._app, "tools", None):
                try:
                    await self._app.tools.reload_mcp_tools()
                except Exception:
                    pass

            self._status.setText(f"{'Enabled' if enabled else 'Disabled'}: {skill_id}")

    class InboxDialog(QDialog):
        def __init__(self, app: "IntelCLawApp", parent: Optional[Any] = None):
            super().__init__(parent)
            self._app = app
            self._session_id: Optional[str] = None

            self.setWindowTitle("Inbox")
            self.setModal(False)
            self.setMinimumSize(640, 480)

            root = QVBoxLayout(self)
            root.setContentsMargins(14, 14, 14, 14)
            root.setSpacing(10)

            header = QHBoxLayout()
            header.setSpacing(8)
            title = QLabel("Inbox")
            title.setObjectName("inboxDialogTitle")
            header.addWidget(title)
            header.addStretch()

            refresh_btn = QPushButton("Refresh")
            refresh_btn.setObjectName("inboxRefreshButton")
            refresh_btn.clicked.connect(lambda: asyncio.create_task(self.refresh()))
            header.addWidget(refresh_btn)
            root.addLayout(header)

            self._text = QTextEdit()
            self._text.setObjectName("inboxText")
            self._text.setReadOnly(True)
            root.addWidget(self._text, stretch=1)

            self._status = QLabel("")
            self._status.setObjectName("inboxStatusLabel")
            root.addWidget(self._status)

            self.setStyleSheet("""
                QDialog {
                    background-color: rgba(30, 30, 40, 250);
                    border: 1px solid rgba(100, 100, 120, 150);
                    border-radius: 12px;
                }
                #inboxDialogTitle {
                    color: white;
                    font-size: 14px;
                    font-weight: bold;
                }
                #inboxRefreshButton {
                    background-color: rgba(50, 50, 60, 220);
                    border: 1px solid rgba(100, 100, 120, 150);
                    border-radius: 8px;
                    color: white;
                    font-size: 12px;
                    padding: 6px 10px;
                }
                #inboxRefreshButton:hover {
                    border: 1px solid rgba(100, 150, 255, 200);
                }
                #inboxText {
                    background-color: rgba(40, 40, 50, 200);
                    border: 1px solid rgba(80, 80, 100, 150);
                    border-radius: 10px;
                    color: white;
                    font-size: 12px;
                    padding: 10px;
                }
                #inboxStatusLabel {
                    color: rgba(150, 150, 170, 200);
                    font-size: 11px;
                }
            """)

        def set_session_id(self, session_id: str) -> None:
            self._session_id = str(session_id or "").strip() or None

        async def refresh(self) -> None:
            mailbox = getattr(self._app, "mailbox", None)
            sid = str(self._session_id or "").strip()
            if not mailbox or not sid:
                self._text.setPlainText("No mailbox data available.")
                return

            try:
                msgs = await mailbox.list_messages(sid, limit=200)
            except Exception as e:
                self._status.setText(f"Failed to load inbox: {e}")
                return

            parts = []
            for m in msgs or []:
                ts = str(m.get("ts") or "")
                kind = str(m.get("kind") or "info")
                from_agent = str(m.get("from_agent") or "")
                title = str(m.get("title") or "")
                body = str(m.get("body") or "")
                parts.append(f"[{ts}] {kind} - {from_agent}\n{title}\n{body}".strip())

            self._text.setPlainText("\n\n---\n\n".join(parts))
            sb = self._text.verticalScrollBar()
            sb.setValue(sb.maximum())
            self._status.setText("")

        def append_message(self, msg: dict) -> None:
            try:
                ts = str(msg.get("ts") or "")
                kind = str(msg.get("kind") or "info")
                from_agent = str(msg.get("from_agent") or "")
                title = str(msg.get("title") or "")
                body = str(msg.get("body") or "")
                chunk = f"[{ts}] {kind} - {from_agent}\n{title}\n{body}".strip()
                cur = self._text.toPlainText().strip()
                cur = (cur + "\n\n---\n\n" + chunk).strip() if cur else chunk
                self._text.setPlainText(cur)
                sb = self._text.verticalScrollBar()
                sb.setValue(sb.maximum())
            except Exception:
                return

    class TransparentOverlay(QMainWindow):
        """PyQt6 transparent overlay window."""
        
        response_received = pyqtSignal(str, str)
        mailbox_message_received = pyqtSignal(object)
        
        def __init__(self, app: "IntelCLawApp"):
            super().__init__()
            self._app = app
            self._active_session_id: Optional[str] = None
            self._skills_dialog: Optional[SkillsDialog] = None
            self._inbox_dialog: Optional[InboxDialog] = None
            self._setup_ui()
            self._setup_style()
            
            # Connect signal
            self.response_received.connect(self._display_response)
            self.mailbox_message_received.connect(self._on_mailbox_message)

            # Subscribe to mailbox events (best-effort)
            try:
                if getattr(self._app, "event_bus", None):
                    asyncio.create_task(self._subscribe_mailbox_events())
            except Exception:
                pass

            # Load sessions on startup (best-effort)
            try:
                asyncio.create_task(self._refresh_sessions())
            except Exception:
                pass
        
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

            skills_btn = QPushButton("Skills")
            skills_btn.setObjectName("skillsButton")
            skills_btn.setFixedHeight(30)
            skills_btn.clicked.connect(lambda: asyncio.create_task(self._open_skills_dialog()))
            header.addWidget(skills_btn)

            inbox_btn = QPushButton("Inbox")
            inbox_btn.setObjectName("inboxButton")
            inbox_btn.setFixedHeight(30)
            inbox_btn.clicked.connect(lambda: asyncio.create_task(self._open_inbox_dialog()))
            header.addWidget(inbox_btn)

            close_btn = QPushButton("✕")
            close_btn.setObjectName("closeButton")
            close_btn.setFixedSize(30, 30)
            close_btn.clicked.connect(self.hide)
            header.addWidget(close_btn)
            
            layout.addLayout(header)

            # Sessions row
            session_row = QHBoxLayout()
            session_row.setSpacing(8)

            session_label = QLabel("Session")
            session_label.setObjectName("sessionLabel")
            session_row.addWidget(session_label)

            self.session_combo = QComboBox()
            self.session_combo.setObjectName("sessionCombo")
            self.session_combo.setMinimumWidth(280)
            self.session_combo.setEnabled(False)
            self.session_combo.currentIndexChanged.connect(self._on_session_selection_changed)
            session_row.addWidget(self.session_combo, stretch=1)

            refresh_btn = QPushButton("⟳")
            refresh_btn.setObjectName("refreshSessionsButton")
            refresh_btn.setFixedSize(34, 28)
            refresh_btn.clicked.connect(lambda: asyncio.create_task(self._refresh_sessions()))
            session_row.addWidget(refresh_btn)

            new_btn = QPushButton("New")
            new_btn.setObjectName("newSessionButton")
            new_btn.setFixedHeight(28)
            new_btn.clicked.connect(lambda: asyncio.create_task(self._create_new_session()))
            session_row.addWidget(new_btn)

            layout.addLayout(session_row)
            
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

        def _get_session_store(self):
            try:
                memory = getattr(self._app, "memory", None)
                store = getattr(memory, "session_store", None) if memory else None
                if store and getattr(store, "is_enabled", False):
                    return store
            except Exception:
                pass
            return None

        def _selected_session_id(self) -> str:
            try:
                if getattr(self, "session_combo", None):
                    sid = self.session_combo.currentData()
                    if sid:
                        return str(sid)
            except Exception:
                pass
            return str(self._active_session_id or "overlay")

        def _on_session_selection_changed(self, index: int) -> None:
            try:
                if not getattr(self, "session_combo", None):
                    return
                sid = self.session_combo.itemData(index)
                if not sid:
                    return
                asyncio.create_task(self._switch_to_session(str(sid)))
            except Exception:
                pass

        async def _refresh_sessions(self, preferred_session_id: Optional[str] = None) -> None:
            store = self._get_session_store()
            if not getattr(self, "session_combo", None):
                return

            if not store:
                self.session_combo.blockSignals(True)
                try:
                    self.session_combo.clear()
                    self.session_combo.addItem("overlay", "overlay")
                    self.session_combo.setCurrentIndex(0)
                    self._active_session_id = "overlay"
                finally:
                    self.session_combo.blockSignals(False)
                self.session_combo.setEnabled(False)
                return

            # List sessions (most-recent first)
            try:
                sessions = await store.list_sessions(limit=200, offset=0)
            except Exception:
                sessions = []

            if not sessions:
                try:
                    sid = await store.create_session(title="New Session")
                    sessions = await store.list_sessions(limit=200, offset=0)
                    preferred_session_id = preferred_session_id or sid
                except Exception:
                    sessions = []

            preferred = (preferred_session_id or self._active_session_id or "").strip() or None
            if preferred is None and sessions:
                preferred = str(sessions[0].get("session_id") or "").strip() or None

            self.session_combo.blockSignals(True)
            try:
                self.session_combo.clear()
                for s in sessions or []:
                    sid = str(s.get("session_id") or "").strip()
                    if not sid:
                        continue
                    title = str(s.get("title") or "").strip() or sid
                    count = s.get("message_count")
                    try:
                        count_i = int(count) if count is not None else 0
                    except Exception:
                        count_i = 0
                    label = f"{title} ({count_i} msg{'s' if count_i != 1 else ''})"
                    self.session_combo.addItem(label, sid)

                # Select preferred (if present)
                if preferred:
                    idx = self.session_combo.findData(preferred)
                    if idx >= 0:
                        self.session_combo.setCurrentIndex(idx)
                if self.session_combo.currentIndex() < 0 and self.session_combo.count() > 0:
                    self.session_combo.setCurrentIndex(0)

                selected = self._selected_session_id()
                self._active_session_id = selected
            finally:
                self.session_combo.blockSignals(False)

            self.session_combo.setEnabled(True)

            # Load messages for selected session.
            await self._load_session_messages(self._selected_session_id())

        async def _create_new_session(self) -> None:
            store = self._get_session_store()
            if not store:
                return
            try:
                sid = await store.create_session(title="New Session")
            except Exception:
                return
            await self._refresh_sessions(preferred_session_id=sid)

        async def _switch_to_session(self, session_id: str) -> None:
            sid = str(session_id or "").strip()
            if not sid:
                return
            self._active_session_id = sid
            await self._load_session_messages(sid)

            if self._inbox_dialog is not None:
                try:
                    self._inbox_dialog.set_session_id(sid)
                    if self._inbox_dialog.isVisible():
                        await self._inbox_dialog.refresh()
                except Exception:
                    pass

        async def _load_session_messages(self, session_id: str) -> None:
            store = self._get_session_store()
            sid = str(session_id or "").strip()
            if not sid:
                sid = "overlay"

            if not store:
                self.chat_display.clear()
                self.chat_display.setPlaceholderText("Session storage not available.")
                return

            try:
                rows = await store.get_messages(sid, limit=None)
            except Exception:
                rows = []

            parts = []
            for r in rows or []:
                role = str(r.get("role") or "").lower()
                content = str(r.get("content") or "")
                if not content:
                    continue
                sender = "IntelCLaw"
                if role == "user":
                    sender = "You"
                elif role == "assistant":
                    sender = "IntelCLaw"
                elif role == "system":
                    sender = "System"
                parts.append(f"**{sender}:**\n{content}")

            self.chat_display.setPlainText("\n\n".join(parts))
            scrollbar = self.chat_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        
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

                #skillsButton,
                #inboxButton {
                    background-color: rgba(50, 50, 60, 220);
                    border: 1px solid rgba(100, 100, 120, 150);
                    border-radius: 8px;
                    color: white;
                    font-size: 12px;
                    padding: 6px 10px;
                }

                #skillsButton:hover,
                #inboxButton:hover {
                    border: 1px solid rgba(100, 150, 255, 200);
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

                #sessionLabel {
                    color: rgba(150, 150, 170, 200);
                    font-size: 12px;
                    padding: 0 4px;
                }

                #sessionCombo {
                    background-color: rgba(50, 50, 60, 220);
                    border: 1px solid rgba(100, 100, 120, 150);
                    border-radius: 8px;
                    color: white;
                    font-size: 12px;
                    padding: 6px 8px;
                }

                #newSessionButton,
                #refreshSessionsButton {
                    background-color: rgba(50, 50, 60, 220);
                    border: 1px solid rgba(100, 100, 120, 150);
                    border-radius: 8px;
                    color: white;
                    font-size: 12px;
                    padding: 6px 10px;
                }

                #newSessionButton:hover,
                #refreshSessionsButton:hover {
                    border: 1px solid rgba(100, 150, 255, 200);
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

            session_id = self._selected_session_id()
            
            # Display user message
            self._append_message("You", message)
            self.input_field.clear()
            
            # Update status
            self.status_label.setText("Thinking...")
            
            # Process asynchronously
            asyncio.create_task(self._process_message(message, session_id))
        
        async def _process_message(self, message: str, session_id: str) -> None:
            """Process message through agent."""
            try:
                response = await self._app.process_user_input(message, session_id=session_id)
                self.response_received.emit(response, session_id)
            except Exception as e:
                self.response_received.emit(f"Error: {str(e)}", session_id)

        async def _open_skills_dialog(self) -> None:
            if self._skills_dialog is None:
                self._skills_dialog = SkillsDialog(self._app, parent=self)
            try:
                await self._skills_dialog.refresh()
            except Exception:
                pass
            self._skills_dialog.show()
            self._skills_dialog.raise_()
            self._skills_dialog.activateWindow()

        async def _open_inbox_dialog(self) -> None:
            if self._inbox_dialog is None:
                self._inbox_dialog = InboxDialog(self._app, parent=self)
            try:
                self._inbox_dialog.set_session_id(self._selected_session_id())
                await self._inbox_dialog.refresh()
            except Exception:
                pass
            self._inbox_dialog.show()
            self._inbox_dialog.raise_()
            self._inbox_dialog.activateWindow()

        async def _subscribe_mailbox_events(self) -> None:
            bus = getattr(self._app, "event_bus", None)
            if not bus:
                return

            async def handler(event):
                try:
                    msg = (event.data or {}).get("message")
                    if msg:
                        self.mailbox_message_received.emit(msg)
                except Exception:
                    return

            try:
                await bus.subscribe("mailbox.message", handler)
            except Exception:
                return

        def _on_mailbox_message(self, msg: Any) -> None:
            try:
                if not isinstance(msg, dict):
                    return
                if not self._inbox_dialog or not self._inbox_dialog.isVisible():
                    return
                current_sid = self._selected_session_id()
                msg_sid = str(msg.get("session_id") or "").strip() or current_sid
                if msg_sid != current_sid:
                    return
                self._inbox_dialog.append_message(msg)
            except Exception:
                return
        
        def _display_response(self, response: str, session_id: str) -> None:
            """Display agent response."""
            # If the user switched sessions while we were processing, don't mix streams.
            if str(session_id or "").strip() == self._selected_session_id():
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
