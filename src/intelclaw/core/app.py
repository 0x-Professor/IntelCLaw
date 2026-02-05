"""
Main application entry point for IntelCLaw.

This module orchestrates all components and manages the application lifecycle.
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional

from loguru import logger

# Qt imports for event loop integration
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    import qasync
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    qasync = None

from intelclaw.core.events import EventBus
from intelclaw.config.manager import ConfigManager
from intelclaw.agent.orchestrator import AgentOrchestrator
from intelclaw.agent.self_improvement import SelfImprovement
from intelclaw.memory.manager import MemoryManager
from intelclaw.memory.data_store import DataStore
from intelclaw.perception.manager import PerceptionManager
from intelclaw.tools.registry import ToolRegistry
from intelclaw.ui.overlay import OverlayWindow
from intelclaw.ui.system_tray import SystemTray
from intelclaw.security.manager import SecurityManager
from intelclaw.integrations.copilot import CopilotIntegration, ModelManager


class IntelCLawApp:
    """
    Main application class that coordinates all IntelCLaw components.
    
    This implements a modular, event-driven architecture with:
    - Agent orchestration for task delegation
    - Multi-tier memory system
    - Screen perception and monitoring
    - Transparent overlay UI
    - MCP-based tool ecosystem
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the application with optional config path."""
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Core components (initialized in startup)
        self.event_bus: EventBus = EventBus()
        self.config: Optional[ConfigManager] = None
        self.security: Optional[SecurityManager] = None
        self.memory: Optional[MemoryManager] = None
        self.data_store: Optional[DataStore] = None
        self.perception: Optional[PerceptionManager] = None
        self.tools: Optional[ToolRegistry] = None
        self.agent: Optional[AgentOrchestrator] = None
        self.overlay: Optional[OverlayWindow] = None
        self.tray: Optional[SystemTray] = None
        
        # Autonomous capabilities
        self.self_improvement: Optional[SelfImprovement] = None
        self.copilot: Optional[CopilotIntegration] = None
        self.model_manager: Optional[ModelManager] = None
        
        self._config_path = config_path
        
        logger.info("IntelCLaw application instance created")
    
    async def startup(self) -> None:
        """Initialize all components in the correct order."""
        logger.info("Starting IntelCLaw...")
        
        # 1. Load configuration
        self.config = ConfigManager(self._config_path)
        await self.config.load()
        logger.info("Configuration loaded")
        
        # 2. Initialize security manager
        self.security = SecurityManager(self.config)
        await self.security.initialize()
        logger.info("Security manager initialized")
        
        # 3. Initialize memory system
        self.memory = MemoryManager(self.config, self.event_bus)
        await self.memory.initialize()
        logger.info("Memory system initialized")
        
        # 3b. Initialize data store (for contacts, notes, etc.)
        self.data_store = DataStore(self.config.get("memory.working_db_path", "data/datastore.db"))
        await self.data_store.initialize()
        logger.info("Data store initialized")
        
        # 4. Initialize perception layer
        self.perception = PerceptionManager(self.config, self.event_bus)
        await self.perception.initialize()
        logger.info("Perception layer initialized")
        
        # 5. Initialize tool registry
        self.tools = ToolRegistry(self.config, self.security)
        await self.tools.initialize()
        logger.info("Tool registry initialized")
        
        # 5b. Initialize GitHub Copilot integration
        self.copilot = CopilotIntegration(self.config)
        await self.copilot.initialize()
        self.model_manager = ModelManager(self.config)
        await self.model_manager.initialize()
        logger.info("Copilot integration initialized")
        
        # 6. Initialize agent orchestrator
        self.agent = AgentOrchestrator(
            config=self.config,
            memory=self.memory,
            tools=self.tools,
            event_bus=self.event_bus,
        )
        await self.agent.initialize()
        logger.info("Agent orchestrator initialized")
        
        # 6b. Initialize self-improvement module
        self.self_improvement = SelfImprovement(self.config)
        await self.self_improvement.initialize(self.agent._llm if hasattr(self.agent, '_llm') else None)
        logger.info("Self-improvement module initialized")
        
        # 7. Initialize UI components (skip in web mode)
        import os
        if not os.environ.get("INTELCLAW_NO_QT"):
            self.tray = SystemTray(self)
            self.overlay = OverlayWindow(self)
            logger.info("UI components initialized")
        else:
            logger.info("Running in web mode - Qt UI disabled")
        
        self._running = True
        logger.success("IntelCLaw started successfully!")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        logger.info("Shutting down IntelCLaw...")
        self._running = False
        
        # Shutdown in reverse order
        if self.overlay:
            await self.overlay.close()
        
        if self.tray:
            self.tray.stop()
        
        if self.agent:
            await self.agent.shutdown()
        
        if self.tools:
            await self.tools.shutdown()
        
        if self.perception:
            await self.perception.shutdown()
        
        if self.data_store:
            await self.data_store.close()
        
        if self.memory:
            await self.memory.shutdown()
        
        if self.security:
            await self.security.shutdown()
        
        if self.config:
            await self.config.save()
        
        self._shutdown_event.set()
        logger.success("IntelCLaw shutdown complete")
    
    async def run(self) -> None:
        """Run the main application loop with Qt integration."""
        
        logger.info("Starting main event loop... Press Ctrl+Shift+Space to open overlay")
        
        # Start background tasks
        async_tasks = []
        
        if self.perception:
            async_tasks.append(asyncio.create_task(self.perception.run()))
        if self.agent:
            async_tasks.append(asyncio.create_task(self.agent.run()))
        async_tasks.append(asyncio.create_task(self._monitor_health()))
        
        try:
            # Wait for shutdown signal
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            # Cleanup
            for task in async_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            await self.shutdown()
    
    async def _monitor_health(self) -> None:
        """Monitor system health and resource usage."""
        while self._running:
            try:
                # Emit health metrics
                await self.event_bus.emit("health.check", {
                    "memory_usage": self.memory.get_stats() if self.memory else {},
                    "perception_active": self.perception.is_active if self.perception else False,
                    "agent_status": self.agent.status if self.agent else "unknown",
                })
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def process_user_input(self, message: str) -> str:
        """
        Process user input through the agent.
        
        Args:
            message: User's natural language input
            
        Returns:
            Agent's response
        """
        if not self.agent:
            return "Agent not initialized"
        
        # Get current context from perception
        perception_context = await self.perception.get_context() if self.perception else {}
        
        # Build AgentContext (required by orchestrator)
        from intelclaw.agent.base import AgentContext
        
        screen_context = None
        if perception_context:
            screen_context = {
                "text": perception_context.get("screen_text"),
                "ui_elements": perception_context.get("ui_elements"),
                "active_window_title": perception_context.get("active_window_title"),
            }
        
        conversation_history = (
            self.memory.get_conversation_history(limit=10) if self.memory else []
        )
        
        agent_context = AgentContext(
            user_message=message,
            conversation_history=conversation_history,
            screen_context=screen_context,
            active_window=perception_context.get("active_window") if perception_context else None,
            clipboard_content=perception_context.get("clipboard_content") if perception_context else None,
            user_preferences={},
        )
        
        # Process through agent
        response = await self.agent.process(agent_context)
        
        return response.answer if response else "I couldn't process that request."
    
    def toggle_overlay(self) -> None:
        """Toggle the overlay window visibility."""
        if self.overlay:
            self.overlay.toggle()
    
    @property
    def is_running(self) -> bool:
        """Check if the application is running."""
        return self._running


@asynccontextmanager
async def create_app(config_path: Optional[str] = None):
    """Context manager for creating and running the app."""
    app = IntelCLawApp(config_path)
    try:
        await app.startup()
        yield app
    finally:
        await app.shutdown()
