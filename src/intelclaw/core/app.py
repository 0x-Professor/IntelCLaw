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

from intelclaw.core.events import EventBus
from intelclaw.config.manager import ConfigManager
from intelclaw.agent.orchestrator import AgentOrchestrator
from intelclaw.memory.manager import MemoryManager
from intelclaw.perception.manager import PerceptionManager
from intelclaw.tools.registry import ToolRegistry
from intelclaw.ui.overlay import OverlayWindow
from intelclaw.ui.system_tray import SystemTray
from intelclaw.security.manager import SecurityManager


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
        self.perception: Optional[PerceptionManager] = None
        self.tools: Optional[ToolRegistry] = None
        self.agent: Optional[AgentOrchestrator] = None
        self.overlay: Optional[OverlayWindow] = None
        self.tray: Optional[SystemTray] = None
        
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
        
        # 4. Initialize perception layer
        self.perception = PerceptionManager(self.config, self.event_bus)
        await self.perception.initialize()
        logger.info("Perception layer initialized")
        
        # 5. Initialize tool registry
        self.tools = ToolRegistry(self.config, self.security)
        await self.tools.initialize()
        logger.info("Tool registry initialized")
        
        # 6. Initialize agent orchestrator
        self.agent = AgentOrchestrator(
            config=self.config,
            memory=self.memory,
            tools=self.tools,
            event_bus=self.event_bus,
        )
        await self.agent.initialize()
        logger.info("Agent orchestrator initialized")
        
        # 7. Initialize UI components
        self.tray = SystemTray(self)
        self.overlay = OverlayWindow(self)
        logger.info("UI components initialized")
        
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
        
        if self.memory:
            await self.memory.shutdown()
        
        if self.security:
            await self.security.shutdown()
        
        if self.config:
            await self.config.save()
        
        self._shutdown_event.set()
        logger.success("IntelCLaw shutdown complete")
    
    async def run(self) -> None:
        """Run the main application loop."""
        await self.startup()
        
        # Setup signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_event_loop().add_signal_handler(
                sig,
                lambda: asyncio.create_task(self.shutdown())
            )
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self.perception.run()),
            asyncio.create_task(self.agent.run()),
            asyncio.create_task(self._monitor_health()),
        ]
        
        # Wait for shutdown signal
        await self._shutdown_event.wait()
        
        # Cancel all tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
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
        context = await self.perception.get_context() if self.perception else {}
        
        # Process through agent
        response = await self.agent.process(message, context)
        
        return response
    
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
