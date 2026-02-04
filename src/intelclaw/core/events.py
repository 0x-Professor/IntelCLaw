"""
Event bus for inter-component communication.

Implements a pub/sub pattern for decoupled component interaction.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from uuid import uuid4

from loguru import logger


@dataclass
class Event:
    """Represents an event in the system."""
    
    name: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid4()))
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }


# Type alias for event handlers
EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """
    Asynchronous event bus for component communication.
    
    Features:
    - Async event handling
    - Wildcard subscriptions
    - Event history
    - Handler priority
    """
    
    def __init__(self, max_history: int = 1000):
        """Initialize the event bus."""
        self._handlers: Dict[str, List[tuple[int, EventHandler]]] = defaultdict(list)
        self._wildcard_handlers: List[tuple[int, EventHandler]] = []
        self._history: List[Event] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()
        
    async def subscribe(
        self,
        event_name: str,
        handler: EventHandler,
        priority: int = 0
    ) -> None:
        """
        Subscribe to an event.
        
        Args:
            event_name: Event name or '*' for all events
            handler: Async function to handle the event
            priority: Higher priority handlers run first
        """
        async with self._lock:
            if event_name == "*":
                self._wildcard_handlers.append((priority, handler))
                self._wildcard_handlers.sort(key=lambda x: -x[0])
            else:
                self._handlers[event_name].append((priority, handler))
                self._handlers[event_name].sort(key=lambda x: -x[0])
        
        logger.debug(f"Handler subscribed to '{event_name}'")
    
    async def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event."""
        async with self._lock:
            if event_name == "*":
                self._wildcard_handlers = [
                    (p, h) for p, h in self._wildcard_handlers if h != handler
                ]
            else:
                self._handlers[event_name] = [
                    (p, h) for p, h in self._handlers[event_name] if h != handler
                ]
    
    async def emit(
        self,
        event_name: str,
        data: Dict[str, Any],
        source: Optional[str] = None,
        wait: bool = False
    ) -> Event:
        """
        Emit an event to all subscribers.
        
        Args:
            event_name: Name of the event
            data: Event data payload
            source: Source component name
            wait: If True, wait for all handlers to complete
            
        Returns:
            The emitted event
        """
        event = Event(name=event_name, data=data, source=source)
        
        # Store in history
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        
        # Gather all handlers
        handlers = [h for _, h in self._handlers.get(event_name, [])]
        handlers.extend([h for _, h in self._wildcard_handlers])
        
        if not handlers:
            return event
        
        # Execute handlers
        async def run_handler(handler: EventHandler) -> None:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Event handler error for '{event_name}': {e}")
        
        if wait:
            await asyncio.gather(*[run_handler(h) for h in handlers])
        else:
            for handler in handlers:
                asyncio.create_task(run_handler(handler))
        
        return event
    
    def get_history(
        self,
        event_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_name: Filter by event name (optional)
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        events = self._history
        if event_name:
            events = [e for e in events if e.name == event_name]
        return events[-limit:]
    
    async def clear_history(self) -> None:
        """Clear event history."""
        async with self._lock:
            self._history.clear()


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus
