"""
Short-Term Memory - Conversation history buffer.

In-memory circular buffer for current session context.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional


@dataclass
class Message:
    """A single message in conversation history."""
    
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ShortTermMemory:
    """
    Short-term conversation memory.
    
    Features:
    - Circular buffer with configurable size
    - Message metadata support
    - Token counting (approximate)
    - Context window management
    """
    
    def __init__(self, max_messages: int = 50):
        """
        Initialize short-term memory.
        
        Args:
            max_messages: Maximum messages to retain
        """
        self._max_messages = max_messages
        self._messages: Deque[Message] = deque(maxlen=max_messages)
        self._session_start = datetime.now()
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to memory.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._messages.append(message)
    
    def get_history(
        self,
        limit: Optional[int] = None,
        roles: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            limit: Maximum messages to return
            roles: Filter by roles
            
        Returns:
            List of message dictionaries
        """
        messages = list(self._messages)
        
        if roles:
            messages = [m for m in messages if m.role in roles]
        
        if limit:
            messages = messages[-limit:]
        
        return [m.to_dict() for m in messages]
    
    def get_last_n_messages(self, n: int) -> List[Dict[str, Any]]:
        """Get the last N messages."""
        return self.get_history(limit=n)
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the most recent user message."""
        for message in reversed(self._messages):
            if message.role == "user":
                return message.content
        return None
    
    def get_last_assistant_message(self) -> Optional[str]:
        """Get the most recent assistant message."""
        for message in reversed(self._messages):
            if message.role == "assistant":
                return message.content
        return None
    
    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self._session_start = datetime.now()
    
    def estimate_tokens(self) -> int:
        """Estimate total tokens in memory (approximate)."""
        total_chars = sum(len(m.content) for m in self._messages)
        # Rough estimate: 4 characters per token
        return total_chars // 4
    
    def trim_to_token_limit(self, max_tokens: int) -> None:
        """Remove oldest messages to fit within token limit."""
        while self.estimate_tokens() > max_tokens and len(self._messages) > 1:
            self._messages.popleft()
    
    def get_context_for_prompt(
        self,
        max_messages: int = 10,
        max_tokens: int = 4000
    ) -> List[Dict[str, str]]:
        """
        Get formatted context for LLM prompt.
        
        Args:
            max_messages: Maximum messages to include
            max_tokens: Maximum estimated tokens
            
        Returns:
            List of {"role": str, "content": str}
        """
        messages = self.get_last_n_messages(max_messages)
        
        # Trim to token limit
        total_chars = sum(len(m["content"]) for m in messages)
        while total_chars > max_tokens * 4 and len(messages) > 1:
            messages.pop(0)
            total_chars = sum(len(m["content"]) for m in messages)
        
        return [{"role": m["role"], "content": m["content"]} for m in messages]
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Simple keyword search in message history.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            Matching messages
        """
        query_lower = query.lower()
        matches = [
            m.to_dict()
            for m in self._messages
            if query_lower in m.content.lower()
        ]
        return matches[-limit:]
    
    @property
    def message_count(self) -> int:
        """Get total message count."""
        return len(self._messages)
    
    @property
    def session_duration_seconds(self) -> float:
        """Get session duration in seconds."""
        return (datetime.now() - self._session_start).total_seconds()
