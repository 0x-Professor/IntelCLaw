"""
Context Builder - Aggregates and filters perception context.

Applies privacy filters and builds unified context for agents.
"""

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from intelclaw.perception.manager import PerceptionContext


class ContextBuilder:
    """
    Builds and filters perception context for agent consumption.
    
    Features:
    - Privacy filtering (PII detection)
    - Sensitive content masking
    - Context aggregation
    - Relevance scoring
    """
    
    # Patterns for PII detection
    PII_PATTERNS = [
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),  # Phone numbers
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Emails
        (r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', '[SSN]'),  # SSN
        (r'\b(?:\d[ -]*?){13,16}\b', '[CARD]'),  # Credit cards
        (r'\b\d{5}(?:[-\s]\d{4})?\b', '[ZIP]'),  # ZIP codes
    ]
    
    # Sensitive window patterns
    SENSITIVE_WINDOWS = [
        "password", "bank", "credit", "login", "signin", "sign in",
        "1password", "lastpass", "bitwarden", "keepass",
        "paypal", "venmo", "cash app",
    ]
    
    def __init__(self, privacy_filter: bool = True):
        """
        Initialize context builder.
        
        Args:
            privacy_filter: Enable privacy filtering
        """
        self._privacy_filter = privacy_filter
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in self.PII_PATTERNS
        ]
    
    def filter_context(self, context: "PerceptionContext") -> "PerceptionContext":
        """
        Apply privacy filters to perception context.
        
        Args:
            context: Raw perception context
            
        Returns:
            Filtered context
        """
        if not self._privacy_filter:
            return context
        
        # Check if window is sensitive
        if self._is_sensitive_window(context.active_window_title):
            logger.debug(f"Sensitive window detected: {context.active_window_title}")
            context.screen_text = None
            context.ui_elements = []
            context.clipboard_content = None
            return context
        
        # Filter screen text
        if context.screen_text:
            context.screen_text = self._mask_pii(context.screen_text)
        
        # Filter clipboard
        if context.clipboard_content:
            context.clipboard_content = self._mask_pii(context.clipboard_content)
        
        return context
    
    def _is_sensitive_window(self, title: Optional[str]) -> bool:
        """Check if window title indicates sensitive content."""
        if not title:
            return False
        
        title_lower = title.lower()
        return any(pattern in title_lower for pattern in self.SENSITIVE_WINDOWS)
    
    def _mask_pii(self, text: str) -> str:
        """Mask personally identifiable information in text."""
        for pattern, replacement in self._compiled_patterns:
            text = pattern.sub(replacement, text)
        return text
    
    def build_agent_context(
        self,
        context: "PerceptionContext",
        user_message: str,
        conversation_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Build context dictionary for agent consumption.
        
        Args:
            context: Perception context
            user_message: Current user message
            conversation_history: Recent messages
            
        Returns:
            Agent-ready context dictionary
        """
        # Apply filters
        filtered_context = self.filter_context(context)
        
        return {
            "user_message": user_message,
            "timestamp": filtered_context.timestamp.isoformat(),
            "active_window": filtered_context.active_window,
            "active_window_title": filtered_context.active_window_title,
            "screen_text": self._truncate(filtered_context.screen_text, 2000),
            "ui_elements_summary": self._summarize_elements(filtered_context.ui_elements),
            "clipboard_available": bool(filtered_context.clipboard_content),
            "recent_messages": conversation_history[-5:],
        }
    
    def _truncate(self, text: Optional[str], max_length: int) -> Optional[str]:
        """Truncate text to max length."""
        if not text:
            return None
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def _summarize_elements(
        self,
        elements: List[Dict[str, Any]],
        max_elements: int = 20
    ) -> List[Dict[str, str]]:
        """Summarize UI elements for agent context."""
        if not elements:
            return []
        
        # Group by type
        by_type: Dict[str, List[str]] = {}
        for elem in elements[:max_elements]:
            elem_type = elem.get("type", "Unknown")
            elem_name = elem.get("name", "")
            if elem_name:
                by_type.setdefault(elem_type, []).append(elem_name)
        
        summary = []
        for elem_type, names in by_type.items():
            summary.append({
                "type": elem_type,
                "count": len(names),
                "examples": names[:3],
            })
        
        return summary
