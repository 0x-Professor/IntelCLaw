"""
Intent Router - Routes requests to appropriate sub-agents.

Uses semantic similarity and keyword matching to determine
the best agent for a given request.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from intelclaw.agent.base import BaseAgent, AgentContext


@dataclass
class RoutingDecision:
    """Result of intent routing."""
    agent_name: str
    confidence: float
    reasoning: str


class IntentRouter:
    """
    Routes user requests to the most appropriate sub-agent.
    
    Uses a combination of:
    - Keyword matching
    - Semantic similarity (when available)
    - Agent self-assessment
    """
    
    def __init__(self):
        """Initialize the router."""
        self._agents: Dict[str, "BaseAgent"] = {}
        
        # Keyword mappings for quick routing
        self._keyword_mappings: Dict[str, List[str]] = {
            "research": [
                "search", "find", "look up", "research", "what is", "who is",
                "explain", "tell me about", "information", "news", "article"
            ],
            "coding": [
                "code", "program", "function", "debug", "error", "python",
                "javascript", "script", "implement", "fix bug", "refactor"
            ],
            "task": [
                "remind", "schedule", "todo", "task", "calendar", "meeting",
                "email", "send", "notify", "deadline", "appointment"
            ],
            "system": [
                "open", "close", "launch", "file", "folder", "window",
                "screenshot", "clipboard", "copy", "paste", "system"
            ],
        }
    
    def register_agent(self, name: str, agent: "BaseAgent") -> None:
        """Register a sub-agent."""
        self._agents[name] = agent
        logger.debug(f"Registered agent: {name}")
    
    async def route(self, context: "AgentContext") -> Optional[RoutingDecision]:
        """
        Route a request to the appropriate agent.
        
        Args:
            context: The request context
            
        Returns:
            Routing decision or None if no specific agent matches
        """
        message_lower = context.user_message.lower()
        
        # Score each agent
        scores: Dict[str, float] = {}
        
        # 1. Keyword matching
        for agent_name, keywords in self._keyword_mappings.items():
            if agent_name not in self._agents:
                continue
            
            keyword_score = sum(
                1.0 for kw in keywords if kw in message_lower
            ) / len(keywords)
            scores[agent_name] = keyword_score * 0.5  # Weight: 50%
        
        # 2. Agent self-assessment
        for agent_name, agent in self._agents.items():
            try:
                confidence = await agent.can_handle(context)
                scores[agent_name] = scores.get(agent_name, 0) + confidence * 0.5
            except Exception as e:
                logger.warning(f"Agent {agent_name} can_handle failed: {e}")
        
        if not scores:
            return None
        
        # Find best match
        best_agent = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_agent]
        
        if best_score < 0.3:  # Minimum threshold
            return None
        
        return RoutingDecision(
            agent_name=best_agent,
            confidence=best_score,
            reasoning=f"Matched based on intent analysis (score: {best_score:.2f})",
        )
    
    def get_agent(self, name: str) -> Optional["BaseAgent"]:
        """Get an agent by name."""
        return self._agents.get(name)
    
    @property
    def available_agents(self) -> List[str]:
        """List available agent names."""
        return list(self._agents.keys())
