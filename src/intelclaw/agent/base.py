"""
Base agent class for all IntelCLaw agents.

Provides common functionality for REACT and specialized agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field
from loguru import logger

if TYPE_CHECKING:
    from intelclaw.memory.manager import MemoryManager
    from intelclaw.tools.registry import ToolRegistry


class AgentStatus(str, Enum):
    """Agent operational status."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"


class AgentThought(BaseModel):
    """Represents a single thought/reasoning step."""
    
    step: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentResponse(BaseModel):
    """Complete response from an agent."""
    
    answer: str
    thoughts: List[AgentThought] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    tokens_used: int = 0
    latency_ms: float = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class AgentContext:
    """Context passed to agent during execution."""
    
    user_message: str
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    screen_context: Optional[Dict[str, Any]] = None
    active_window: Optional[str] = None
    clipboard_content: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_facts: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class BaseAgent(ABC):
    """
    Abstract base class for all IntelCLaw agents.
    
    Implements the REACT pattern:
    - Reason: Analyze the task and plan
    - Act: Execute tools or delegate
    - Observe: Process results
    - Reflect: Determine next steps or conclude
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        memory: Optional["MemoryManager"] = None,
        tools: Optional["ToolRegistry"] = None,
    ):
        """
        Initialize base agent.
        
        Args:
            name: Agent identifier
            description: What this agent does
            memory: Memory manager for context
            tools: Available tools registry
        """
        self.name = name
        self.description = description
        self.memory = memory
        self.tools = tools
        self.status = AgentStatus.IDLE
        self._current_thoughts: List[AgentThought] = []
        
        logger.debug(f"Agent '{name}' initialized")
    
    @abstractmethod
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process a request through the agent.
        
        Args:
            context: Full context for the request
            
        Returns:
            Complete agent response
        """
        pass
    
    @abstractmethod
    async def can_handle(self, context: AgentContext) -> float:
        """
        Determine if this agent can handle the request.
        
        Args:
            context: Request context
            
        Returns:
            Confidence score 0.0 to 1.0
        """
        pass
    
    async def think(self, thought: str, step: int) -> AgentThought:
        """
        Record a thinking step.
        
        Args:
            thought: The reasoning text
            step: Step number in chain
            
        Returns:
            AgentThought object
        """
        agent_thought = AgentThought(step=step, thought=thought)
        self._current_thoughts.append(agent_thought)
        logger.debug(f"[{self.name}] Think #{step}: {thought[:100]}...")
        return agent_thought
    
    async def act(
        self,
        action: str,
        action_input: Dict[str, Any],
        step: int
    ) -> AgentThought:
        """
        Record an action step.
        
        Args:
            action: Tool/action name
            action_input: Input parameters
            step: Step number
            
        Returns:
            AgentThought with action details
        """
        agent_thought = AgentThought(
            step=step,
            thought=f"Executing action: {action}",
            action=action,
            action_input=action_input,
        )
        self._current_thoughts.append(agent_thought)
        self.status = AgentStatus.EXECUTING
        logger.debug(f"[{self.name}] Act #{step}: {action}")
        return agent_thought
    
    async def observe(self, observation: str, step: int) -> None:
        """
        Record an observation from action result.
        
        Args:
            observation: Result of the action
            step: Step number
        """
        # Update the last thought with observation
        if self._current_thoughts and self._current_thoughts[-1].step == step:
            self._current_thoughts[-1].observation = observation
        logger.debug(f"[{self.name}] Observe #{step}: {observation[:100]}...")
    
    def clear_thoughts(self) -> None:
        """Clear current thinking chain."""
        self._current_thoughts.clear()
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return f"""You are {self.name}, an AI assistant specialized in: {self.description}

You follow the REACT pattern:
1. THINK: Analyze the request and plan your approach
2. ACT: Use tools or take actions to accomplish the task
3. OBSERVE: Review the results of your actions
4. REFLECT: Decide if more actions needed or provide final answer

Always be helpful, accurate, and security-conscious.
If you're unsure about something, ask for clarification.
"""
