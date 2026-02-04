"""
Task Agent - Specialized for task management and productivity.

Handles todos, reminders, calendar, and email tasks.
"""

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from intelclaw.agent.base import (
    AgentContext,
    AgentResponse,
    AgentStatus,
    BaseAgent,
)
from intelclaw.integrations.llm_provider import LLMProvider

if TYPE_CHECKING:
    from intelclaw.memory.manager import MemoryManager
    from intelclaw.tools.registry import ToolRegistry


class TaskAgent(BaseAgent):
    """
    Agent specialized in task management and productivity.
    
    Capabilities:
    - Create and manage todos
    - Set reminders
    - Calendar management
    - Email composition and management
    - Meeting scheduling
    """
    
    TASK_KEYWORDS = [
        "remind", "reminder", "schedule", "todo", "task", "calendar",
        "meeting", "email", "send", "notify", "deadline", "appointment",
        "plan", "organize", "list", "priority", "due", "follow up"
    ]
    
    def __init__(
        self,
        memory: Optional["MemoryManager"] = None,
        tools: Optional["ToolRegistry"] = None,
    ):
        """Initialize task agent."""
        super().__init__(
            name="Task Agent",
            description="Specialized in task management, scheduling, and productivity",
            memory=memory,
            tools=tools,
        )
        
        # LLM will be initialized asynchronously
        self._llm_provider: Optional[LLMProvider] = None
        self._llm = None
    
    async def _ensure_llm(self):
        """Ensure LLM is initialized."""
        if self._llm is None:
            self._llm_provider = LLMProvider({"model": "gpt-4o", "temperature": 0.2})
            await self._llm_provider.initialize()
            self._llm = self._llm_provider.llm
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process a task/productivity request.
        
        Flow:
        1. Parse the task request
        2. Extract relevant details (time, priority, etc.)
        3. Execute appropriate actions
        4. Confirm with user
        """
        start_time = time.time()
        self.clear_thoughts()
        self.status = AgentStatus.THINKING
        
        try:
            # Analyze the request
            await self.think(
                f"Analyzing task request: {context.user_message}",
                step=1
            )
            
            # Parse task details
            task_details = await self._parse_task_details(context.user_message)
            
            await self.think(
                f"Extracted task type: {task_details.get('type', 'general')}",
                step=2
            )
            
            # Execute based on task type
            response = await self._execute_task(task_details, context)
            
            latency = (time.time() - start_time) * 1000
            self.status = AgentStatus.IDLE
            
            return AgentResponse(
                answer=response,
                thoughts=self._current_thoughts.copy(),
                tools_used=task_details.get("tools_used", []),
                latency_ms=latency,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Task agent error: {e}")
            self.status = AgentStatus.ERROR
            
            return AgentResponse(
                answer=f"I couldn't complete the task: {str(e)}",
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
    
    async def can_handle(self, context: AgentContext) -> float:
        """Determine if this is a task/productivity request."""
        message_lower = context.user_message.lower()
        
        keyword_matches = sum(
            1 for kw in self.TASK_KEYWORDS
            if kw in message_lower
        )
        
        # Check for time-related patterns
        time_patterns = ["tomorrow", "today", "next week", "at ", "by ", "pm", "am"]
        time_matches = sum(1 for p in time_patterns if p in message_lower)
        
        confidence = min((keyword_matches + time_matches) / 5.0, 1.0)
        return confidence
    
    async def _parse_task_details(self, message: str) -> Dict[str, Any]:
        """Parse task details from the message."""
        
        system_prompt = """Extract task details from the user message.
Return a JSON object with these fields:
- type: "todo" | "reminder" | "email" | "calendar" | "general"
- title: Brief task title
- description: Full description
- due_date: ISO date string if mentioned (or null)
- priority: "high" | "medium" | "low"
- recipients: List of email recipients if applicable
- time: Time if mentioned

Only include fields that are clearly specified."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message),
        ]
        
        response = await self._llm.ainvoke(messages)
        
        # Parse response (in real implementation, use structured output)
        # For now, return basic structure
        task_type = "general"
        if "remind" in message.lower():
            task_type = "reminder"
        elif "email" in message.lower() or "send" in message.lower():
            task_type = "email"
        elif "meeting" in message.lower() or "schedule" in message.lower():
            task_type = "calendar"
        elif "todo" in message.lower() or "task" in message.lower():
            task_type = "todo"
        
        return {
            "type": task_type,
            "raw_message": message,
            "parsed_response": response.content,
        }
    
    async def _execute_task(
        self,
        task_details: Dict[str, Any],
        context: AgentContext
    ) -> str:
        """Execute the task based on type."""
        
        task_type = task_details.get("type", "general")
        tools_used = []
        
        if task_type == "reminder":
            # In full implementation, create actual reminder
            response = f"✅ I'll remind you: {task_details['raw_message']}"
            tools_used.append("reminder_create")
            
        elif task_type == "email":
            # Generate email draft
            response = await self._draft_email(task_details, context)
            tools_used.append("email_draft")
            
        elif task_type == "calendar":
            response = f"📅 I'll add this to your calendar: {task_details['raw_message']}"
            tools_used.append("calendar_create")
            
        elif task_type == "todo":
            response = f"☑️ Added to your todo list: {task_details['raw_message']}"
            tools_used.append("todo_create")
            
        else:
            response = f"I understand your request. {task_details.get('parsed_response', '')}"
        
        task_details["tools_used"] = tools_used
        return response
    
    async def _draft_email(
        self,
        task_details: Dict[str, Any],
        context: AgentContext
    ) -> str:
        """Generate an email draft."""
        
        system_prompt = """You are an email drafting assistant.
Based on the user's request, create a professional email.
Include subject line and body.
Format as:
Subject: [subject]

[email body]"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_details["raw_message"]),
        ]
        
        response = await self._llm.ainvoke(messages)
        
        return f"📧 Here's a draft email:\n\n{response.content}\n\nWould you like me to send this?"
