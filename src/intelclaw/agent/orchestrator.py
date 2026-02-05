"""
Agent Orchestrator - Central coordinator using LangGraph REACT pattern.

This is the root agent that:
- Parses user intent
- Creates task plans for complex multi-step workflows
- Routes to appropriate sub-agents
- Manages multi-step workflows with task queue
- Coordinates tool execution with proper step-wise progress
- Integrates web search and RAG for information gathering
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Annotated, Awaitable, Dict, List, Optional, Sequence, TypedDict, TYPE_CHECKING

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from loguru import logger
from pydantic import BaseModel, Field

from intelclaw.agent.base import (
    AgentContext,
    AgentResponse,
    AgentStatus,
    AgentThought,
    BaseAgent,
)
from intelclaw.agent.router import IntentRouter
from intelclaw.agent.task_planner import TaskPlanner, TaskPlan, TaskStep, TaskStatus as PlanTaskStatus, TaskQueue, QueuedTask, TaskPriority
from intelclaw.integrations.llm_provider import LLMProvider

if TYPE_CHECKING:
    from intelclaw.config.manager import ConfigManager
    from intelclaw.memory.manager import MemoryManager
    from intelclaw.tools.registry import ToolRegistry
    from intelclaw.core.events import EventBus


class AgentState(TypedDict):
    """State maintained throughout the agent execution."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: Dict[str, Any]
    thoughts: List[Dict[str, Any]]
    tools_used: List[str]
    iteration: int
    max_iterations: int
    plan: List[str]
    current_step: int
    completed_steps: List[str]
    consecutive_errors: int
    needs_replan: bool
    last_tool_error: Optional[str]


class AgentOrchestrator(BaseAgent):
    """
    Main orchestrator agent using LangGraph for REACT execution.
    
    Architecture:
    - Uses LangGraph's StateGraph for workflow management
    - Implements REACT loop: Reason -> Act -> Observe -> Repeat
    - Routes complex tasks to specialized sub-agents
    - Maintains conversation context across invocations
    - Integrates with memory system for personalization
    - Loads persona from markdown files (OpenClaw-style)
    """
    
    MODEL_NAME = "gpt-4o-mini"  # Default model
    
    # Persona files to load (OpenClaw-style)
    PERSONA_DIR = Path(__file__).parent.parent.parent.parent / "persona"
    PERSONA_FILES = ["AGENT.md", "SOUL.md", "SKILLS.md", "TOOLS.md", "USER.md"]
    
    # Context files injected into system prompt
    BOOTSTRAP_MAX_CHARS = 20000
    
    def __init__(
        self,
        config: "ConfigManager",
        memory: "MemoryManager",
        tools: "ToolRegistry",
        event_bus: "EventBus",
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Configuration manager
            memory: Memory system
            tools: Tool registry
            event_bus: Event bus for notifications
        """
        super().__init__(
            name="IntelCLaw Orchestrator",
            description="Central AI coordinator for task delegation and execution",
            memory=memory,
            tools=tools,
        )
        
        self.config = config
        self.event_bus = event_bus
        self.router = IntentRouter()
        self.sub_agents: Dict[str, BaseAgent] = {}
        
        # LangGraph components (initialized later)
        self._llm_provider: Optional[LLMProvider] = None
        self._llm = None  # Will be set from provider
        self._graph: Optional[StateGraph] = None
        self._compiled_graph = None
        self._langchain_tools: List[BaseTool] = []
        
        # Conversation history (short-term memory)
        self._conversation_history: List[BaseMessage] = []
        self._max_history = 20
        
        # Workflow controls (configurable)
        self._planning_enabled = self.config.get("agent.planning.enabled", True)
        self._planning_mode = self.config.get("agent.planning.mode", "always")
        self._planning_max_steps = self.config.get("agent.planning.max_steps", 8)
        self._planning_replan_on_failure = self.config.get("agent.planning.replan_on_failure", True)
        self._max_iterations = self.config.get("agent.execution.max_iterations", 12)
        self._max_tool_calls_per_iteration = self.config.get("agent.execution.max_tool_calls_per_iteration", 3)
        self._max_consecutive_errors = self.config.get("agent.execution.max_consecutive_errors", 3)

        # Research & retrieval controls
        self._auto_web_search = self.config.get("agent.research.auto_web_search", True)
        self._web_search_max_results = self.config.get("agent.research.max_results", 5)
        self._web_search_depth = self.config.get("agent.research.search_depth", "basic")
        self._web_context_max_chars = self.config.get("agent.research.max_context_chars", 2000)

        # Task Planner and Queue for autonomous execution
        self._task_planner: Optional[TaskPlanner] = None
        self._task_queue: TaskQueue = TaskQueue(max_concurrent=1)
        self._active_plan: Optional[TaskPlan] = None

        # Live workflow state for UI
        self._workflow_state: Dict[str, Any] = {
            "status": self.status.value,
            "plan": [],
            "current_step": 0,
            "current_step_title": None,
            "completed_steps": [],
            "next_step": None,
            "last_tool": None,
            "last_tool_error": None,
            "queue": [],
            "progress": 0,
        }
        # Concurrency controls
        self._state_lock = asyncio.Lock()
        self._pending_tasks: set[asyncio.Task] = set()
        self._shutdown = False
        
        logger.info("AgentOrchestrator created")
    
    async def initialize(self) -> None:
        """Initialize the orchestrator and build the LangGraph."""
        logger.info("Initializing AgentOrchestrator...")
        
        # Initialize LLM via unified provider (supports Copilot, OpenAI, Anthropic)
        model_config = self.config.get("models", {})
        llm_config = self.config.get("llm", {})
        
        self._llm_provider = LLMProvider({
            "model": model_config.get("primary", self.MODEL_NAME),
            "temperature": 1.0,
            "provider": llm_config.get("provider", "auto"),
        })
        await self._llm_provider.initialize()
        self._llm = self._llm_provider.llm
        
        logger.info(f"Using LLM provider: {self._llm_provider.active_provider}")
        
        # Get tools from registry
        if self.tools:
            self._langchain_tools = await self.tools.get_langchain_tools()
        
        # Initialize Task Planner for autonomous workflows
        self._task_planner = TaskPlanner(
            llm_provider=self._llm_provider,
            tools=self.tools,
            memory=self.memory,
            max_steps=self._planning_max_steps,
            enable_web_search=self._auto_web_search,
            enable_rag=True,
        )
        
        # Set up planner callbacks for UI updates
        self._task_planner._on_plan_created = self._on_plan_created
        self._task_planner._on_step_started = self._on_step_started
        self._task_planner._on_step_completed = self._on_step_completed
        self._task_planner._on_plan_completed = self._on_plan_completed
        
        # Initialize task queue for queuing multiple tasks
        self._task_queue = TaskQueue(max_concurrent=1)
        
        logger.info("TaskPlanner and TaskQueue initialized for autonomous workflows")
        
        # Build the REACT graph
        self._build_graph()
        
        # Initialize sub-agents
        await self._initialize_sub_agents()
        
        self.status = AgentStatus.IDLE
        await self._update_workflow_state(status=self.status.value)
        logger.success("AgentOrchestrator initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        logger.info("Shutting down AgentOrchestrator...")
        self._shutdown = True
        self.status = AgentStatus.IDLE
        await self._update_workflow_state(status=self.status.value)

        # Cancel any pending background tasks (plan callbacks, etc.)
        if self._pending_tasks:
            for task in list(self._pending_tasks):
                task.cancel()
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()
        
        # Shutdown sub-agents
        for agent in self.sub_agents.values():
            if hasattr(agent, 'shutdown'):
                await agent.shutdown()

        # Cleanup LLM provider if supported
        if self._llm_provider:
            cleanup = getattr(self._llm_provider, "cleanup", None)
            close = getattr(self._llm_provider, "close", None)
            try:
                if callable(cleanup):
                    result = cleanup()
                    if asyncio.iscoroutine(result):
                        await result
                elif callable(close):
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as e:
                logger.debug(f"LLM provider cleanup failed: {e}")

        # Clear state to avoid leaks
        self._conversation_history.clear()
        self._compiled_graph = None
        self._graph = None
        self._llm = None
        self._llm_provider = None

        # Break potential callback cycles
        if self._task_planner:
            self._task_planner._on_plan_created = None
            self._task_planner._on_step_started = None
            self._task_planner._on_step_completed = None
            self._task_planner._on_plan_completed = None

        # Note: Orchestrator does not subscribe to the event bus, so no unsubscribe needed.
        
        logger.info("AgentOrchestrator shutdown complete")
    
    async def _try_reinitialize_llm(self) -> bool:
        """
        Try to reinitialize the LLM connection.
        
        This is useful when the Copilot token has expired and been refreshed.
        """
        try:
            logger.info("Attempting to reinitialize LLM connection...")
            
            model_config = self.config.get("models", {})
            llm_config = self.config.get("llm", {})
            
            self._llm_provider = LLMProvider({
                "model": model_config.get("primary", self.MODEL_NAME),
                "temperature": 0.1,
                "provider": llm_config.get("provider", "auto"),
            })
            
            if await self._llm_provider.initialize():
                self._llm = self._llm_provider.llm
                logger.success(f"LLM reinitialized: {self._llm_provider.active_provider}")
                return True
            else:
                logger.warning("LLM reinitialization failed")
                return False
                
        except Exception as e:
            logger.error(f"LLM reinitialization error: {e}")
            return False
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history. Useful when switching models."""
        self._conversation_history = []
        logger.info("Conversation history cleared")
    
    async def run(self) -> None:
        """Background run loop for the orchestrator."""
        logger.info("AgentOrchestrator run loop started")
        while self.status != AgentStatus.ERROR:
            await asyncio.sleep(1)  # Heartbeat
    
    def _build_graph(self) -> None:
        """Build the LangGraph REACT workflow."""
        
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("reflect", self._reflect_node)
        
        # Add conditional edges
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "reason")
        workflow.add_conditional_edges(
            "reason",
            self._should_continue,
            {
                "act": "act",
                "end": END,
            }
        )
        workflow.add_edge("act", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            self._route_after_reflect,
            {
                "plan": "plan",
                "reason": "reason",
                "end": END,
            }
        )
        
        # Compile the graph
        self._graph = workflow
        self._compiled_graph = workflow.compile()
        
        logger.debug("LangGraph REACT workflow built")

    async def _emit_event(self, name: str, data: Dict[str, Any]) -> None:
        """Emit an event on the global bus (if available)."""
        if not self.event_bus:
            return
        try:
            await self.event_bus.emit(name, data, source="agent")
        except Exception as e:
            logger.debug(f"Event emit failed ({name}): {e}")

    async def _update_workflow_state(self, **updates: Any) -> None:
        """Update internal workflow state and broadcast to UI."""
        async with self._state_lock:
            self._workflow_state.update(updates)
            snapshot = dict(self._workflow_state)
        await self._emit_event("agent.workflow", snapshot)

    def _track_task(self, coro: Awaitable[Any]) -> Optional[asyncio.Task]:
        """Track background tasks for proper cleanup on shutdown."""
        if self._shutdown:
            return None
        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(lambda t: self._pending_tasks.discard(t))
        return task

    async def _set_active_plan(self, plan: Optional[TaskPlan]) -> None:
        """Set the active plan with concurrency protection."""
        async with self._state_lock:
            self._active_plan = plan

    async def _get_active_plan(self) -> Optional[TaskPlan]:
        """Get the active plan with concurrency protection."""
        async with self._state_lock:
            return self._active_plan
    
    # =========================================================================
    # Task Planner Callbacks for UI Updates
    # =========================================================================
    
    def _on_plan_created(self, plan: TaskPlan) -> None:
        """Callback when a new plan is created."""
        self._track_task(self._async_on_plan_created(plan))
    
    async def _async_on_plan_created(self, plan: TaskPlan) -> None:
        """Async handler for plan creation."""
        await self._set_active_plan(plan)
        step_titles = [s.title for s in plan.steps]
        
        await self._update_workflow_state(
            status=AgentStatus.THINKING.value,
            plan=step_titles,
            current_step=0,
            current_step_title=plan.steps[0].title if plan.steps else None,
            next_step=plan.steps[1].title if len(plan.steps) > 1 else None,
            completed_steps=[],
            progress=0,
        )
        
        await self._emit_event("agent.plan.created", {
            "plan_id": plan.id,
            "goal": plan.goal,
            "steps": [s.to_dict() for s in plan.steps],
            "total_steps": len(plan.steps),
        })
        
        logger.info(f"Plan created with {len(plan.steps)} steps: {step_titles}")
    
    def _on_step_started(self, step: TaskStep) -> None:
        """Callback when a step starts execution."""
        self._track_task(self._async_on_step_started(step))
    
    async def _async_on_step_started(self, step: TaskStep) -> None:
        """Async handler for step start."""
        plan = await self._get_active_plan()
        if not plan:
            return
        
        step_idx = next((i for i, s in enumerate(plan.steps) if s.id == step.id), 0)
        next_step = plan.steps[step_idx + 1] if step_idx + 1 < len(plan.steps) else None
        
        await self._update_workflow_state(
            status=AgentStatus.EXECUTING.value,
            current_step=step_idx,
            current_step_title=step.title,
            next_step=next_step.title if next_step else None,
            last_tool=step.tool,
        )
        
        await self._emit_event("agent.step.started", {
            "step_id": step.id,
            "step_index": step_idx,
            "title": step.title,
            "tool": step.tool,
        })
        
        logger.info(f"Executing step {step_idx + 1}/{len(plan.steps)}: {step.title}")
    
    def _on_step_completed(self, step: TaskStep) -> None:
        """Callback when a step completes."""
        self._track_task(self._async_on_step_completed(step))
    
    async def _async_on_step_completed(self, step: TaskStep) -> None:
        """Async handler for step completion."""
        plan = await self._get_active_plan()
        if not plan:
            return
        
        completed = [s.title for s in plan.completed_steps]
        progress = plan.progress_percent
        
        await self._update_workflow_state(
            completed_steps=completed,
            progress=progress,
            last_tool_error=step.error if step.status == PlanTaskStatus.FAILED else None,
        )
        
        await self._emit_event("agent.step.completed", {
            "step_id": step.id,
            "title": step.title,
            "status": step.status.value,
            "result": step.result[:2000] if step.result else None,
            "error": step.error,
            "progress": progress,
        })
        
        status = "✓" if step.status == PlanTaskStatus.COMPLETED else "✗"
        logger.info(f"Step {status} {step.title}: {step.result[:100] if step.result else step.error or 'done'}")
    
    def _on_plan_completed(self, plan: TaskPlan) -> None:
        """Callback when a plan completes."""
        self._track_task(self._async_on_plan_completed(plan))
    
    async def _async_on_plan_completed(self, plan: TaskPlan) -> None:
        """Async handler for plan completion."""
        await self._update_workflow_state(
            status=AgentStatus.IDLE.value,
            progress=100 if plan.is_complete else plan.progress_percent,
        )
        
        await self._emit_event("agent.plan.completed", {
            "plan_id": plan.id,
            "status": plan.status.value,
            "completed_steps": len(plan.completed_steps),
            "failed_steps": len(plan.failed_steps),
        })
        
        await self._set_active_plan(None)
        logger.info(f"Plan {plan.id} completed: {plan.status.value}")

    def _needs_web_research(self, message: str) -> bool:
        """Heuristic: decide if a query needs web search."""
        msg = (message or "").lower().strip()
        if not msg:
            return False

        # Explicit opt-out phrases
        opt_out = [
            "don't search", "do not search", "no web search", "without web",
            "without browsing", "offline", "no internet", "skip web", "avoid web"
        ]
        if any(phrase in msg for phrase in opt_out):
            return False

        indicators = [
            "latest", "recent", "today", "current", "news", "trend", "trends",
            "this week", "this month", "update", "release", "announcement",
            "breaking", "market", "report", "survey", "benchmark",
            "pricing", "price", "writeup", "cve", "vulnerability", "patch",
            "release notes", "changelog", "new version", "version",
            "search the web", "web search", "look up", "browse the web"
        ]

        # Tokenize for negation-aware matching
        words = re.findall(r"[a-z0-9']+", msg)
        negations = {"no", "not", "don't", "dont", "without", "avoid", "skip", "never"}

        def is_negated(start_idx: int) -> bool:
            window_start = max(0, start_idx - 3)
            return any(w in negations for w in words[window_start:start_idx])

        # Check indicators with negation window
        for indicator in indicators:
            indicator_words = re.findall(r"[a-z0-9']+", indicator)
            if not indicator_words:
                continue
            for idx in range(len(words) - len(indicator_words) + 1):
                if words[idx:idx + len(indicator_words)] == indicator_words:
                    if not is_negated(idx):
                        return True

        # Year-based recency cues (e.g., 2024/2025/2026)
        if re.search(r"\b20(1\d|2\d)\b", msg):
            return True

        return False

    async def _fetch_web_context(self, query: str) -> str:
        """Fetch web search results for a query using available tools."""
        if not self.tools:
            return ""
        try:
            results = await self.tools.execute(
                "tavily_search",
                {
                    "query": query,
                    "max_results": self._web_search_max_results,
                    "search_depth": self._web_search_depth,
                },
            )
        except Exception as e:
            logger.debug(f"Web search failed: {e}")
            return ""
        
        if not results or not isinstance(results, list):
            return ""
        
        lines = []
        for idx, item in enumerate(results[: self._web_search_max_results], start=1):
            title = item.get("title", "Untitled")
            snippet = item.get("content", item.get("snippet", ""))[:300]
            url = item.get("url", "")
            lines.append(f"{idx}. {title}\n{snippet}\nSource: {url}")
        
        context = "\n\n".join(lines)
        if len(context) > self._web_context_max_chars:
            context = context[: self._web_context_max_chars] + "..."
        return context

    async def _gather_request_context(self, context: AgentContext) -> Dict[str, Any]:
        """Gather RAG/web/screen context once per request."""
        gathered: Dict[str, Any] = {
            "screen_context": context.screen_context,
            "active_window": context.active_window,
        }
        tasks: List[tuple[str, Any]] = []

        if self.memory and hasattr(self.memory, 'get_rag_context'):
            tasks.append((
                "rag_context",
                self.memory.get_rag_context(
                    query=context.user_message,
                    include_persona=True,
                    include_session=True,
                    max_context_chars=5000,
                ),
            ))

        if self._auto_web_search and self._needs_web_research(context.user_message):
            tasks.append(("web_context", self._fetch_web_context(context.user_message)))

        if tasks:
            results = await asyncio.gather(
                *[coro for _, coro in tasks],
                return_exceptions=True,
            )
            for (key, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to gather {key}: {result}")
                    continue
                if result:
                    gathered[key] = result

        return gathered

    def _should_plan(self, user_message: str) -> bool:
        """Heuristic: determine if the task benefits from a multi-step plan."""
        msg = (user_message or "").lower()

        if self._requires_tool_use(user_message):
            return True
        
        # Explicit multi-step signals
        indicators = [
            "and then", "then", "after that", "next", "first", "second", "third",
            "step", "steps", "multi-step", "workflow", "pipeline",
            "plan", "roadmap", "sequence", "in order to"
        ]
        if any(ind in msg for ind in indicators):
            return True
        
        # Multiple action verbs or conjunctions often imply multi-step
        action_verbs = ["create", "build", "search", "analyze", "summarize", "write", "fix", "refactor", "deploy", "test", "run", "generate"]
        verb_hits = sum(1 for v in action_verbs if v in msg)
        if verb_hits >= 2:
            return True
        
        # Longer requests usually benefit from planning
        return len(msg) > 200

    def _requires_tool_use(self, user_message: str) -> bool:
        """
        Heuristic: detect if the request explicitly requires tool execution.
        
        This helps route action-oriented tasks through TaskPlanner so tools
        are actually executed (even when models don't emit tool_calls).
        """
        msg = (user_message or "").lower()
        if not msg:
            return False
        
        action_phrases = [
            # Web / search
            "search the web", "search web", "web search", "look up", "browse the web",
            # File operations
            "save to", "write to", "write a file", "create a file", "create file",
            "save in", "save as", "export to",
            "delete file", "remove file", "move file", "copy file", "rename file",
            "list files", "list directory", "open file", "read file",
            # Shell / command
            "run command", "execute command", "shell command", "powershell",
            "download", "install", "scrape", "fetch url", "collect urls",
            # Process / task management (THE KEY FIX for "check running tasks")
            "running task", "running process", "running app", "running program",
            "running service", "running software",
            "list process", "list task", "show process", "show task",
            "check process", "check task", "check running",
            "kill process", "kill task", "stop process", "end task",
            "task manager", "tasklist", "get-process", "top process",
            "what is running", "what's running", "whats running",
            "active process", "active task", "active app",
            "checkout the running", "check the running",
            # Windows services
            "list service", "show service", "check service", "start service",
            "stop service", "restart service", "service status",
            # Windows scheduled tasks
            "scheduled task", "task scheduler", "cron job",
            # Registry
            "registry", "reg key", "regedit",
            # Event log
            "event log", "eventlog", "windows log", "system log", "error log",
            # Network
            "network", "ip address", "ping", "traceroute", "wifi",
            "port", "listening port", "connection", "active connection",
            "firewall", "firewall rule",
            # Disk / storage
            "disk space", "disk usage", "storage", "free space", "drive",
            "large file", "folder size",
            # System info / performance
            "system info", "cpu usage", "memory usage", "ram usage",
            "gpu info", "battery", "uptime", "hardware info",
            "system performance", "system monitor",
            # Apps / software
            "installed app", "installed program", "installed software",
            "startup app", "startup program",
            # Environment
            "environment variable", "env var", "path variable",
            # Users / security
            "user account", "local user", "current user", "user group",
            "user session", "logged in user",
            # Windows update
            "windows update", "system update", "pending update",
            # Screenshot / clipboard
            "take screenshot", "screenshot", "clipboard", "copy to clipboard",
            # UI Automation
            "click button", "click on", "automate ui", "find window",
            "window title", "ui automation",
            # WMI/CIM
            "wmi", "cim", "win32_",
        ]
        if any(phrase in msg for phrase in action_phrases):
            return True
        
        # File extension or explicit path hints indicate file operations
        if re.search(r"\b\w+\.(txt|md|csv|json|yaml|yml|log|py|js|ts|html|css)\b", msg):
            return True
        
        # Current directory + action verb implies file output
        if "current directory" in msg or "this directory" in msg or "this folder" in msg:
            if any(word in msg for word in ["save", "write", "create", "output", "export"]):
                return True
        
        # PowerShell cmdlet patterns (Get-Process, Get-Service, etc.)
        if re.search(r"\b(get|set|start|stop|restart|new|remove|enable|disable)-(process|service|scheduledtask|netfirewallrule|item|childitem|content|hotfix|appxpackage|localuser|localgroup|netadapter|nettcpconnection|ciminstance)", msg, re.IGNORECASE):
            return True
        
        # Windows-specific command patterns
        if re.search(r"\b(tasklist|taskkill|netstat|ipconfig|systeminfo|wmic|sfc|dism|chkdsk|diskpart)\b", msg, re.IGNORECASE):
            return True
        
        return False
    
    def _is_complex_task(self, user_message: str) -> bool:
        """
        Determine if a task is complex enough to use the TaskPlanner.
        
        Complex tasks benefit from structured planning, step-by-step execution,
        and progress tracking via the TaskPlanner.
        
        Simple system queries (like "check running tasks") should NOT be complex
        but still need tool execution. They go through REACT with tool binding.
        """
        msg = (user_message or "").lower()

        # =====================================================================
        # SIMPLE SYSTEM QUERIES → NOT complex (handled by REACT with tools)
        # These are single-step tool operations that don't need a planner.
        # =====================================================================
        simple_system_patterns = [
            # Process queries
            "running task", "running process", "list process", "show process",
            "check process", "check task", "check running", "what is running",
            "what's running", "whats running", "active process", "active task",
            "checkout the running", "check the running", "top process",
            "kill process", "end task", "stop process",
            # Service queries
            "list service", "show service", "check service", "service status",
            "start service", "stop service", "restart service",
            # Network queries
            "network adapter", "ip address", "listening port", "active connection",
            "ping ", "public ip", "wifi profile",
            # System info queries
            "system info", "cpu usage", "memory usage", "ram usage", "gpu info",
            "battery", "uptime", "hardware info", "system performance",
            "disk space", "free space",
            # User queries
            "current user", "list user", "user session",
            # Simple queries
            "clipboard", "screenshot", "firewall status",
            "installed app", "installed program", "startup app",
            "environment variable", "path variable",
            "windows update", "pending update", "installed update",
            "event log", "system log",
        ]
        
        # If the message matches a simple system pattern and is short, it's NOT complex
        if len(msg) < 200 and any(pattern in msg for pattern in simple_system_patterns):
            # But if it has multiple "and" or sequential indicators, it IS complex
            if msg.count(" and ") < 2 and not any(kw in msg for kw in ["then", "after that", "first", "second"]):
                return False

        # If tools are explicitly required via multi-step phrasing, treat as complex
        if self._requires_tool_use(user_message):
            # Only complex if message is long or has multi-step indicators
            multi_step_indicators = [
                "and then", "then", "after that", "first", "second", "third",
                "step by step", "multi-step", "multiple steps", "workflow",
            ]
            if any(ind in msg for ind in multi_step_indicators):
                return True
            if len(msg) > 300:
                return True
            if msg.count(" and ") >= 2:
                return True
            # Single tool operation — not complex
            return False
        
        # Strong indicators of complex multi-step tasks
        complex_indicators = [
            # Explicit complexity markers
            "step by step", "step-by-step", "multi-step", "multiple steps",
            "workflow", "pipeline", "automation", "autonomous",
            
            # Sequential operations
            "first", "second", "third", "then", "after that", "finally",
            "in order", "sequence", "series of",
            
            # Project-level tasks
            "project", "application", "system", "full", "complete",
            "build", "create", "implement", "develop",
            
            # Research and analysis
            "research", "investigate", "analyze", "compare",
            "find and", "search and", "gather and",
            
            # Multiple outputs
            "several", "multiple", "various", "all", "each",
        ]
        
        if any(ind in msg for ind in complex_indicators):
            return True
        
        # Count distinct action verbs - 3+ suggests complexity
        action_verbs = [
            "create", "build", "search", "find", "analyze", "summarize",
            "write", "fix", "refactor", "deploy", "test", "run", "generate",
            "install", "configure", "setup", "update", "modify", "edit",
            "delete", "remove", "add", "integrate", "implement", "design"
        ]
        verb_hits = sum(1 for v in action_verbs if v in msg)
        if verb_hits >= 3:
            return True
        
        # Very long requests often indicate complex tasks
        if len(msg) > 300:
            return True
        
        # Check for "and" conjunctions suggesting multiple operations
        and_count = msg.count(" and ")
        if and_count >= 2:
            return True
        
        return False

    def _parse_plan_json_candidates(self, content: str) -> List[str]:
        """Extract top-level JSON array candidates from text using bracket balancing."""
        if not content:
            return []
        candidates: List[str] = []
        depth = 0
        start_idx: Optional[int] = None
        in_str = False
        escape = False
        for i, ch in enumerate(content):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
            if in_str:
                continue
            if ch == "[":
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == "]" and depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    candidates.append(content[start_idx:i + 1])
                    start_idx = None
        return candidates

    def _extract_json_array(self, content: str) -> Optional[Any]:
        """Extract JSON content (array or object) from LLM output."""
        if not content:
            return None

        # 1) Prefer fenced code blocks (```json ... ```)
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE):
            block = match.group(1).strip()
            try:
                return json.loads(block)
            except Exception:
                continue

        # 2) Scan for top-level JSON arrays
        for candidate in self._parse_plan_json_candidates(content):
            try:
                return json.loads(candidate)
            except Exception:
                continue

        return None

    def _normalize_plan_steps(self, payload: Any) -> List[str]:
        """Normalize a JSON payload into a list of non-empty step strings."""
        steps: List[str] = []

        if isinstance(payload, dict):
            for key in ("steps", "plan", "tasks"):
                if key in payload:
                    return self._normalize_plan_steps(payload.get(key))
            return steps

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    steps.append(item)
                elif isinstance(item, dict):
                    for key in ("title", "step", "description", "task", "name"):
                        val = item.get(key)
                        if val:
                            steps.append(str(val))
                            break
            return steps

        return steps

    def _parse_plan(self, content: str) -> List[str]:
        """Parse a plan from LLM output (JSON list or bullet list)."""
        if not content:
            return []
        
        # Try robust JSON extraction first
        parsed = self._extract_json_array(content)
        if parsed is not None:
            steps = self._normalize_plan_steps(parsed)
            steps = [s.strip() for s in steps if s and str(s).strip()]
            if steps:
                return steps[: self._planning_max_steps]
        
        # Fallback to line-based parsing
        steps = []
        for line in content.splitlines():
            stripped = line.strip().lstrip("-").lstrip("*").strip()
            if not stripped:
                continue
            # Remove "1.", "2)" prefixes
            if stripped[0].isdigit():
                stripped = stripped.lstrip("0123456789").lstrip(".").lstrip(")").strip()
            if stripped:
                steps.append(stripped)
        
        return steps[: self._planning_max_steps]

    async def _plan_node(self, state: AgentState) -> AgentState:
        """Planning node - generate a multi-step plan when needed."""
        if not self._planning_enabled:
            return state
        
        # Only plan if we don't already have a plan or a replan is requested
        if state.get("plan") and not state.get("needs_replan"):
            return state
        
        # Determine if the task warrants planning
        user_message = ""
        if state.get("messages"):
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "content"):
                user_message = last_msg.content
        
        if self._planning_mode == "adaptive" and not self._should_plan(user_message):
            return state
        
        # Ensure LLM available
        if self._llm is None:
            await self._try_reinitialize_llm()
            if self._llm is None:
                return state
        
        system_prompt = (
            "You are a senior software agent planner. Create a concise, actionable plan "
            f"with at most {self._planning_max_steps} steps. "
            "Return ONLY a JSON array of strings (no extra text)."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message or "Create a plan for the task.")
        ]
        
        try:
            response = await self._llm.ainvoke(messages)
            plan_steps = self._parse_plan(response.content if hasattr(response, "content") else str(response))
        except Exception as e:
            logger.debug(f"Planning failed: {e}")
            plan_steps = []
        
        if not plan_steps:
            return state
        
        await self._update_workflow_state(
            status=AgentStatus.THINKING.value,
            plan=plan_steps,
            current_step=0,
            completed_steps=[],
        )
        await self._emit_event("agent.plan", {
            "plan": plan_steps,
            "current_step": 0,
            "completed_steps": [],
        })
        
        return {
            **state,
            "plan": plan_steps,
            "current_step": 0,
            "completed_steps": [],
            "needs_replan": False,
        }
    
    async def _reason_node(self, state: AgentState) -> AgentState:
        """
        Reasoning node - Analyze and plan.
        
        This is where the LLM thinks about the task and decides
        whether to use tools or provide a direct response.
        """
        self.status = AgentStatus.THINKING
        await self._update_workflow_state(status=self.status.value)
        
        # Check if LLM is initialized
        if self._llm is None:
            # Try to re-initialize (token may have been refreshed)
            await self._try_reinitialize_llm()
            
            if self._llm is None:
                logger.error("LLM not available - GitHub Copilot subscription may be expired or unavailable")
                from langchain_core.messages import AIMessage
                return {
                    **state,
                    "messages": list(state["messages"]) + [AIMessage(content="I apologize, but I'm unable to process your request. The GitHub Copilot connection is not available. Please run 'uv run python -m intelclaw onboard' to refresh your authentication.")],
                    "thoughts": list(state["thoughts"]) + [{"step": state["iteration"], "type": "error", "content": "LLM not initialized", "timestamp": datetime.now().isoformat()}],
                    "iteration": state["iteration"] + 1,
                }
        
        # Build prompt with tools and workflow context
        prompt_context = dict(state["context"])
        if state.get("plan"):
            prompt_context["plan"] = state.get("plan", [])
            prompt_context["current_step"] = state.get("current_step", 0)
            prompt_context["completed_steps"] = state.get("completed_steps", [])
        if state.get("last_tool_error"):
            prompt_context["last_tool_error"] = state.get("last_tool_error")
        prompt_context["iteration"] = state.get("iteration", 0)
        prompt_context["max_iterations"] = state.get("max_iterations", self._max_iterations)
        
        system_prompt = await self._get_system_prompt(prompt_context)
        
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        
        # Get LLM response with tool binding
        if self._langchain_tools:
            # On the first iteration, if the query clearly requires tool use,
            # force the LLM to call a tool instead of responding with text
            tool_choice = "auto"
            if state["iteration"] == 0:
                # Extract user message from state
                user_msg = ""
                for msg in reversed(state["messages"]):
                    if hasattr(msg, 'content') and isinstance(msg, HumanMessage):
                        user_msg = msg.content
                        break
                if user_msg and self._requires_tool_use(user_msg):
                    tool_choice = "required"
                    logger.info(f"Forcing tool_choice='required' for tool-required query: {user_msg[:80]}")
            
            llm_with_tools = self._llm.bind_tools(self._langchain_tools, tool_choice=tool_choice)
            response = await llm_with_tools.ainvoke(messages)
        else:
            response = await self._llm.ainvoke(messages)
        
        # Record thought
        thought = {
            "step": state["iteration"],
            "type": "reason",
            "content": response.content if hasattr(response, 'content') else str(response),
            "has_tool_calls": hasattr(response, 'tool_calls') and bool(response.tool_calls),
            "timestamp": datetime.now().isoformat(),
        }
        
        new_messages = list(state["messages"]) + [response]
        new_thoughts = list(state["thoughts"]) + [thought]
        
        return {
            **state,
            "messages": new_messages,
            "thoughts": new_thoughts,
            "iteration": state["iteration"] + 1,
        }
    
    async def _act_node(self, state: AgentState) -> AgentState:
        """
        Action node - Execute tools.
        
        Processes tool calls from the reasoning step.
        """
        self.status = AgentStatus.EXECUTING
        await self._update_workflow_state(status=self.status.value)
        
        last_message = state["messages"][-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return state
        
        tool_results = []
        tools_used = list(state["tools_used"])
        consecutive_errors = state.get("consecutive_errors", 0)
        last_tool_error = state.get("last_tool_error")
        
        tool_calls = list(last_message.tool_calls)
        if self._max_tool_calls_per_iteration and len(tool_calls) > self._max_tool_calls_per_iteration:
            logger.warning(
                f"Too many tool calls ({len(tool_calls)}). Limiting to {self._max_tool_calls_per_iteration}."
            )
            tool_calls = tool_calls[: self._max_tool_calls_per_iteration]
        
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", tool_call.get("arguments"))
                tool_call_id = tool_call.get("id")
            else:
                tool_name = getattr(tool_call, "name", None)
                tool_args = getattr(tool_call, "args", None) or getattr(tool_call, "arguments", None)
                tool_call_id = getattr(tool_call, "id", None)
            
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {"input": tool_args}
            
            if tool_args is None:
                tool_args = {}
            
            if not tool_call_id:
                tool_call_id = f"call_{time.time_ns()}"
            
            logger.debug(f"Executing tool: {tool_name}")
            tools_used.append(tool_name)
            
            await self._emit_event("tool.call", {
                "id": tool_call_id,
                "name": tool_name,
                "args": tool_args,
            })
            
            # Execute the tool
            result = await self._execute_tool(tool_name, tool_args)
            
            result_text = str(result)
            if result_text.lower().startswith("error"):
                consecutive_errors += 1
                last_tool_error = result_text[:500]
            else:
                consecutive_errors = 0
                last_tool_error = None
            
            await self._emit_event("tool.result", {
                "id": tool_call_id,
                "name": tool_name,
                "success": not result_text.lower().startswith("error"),
                "result": result_text,
            })
            
            tool_results.append(
                ToolMessage(
                    content=result_text,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )
        
        new_messages = list(state["messages"]) + tool_results
        if tools_used:
            await self._update_workflow_state(
                last_tool=tools_used[-1],
                last_tool_error=last_tool_error,
            )
        
        return {
            **state,
            "messages": new_messages,
            "tools_used": tools_used,
            "consecutive_errors": consecutive_errors,
            "last_tool_error": last_tool_error,
        }
    
    async def _reflect_node(self, state: AgentState) -> AgentState:
        """
        Reflection node - Process observations and decide next steps.
        """
        # Check if we have tool results to process
        last_message = state["messages"][-1]
        
        if isinstance(last_message, ToolMessage):
            thought = {
                "step": state["iteration"],
                "type": "observe",
                "tool": last_message.name,
                "observation": last_message.content[:5000],  # Keep full context for LLM
                "timestamp": datetime.now().isoformat(),
            }
            new_thoughts = list(state["thoughts"]) + [thought]
            
            # Update plan progress (best-effort)
            current_step = state.get("current_step", 0)
            plan = state.get("plan", [])
            completed_steps = list(state.get("completed_steps", []))
            if plan:
                step_idx = min(current_step, max(len(plan) - 1, 0))
                if plan and step_idx < len(plan):
                    step_text = plan[step_idx]
                    if step_text not in completed_steps:
                        completed_steps.append(step_text)
                current_step = min(step_idx + 1, len(plan))
            
            # Trigger replan if too many consecutive errors
            needs_replan = state.get("needs_replan", False)
            current_error = state.get("last_tool_error")
            if self._planning_replan_on_failure and state.get("consecutive_errors", 0) >= self._max_consecutive_errors:
                needs_replan = True
            
            await self._update_workflow_state(
                status=AgentStatus.THINKING.value,
                current_step=current_step,
                completed_steps=completed_steps,
                last_tool_error=current_error if needs_replan else None,
            )
            await self._emit_event("agent.plan.update", {
                "plan": plan,
                "current_step": current_step,
                "completed_steps": completed_steps,
                "needs_replan": needs_replan,
            })
            
            return {
                **state,
                "thoughts": new_thoughts,
                "current_step": current_step,
                "completed_steps": completed_steps,
                "needs_replan": needs_replan,
            }
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """
        Determine if the agent should continue or end.
        
        Returns:
            "act" if tools should be executed
            "reason" if more thinking needed
            "end" if task is complete
        """
        # Log current iteration (no hard limit - let LLM decide when complete)
        if state["iteration"] > 0:
            logger.debug(f"REACT iteration: {state['iteration']}")
        
        # Enforce iteration limit to avoid infinite loops
        if state.get("max_iterations") and state["iteration"] >= state["max_iterations"]:
            logger.warning(f"Reached max iterations ({state['max_iterations']}). Ending loop.")
            return "end"
        
        # Check last message
        last_message = state["messages"][-1] if state["messages"] else None
        
        if last_message is None:
            return "end"
        
        # If AI message with tool calls, go to act
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "act"
        
        # If it's a tool message, go back to reason
        if isinstance(last_message, ToolMessage):
            return "reason"
        
        # If AI message without tool calls, we're done
        if isinstance(last_message, AIMessage) and not getattr(last_message, 'tool_calls', None):
            return "end"
        
        return "end"

    def _route_after_reflect(self, state: AgentState) -> str:
        """Route after reflect: replan, continue, or end."""
        # Stop if max iterations reached
        if state.get("max_iterations") and state["iteration"] >= state["max_iterations"]:
            return "end"
        
        if state.get("needs_replan"):
            return "plan"
        
        return self._should_continue(state)

    def _append_history(self, *messages: BaseMessage) -> None:
        """Append messages to conversation history and keep it bounded."""
        for msg in messages:
            if msg:
                self._conversation_history.append(msg)
        if len(self._conversation_history) > self._max_history:
            self._conversation_history = self._conversation_history[-self._max_history:]
    
    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a tool through the registry."""
        if not self.tools:
            return f"Error: Tool registry not available"
        
        # Ensure tool_args is a dict, not None
        if tool_args is None:
            tool_args = {}
        
        # Log the tool call for debugging
        logger.debug(f"Executing tool '{tool_name}' with args: {tool_args}")
        
        try:
            result = await self.tools.execute(tool_name, tool_args)
            
            # Emit event
            await self.event_bus.emit("tool.executed", {
                "tool": tool_name,
                "args": tool_args,
                "success": True,
            })
            
            return str(result)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool execution error for '{tool_name}': {error_msg}")
            
            # Provide helpful error message for missing parameters
            if "Missing required parameter" in error_msg:
                tool_def = self.tools.get_definition(tool_name)
                if tool_def and tool_def.parameters:
                    required = tool_def.parameters.get("required", [])
                    props = tool_def.parameters.get("properties", {})
                    hint = f"Required parameters: {required}. "
                    hint += f"Got: {list(tool_args.keys())}"
                    return f"Error: {error_msg}. {hint}"
            
            return f"Error executing {tool_name}: {error_msg}"
    
    async def _load_persona_files(self) -> Dict[str, str]:
        """
        Load persona markdown files (OpenClaw-style context injection).
        
        Returns:
            Dict mapping filename to content
        """
        persona_content: Dict[str, str] = {}
        
        try:
            import aiofiles  # type: ignore
        except Exception as e:
            logger.warning(f"aiofiles not available, falling back to thread-based reads: {e}")
            aiofiles = None
        
        for filename in self.PERSONA_FILES:
            filepath = self.PERSONA_DIR / filename
            try:
                if filepath.exists():
                    if aiofiles:
                        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
                            content = await f.read()
                    else:
                        content = await asyncio.to_thread(filepath.read_text, encoding="utf-8")
                    # Truncate large files
                    if len(content) > self.BOOTSTRAP_MAX_CHARS:
                        content = content[:self.BOOTSTRAP_MAX_CHARS] + "\n\n[... truncated ...]"
                    persona_content[filename] = content
                    logger.debug(f"Loaded persona file: {filename} ({len(content)} chars)")
            except Exception as e:
                logger.warning(f"Failed to load persona file {filename}: {e}")
        
        return persona_content
    
    async def _get_system_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build the system prompt using persona files (OpenClaw-style).
        
        Loads and injects AGENT.md, SOUL.md, SKILLS.md, TOOLS.md, USER.md
        to define agent identity, personality, and capabilities.
        """
        
        # Get current model name from LLM provider
        current_model = "unknown"
        if self._llm_provider and hasattr(self._llm_provider, '_llm'):
            llm = self._llm_provider._llm
            if hasattr(llm, 'model'):
                current_model = llm.model
        
        # Load persona files
        persona = await self._load_persona_files()
        
        # Build system prompt sections
        sections = []
        
        # Runtime info (model, platform)
        sections.append(f"""## Runtime
- Model: {current_model}
- Provider: GitHub Copilot
- Platform: Windows
- Host: IntelCLaw
""")
        
        # Project Context - Persona Files (OpenClaw-style injection)
        if persona:
            sections.append("# Project Context\n")
            sections.append("The following persona files define your identity and behavior:\n")
            
            # Check if SOUL.md is present (like OpenClaw)
            has_soul = "SOUL.md" in persona
            if has_soul:
                sections.append(
                    "If SOUL.md is present, embody its persona and tone. "
                    "Avoid stiff, generic replies; follow its guidance.\n"
                )
            
            # Inject each persona file
            for filename, content in persona.items():
                sections.append(f"## {filename}\n")
                sections.append(content)
                sections.append("\n")
        
        # Tool usage guidance (minimal - let TOOLS.md handle details)
        # Include available tools list for better tool calling
        available_tool_names = []
        if self.tools:
            try:
                tool_defs = self.tools.list_tools()
                available_tool_names = [t.name for t in tool_defs]
            except Exception:
                pass
        
        tool_list_str = ", ".join(available_tool_names) if available_tool_names else "various tools"
        
        sections.append(f"""## Tool Usage
You MUST use tools when the user asks you to perform actions. DO NOT just explain what you would do - actually DO IT by calling the appropriate tool.

AVAILABLE TOOLS: {tool_list_str}

KEY TOOL MAPPINGS:

### File Operations
- DELETE a file: use "file_delete" with {{"path": "filename"}}
- CREATE/WRITE a file: use "file_write" with {{"path": "filename", "content": "..."}}
- READ a file: use "file_read" with {{"path": "filename"}}
- SEARCH for files: use "file_search" with {{"directory": ".", "pattern": "*.md"}}
- LIST directory: use "list_directory" with {{"path": "."}}
- MOVE/RENAME: use "file_move" with {{"source": "old", "destination": "new"}}
- COPY: use "file_copy" with {{"source": "src", "destination": "dst"}}
- GET CURRENT DIR: use "get_cwd"

### Search & Web
- WEB SEARCH: use "tavily_search" with {{"query": "search terms"}}
- SCRAPE WEBSITE: use "web_scrape" with {{"url": "..."}}

### Shell & Code
- RUN COMMAND: use "shell_command" with {{"command": "..."}}
- POWERSHELL: use "powershell" with {{"script": "..."}}
- EXECUTE CODE: use "execute_code" with {{"code": "..."}}
- PIP INSTALL: use "pip_install" with {{"packages": ["pkg1"]}}

### Process & Task Management (CRITICAL for system queries)
- CHECK RUNNING TASKS/PROCESSES: use "process_management" with {{"action": "list"}} or {{"action": "top_cpu"}} or {{"action": "top_memory"}}
- FIND A PROCESS: use "process_management" with {{"action": "find", "name": "chrome"}}
- KILL A PROCESS: use "process_management" with {{"action": "kill", "pid": 1234}} or {{"action": "kill", "name": "notepad"}}
- PROCESS DETAILS: use "process_management" with {{"action": "details", "pid": 1234}}
- PROCESS TREE: use "process_management" with {{"action": "tree"}}

### Windows Services
- LIST SERVICES: use "windows_services" with {{"action": "list"}}
- SERVICE STATUS: use "windows_services" with {{"action": "get", "name": "svcname"}}
- START/STOP/RESTART: use "windows_services" with {{"action": "start", "name": "svcname"}}

### Scheduled Tasks
- LIST TASKS: use "windows_tasks" with {{"action": "list"}}
- CREATE TASK: use "windows_tasks" with {{"action": "create", "name": "...", "action_path": "...", "schedule": "daily", "start_time": "09:00"}}

### Registry
- READ REGISTRY: use "windows_registry" with {{"action": "get", "path": "HKLM:\\...", "name": "value"}}
- LIST KEYS: use "windows_registry" with {{"action": "list_keys", "path": "HKLM:\\..."}}

### Event Logs
- QUERY EVENTS: use "windows_eventlog" with {{"action": "query", "log_name": "System", "levels": ["Error", "Warning"]}}
- LIST LOGS: use "windows_eventlog" with {{"action": "list_logs"}}

### Network
- NETWORK ADAPTERS: use "network_info" with {{"action": "adapters"}}
- ACTIVE CONNECTIONS: use "network_info" with {{"action": "connections"}}
- LISTENING PORTS: use "network_info" with {{"action": "listening"}}
- PING: use "network_info" with {{"action": "ping", "target": "google.com"}}
- PUBLIC IP: use "network_info" with {{"action": "public_ip"}}

### Disk & Storage
- DISK SPACE: use "disk_management" with {{"action": "space"}}
- VOLUMES: use "disk_management" with {{"action": "volumes"}}
- LARGE FILES: use "disk_management" with {{"action": "large_files", "path": "C:\\"}}

### Firewall
- FIREWALL STATUS: use "windows_firewall" with {{"action": "status"}}
- LIST RULES: use "windows_firewall" with {{"action": "list_rules"}}

### System Info & Performance
- SYSTEM INFO: use "system_info" with {{"info_type": "all"}}
- PERFORMANCE OVERVIEW: use "system_performance" with {{"action": "overview"}}
- CPU/MEMORY/GPU: use "system_performance" with {{"action": "cpu"}} or {{"action": "memory"}} or {{"action": "gpu"}}
- HARDWARE INFO: use "system_performance" with {{"action": "hardware"}}
- BATTERY: use "system_performance" with {{"action": "battery"}}
- UPTIME: use "system_performance" with {{"action": "uptime"}}

### Installed Applications
- LIST APPS: use "installed_apps" with {{"action": "list"}}
- SEARCH APP: use "installed_apps" with {{"action": "search", "name": "chrome"}}
- STARTUP APPS: use "installed_apps" with {{"action": "startup_apps"}}

### Environment Variables
- LIST ENV VARS: use "environment_vars" with {{"action": "list"}}
- GET PATH: use "environment_vars" with {{"action": "path"}}
- SET VAR: use "environment_vars" with {{"action": "set", "name": "VAR", "value": "val"}}

### Windows Updates
- CHECK UPDATES: use "windows_update" with {{"action": "pending"}}
- INSTALLED UPDATES: use "windows_update" with {{"action": "installed"}}

### Users & Security
- CURRENT USER: use "user_security" with {{"action": "current_user"}}
- LIST USERS: use "user_security" with {{"action": "list_users"}}

### WMI/CIM Queries
- QUERY WMI: use "windows_cim" with {{"class_name": "Win32_OperatingSystem"}}

### UI Automation
- FIND WINDOW: use "windows_ui_automation" with {{"action": "find_window", "window_title": "..."}}
- CLICK/INVOKE: use "windows_ui_automation" with {{"action": "click", "window_title": "...", "control_title": "..."}}

### Other
- SCREENSHOT: use "screenshot"
- CLIPBOARD: use "clipboard" with {{"action": "read"}} or {{"action": "write", "content": "..."}}
- LAUNCH APP: use "launch_app" with {{"target": "app_name"}}

IMPORTANT RULES:
1. When user asks about running tasks/processes -> IMMEDIATELY use process_management tool
2. When user says "delete file X" -> immediately call file_delete
3. NEVER say "I will do X" without actually calling the tool
4. NEVER claim you completed an action unless a tool result confirms it
5. If a tool fails, report the actual error message
6. For system queries (processes, services, network, etc.) -> ALWAYS use the appropriate Windows tool
7. The user is on Windows - use Windows-specific tools and PowerShell commands

Think step by step before taking action.
Be security-conscious - ask for confirmation for sensitive operations.
If an execution plan is present, follow it step-by-step and use tools to complete each step.
If a step fails repeatedly, consider replanning or ask the user for clarification.
Focus only on the current plan step unless the user changes the goal.
""")
        
        # Current runtime context
        sections.append("## Current Context\n")
        
        if context.get("active_window"):
            sections.append(f"Active Window: {context['active_window']}\n")
        
        if context.get("screen_text"):
            sections.append(f"Visible Text: {context['screen_text'][:500]}...\n")
        
        # Workflow plan context
        if context.get("plan"):
            plan = context.get("plan", [])
            completed = set(context.get("completed_steps", []))
            current_step = context.get("current_step", 0)
            sections.append("## Execution Plan\n")
            for idx, step in enumerate(plan, start=1):
                status = "done" if step in completed else "next" if idx - 1 == current_step else "pending"
                sections.append(f"{idx}. [{status}] {step}\n")
            sections.append(f"Current step index: {current_step}\n")
        
        if context.get("last_tool_error"):
            sections.append("## Last Tool Error\n")
            sections.append(f"{context['last_tool_error']}\n")
        
        # User preferences (from Mem0 and context)
        if context.get("user_preferences"):
            prefs = context["user_preferences"]
            sections.append("### User Preferences (remembered from past interactions)\n")
            if isinstance(prefs, dict):
                if prefs.get("mem0_preferences"):
                    sections.append(prefs["mem0_preferences"])
                    sections.append("\n")
                for k, v in prefs.items():
                    if k != "mem0_preferences":
                        sections.append(f"- {k}: {v}\n")
            else:
                sections.append(f"{prefs}\n")
        
        # Add RAG context if query is provided
        if context.get("rag_context"):
            sections.append("## Retrieved Context (from persona and memory)\n")
            sections.append(context["rag_context"])
            sections.append("\n")

        # Add Web context if available
        if context.get("web_context"):
            sections.append("## Web Context (fresh search results)\n")
            sections.append(context["web_context"])
            sections.append("\n")
        
        return "\n".join(sections)
    
    async def _initialize_sub_agents(self) -> None:
        """Initialize specialized sub-agents."""
        from intelclaw.agent.sub_agents.research_agent import ResearchAgent
        from intelclaw.agent.sub_agents.coding_agent import CodingAgent
        from intelclaw.agent.sub_agents.task_agent import TaskAgent
        from intelclaw.agent.sub_agents.system_agent import SystemAgent
        
        self.sub_agents = {
            "research": ResearchAgent(memory=self.memory, tools=self.tools),
            "coding": CodingAgent(memory=self.memory, tools=self.tools),
            "task": TaskAgent(memory=self.memory, tools=self.tools),
            "system": SystemAgent(memory=self.memory, tools=self.tools),
        }
        
        # Register sub-agents with router
        for name, agent in self.sub_agents.items():
            self.router.register_agent(name, agent)
        
        logger.info(f"Initialized {len(self.sub_agents)} sub-agents")
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process a request through the REACT pipeline.
        
        For complex multi-step tasks, uses TaskPlanner for structured planning
        and step-by-step execution with progress tracking.
        
        For simple system queries, tries direct tool execution first for speed.
        
        Args:
            context: Full context for the request
            
        Returns:
            Complete agent response
        """
        start_time = time.time()
        self.clear_thoughts()
        await self._update_workflow_state(
            status=AgentStatus.THINKING.value,
            plan=[],
            current_step=0,
            completed_steps=[],
            last_tool=None,
            last_tool_error=None,
            progress=0,
        )
        
        # Check if should delegate to sub-agent
        delegation = await self.router.route(context)
        if delegation and delegation.confidence > 0.8 and not self._should_plan(context.user_message):
            sub_agent = self.sub_agents.get(delegation.agent_name)
            if sub_agent:
                logger.info(f"Delegating to {delegation.agent_name}")
                return await sub_agent.process(context)

        # Gather shared context once per request
        shared_context = await self._gather_request_context(context)
        
        # =========================================================================
        # COMPLEX TASK HANDLING: Use TaskPlanner for multi-step tasks
        # =========================================================================
        if self._task_planner and self._is_complex_task(context.user_message):
            logger.info("Complex task detected - using TaskPlanner for structured execution")
            return await self._process_with_task_planner(context, start_time, shared_context)
        
        # =========================================================================
        # SIMPLE TASK HANDLING: Use standard REACT workflow
        # =========================================================================
        react_response = await self._process_with_react(context, start_time, shared_context)
        
        # If the request clearly requires tools but none were used, fall back to TaskPlanner
        if (
            self._task_planner
            and self._planning_enabled
            and self._requires_tool_use(context.user_message)
            and not react_response.tools_used
        ):
            logger.info("No tools were used for a tool-required request; falling back to TaskPlanner execution.")
            return await self._process_with_task_planner(context, start_time, shared_context)
        
        return react_response
    
    async def _process_with_task_planner(
        self,
        context: AgentContext,
        start_time: float,
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Process a complex request using the TaskPlanner for structured execution.
        
        This method handles multi-step tasks by:
        1. Creating a detailed plan with the TaskPlanner
        2. Executing each step with progress tracking
        3. Broadcasting updates to the UI via events
        4. Handling failures with automatic replanning
        
        Args:
            context: Full context for the request
            start_time: Request start time for latency calculation
            
        Returns:
            Complete agent response with execution results
        """
        logger.info(f"Processing complex task with TaskPlanner: {context.user_message[:100]}...")
        
        # Gather additional context for planning
        additional_context = {}
        
        shared_context = shared_context or {}

        # Use shared RAG/web context if available
        if shared_context.get("rag_context"):
            additional_context["rag_context"] = shared_context["rag_context"]
        if shared_context.get("web_context"):
            additional_context["web_context"] = shared_context["web_context"]

        # Add screen context
        screen_context = shared_context.get("screen_context") or context.screen_context
        if screen_context:
            additional_context["screen_context"] = screen_context.get("text", "")[:1000]
        active_window = shared_context.get("active_window") or context.active_window
        if active_window:
            additional_context["active_window"] = active_window
        
        try:
            # Step 1: Create the plan
            await self._update_workflow_state(
                status=AgentStatus.THINKING.value,
                current_step_title="Creating execution plan...",
            )
            
            plan = await self._task_planner.create_plan(
                goal=context.user_message,
                context=additional_context
            )
            
            if not plan or not plan.steps:
                logger.warning("TaskPlanner failed to create a plan, falling back to REACT")
                return await self._process_with_react(context, start_time, shared_context)
            
            # Step 2: Execute the plan
            await self._update_workflow_state(
                status=AgentStatus.EXECUTING.value,
                current_step_title=plan.steps[0].title if plan.steps else "Executing...",
            )
            
            executed_plan = await self._task_planner.execute_plan(plan)
            
            # Step 3: Build response from execution results
            thoughts = []
            tools_used = []
            
            for i, step in enumerate(executed_plan.steps):
                thought = AgentThought(
                    step=i + 1,
                    thought=f"{step.title}: {step.result or step.error or 'completed'}",
                    action=step.tool,
                    timestamp=datetime.now(),
                )
                thoughts.append(thought)
                if step.tool:
                    tools_used.append(step.tool)
            
            # Build the final answer
            if executed_plan.is_complete:
                # Synthesize a final answer from the results
                answer = await self._synthesize_plan_response(executed_plan, context.user_message)
            else:
                # Plan partially completed or failed
                completed = [s.title for s in executed_plan.completed_steps]
                failed = [f"{s.title}: {s.error}" for s in executed_plan.failed_steps]
                
                answer = f"I completed {len(completed)} of {len(executed_plan.steps)} steps.\n\n"
                if completed:
                    answer += "✓ Completed:\n" + "\n".join(f"  - {c}" for c in completed) + "\n\n"
                if failed:
                    answer += "✗ Failed:\n" + "\n".join(f"  - {f}" for f in failed) + "\n\n"
                
                # Add last step result as context
                last_result = None
                for s in reversed(executed_plan.steps):
                    if s.result:
                        last_result = s.result
                        break
                if last_result:
                    answer += f"Last result: {last_result}"
            
            latency = (time.time() - start_time) * 1000
            
            # Store in memory
            if self.memory:
                await self.memory.store_interaction(
                    user_message=context.user_message,
                    agent_response=answer,
                    tools_used=tools_used,
                )
            
            self.status = AgentStatus.IDLE
            await self._update_workflow_state(
                status=self.status.value,
                progress=100 if executed_plan.is_complete else executed_plan.progress_percent,
            )
            
            return AgentResponse(
                answer=answer,
                thoughts=thoughts,
                tools_used=tools_used,
                latency_ms=latency,
                success=plan.is_complete,
            )
            
        except Exception as e:
            logger.error(f"TaskPlanner execution failed: {e}")
            # Fallback to REACT workflow
            logger.info("Falling back to standard REACT workflow")
            return await self._process_with_react(context, start_time, shared_context)
    
    async def _synthesize_plan_response(self, plan: TaskPlan, original_goal: str) -> str:
        """
        Synthesize a coherent final response from plan execution results.
        
        Uses the LLM to create a natural response summarizing what was accomplished.
        """
        if not self._llm:
            # Fallback to simple concatenation
            results = []
            for step in plan.completed_steps:
                if step.result:
                    results.append(f"• {step.title}: {step.result}")
            return "Here's what I accomplished:\n\n" + "\n".join(results)
        
        # Build a summary of what was done
        step_summaries = []
        for step in plan.steps:
            status = "✓" if step.status == PlanTaskStatus.COMPLETED else "✗"
            result_preview = (step.result or step.error or "")[:2000]
            step_summaries.append(f"{status} {step.title}: {result_preview}")
        
        synthesis_prompt = f"""You are summarizing the results of a multi-step task execution.

Original Goal: {original_goal}

Execution Results:
{chr(10).join(step_summaries)}

Please provide a concise, helpful summary of what was accomplished. 
Focus on the key outcomes and any important information discovered.
Be conversational and natural in your response."""

        try:
            from langchain_core.messages import HumanMessage as LCHumanMessage
            response = await self._llm.ainvoke([LCHumanMessage(content=synthesis_prompt)])
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.warning(f"Failed to synthesize response: {e}")
            # Fallback
            results = [f"• {s.title}: {s.result if s.result else 'done'}" 
                      for s in plan.completed_steps]
            return "Here's what I accomplished:\n\n" + "\n".join(results)
    
    async def _process_with_react(
        self,
        context: AgentContext,
        start_time: float,
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Process using the standard REACT workflow (existing implementation).
        
        This is the original REACT graph-based processing, extracted to a separate
        method for use as a fallback when TaskPlanner isn't suitable.
        """
        # PRIORITY 1: First search persona files and vector memory for relevant data
        shared_context = shared_context or {}
        rag_context = shared_context.get("rag_context", "")
        web_context = shared_context.get("web_context", "")
        user_preferences_context = ""

        if self.memory and hasattr(self.memory, 'agentic_rag') and self.memory.agentic_rag:
            try:
                prefs = await self.memory.agentic_rag.get_user_preferences(context.user_message)
                if prefs:
                    user_preferences_context = "\n".join([
                        f"- {p.get('content', '')}" for p in prefs[:5]
                    ])
            except Exception as e:
                logger.warning(f"Failed to get user preferences: {e}")
        
        user_message = HumanMessage(content=context.user_message)
        history_messages = self._conversation_history[-self._max_history:]
        
        combined_preferences = context.user_preferences or {}
        if user_preferences_context:
            combined_preferences["mem0_preferences"] = user_preferences_context
        
        initial_state: AgentState = {
            "messages": history_messages + [user_message],
            "context": {
                "screen_text": (
                    (shared_context.get("screen_context") or context.screen_context or {}).get("text")
                ),
                "active_window": shared_context.get("active_window") or context.active_window,
                "user_preferences": combined_preferences,
                "rag_context": rag_context,
                "web_context": web_context,
            },
            "thoughts": [],
            "tools_used": [],
            "iteration": 0,
            "max_iterations": self._max_iterations,
            "plan": [],
            "current_step": 0,
            "completed_steps": [],
            "consecutive_errors": 0,
            "needs_replan": False,
            "last_tool_error": None,
        }
        
        try:
            if self._compiled_graph is None:
                logger.warning("No compiled graph, using direct LLM")
                return await self._direct_llm_response(context, start_time)
            
            final_state = await self._compiled_graph.ainvoke(initial_state)
            
            last_ai_message = None
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg
                    break
            
            answer = last_ai_message.content if last_ai_message else ""
            if not answer:
                last_tool_message = None
                for msg in reversed(final_state["messages"]):
                    if isinstance(msg, ToolMessage):
                        last_tool_message = msg
                        break
                
                if final_state.get("iteration", 0) >= final_state.get("max_iterations", 0):
                    answer = "I paused after reaching the maximum planning/execution steps. "
                else:
                    answer = "I couldn't generate a final response. "
                
                if last_tool_message:
                    answer += f"Last tool output ({last_tool_message.name}): {last_tool_message.content[:5000]}"
                else:
                    answer += "Please clarify if you'd like me to continue or adjust the plan."
            
            if last_ai_message:
                self._append_history(user_message, last_ai_message)
            else:
                self._append_history(user_message)
            
            thoughts = [
                AgentThought(
                    step=t.get("step", 0),
                    thought=t.get("content", t.get("observation", "")),
                    action=t.get("tool"),
                    timestamp=datetime.fromisoformat(t["timestamp"]) if "timestamp" in t else datetime.now(),
                )
                for t in final_state["thoughts"]
            ]
            
            latency = (time.time() - start_time) * 1000
            
            if self.memory:
                await self.memory.store_interaction(
                    user_message=context.user_message,
                    agent_response=answer,
                    tools_used=final_state["tools_used"],
                )
                
                if hasattr(self.memory, 'rag_store_session'):
                    import uuid
                    session_messages = [
                        {"role": "user", "content": context.user_message},
                        {"role": "assistant", "content": answer}
                    ]
                    try:
                        await self.memory.rag_store_session(
                            session_id=str(uuid.uuid4())[:8],
                            messages=session_messages,
                            metadata={"tools_used": final_state["tools_used"]}
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store RAG session: {e}")
                
                await self._extract_and_store_preferences(
                    user_message=context.user_message,
                    agent_response=answer
                )
            
            self.status = AgentStatus.IDLE
            await self._update_workflow_state(status=self.status.value)
            
            return AgentResponse(
                answer=answer,
                thoughts=thoughts,
                tools_used=final_state["tools_used"],
                latency_ms=latency,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"REACT processing error: {e}")
            return await self._direct_llm_response(context, start_time)
    
    async def _direct_llm_response(self, context: AgentContext, start_time: float) -> AgentResponse:
        """
        Generate a response directly from the LLM without the graph.
        Used as fallback when the graph isn't working.
        """
        if not self._llm:
            return AgentResponse(
                answer="I'm sorry, but I'm not properly initialized. Please check the LLM configuration.",
                success=False,
                error="LLM not initialized",
                latency_ms=(time.time() - start_time) * 1000,
            )
        
        try:
            # Get current model name
            current_model = "unknown"
            if hasattr(self._llm, 'model'):
                current_model = self._llm.model
            
            # Build prompt using persona files (same as _get_system_prompt)
            system_prompt = await self._get_system_prompt({
                "active_window": getattr(context, 'active_window', None),
                "screen_text": context.screen_context.get("text") if context.screen_context else None,
                "user_preferences": getattr(context, 'user_preferences', None),
            })
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context.user_message)
            ]
            
            # Call the LLM directly
            response = await self._llm.ainvoke(messages)
            
            answer = response.content if hasattr(response, 'content') else str(response)
            
            latency = (time.time() - start_time) * 1000
            
            self.status = AgentStatus.IDLE
            await self._update_workflow_state(status=self.status.value)
            
            return AgentResponse(
                answer=answer,
                thoughts=[],
                tools_used=[],
                latency_ms=latency,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Direct LLM error: {e}")
            raise
    
    async def can_handle(self, context: AgentContext) -> float:
        """Orchestrator can handle anything."""
        return 1.0
    
    async def change_model(self, model_name: str, provider: Optional[str] = None) -> bool:
        """
        Dynamically change the LLM model.
        
        Args:
            model_name: Name of the model to switch to
            provider: Optional provider override (copilot, openai, anthropic)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Changing model to: {model_name} (provider: {provider or 'auto'})")
            
            new_config = {
                "model": model_name,
                "temperature": 0.1,
            }
            if provider:
                new_config["provider"] = provider
            
            new_provider = LLMProvider(new_config)
            await new_provider.initialize()
            
            # Swap the provider
            self._llm_provider = new_provider
            self._llm = new_provider.llm
            
            # Rebuild the graph with new LLM
            self._build_graph()
            
            logger.info(f"Successfully changed to model: {model_name} via {new_provider.active_provider}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to change model: {e}")
            return False
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the current LLM configuration."""
        if self._llm_provider:
            return {
                "provider": self._llm_provider.active_provider,
                "model": getattr(self._llm, "model_name", self.MODEL_NAME),
                "available_providers": self._llm_provider.available_providers,
            }
        return {"provider": "unknown", "model": self.MODEL_NAME}

    def get_workflow_state(self) -> Dict[str, Any]:
        """Get a copy of the current workflow state for UI."""
        return dict(self._workflow_state)
    
    async def _extract_and_store_preferences(
        self,
        user_message: str,
        agent_response: str
    ) -> None:
        """
        Extract user preferences from conversation and store in Mem0.
        
        This enables the agent to learn and remember:
        - User's preferred coding style, language preferences
        - Common tools and workflows the user uses
        - User's communication preferences
        - Important information shared by the user
        """
        if not self.memory or not hasattr(self.memory, 'agentic_rag'):
            return
        
        rag = self.memory.agentic_rag
        if not rag or not rag._mem0:
            return
        
        # Keywords that indicate user preferences
        preference_indicators = [
            "i prefer", "i like", "always use", "i want", "my favorite",
            "i usually", "i always", "please use", "don't use", "never use",
            "i work with", "my name is", "call me", "i'm a", "i am a",
            "my project", "i code in", "my style", "i need"
        ]
        
        message_lower = user_message.lower()
        
        # Check if message contains preference indicators
        for indicator in preference_indicators:
            if indicator in message_lower:
                # Store this as a user preference
                try:
                    await rag._mem0_add(
                        f"User stated: {user_message}",
                        metadata={
                            "type": "user_preference",
                            "indicator": indicator,
                            "timestamp": datetime.now().isoformat(),
                            "context": agent_response[:200]  # Include some response context
                        }
                    )
                    logger.debug(f"Stored user preference (indicator: {indicator})")
                except Exception as e:
                    logger.debug(f"Failed to store preference: {e}")
                break  # Only store once per message
