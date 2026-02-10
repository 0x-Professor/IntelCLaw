"""
Task Planner - Autonomous multi-step task planning and execution.

This module implements a sophisticated task planning system that:
1. Analyzes user requests and breaks them into actionable steps
2. Determines dependencies between steps
3. Identifies which steps can run in parallel
4. Integrates web search and RAG for information gathering
5. Tracks progress and handles failures with replanning
"""

from __future__ import annotations

import asyncio
import re
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from langchain_core.messages import HumanMessage
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from intelclaw.integrations.llm_provider import LLMProvider
    from intelclaw.tools.registry import ToolRegistry
    from intelclaw.memory.manager import MemoryManager


class TaskStatus(str, Enum):
    """Status of a task step."""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class TaskPriority(str, Enum):
    """Priority level for tasks."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskStep:
    """
    A single step in a multi-step task plan.
    
    Attributes:
        id: Unique identifier for this step
        title: Short description of the step
        description: Detailed description of what needs to be done
        status: Current execution status
        tool: Optional tool to use for execution
        tool_args: Arguments to pass to the tool
        dependencies: IDs of steps that must complete before this one
        result: Output from execution
        error: Error message if failed
        started_at: When execution started
        completed_at: When execution completed
        retries: Number of retry attempts
        max_retries: Maximum allowed retries
    """
    id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    tool: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 2
    priority: TaskPriority = TaskPriority.NORMAL
    requires_confirmation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "tool": self.tool,
            "tool_args": self.tool_args,
            "dependencies": self.dependencies,
            "result": self.result if self.result else None,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retries": self.retries,
            "priority": self.priority.value,
        }


@dataclass  
class TaskPlan:
    """
    A complete task plan with multiple steps.
    
    Attributes:
        id: Unique plan identifier
        goal: The original user goal/request
        steps: Ordered list of task steps
        created_at: When the plan was created
        status: Overall plan status
        context: Additional context for execution
    """
    id: str
    goal: str
    steps: List[TaskStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def current_step(self) -> Optional[TaskStep]:
        """Get the current step being executed."""
        for step in self.steps:
            if step.status == TaskStatus.IN_PROGRESS:
                return step
        return None
    
    @property
    def next_step(self) -> Optional[TaskStep]:
        """Get the next step to execute."""
        for step in self.steps:
            if step.status == TaskStatus.PENDING:
                # Check if all dependencies are complete
                deps_complete = all(
                    self.get_step(dep_id) and self.get_step(dep_id).status == TaskStatus.COMPLETED
                    for dep_id in step.dependencies
                )
                if deps_complete:
                    return step
        return None
    
    @property
    def pending_steps(self) -> List[TaskStep]:
        """Get all pending steps."""
        return [s for s in self.steps if s.status == TaskStatus.PENDING]
    
    @property
    def completed_steps(self) -> List[TaskStep]:
        """Get all completed steps."""
        return [s for s in self.steps if s.status == TaskStatus.COMPLETED]
    
    @property
    def failed_steps(self) -> List[TaskStep]:
        """Get all failed steps."""
        return [s for s in self.steps if s.status == TaskStatus.FAILED]
    
    @property
    def progress_percent(self) -> float:
        """Calculate completion percentage."""
        if not self.steps:
            return 0.0
        completed = len([s for s in self.steps if s.status == TaskStatus.COMPLETED])
        return (completed / len(self.steps)) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if the plan is fully executed."""
        return all(s.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED) for s in self.steps)
    
    @property
    def has_failed(self) -> bool:
        """Check if the plan has any failed steps."""
        return any(s.status == TaskStatus.FAILED for s in self.steps)
    
    def get_step(self, step_id: str) -> Optional[TaskStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def get_runnable_steps(self) -> List[TaskStep]:
        """Get all steps that can run now (dependencies satisfied)."""
        runnable = []
        completed_ids = {s.id for s in self.steps if s.status == TaskStatus.COMPLETED}
        
        for step in self.steps:
            if step.status != TaskStatus.PENDING:
                continue
            
            # Check dependencies
            deps_satisfied = all(dep_id in completed_ids for dep_id in step.dependencies)
            if deps_satisfied:
                runnable.append(step)
        
        return runnable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "progress": self.progress_percent,
            "current_step": self.current_step.to_dict() if self.current_step else None,
            "next_step": self.next_step.to_dict() if self.next_step else None,
        }


class TaskPlanner:
    """
    Intelligent task planner that creates and manages multi-step execution plans.
    
    Features:
    - Breaks down complex requests into actionable steps
    - Identifies tool requirements for each step
    - Determines step dependencies and parallelization opportunities
    - Integrates web search for information gathering
    - Supports replanning on failure
    """
    
    # Planning prompts
    PLANNING_SYSTEM_PROMPT = """You are an expert task planner for an autonomous Windows AI agent.
Your job is to break down user requests into clear, actionable steps.

For each step, determine:
1. A clear, concise title (what to do)
2. A detailed description (how to do it)
3. The EXACT tool name to use (from the available tools list - use EXACTLY as shown)
4. Dependencies on other steps

AVAILABLE TOOLS (use these EXACT names):
{tools}

IMPORTANT TOOL MAPPING:

### File Operations
- For reading files: use "file_read" with args {{"path": "file/path"}}
- For writing files: use "file_write" with args {{"path": "file/path", "content": "..."}}
- For deleting files: use "file_delete" with args {{"path": "file/path"}}
- For copying files: use "file_copy" with args {{"source": "src", "destination": "dst"}}
- For moving/renaming: use "file_move" with args {{"source": "old", "destination": "new"}}
- For finding files: use "file_search" with args {{"directory": ".", "pattern": "*.txt"}}
- For listing directories: use "list_directory" with args {{"path": "."}}
- For current directory: use "get_cwd"

### Search & Web
- For web search: use "tavily_search" with args {{"query": "search terms"}}
- For web scraping: use "web_scrape" with args {{"url": "..."}}

### Contacts
- To look up a saved contact (data/contacts.md): use "contacts_lookup" with args {{"query": "Alice"}}
- To add/update a contact: use "contacts_upsert" with args {{"name": "Alice", "phone": "+923171156353", "gender": "female", "persona": "Friend. Keep tone casual.", "inbound_allowed": true, "resolve_whatsapp": true}}
- To enable/disable inbound auto-replies: use "contacts_set_inbound_allowed" with args {{"query": "Alice", "allowed": true}}

### Shell & Code
- For running commands: use "shell_command" with args {{"command": "..."}}
- For PowerShell: use "powershell" with args {{"script": "..."}}

### Process & Task Management (Windows)
- For listing processes/tasks: use "process_management" with args {{"action": "list"}}
- For finding a process: use "process_management" with args {{"action": "find", "name": "process_name"}}
- For process details: use "process_management" with args {{"action": "details", "pid": 1234}}
- For killing a process: use "process_management" with args {{"action": "kill", "pid": 1234}}
- For top CPU consumers: use "process_management" with args {{"action": "top_cpu"}}
- For top memory consumers: use "process_management" with args {{"action": "top_memory"}}

### Windows Services
- For listing services: use "windows_services" with args {{"action": "list"}}
- For service details: use "windows_services" with args {{"action": "get", "name": "service_name"}}
- For start/stop/restart: use "windows_services" with args {{"action": "start", "name": "service_name"}}

### Scheduled Tasks
- For listing tasks: use "windows_tasks" with args {{"action": "list"}}
- For creating tasks: use "windows_tasks" with args {{"action": "create", "name": "...", "action_path": "...", "schedule": "daily", "start_time": "09:00"}}

### Registry
- For reading registry: use "windows_registry" with args {{"action": "get", "path": "HKLM:\\\\...", "name": "value"}}
- For listing keys: use "windows_registry" with args {{"action": "list_keys", "path": "HKLM:\\\\..."}}

### Event Logs
- For querying events: use "windows_eventlog" with args {{"action": "query", "log_name": "System"}}
- For listing log names: use "windows_eventlog" with args {{"action": "list_logs"}}

### Network
- For network adapters: use "network_info" with args {{"action": "adapters"}}
- For active connections: use "network_info" with args {{"action": "connections"}}
- For listening ports: use "network_info" with args {{"action": "listening"}}
- For ping: use "network_info" with args {{"action": "ping", "target": "host"}}
- For public IP: use "network_info" with args {{"action": "public_ip"}}

### Disk & Storage
- For disk space: use "disk_management" with args {{"action": "space"}}
- For volumes: use "disk_management" with args {{"action": "volumes"}}
- For large files: use "disk_management" with args {{"action": "large_files", "path": "C:\\\\"}}

### System Performance
- For system overview: use "system_performance" with args {{"action": "overview"}}
- For CPU info: use "system_performance" with args {{"action": "cpu"}}
- For memory info: use "system_performance" with args {{"action": "memory"}}
- For GPU info: use "system_performance" with args {{"action": "gpu"}}
- For hardware: use "system_performance" with args {{"action": "hardware"}}
- For battery: use "system_performance" with args {{"action": "battery"}}
- For uptime: use "system_performance" with args {{"action": "uptime"}}

### Firewall
- For firewall status: use "windows_firewall" with args {{"action": "status"}}
- For listing rules: use "windows_firewall" with args {{"action": "list_rules"}}

### Installed Applications
- For listing apps: use "installed_apps" with args {{"action": "list"}}
- For searching apps: use "installed_apps" with args {{"action": "search", "name": "chrome"}}
- For startup apps: use "installed_apps" with args {{"action": "startup_apps"}}

### Environment Variables
- For listing vars: use "environment_vars" with args {{"action": "list"}}
- For PATH: use "environment_vars" with args {{"action": "path"}}

### Windows Updates
- For pending updates: use "windows_update" with args {{"action": "pending"}}
- For installed updates: use "windows_update" with args {{"action": "installed"}}

### Users & Security
- For current user: use "user_security" with args {{"action": "current_user"}}
- For local users: use "user_security" with args {{"action": "list_users"}}

### WMI/CIM
- For CIM queries: use "windows_cim" with args {{"class_name": "Win32_OperatingSystem"}}

### Windows MCP UI Automation (mcp_windows__*)
- Use MCP tools when available (tool names starting with "mcp_").
- IMPORTANT: For any MCP tool, only use keys that exist in that tool's schema (do not invent keys like "url" unless the schema includes it).
- Launch/switch/resize an app window: use "mcp_windows__app" with args {{"mode": "launch|switch|resize", "name": "chrome"}}
  - Do NOT pass "app" or "url" to "mcp_windows__app" (it doesn't accept those).
- Open a URL in a browser: use "mcp_windows__shell" with args {{"command": "Start-Process chrome 'https://www.youtube.com'"}}
- Run an arbitrary command: use "mcp_windows__shell" with args {{"command": "...", "timeout": 10}}

### WhatsApp MCP (mcp_whatsapp__*)
- Phone numbers must be digits only (no "+", spaces, or symbols). Example: "923171156353"
- Prefer using a chat JID when you have it (e.g. "923171156353@s.whatsapp.net" or "...@lid").
- Resolve recipient by name: use "contacts_lookup" first, then send.
- If the user explicitly asks to send a message, treat sending as approved (do not add a separate confirmation step) unless the recipient is ambiguous.
- Send a message: use "mcp_whatsapp__send_message" with args {{"recipient": "923171156353", "message": "hi"}}
- Never send tool errors/placeholders as the WhatsApp message text.

### Other
- For taking screenshot: use "screenshot"
- For clipboard: use "clipboard" with args {{"action": "read"}}
- For launching apps: use "launch_app" with args {{"target": "app_name"}}
- For system info: use "system_info"

Guidelines:
- Each step should be atomic and achievable
- Identify which steps can run in parallel (no dependencies between them)
- Use tavily_search for web research, not "web_search"
- Include verification/validation steps for important operations
- Maximum {max_steps} steps per plan
- For analysis/synthesis steps without a specific tool, set tool to null
- When user asks about "running tasks" or "processes", use "process_management" tool
- Always prefer specific Windows tools over generic shell_command/powershell

Return ONLY a JSON array with this structure:
[
  {{
    "id": "step_1",
    "title": "Short action title",
    "description": "Detailed description of what to do",
    "tool": "exact_tool_name_or_null",
    "tool_args": {{"arg": "value"}} or null,
    "dependencies": ["step_id"] or [],
    "requires_info": true/false
  }}
]"""

    RESEARCH_PROMPT = """Based on this task, what information do I need to gather first?
Task: {task}

Consider:
- Do I need current/latest information? (use web search)
- Do I need project-specific context? (use RAG/memory)
- Do I need to verify any assumptions?

Return a JSON object:
{{
  "needs_web_search": true/false,
  "web_search_queries": ["query1", "query2"],
  "needs_rag": true/false,
  "rag_queries": ["query1"],
  "assumptions_to_verify": ["assumption1"]
}}"""
    
    def __init__(
        self,
        llm_provider: Optional["LLMProvider"] = None,
        tools: Optional["ToolRegistry"] = None,
        memory: Optional["MemoryManager"] = None,
        max_steps: int = 10,
        enable_web_search: bool = True,
        enable_rag: bool = True,
        step_timeout_secs: int = 120,
        plan_timeout_secs: int = 900,
        stall_timeout_secs: int = 180,
    ):
        self.llm_provider = llm_provider
        self.tools = tools
        self.memory = memory
        self.max_steps = max_steps
        self.enable_web_search = enable_web_search
        self.enable_rag = enable_rag
        self.step_timeout_secs = step_timeout_secs
        self.plan_timeout_secs = plan_timeout_secs
        self.stall_timeout_secs = stall_timeout_secs
        self._last_progress_ts = time.time()
        
        # Active plans
        self._plans: Dict[str, TaskPlan] = {}
        self._current_plan: Optional[str] = None
        
        # Event callbacks
        self._on_plan_created: Optional[Callable[[TaskPlan], None]] = None
        self._on_step_started: Optional[Callable[[TaskStep], None]] = None
        self._on_step_completed: Optional[Callable[[TaskStep], None]] = None
        self._on_plan_completed: Optional[Callable[[TaskPlan], None]] = None
    
    async def create_plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        force_research: bool = False,
    ) -> TaskPlan:
        """
        Create a new task plan for the given goal.
        
        Args:
            goal: The user's goal/request
            context: Additional context for planning
            force_research: Force information gathering step
            
        Returns:
            A TaskPlan with steps to achieve the goal
        """
        plan_id = str(uuid.uuid4())[:8]
        logger.info(f"Creating plan {plan_id} for: {goal[:100]}...")
        
        # Step 1: Determine if we need to gather information first
        research_needed = await self._analyze_research_needs(goal, force_research)
        
        # Step 2: Gather information if needed
        gathered_context = {}
        if research_needed.get("needs_web_search") and self.enable_web_search:
            queries = research_needed.get("web_search_queries", [goal])
            web_results = await self._perform_web_search(queries)
            gathered_context["web_search"] = web_results
        
        if research_needed.get("needs_rag") and self.enable_rag:
            queries = research_needed.get("rag_queries", [goal])
            sid = None
            if isinstance(context, dict):
                sid = context.get("session_id")
            rag_results = await self._perform_rag_search(queries, session_id=str(sid) if sid else None)
            gathered_context["rag"] = rag_results
        
        # Step 3: Generate the plan using LLM
        steps = await self._generate_plan_steps(goal, context, gathered_context)
        
        # Step 4: Create and store the plan
        plan = TaskPlan(
            id=plan_id,
            goal=goal,
            steps=steps,
            context=context or {},
            metadata={
                "research": research_needed,
                "gathered_context": gathered_context,
            }
        )
        
        self._plans[plan_id] = plan
        self._current_plan = plan_id
        
        logger.info(f"Created plan {plan_id} with {len(steps)} steps")
        
        if self._on_plan_created:
            self._on_plan_created(plan)
        
        return plan
    
    async def _analyze_research_needs(
        self,
        goal: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """Analyze what information gathering is needed."""
        if force:
            return {
                "needs_web_search": True,
                "web_search_queries": [goal],
                "needs_rag": True,
                "rag_queries": [goal],
            }
        
        # Quick heuristic check
        goal_lower = goal.lower()
        
        # Keywords indicating need for fresh information
        web_keywords = [
            "latest", "recent", "current", "today", "news", "trend",
            "2024", "2025", "2026", "update", "new", "release"
        ]
        
        needs_web = any(kw in goal_lower for kw in web_keywords)
        
        # Keywords indicating need for project context
        rag_keywords = [
            "project", "codebase", "my code", "existing", "current",
            "our", "this", "the file", "our system"
        ]
        
        needs_rag = any(kw in goal_lower for kw in rag_keywords)
        
        result = {
            "needs_web_search": needs_web,
            "web_search_queries": [goal] if needs_web else [],
            "needs_rag": needs_rag,
            "rag_queries": [goal] if needs_rag else [],
        }
        
        # For complex goals, use LLM to analyze
        if self.llm_provider and len(goal) > 100:
            try:
                llm_analysis = await self._llm_analyze_research(goal)
                result.update(llm_analysis)
            except Exception as e:
                logger.debug(f"LLM research analysis failed: {e}")
        
        return result
    
    async def _llm_analyze_research(self, goal: str) -> Dict[str, Any]:
        """Use LLM to analyze research needs."""
        if not self.llm_provider or not self.llm_provider.llm:
            return {}
        
        from langchain_core.messages import SystemMessage, HumanMessage
        
        prompt = self.RESEARCH_PROMPT.format(task=goal)
        
        try:
            response = await self.llm_provider.llm.ainvoke([
                SystemMessage(content="You are a research planning assistant. Return valid JSON only."),
                HumanMessage(content=prompt),
            ])
            
            content = response.content if hasattr(response, "content") else str(response)
            
            # Parse JSON from response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(content[json_start:json_end])
        except Exception as e:
            logger.debug(f"Research analysis parsing failed: {e}")
        
        return {}
    
    async def _perform_web_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Perform web search for the given queries."""
        results = []
        
        if not self.tools:
            return results
        
        for query in queries[:3]:  # Limit to 3 queries
            try:
                search_result = await self.tools.execute(
                    "tavily_search",
                    {"query": query, "max_results": 3}
                )
                payload = self._unwrap_tool_data(search_result)
                if payload:
                    results.extend(payload if isinstance(payload, list) else [payload])
            except Exception as e:
                logger.debug(f"Web search failed for '{query}': {e}")
        
        return results
    
    async def _perform_rag_search(
        self, queries: List[str], *, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform RAG search for the given queries."""
        results = []
        
        if not self.memory or not hasattr(self.memory, "get_rag_context"):
            return results
        
        for query in queries[:2]:  # Limit to 2 queries
            try:
                context = await self.memory.get_rag_context(
                    query=query,
                    include_persona=True,
                    max_context_chars=2000,
                    session_id=session_id,
                )
                if context:
                    results.append({"query": query, "context": context})
            except Exception as e:
                logger.debug(f"RAG search failed for '{query}': {e}")
        
        return results
    
    async def _generate_plan_steps(
        self,
        goal: str,
        context: Optional[Dict[str, Any]],
        gathered_context: Dict[str, Any],
    ) -> List[TaskStep]:
        """Generate plan steps using LLM."""
        if not self.llm_provider or not self.llm_provider.llm:
            # Fallback: single-step plan
            return [TaskStep(
                id="step_1",
                title="Execute request",
                description=goal,
                status=TaskStatus.PENDING,
            )]
        
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Get available tools with descriptions
        tool_info = []
        if self.tools:
            try:
                tool_defs = self.tools.list_tools()
                for td in tool_defs:
                    desc = " ".join(str(td.description or "").split())
                    line = f"- {td.name}: {desc[:150]}"
                    if str(td.name or "").startswith("mcp_"):
                        try:
                            schema = json.dumps(td.parameters, ensure_ascii=False)
                            if len(schema) > 500:
                                schema = schema[:500] + "..."
                            line += f" | schema: {schema}"
                        except Exception:
                            pass
                    tool_info.append(line)
            except Exception as e:
                logger.debug(f"Failed to get tools: {e}")
        
        tools_str = "\n".join(tool_info) if tool_info else "No tools available"
        
        # Build context string
        context_str = ""
        if gathered_context.get("web_search"):
            context_str += "\n\nWeb Search Results:\n"
            for item in gathered_context["web_search"][:5]:
                title = item.get("title", "")
                snippet = item.get("content", item.get("snippet", ""))[:200]
                context_str += f"- {title}: {snippet}\n"
        
        if gathered_context.get("rag"):
            context_str += "\n\nProject Context:\n"
            for item in gathered_context["rag"]:
                context_str += item.get("context", "")[:500] + "\n"

        # Include recent raw conversation (no summarization) when provided.
        if isinstance(context, dict):
            recent = context.get("conversation_recent")
            if recent:
                context_str += "\n\nRecent Conversation:\n"
                context_str += str(recent)[:2000] + "\n"
            instructions = context.get("agent_instructions")
            if instructions:
                context_str += "\n\nAgent Instructions:\n"
                context_str += str(instructions)[:2000] + "\n"
        
        system_prompt = self.PLANNING_SYSTEM_PROMPT.format(
            tools=tools_str,
            max_steps=self.max_steps,
        )
        
        user_prompt = f"Create a step-by-step plan for: {goal}"
        if context_str:
            user_prompt += f"\n\nAvailable Context:{context_str}"
        
        try:
            response = await self.llm_provider.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            
            content = response.content if hasattr(response, "content") else str(response)
            steps = self._parse_plan_response(content)
            
            if steps:
                # Normalize/validate tool names against registry
                if self.tools:
                    try:
                        available_tools = {t.name for t in self.tools.list_tools()}
                    except Exception:
                        available_tools = set()
                    
                    if available_tools:
                        for step in steps:
                            if not step.tool:
                                continue
                            
                            normalized = step.tool
                            for prefix in ("functions.", "tools.", "tool."):
                                if normalized.startswith(prefix):
                                    normalized = normalized[len(prefix):]
                                    break
                            
                            # Preserve Codex parallel wrapper; warn if no args provided
                            if normalized == "multi_tool_use.parallel":
                                step.tool = "multi_tool_use.parallel"
                                if not step.tool_args:
                                    logger.warning(
                                        f"Plan step '{step.title}' uses multi_tool_use.parallel without tool_args"
                                    )
                                    step.tool_args = step.tool_args or {}
                                continue
                            
                            # Use normalized tool name if available
                            if normalized in available_tools:
                                step.tool = normalized
                                continue
                            
                            # Unknown tool: fall back to LLM execution
                            if step.tool not in available_tools:
                                logger.warning(
                                    f"Unknown tool in plan '{step.tool}' for step '{step.title}', falling back to LLM"
                                )
                                step.tool = None
                
                return steps
        except Exception as e:
            logger.warning(f"Plan generation failed: {e}")
        
        # Fallback
        return [TaskStep(
            id="step_1",
            title="Execute request",
            description=goal,
            status=TaskStatus.PENDING,
        )]
    
    def _parse_plan_response(self, content: str) -> List[TaskStep]:
        """Parse LLM response into TaskSteps."""
        steps = []
        
        try:
            # Find JSON array in response
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            
            if json_start != -1 and json_end > json_start:
                parsed = json.loads(content[json_start:json_end])
                
                for idx, item in enumerate(parsed):
                    if not isinstance(item, dict):
                        continue
                    
                    step = TaskStep(
                        id=item.get("id", f"step_{idx + 1}"),
                        title=item.get("title", f"Step {idx + 1}"),
                        description=item.get("description", ""),
                        tool=item.get("tool"),
                        tool_args=item.get("tool_args"),
                        dependencies=item.get("dependencies", []),
                        status=TaskStatus.PENDING,
                    )
                    steps.append(step)
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse plan JSON: {e}")
        except Exception as e:
            logger.debug(f"Plan parsing error: {e}")
        
        return steps[:self.max_steps]
    
    async def execute_plan(
        self,
        plan: TaskPlan,
        executor: Optional[Callable[[TaskStep], Any]] = None,
        parallel: bool = True,
    ) -> TaskPlan:
        """
        Execute a task plan step by step.
        
        Args:
            plan: The plan to execute
            executor: Callable to execute each step
            parallel: Whether to run independent steps in parallel
            
        Returns:
            The completed plan with results
        """
        plan.status = TaskStatus.IN_PROGRESS
        plan_start = time.time()
        self._last_progress_ts = plan_start

        try:
            while not plan.is_complete and not plan.has_failed:
                # Plan timeout guard
                if self.plan_timeout_secs and (time.time() - plan_start) > self.plan_timeout_secs:
                    for step in plan.pending_steps:
                        step.status = TaskStatus.FAILED
                        step.error = f"Plan timed out after {self.plan_timeout_secs} seconds"
                    plan.status = TaskStatus.FAILED
                    break

                # Stall timeout guard (no progress)
                if self.stall_timeout_secs and (time.time() - self._last_progress_ts) > self.stall_timeout_secs:
                    for step in plan.pending_steps:
                        step.status = TaskStatus.FAILED
                        step.error = f"Plan stalled after {self.stall_timeout_secs} seconds without progress"
                    plan.status = TaskStatus.FAILED
                    break

                runnable = plan.get_runnable_steps()
            
                if not runnable:
                    # No more steps can run - either done or blocked
                    if plan.pending_steps:
                        # Steps are blocked
                        for step in plan.pending_steps:
                            step.status = TaskStatus.BLOCKED
                    break
                
                if parallel and len(runnable) > 1:
                    # Execute in parallel
                    await asyncio.gather(*[
                        self._execute_step(step, plan, executor)
                        for step in runnable
                    ])
                else:
                    # Execute sequentially
                    for step in runnable:
                        await self._execute_step(step, plan, executor)
                        
                        # Check for failure
                        if step.status == TaskStatus.FAILED and step.retries >= step.max_retries:
                            break
        except asyncio.CancelledError:
            for step in plan.pending_steps:
                step.status = TaskStatus.FAILED
                step.error = "Plan execution cancelled"
            plan.status = TaskStatus.FAILED
        
        # Determine final plan status
        if plan.is_complete:
            plan.status = TaskStatus.COMPLETED
        elif plan.has_failed:
            plan.status = TaskStatus.FAILED
        
        if self._on_plan_completed:
            self._on_plan_completed(plan)
        
        return plan

    def _rewrite_common_tool_invocations(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Best-effort rewrite of common tool-call mistakes.

        This is intentionally conservative and only handles a small set of
        high-frequency issues (mainly around MCP tools) to prevent permanent
        failures on otherwise valid plans.
        """
        name = str(tool_name or "").strip()
        args: Dict[str, Any] = dict(tool_args or {})
        if not name:
            return name, args

        # Legacy/generic: Some planners call a non-MCP `app_tool` with {app, url}
        # to mean "open <url> in <app>". Rewrite to an actually supported tool.
        if name == "app_tool":
            if "app" in args and "name" not in args:
                args["name"] = args.pop("app")

            url = args.get("url")
            if url:
                url_str = str(url)
                browser = str(args.get("name") or "chrome")

                try:
                    if self.tools and hasattr(self.tools, "get_tool") and self.tools.get_tool("mcp_windows__shell"):
                        return "mcp_windows__shell", {"command": f"Start-Process {browser} '{url_str}'"}
                except Exception:
                    pass

                try:
                    if self.tools and hasattr(self.tools, "get_tool") and self.tools.get_tool("powershell"):
                        return "powershell", {"script": f"Start-Process {browser} '{url_str}'"}
                except Exception:
                    pass

                try:
                    if self.tools and hasattr(self.tools, "get_tool") and self.tools.get_tool("launch_app"):
                        return "launch_app", {"target": url_str}
                except Exception:
                    pass

                # If we can't rewrite, at least strip url to avoid schema validation errors.
                args.pop("url", None)
            return name, args

        # Windows MCP: App tool does not accept URL navigation.
        if name == "mcp_windows__app":
            # Common mistake: app -> name
            if "app" in args and "name" not in args:
                args["name"] = args.pop("app")

            # Common mistake: trying to pass url/app to the App tool.
            url = args.get("url")
            if url:
                url_str = str(url)
                browser = str(args.get("name") or "chrome")

                # Prefer MCP shell if available
                try:
                    if self.tools and hasattr(self.tools, "get_tool") and self.tools.get_tool("mcp_windows__shell"):
                        return "mcp_windows__shell", {"command": f"Start-Process {browser} '{url_str}'"}
                except Exception:
                    pass

                # Fall back to PowerShell tool
                try:
                    if self.tools and hasattr(self.tools, "get_tool") and self.tools.get_tool("powershell"):
                        return "powershell", {"script": f"Start-Process {browser} '{url_str}'"}
                except Exception:
                    pass

                # Last resort: launch_app can open URLs via default browser
                try:
                    if self.tools and hasattr(self.tools, "get_tool") and self.tools.get_tool("launch_app"):
                        return "launch_app", {"target": url_str}
                except Exception:
                    pass

                # Can't rewrite; strip the invalid key to avoid server-side validation errors.
                args.pop("url", None)
                return name, args

        # Windows MCP: Shell tool expects `command`, but LLMs sometimes use `script`/`cmd`.
        if name == "mcp_windows__shell":
            if "command" not in args:
                if "script" in args:
                    args["command"] = args.pop("script")
                elif "cmd" in args:
                    args["command"] = args.pop("cmd")
            if "timeout" not in args and "timeout_seconds" in args:
                args["timeout"] = args.pop("timeout_seconds")
            return name, args

        # WhatsApp MCP: normalize recipient phone numbers (strip '+' and formatting).
        if name in {"mcp_whatsapp__send_message", "mcp_whatsapp__send_file", "mcp_whatsapp__send_audio_message"}:
            if "recipient" not in args:
                for alt in (
                    "to",
                    "phone",
                    "phone_number",
                    "jid",
                    "chat_jid",
                    "recipient_jid",
                    "contact",
                    "contact_name",
                    "name",
                    "recipient_name",
                    "person",
                    "user",
                    "target",
                ):
                    if alt in args:
                        args["recipient"] = args.pop(alt)
                        break
            if "message" not in args:
                for alt in ("text", "content", "body", "msg"):
                    if alt in args:
                        args["message"] = args.pop(alt)
                        break
            rec = args.get("recipient")
            if isinstance(rec, str):
                rec_s = rec.strip()
                if rec_s and "@" not in rec_s:
                    digits = re.sub(r"\\D+", "", rec_s)
                    if digits:
                        args["recipient"] = digits
                    else:
                        # Best-effort: treat as a contact name and resolve from contacts.md.
                        try:
                            from pathlib import Path

                            from intelclaw.contacts.store import ContactsStore

                            store = ContactsStore(Path("data") / "contacts.md")
                            matches = store.lookup(rec_s)
                            if matches:
                                exact = [m for m in matches if (m.name or "").strip().lower() == rec_s.lower()]
                                m = exact[0] if exact else matches[0]
                                args["recipient"] = m.whatsapp_jid or m.phone
                        except Exception:
                            pass
            return name, args

        return name, args

    @staticmethod
    def _try_parse_json(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return None
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    def _infer_whatsapp_recipient_from_plan(self, plan: TaskPlan, step: TaskStep) -> Optional[str]:
        """
        When an LLM omits the required `recipient` argument for WhatsApp send tools,
        best-effort infer it from earlier contacts_lookup results.
        """
        haystack = f"{step.title}\n{step.description}".lower()
        candidates: List[tuple[str, str]] = []

        for s in reversed(plan.completed_steps):
            if s.tool != "contacts_lookup" or not s.result:
                continue
            parsed = self._try_parse_json(s.result)
            if not parsed:
                continue
            rows = parsed if isinstance(parsed, list) else [parsed]
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name") or "").strip()
                recipient = str(row.get("whatsapp_jid") or row.get("phone") or "").strip()
                if not recipient:
                    continue
                candidates.append((name, recipient))

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][1]

        # If we have multiple, try to match the name embedded in the step title/description.
        matches = []
        for name, recipient in candidates:
            if name and name.lower() in haystack:
                matches.append(recipient)

        if len(matches) == 1:
            return matches[0]

        # Ambiguous; do not guess.
        return None

    def _infer_whatsapp_message_from_plan(self, plan: TaskPlan) -> Optional[str]:
        """
        Best-effort infer a WhatsApp message body from a prior "draft" step.
        """
        for s in reversed(plan.completed_steps):
            t = str(s.title or "").lower()
            if "draft" not in t and "compose" not in t and "write" not in t:
                continue
            if not s.result:
                continue
            msg = str(s.result).strip()
            if msg:
                return msg
        return None
    
    async def _execute_step(
        self,
        step: TaskStep,
        plan: TaskPlan,
        executor: Optional[Callable[[TaskStep], Any]],
    ) -> None:
        """Execute a single step."""
        step.status = TaskStatus.IN_PROGRESS
        step.started_at = datetime.now()
        self._last_progress_ts = time.time()
        
        if self._on_step_started:
            self._on_step_started(step)
        
        try:
            async def run_step() -> Any:
                if executor:
                    # Use custom executor
                    return await executor(step) if asyncio.iscoroutinefunction(executor) else executor(step)
                if step.tool and self.tools:
                    # Execute the specified tool
                    logger.info(f"Executing tool {step.tool} for step: {step.title}")
                    
                    # Special handling for file_write - populate content from previous steps
                    tool_args = dict(step.tool_args) if step.tool_args else {}
                    if step.tool == "file_write":
                        content = tool_args.get("content", "")
                        # If content is a placeholder or too short, synthesize from completed steps
                        if not content or len(content) < 50 or "will be generated" in content.lower() or "..." in content:
                            logger.info("File write content is placeholder, synthesizing from completed steps...")
                            synthesized_content = await self._synthesize_content_for_file(plan, step)
                            if synthesized_content:
                                tool_args["content"] = synthesized_content

                    tool_name = step.tool
                    rewritten_name, rewritten_args = self._rewrite_common_tool_invocations(tool_name, tool_args)
                    if rewritten_name != tool_name or rewritten_args != tool_args:
                        logger.info(
                            f"Rewriting tool call for step {step.id}: {tool_name} -> {rewritten_name}"
                        )
                        step.tool = rewritten_name
                        step.tool_args = dict(rewritten_args or {})
                        tool_name = rewritten_name
                        tool_args = dict(rewritten_args or {})

                    # Backfill missing WhatsApp send args from previous steps (common LLM omission).
                    if tool_name in {"mcp_whatsapp__send_message", "mcp_whatsapp__send_file", "mcp_whatsapp__send_audio_message"}:
                        if not tool_args.get("recipient"):
                            inferred = self._infer_whatsapp_recipient_from_plan(plan, step)
                            if inferred:
                                tool_args["recipient"] = inferred
                                step.tool_args = dict(tool_args)
                        if tool_name == "mcp_whatsapp__send_message" and not tool_args.get("message"):
                            inferred_msg = self._infer_whatsapp_message_from_plan(plan)
                            if inferred_msg:
                                tool_args["message"] = inferred_msg
                                step.tool_args = dict(tool_args)

                    return await self.tools.execute(tool_name, tool_args)

                # No specific tool - use LLM to reason and execute
                logger.info(f"Using LLM to execute step: {step.title}")
                return await self._llm_execute_step(step, plan)

            if self.step_timeout_secs:
                result = await asyncio.wait_for(run_step(), timeout=self.step_timeout_secs)
            else:
                result = await run_step()

            if step.tool and self.tools:
                step.result = self._format_tool_result(result)
            else:
                step.result = str(result) if result else "Completed"
            
            step.status = TaskStatus.COMPLETED
            logger.info(f"Step completed: {step.title} - {step.result[:2000] if step.result else 'done'}")
            
        except asyncio.TimeoutError:
            step.error = f"Step timed out after {self.step_timeout_secs} seconds"
            step.retries += 1
            
            if step.retries < step.max_retries:
                step.status = TaskStatus.PENDING  # Will retry
                logger.warning(f"Step {step.id} timed out, will retry ({step.retries}/{step.max_retries})")
            else:
                step.status = TaskStatus.FAILED
                logger.error(f"Step {step.id} timed out permanently")
        except Exception as e:
            step.error = str(e)
            step.retries += 1
            
            if step.retries < step.max_retries:
                step.status = TaskStatus.PENDING  # Will retry
                logger.warning(f"Step {step.id} failed, will retry ({step.retries}/{step.max_retries}): {e}")
            else:
                step.status = TaskStatus.FAILED
                logger.error(f"Step {step.id} failed permanently: {e}")
        
        step.completed_at = datetime.now()
        self._last_progress_ts = time.time()
        
        if self._on_step_completed:
            self._on_step_completed(step)

    @staticmethod
    def _unwrap_tool_data(result: Any) -> Any:
        """Extract the data payload from structured tool results."""
        if isinstance(result, dict) and "data" in result and "success" in result:
            return result.get("data")
        return result

    @staticmethod
    def _format_tool_result(result: Any) -> str:
        """Format tool results for storage without truncation."""
        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result, ensure_ascii=False)
            except Exception:
                return str(result)
        return str(result) if result is not None else ""
    
    async def _llm_execute_step(self, step: TaskStep, plan: TaskPlan) -> str:
        """
        Execute a step using the LLM when no specific tool is assigned.
        
        This method allows the LLM to reason about the step and either:
        1. Use available tools to complete it
        2. Generate content/analysis directly
        
        Args:
            step: The step to execute
            plan: The parent plan for context
            
        Returns:
            Result of the step execution
        """
        if not self.llm_provider or not self.llm_provider.llm:
            return "LLM not available for step execution"
        
        llm = self.llm_provider.llm
        
        # Build context from completed steps - include full results for synthesis
        completed_context = []
        full_results = {}  # Store full results for file writing
        for s in plan.completed_steps:
            # For synthesis/write steps, provide more context
            if step.title.lower().find("write") >= 0 or step.title.lower().find("save") >= 0 or step.title.lower().find("synthesize") >= 0 or step.title.lower().find("analyze") >= 0:
                # Include full result for write/synthesis steps
                completed_context.append(f"- {s.title}:\n{s.result if s.result else 'done'}")
                full_results[s.id] = s.result
            else:
                # Include full results for proper context
                completed_context.append(f"- {s.title}: {s.result if s.result else 'done'}")
        
        # Get available tools
        available_tools = []
        if self.tools:
            try:
                tool_list = self.tools.list_tools()
                available_tools = [t.name for t in tool_list]
            except Exception:
                pass
        
        # Extract step number from ID (e.g., "step_1" -> 1)
        step_order = step.id.split("_")[-1] if "_" in step.id else step.id
        
        # Determine if this is a file writing step
        is_write_step = any(kw in step.title.lower() for kw in ["save", "write", "create file", "output"])
        
        # Build the file writing instruction if applicable
        file_write_instruction = ""
        if is_write_step:
            file_write_instruction = """
IMPORTANT: This is a FILE WRITING step. You MUST:
1. Use the file_write tool with the FULL synthesized content
2. Include ALL the information from the completed steps above
3. Format the content as proper markdown
4. Use: TOOL: file_write ARGS: {"path": "filename.md", "content": "...full content here..."}
"""
        
        prompt = f"""You are executing step {step_order} of a multi-step plan.

GOAL: {plan.goal}

CURRENT STEP: {step.title}
STEP DESCRIPTION: {step.description or step.title}
{file_write_instruction}
{"COMPLETED STEPS WITH RESULTS:" + chr(10) + chr(10).join(completed_context) if completed_context else "This is the first step."}

IMPORTANT MCP NOTE:
- For any tool starting with "mcp_", only use arguments allowed by that tool's schema. Do not invent keys.
- Windows MCP quick reference:
  - Launch/switch/resize app: TOOL: mcp_windows__app ARGS: {{"mode": "launch|switch|resize", "name": "chrome"}}
  - Open URL / run command: TOOL: mcp_windows__shell ARGS: {{"command": "Start-Process chrome 'https://www.youtube.com'", "timeout": 10}}

{"AVAILABLE TOOLS: " + ", ".join(available_tools) if available_tools else "No tools available - provide analysis/content directly."}

Execute this step by:
1. If a tool is needed, specify which tool and arguments to use
2. If this is an analysis/synthesis step, provide the result directly
3. Be specific and actionable in your response
4. For file writing, use the file_write tool with the COMPLETE content

Respond with either:
- TOOL: <tool_name> ARGS: <json_args>
- RESULT: <direct result of the step>
"""

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Check if LLM wants to use a tool
            if content.strip().startswith("TOOL:"):
                # Parse tool call - handle multiline content
                try:
                    # Find the tool name first
                    content_stripped = content.strip()
                    tool_line_end = content_stripped.find("\n")
                    first_line = content_stripped[:tool_line_end] if tool_line_end > 0 else content_stripped
                    tool_part = first_line.split("TOOL:")[1].strip()
                    
                    if "ARGS:" in content_stripped:
                        # Extract tool name (before ARGS)
                        tool_name = tool_part.split("ARGS:")[0].strip() if "ARGS:" in tool_part else tool_part.strip()
                        
                        # Extract JSON args - may be multiline
                        args_start = content_stripped.find("ARGS:") + 5
                        args_str = content_stripped[args_start:].strip()
                        
                        # Try to find valid JSON - could be on same line or span multiple lines
                        try:
                            args = json.loads(args_str)
                        except json.JSONDecodeError:
                            # Try to extract JSON object/array from the string
                            import re
                            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])', args_str, re.DOTALL)
                            if json_match:
                                args = json.loads(json_match.group(1))
                            else:
                                args = {}
                    else:
                        tool_name = tool_part.strip()
                        args = {}
                    
                    # Execute the tool
                    logger.info(f"Executing tool {tool_name} with args: {str(args)[:200]}...")
                    if self.tools:
                        result = await self.tools.execute(tool_name, args)
                        return str(result) if result else f"Tool {tool_name} executed"
                    else:
                        return f"Tool {tool_name} requested but no tool registry available"
                        
                except Exception as e:
                    logger.warning(f"Failed to parse tool call: {e}")
                    # Fall back to using the response as-is
                    return content
            
            elif content.strip().startswith("RESULT:"):
                return content.split("RESULT:", 1)[1].strip()
            
            else:
                # Use the response as the result
                return content
                
        except Exception as e:
            logger.error(f"LLM execution failed for step {step.title}: {e}")
            raise
    
    async def _synthesize_content_for_file(self, plan: TaskPlan, step: TaskStep) -> Optional[str]:
        """
        Synthesize file content from completed step results using the LLM.
        
        Args:
            plan: The task plan with completed steps
            step: The file write step
            
        Returns:
            Synthesized content for the file
        """
        if not self.llm_provider or not self.llm_provider.llm:
            return None
        
        llm = self.llm_provider.llm
        
        # Gather all completed step results
        step_results = []
        for s in plan.completed_steps:
            if s.result:
                step_results.append(f"## {s.title}\n\n{s.result}")
        
        if not step_results:
            return None
        
        # Get file path for context
        file_path = step.tool_args.get("path", "output.md") if step.tool_args else "output.md"
        
        prompt = f"""Based on the research and analysis completed, create a well-formatted document for saving.

ORIGINAL GOAL: {plan.goal}

STEP BEING EXECUTED: {step.title}
TARGET FILE: {file_path}

RESEARCH RESULTS TO INCLUDE:
{chr(10).join(step_results)}

Create a comprehensive, well-structured markdown document that:
1. Has a clear title and introduction
2. Organizes the findings into logical sections
3. Includes all key insights from the research
4. Has a conclusion or summary

Return ONLY the document content (no code blocks, no explanations, just the markdown content):"""

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, 'content') else str(response)
            return content.strip()
        except Exception as e:
            logger.error(f"Failed to synthesize file content: {e}")
            # Fallback: just concatenate the results
            return f"# {plan.goal}\n\n" + "\n\n".join(step_results)
    
    async def replan(
        self,
        plan: TaskPlan,
        from_step: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> TaskPlan:
        """
        Create a new plan based on a failed or incomplete plan.
        
        Args:
            plan: The original plan
            from_step: Step ID to replan from
            reason: Reason for replanning
            
        Returns:
            New task plan
        """
        # Gather context from completed steps
        completed_context = []
        for step in plan.completed_steps:
            completed_context.append(f"✓ {step.title}: {step.result[:2000] if step.result else 'done'}")
        
        # Get failed step info
        failed_info = ""
        for step in plan.failed_steps:
            failed_info += f"✗ {step.title} failed: {step.error}\n"
        
        # Build new goal with context
        new_goal = f"""Continue this task: {plan.goal}

Already completed:
{chr(10).join(completed_context) if completed_context else 'Nothing yet'}

Failed steps:
{failed_info if failed_info else 'None'}

{f'Reason for replanning: {reason}' if reason else ''}

Please create a revised plan to complete the remaining work."""
        
        return await self.create_plan(new_goal, plan.context)
    
    def get_current_plan(self) -> Optional[TaskPlan]:
        """Get the currently active plan."""
        if self._current_plan and self._current_plan in self._plans:
            return self._plans[self._current_plan]
        return None
    
    def get_plan(self, plan_id: str) -> Optional[TaskPlan]:
        """Get a plan by ID."""
        return self._plans.get(plan_id)
    
    def get_all_plans(self) -> List[TaskPlan]:
        """Get all stored plans."""
        return list(self._plans.values())
    
    def clear_plans(self):
        """Clear all stored plans."""
        self._plans.clear()
        self._current_plan = None


# =============================================================================
# Task Queue for managing multiple concurrent tasks
# =============================================================================

@dataclass
class QueuedTask:
    """A task in the execution queue."""
    id: str
    prompt: str
    priority: TaskPriority
    plan: Optional[TaskPlan] = None
    status: TaskStatus = TaskStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt[:100],
            "priority": self.priority.value,
            "status": self.status.value,
            "plan": self.plan.to_dict() if self.plan else None,
            "created_at": self.created_at.isoformat(),
        }


class TaskQueue:
    """
    Queue for managing multiple tasks with priority.
    
    Supports:
    - Priority-based ordering
    - Concurrent execution limits
    - Task dependencies
    - Progress tracking
    """
    
    def __init__(self, max_concurrent: int = 1):
        self._queue: List[QueuedTask] = []
        self._active: Dict[str, QueuedTask] = {}
        self._completed: List[QueuedTask] = []
        self._max_concurrent = max_concurrent
        self._lock = asyncio.Lock()
        
        # Callbacks
        self._on_task_queued: Optional[Callable[[QueuedTask], None]] = None
        self._on_task_started: Optional[Callable[[QueuedTask], None]] = None
        self._on_task_completed: Optional[Callable[[QueuedTask], None]] = None
    
    async def add(
        self,
        prompt: str,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> QueuedTask:
        """Add a task to the queue."""
        async with self._lock:
            task = QueuedTask(
                id=str(uuid.uuid4())[:8],
                prompt=prompt,
                priority=priority,
            )
            
            # Insert based on priority
            insert_idx = len(self._queue)
            for idx, existing in enumerate(self._queue):
                if self._priority_value(priority) > self._priority_value(existing.priority):
                    insert_idx = idx
                    break
            
            self._queue.insert(insert_idx, task)
            
            if self._on_task_queued:
                self._on_task_queued(task)
            
            return task
    
    def _priority_value(self, priority: TaskPriority) -> int:
        """Convert priority to numeric value."""
        return {
            TaskPriority.LOW: 0,
            TaskPriority.NORMAL: 1,
            TaskPriority.HIGH: 2,
            TaskPriority.CRITICAL: 3,
        }.get(priority, 1)
    
    async def get_next(self) -> Optional[QueuedTask]:
        """Get the next task to execute."""
        async with self._lock:
            if len(self._active) >= self._max_concurrent:
                return None
            
            if not self._queue:
                return None
            
            task = self._queue.pop(0)
            task.status = TaskStatus.IN_PROGRESS
            self._active[task.id] = task
            
            if self._on_task_started:
                self._on_task_started(task)
            
            return task
    
    async def complete(self, task_id: str, success: bool = True) -> None:
        """Mark a task as completed."""
        async with self._lock:
            if task_id in self._active:
                task = self._active.pop(task_id)
                task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                self._completed.append(task)
                
                if self._on_task_completed:
                    self._on_task_completed(task)
    
    def get_queue_state(self) -> Dict[str, Any]:
        """Get current queue state for UI."""
        return {
            "queued": [t.to_dict() for t in self._queue],
            "active": [t.to_dict() for t in self._active.values()],
            "completed": [t.to_dict() for t in self._completed[-10:]],  # Last 10
            "counts": {
                "queued": len(self._queue),
                "active": len(self._active),
                "completed": len(self._completed),
            }
        }
    
    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0 and len(self._active) == 0
    
    @property
    def pending_count(self) -> int:
        """Get number of pending tasks."""
        return len(self._queue)
    
    @property
    def active_count(self) -> int:
        """Get number of active tasks."""
        return len(self._active)
