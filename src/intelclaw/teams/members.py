"""
Team members - specialized agents with tool allowlists and skill instructions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from loguru import logger

from intelclaw.agent.base import AgentContext, AgentResponse
from intelclaw.agent.task_planner import TaskPlanner, TaskPlan, TaskStep

if TYPE_CHECKING:
    from intelclaw.integrations.llm_provider import LLMProvider
    from intelclaw.tools.registry import ToolRegistry
    from intelclaw.memory.manager import MemoryManager


class FilteredToolRegistry:
    """A thin wrapper that restricts list_tools/execute to an allowlist."""

    def __init__(self, base: "ToolRegistry", allowlist: Set[str]) -> None:
        self._base = base
        self._allow = set(allowlist)

    def list_tools(self, category: Any = None) -> List[Any]:
        try:
            return self._base.list_tools(category, allowlist=self._allow)
        except TypeError:
            # Backwards compatibility if allowlist arg isn't present.
            return [t for t in self._base.list_tools(category) if t.name in self._allow]

    async def execute(self, name: str, params: Dict[str, Any], check_permissions: bool = True) -> Any:
        tool_name = str(name or "").strip()
        if tool_name not in self._allow:
            raise RuntimeError(f"Tool not allowed for this agent: {tool_name}")
        return await self._base.execute(tool_name, params, check_permissions=check_permissions)


@dataclass(frozen=True)
class TeamMember:
    member_id: str
    display_name: str
    agent_instructions: str

    llm_provider: "LLMProvider"
    tools: "ToolRegistry"
    memory: Optional["MemoryManager"] = None

    async def process(self, context: AgentContext, *, allow_tools: Set[str]) -> AgentResponse:
        start = time.time()
        goal = str(context.user_message or "")

        filtered_tools = FilteredToolRegistry(self.tools, allow_tools)

        planner = TaskPlanner(
            llm_provider=self.llm_provider,
            tools=filtered_tools,  # type: ignore[arg-type]
            memory=self.memory,
            max_steps=8,
            enable_web_search=True,
            enable_rag=True,
        )

        # Provide member instructions to planning context (best-effort).
        plan = await planner.create_plan(
            goal=goal,
            context={
                "agent_instructions": self.agent_instructions,
                "member_id": self.member_id,
                "session_id": getattr(context, "session_id", None),
            },
        )

        executed = await planner.execute_plan(plan)

        tools_used = [s.tool for s in executed.steps if s.tool] if executed and executed.steps else []
        answer = await _synthesize_plan_response(self.llm_provider, executed, goal)

        return AgentResponse(
            answer=answer,
            thoughts=[],
            tools_used=[t for t in tools_used if t],
            latency_ms=(time.time() - start) * 1000,
            success=bool(executed and executed.is_complete),
            error=None if executed and executed.is_complete else "Task did not fully complete",
        )


async def _synthesize_plan_response(llm_provider: "LLMProvider", plan: TaskPlan, goal: str) -> str:
    llm = getattr(llm_provider, "llm", None)
    if llm is None:
        lines = []
        for step in plan.steps:
            if step.status.value == "completed":
                lines.append(f"- {step.title}: {step.result or ''}".strip())
        return "Completed:\n" + "\n".join(lines) if lines else "Completed."

    summaries = []
    for step in plan.steps:
        status = "✓" if step.status.value == "completed" else "✗"
        preview = (step.result or step.error or "").strip()[:1200]
        summaries.append(f"{status} {step.title}: {preview}")

    prompt = (
        "You are summarizing the results of a multi-step execution by a specialist agent.\n\n"
        f"Goal: {goal}\n\n"
        "Results:\n" + "\n".join(summaries) + "\n\n"
        "Write a concise, helpful summary focusing on outcomes and next steps."
    )

    try:
        from langchain_core.messages import HumanMessage

        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        logger.debug(f"Member synthesis failed: {e}")
        return "Completed."

