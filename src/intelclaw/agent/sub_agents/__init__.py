"""Sub-agents for specialized tasks."""

from intelclaw.agent.sub_agents.research_agent import ResearchAgent
from intelclaw.agent.sub_agents.coding_agent import CodingAgent
from intelclaw.agent.sub_agents.task_agent import TaskAgent
from intelclaw.agent.sub_agents.system_agent import SystemAgent
from intelclaw.agent.sub_agents.autonomous_agent import AutonomousAgent

__all__ = ["ResearchAgent", "CodingAgent", "TaskAgent", "SystemAgent", "AutonomousAgent"]
