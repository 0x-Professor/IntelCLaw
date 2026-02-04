"""Integrations module - External service integrations."""

from intelclaw.integrations.copilot import CopilotIntegration, ModelManager
from intelclaw.integrations.llm_provider import LLMProvider, CopilotLLM, GitHubAuth

__all__ = ["CopilotIntegration", "ModelManager", "LLMProvider", "CopilotLLM", "GitHubAuth"]
