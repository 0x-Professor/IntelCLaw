"""
MCP runtime - manage MCP server processes and tool calls.

This is the client-side integration used by ToolRegistry to expose MCP tools
as IntelCLaw tools.
"""

from __future__ import annotations

from intelclaw.mcp.connection import MCPServerSpec
from intelclaw.mcp.manager import MCPManager

__all__ = ["MCPManager", "MCPServerSpec"]

