import shutil

import pytest

from intelclaw.mcp.connection import MCPServerSpec
from intelclaw.mcp.manager import MCPManager, MCPTimeouts


@pytest.mark.asyncio
async def test_windows_mcp_lists_tools_smoke():
    if not shutil.which("windows-mcp"):
        pytest.skip("windows-mcp not installed / not on PATH")

    mgr = MCPManager(timeouts=MCPTimeouts(start_seconds=15, list_tools_seconds=15, call_tool_seconds=60))
    spec = MCPServerSpec(
        skill_id="windows",
        server_id="windows_mcp",
        transport="stdio",
        command="windows-mcp",
        args=["--transport", "stdio"],
        env={"ANONYMIZED_TELEMETRY": "false"},
        cwd=None,
        tool_namespace="windows",
        tool_allowlist=[],
        tool_denylist=[],
    )

    try:
        tools = await mgr.get_tools(spec, refresh=True)
        assert len(list(tools or [])) > 0
    finally:
        await mgr.shutdown_unused(set())
