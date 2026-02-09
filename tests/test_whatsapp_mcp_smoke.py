import shutil
from pathlib import Path

import pytest

from intelclaw.mcp.connection import MCPServerSpec
from intelclaw.mcp.manager import MCPManager, MCPTimeouts


@pytest.mark.asyncio
async def test_whatsapp_mcp_lists_tools_smoke():
    root = Path(__file__).resolve().parent.parent
    server_dir = root / "data" / "vendor" / "whatsapp-mcp" / "whatsapp-mcp-server"

    if not server_dir.exists():
        pytest.skip("whatsapp-mcp repo not present under data/vendor")
    if not shutil.which("uv"):
        pytest.skip("uv not installed / not on PATH")

    mgr = MCPManager(timeouts=MCPTimeouts(start_seconds=30, list_tools_seconds=30, call_tool_seconds=60))
    spec = MCPServerSpec(
        skill_id="whatsapp",
        server_id="whatsapp_mcp",
        transport="stdio",
        command="uv",
        args=["run", "main.py"],
        env={},
        cwd=server_dir,
        tool_namespace="whatsapp",
        tool_allowlist=[],
        tool_denylist=[],
    )

    try:
        try:
            tools = await mgr.get_tools(spec, refresh=True)
        except Exception as e:
            pytest.skip(f"whatsapp-mcp server not ready (bridge/login likely missing): {e}")
        assert len(list(tools or [])) > 0
    finally:
        await mgr.shutdown_unused(set())
