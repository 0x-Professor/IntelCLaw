import asyncio
import sys
from pathlib import Path

import pytest

from intelclaw.skills.manager import SkillManager
from intelclaw.tools.registry import ToolRegistry


class DummyConfig:
    def get(self, key, default=None):
        if key == "mcp":
            return {"enabled": True}
        if key == "mcp.timeouts.start_seconds":
            return 0.2
        if key == "mcp.timeouts.list_tools_seconds":
            return 0.2
        if key == "mcp.timeouts.call_tool_seconds":
            return 0.2
        return default


class DummySecurity:
    async def has_permission(self, _perm: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_mcp_hanging_server_times_out_and_does_not_block(tmp_path):
    builtin = tmp_path / "skills"
    user = tmp_path / "data" / "skills"
    state = tmp_path / "data" / "skills_state.json"

    (builtin / "hang").mkdir(parents=True, exist_ok=True)
    server_script = Path(__file__).resolve().parent / "fixtures" / "hanging_mcp_server.py"

    (builtin / "hang" / "skill.yaml").write_text(
        "\n".join(
            [
                "id: hang",
                "name: Hang",
                "enabled_by_default: true",
                "mcp_servers:",
                "  - id: hang_mcp",
                "    transport: stdio",
                f"    command: '{sys.executable}'",
                f"    args: ['{str(server_script)}']",
                "    env: {}",
                "    cwd: null",
                "    tool_namespace: hang",
                "",
            ]
        ),
        encoding="utf-8",
    )

    skills = SkillManager(None, None, builtin_dir=builtin, user_dir=user, state_path=state)
    await skills.initialize()

    tools = ToolRegistry(DummyConfig(), DummySecurity(), skills=skills, event_bus=None)
    try:
        # Ensure the call returns quickly even if the server never responds.
        await asyncio.wait_for(tools._load_mcp_tools(), timeout=15.0)

        health = tools.get_mcp_server_health("hang", "hang_mcp")
        assert health["healthy"] is False
        assert "timeout" in str(health.get("last_error", "")).lower()
    finally:
        await tools.shutdown()
