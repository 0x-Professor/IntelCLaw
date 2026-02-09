import sys
from pathlib import Path

import pytest

from intelclaw.skills.manager import SkillManager
from intelclaw.tools.registry import ToolRegistry


class DummyConfig:
    def get(self, key, default=None):
        if key == "mcp":
            return {"enabled": True}
        return default


class DummySecurity:
    async def has_permission(self, _perm: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_tool_registry_loads_and_executes_mcp_tools(tmp_path):
    builtin = tmp_path / "skills"
    user = tmp_path / "data" / "skills"
    state = tmp_path / "data" / "skills_state.json"

    (builtin / "fake").mkdir(parents=True, exist_ok=True)
    server_script = Path(__file__).resolve().parent / "fixtures" / "fake_mcp_server.py"
    (builtin / "fake" / "skill.yaml").write_text(
        "\n".join(
            [
                "id: fake",
                "name: Fake",
                "enabled_by_default: true",
                "mcp_servers:",
                "  - id: fake_mcp",
                "    transport: stdio",
                f"    command: '{sys.executable}'",
                f"    args: ['{str(server_script)}']",
                "    env: {}",
                "    cwd: null",
                "    tool_namespace: fake",
                "",
            ]
        ),
        encoding="utf-8",
    )

    skills = SkillManager(None, None, builtin_dir=builtin, user_dir=user, state_path=state)
    await skills.initialize()

    tools = ToolRegistry(DummyConfig(), DummySecurity(), skills=skills, event_bus=None)
    # For test speed/stability, load only MCP tools (skip registering all built-ins).
    await tools._load_mcp_tools()

    try:
        names = {d.name for d in tools.list_tools()}
        assert "mcp_fake__echo_tool" in names  # normalized from "Echo-Tool!"
        assert "mcp_fake__add" in names

        assert tools.get_skill_id_for_tool("mcp_fake__add") == "fake"
        assert set(tools.get_mcp_tool_names_for_skill("fake")) >= {"mcp_fake__echo_tool", "mcp_fake__add"}

        res = await tools.execute("mcp_fake__add", {"a": 2, "b": 3}, check_permissions=False)
        assert isinstance(res, dict)
        assert res["sum"] == 5

        res2 = await tools.execute("mcp_fake__echo_tool", {"text": "hi"}, check_permissions=False)
        assert isinstance(res2, dict)
        assert res2["echo"] == "hi"
    finally:
        await tools.shutdown()
