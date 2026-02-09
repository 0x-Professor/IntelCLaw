from intelclaw.agent.task_planner import TaskPlanner


class _DummyTools:
    def __init__(self, available: set[str]) -> None:
        self._available = set(available)

    def get_tool(self, name: str):
        return object() if name in self._available else None


def test_rewrite_windows_app_url_to_shell():
    planner = TaskPlanner(tools=_DummyTools({"mcp_windows__shell"}))
    tool, args = planner._rewrite_common_tool_invocations(
        "mcp_windows__app",
        {"app": "chrome", "url": "https://www.youtube.com"},
    )
    assert tool == "mcp_windows__shell"
    assert args["command"] == "Start-Process chrome 'https://www.youtube.com'"


def test_rewrite_legacy_app_tool_url_to_shell():
    planner = TaskPlanner(tools=_DummyTools({"mcp_windows__shell"}))
    tool, args = planner._rewrite_common_tool_invocations(
        "app_tool",
        {"app": "chrome", "url": "https://www.youtube.com"},
    )
    assert tool == "mcp_windows__shell"
    assert args["command"] == "Start-Process chrome 'https://www.youtube.com'"


def test_rewrite_windows_shell_script_to_command():
    planner = TaskPlanner(tools=_DummyTools(set()))
    tool, args = planner._rewrite_common_tool_invocations(
        "mcp_windows__shell",
        {"script": "Write-Host hi", "timeout_seconds": 5},
    )
    assert tool == "mcp_windows__shell"
    assert args["command"] == "Write-Host hi"
    assert args["timeout"] == 5


def test_rewrite_windows_app_strips_url_when_no_fallback_available():
    planner = TaskPlanner(tools=_DummyTools(set()))
    tool, args = planner._rewrite_common_tool_invocations(
        "mcp_windows__app",
        {"name": "chrome", "url": "https://www.youtube.com"},
    )
    assert tool == "mcp_windows__app"
    assert "url" not in args
