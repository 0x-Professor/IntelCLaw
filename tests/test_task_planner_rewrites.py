import pytest

from intelclaw.agent.task_planner import TaskPlanner
from intelclaw.agent.task_planner import TaskPlan, TaskStatus, TaskStep


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


class _SpyTools:
    def __init__(self) -> None:
        self.calls = []

    def get_tool(self, _name: str):
        return object()

    async def execute(self, name: str, args: dict):
        self.calls.append((name, dict(args)))
        return {"success": True, "data": {"ok": True}}


@pytest.mark.asyncio
async def test_backfill_whatsapp_recipient_from_contacts_lookup_result():
    import json

    spy = _SpyTools()
    planner = TaskPlanner(tools=spy)

    step_1 = TaskStep(
        id="step_1",
        title="Resolve Talha contact information",
        tool="contacts_lookup",
        tool_args={"query": "Talha Bilal AU CYS"},
        status=TaskStatus.COMPLETED,
    )
    step_1.result = json.dumps(
        [
            {
                "name": "Talha Bilal AU CYS",
                "phone": "923171156353",
                "whatsapp_jid": "923171156353@s.whatsapp.net",
            }
        ]
    )

    step_2 = TaskStep(
        id="step_2",
        title="Draft introduction message as IntelCLaw",
        tool=None,
        tool_args=None,
        status=TaskStatus.COMPLETED,
    )
    step_2.result = "Hello Talha Bilal,\n\nThis is a test.\n"

    step_3 = TaskStep(
        id="step_3",
        title="Send WhatsApp message to Talha Bilal AU CYS",
        tool="mcp_whatsapp__send_message",
        tool_args={"message": "hi"},
        dependencies=["step_1", "step_2"],
        status=TaskStatus.PENDING,
    )

    plan = TaskPlan(
        id="plan_1",
        goal="Send a message",
        steps=[step_1, step_2, step_3],
        context={},
        metadata={},
    )

    await planner._execute_step(step_3, plan, executor=None)

    assert spy.calls, "Expected the send_message tool to be executed"
    tool_name, tool_args = spy.calls[0]
    assert tool_name == "mcp_whatsapp__send_message"
    assert tool_args["recipient"] == "923171156353@s.whatsapp.net"
