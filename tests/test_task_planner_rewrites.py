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
        title="Resolve contact information",
        tool="contacts_lookup",
        tool_args={"query": "Alex Example"},
        status=TaskStatus.COMPLETED,
    )
    step_1.result = json.dumps(
        [
            {
                "name": "Alex Example",
                "phone": "15551234567",
                "whatsapp_jid": "15551234567@s.whatsapp.net",
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
    step_2.result = "Hello Alex,\n\nThis is a test.\n"

    step_3 = TaskStep(
        id="step_3",
        title="Send WhatsApp message to Alex Example",
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
    assert tool_args["recipient"] == "15551234567@s.whatsapp.net"


@pytest.mark.asyncio
async def test_backfill_whatsapp_message_from_goal_when_missing():
    import json

    spy = _SpyTools()
    planner = TaskPlanner(tools=spy)

    step_1 = TaskStep(
        id="step_1",
        title="Resolve contact information",
        tool="contacts_lookup",
        tool_args={"query": "Alex Example"},
        status=TaskStatus.COMPLETED,
    )
    step_1.result = json.dumps(
        [
            {
                "name": "Alex Example",
                "phone": "15551234567",
                "whatsapp_jid": "15551234567@s.whatsapp.net",
            }
        ]
    )

    step_2 = TaskStep(
        id="step_2",
        title="Send WhatsApp message to Alex Example",
        tool="mcp_whatsapp__send_message",
        tool_args={},  # LLM omitted required `message`
        dependencies=["step_1"],
        status=TaskStatus.PENDING,
    )

    plan = TaskPlan(
        id="plan_1",
        goal="Please send a WhatsApp message to Alex and ask him when he will return from the job?",
        steps=[step_1, step_2],
        context={},
        metadata={},
    )

    await planner._execute_step(step_2, plan, executor=None)

    assert spy.calls, "Expected the send_message tool to be executed"
    tool_name, tool_args = spy.calls[0]
    assert tool_name == "mcp_whatsapp__send_message"
    assert tool_args["recipient"] == "15551234567@s.whatsapp.net"
    msg = str(tool_args.get("message") or "")
    assert msg, "Expected message to be backfilled"
    assert "when will you return" in msg.lower()


@pytest.mark.asyncio
async def test_backfill_contacts_upsert_phone_from_contacts_lookup_result():
    import json

    spy = _SpyTools()
    planner = TaskPlanner(tools=spy)

    step_1 = TaskStep(
        id="step_1",
        title="Resolve contact information",
        tool="contacts_lookup",
        tool_args={"query": "Alex Example"},
        status=TaskStatus.COMPLETED,
    )
    step_1.result = json.dumps(
        [
            {
                "name": "Alex Example",
                "phone": "15551234567",
                "whatsapp_jid": "15551234567@s.whatsapp.net",
            }
        ]
    )

    step_2 = TaskStep(
        id="step_2",
        title="(Optional) Save resolved contact locally",
        tool="contacts_upsert",
        tool_args={"name": "Alex Example"},  # LLM omitted required `phone`
        dependencies=["step_1"],
        status=TaskStatus.PENDING,
    )

    plan = TaskPlan(
        id="plan_1",
        goal="Save Alex Example to contacts",
        steps=[step_1, step_2],
        context={},
        metadata={},
    )

    await planner._execute_step(step_2, plan, executor=None)

    assert spy.calls, "Expected contacts_upsert to be executed"
    tool_name, tool_args = spy.calls[0]
    assert tool_name == "contacts_upsert"
    assert tool_args["name"] == "Alex Example"
    assert tool_args["phone"] == "15551234567"


@pytest.mark.asyncio
async def test_fix_invalid_whatsapp_recipient_by_inferencing_from_plan(tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)

    spy = _SpyTools()
    planner = TaskPlanner(tools=spy)

    step_1 = TaskStep(
        id="step_1",
        title="Resolve contact information",
        tool="contacts_lookup",
        tool_args={"query": "Alex Example"},
        status=TaskStatus.COMPLETED,
    )
    step_1.result = json.dumps(
        [
            {
                "name": "Alex Example",
                "phone": "15551234567",
                "whatsapp_jid": "15551234567@s.whatsapp.net",
            }
        ]
    )

    step_2 = TaskStep(
        id="step_2",
        title="Send WhatsApp message",
        tool="mcp_whatsapp__send_message",
        tool_args={"recipient": "Alex on WhatsApp", "message": "hi"},
        dependencies=["step_1"],
        status=TaskStatus.PENDING,
    )

    plan = TaskPlan(
        id="plan_1",
        goal="Send WhatsApp message hi to Alex",
        steps=[step_1, step_2],
        context={},
        metadata={},
    )

    await planner._execute_step(step_2, plan, executor=None)

    assert spy.calls, "Expected the send_message tool to be executed"
    tool_name, tool_args = spy.calls[0]
    assert tool_name == "mcp_whatsapp__send_message"
    assert tool_args["recipient"] == "15551234567@s.whatsapp.net"
