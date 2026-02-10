import asyncio
from pathlib import Path

import pytest

from intelclaw.mcp.connection import MCPServerSpec
from intelclaw.tools.mcp_tool import MCPRemoteTool


class _FakeRes:
    def __init__(self, *, is_error: bool = False, structured=None, content=None):
        self.isError = is_error
        self.structuredContent = structured
        self.content = content


class _FakeMCP:
    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    async def call(self, spec, tool_name, args=None):
        self.calls.append({"spec": spec, "tool_name": tool_name, "args": dict(args or {})})
        if not self._results:
            raise RuntimeError("no more results")
        return self._results.pop(0)


@pytest.mark.asyncio
async def test_mcp_payload_success_false_becomes_tool_failure(monkeypatch):
    fake = _FakeMCP([_FakeRes(structured={"success": False, "message": "boom"})])
    spec = MCPServerSpec(
        skill_id="whatsapp",
        server_id="whatsapp_mcp",
        transport="stdio",
        command="uv",
        args=["run", "main.py"],
        env={},
        cwd=Path("."),
        tool_namespace="whatsapp",
        tool_allowlist=[],
        tool_denylist=[],
    )
    tool = MCPRemoteTool(mcp_manager=fake, spec=spec, mcp_tool_name="send_message", description="", input_schema={})
    res = await tool.execute(recipient="+92 317-115-6353", message="hi")
    assert res.success is False
    assert "boom" in (res.error or "")
    # Recipient normalized (digits-only phone or a resolved JID).
    sent_recipient = fake.calls[0]["args"]["recipient"]
    if "@" in sent_recipient:
        assert sent_recipient.split("@", 1)[0] == "923171156353"
    else:
        assert sent_recipient == "923171156353"


@pytest.mark.asyncio
async def test_mcp_whatsapp_iq_timeout_retries(monkeypatch):
    async def _fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    fake = _FakeMCP(
        [
            _FakeRes(structured={"success": False, "message": "Error sending message: info query timed out"}),
            _FakeRes(structured={"success": True, "message": "ok"}),
        ]
    )
    spec = MCPServerSpec(
        skill_id="whatsapp",
        server_id="whatsapp_mcp",
        transport="stdio",
        command="uv",
        args=["run", "main.py"],
        env={},
        cwd=Path("."),
        tool_namespace="whatsapp",
        tool_allowlist=[],
        tool_denylist=[],
    )
    tool = MCPRemoteTool(mcp_manager=fake, spec=spec, mcp_tool_name="send_message", description="", input_schema={})
    res = await tool.execute(recipient="923171156353", message="hi")
    assert res.success is True
    assert fake.calls and len(fake.calls) == 2
