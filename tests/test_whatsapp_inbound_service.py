import sqlite3
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from intelclaw.integrations.whatsapp.inbound import WhatsAppInboundMessage, WhatsAppInboundService


class _SpyTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, name: str, args: dict):
        self.calls.append((name, dict(args)))
        return {"success": True}


class _DummyLLM:
    def __init__(self, reply_text: str) -> None:
        self._reply_text = reply_text

    async def ainvoke(self, _messages):
        return AIMessage(content=self._reply_text)


class _DummyProvider:
    def __init__(self, reply_text: str) -> None:
        self.llm = _DummyLLM(reply_text)


def _create_messages_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE messages (
                id TEXT,
                chat_jid TEXT,
                sender TEXT,
                content TEXT,
                timestamp TEXT,
                is_from_me BOOLEAN
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_message(
    db_path: Path,
    *,
    chat_jid: str,
    sender: str,
    content: str,
    timestamp: str,
    is_from_me: int,
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO messages (id, chat_jid, sender, content, timestamp, is_from_me) VALUES (?,?,?,?,?,?)",
            ("id1", chat_jid, sender, content, timestamp, int(is_from_me)),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_inbound_autoreply_sends_message_when_contact_missing(tmp_path: Path):
    db_path = tmp_path / "messages.db"
    _create_messages_db(db_path)

    chat_jid = "15551234567@s.whatsapp.net"
    _insert_message(
        db_path,
        chat_jid=chat_jid,
        sender="15551234567",
        content="Hello there",
        timestamp="2026-02-10 10:00:00+00:00",
        is_from_me=0,
    )

    tools = _SpyTools()
    svc = WhatsAppInboundService(
        config={"whatsapp.inbound.enabled": True},
        skills=None,
        tools=tools,
        llm_provider=_DummyProvider("Hi! Doing well - what's up?"),
        mailbox=None,
        event_bus=None,
    )

    msg = WhatsAppInboundMessage(
        rowid=1,
        msg_id="id1",
        chat_jid=chat_jid,
        sender="15551234567",
        content="Hello there",
        timestamp="2026-02-10 10:00:00+00:00",
    )

    await svc._handle_inbound_message(msg, db_path, context_n=10, contact=None)

    assert tools.calls, "Expected WhatsApp send_message tool to be executed"
    tool_name, tool_args = tools.calls[0]
    assert tool_name == "mcp_whatsapp__send_message"
    assert tool_args["recipient"] == chat_jid
    assert tool_args["message"].strip()
