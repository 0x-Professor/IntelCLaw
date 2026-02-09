"""
TeamMailbox - shared inbox for agent-to-agent notes and user notifications.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional
from uuid import uuid4

from loguru import logger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MailboxMessage:
    id: str = field(default_factory=lambda: str(uuid4()))
    ts: str = field(default_factory=_utc_now_iso)
    from_agent: str = "team_lead"
    to: str = "user"
    kind: str = "info"  # info|note|progress|error|completed
    title: str = ""
    body: str = ""
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "from_agent": self.from_agent,
            "to": self.to,
            "kind": self.kind,
            "title": self.title,
            "body": self.body,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "meta": dict(self.meta or {}),
        }


class TeamMailbox:
    def __init__(self, event_bus: Any, *, max_per_session: int = 500) -> None:
        self._event_bus = event_bus
        self._max = int(max_per_session)
        self._lock = asyncio.Lock()
        self._by_session: Dict[str, Deque[MailboxMessage]] = defaultdict(
            lambda: deque(maxlen=self._max)
        )

    async def post(self, msg: MailboxMessage) -> MailboxMessage:
        session_id = str(msg.session_id or "default").strip() or "default"
        async with self._lock:
            self._by_session[session_id].append(msg)

        if self._event_bus:
            try:
                await self._event_bus.emit(
                    "mailbox.message",
                    {"message": msg.to_dict()},
                    source="mailbox",
                )
            except Exception as e:
                logger.debug(f"Mailbox emit failed: {e}")

        return msg

    async def list_messages(self, session_id: str, *, limit: int = 200) -> List[Dict[str, Any]]:
        sid = str(session_id or "default").strip() or "default"
        n = max(1, min(int(limit or 200), self._max))
        async with self._lock:
            msgs = list(self._by_session.get(sid, deque()))
        return [m.to_dict() for m in msgs[-n:]]

    async def clear(self, session_id: str) -> None:
        sid = str(session_id or "default").strip() or "default"
        async with self._lock:
            self._by_session.pop(sid, None)

