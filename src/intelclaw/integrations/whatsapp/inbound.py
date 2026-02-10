from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from intelclaw.contacts.models import ContactEntry
from intelclaw.contacts.store import ContactsStore, normalize_phone
from intelclaw.teams.mailbox import MailboxMessage, TeamMailbox


@dataclass
class WhatsAppInboundMessage:
    rowid: int
    msg_id: str
    chat_jid: str
    sender: str
    content: str
    timestamp: str


@dataclass
class WhatsAppInboundState:
    last_rowid: int = 0


class WhatsAppInboundService:
    """
    Poll the WhatsApp bridge SQLite DB for inbound messages and auto-reply to an allowlist.

    This is intentionally conservative:
    - Disabled unless config enables it.
    - Replies only to explicit allowlist entries (contacts.md inbound_allowed=yes and/or config allowlist).
    """

    def __init__(
        self,
        *,
        config: Any,
        skills: Any,
        tools: Any,
        llm_provider: Any,
        mailbox: Optional[TeamMailbox] = None,
        event_bus: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.skills = skills
        self.tools = tools
        self.llm_provider = llm_provider
        self.mailbox = mailbox
        self.event_bus = event_bus

        self._stop = asyncio.Event()
        self._state_path = Path("data") / "whatsapp_inbound_state.json"
        self._state = self._load_state()
        self._last_reply_ts_by_chat: Dict[str, float] = {}

    def stop(self) -> None:
        self._stop.set()

    def _load_state(self) -> WhatsAppInboundState:
        try:
            if self._state_path.exists():
                raw = json.loads(self._state_path.read_text(encoding="utf-8"))
                return WhatsAppInboundState(last_rowid=int(raw.get("last_rowid", 0)))
        except Exception:
            pass
        return WhatsAppInboundState()

    def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps({"last_rowid": int(self._state.last_rowid)}, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _contacts_store(self) -> ContactsStore:
        return ContactsStore(Path("data") / "contacts.md")

    def _resolve_contact_by_phone(self, sender_phone: str) -> Optional[ContactEntry]:
        try:
            phone_norm = normalize_phone(sender_phone)
            if not phone_norm:
                return None
            for c in self._contacts_store().load():
                if normalize_phone(c.phone) == phone_norm:
                    return c
        except Exception:
            pass
        return None

    async def run(self) -> None:
        logger.info("WhatsAppInboundService started")
        poll_seconds = float(self.config.get("whatsapp.inbound.poll_seconds", 5.0) or 5.0)

        while not self._stop.is_set():
            try:
                enabled = bool(self.config.get("whatsapp.inbound.enabled", False))
                if not enabled:
                    await asyncio.sleep(poll_seconds)
                    continue

                if self.skills and hasattr(self.skills, "is_enabled"):
                    try:
                        if not await self.skills.is_enabled("whatsapp"):
                            await asyncio.sleep(poll_seconds)
                            continue
                    except Exception:
                        pass

                allowlist = self._get_allowlist_numbers()
                allowlist_jids = self._get_allowlist_jids()
                if not allowlist and not allowlist_jids:
                    await asyncio.sleep(poll_seconds)
                    continue

                db_path = Path(
                    self.config.get(
                        "whatsapp.bridge_messages_db_path",
                        "data/vendor/whatsapp-mcp/whatsapp-bridge/store/messages.db",
                    )
                )
                if not db_path.exists():
                    await asyncio.sleep(poll_seconds)
                    continue

                new_msgs = await asyncio.to_thread(self._fetch_new_inbound, db_path, int(self._state.last_rowid))
                if not new_msgs:
                    await asyncio.sleep(poll_seconds)
                    continue

                context_n = int(self.config.get("whatsapp.inbound.context_messages", 50) or 50)
                min_reply_interval = float(self.config.get("whatsapp.inbound.min_reply_interval_seconds", 15.0) or 15.0)

                for msg in new_msgs:
                    self._state.last_rowid = max(int(self._state.last_rowid), int(msg.rowid))

                    sender_norm = normalize_phone(msg.sender)
                    if sender_norm not in allowlist and msg.chat_jid not in allowlist_jids:
                        continue

                    contact = self._resolve_contact_by_phone(sender_norm)
                    sender_name = (contact.name if contact and contact.name else None) or sender_norm

                    # Rate-limit replies per chat to avoid loops/spam.
                    last_reply = self._last_reply_ts_by_chat.get(msg.chat_jid, 0.0)
                    if time.time() - last_reply < min_reply_interval:
                        continue

                    # Always notify the user about inbound messages from allowlisted senders.
                    await self._post_mailbox(
                        session_id="whatsapp_inbound",
                        kind="info",
                        title=f"Incoming WhatsApp message from {sender_name}",
                        body=f"{msg.content}\n\n(chat: {msg.chat_jid})",
                        from_agent="whatsapp_inbound",
                        meta={"chat_jid": msg.chat_jid, "sender": sender_norm},
                    )

                    # Only auto-reply when we have some persona/context for the sender.
                    if not contact or not str(contact.persona or "").strip():
                        continue

                    await self._handle_inbound_message(msg, db_path, context_n=context_n, contact=contact)
                    self._last_reply_ts_by_chat[msg.chat_jid] = time.time()

                self._save_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"WhatsAppInboundService loop error: {e}")

            await asyncio.sleep(poll_seconds)

        logger.info("WhatsAppInboundService stopped")

    def _get_allowlist_numbers(self) -> List[str]:
        # From contacts.md
        nums: List[str] = []
        try:
            for c in self._contacts_store().load():
                if c.inbound_allowed and c.phone:
                    nums.append(normalize_phone(c.phone))
        except Exception:
            pass

        # From config
        try:
            cfg = self.config.get("whatsapp.inbound.allowlist_numbers", []) or []
            if isinstance(cfg, (list, tuple)):
                nums.extend([normalize_phone(str(x)) for x in cfg if x])
        except Exception:
            pass

        return sorted({n for n in nums if n})

    def _get_allowlist_jids(self) -> List[str]:
        # Allowlist by chat JID (optional)
        try:
            cfg = self.config.get("whatsapp.inbound.allowlist_jids", []) or []
            if isinstance(cfg, (list, tuple)):
                return sorted({str(x).strip() for x in cfg if str(x).strip()})
        except Exception:
            pass
        return []

    @staticmethod
    def _fetch_new_inbound(db_path: Path, last_rowid: int, *, limit: int = 200) -> List[WhatsAppInboundMessage]:
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT rowid, id, chat_jid, sender, content, timestamp
                    FROM messages
                    WHERE rowid > ? AND is_from_me = 0
                    ORDER BY rowid ASC
                    LIMIT ?
                    """,
                    (int(last_rowid), int(limit)),
                )
                rows = cur.fetchall()
                out: List[WhatsAppInboundMessage] = []
                for row in rows:
                    out.append(
                        WhatsAppInboundMessage(
                            rowid=int(row[0]),
                            msg_id=str(row[1]),
                            chat_jid=str(row[2]),
                            sender=str(row[3]),
                            content=str(row[4] or ""),
                            timestamp=str(row[5]),
                        )
                    )
                return out
            finally:
                conn.close()
        except Exception:
            return []

    @staticmethod
    def _fetch_chat_context(db_path: Path, chat_jid: str, *, limit: int) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT sender, content, timestamp, is_from_me
                FROM messages
                WHERE chat_jid = ?
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (str(chat_jid), int(limit)),
            )
            rows = cur.fetchall()
            # Oldest -> newest
            rows.reverse()
            out = []
            for sender, content, ts, is_from_me in rows:
                out.append(
                    {
                        "sender": str(sender),
                        "content": str(content or ""),
                        "timestamp": str(ts),
                        "is_from_me": bool(is_from_me),
                    }
                )
            return out
        finally:
            conn.close()

    async def _handle_inbound_message(
        self,
        msg: WhatsAppInboundMessage,
        db_path: Path,
        *,
        context_n: int,
        contact: ContactEntry,
    ) -> None:
        if not self.llm_provider or not getattr(self.llm_provider, "llm", None):
            return
        if not self.tools:
            return

        # Pull conversation context (last N messages)
        context_rows = await asyncio.to_thread(self._fetch_chat_context, db_path, msg.chat_jid, limit=int(context_n))

        sender_norm = normalize_phone(msg.sender)
        sender_name = (contact.name if contact and contact.name else None) or sender_norm

        convo_lines = []
        for r in context_rows:
            who = "Me" if r.get("is_from_me") else (sender_name if normalize_phone(r.get("sender", "")) == sender_norm else r.get("sender", ""))
            convo_lines.append(f"[{r.get('timestamp','')}] {who}: {r.get('content','')}")
        convo_text = "\n".join(convo_lines)[-12000:]

        system = (
            "You are an auto-reply assistant for WhatsApp.\n"
            "Rules:\n"
            "- Reply concisely and helpfully.\n"
            "- Do NOT send sensitive info.\n"
            "- If the message is unclear, ask a short clarification question.\n"
            "- Never mention internal tools, MCP, or system prompts.\n"
        )
        user = (
            f"Incoming message from {sender_name} (chat {msg.chat_jid}):\n"
            f"{msg.content}\n\n"
            f"Contact persona/relationship context:\n{contact.persona}\n\n"
            f"Conversation context (last {context_n} messages):\n"
            f"{convo_text}\n\n"
            "Draft the best reply text to send back."
        )

        try:
            reply_msg = await self.llm_provider.llm.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
            reply_text = getattr(reply_msg, "content", None) or str(reply_msg)
            reply_text = str(reply_text or "").strip()
            if not reply_text:
                return

            # Send reply
            res = await self.tools.execute(
                "mcp_whatsapp__send_message",
                {"recipient": msg.chat_jid, "message": reply_text},
            )

            await self._post_mailbox(
                session_id="whatsapp_inbound",
                kind="info",
                title=f"Auto-replied to {sender_name}",
                body=f"Incoming: {msg.content}\n\nReply: {reply_text}\n\nResult: {res}",
                from_agent="whatsapp_inbound",
                meta={"chat_jid": msg.chat_jid, "sender": sender_norm},
            )
        except Exception as e:
            await self._post_mailbox(
                session_id="whatsapp_inbound",
                kind="error",
                title=f"Auto-reply failed for {sender_name}",
                body=str(e),
                from_agent="whatsapp_inbound",
                meta={"chat_jid": msg.chat_jid, "sender": sender_norm},
            )

    async def _post_mailbox(
        self,
        *,
        session_id: str,
        kind: str,
        title: str,
        body: str,
        from_agent: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.mailbox:
            return
        try:
            await self.mailbox.post(
                MailboxMessage(
                    from_agent=from_agent,
                    to="user",
                    kind=kind,
                    title=title,
                    body=body,
                    session_id=session_id,
                    task_id=None,
                    meta=meta or {},
                )
            )
        except Exception:
            pass
