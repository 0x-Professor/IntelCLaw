"""
Contacts tools - manage `data/contacts.md`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from intelclaw.contacts.store import ContactsStore, normalize_phone
from intelclaw.tools.base import BaseTool, ToolCategory, ToolDefinition, ToolPermission, ToolResult


def _default_contacts_path() -> Path:
    return Path("data") / "contacts.md"


def _default_whatsapp_messages_db_path() -> Path:
    return Path("data") / "vendor" / "whatsapp-mcp" / "whatsapp-bridge" / "store" / "messages.db"


def _best_effort_resolve_whatsapp_jid(*, phone: str, name: str) -> Optional[str]:
    """
    Resolve a WhatsApp chat JID for a contact from the local WhatsApp bridge DB.

    This mirrors the WhatsApp MCP server's `search_contacts` behavior (based on the same DB),
    but keeps contacts management functional even when the MCP server isn't running.
    """
    db_path = _default_whatsapp_messages_db_path()
    if not db_path.exists():
        return None

    phone_norm = normalize_phone(phone)
    name_norm = str(name or "").strip()
    if not phone_norm and not name_norm:
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            # Prefer phone match if available.
            if phone_norm:
                cur.execute(
                    """
                    SELECT jid, name
                    FROM chats
                    WHERE jid LIKE ? AND jid NOT LIKE '%@g.us'
                    ORDER BY last_message_time DESC
                    LIMIT 10
                    """,
                    (f"%{phone_norm}%",),
                )
                rows = cur.fetchall()
                for jid, _nm in rows:
                    jid_str = str(jid or "").strip()
                    if jid_str and jid_str.split("@", 1)[0] == phone_norm:
                        return jid_str
                if rows:
                    jid0 = str(rows[0][0] or "").strip()
                    return jid0 or None

            # Fall back to name search.
            if name_norm:
                cur.execute(
                    """
                    SELECT jid, name
                    FROM chats
                    WHERE LOWER(name) LIKE LOWER(?) AND jid NOT LIKE '%@g.us'
                    ORDER BY last_message_time DESC
                    LIMIT 10
                    """,
                    (f"%{name_norm}%",),
                )
                rows = cur.fetchall()
                if not rows:
                    return None
                for jid, nm in rows:
                    if str(nm or "").strip().lower() == name_norm.lower():
                        jid_str = str(jid or "").strip()
                        return jid_str or None
                jid0 = str(rows[0][0] or "").strip()
                return jid0 or None
        finally:
            conn.close()
    except Exception:
        return None


class ContactsLookupTool(BaseTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="contacts_lookup",
            description="Lookup a contact from data/contacts.md by name or phone substring.",
            category=ToolCategory.PRODUCTIVITY,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Name or phone substring to search"},
                    "inbound_only": {
                        "type": "boolean",
                        "description": "Only return contacts with inbound_allowed=yes",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
            returns="list of contacts",
        )

    async def execute(self, query: str, inbound_only: bool = False, **kwargs) -> ToolResult:
        try:
            store = ContactsStore(_default_contacts_path())
            matches = store.lookup(query, inbound_only=bool(inbound_only))
            data: List[Dict[str, Any]] = [
                {
                    "name": m.name,
                    "phone": m.phone,
                    "gender": m.gender,
                    "whatsapp_jid": m.whatsapp_jid,
                    "inbound_allowed": bool(m.inbound_allowed),
                    "persona": m.persona,
                    "notes": m.notes,
                }
                for m in matches
            ]
            return ToolResult(success=True, data=data)
        except Exception as e:
            logger.error(f"contacts_lookup failed: {e}")
            return ToolResult(success=False, error=str(e))


class ContactsUpsertTool(BaseTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="contacts_upsert",
            description="Add or update a contact entry in data/contacts.md.",
            category=ToolCategory.PRODUCTIVITY,
            permissions=[ToolPermission.WRITE],
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Contact display name"},
                    "phone": {
                        "type": "string",
                        "description": "Phone number (any formatting ok; will be normalized to digits).",
                    },
                    "gender": {
                        "type": "string",
                        "description": "Optional gender (e.g. male/female/other).",
                    },
                    "whatsapp_jid": {
                        "type": "string",
                        "description": "Optional WhatsApp JID (e.g. 123@s.whatsapp.net or ...@lid)",
                    },
                    "inbound_allowed": {
                        "type": "boolean",
                        "description": "Whether inbound auto-replies are allowed for this contact",
                        "default": False,
                    },
                    "persona": {
                        "type": "string",
                        "description": "Optional short persona/relationship notes for how to message this person.",
                    },
                    "notes": {"type": "string", "description": "Optional notes"},
                    "resolve_whatsapp": {
                        "type": "boolean",
                        "description": "Best-effort resolve WhatsApp JID from local bridge DB if available.",
                        "default": True,
                    },
                },
                "required": ["name", "phone"],
            },
            returns="the updated contact entry",
        )

    async def execute(
        self,
        name: str,
        phone: str,
        gender: Optional[str] = None,
        whatsapp_jid: Optional[str] = None,
        inbound_allowed: bool = False,
        persona: Optional[str] = None,
        notes: Optional[str] = None,
        resolve_whatsapp: bool = True,
        **kwargs,
    ) -> ToolResult:
        try:
            store = ContactsStore(_default_contacts_path())
            jid_val = whatsapp_jid
            if resolve_whatsapp and (not isinstance(jid_val, str) or not jid_val.strip()):
                jid_val = _best_effort_resolve_whatsapp_jid(phone=phone, name=name)
            entry = store.upsert(
                name=name,
                phone=phone,
                gender=gender,
                whatsapp_jid=jid_val,
                inbound_allowed=bool(inbound_allowed),
                persona=persona,
                notes=notes,
            )
            return ToolResult(
                success=True,
                data={
                    "name": entry.name,
                    "phone": entry.phone,
                    "gender": entry.gender,
                    "whatsapp_jid": entry.whatsapp_jid,
                    "inbound_allowed": bool(entry.inbound_allowed),
                    "persona": entry.persona,
                    "notes": entry.notes,
                },
            )
        except Exception as e:
            logger.error(f"contacts_upsert failed: {e}")
            return ToolResult(success=False, error=str(e))


class ContactsSetInboundAllowedTool(BaseTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="contacts_set_inbound_allowed",
            description="Enable/disable inbound auto-replies for a contact in data/contacts.md.",
            category=ToolCategory.PRODUCTIVITY,
            permissions=[ToolPermission.WRITE],
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Contact name or phone to match"},
                    "allowed": {"type": "boolean", "description": "Whether inbound is allowed"},
                },
                "required": ["query", "allowed"],
            },
            returns="the updated contact entry",
        )

    async def execute(self, query: str, allowed: bool, **kwargs) -> ToolResult:
        try:
            store = ContactsStore(_default_contacts_path())
            q = str(query or "").strip()
            if not q:
                return ToolResult(success=False, error="query is required")

            phone_norm = normalize_phone(q)
            all_entries = store.load()

            exact = [
                e
                for e in all_entries
                if (e.name or "").strip().lower() == q.lower() or (phone_norm and normalize_phone(e.phone) == phone_norm)
            ]
            matches = exact if exact else store.lookup(q)

            if not matches:
                return ToolResult(success=False, error=f"No contact matched query: {q}")
            if len(matches) > 1:
                return ToolResult(
                    success=False,
                    error=f"Multiple contacts matched '{q}'. Use a more specific name/number.",
                    data=[
                        {
                            "name": m.name,
                            "phone": m.phone,
                            "whatsapp_jid": m.whatsapp_jid,
                            "inbound_allowed": bool(m.inbound_allowed),
                        }
                        for m in matches[:10]
                    ],
                )

            m = matches[0]
            entry = store.upsert(
                name=m.name,
                phone=m.phone,
                gender=None,
                whatsapp_jid=None,
                inbound_allowed=bool(allowed),
                persona=None,
                notes=None,
            )
            return ToolResult(
                success=True,
                data={
                    "name": entry.name,
                    "phone": entry.phone,
                    "gender": entry.gender,
                    "whatsapp_jid": entry.whatsapp_jid,
                    "inbound_allowed": bool(entry.inbound_allowed),
                    "persona": entry.persona,
                    "notes": entry.notes,
                },
            )
        except Exception as e:
            logger.error(f"contacts_set_inbound_allowed failed: {e}")
            return ToolResult(success=False, error=str(e))
