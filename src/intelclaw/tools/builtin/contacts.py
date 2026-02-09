"""
Contacts tools - manage `data/contacts.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from intelclaw.contacts.store import ContactsStore
from intelclaw.tools.base import BaseTool, ToolCategory, ToolDefinition, ToolPermission, ToolResult


def _default_contacts_path() -> Path:
    return Path("data") / "contacts.md"


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
                    "whatsapp_jid": m.whatsapp_jid,
                    "inbound_allowed": bool(m.inbound_allowed),
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
                    "whatsapp_jid": {
                        "type": "string",
                        "description": "Optional WhatsApp JID (e.g. 123@s.whatsapp.net or ...@lid)",
                    },
                    "inbound_allowed": {
                        "type": "boolean",
                        "description": "Whether inbound auto-replies are allowed for this contact",
                        "default": False,
                    },
                    "notes": {"type": "string", "description": "Optional notes"},
                },
                "required": ["name", "phone"],
            },
            returns="the updated contact entry",
        )

    async def execute(
        self,
        name: str,
        phone: str,
        whatsapp_jid: Optional[str] = None,
        inbound_allowed: bool = False,
        notes: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        try:
            store = ContactsStore(_default_contacts_path())
            entry = store.upsert(
                name=name,
                phone=phone,
                whatsapp_jid=whatsapp_jid,
                inbound_allowed=bool(inbound_allowed),
                notes=notes,
            )
            return ToolResult(
                success=True,
                data={
                    "name": entry.name,
                    "phone": entry.phone,
                    "whatsapp_jid": entry.whatsapp_jid,
                    "inbound_allowed": bool(entry.inbound_allowed),
                    "notes": entry.notes,
                },
            )
        except Exception as e:
            logger.error(f"contacts_upsert failed: {e}")
            return ToolResult(success=False, error=str(e))

