"""
MCPRemoteTool - expose an MCP server tool as an IntelCLaw tool.
"""

from __future__ import annotations

import re
import asyncio
from typing import Any, Dict, Optional

from loguru import logger

from intelclaw.mcp.connection import MCPServerSpec
from intelclaw.mcp.manager import MCPManager
from intelclaw.tools.base import BaseTool, ToolCategory, ToolDefinition, ToolPermission, ToolResult


def normalize_tool_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "tool"
    if s[0].isdigit():
        s = "tool_" + s
    return s


def _ensure_object_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    if schema.get("type") != "object":
        # Some MCP servers provide only properties/required; normalize.
        if "properties" in schema or "required" in schema:
            schema = {"type": "object", **schema}
        else:
            schema = {"type": "object", "properties": schema}
    schema.setdefault("properties", {})
    return schema


def _flatten_mcp_content(content: Any) -> str:
    if not content:
        return ""
    parts = []
    for block in list(content):
        try:
            # Pydantic models (mcp.types.*)
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
                continue

            # dict-like
            if isinstance(block, dict):
                if isinstance(block.get("text"), str) and block["text"].strip():
                    parts.append(block["text"].strip())
                    continue

            parts.append(str(block))
        except Exception:
            parts.append(str(block))
    return "\n".join([p for p in parts if p])


def _payload_indicates_failure(data: Any) -> Optional[str]:
    """
    Some MCP servers return a structured payload with {success: false, message: "..."} while
    the MCP call itself is not marked as isError. Treat these as tool failures so the rest
    of the agent stack can retry/handle appropriately.
    """
    if not isinstance(data, dict):
        return None
    if data.get("success") is False:
        msg = data.get("message") or data.get("error") or "MCP tool reported success=false"
        return str(msg)
    return None


def _normalize_whatsapp_recipient(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return s
    # If it's already a JID, keep as-is.
    if "@" in s:
        return s
    # Strip any formatting like +, spaces, hyphens, parentheses.
    digits = re.sub(r"\D+", "", s)
    return digits or s


def _resolve_whatsapp_recipient_via_contacts(value: Any) -> Any:
    """
    Resolve a WhatsApp recipient string into the format expected by WhatsApp MCP tools:
    - Prefer returning a JID if available (contains '@')
    - Otherwise return digits-only phone number

    If the value looks like a contact name (non-empty, no '@', no digits),
    attempt to resolve via `data/contacts.md`.
    """
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return raw
    if "@" in raw:
        return raw

    digits = re.sub(r"\D+", "", raw)
    if digits:
        return digits

    # Treat as contact name
    try:
        from pathlib import Path

        from intelclaw.contacts.store import ContactsStore

        store = ContactsStore(Path("data") / "contacts.md")
        candidates = [raw]
        cleaned = raw
        cleaned = re.sub(r"(?i)\b(on|via)\s+whatsapp\b", "", cleaned).strip()
        cleaned = re.sub(r"(?i)\bwhatsapp\b", "", cleaned).strip()
        cleaned = re.sub(r"(?i)^(send|text|message)\s+(a\s+)?(whatsapp\s+)?(message\s+)?to\s+", "", cleaned).strip()
        cleaned = re.sub(r"(?i)^to\s+", "", cleaned).strip()
        cleaned = cleaned.strip(" \"'")
        if cleaned and cleaned.lower() != raw.lower():
            candidates.append(cleaned)

        for q in candidates:
            matches = store.lookup(q)
            if not matches:
                continue
            exact = [m for m in matches if (m.name or "").strip().lower() == q.lower()]
            m = exact[0] if exact else matches[0]
            resolved = m.whatsapp_jid or m.phone
            if resolved:
                return resolved

        return raw
    except Exception:
        return raw


def _looks_like_internal_error_text(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return True
    sl = s.lower()
    if sl.startswith("error:") or sl.startswith("tool execution failed"):
        return True
    if "pydantic" in sl or "validation error" in sl:
        return True
    if "traceback" in sl or "exception" in sl:
        return True
    if "http 500" in sl or "http 400" in sl:
        return True
    if "no required module provides package" in sl:
        return True
    if sl in {"...", "tbd", "todo"}:
        return True
    # Common tool-result payload accidentally forwarded as message text.
    if sl.startswith("{") and ("\"success\"" in sl or "'success'" in sl) and ("\"error\"" in sl or "'error'" in sl):
        return True
    return False


def _best_effort_whatsapp_jid_from_bridge_db(phone_digits: str) -> Optional[str]:
    """
    Best-effort resolve a chat JID (e.g. ...@s.whatsapp.net or ...@lid) from the local
    WhatsApp bridge SQLite DB. This avoids extra MCP round-trips just to convert phone->JID.
    """
    digits = re.sub(r"\D+", "", str(phone_digits or ""))
    if not digits:
        return None
    try:
        import sqlite3
        from pathlib import Path

        db_path = Path("data") / "vendor" / "whatsapp-mcp" / "whatsapp-bridge" / "store" / "messages.db"
        if not db_path.exists():
            return None
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT jid
                FROM chats
                WHERE jid LIKE ? AND jid NOT LIKE '%@g.us'
                ORDER BY last_message_time DESC
                LIMIT 1
                """,
                (f"%{digits}%",),
            )
            row = cur.fetchone()
            if not row:
                return None
            jid = str(row[0] or "").strip()
            return jid or None
        finally:
            conn.close()
    except Exception:
        return None


class MCPRemoteTool(BaseTool):
    def __init__(
        self,
        *,
        mcp_manager: MCPManager,
        spec: MCPServerSpec,
        mcp_tool_name: str,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._mcp = mcp_manager
        self._spec = spec
        self._mcp_tool_name = str(mcp_tool_name or "").strip()

        normalized = normalize_tool_name(self._mcp_tool_name)
        self._name = f"mcp_{normalize_tool_name(spec.tool_namespace)}__{normalized}"

        self._definition = ToolDefinition(
            name=self._name,
            description=description or f"MCP tool '{self._mcp_tool_name}' from {spec.skill_id}/{spec.server_id}",
            category=ToolCategory.CUSTOM,
            permissions=[ToolPermission.EXECUTE],
            parameters=_ensure_object_schema(input_schema or {}),
            returns="any",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    @property
    def skill_id(self) -> str:
        return self._spec.skill_id

    @property
    def server_id(self) -> str:
        return self._spec.server_id

    @property
    def mcp_tool_name(self) -> str:
        return self._mcp_tool_name

    async def execute(self, **kwargs) -> ToolResult:
        try:
            call_args = dict(kwargs or {})

            # Skill-specific pre-processing for common WhatsApp inputs.
            if normalize_tool_name(self._spec.tool_namespace) == "whatsapp" and self._mcp_tool_name in {
                "send_message",
                "send_file",
                "send_audio_message",
            }:
                if "recipient" in call_args:
                    resolved = _resolve_whatsapp_recipient_via_contacts(call_args.get("recipient"))

                    if isinstance(resolved, str):
                        rec_s = resolved.strip()
                        # Upgrade digits-only recipients to a real chat JID when possible.
                        if rec_s and "@" not in rec_s and rec_s.isdigit():
                            jid = _best_effort_whatsapp_jid_from_bridge_db(rec_s)
                            if jid:
                                resolved = jid
                                # Best-effort: persist back to contacts.md so future sends prefer JID.
                                try:
                                    from pathlib import Path

                                    from intelclaw.contacts.store import ContactsStore, normalize_phone

                                    store = ContactsStore(Path("data") / "contacts.md")
                                    phone_norm = normalize_phone(rec_s)
                                    entries = [e for e in store.load() if normalize_phone(e.phone) == phone_norm]
                                    if len(entries) == 1 and not str(entries[0].whatsapp_jid or "").strip():
                                        store.upsert(
                                            name=entries[0].name,
                                            phone=entries[0].phone,
                                            gender=None,
                                            whatsapp_jid=str(jid),
                                            inbound_allowed=None,
                                            persona=None,
                                            notes=None,
                                        )
                                except Exception:
                                    pass

                        # Refuse to send if recipient still isn't in an acceptable format.
                        rec_final = str(resolved).strip()
                        if rec_final and "@" not in rec_final and not rec_final.isdigit():
                            return ToolResult(
                                success=False,
                                error=(
                                    "Invalid WhatsApp recipient. Use a digits-only phone number, a JID, "
                                    "or save the person in contacts.md and reference them by name."
                                ),
                                metadata={
                                    "skill_id": self._spec.skill_id,
                                    "server_id": self._spec.server_id,
                                    "mcp_tool_name": self._mcp_tool_name,
                                },
                            )

                    call_args["recipient"] = resolved

                if self._mcp_tool_name == "send_message":
                    msg_text = call_args.get("message")
                    if not isinstance(msg_text, str) or _looks_like_internal_error_text(msg_text):
                        return ToolResult(
                            success=False,
                            error="Refusing to send placeholder/internal error text as a WhatsApp message.",
                            metadata={
                                "skill_id": self._spec.skill_id,
                                "server_id": self._spec.server_id,
                                "mcp_tool_name": self._mcp_tool_name,
                            },
                        )

            async def _call_once() -> Any:
                return await self._mcp.call(self._spec, self._mcp_tool_name, args=dict(call_args or {}))

            res = await _call_once()

            is_error = bool(getattr(res, "isError", False))
            structured = getattr(res, "structuredContent", None)
            content = getattr(res, "content", None)

            if is_error:
                msg = _flatten_mcp_content(content) or "MCP tool returned an error"
                return ToolResult(
                    success=False,
                    error=msg,
                    metadata={
                        "skill_id": self._spec.skill_id,
                        "server_id": self._spec.server_id,
                        "mcp_tool_name": self._mcp_tool_name,
                    },
                )

            data: Any = structured if structured is not None else _flatten_mcp_content(content)

            # Normalize "success: false" payloads into tool failures.
            payload_error = _payload_indicates_failure(data)
            if payload_error:
                # Best-effort retry for transient WhatsApp usync/device-list timeouts.
                if (
                    normalize_tool_name(self._spec.tool_namespace) == "whatsapp"
                    and any(x in payload_error.lower() for x in ("info query timed out", "failed to get device list", "usync query"))
                ):
                    try:
                        await asyncio.sleep(3.0)
                        res2 = await _call_once()
                        is_error2 = bool(getattr(res2, "isError", False))
                        structured2 = getattr(res2, "structuredContent", None)
                        content2 = getattr(res2, "content", None)
                        if is_error2:
                            msg2 = _flatten_mcp_content(content2) or payload_error
                            return ToolResult(
                                success=False,
                                error=msg2,
                                metadata={
                                    "skill_id": self._spec.skill_id,
                                    "server_id": self._spec.server_id,
                                    "mcp_tool_name": self._mcp_tool_name,
                                },
                            )
                        data2: Any = structured2 if structured2 is not None else _flatten_mcp_content(content2)
                        payload_error2 = _payload_indicates_failure(data2)
                        if not payload_error2:
                            return ToolResult(
                                success=True,
                                data=data2,
                                metadata={
                                    "skill_id": self._spec.skill_id,
                                    "server_id": self._spec.server_id,
                                    "mcp_tool_name": self._mcp_tool_name,
                                },
                            )
                        payload_error = payload_error2
                    except Exception:
                        pass

                return ToolResult(
                    success=False,
                    error=payload_error,
                    metadata={
                        "skill_id": self._spec.skill_id,
                        "server_id": self._spec.server_id,
                        "mcp_tool_name": self._mcp_tool_name,
                    },
                )

            return ToolResult(
                success=True,
                data=data,
                metadata={
                    "skill_id": self._spec.skill_id,
                    "server_id": self._spec.server_id,
                    "mcp_tool_name": self._mcp_tool_name,
                },
            )
        except Exception as e:
            logger.debug(f"MCPRemoteTool execute failed ({self._name}): {e}")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={
                    "skill_id": self._spec.skill_id,
                    "server_id": self._spec.server_id,
                    "mcp_tool_name": self._mcp_tool_name,
                },
            )
