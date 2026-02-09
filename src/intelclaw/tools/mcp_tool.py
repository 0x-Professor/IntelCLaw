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
                    call_args["recipient"] = _normalize_whatsapp_recipient(call_args.get("recipient"))

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
