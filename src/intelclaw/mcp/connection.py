"""
MCPServerConnection - stdio MCP client connection lifecycle.

Uses the official `mcp` Python client to spawn an MCP server and communicate
over stdin/stdout (stdio transport).
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    from mcp import ClientSession, StdioServerParameters, stdio_client  # type: ignore
    from mcp.types import CallToolResult, Tool  # type: ignore

    MCP_AVAILABLE = True
except Exception:  # pragma: no cover
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    stdio_client = None  # type: ignore
    CallToolResult = None  # type: ignore
    Tool = None  # type: ignore
    MCP_AVAILABLE = False


@dataclass(frozen=True)
class MCPServerSpec:
    skill_id: str
    server_id: str
    transport: str
    command: str
    args: List[str]
    env: Dict[str, str]
    cwd: Optional[Path]
    tool_namespace: str
    tool_allowlist: List[str]
    tool_denylist: List[str]

    def key(self) -> Tuple[str, str]:
        return (self.skill_id, self.server_id)


class MCPServerConnection:
    def __init__(self, spec: MCPServerSpec) -> None:
        self.spec = spec
        self._stack: Optional[AsyncExitStack] = None
        self._session: Optional[Any] = None
        self._started = False
        self._lock = asyncio.Lock()

    @property
    def is_started(self) -> bool:
        return self._started

    async def start(self) -> None:
        if self._started:
            return
        if not MCP_AVAILABLE:
            raise RuntimeError("mcp package not available")
        if self.spec.transport != "stdio":
            raise RuntimeError(f"Unsupported MCP transport: {self.spec.transport}")

        logger.info(
            f"Starting MCP server (skill={self.spec.skill_id}, server={self.spec.server_id}): "
            f"{self.spec.command} {' '.join(self.spec.args)}"
        )

        stack = AsyncExitStack()
        # Assign early so a timeout/cancellation can still be cleaned up via shutdown().
        self._stack = stack
        try:
            params = StdioServerParameters(  # type: ignore[misc]
                command=self.spec.command,
                args=list(self.spec.args),
                env=dict(self.spec.env) if self.spec.env else None,
                cwd=str(self.spec.cwd) if self.spec.cwd else None,
            )

            read_stream, write_stream = await stack.enter_async_context(stdio_client(params))  # type: ignore[misc]
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))  # type: ignore[misc]
            await session.initialize()

            self._session = session
            self._started = True
        except asyncio.CancelledError:
            try:
                await stack.aclose()
            finally:
                self._stack = None
                self._session = None
                self._started = False
            raise
        except Exception:
            try:
                await stack.aclose()
            finally:
                self._stack = None
                self._session = None
                self._started = False
            raise

    async def shutdown(self) -> None:
        # Close even if we didn't fully start (e.g., timed out during initialize).
        if self._stack is None:
            self._session = None
            self._started = False
            return
        try:
            await self._stack.aclose()
        finally:
            self._stack = None
            self._session = None
            self._started = False

    async def list_tools(self) -> List[Any]:
        await self.start()
        async with self._lock:
            if not self._session:
                raise RuntimeError("MCP session not available")
            res = await self._session.list_tools()
            tools = getattr(res, "tools", None)
            return list(tools or [])

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        await self.start()
        async with self._lock:
            if not self._session:
                raise RuntimeError("MCP session not available")
            return await self._session.call_tool(name=name, arguments=arguments or {})
