"""
MCPManager - connection pool and tool cache for MCP servers.
"""

from __future__ import annotations

import asyncio
import fnmatch
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from intelclaw.mcp.connection import MCPServerConnection, MCPServerSpec


@dataclass
class MCPServerHealth:
    healthy: bool
    last_error: Optional[str] = None


@dataclass(frozen=True)
class MCPTimeouts:
    start_seconds: float = 30.0
    list_tools_seconds: float = 30.0
    call_tool_seconds: float = 300.0


class MCPManager:
    def __init__(self, *, timeouts: Optional[MCPTimeouts] = None) -> None:
        self._lock = asyncio.Lock()
        self._conns: Dict[Tuple[str, str], MCPServerConnection] = {}
        self._tools_cache: Dict[Tuple[str, str], List[Any]] = {}
        self._health: Dict[Tuple[str, str], MCPServerHealth] = {}
        self._timeouts = timeouts or MCPTimeouts()
        self._shutdown_timeout_seconds = 5.0

    async def _shutdown_and_drop(self, key: Tuple[str, str], conn: MCPServerConnection, *, reason: str) -> None:
        try:
            await asyncio.wait_for(conn.shutdown(), timeout=float(self._shutdown_timeout_seconds))
        except asyncio.TimeoutError:
            logger.debug(f"MCP shutdown timed out ({key}) while handling {reason}")
        except Exception as e:
            logger.debug(f"MCP shutdown failed ({key}) while handling {reason}: {e}")
        async with self._lock:
            self._conns.pop(key, None)
            self._tools_cache.pop(key, None)

    def get_health(self, spec: MCPServerSpec) -> MCPServerHealth:
        return self._health.get(spec.key(), MCPServerHealth(healthy=False, last_error="not started"))

    def get_health_for_key(self, skill_id: str, server_id: str) -> MCPServerHealth:
        key = (str(skill_id or "").strip(), str(server_id or "").strip())
        return self._health.get(key, MCPServerHealth(healthy=False, last_error="not started"))

    async def ensure_started(self, spec: MCPServerSpec) -> MCPServerConnection:
        key = spec.key()
        async with self._lock:
            conn = self._conns.get(key)
            if conn is None:
                conn = MCPServerConnection(spec)
                self._conns[key] = conn

        try:
            timeout_s = float(self._timeouts.start_seconds)
            await asyncio.wait_for(conn.start(), timeout=timeout_s)
            self._health[key] = MCPServerHealth(healthy=True, last_error=None)
            return conn
        except asyncio.TimeoutError:
            self._health[key] = MCPServerHealth(
                healthy=False, last_error=f"timeout after {self._timeouts.start_seconds}s during start"
            )
            await self._shutdown_and_drop(key, conn, reason="start timeout")
            raise
        except Exception as e:
            self._health[key] = MCPServerHealth(healthy=False, last_error=str(e))
            await self._shutdown_and_drop(key, conn, reason="start error")
            raise

    async def shutdown_unused(self, keep_keys: Iterable[Tuple[str, str]]) -> None:
        keep = set(keep_keys)
        async with self._lock:
            keys = list(self._conns.keys())

        for key in keys:
            if key in keep:
                continue
            conn = self._conns.get(key)
            if not conn:
                continue
            await self._shutdown_and_drop(key, conn, reason="shutdown_unused")
            async with self._lock:
                self._health.pop(key, None)

    @staticmethod
    def _tool_allowed(name: str, allowlist: List[str], denylist: List[str]) -> bool:
        n = str(name or "")
        if denylist:
            for pat in denylist:
                if not pat:
                    continue
                if fnmatch.fnmatchcase(n, pat):
                    return False
        if allowlist:
            for pat in allowlist:
                if not pat:
                    continue
                if fnmatch.fnmatchcase(n, pat):
                    return True
            return False
        return True

    async def get_tools(self, spec: MCPServerSpec, *, refresh: bool = False) -> List[Any]:
        key = spec.key()
        if not refresh and key in self._tools_cache:
            return list(self._tools_cache[key])

        conn = await self.ensure_started(spec)
        try:
            timeout_s = float(self._timeouts.list_tools_seconds)
            tools = await asyncio.wait_for(conn.list_tools(), timeout=timeout_s)
            filtered = [t for t in tools if self._tool_allowed(getattr(t, "name", ""), spec.tool_allowlist, spec.tool_denylist)]
            self._tools_cache[key] = filtered
            self._health[key] = MCPServerHealth(healthy=True, last_error=None)
            return list(filtered)
        except asyncio.TimeoutError:
            self._health[key] = MCPServerHealth(
                healthy=False, last_error=f"timeout after {self._timeouts.list_tools_seconds}s during list_tools"
            )
            await self._shutdown_and_drop(key, conn, reason="list_tools timeout")
            raise
        except Exception as e:
            self._health[key] = MCPServerHealth(healthy=False, last_error=str(e))
            await self._shutdown_and_drop(key, conn, reason="list_tools error")
            raise

    async def call(self, spec: MCPServerSpec, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Any:
        key = spec.key()

        async def _attempt() -> Any:
            conn = await self.ensure_started(spec)
            timeout_s = float(self._timeouts.call_tool_seconds)
            return await asyncio.wait_for(conn.call_tool(tool_name, args or {}), timeout=timeout_s)

        try:
            res = await _attempt()
            self._health[key] = MCPServerHealth(healthy=True, last_error=None)
            return res
        except asyncio.TimeoutError:
            self._health[key] = MCPServerHealth(
                healthy=False, last_error=f"timeout after {self._timeouts.call_tool_seconds}s during call_tool"
            )
            try:
                conn = self._conns.get(key)
                if conn:
                    await self._shutdown_and_drop(key, conn, reason="call_tool timeout")
            except Exception:
                pass
            raise
        except Exception as e:
            # One retry after restart
            logger.warning(f"MCP call failed ({key}, tool={tool_name}), retrying once: {e}")
            try:
                conn = self._conns.get(key)
                if conn:
                    await self._shutdown_and_drop(key, conn, reason="call_tool error retry")
            except Exception:
                pass

            try:
                res = await _attempt()
                self._health[key] = MCPServerHealth(healthy=True, last_error=None)
                return res
            except Exception as e2:
                self._health[key] = MCPServerHealth(healthy=False, last_error=str(e2))
                raise
