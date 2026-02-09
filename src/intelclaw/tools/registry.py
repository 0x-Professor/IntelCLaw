"""
Tool Registry - Central registry for all tools.

Manages tool discovery, registration, and execution.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from langchain_core.tools import BaseTool as LangChainBaseTool
from loguru import logger

from intelclaw.tools.base import BaseTool, ToolDefinition, ToolResult, ToolCategory
from intelclaw.tools.converter import convert_tools_to_openai, convert_tools_to_mcp
from intelclaw.mcp.manager import MCPManager, MCPTimeouts
from intelclaw.mcp.connection import MCPServerSpec

if TYPE_CHECKING:
    from intelclaw.config.manager import ConfigManager
    from intelclaw.memory.manager import MemoryManager
    from intelclaw.security.manager import SecurityManager
    from intelclaw.skills.manager import SkillManager
    from intelclaw.core.events import EventBus


class ToolRegistry:
    """
    Central registry for all IntelCLaw tools.
    
    Features:
    - Tool registration and discovery
    - Permission checking
    - Rate limiting
    - LangChain tool conversion
    - MCP server integration
    """
    
    def __init__(
        self,
        config: "ConfigManager",
        security: "SecurityManager",
        memory: Optional["MemoryManager"] = None,
        *,
        skills: Optional["SkillManager"] = None,
        event_bus: Optional["EventBus"] = None,
    ):
        """
        Initialize tool registry.
        
        Args:
            config: Configuration manager
            security: Security manager for permissions
        """
        self.config = config
        self.security = security
        self.memory = memory
        self.skills = skills
        self.event_bus = event_bus
        
        self._tools: Dict[str, BaseTool] = {}
        self._definitions: Dict[str, ToolDefinition] = {}
        self._call_counts: Dict[str, int] = {}
        self._initialized = False
        self._revision = 0

        # MCP integration
        def _timeout(key: str, default: float) -> float:
            try:
                return float(self.config.get(key, default))
            except Exception:
                return float(default)

        self._mcp = MCPManager(
            timeouts=MCPTimeouts(
                start_seconds=_timeout("mcp.timeouts.start_seconds", 30.0),
                list_tools_seconds=_timeout("mcp.timeouts.list_tools_seconds", 30.0),
                call_tool_seconds=_timeout("mcp.timeouts.call_tool_seconds", 300.0),
            )
        )
        self._mcp_reload_lock = asyncio.Lock()
        self._mcp_tool_names: Set[str] = set()
        self._mcp_tools_by_skill: Dict[str, List[str]] = {}
        self._mcp_tool_to_skill: Dict[str, str] = {}
        self._mcp_tool_to_server: Dict[str, Tuple[str, str]] = {}

        # LangChain wrapper cache
        self._lc_cache_revision = -1
        self._lc_tool_cache: Dict[str, LangChainBaseTool] = {}
        
        logger.debug("ToolRegistry created")

    @property
    def revision(self) -> int:
        return int(self._revision)

    def _bump_revision(self) -> None:
        self._revision += 1
        # Invalidate LangChain cache on registry changes
        self._lc_tool_cache.clear()
        self._lc_cache_revision = self._revision
    
    async def initialize(self) -> None:
        """Initialize and register built-in tools."""
        logger.info("Initializing tool registry...")
        
        # Register built-in tools
        await self._register_builtin_tools()
        
        # Load MCP tools if configured
        mcp_config = self.config.get("mcp", {})
        if mcp_config.get("enabled", True):
            await self._load_mcp_tools()

        # Auto-reload MCP tools on skill changes (best-effort)
        if self.event_bus:
            try:
                await self.event_bus.subscribe("skills.changed", self._on_skills_changed)
            except Exception as e:
                logger.debug(f"Failed to subscribe to skills.changed: {e}")
        
        self._initialized = True
        logger.success(f"Tool registry initialized with {len(self._tools)} tools")
    
    async def shutdown(self) -> None:
        """Shutdown tool registry."""
        logger.info("Shutting down tool registry...")
        try:
            await self._mcp.shutdown_unused(set())
        except Exception:
            pass
        self._tools.clear()
        self._definitions.clear()
        logger.info("Tool registry shutdown complete")
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        definition = tool.definition
        self._tools[definition.name] = tool
        self._definitions[definition.name] = definition
        self._call_counts[definition.name] = 0
        self._bump_revision()
        logger.debug(f"Registered tool: {definition.name}")
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            del self._definitions[name]
            del self._call_counts[name]
            self._bump_revision()
            return True
        return False
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self._definitions.get(name)
    
    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        *,
        allowlist: Optional[Set[str]] = None,
    ) -> List[ToolDefinition]:
        """
        List all registered tools.
        
        Args:
            category: Filter by category
            
        Returns:
            List of tool definitions
        """
        definitions = list(self._definitions.values())
        if allowlist is not None:
            allowed = set(allowlist)
            definitions = [d for d in definitions if d.name in allowed]
        
        if category:
            definitions = [d for d in definitions if d.category == category]
        
        return definitions

    def get_mcp_tool_names_for_skill(self, skill_id: str) -> List[str]:
        return list(self._mcp_tools_by_skill.get(str(skill_id or "").strip(), []))

    def get_skill_id_for_tool(self, tool_name: str) -> Optional[str]:
        return self._mcp_tool_to_skill.get(str(tool_name or "").strip())

    def get_mcp_server_health(self, skill_id: str, server_id: str) -> Dict[str, Any]:
        """Best-effort health snapshot for a given MCP server key."""
        try:
            h = self._mcp.get_health_for_key(skill_id, server_id)
            return {"healthy": bool(h.healthy), "last_error": h.last_error}
        except Exception as e:
            return {"healthy": False, "last_error": str(e)}

    async def _on_skills_changed(self, event: Any) -> None:
        # Best-effort: keep MCP tools synced with enabled skills.
        try:
            await self.reload_mcp_tools()
        except Exception as e:
            logger.debug(f"Auto MCP reload on skills.changed failed: {e}")
    
    async def execute(
        self,
        name: str,
        params: Dict[str, Any],
        check_permissions: bool = True
    ) -> Any:
        """
        Execute a tool.
        
        Args:
            name: Tool name
            params: Execution parameters
            check_permissions: Whether to check permissions
            
        Returns:
            Tool result
        """
        if name == "multi_tool_use.parallel":
            return await self._execute_parallel_tool_uses(params or {}, check_permissions=check_permissions)

        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        
        definition = tool.definition
        
        # Check permissions
        if check_permissions:
            for perm in definition.permissions:
                if not await self.security.has_permission(perm.value):
                    raise PermissionError(f"Missing permission: {perm.value}")
        
        # Check rate limit
        if definition.rate_limit:
            if self._call_counts[name] >= definition.rate_limit:
                raise RuntimeError(f"Rate limit exceeded for {name}")
            self._call_counts[name] += 1
        
        # Execute
        result = await tool.safe_execute(**params)
        
        if result.success:
            return result.data
        else:
            raise RuntimeError(f"Tool execution failed: {result.error}")

    @staticmethod
    def _normalize_tool_name(name: Any) -> str:
        """Normalize tool name, stripping known prefixes."""
        if not name:
            return ""
        name_str = str(name).strip()
        for prefix in ("functions.", "tools.", "tool."):
            if name_str.startswith(prefix):
                return name_str[len(prefix):]
        return name_str

    @staticmethod
    def _normalize_tool_args(args: Any) -> Dict[str, Any]:
        """Normalize tool arguments to a dict."""
        if args is None:
            return {}
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                return parsed if isinstance(parsed, dict) else {"input": parsed}
            except Exception:
                return {"input": args}
        if isinstance(args, dict):
            return args
        return {"input": args}

    async def _execute_parallel_tool_uses(
        self,
        params: Dict[str, Any],
        check_permissions: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a batch of tool calls (Codex-style parallel wrapper).

        Expects params to include tool_uses (or tool_calls/calls) list.
        """
        tool_uses = (
            params.get("tool_uses")
            or params.get("tool_calls")
            or params.get("calls")
            or []
        )

        if not isinstance(tool_uses, list):
            raise RuntimeError("multi_tool_use.parallel expects a list under 'tool_uses'")

        if not tool_uses:
            raise RuntimeError("multi_tool_use.parallel requires non-empty tool_uses")

        entries: List[Dict[str, Any]] = []
        tasks: List[Any] = []

        for call in tool_uses:
            if not isinstance(call, dict):
                entries.append({
                    "tool": None,
                    "success": False,
                    "error": "Invalid tool call format (expected object)",
                })
                continue

            raw_name = call.get("recipient_name") or call.get("name") or call.get("tool_name")
            tool_name = self._normalize_tool_name(raw_name)
            call_id = call.get("id") or call.get("tool_call_id")

            if not tool_name:
                entries.append({
                    "tool": raw_name,
                    "id": call_id,
                    "success": False,
                    "error": "Missing tool name",
                })
                continue

            if tool_name == "multi_tool_use.parallel":
                entries.append({
                    "tool": tool_name,
                    "id": call_id,
                    "success": False,
                    "error": "Nested multi_tool_use.parallel is not supported",
                })
                continue

            raw_args = call.get("parameters")
            if raw_args is None:
                raw_args = call.get("args")
            if raw_args is None:
                raw_args = call.get("arguments")

            tool_args = self._normalize_tool_args(raw_args)

            entries.append({
                "tool": tool_name,
                "id": call_id,
                "args": tool_args,
                "task_index": len(tasks),
            })
            tasks.append(self.execute(tool_name, tool_args, check_permissions=check_permissions))

        if not tasks:
            raise RuntimeError("multi_tool_use.parallel received no valid tool calls")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output_results: List[Dict[str, Any]] = []
        success_count = 0

        for entry in entries:
            if "task_index" not in entry:
                output_results.append(entry)
                continue

            result = results[entry["task_index"]]
            if isinstance(result, Exception):
                output_results.append({
                    "tool": entry.get("tool"),
                    "id": entry.get("id"),
                    "args": entry.get("args", {}),
                    "success": False,
                    "error": str(result),
                })
            else:
                success_count += 1
                output_results.append({
                    "tool": entry.get("tool"),
                    "id": entry.get("id"),
                    "args": entry.get("args", {}),
                    "success": True,
                    "result": result,
                })

        summary = {
            "total": len(output_results),
            "succeeded": success_count,
            "failed": len(output_results) - success_count,
        }

        return {"results": output_results, "summary": summary}
    
    async def get_langchain_tools(self, *, allowlist: Optional[Set[str]] = None) -> List[LangChainBaseTool]:
        """
        Convert registered tools to LangChain tools.
        
        Returns:
            List of LangChain-compatible tools
        """
        return await self.get_langchain_tools_filtered(allowlist=allowlist)

    async def get_langchain_tools_filtered(
        self, *, allowlist: Optional[Set[str]] = None
    ) -> List[LangChainBaseTool]:
        """
        Convert registered tools to LangChain tools, optionally filtered by allowlist.
        """
        if self._lc_cache_revision != self._revision:
            self._lc_tool_cache.clear()
            self._lc_cache_revision = self._revision

        wanted = set(allowlist) if allowlist is not None else set(self._tools.keys())
        out: List[LangChainBaseTool] = []

        for name in sorted(wanted):
            internal_tool = self._tools.get(name)
            if not internal_tool:
                continue

            cached = self._lc_tool_cache.get(name)
            if cached is not None:
                out.append(cached)
                continue

            definition = internal_tool.definition

            tool_ref = internal_tool

            async def executor(**kwargs):
                result = await tool_ref.safe_execute(**kwargs)
                if result.success:
                    return result.data
                return f"Error: {result.error}"

            # Build args_schema from definition parameters
            args_schema = None
            if definition.parameters and "properties" in definition.parameters:
                from pydantic import Field, create_model
                from typing import Optional as TypingOptional

                fields: Dict[str, Any] = {}
                props = definition.parameters.get("properties", {}) or {}
                required = set(definition.parameters.get("required", []) or [])

                for prop_name, prop_info in props.items():
                    prop_type: Any = str  # Default to string
                    if isinstance(prop_info, dict):
                        if prop_info.get("type") == "integer":
                            prop_type = int
                        elif prop_info.get("type") == "boolean":
                            prop_type = bool
                        elif prop_info.get("type") == "number":
                            prop_type = float

                    default_val = prop_info.get("default", ...) if isinstance(prop_info, dict) else ...
                    if prop_name not in required and default_val is ...:
                        default_val = None
                        prop_type = TypingOptional[prop_type]

                    fields[prop_name] = (
                        prop_type,
                        Field(default=default_val, description=(prop_info or {}).get("description", "") if isinstance(prop_info, dict) else ""),
                    )

                if fields:
                    args_schema = create_model(f"{definition.name}Args", **fields)

            from langchain_core.tools import StructuredTool

            def sync_executor(**kwargs):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    return asyncio.run(executor(**kwargs))
                raise RuntimeError("Synchronous tool execution is not supported in an active event loop")

            lc_tool = StructuredTool.from_function(
                func=sync_executor,
                coroutine=executor,
                name=definition.name,
                description=definition.description,
                args_schema=args_schema,
            )

            self._lc_tool_cache[name] = lc_tool
            out.append(lc_tool)

        return out

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Convert registered tools to OpenAI tool schema.
        
        Returns:
            List of OpenAI-compatible tool schemas
        """
        return convert_tools_to_openai(list(self._tools.values()))

    def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Convert registered tools to MCP tool schema.
        
        Returns:
            List of MCP-compatible tool schemas
        """
        return convert_tools_to_mcp(list(self._tools.values()))
    
    async def _register_builtin_tools(self) -> None:
        """Register built-in tools."""
        from intelclaw.tools.builtin.search import TavilySearchTool
        from intelclaw.tools.builtin.file_ops import (
            FileReadTool, FileWriteTool, FileSearchTool,
            DirectoryListTool, GetCurrentDirectoryTool,
            FileDeleteTool, FileCopyTool, FileMoveTool
        )
        from intelclaw.tools.builtin.system import ScreenshotTool, ClipboardTool, LaunchAppTool
        from intelclaw.tools.builtin.web import WebScrapeTool
        from intelclaw.tools.builtin.shell import (
            ShellCommandTool, CodeExecutionTool, PipInstallTool,
            PowerShellTool, SystemInfoTool
        )
        from intelclaw.tools.builtin.windows import (
            WindowsServicesTool,
            WindowsTasksTool,
            WindowsRegistryTool,
            WindowsEventLogTool,
            WindowsUIAutomationTool,
            WindowsCIMTool,
        )
        from intelclaw.tools.builtin.windows_extended import (
            ProcessManagementTool,
            NetworkInfoTool,
            DiskManagementTool,
            FirewallTool,
            InstalledAppsTool,
            EnvironmentTool,
            WindowsUpdateTool,
            SystemPerformanceTool,
            UserSecurityTool,
        )
        from intelclaw.tools.builtin.rag import (
            RagIndexPathTool,
            RagListDocumentsTool,
            RagDeleteDocumentTool,
        )
        from intelclaw.tools.builtin.contacts import ContactsLookupTool, ContactsUpsertTool
        
        builtin_tools = [
            # Search tools
            TavilySearchTool(),
            # File operation tools
            FileReadTool(),
            FileWriteTool(),
            FileDeleteTool(),
            FileCopyTool(),
            FileMoveTool(),
            FileSearchTool(),
            DirectoryListTool(),
            GetCurrentDirectoryTool(),
            # System tools
            ScreenshotTool(),
            ClipboardTool(),
            LaunchAppTool(),
            SystemInfoTool(),
            # Web tools
            WebScrapeTool(),
            # Shell and code execution tools
            ShellCommandTool(),
            CodeExecutionTool(),
            PipInstallTool(),
            PowerShellTool(),
            # Windows native tools (core)
            WindowsServicesTool(),
            WindowsTasksTool(),
            WindowsRegistryTool(),
            WindowsEventLogTool(),
            WindowsUIAutomationTool(),
            WindowsCIMTool(),
            # Windows extended tools
            ProcessManagementTool(),
            NetworkInfoTool(),
            DiskManagementTool(),
            FirewallTool(),
            InstalledAppsTool(),
            EnvironmentTool(),
            WindowsUpdateTool(),
            SystemPerformanceTool(),
            UserSecurityTool(),

            # Contacts (data/contacts.md)
            ContactsLookupTool(),
            ContactsUpsertTool(),
        ]

        # RAG ingestion tools (require MemoryManager)
        builtin_tools.extend(
            [
                RagIndexPathTool(memory=self.memory),
                RagListDocumentsTool(memory=self.memory),
                RagDeleteDocumentTool(memory=self.memory),
            ]
        )
        
        for tool in builtin_tools:
            self.register(tool)
    
    async def _load_mcp_tools(self) -> None:
        """Load tools from MCP servers."""
        async with self._mcp_reload_lock:
            specs = await self._collect_enabled_mcp_specs()
            await self._register_mcp_tools(specs)
            await self._mcp.shutdown_unused({s.key() for s in specs})

    async def reload_mcp_tools(self) -> None:
        """Reload MCP tools according to currently enabled skills."""
        async with self._mcp_reload_lock:
            # Unregister previous MCP tools
            for name in list(self._mcp_tool_names):
                try:
                    self.unregister(name)
                except Exception:
                    pass

            self._mcp_tool_names.clear()
            self._mcp_tools_by_skill.clear()
            self._mcp_tool_to_skill.clear()
            self._mcp_tool_to_server.clear()

            specs = await self._collect_enabled_mcp_specs()
            await self._register_mcp_tools(specs)
            await self._mcp.shutdown_unused({s.key() for s in specs})

    async def _collect_enabled_mcp_specs(self) -> List[MCPServerSpec]:
        if not self.skills:
            return []
        try:
            entries = await self.skills.get_enabled_mcp_server_specs()
        except Exception as e:
            logger.debug(f"Failed to get enabled MCP server specs: {e}")
            return []

        specs: List[MCPServerSpec] = []
        for skill_id, server, skill_dir in entries:
            try:
                cwd_val = getattr(server, "cwd", None)
                cwd_path: Optional[Path] = None
                if cwd_val:
                    p = Path(str(cwd_val))
                    cwd_path = p if p.is_absolute() else (Path(skill_dir) / p).resolve()

                specs.append(
                    MCPServerSpec(
                        skill_id=str(skill_id),
                        server_id=str(server.id),
                        transport=str(server.transport),
                        command=str(server.command),
                        args=[str(a) for a in (server.args or [])],
                        env={str(k): str(v) for k, v in (server.env or {}).items()},
                        cwd=cwd_path,
                        tool_namespace=str(server.tool_namespace or "default"),
                        tool_allowlist=[str(x) for x in (server.tool_allowlist or [])],
                        tool_denylist=[str(x) for x in (server.tool_denylist or [])],
                    )
                )
            except Exception as e:
                logger.debug(f"Skipping MCP server spec due to parse error: {e}")

        return specs

    async def _register_mcp_tools(self, specs: List[MCPServerSpec]) -> None:
        if not specs:
            return

        from intelclaw.tools.mcp_tool import MCPRemoteTool

        for spec in specs:
            try:
                tools = await self._mcp.get_tools(spec, refresh=True)
            except Exception as e:
                logger.warning(
                    f"Failed to list MCP tools for {spec.skill_id}/{spec.server_id}: {e}"
                )
                continue

            for t in tools:
                try:
                    mcp_tool_name = getattr(t, "name", None) or ""
                    if not str(mcp_tool_name).strip():
                        continue
                    description = getattr(t, "description", None) or ""
                    input_schema = getattr(t, "inputSchema", None) or {}

                    remote = MCPRemoteTool(
                        mcp_manager=self._mcp,
                        spec=spec,
                        mcp_tool_name=str(mcp_tool_name),
                        description=str(description or ""),
                        input_schema=input_schema if isinstance(input_schema, dict) else {},
                    )

                    self.register(remote)
                    name = remote.definition.name
                    self._mcp_tool_names.add(name)
                    self._mcp_tool_to_skill[name] = spec.skill_id
                    self._mcp_tool_to_server[name] = spec.key()
                    self._mcp_tools_by_skill.setdefault(spec.skill_id, []).append(name)
                except Exception as e:
                    logger.debug(f"Failed to register MCP tool: {e}")
    
    def reset_rate_limits(self) -> None:
        """Reset all rate limit counters."""
        for name in self._call_counts:
            self._call_counts[name] = 0
