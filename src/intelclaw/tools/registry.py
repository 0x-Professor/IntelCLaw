"""
Tool Registry - Central registry for all tools.

Manages tool discovery, registration, and execution.
"""

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from langchain_core.tools import BaseTool as LangChainBaseTool
from loguru import logger

from intelclaw.tools.base import BaseTool, ToolDefinition, ToolResult, ToolCategory
from intelclaw.tools.converter import convert_tools_to_openai, convert_tools_to_mcp

if TYPE_CHECKING:
    from intelclaw.config.manager import ConfigManager
    from intelclaw.memory.manager import MemoryManager
    from intelclaw.security.manager import SecurityManager


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
        
        self._tools: Dict[str, BaseTool] = {}
        self._definitions: Dict[str, ToolDefinition] = {}
        self._call_counts: Dict[str, int] = {}
        self._initialized = False
        
        logger.debug("ToolRegistry created")
    
    async def initialize(self) -> None:
        """Initialize and register built-in tools."""
        logger.info("Initializing tool registry...")
        
        # Register built-in tools
        await self._register_builtin_tools()
        
        # Load MCP tools if configured
        mcp_config = self.config.get("mcp", {})
        if mcp_config.get("enabled", True):
            await self._load_mcp_tools()
        
        self._initialized = True
        logger.success(f"Tool registry initialized with {len(self._tools)} tools")
    
    async def shutdown(self) -> None:
        """Shutdown tool registry."""
        logger.info("Shutting down tool registry...")
        self._tools.clear()
        self._definitions.clear()
        logger.info("Tool registry shutdown complete")
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        definition = tool.definition
        self._tools[definition.name] = tool
        self._definitions[definition.name] = definition
        self._call_counts[definition.name] = 0
        logger.debug(f"Registered tool: {definition.name}")
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            del self._definitions[name]
            del self._call_counts[name]
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
        category: Optional[ToolCategory] = None
    ) -> List[ToolDefinition]:
        """
        List all registered tools.
        
        Args:
            category: Filter by category
            
        Returns:
            List of tool definitions
        """
        definitions = list(self._definitions.values())
        
        if category:
            definitions = [d for d in definitions if d.category == category]
        
        return definitions
    
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
    
    async def get_langchain_tools(self) -> List[LangChainBaseTool]:
        """
        Convert registered tools to LangChain tools.
        
        Returns:
            List of LangChain-compatible tools
        """
        langchain_tools = []
        
        for name, internal_tool in self._tools.items():
            definition = internal_tool.definition
            
            # Create async wrapper
            async def make_executor(t=internal_tool):
                async def executor(**kwargs):
                    result = await t.safe_execute(**kwargs)
                    if result.success:
                        return result.data
                    return f"Error: {result.error}"
                return executor
            
            executor = await make_executor()
            
            # Build args_schema from definition parameters
            args_schema = None
            if definition.parameters and "properties" in definition.parameters:
                from pydantic import create_model, Field
                from typing import Optional
                
                fields = {}
                props = definition.parameters.get("properties", {})
                required = definition.parameters.get("required", [])
                
                for prop_name, prop_info in props.items():
                    prop_type = str  # Default to string
                    if prop_info.get("type") == "integer":
                        prop_type = int
                    elif prop_info.get("type") == "boolean":
                        prop_type = bool
                    elif prop_info.get("type") == "number":
                        prop_type = float
                    
                    default_val = prop_info.get("default", ...)
                    if prop_name not in required and default_val is ...:
                        default_val = None
                        prop_type = Optional[prop_type]
                    
                    fields[prop_name] = (prop_type, Field(
                        default=default_val,
                        description=prop_info.get("description", "")
                    ))
                
                if fields:
                    args_schema = create_model(f"{definition.name}Args", **fields)
            
            # Use StructuredTool for proper async support
            from langchain_core.tools import StructuredTool
            
            lc_tool = StructuredTool.from_function(
                func=lambda **kw: asyncio.run(executor(**kw)),
                coroutine=executor,
                name=definition.name,
                description=definition.description,
                args_schema=args_schema,
            )
            
            langchain_tools.append(lc_tool)
        
        return langchain_tools

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
        # MCP integration would go here
        # For now, this is a placeholder
        logger.debug("MCP tool loading not yet implemented")
    
    def reset_rate_limits(self) -> None:
        """Reset all rate limit counters."""
        for name in self._call_counts:
            self._call_counts[name] = 0
