"""
Tool Registry - Central registry for all tools.

Manages tool discovery, registration, and execution.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from langchain_core.tools import BaseTool as LangChainBaseTool, tool
from loguru import logger

from intelclaw.tools.base import BaseTool, ToolDefinition, ToolResult, ToolCategory

if TYPE_CHECKING:
    from intelclaw.config.manager import ConfigManager
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
    ):
        """
        Initialize tool registry.
        
        Args:
            config: Configuration manager
            security: Security manager for permissions
        """
        self.config = config
        self.security = security
        
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
            
            # Create LangChain tool using the @tool decorator pattern
            @tool(name=definition.name, description=definition.description)
            async def langchain_wrapper(**kwargs):
                return await executor(**kwargs)
            
            # Actually, let's use StructuredTool for better control
            from langchain_core.tools import StructuredTool
            
            lc_tool = StructuredTool.from_function(
                func=lambda **kw: asyncio.run(executor(**kw)),
                coroutine=executor,
                name=definition.name,
                description=definition.description,
            )
            
            langchain_tools.append(lc_tool)
        
        return langchain_tools
    
    async def _register_builtin_tools(self) -> None:
        """Register built-in tools."""
        from intelclaw.tools.builtin.search import TavilySearchTool
        from intelclaw.tools.builtin.file_ops import FileReadTool, FileWriteTool, FileSearchTool
        from intelclaw.tools.builtin.system import ScreenshotTool, ClipboardTool, LaunchAppTool
        from intelclaw.tools.builtin.web import WebScrapeTool
        
        builtin_tools = [
            TavilySearchTool(),
            FileReadTool(),
            FileWriteTool(),
            FileSearchTool(),
            ScreenshotTool(),
            ClipboardTool(),
            LaunchAppTool(),
            WebScrapeTool(),
        ]
        
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
