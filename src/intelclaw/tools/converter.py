"""
Tool Schema Converter - normalize tool schemas across OpenAI, MCP, LangChain, and OpenAI Agent SDK.

This module provides conversion helpers so IntelCLaw can interoperate with:
- OpenAI tool/function schemas (Chat Completions API)
- OpenAI Agent SDK tool schemas (Assistants/Swarm API)
- MCP tool schemas (Model Context Protocol)
- LangChain tool definitions
- LangGraph tool wrappers
- Native IntelCLaw tool definitions

Supports bidirectional conversion between all formats for maximum interoperability.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, get_type_hints

from loguru import logger

from intelclaw.tools.base import ToolDefinition, BaseTool

# Type stubs for optional dependencies
try:
    from langchain_core.tools import BaseTool as LangChainBaseTool
    from langchain_core.tools import StructuredTool, tool as langchain_tool_decorator
except Exception:  # pragma: no cover
    LangChainBaseTool = None
    StructuredTool = None
    langchain_tool_decorator = None

try:
    from pydantic import BaseModel, Field, create_model
except Exception:  # pragma: no cover
    BaseModel = None
    Field = None
    create_model = None

# Type aliases for different tool formats
OpenAITool = Dict[str, Any]
OpenAIAgentSDKTool = Dict[str, Any]
MCPTool = Dict[str, Any]
LangChainToolDict = Dict[str, Any]
ToolExecutor = Callable[..., Any]


class ToolFormat(str, Enum):
    """Supported tool schema formats."""
    OPENAI = "openai"               # OpenAI Chat Completions API
    OPENAI_AGENT_SDK = "agent_sdk"  # OpenAI Agents SDK / Swarm
    MCP = "mcp"                     # Model Context Protocol
    LANGCHAIN = "langchain"         # LangChain BaseTool
    LANGGRAPH = "langgraph"         # LangGraph compatible
    INTELCLAW = "intelclaw"         # Native IntelCLaw format


@dataclass
class UnifiedTool:
    """
    Unified tool representation that can be converted to any format.
    
    This is the canonical internal representation that bridges all frameworks.
    """
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format
    executor: Optional[ToolExecutor] = None
    is_async: bool = False
    category: str = "general"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_openai(self) -> OpenAITool:
        """Convert to OpenAI Chat Completions tool format."""
        return _unified_to_openai(self)
    
    def to_openai_agent_sdk(self) -> OpenAIAgentSDKTool:
        """Convert to OpenAI Agent SDK tool format."""
        return _unified_to_openai_agent_sdk(self)
    
    def to_mcp(self) -> MCPTool:
        """Convert to MCP tool format."""
        return _unified_to_mcp(self)
    
    def to_langchain(self) -> Any:
        """Convert to LangChain StructuredTool."""
        return _unified_to_langchain(self)
    
    def to_intelclaw(self) -> ToolDefinition:
        """Convert to IntelClaw ToolDefinition."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ToolConverter:
    """
    Central tool conversion hub for multi-framework interoperability.
    
    Supports:
    - OpenAI Chat API tools
    - OpenAI Agent SDK tools (Swarm-style)
    - MCP (Model Context Protocol) tools
    - LangChain tools
    - LangGraph tools
    - IntelClaw native tools
    
    Usage:
        converter = ToolConverter()
        
        # Register tools from any framework
        converter.register_openai_tool(openai_tool_dict)
        converter.register_mcp_tool(mcp_tool_dict)
        converter.register_langchain_tool(langchain_tool)
        
        # Get tools in any format
        openai_tools = converter.get_tools(ToolFormat.OPENAI)
        mcp_tools = converter.get_tools(ToolFormat.MCP)
    """
    
    def __init__(self):
        self._tools: Dict[str, UnifiedTool] = {}
        self._executors: Dict[str, ToolExecutor] = {}
    
    def register(self, tool: Any, executor: Optional[ToolExecutor] = None) -> str:
        """
        Register a tool from any supported framework.
        
        Args:
            tool: Tool in any supported format
            executor: Optional callable to execute the tool
            
        Returns:
            The registered tool name
        """
        unified = to_unified_tool(tool, executor)
        if unified:
            self._tools[unified.name] = unified
            if unified.executor:
                self._executors[unified.name] = unified.executor
            logger.debug(f"Registered tool: {unified.name}")
            return unified.name
        return ""
    
    def register_openai_tool(self, tool: OpenAITool, executor: Optional[ToolExecutor] = None) -> str:
        """Register an OpenAI format tool."""
        return self.register(tool, executor)
    
    def register_mcp_tool(self, tool: MCPTool, executor: Optional[ToolExecutor] = None) -> str:
        """Register an MCP format tool."""
        return self.register(tool, executor)
    
    def register_langchain_tool(self, tool: Any) -> str:
        """Register a LangChain tool."""
        return self.register(tool)
    
    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Register a Python function as a tool.
        
        Automatically extracts parameter schema from type hints.
        """
        unified = _function_to_unified(func, name, description)
        if unified:
            self._tools[unified.name] = unified
            self._executors[unified.name] = unified.executor
            return unified.name
        return ""
    
    def get_tools(self, format: ToolFormat = ToolFormat.OPENAI) -> List[Any]:
        """
        Get all registered tools in the specified format.
        
        Args:
            format: Target tool format
            
        Returns:
            List of tools in the requested format
        """
        tools = []
        for unified in self._tools.values():
            try:
                if format == ToolFormat.OPENAI:
                    tools.append(unified.to_openai())
                elif format == ToolFormat.OPENAI_AGENT_SDK:
                    tools.append(unified.to_openai_agent_sdk())
                elif format == ToolFormat.MCP:
                    tools.append(unified.to_mcp())
                elif format == ToolFormat.LANGCHAIN:
                    lc_tool = unified.to_langchain()
                    if lc_tool:
                        tools.append(lc_tool)
                elif format == ToolFormat.INTELCLAW:
                    tools.append(unified.to_intelclaw())
            except Exception as e:
                logger.warning(f"Failed to convert tool {unified.name} to {format}: {e}")
        return tools
    
    def get_tool(self, name: str, format: ToolFormat = ToolFormat.OPENAI) -> Optional[Any]:
        """Get a specific tool by name in the specified format."""
        unified = self._tools.get(name)
        if not unified:
            return None
        
        if format == ToolFormat.OPENAI:
            return unified.to_openai()
        elif format == ToolFormat.MCP:
            return unified.to_mcp()
        elif format == ToolFormat.LANGCHAIN:
            return unified.to_langchain()
        elif format == ToolFormat.INTELCLAW:
            return unified.to_intelclaw()
        return None
    
    async def execute(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a registered tool by name.
        
        Args:
            name: Tool name
            args: Tool arguments
            
        Returns:
            Tool execution result
        """
        executor = self._executors.get(name)
        if not executor:
            raise ValueError(f"No executor registered for tool: {name}")
        
        # Handle async executors
        if asyncio.iscoroutinefunction(executor):
            return await executor(**args)
        else:
            return executor(**args)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def clear(self):
        """Clear all registered tools."""
        self._tools.clear()
        self._executors.clear()


# =============================================================================
# Unified Tool Conversion Functions
# =============================================================================

def to_unified_tool(tool: Any, executor: Optional[ToolExecutor] = None) -> Optional[UnifiedTool]:
    """
    Convert any tool format to UnifiedTool.
    
    Supports:
    - Dict (OpenAI, MCP, generic)
    - IntelClaw BaseTool / ToolDefinition
    - LangChain BaseTool / StructuredTool
    - Callable (function with type hints)
    """
    if tool is None:
        return None
    
    # Already unified
    if isinstance(tool, UnifiedTool):
        if executor and not tool.executor:
            tool.executor = executor
        return tool
    
    # Callable function
    if callable(tool) and not isinstance(tool, type):
        return _function_to_unified(tool)
    
    # Dict-based formats
    if isinstance(tool, dict):
        return _dict_to_unified(tool, executor)
    
    # IntelClaw types
    if isinstance(tool, BaseTool):
        return UnifiedTool(
            name=tool.definition.name,
            description=tool.definition.description or tool.definition.name,
            parameters=_ensure_object_schema(tool.definition.parameters or {}),
            executor=executor or (tool.execute if hasattr(tool, 'execute') else None),
            is_async=hasattr(tool, 'aexecute'),
        )
    
    if isinstance(tool, ToolDefinition):
        return UnifiedTool(
            name=tool.name,
            description=tool.description or tool.name,
            parameters=_ensure_object_schema(tool.parameters or {}),
            executor=executor,
        )
    
    # LangChain types
    if LangChainBaseTool and isinstance(tool, LangChainBaseTool):
        return _langchain_to_unified(tool)
    
    logger.debug(f"Unsupported tool type: {type(tool)}")
    return None


def _dict_to_unified(tool: Dict[str, Any], executor: Optional[ToolExecutor] = None) -> Optional[UnifiedTool]:
    """Convert a dict-based tool schema to UnifiedTool."""
    
    # OpenAI format: {"type": "function", "function": {...}}
    if tool.get("type") == "function" and "function" in tool:
        func = tool["function"]
        return UnifiedTool(
            name=func.get("name", "unknown"),
            description=func.get("description", ""),
            parameters=_ensure_object_schema(func.get("parameters", {})),
            executor=executor,
        )
    
    # MCP format: {"name": "...", "inputSchema": {...}}
    if "inputSchema" in tool and "name" in tool:
        return UnifiedTool(
            name=tool.get("name", "unknown"),
            description=tool.get("description", ""),
            parameters=_ensure_object_schema(tool.get("inputSchema", {})),
            executor=executor,
        )
    
    # OpenAI Agent SDK format: {"name": "...", "parameters": {...}, "type": "function"}
    if "name" in tool and "parameters" in tool:
        return UnifiedTool(
            name=tool.get("name", "unknown"),
            description=tool.get("description", ""),
            parameters=_ensure_object_schema(tool.get("parameters", {})),
            executor=executor,
        )
    
    # Generic dict with name
    if "name" in tool:
        return UnifiedTool(
            name=tool.get("name", "unknown"),
            description=tool.get("description", ""),
            parameters=_ensure_object_schema(tool.get("parameters", tool.get("schema", {}))),
            executor=executor,
        )
    
    return None


def _langchain_to_unified(tool: Any) -> Optional[UnifiedTool]:
    """Convert LangChain tool to UnifiedTool."""
    if not LangChainBaseTool or not isinstance(tool, LangChainBaseTool):
        return None
    
    schema = _ensure_object_schema(_pydantic_schema(getattr(tool, "args_schema", None)))
    
    # Get the executor
    executor = None
    if hasattr(tool, "_run"):
        executor = tool._run
    elif hasattr(tool, "invoke"):
        executor = tool.invoke
    
    return UnifiedTool(
        name=getattr(tool, "name", "unknown_tool"),
        description=getattr(tool, "description", ""),
        parameters=schema,
        executor=executor,
        is_async=hasattr(tool, "_arun"),
    )


def _function_to_unified(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Optional[UnifiedTool]:
    """Convert a Python function to UnifiedTool using type hints."""
    try:
        func_name = name or func.__name__
        func_doc = description or func.__doc__ or f"Execute {func_name}"
        
        # Get type hints
        hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
        sig = inspect.signature(func)
        
        # Build JSON Schema from type hints
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue
            
            param_type = hints.get(param_name, Any)
            param_schema = _python_type_to_json_schema(param_type)
            
            # Add description from docstring if available
            properties[param_name] = param_schema
            
            # Required if no default
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        
        return UnifiedTool(
            name=func_name,
            description=func_doc.strip(),
            parameters=parameters,
            executor=func,
            is_async=asyncio.iscoroutinefunction(func),
        )
    except Exception as e:
        logger.debug(f"Failed to convert function {func}: {e}")
        return None


def _python_type_to_json_schema(python_type: Type) -> Dict[str, Any]:
    """Convert Python type hint to JSON Schema."""
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        type(None): {"type": "null"},
    }
    
    # Handle basic types
    if python_type in type_mapping:
        return type_mapping[python_type]
    
    # Handle Optional, List, Dict from typing module
    origin = getattr(python_type, '__origin__', None)
    args = getattr(python_type, '__args__', ())
    
    if origin is list:
        item_type = args[0] if args else Any
        return {"type": "array", "items": _python_type_to_json_schema(item_type)}
    
    if origin is dict:
        return {"type": "object"}
    
    if origin is Union:
        # Handle Optional (Union[X, None])
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])
        return {"anyOf": [_python_type_to_json_schema(a) for a in non_none]}
    
    # Default to string
    return {"type": "string"}


# =============================================================================
# Format-specific conversion functions
# =============================================================================

def _unified_to_openai(unified: UnifiedTool) -> OpenAITool:
    """Convert UnifiedTool to OpenAI Chat Completions format."""
    return {
        "type": "function",
        "function": {
            "name": unified.name,
            "description": unified.description,
            "parameters": unified.parameters,
        },
    }


def _unified_to_openai_agent_sdk(unified: UnifiedTool) -> OpenAIAgentSDKTool:
    """
    Convert UnifiedTool to OpenAI Agent SDK format.
    
    This format is used by the OpenAI Agents SDK (Swarm-style agents).
    """
    return {
        "type": "function",
        "name": unified.name,
        "description": unified.description,
        "parameters": unified.parameters,
    }


def _unified_to_mcp(unified: UnifiedTool) -> MCPTool:
    """Convert UnifiedTool to MCP format."""
    return {
        "name": unified.name,
        "description": unified.description,
        "inputSchema": unified.parameters,
    }


def _unified_to_langchain(unified: UnifiedTool) -> Any:
    """Convert UnifiedTool to LangChain StructuredTool."""
    if StructuredTool is None or BaseModel is None:
        logger.debug("LangChain not available for conversion")
        return None
    
    try:
        # Create Pydantic model for args
        fields = {}
        for prop_name, prop_schema in unified.parameters.get("properties", {}).items():
            field_type = _json_schema_to_python_type(prop_schema)
            default = ... if prop_name in unified.parameters.get("required", []) else None
            fields[prop_name] = (field_type, default)
        
        if fields:
            ArgsModel = create_model(f"{unified.name}Args", **fields)
        else:
            ArgsModel = None
        
        # Create the structured tool
        def tool_func(**kwargs):
            if unified.executor:
                return unified.executor(**kwargs)
            return f"Executed {unified.name}"
        
        async def async_tool_func(**kwargs):
            if unified.executor:
                if asyncio.iscoroutinefunction(unified.executor):
                    return await unified.executor(**kwargs)
                return unified.executor(**kwargs)
            return f"Executed {unified.name}"
        
        return StructuredTool(
            name=unified.name,
            description=unified.description,
            func=tool_func,
            coroutine=async_tool_func if unified.is_async else None,
            args_schema=ArgsModel,
        )
    except Exception as e:
        logger.debug(f"Failed to create LangChain tool: {e}")
        return None


def _json_schema_to_python_type(schema: Dict[str, Any]) -> Type:
    """Convert JSON Schema type to Python type."""
    json_type = schema.get("type", "string")
    
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    
    return type_mapping.get(json_type, str)


# Import Union for type hints
from typing import Union


# =============================================================================
# Legacy compatibility functions (original API preserved)
# =============================================================================

def _pydantic_schema(model: Any) -> Dict[str, Any]:
    """Extract JSON schema from a Pydantic model class."""
    if model is None:
        return {}
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    if hasattr(model, "schema"):
        return model.schema()
    return {}


def _ensure_object_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure schema is a JSON Schema object with properties."""
    if not schema:
        return {"type": "object", "properties": {}, "required": []}
    schema = dict(schema)
    if schema.get("type") != "object":
        schema = {"type": "object", "properties": {"input": schema}, "required": ["input"]}
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    return schema


def _tool_def_to_openai(defn: ToolDefinition) -> OpenAITool:
    schema = _ensure_object_schema(defn.parameters or {})
    return {
        "type": "function",
        "function": {
            "name": defn.name,
            "description": defn.description or defn.name,
            "parameters": schema,
        },
    }


def _langchain_tool_to_openai(tool: Any) -> OpenAITool:
    schema = _ensure_object_schema(_pydantic_schema(getattr(tool, "args_schema", None)))
    return {
        "type": "function",
        "function": {
            "name": getattr(tool, "name", "unknown_tool"),
            "description": getattr(tool, "description", "") or getattr(tool, "name", ""),
            "parameters": schema,
        },
    }


def _mcp_tool_to_openai(tool: MCPTool) -> OpenAITool:
    schema = _ensure_object_schema(tool.get("inputSchema", tool.get("schema", {})))
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", "unknown_tool"),
            "description": tool.get("description", "") or tool.get("name", ""),
            "parameters": schema,
        },
    }


def to_openai_tool(tool: Any) -> Optional[OpenAITool]:
    """Convert an arbitrary tool (IntelClaw/LangChain/MCP/OpenAI) to OpenAI tool schema."""
    if tool is None:
        return None

    # Already OpenAI tool schema
    if isinstance(tool, dict):
        if tool.get("type") == "function" and "function" in tool:
            return tool
        if "inputSchema" in tool or "schema" in tool:
            return _mcp_tool_to_openai(tool)
        if "name" in tool and "parameters" in tool:
            return {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": _ensure_object_schema(tool.get("parameters", {})),
                },
            }

    # IntelClaw tool or definition
    if isinstance(tool, BaseTool):
        return _tool_def_to_openai(tool.definition)
    if isinstance(tool, ToolDefinition):
        return _tool_def_to_openai(tool)

    # LangChain tool
    if LangChainBaseTool and isinstance(tool, LangChainBaseTool):
        return _langchain_tool_to_openai(tool)
    
    # Try UnifiedTool
    unified = to_unified_tool(tool)
    if unified:
        return unified.to_openai()

    logger.debug(f"Unsupported tool type for OpenAI conversion: {type(tool)}")
    return None


def to_mcp_tool(tool: Any) -> Optional[MCPTool]:
    """Convert tool to MCP-compatible schema (name, description, inputSchema)."""
    if tool is None:
        return None

    if isinstance(tool, dict):
        if "inputSchema" in tool and "name" in tool:
            return tool
        if tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            return {
                "name": func.get("name"),
                "description": func.get("description", ""),
                "inputSchema": _ensure_object_schema(func.get("parameters", {})),
            }

    if isinstance(tool, BaseTool):
        defn = tool.definition
        return {
            "name": defn.name,
            "description": defn.description or defn.name,
            "inputSchema": _ensure_object_schema(defn.parameters or {}),
        }
    if isinstance(tool, ToolDefinition):
        return {
            "name": tool.name,
            "description": tool.description or tool.name,
            "inputSchema": _ensure_object_schema(tool.parameters or {}),
        }

    if LangChainBaseTool and isinstance(tool, LangChainBaseTool):
        schema = _ensure_object_schema(_pydantic_schema(getattr(tool, "args_schema", None)))
        return {
            "name": getattr(tool, "name", "unknown_tool"),
            "description": getattr(tool, "description", ""),
            "inputSchema": schema,
        }
    
    # Try UnifiedTool
    unified = to_unified_tool(tool)
    if unified:
        return unified.to_mcp()

    logger.debug(f"Unsupported tool type for MCP conversion: {type(tool)}")
    return None


def to_openai_agent_sdk_tool(tool: Any) -> Optional[OpenAIAgentSDKTool]:
    """Convert tool to OpenAI Agent SDK format."""
    unified = to_unified_tool(tool)
    if unified:
        return unified.to_openai_agent_sdk()
    return None


def to_intelclaw_definition(tool: Any) -> Optional[ToolDefinition]:
    """Convert a tool schema into an IntelClaw ToolDefinition (best-effort)."""
    if tool is None:
        return None

    if isinstance(tool, ToolDefinition):
        return tool

    if isinstance(tool, BaseTool):
        return tool.definition

    if isinstance(tool, dict):
        if tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            return ToolDefinition(
                name=func.get("name", "unknown_tool"),
                description=func.get("description", ""),
                parameters=_ensure_object_schema(func.get("parameters", {})),
            )
        if "inputSchema" in tool and "name" in tool:
            return ToolDefinition(
                name=tool.get("name", "unknown_tool"),
                description=tool.get("description", ""),
                parameters=_ensure_object_schema(tool.get("inputSchema", {})),
            )

    if LangChainBaseTool and isinstance(tool, LangChainBaseTool):
        schema = _ensure_object_schema(_pydantic_schema(getattr(tool, "args_schema", None)))
        return ToolDefinition(
            name=getattr(tool, "name", "unknown_tool"),
            description=getattr(tool, "description", ""),
            parameters=schema,
        )
    
    # Try UnifiedTool
    unified = to_unified_tool(tool)
    if unified:
        return unified.to_intelclaw()

    return None


def convert_tools_to_openai(tools: List[Any]) -> List[OpenAITool]:
    """Convert a list of tools to OpenAI tool schemas."""
    converted: List[OpenAITool] = []
    for tool in tools:
        try:
            schema = to_openai_tool(tool)
            if schema:
                converted.append(schema)
        except Exception as e:
            logger.warning(f"Tool conversion failed: {e}")
    return converted


def convert_tools_to_mcp(tools: List[Any]) -> List[MCPTool]:
    """Convert a list of tools to MCP tool schemas."""
    converted: List[MCPTool] = []
    for tool in tools:
        try:
            schema = to_mcp_tool(tool)
            if schema:
                converted.append(schema)
        except Exception as e:
            logger.warning(f"MCP tool conversion failed: {e}")
    return converted


def convert_tools_to_agent_sdk(tools: List[Any]) -> List[OpenAIAgentSDKTool]:
    """Convert a list of tools to OpenAI Agent SDK format."""
    converted: List[OpenAIAgentSDKTool] = []
    for tool in tools:
        try:
            schema = to_openai_agent_sdk_tool(tool)
            if schema:
                converted.append(schema)
        except Exception as e:
            logger.warning(f"Agent SDK tool conversion failed: {e}")
    return converted


def convert_tools_to_langchain(tools: List[Any]) -> List[Any]:
    """Convert a list of tools to LangChain StructuredTool format."""
    if StructuredTool is None:
        logger.warning("LangChain not available")
        return []
    
    converted = []
    for tool in tools:
        try:
            unified = to_unified_tool(tool)
            if unified:
                lc_tool = unified.to_langchain()
                if lc_tool:
                    converted.append(lc_tool)
        except Exception as e:
            logger.warning(f"LangChain tool conversion failed: {e}")
    return converted


# =============================================================================
# Tool decorator for easy registration
# =============================================================================

def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "general"
):
    """
    Decorator to convert a function into a registrable tool.
    
    Usage:
        @tool(name="my_tool", description="Does something useful")
        def my_function(arg1: str, arg2: int = 10) -> str:
            return f"Result: {arg1}, {arg2}"
    """
    def decorator(func: Callable) -> UnifiedTool:
        unified = _function_to_unified(func, name, description)
        if unified:
            unified.category = category
        return unified
    return decorator


# =============================================================================
# Global converter instance for convenience
# =============================================================================

_global_converter: Optional[ToolConverter] = None


def get_global_converter() -> ToolConverter:
    """Get or create the global ToolConverter instance."""
    global _global_converter
    if _global_converter is None:
        _global_converter = ToolConverter()
    return _global_converter
