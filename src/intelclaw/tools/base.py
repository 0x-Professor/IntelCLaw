"""
Base Tool - Abstract base class for all tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    """Tool categories."""
    SYSTEM = "system"
    SEARCH = "search"
    PRODUCTIVITY = "productivity"
    DEVELOPMENT = "development"
    COMMUNICATION = "communication"
    CODE = "code"  # Code execution tools
    CUSTOM = "custom"


class ToolPermission(str, Enum):
    """Tool permission levels."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    SENSITIVE = "sensitive"


@dataclass
class ToolResult:
    """Result from tool execution."""
    
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


class ToolDefinition(BaseModel):
    """Tool definition for registration."""
    
    name: str
    description: str
    category: ToolCategory = ToolCategory.CUSTOM
    permissions: List[ToolPermission] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)  # JSON Schema
    returns: str = "any"
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    rate_limit: Optional[int] = None  # Calls per minute
    requires_confirmation: bool = False


class BaseTool(ABC):
    """
    Abstract base class for IntelCLaw tools.
    
    All tools must implement:
    - definition: Tool metadata
    - execute: Core execution logic
    """
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Get tool definition."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with execution outcome
        """
        pass
    
    def validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """
        Validate input parameters.
        
        Args:
            params: Input parameters
            
        Returns:
            Error message if invalid, None if valid
        """
        # Basic validation - override for specific tools
        schema = self.definition.parameters
        required = schema.get("required", [])
        
        for param in required:
            if param not in params:
                return f"Missing required parameter: {param}"
        
        return None
    
    async def safe_execute(self, **kwargs) -> ToolResult:
        """Execute with validation and error handling."""
        import time
        
        start_time = time.time()
        
        # Validate
        error = self.validate_params(kwargs)
        if error:
            return ToolResult(
                success=False,
                error=error,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Execute
        try:
            result = await self.execute(**kwargs)
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
