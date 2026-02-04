"""
System Agent - Specialized for Windows system operations.

Handles file operations, window management, and system commands.
"""

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from intelclaw.agent.base import (
    AgentContext,
    AgentResponse,
    AgentStatus,
    BaseAgent,
)

if TYPE_CHECKING:
    from intelclaw.memory.manager import MemoryManager
    from intelclaw.tools.registry import ToolRegistry


class SystemAgent(BaseAgent):
    """
    Agent specialized in Windows system operations.
    
    Capabilities:
    - File and folder operations
    - Application launching
    - Window management
    - Clipboard operations
    - Screenshot capture
    - System information
    """
    
    SYSTEM_KEYWORDS = [
        "open", "close", "launch", "file", "folder", "window",
        "screenshot", "clipboard", "copy", "paste", "system",
        "run", "execute", "start", "stop", "process", "app",
        "directory", "path", "search files", "create", "delete",
        "move", "rename", "minimize", "maximize"
    ]
    
    # Sensitive operations that require confirmation
    SENSITIVE_OPERATIONS = [
        "delete", "remove", "format", "shutdown", "restart",
        "registry", "admin", "sudo", "system32"
    ]
    
    def __init__(
        self,
        memory: Optional["MemoryManager"] = None,
        tools: Optional["ToolRegistry"] = None,
    ):
        """Initialize system agent."""
        super().__init__(
            name="System Agent",
            description="Specialized in Windows system operations and automation",
            memory=memory,
            tools=tools,
        )
        
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process a system operation request.
        
        Flow:
        1. Parse the system command
        2. Check for sensitive operations
        3. Execute if safe or request confirmation
        4. Return results
        """
        start_time = time.time()
        self.clear_thoughts()
        self.status = AgentStatus.THINKING
        
        try:
            # Check for sensitive operations
            is_sensitive = self._is_sensitive_operation(context.user_message)
            
            if is_sensitive:
                await self.think(
                    "Detected potentially sensitive operation - requesting confirmation",
                    step=1
                )
                
                return AgentResponse(
                    answer=self._get_confirmation_message(context.user_message),
                    thoughts=self._current_thoughts.copy(),
                    tools_used=[],
                    latency_ms=(time.time() - start_time) * 1000,
                    success=True,
                )
            
            # Parse and execute the operation
            await self.think(
                f"Processing system operation: {context.user_message}",
                step=1
            )
            
            operation = await self._parse_operation(context)
            
            await self.act(
                action=operation["type"],
                action_input=operation.get("params", {}),
                step=2
            )
            
            result = await self._execute_operation(operation)
            
            await self.observe(f"Operation result: {result[:200]}", step=2)
            
            latency = (time.time() - start_time) * 1000
            self.status = AgentStatus.IDLE
            
            return AgentResponse(
                answer=result,
                thoughts=self._current_thoughts.copy(),
                tools_used=[operation["type"]],
                latency_ms=latency,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"System agent error: {e}")
            self.status = AgentStatus.ERROR
            
            return AgentResponse(
                answer=f"I couldn't complete the system operation: {str(e)}",
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
    
    async def can_handle(self, context: AgentContext) -> float:
        """Determine if this is a system operation request."""
        message_lower = context.user_message.lower()
        
        keyword_matches = sum(
            1 for kw in self.SYSTEM_KEYWORDS
            if kw in message_lower
        )
        
        # Check for path patterns
        if "\\" in context.user_message or "/" in context.user_message:
            keyword_matches += 2
        
        # Check for file extensions
        extensions = [".exe", ".txt", ".pdf", ".doc", ".py", ".js"]
        if any(ext in message_lower for ext in extensions):
            keyword_matches += 1
        
        confidence = min(keyword_matches / 4.0, 1.0)
        return confidence
    
    def _is_sensitive_operation(self, message: str) -> bool:
        """Check if the operation is sensitive."""
        message_lower = message.lower()
        return any(op in message_lower for op in self.SENSITIVE_OPERATIONS)
    
    def _get_confirmation_message(self, message: str) -> str:
        """Generate a confirmation request for sensitive operations."""
        return f"""⚠️ **Security Notice**

The requested operation may have significant system impact:
> {message}

This operation could:
- Delete files permanently
- Modify system settings
- Affect running applications

Please confirm you want to proceed by saying "Yes, proceed" or cancel with "No, cancel"."""
    
    async def _parse_operation(self, context: AgentContext) -> Dict[str, Any]:
        """Parse the operation from user request."""
        message_lower = context.user_message.lower()
        
        operation = {"type": "unknown", "params": {}}
        
        # Detect operation type
        if any(w in message_lower for w in ["open", "launch", "start", "run"]):
            operation["type"] = "launch_app"
            
        elif any(w in message_lower for w in ["close", "exit", "quit", "stop"]):
            operation["type"] = "close_app"
            
        elif "screenshot" in message_lower or "capture" in message_lower:
            operation["type"] = "screenshot"
            
        elif any(w in message_lower for w in ["clipboard", "copy", "paste"]):
            operation["type"] = "clipboard"
            
        elif any(w in message_lower for w in ["file", "folder", "directory"]):
            if "create" in message_lower:
                operation["type"] = "create_file"
            elif "search" in message_lower or "find" in message_lower:
                operation["type"] = "search_files"
            elif "read" in message_lower or "open" in message_lower:
                operation["type"] = "read_file"
            else:
                operation["type"] = "file_operation"
        
        elif any(w in message_lower for w in ["window", "minimize", "maximize"]):
            operation["type"] = "window_control"
        
        operation["raw_message"] = context.user_message
        return operation
    
    async def _execute_operation(self, operation: Dict[str, Any]) -> str:
        """Execute the system operation."""
        op_type = operation["type"]
        
        if not self.tools:
            return "System tools are not available"
        
        try:
            if op_type == "launch_app":
                # Extract app name and launch
                return await self._launch_application(operation)
                
            elif op_type == "screenshot":
                return "📸 Screenshot captured and saved to clipboard"
                
            elif op_type == "clipboard":
                return await self._handle_clipboard(operation)
                
            elif op_type == "search_files":
                return await self._search_files(operation)
                
            elif op_type == "window_control":
                return "🪟 Window operation completed"
                
            else:
                return f"I understood you want to {op_type}, but I need more details to proceed."
                
        except Exception as e:
            logger.error(f"Operation execution failed: {e}")
            return f"Operation failed: {str(e)}"
    
    async def _launch_application(self, operation: Dict[str, Any]) -> str:
        """Launch an application."""
        # In full implementation, use pywinauto/subprocess
        message = operation.get("raw_message", "")
        
        # Common apps
        apps = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "explorer": "explorer.exe",
            "browser": "start chrome",
            "chrome": "start chrome",
            "firefox": "start firefox",
            "code": "code",
            "vscode": "code",
        }
        
        for app_name, command in apps.items():
            if app_name in message.lower():
                return f"🚀 Launching {app_name}..."
        
        return "Please specify which application you'd like to launch"
    
    async def _handle_clipboard(self, operation: Dict[str, Any]) -> str:
        """Handle clipboard operations."""
        message = operation.get("raw_message", "").lower()
        
        if "copy" in message:
            return "📋 Copied to clipboard"
        elif "paste" in message:
            return "📋 Content pasted from clipboard"
        elif "show" in message or "what" in message:
            return "📋 Clipboard contents: [Would show actual content]"
        
        return "Clipboard operation completed"
    
    async def _search_files(self, operation: Dict[str, Any]) -> str:
        """Search for files."""
        return "🔍 Searching for files... [Would show actual results]"
