"""
System Tools - Screenshots, clipboard, app launching.
"""

import asyncio
import subprocess
from typing import Any, Dict, Optional

from loguru import logger

from intelclaw.tools.base import BaseTool, ToolDefinition, ToolResult, ToolCategory, ToolPermission


class ScreenshotTool(BaseTool):
    """Capture screenshots."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="screenshot",
            description="Capture a screenshot of the screen or active window.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {
                    "active_window_only": {
                        "type": "boolean",
                        "description": "Capture only the active window",
                        "default": False
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Path to save the screenshot (optional)"
                    }
                },
                "required": []
            },
            returns="bytes"
        )
    
    async def execute(
        self,
        active_window_only: bool = False,
        save_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Capture screenshot."""
        try:
            from intelclaw.perception.screen_capture import ScreenCapture
            
            capture = ScreenCapture()
            await capture.initialize()
            
            if active_window_only:
                image_bytes = await capture.capture_active_window()
            else:
                image_bytes = await capture.capture(save_path=save_path)
            
            await capture.shutdown()
            
            if image_bytes:
                return ToolResult(
                    success=True,
                    data=image_bytes,
                    metadata={"size": len(image_bytes)}
                )
            else:
                return ToolResult(success=False, error="Screenshot capture failed")
                
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return ToolResult(success=False, error=str(e))


class ClipboardTool(BaseTool):
    """Read/write clipboard."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="clipboard",
            description="Read from or write to the system clipboard.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ, ToolPermission.WRITE],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write"],
                        "description": "Action to perform"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (for write action)"
                    }
                },
                "required": ["action"]
            },
            returns="string"
        )
    
    async def execute(
        self,
        action: str,
        content: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Handle clipboard operations."""
        try:
            import win32clipboard
            import win32con
            
            if action == "read":
                win32clipboard.OpenClipboard()
                try:
                    if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                        data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                        return ToolResult(success=True, data=str(data))
                    return ToolResult(success=True, data="")
                finally:
                    win32clipboard.CloseClipboard()
                    
            elif action == "write":
                if not content:
                    return ToolResult(success=False, error="Content required for write")
                
                win32clipboard.OpenClipboard()
                try:
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardText(content, win32con.CF_UNICODETEXT)
                    return ToolResult(success=True, data=True)
                finally:
                    win32clipboard.CloseClipboard()
            
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Clipboard operation failed: {e}")
            return ToolResult(success=False, error=str(e))


class LaunchAppTool(BaseTool):
    """Launch applications."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="launch_app",
            description="Launch an application or open a file with its default application.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Application name, path, or file to open"
                    },
                    "arguments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command line arguments"
                    }
                },
                "required": ["target"]
            },
            returns="boolean",
            requires_confirmation=True
        )
    
    async def execute(
        self,
        target: str,
        arguments: Optional[list] = None,
        **kwargs
    ) -> ToolResult:
        """Launch application."""
        try:
            import os
            
            # Use os.startfile for Windows
            if arguments:
                cmd = [target] + arguments
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            else:
                # Use startfile for simple launches
                await asyncio.to_thread(os.startfile, target)
            
            return ToolResult(
                success=True,
                data=True,
                metadata={"target": target}
            )
            
        except Exception as e:
            logger.error(f"Launch app failed: {e}")
            return ToolResult(success=False, error=str(e))
