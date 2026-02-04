"""
File Operations Tools - Read, write, and search files.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from intelclaw.tools.base import BaseTool, ToolDefinition, ToolResult, ToolCategory, ToolPermission


class FileReadTool(BaseTool):
    """Read file contents."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_read",
            description="Read the contents of a file. Supports text files.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8)",
                        "default": "utf-8"
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum lines to read (default: all)",
                        "default": None
                    }
                },
                "required": ["path"]
            },
            returns="string"
        )
    
    async def execute(
        self,
        path: str,
        encoding: str = "utf-8",
        max_lines: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        """Read file contents."""
        try:
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                return ToolResult(success=False, error=f"File not found: {path}")
            
            if not file_path.is_file():
                return ToolResult(success=False, error=f"Not a file: {path}")
            
            # Check file size
            size = file_path.stat().st_size
            if size > 10 * 1024 * 1024:  # 10MB limit
                return ToolResult(success=False, error="File too large (>10MB)")
            
            content = await asyncio.to_thread(
                file_path.read_text,
                encoding=encoding
            )
            
            if max_lines:
                lines = content.split("\n")[:max_lines]
                content = "\n".join(lines)
            
            return ToolResult(
                success=True,
                data=content,
                metadata={"path": str(file_path), "size": size}
            )
            
        except Exception as e:
            logger.error(f"File read failed: {e}")
            return ToolResult(success=False, error=str(e))


class FileWriteTool(BaseTool):
    """Write content to a file."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_write",
            description="Write content to a file. Creates parent directories if needed.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.WRITE],
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["write", "append"],
                        "description": "Write mode (default: write)",
                        "default": "write"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8)",
                        "default": "utf-8"
                    }
                },
                "required": ["path", "content"]
            },
            returns="boolean",
            requires_confirmation=True
        )
    
    async def execute(
        self,
        path: str,
        content: str,
        mode: str = "write",
        encoding: str = "utf-8",
        **kwargs
    ) -> ToolResult:
        """Write to file."""
        try:
            file_path = Path(path).expanduser().resolve()
            
            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            write_mode = "w" if mode == "write" else "a"
            
            await asyncio.to_thread(
                file_path.write_text if mode == "write" else 
                lambda c, e: file_path.open("a", encoding=e).write(c),
                content,
                encoding
            )
            
            return ToolResult(
                success=True,
                data=True,
                metadata={"path": str(file_path), "mode": mode}
            )
            
        except Exception as e:
            logger.error(f"File write failed: {e}")
            return ToolResult(success=False, error=str(e))


class FileSearchTool(BaseTool):
    """Search for files by pattern."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_search",
            description="Search for files matching a pattern in a directory.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '*.py', '**/*.txt')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 100)",
                        "default": 100
                    }
                },
                "required": ["directory", "pattern"]
            },
            returns="list[string]"
        )
    
    async def execute(
        self,
        directory: str,
        pattern: str,
        max_results: int = 100,
        **kwargs
    ) -> ToolResult:
        """Search for files."""
        try:
            dir_path = Path(directory).expanduser().resolve()
            
            if not dir_path.exists():
                return ToolResult(success=False, error=f"Directory not found: {directory}")
            
            if not dir_path.is_dir():
                return ToolResult(success=False, error=f"Not a directory: {directory}")
            
            # Search
            matches = list(dir_path.glob(pattern))[:max_results]
            
            results = [str(p) for p in matches]
            
            return ToolResult(
                success=True,
                data=results,
                metadata={"directory": str(dir_path), "pattern": pattern, "count": len(results)}
            )
            
        except Exception as e:
            logger.error(f"File search failed: {e}")
            return ToolResult(success=False, error=str(e))
