"""
File Operations Tools - Read, write, list, and search files.
"""

import asyncio
import os
import shutil
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
            description="Read the contents of a file. Supports text files. Use this to read code, config files, logs, etc.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to read. Use forward slashes or escaped backslashes."
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8)",
                        "default": "utf-8"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Start reading from this line (1-indexed, optional)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Stop reading at this line (1-indexed, optional)"
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
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        """Read file contents."""
        try:
            # Handle various path formats
            path = path.strip().strip('"').strip("'")
            file_path = Path(path).expanduser().resolve()
            
            if not file_path.exists():
                # Try relative to current directory
                alt_path = Path.cwd() / path
                if alt_path.exists():
                    file_path = alt_path
                else:
                    return ToolResult(
                        success=False, 
                        error=f"File not found: {path}. Tried: {file_path} and {alt_path}"
                    )
            
            if not file_path.is_file():
                return ToolResult(success=False, error=f"Not a file: {path}")
            
            # Check file size
            size = file_path.stat().st_size
            if size > 10 * 1024 * 1024:  # 10MB limit
                return ToolResult(
                    success=False, 
                    error=f"File too large ({size / 1024 / 1024:.1f}MB). Max 10MB."
                )
            
            # Try multiple encodings
            content = None
            tried_encodings = [encoding, "utf-8", "utf-8-sig", "latin-1", "cp1252"]
            for enc in tried_encodings:
                try:
                    content = await asyncio.to_thread(file_path.read_text, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                # Fall back to binary read
                content = await asyncio.to_thread(file_path.read_bytes)
                content = content.decode("utf-8", errors="replace")
            
            # Handle line ranges
            if start_line or end_line:
                lines = content.split("\n")
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else len(lines)
                content = "\n".join(lines[start_idx:end_idx])
            
            return ToolResult(
                success=True,
                data=content,
                metadata={
                    "path": str(file_path), 
                    "size": size,
                    "lines": content.count("\n") + 1
                }
            )
            
        except Exception as e:
            logger.error(f"File read failed: {e}")
            return ToolResult(success=False, error=f"Failed to read file: {str(e)}")


class FileWriteTool(BaseTool):
    """Write content to a file."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_write",
            description="Write or append content to a file. Creates parent directories if needed. Use this to create new files or modify existing ones.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.WRITE],
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file. Parent directories will be created automatically."
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["write", "append"],
                        "description": "Write mode: 'write' overwrites file, 'append' adds to end (default: write)",
                        "default": "write"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8)",
                        "default": "utf-8"
                    },
                    "create_backup": {
                        "type": "boolean",
                        "description": "Create backup of existing file before overwriting",
                        "default": False
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
        create_backup: bool = False,
        **kwargs
    ) -> ToolResult:
        """Write to file with proper handling."""
        try:
            # Handle various path formats
            path = path.strip().strip('"').strip("'")
            file_path = Path(path).expanduser().resolve()
            
            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if requested and file exists
            if create_backup and file_path.exists():
                backup_path = file_path.with_suffix(file_path.suffix + ".bak")
                await asyncio.to_thread(shutil.copy2, file_path, backup_path)
                logger.debug(f"Created backup: {backup_path}")
            
            # Write the file properly
            if mode == "write":
                await asyncio.to_thread(
                    lambda: file_path.write_text(content, encoding=encoding)
                )
            else:  # append
                def append_content():
                    with file_path.open("a", encoding=encoding) as f:
                        f.write(content)
                await asyncio.to_thread(append_content)
            
            # Get file info after write
            size = file_path.stat().st_size
            
            return ToolResult(
                success=True,
                data=True,
                metadata={
                    "path": str(file_path), 
                    "mode": mode,
                    "size": size,
                    "lines": content.count("\n") + 1
                }
            )
            
        except PermissionError:
            return ToolResult(
                success=False, 
                error=f"Permission denied: Cannot write to {path}"
            )
        except Exception as e:
            logger.error(f"File write failed: {e}")
            return ToolResult(success=False, error=f"Failed to write file: {str(e)}")


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
