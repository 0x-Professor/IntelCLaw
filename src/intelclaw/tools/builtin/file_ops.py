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


class FileDeleteTool(BaseTool):
    """Delete a file or directory."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_delete",
            description="Delete a file or directory. Can delete files or entire directories (with force option). Use with caution!",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.WRITE],
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory to delete"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force delete directories with contents (default: False)",
                        "default": False
                    }
                },
                "required": ["path"]
            },
            returns="boolean",
            requires_confirmation=True
        )
    
    async def execute(
        self,
        path: str,
        force: bool = False,
        **kwargs
    ) -> ToolResult:
        """Delete file or directory."""
        try:
            path = path.strip().strip('"').strip("'")
            target_path = Path(path).expanduser().resolve()
            
            if not target_path.exists():
                # Try relative path
                alt_path = Path.cwd() / path
                if alt_path.exists():
                    target_path = alt_path
                else:
                    return ToolResult(
                        success=False, 
                        error=f"Path not found: {path}"
                    )
            
            if target_path.is_file():
                await asyncio.to_thread(target_path.unlink)
                return ToolResult(
                    success=True,
                    data=True,
                    metadata={"path": str(target_path), "type": "file", "deleted": True}
                )
            elif target_path.is_dir():
                if force:
                    await asyncio.to_thread(shutil.rmtree, target_path)
                else:
                    await asyncio.to_thread(target_path.rmdir)  # Only works if empty
                return ToolResult(
                    success=True,
                    data=True,
                    metadata={"path": str(target_path), "type": "directory", "deleted": True}
                )
            else:
                return ToolResult(success=False, error=f"Unknown path type: {path}")
                
        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: Cannot delete {path}")
        except OSError as e:
            if "not empty" in str(e).lower():
                return ToolResult(success=False, error=f"Directory not empty. Use force=True to delete: {path}")
            return ToolResult(success=False, error=f"Failed to delete: {str(e)}")
        except Exception as e:
            logger.error(f"File delete failed: {e}")
            return ToolResult(success=False, error=f"Failed to delete: {str(e)}")


class FileCopyTool(BaseTool):
    """Copy a file or directory."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_copy",
            description="Copy a file or directory to a new location. Creates destination directories automatically.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ, ToolPermission.WRITE],
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source file or directory path"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination path"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Overwrite if destination exists (default: False)",
                        "default": False
                    }
                },
                "required": ["source", "destination"]
            },
            returns="boolean"
        )
    
    async def execute(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
        **kwargs
    ) -> ToolResult:
        """Copy file or directory."""
        try:
            source = source.strip().strip('"').strip("'")
            destination = destination.strip().strip('"').strip("'")
            src_path = Path(source).expanduser().resolve()
            dst_path = Path(destination).expanduser().resolve()
            
            if not src_path.exists():
                return ToolResult(success=False, error=f"Source not found: {source}")
            
            if dst_path.exists() and not overwrite:
                return ToolResult(success=False, error=f"Destination exists: {destination}. Use overwrite=True")
            
            # Create parent directories
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            if src_path.is_file():
                await asyncio.to_thread(shutil.copy2, src_path, dst_path)
            else:
                if dst_path.exists() and overwrite:
                    await asyncio.to_thread(shutil.rmtree, dst_path)
                await asyncio.to_thread(shutil.copytree, src_path, dst_path)
            
            return ToolResult(
                success=True,
                data=True,
                metadata={
                    "source": str(src_path),
                    "destination": str(dst_path),
                    "type": "file" if src_path.is_file() else "directory"
                }
            )
            
        except Exception as e:
            logger.error(f"File copy failed: {e}")
            return ToolResult(success=False, error=f"Failed to copy: {str(e)}")


class FileMoveTool(BaseTool):
    """Move or rename a file or directory."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_move",
            description="Move or rename a file or directory. Use this to rename files or move them to a new location.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ, ToolPermission.WRITE],
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source file or directory path"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination path (new name or location)"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Overwrite if destination exists (default: False)",
                        "default": False
                    }
                },
                "required": ["source", "destination"]
            },
            returns="boolean"
        )
    
    async def execute(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
        **kwargs
    ) -> ToolResult:
        """Move or rename file/directory."""
        try:
            source = source.strip().strip('"').strip("'")
            destination = destination.strip().strip('"').strip("'")
            src_path = Path(source).expanduser().resolve()
            dst_path = Path(destination).expanduser().resolve()
            
            if not src_path.exists():
                return ToolResult(success=False, error=f"Source not found: {source}")
            
            if dst_path.exists():
                if not overwrite:
                    return ToolResult(success=False, error=f"Destination exists: {destination}. Use overwrite=True")
                if dst_path.is_dir():
                    await asyncio.to_thread(shutil.rmtree, dst_path)
                else:
                    await asyncio.to_thread(dst_path.unlink)
            
            # Create parent directories
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            await asyncio.to_thread(shutil.move, str(src_path), str(dst_path))
            
            return ToolResult(
                success=True,
                data=True,
                metadata={
                    "source": str(src_path),
                    "destination": str(dst_path),
                    "type": "file" if dst_path.is_file() else "directory"
                }
            )
            
        except Exception as e:
            logger.error(f"File move failed: {e}")
            return ToolResult(success=False, error=f"Failed to move: {str(e)}")


class FileSearchTool(BaseTool):
    """Search for files by pattern."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_search",
            description="Search for files matching a glob pattern in a directory. Use '**/*.py' for recursive search.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in. Use '.' for current directory."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '*.py', '**/*.txt', '*.{js,ts}')"
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
            directory = directory.strip().strip('"').strip("'")
            dir_path = Path(directory).expanduser().resolve()
            
            if not dir_path.exists():
                # Try current directory
                if directory == ".":
                    dir_path = Path.cwd()
                else:
                    return ToolResult(
                        success=False, 
                        error=f"Directory not found: {directory}"
                    )
            
            if not dir_path.is_dir():
                return ToolResult(success=False, error=f"Not a directory: {directory}")
            
            # Search
            matches = list(dir_path.glob(pattern))[:max_results]
            
            results = [str(p) for p in matches]
            
            return ToolResult(
                success=True,
                data=results,
                metadata={
                    "directory": str(dir_path), 
                    "pattern": pattern, 
                    "count": len(results),
                    "truncated": len(matches) >= max_results
                }
            )
            
        except Exception as e:
            logger.error(f"File search failed: {e}")
            return ToolResult(success=False, error=f"Search failed: {str(e)}")


class DirectoryListTool(BaseTool):
    """List directory contents."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_directory",
            description="List files and folders in a directory. Shows file sizes and types. Use this to explore directory structure.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Use '.' for current directory, '..' for parent."
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Show hidden files (starting with .)",
                        "default": False
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively (shows tree structure)",
                        "default": False
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum recursion depth (default: 3)",
                        "default": 3
                    }
                },
                "required": ["path"]
            },
            returns="list[dict]"
        )
    
    async def execute(
        self,
        path: str,
        show_hidden: bool = False,
        recursive: bool = False,
        max_depth: int = 3,
        **kwargs
    ) -> ToolResult:
        """List directory contents."""
        try:
            path = path.strip().strip('"').strip("'")
            dir_path = Path(path).expanduser().resolve()
            
            if not dir_path.exists():
                return ToolResult(
                    success=False, 
                    error=f"Path not found: {path}"
                )
            
            if not dir_path.is_dir():
                return ToolResult(
                    success=False, 
                    error=f"Not a directory: {path}"
                )
            
            def format_size(size: int) -> str:
                """Format file size in human-readable form."""
                for unit in ["B", "KB", "MB", "GB"]:
                    if size < 1024:
                        return f"{size:.1f}{unit}"
                    size /= 1024
                return f"{size:.1f}TB"
            
            def list_dir(dir_p: Path, depth: int = 0) -> List[Dict[str, Any]]:
                """List directory recursively."""
                items = []
                try:
                    for item in sorted(dir_p.iterdir()):
                        # Skip hidden files unless requested
                        if not show_hidden and item.name.startswith("."):
                            continue
                        
                        is_dir = item.is_dir()
                        info = {
                            "name": item.name,
                            "path": str(item),
                            "type": "directory" if is_dir else "file",
                            "size": format_size(item.stat().st_size) if not is_dir else None,
                        }
                        
                        if is_dir:
                            info["children_count"] = len(list(item.iterdir())) if item.is_dir() else 0
                        else:
                            info["extension"] = item.suffix.lower()
                        
                        items.append(info)
                        
                        # Recurse if requested
                        if recursive and is_dir and depth < max_depth:
                            children = list_dir(item, depth + 1)
                            if children:
                                info["children"] = children
                                
                except PermissionError:
                    pass
                
                return items
            
            items = await asyncio.to_thread(list_dir, dir_path)
            
            # Create summary
            files = [i for i in items if i["type"] == "file"]
            dirs = [i for i in items if i["type"] == "directory"]
            
            return ToolResult(
                success=True,
                data=items,
                metadata={
                    "path": str(dir_path),
                    "total_items": len(items),
                    "files": len(files),
                    "directories": len(dirs)
                }
            )
            
        except PermissionError:
            return ToolResult(
                success=False, 
                error=f"Permission denied: Cannot access {path}"
            )
        except Exception as e:
            logger.error(f"Directory list failed: {e}")
            return ToolResult(success=False, error=f"Failed to list directory: {str(e)}")


class GetCurrentDirectoryTool(BaseTool):
    """Get the current working directory."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_cwd",
            description="Get the current working directory. Use this to find out where you are in the file system.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            returns="string"
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Get current working directory."""
        try:
            cwd = Path.cwd()
            
            # Get additional info
            parent = cwd.parent
            home = Path.home()
            
            return ToolResult(
                success=True,
                data=str(cwd),
                metadata={
                    "path": str(cwd),
                    "parent": str(parent),
                    "home": str(home),
                    "drive": cwd.drive if cwd.drive else "/"
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
