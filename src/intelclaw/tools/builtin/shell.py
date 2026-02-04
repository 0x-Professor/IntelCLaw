"""
Shell and Code Execution Tools - Run shell commands and execute code.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from intelclaw.tools.base import BaseTool, ToolDefinition, ToolResult, ToolCategory, ToolPermission


class ShellCommandTool(BaseTool):
    """Execute shell commands safely."""
    
    # Commands that are blocked for security
    BLOCKED_COMMANDS = [
        "rm -rf /", "format", "del /f /s /q",
        "shutdown", "reboot", "halt",
        ":(){:|:&};:",  # Fork bomb
    ]
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="shell_command",
            description="Execute a shell command and return the output. Use for system tasks, file operations, git commands, etc.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for the command (optional)",
                        "default": None
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Command timeout in seconds (default: 60)",
                        "default": 60
                    },
                    "capture_stderr": {
                        "type": "boolean",
                        "description": "Include stderr in output (default: true)",
                        "default": True
                    }
                },
                "required": ["command"]
            },
            returns="dict with stdout, stderr, return_code",
            requires_confirmation=True,
            rate_limit=30
        )
    
    def _is_command_safe(self, command: str) -> bool:
        """Check if command is safe to execute."""
        command_lower = command.lower().strip()
        
        for blocked in self.BLOCKED_COMMANDS:
            if blocked in command_lower:
                return False
        
        return True
    
    async def execute(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: int = 60,
        capture_stderr: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute shell command."""
        # Security check
        if not self._is_command_safe(command):
            return ToolResult(
                success=False,
                error="Command blocked for security reasons"
            )
        
        try:
            # Resolve working directory
            cwd = None
            if working_dir:
                cwd = Path(working_dir).expanduser().resolve()
                if not cwd.exists():
                    return ToolResult(
                        success=False,
                        error=f"Working directory not found: {working_dir}"
                    )
            
            # Determine shell based on OS
            if sys.platform == "win32":
                shell = True
                executable = None
            else:
                shell = True
                executable = "/bin/bash"
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE if capture_stderr else None,
                cwd=cwd,
                shell=shell,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return ToolResult(
                    success=False,
                    error=f"Command timed out after {timeout} seconds"
                )
            
            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""
            
            return ToolResult(
                success=process.returncode == 0,
                data={
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "return_code": process.returncode,
                    "command": command,
                },
                metadata={
                    "working_dir": str(cwd) if cwd else os.getcwd(),
                    "timeout": timeout
                }
            )
            
        except Exception as e:
            logger.error(f"Shell command failed: {e}")
            return ToolResult(success=False, error=str(e))


class CodeExecutionTool(BaseTool):
    """Execute Python code safely."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="execute_code",
            description="Execute Python code and return the output. Useful for calculations, data processing, and testing code snippets.",
            category=ToolCategory.CODE,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds (default: 30)",
                        "default": 30
                    }
                },
                "required": ["code"]
            },
            returns="dict with output, error, success",
            requires_confirmation=True,
            rate_limit=20
        )
    
    async def execute(
        self,
        code: str,
        timeout: int = 30,
        **kwargs
    ) -> ToolResult:
        """Execute Python code."""
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8"
            ) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute the code in a subprocess for isolation
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    return ToolResult(
                        success=False,
                        error=f"Code execution timed out after {timeout} seconds"
                    )
                
                stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
                stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""
                
                return ToolResult(
                    success=process.returncode == 0,
                    data={
                        "output": stdout_str,
                        "error": stderr_str,
                        "return_code": process.returncode,
                    },
                    metadata={"timeout": timeout}
                )
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return ToolResult(success=False, error=str(e))


class PipInstallTool(BaseTool):
    """Install Python packages via pip."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="pip_install",
            description="Install Python packages using pip. Useful for installing required dependencies.",
            category=ToolCategory.CODE,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of packages to install"
                    },
                    "upgrade": {
                        "type": "boolean",
                        "description": "Upgrade packages if already installed",
                        "default": False
                    }
                },
                "required": ["packages"]
            },
            returns="dict with installed packages and status",
            requires_confirmation=True,
            rate_limit=10
        )
    
    async def execute(
        self,
        packages: List[str],
        upgrade: bool = False,
        **kwargs
    ) -> ToolResult:
        """Install packages via pip."""
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            
            if upgrade:
                cmd.append("--upgrade")
            
            cmd.extend(packages)
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""
            
            return ToolResult(
                success=process.returncode == 0,
                data={
                    "packages": packages,
                    "output": stdout_str,
                    "error": stderr_str,
                    "upgraded": upgrade
                },
                metadata={"return_code": process.returncode}
            )
            
        except Exception as e:
            logger.error(f"Pip install failed: {e}")
            return ToolResult(success=False, error=str(e))
