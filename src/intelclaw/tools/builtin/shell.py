"""
Shell and Code Execution Tools - Run shell commands and execute code.
Optimized for Windows PowerShell with cross-platform support.
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
    """Execute shell commands safely with PowerShell/CMD support."""
    
    # Commands that are blocked for security
    BLOCKED_COMMANDS = [
        "rm -rf /", "format c:", "del /f /s /q c:",
        "shutdown", "reboot", "halt",
        ":(){:|:&};:",  # Fork bomb
        "remove-item -recurse -force c:",
    ]
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="shell_command",
            description="""Execute a shell command (PowerShell on Windows, bash on Linux/Mac).
Use for: running git commands, npm/pip commands, file operations, system info.
Examples: 'dir', 'Get-ChildItem', 'git status', 'npm install', 'python --version'""",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute. Use PowerShell cmdlets on Windows."
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for the command. Use '.' for current directory.",
                        "default": None
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Command timeout in seconds (default: 60)",
                        "default": 60
                    },
                    "use_powershell": {
                        "type": "boolean",
                        "description": "Force PowerShell on Windows (default: true)",
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
            if blocked.lower() in command_lower:
                return False
        
        return True
    
    async def execute(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: int = 60,
        use_powershell: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute shell command with proper Windows/PowerShell handling."""
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
                working_dir = working_dir.strip().strip('"').strip("'")
                if working_dir == ".":
                    cwd = Path.cwd()
                else:
                    cwd = Path(working_dir).expanduser().resolve()
                    if not cwd.exists():
                        return ToolResult(
                            success=False,
                            error=f"Working directory not found: {working_dir}"
                        )
            
            # Determine shell based on OS
            if sys.platform == "win32":
                if use_powershell:
                    # Use PowerShell with proper encoding
                    full_command = f'powershell.exe -NoProfile -NonInteractive -Command "{command}"'
                else:
                    full_command = command
                shell = True
                executable = None
            else:
                full_command = command
                shell = True
                executable = "/bin/bash"
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                full_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
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
            
            # Decode output with fallback encodings
            def decode_output(data: bytes) -> str:
                if not data:
                    return ""
                for encoding in ["utf-8", "cp1252", "latin-1"]:
                    try:
                        return data.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                return data.decode("utf-8", errors="replace")
            
            stdout_str = decode_output(stdout)
            stderr_str = decode_output(stderr)
            
            return ToolResult(
                success=process.returncode == 0,
                data={
                    "stdout": stdout_str.strip(),
                    "stderr": stderr_str.strip(),
                    "return_code": process.returncode,
                    "command": command,
                },
                metadata={
                    "working_dir": str(cwd) if cwd else os.getcwd(),
                    "timeout": timeout,
                    "shell": "powershell" if sys.platform == "win32" and use_powershell else "bash"
                }
            )
            
        except Exception as e:
            logger.error(f"Shell command failed: {e}")
            return ToolResult(success=False, error=f"Command execution failed: {str(e)}")


class PowerShellTool(BaseTool):
    """Execute PowerShell-specific commands with better integration."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="powershell",
            description="""Execute PowerShell commands. Best for Windows system administration.
Use for: Get-ChildItem, Get-Process, Get-Service, file operations, registry, etc.
Examples: 
- 'Get-ChildItem -Path C:\\ -Recurse -Filter *.txt'
- 'Get-Process | Where-Object {$_.CPU -gt 10}'
- 'Get-PSDrive' (list drives)""",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "PowerShell script or command to execute"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60)",
                        "default": 60
                    }
                },
                "required": ["script"]
            },
            returns="dict with output and errors",
            requires_confirmation=True
        )
    
    async def execute(
        self,
        script: str,
        working_dir: Optional[str] = None,
        timeout: int = 60,
        **kwargs
    ) -> ToolResult:
        """Execute PowerShell script."""
        if sys.platform != "win32":
            return ToolResult(
                success=False,
                error="PowerShell is only available on Windows"
            )
        
        try:
            cwd = None
            if working_dir:
                cwd = Path(working_dir).expanduser().resolve()
                if not cwd.exists():
                    return ToolResult(
                        success=False,
                        error=f"Working directory not found: {working_dir}"
                    )
            
            # Build PowerShell command
            ps_args = [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-Command", script
            ]
            
            process = await asyncio.create_subprocess_exec(
                *ps_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
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
                    error=f"PowerShell timed out after {timeout} seconds"
                )
            
            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""
            
            return ToolResult(
                success=process.returncode == 0,
                data={
                    "output": stdout_str.strip(),
                    "errors": stderr_str.strip(),
                    "return_code": process.returncode,
                },
                metadata={"script": script[:100]}
            )
            
        except Exception as e:
            logger.error(f"PowerShell execution failed: {e}")
            return ToolResult(success=False, error=str(e))


class SystemInfoTool(BaseTool):
    """Get system information - drives, memory, OS, etc."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system_info",
            description="Get system information including drives, disks, memory, OS version, and environment.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {
                    "info_type": {
                        "type": "string",
                        "enum": ["drives", "memory", "os", "env", "all"],
                        "description": "Type of information to retrieve",
                        "default": "all"
                    }
                },
                "required": []
            },
            returns="dict with system information"
        )
    
    async def execute(
        self,
        info_type: str = "all",
        **kwargs
    ) -> ToolResult:
        """Get system information."""
        try:
            import platform
            
            result = {}
            
            if info_type in ["drives", "all"]:
                # Get drive information
                if sys.platform == "win32":
                    import ctypes
                    drives = []
                    bitmask = ctypes.windll.kernel32.GetLogicalDrives()
                    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        if bitmask & 1:
                            drive_path = f"{letter}:\\"
                            try:
                                total, used, free = 0, 0, 0
                                if Path(drive_path).exists():
                                    import shutil
                                    usage = shutil.disk_usage(drive_path)
                                    total = usage.total // (1024**3)  # GB
                                    free = usage.free // (1024**3)
                                    used = usage.used // (1024**3)
                                drives.append({
                                    "drive": drive_path,
                                    "total_gb": total,
                                    "used_gb": used,
                                    "free_gb": free
                                })
                            except:
                                drives.append({"drive": drive_path, "accessible": False})
                        bitmask >>= 1
                    result["drives"] = drives
                else:
                    # Unix - get mounted filesystems
                    import shutil
                    result["drives"] = [{
                        "path": "/",
                        **dict(zip(["total_gb", "used_gb", "free_gb"], 
                                   [x // (1024**3) for x in shutil.disk_usage("/")]))
                    }]
            
            if info_type in ["memory", "all"]:
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    result["memory"] = {
                        "total_gb": round(mem.total / (1024**3), 2),
                        "available_gb": round(mem.available / (1024**3), 2),
                        "used_percent": mem.percent
                    }
                except ImportError:
                    result["memory"] = {"error": "psutil not installed"}
            
            if info_type in ["os", "all"]:
                result["os"] = {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "python_version": platform.python_version()
                }
            
            if info_type in ["env", "all"]:
                # Safe subset of environment variables
                safe_vars = ["PATH", "HOME", "USERPROFILE", "USERNAME", "USER", 
                             "COMPUTERNAME", "HOSTNAME", "VIRTUAL_ENV"]
                result["environment"] = {
                    k: os.environ.get(k, "") 
                    for k in safe_vars if k in os.environ
                }
                result["cwd"] = str(Path.cwd())
            
            return ToolResult(
                success=True,
                data=result,
                metadata={"info_type": info_type}
            )
            
        except Exception as e:
            logger.error(f"System info failed: {e}")
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
