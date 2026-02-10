"""
Windows Extended Tools - Process Management, Network, Disk, Firewall, Installed Apps, Environment.

These tools complement the core Windows tools (windows.py) to provide comprehensive
Windows system automation capabilities for the IntelCLaw autonomous agent.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from loguru import logger

from intelclaw.tools.base import BaseTool, ToolDefinition, ToolResult, ToolCategory, ToolPermission


async def _run_powershell(script: str, timeout: int = 60) -> ToolResult:
    """Run a PowerShell script and return stdout/stderr in a ToolResult."""
    if sys.platform != "win32":
        return ToolResult(success=False, error="PowerShell is only available on Windows")

    try:
        ps_args = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy", "Bypass",
            "-Command", script,
        ]
        process = await asyncio.create_subprocess_exec(
            *ps_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            return ToolResult(success=False, error=f"PowerShell timed out after {timeout} seconds")

        stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

        success = process.returncode == 0
        error_msg = stderr_str.strip() if not success else None
        if not success and not error_msg:
            error_msg = f"PowerShell failed with return code {process.returncode}"
        return ToolResult(
            success=success,
            data={
                "stdout": stdout_str.strip(),
                "stderr": stderr_str.strip(),
                "return_code": process.returncode,
            },
            error=error_msg,
            metadata={"script": script[:200]},
        )
    except Exception as e:
        logger.error(f"PowerShell execution failed: {e}")
        return ToolResult(success=False, error=str(e))


def _parse_json_output(output: str) -> Any:
    """Parse JSON output from PowerShell, with fallback to raw text."""
    if not output:
        return None
    try:
        return json.loads(output)
    except Exception:
        return output


# =============================================================================
# Process Management Tool
# =============================================================================


class ProcessManagementTool(BaseTool):
    """Manage Windows processes - list, find, kill, get details."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="process_management",
            description=(
                "Manage Windows processes: list running processes/tasks, find by name or PID, "
                "get process details (CPU, memory, command line), kill/stop processes. "
                "Use this when the user asks about running tasks, processes, or applications."
            ),
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "find", "details", "kill", "top_cpu", "top_memory", "tree"],
                        "description": (
                            "Action: list=all processes, find=search by name, details=full info for PID/name, "
                            "kill=terminate process, top_cpu=top CPU consumers, top_memory=top memory consumers, "
                            "tree=process tree"
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": "Process name to search for (supports wildcards like *chrome*)",
                    },
                    "pid": {
                        "type": "integer",
                        "description": "Process ID for details/kill",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default: 50)",
                        "default": 50,
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["name", "cpu", "memory", "pid"],
                        "description": "Sort order for list (default: memory)",
                        "default": "memory",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force kill (default: false)",
                        "default": False,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60)",
                        "default": 60,
                    },
                },
                "required": ["action"],
            },
            returns="dict with process data",
            requires_confirmation=True,
        )

    async def execute(
        self,
        action: str,
        name: Optional[str] = None,
        pid: Optional[int] = None,
        limit: int = 50,
        sort_by: str = "memory",
        force: bool = False,
        timeout: int = 60,
        **kwargs,
    ) -> ToolResult:
        action = (action or "").lower()

        ps = ["$ErrorActionPreference='Stop'"]

        if action == "list":
            sort_map = {
                "name": "ProcessName",
                "cpu": "CPU",
                "memory": "WorkingSet64",
                "pid": "Id",
            }
            sort_col = sort_map.get(sort_by, "WorkingSet64")
            ps.append(
                f"Get-Process | Sort-Object {sort_col} -Descending | "
                f"Select-Object -First {int(limit)} Id, ProcessName, "
                "@{Name='CPU_Seconds';Expression={[math]::Round($_.CPU,2)}}, "
                "@{Name='Memory_MB';Expression={[math]::Round($_.WorkingSet64/1MB,1)}}, "
                "MainWindowTitle | ConvertTo-Json -Depth 3"
            )
        elif action == "find":
            if not name:
                return ToolResult(success=False, error="'name' is required for find action")
            ps.append(
                f"Get-Process -Name '{name}' -ErrorAction SilentlyContinue | "
                f"Select-Object -First {int(limit)} Id, ProcessName, "
                "@{Name='CPU_Seconds';Expression={[math]::Round($_.CPU,2)}}, "
                "@{Name='Memory_MB';Expression={[math]::Round($_.WorkingSet64/1MB,1)}}, "
                "MainWindowTitle | ConvertTo-Json -Depth 3"
            )
        elif action == "details":
            if pid:
                ps.append(f"$proc = Get-Process -Id {int(pid)} -ErrorAction Stop")
            elif name:
                ps.append(f"$proc = Get-Process -Name '{name}' -ErrorAction Stop | Select-Object -First 1")
            else:
                return ToolResult(success=False, error="'pid' or 'name' is required for details")

            ps.append("$start = $null; try { $start = $proc.StartTime } catch {}")
            ps.append("$exe = $null; $cmd = $null; try { $cim = Get-CimInstance Win32_Process -Filter \"ProcessId = $($proc.Id)\" -ErrorAction SilentlyContinue; if ($cim) { $exe = $cim.ExecutablePath; $cmd = $cim.CommandLine } } catch {}")
            ps.append(
                "[PSCustomObject]@{"
                "Id=$proc.Id;"
                "ProcessName=$proc.ProcessName;"
                "CPU_Seconds=[math]::Round($proc.CPU,2);"
                "Memory_MB=[math]::Round($proc.WorkingSet64/1MB,1);"
                "VirtualMemory_MB=[math]::Round($proc.VirtualMemorySize64/1MB,1);"
                "MainWindowTitle=$proc.MainWindowTitle;"
                "StartTime=$start;"
                "ExecutablePath=$exe;"
                "CommandLine=$cmd;"
                "ThreadCount=$proc.Threads.Count;"
                "HandleCount=$proc.HandleCount;"
                "Responding=$proc.Responding"
                "} | ConvertTo-Json -Depth 4"
            )
        elif action == "kill":
            if not pid and not name:
                return ToolResult(success=False, error="'pid' or 'name' is required for kill")
            if pid:
                force_flag = " -Force" if force else ""
                ps.append(f"Stop-Process -Id {int(pid)}{force_flag}")
                ps.append(f"@{{killed=$true; pid={int(pid)}}} | ConvertTo-Json")
            else:
                force_flag = " -Force" if force else ""
                ps.append(f"Stop-Process -Name '{name}'{force_flag}")
                ps.append(f"@{{killed=$true; name='{name}'}} | ConvertTo-Json")
        elif action == "top_cpu":
            ps.append(
                f"Get-Process | Sort-Object CPU -Descending | "
                f"Select-Object -First {int(limit)} Id, ProcessName, "
                "@{Name='CPU_Seconds';Expression={[math]::Round($_.CPU,2)}}, "
                "@{Name='Memory_MB';Expression={[math]::Round($_.WorkingSet64/1MB,1)}} | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "top_memory":
            ps.append(
                f"Get-Process | Sort-Object WorkingSet64 -Descending | "
                f"Select-Object -First {int(limit)} Id, ProcessName, "
                "@{Name='Memory_MB';Expression={[math]::Round($_.WorkingSet64/1MB,1)}}, "
                "@{Name='CPU_Seconds';Expression={[math]::Round($_.CPU,2)}} | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "tree":
            ps.append(
                "Get-CimInstance Win32_Process | Select-Object ProcessId, Name, ParentProcessId, "
                "@{Name='Memory_MB';Expression={[math]::Round($_.WorkingSetSize/1MB,1)}}, "
                f"CommandLine | Select-Object -First {int(limit)} | ConvertTo-Json -Depth 3"
            )
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        return ToolResult(success=True, data={"processes": parsed}, metadata={"action": action})


# =============================================================================
# Network Information Tool
# =============================================================================


class NetworkInfoTool(BaseTool):
    """Get network configuration, connections, adapters, and diagnostics."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="network_info",
            description=(
                "Get network information: adapters, IP addresses, active connections, "
                "listening ports, DNS cache, ping, traceroute, Wi-Fi profiles."
            ),
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "adapters", "connections", "listening", "dns_cache",
                            "ping", "traceroute", "wifi_profiles", "public_ip",
                            "speed_test", "arp_table",
                        ],
                        "description": "Network action to perform",
                    },
                    "target": {
                        "type": "string",
                        "description": "Target host for ping/traceroute",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 100)",
                        "default": 100,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60)",
                        "default": 60,
                    },
                },
                "required": ["action"],
            },
            returns="dict with network data",
            requires_confirmation=False,
        )

    async def execute(
        self,
        action: str,
        target: Optional[str] = None,
        limit: int = 100,
        timeout: int = 60,
        **kwargs,
    ) -> ToolResult:
        action = (action or "").lower()
        ps = ["$ErrorActionPreference='Stop'"]

        if action == "adapters":
            ps.append(
                "Get-NetAdapter | Select-Object Name, InterfaceDescription, Status, "
                "LinkSpeed, MacAddress, MediaType | ConvertTo-Json -Depth 3"
            )
        elif action == "connections":
            ps.append(
                f"Get-NetTCPConnection -State Established | "
                f"Select-Object -First {int(limit)} LocalAddress, LocalPort, "
                "RemoteAddress, RemotePort, State, OwningProcess, "
                "@{Name='ProcessName';Expression={(Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName}} | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "listening":
            ps.append(
                f"Get-NetTCPConnection -State Listen | "
                f"Select-Object -First {int(limit)} LocalAddress, LocalPort, "
                "OwningProcess, "
                "@{Name='ProcessName';Expression={(Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName}} | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "dns_cache":
            ps.append(
                f"Get-DnsClientCache | Select-Object -First {int(limit)} Entry, Name, Data, Type, TimeToLive | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "ping":
            if not target:
                return ToolResult(success=False, error="'target' is required for ping")
            ps.append(
                f"Test-Connection -ComputerName '{target}' -Count 4 | "
                "Select-Object Address, Latency, Status, BufferSize | ConvertTo-Json -Depth 3"
            )
        elif action == "traceroute":
            if not target:
                return ToolResult(success=False, error="'target' is required for traceroute")
            ps.append(
                f"Test-NetConnection -ComputerName '{target}' -TraceRoute | "
                "Select-Object ComputerName, RemoteAddress, PingSucceeded, "
                "@{Name='TraceRoute';Expression={$_.TraceRoute -join ', '}} | ConvertTo-Json -Depth 3"
            )
        elif action == "wifi_profiles":
            ps.append(
                "netsh wlan show profiles | "
                "Select-String 'All User Profile' | "
                "ForEach-Object { $_ -replace '.*: ', '' } | "
                "ConvertTo-Json"
            )
        elif action == "public_ip":
            ps.append(
                "(Invoke-RestMethod -Uri 'https://api.ipify.org?format=json' -TimeoutSec 10).ip"
            )
        elif action == "arp_table":
            ps.append(
                f"Get-NetNeighbor | Select-Object -First {int(limit)} "
                "IPAddress, LinkLayerAddress, State, InterfaceAlias | ConvertTo-Json -Depth 3"
            )
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        return ToolResult(success=True, data={"network": parsed}, metadata={"action": action})


# =============================================================================
# Disk Management Tool
# =============================================================================


class DiskManagementTool(BaseTool):
    """Disk and volume management."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="disk_management",
            description=(
                "Manage disks and volumes: list physical disks, partitions, volumes, "
                "check disk space, find large files, analyze folder sizes."
            ),
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["disks", "volumes", "partitions", "space", "large_files", "folder_size"],
                        "description": "Disk action to perform",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path for large_files/folder_size (default: C:\\)",
                    },
                    "min_size_mb": {
                        "type": "integer",
                        "description": "Min file size in MB for large_files (default: 100)",
                        "default": 100,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 50)",
                        "default": 50,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 120)",
                        "default": 120,
                    },
                },
                "required": ["action"],
            },
            returns="dict with disk data",
        )

    async def execute(
        self,
        action: str,
        path: Optional[str] = None,
        min_size_mb: int = 100,
        limit: int = 50,
        timeout: int = 120,
        **kwargs,
    ) -> ToolResult:
        action = (action or "").lower()
        ps = ["$ErrorActionPreference='Stop'"]

        if action == "disks":
            ps.append(
                "Get-PhysicalDisk | Select-Object FriendlyName, MediaType, "
                "@{Name='Size_GB';Expression={[math]::Round($_.Size/1GB,1)}}, "
                "HealthStatus, BusType, Model | ConvertTo-Json -Depth 3"
            )
        elif action == "volumes":
            ps.append(
                "Get-Volume | Where-Object {$_.DriveLetter} | Select-Object DriveLetter, FileSystemLabel, "
                "FileSystem, DriveType, HealthStatus, "
                "@{Name='Size_GB';Expression={[math]::Round($_.Size/1GB,1)}}, "
                "@{Name='Free_GB';Expression={[math]::Round($_.SizeRemaining/1GB,1)}}, "
                "@{Name='Used_Percent';Expression={if($_.Size -gt 0){[math]::Round(($_.Size-$_.SizeRemaining)/$_.Size*100,1)}else{0}}} | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "partitions":
            ps.append(
                "Get-Partition | Select-Object DiskNumber, PartitionNumber, DriveLetter, "
                "@{Name='Size_GB';Expression={[math]::Round($_.Size/1GB,1)}}, Type, IsActive | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "space":
            ps.append(
                "Get-PSDrive -PSProvider FileSystem | Select-Object Name, "
                "@{Name='Used_GB';Expression={[math]::Round($_.Used/1GB,1)}}, "
                "@{Name='Free_GB';Expression={[math]::Round($_.Free/1GB,1)}}, "
                "@{Name='Total_GB';Expression={[math]::Round(($_.Used+$_.Free)/1GB,1)}} | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "large_files":
            search_path = path or "C:\\"
            ps.append(
                f"Get-ChildItem -Path '{search_path}' -Recurse -File -ErrorAction SilentlyContinue | "
                f"Where-Object {{$_.Length -gt {int(min_size_mb) * 1024 * 1024}}} | "
                f"Sort-Object Length -Descending | Select-Object -First {int(limit)} "
                "FullName, @{Name='Size_MB';Expression={[math]::Round($_.Length/1MB,1)}}, "
                "LastWriteTime, Extension | ConvertTo-Json -Depth 3"
            )
        elif action == "folder_size":
            search_path = path or "C:\\"
            ps.append(
                f"Get-ChildItem -Path '{search_path}' -Directory -ErrorAction SilentlyContinue | "
                f"Select-Object -First {int(limit)} FullName, "
                "@{Name='Size_MB';Expression={[math]::Round(("
                "Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | "
                "Measure-Object Length -Sum).Sum/1MB,1)}} | "
                "Sort-Object Size_MB -Descending | ConvertTo-Json -Depth 3"
            )
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        return ToolResult(success=True, data={"disk": parsed}, metadata={"action": action})


# =============================================================================
# Firewall Management Tool
# =============================================================================


class FirewallTool(BaseTool):
    """Manage Windows Firewall rules."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="windows_firewall",
            description=(
                "Manage Windows Firewall: list rules, get rule details, "
                "enable/disable rules, create new rules, check firewall status."
            ),
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "list_rules", "get_rule", "enable_rule", "disable_rule", "create_rule", "delete_rule"],
                        "description": "Firewall action to perform",
                    },
                    "name": {"type": "string", "description": "Rule display name"},
                    "direction": {
                        "type": "string",
                        "enum": ["Inbound", "Outbound"],
                        "description": "Rule direction for create_rule",
                    },
                    "action_type": {
                        "type": "string",
                        "enum": ["Allow", "Block"],
                        "description": "Allow or Block for create_rule",
                    },
                    "protocol": {
                        "type": "string",
                        "enum": ["TCP", "UDP", "Any"],
                        "description": "Protocol for create_rule",
                    },
                    "port": {"type": "string", "description": "Port number(s) for create_rule (e.g., '80' or '80,443')"},
                    "program": {"type": "string", "description": "Program path for create_rule"},
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 100)",
                        "default": 100,
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Confirm destructive actions",
                        "default": False,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60)",
                        "default": 60,
                    },
                },
                "required": ["action"],
            },
            returns="dict with firewall data",
            requires_confirmation=True,
        )

    async def execute(
        self,
        action: str,
        name: Optional[str] = None,
        direction: Optional[str] = None,
        action_type: Optional[str] = None,
        protocol: Optional[str] = None,
        port: Optional[str] = None,
        program: Optional[str] = None,
        limit: int = 100,
        confirm: bool = False,
        timeout: int = 60,
        **kwargs,
    ) -> ToolResult:
        action = (action or "").lower()
        ps = ["$ErrorActionPreference='Stop'"]

        if action == "status":
            ps.append(
                "Get-NetFirewallProfile | Select-Object Name, Enabled, "
                "DefaultInboundAction, DefaultOutboundAction, LogFileName | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "list_rules":
            ps.append(
                f"Get-NetFirewallRule | Where-Object {{$_.Enabled -eq 'True'}} | "
                f"Select-Object -First {int(limit)} DisplayName, Direction, Action, "
                "Enabled, Profile | ConvertTo-Json -Depth 3"
            )
        elif action == "get_rule":
            if not name:
                return ToolResult(success=False, error="'name' is required for get_rule")
            ps.append(
                f"Get-NetFirewallRule -DisplayName '{name}' | "
                "Select-Object DisplayName, Direction, Action, Enabled, Profile, Description | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "enable_rule":
            if not name:
                return ToolResult(success=False, error="'name' is required")
            ps.append(f"Enable-NetFirewallRule -DisplayName '{name}'")
            ps.append(f"Get-NetFirewallRule -DisplayName '{name}' | Select-Object DisplayName, Enabled | ConvertTo-Json -Depth 3")
        elif action == "disable_rule":
            if not name:
                return ToolResult(success=False, error="'name' is required")
            ps.append(f"Disable-NetFirewallRule -DisplayName '{name}'")
            ps.append(f"Get-NetFirewallRule -DisplayName '{name}' | Select-Object DisplayName, Enabled | ConvertTo-Json -Depth 3")
        elif action == "create_rule":
            if not name or not direction or not action_type:
                return ToolResult(success=False, error="name, direction, and action_type are required for create_rule")
            cmd = f"New-NetFirewallRule -DisplayName '{name}' -Direction {direction} -Action {action_type}"
            if protocol:
                cmd += f" -Protocol {protocol}"
            if port:
                cmd += f" -LocalPort {port}"
            if program:
                cmd += f" -Program '{program}'"
            ps.append(cmd)
            ps.append(f"Get-NetFirewallRule -DisplayName '{name}' | Select-Object DisplayName, Direction, Action, Enabled | ConvertTo-Json -Depth 3")
        elif action == "delete_rule":
            if not name:
                return ToolResult(success=False, error="'name' is required")
            if not confirm:
                return ToolResult(success=False, error="Confirmation required to delete firewall rule (confirm=true)")
            ps.append(f"Remove-NetFirewallRule -DisplayName '{name}'")
            ps.append("@{deleted=$true} | ConvertTo-Json")
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        return ToolResult(success=True, data={"firewall": parsed}, metadata={"action": action})


# =============================================================================
# Installed Applications Tool
# =============================================================================


class InstalledAppsTool(BaseTool):
    """Query and manage installed applications."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="installed_apps",
            description=(
                "Query installed applications: list all installed programs, "
                "search by name, get details, check for updates, uninstall."
            ),
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "search", "details", "startup_apps", "store_apps"],
                        "description": "Action to perform",
                    },
                    "name": {"type": "string", "description": "Application name to search for"},
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 100)",
                        "default": 100,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 120)",
                        "default": 120,
                    },
                },
                "required": ["action"],
            },
            returns="dict with application data",
        )

    async def execute(
        self,
        action: str,
        name: Optional[str] = None,
        limit: int = 100,
        timeout: int = 120,
        **kwargs,
    ) -> ToolResult:
        action = (action or "").lower()
        ps = ["$ErrorActionPreference='Stop'"]

        if action == "list":
            ps.append(
                "Get-ItemProperty HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*, "
                "HKLM:\\Software\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\* "
                "-ErrorAction SilentlyContinue | "
                "Where-Object {$_.DisplayName} | "
                f"Select-Object -First {int(limit)} DisplayName, DisplayVersion, Publisher, "
                "InstallDate, EstimatedSize, InstallLocation | "
                "Sort-Object DisplayName | ConvertTo-Json -Depth 3"
            )
        elif action == "search":
            if not name:
                return ToolResult(success=False, error="'name' is required for search")
            ps.append(
                "Get-ItemProperty HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*, "
                "HKLM:\\Software\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\* "
                "-ErrorAction SilentlyContinue | "
                f"Where-Object {{$_.DisplayName -like '*{name}*'}} | "
                "Select-Object DisplayName, DisplayVersion, Publisher, "
                "InstallDate, EstimatedSize, InstallLocation, UninstallString | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "details":
            if not name:
                return ToolResult(success=False, error="'name' is required for details")
            ps.append(
                "Get-ItemProperty HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*, "
                "HKLM:\\Software\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\* "
                "-ErrorAction SilentlyContinue | "
                f"Where-Object {{$_.DisplayName -like '*{name}*'}} | Select-Object -First 1 "
                "DisplayName, DisplayVersion, Publisher, InstallDate, "
                "InstallLocation, UninstallString, EstimatedSize, URLInfoAbout, "
                "HelpLink | ConvertTo-Json -Depth 3"
            )
        elif action == "startup_apps":
            ps.append(
                "Get-CimInstance Win32_StartupCommand | "
                f"Select-Object -First {int(limit)} Name, Command, Location, User | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "store_apps":
            ps.append(
                f"Get-AppxPackage | Select-Object -First {int(limit)} Name, "
                "PackageFullName, Version, Publisher, Status | ConvertTo-Json -Depth 3"
            )
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        return ToolResult(success=True, data={"apps": parsed}, metadata={"action": action})


# =============================================================================
# Environment Variables Tool
# =============================================================================


class EnvironmentTool(BaseTool):
    """Manage Windows environment variables."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="environment_vars",
            description=(
                "Manage environment variables: list all, get specific, set, remove. "
                "Supports User, Machine, and Process scope."
            ),
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "get", "set", "remove", "path"],
                        "description": "Action: list=all vars, get=specific var, set=create/update, remove=delete, path=show PATH entries",
                    },
                    "name": {"type": "string", "description": "Variable name"},
                    "value": {"type": "string", "description": "Variable value for set"},
                    "scope": {
                        "type": "string",
                        "enum": ["User", "Machine", "Process"],
                        "description": "Scope (default: User)",
                        "default": "User",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30)",
                        "default": 30,
                    },
                },
                "required": ["action"],
            },
            returns="dict with environment data",
            requires_confirmation=True,
        )

    async def execute(
        self,
        action: str,
        name: Optional[str] = None,
        value: Optional[str] = None,
        scope: str = "User",
        timeout: int = 30,
        **kwargs,
    ) -> ToolResult:
        action = (action or "").lower()
        ps = ["$ErrorActionPreference='Stop'"]

        if action == "list":
            ps.append(
                f"[Environment]::GetEnvironmentVariables('{scope}') | "
                "ForEach-Object { $_.GetEnumerator() | Select-Object Key, Value } | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "get":
            if not name:
                return ToolResult(success=False, error="'name' is required for get")
            ps.append(
                f"@{{Name='{name}'; Value=[Environment]::GetEnvironmentVariable('{name}', '{scope}')}} | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "set":
            if not name or value is None:
                return ToolResult(success=False, error="'name' and 'value' are required for set")
            ps.append(
                f"[Environment]::SetEnvironmentVariable('{name}', '{value}', '{scope}')"
            )
            ps.append(
                f"@{{Name='{name}'; Value='{value}'; Scope='{scope}'; Set=$true}} | ConvertTo-Json -Depth 3"
            )
        elif action == "remove":
            if not name:
                return ToolResult(success=False, error="'name' is required for remove")
            ps.append(
                f"[Environment]::SetEnvironmentVariable('{name}', $null, '{scope}')"
            )
            ps.append(f"@{{Name='{name}'; Removed=$true}} | ConvertTo-Json -Depth 3")
        elif action == "path":
            ps.append(
                f"[Environment]::GetEnvironmentVariable('PATH', '{scope}') -split ';' | "
                "Where-Object {$_} | ConvertTo-Json -Depth 3"
            )
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        return ToolResult(success=True, data={"environment": parsed}, metadata={"action": action})


# =============================================================================
# Windows Update Tool
# =============================================================================


class WindowsUpdateTool(BaseTool):
    """Check and manage Windows Updates."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="windows_update",
            description=(
                "Check and manage Windows Updates: list installed updates, "
                "check for pending updates, get update history."
            ),
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["installed", "history", "pending"],
                        "description": "Update action to perform",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 50)",
                        "default": 50,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 120)",
                        "default": 120,
                    },
                },
                "required": ["action"],
            },
            returns="dict with update data",
        )

    async def execute(
        self,
        action: str,
        limit: int = 50,
        timeout: int = 120,
        **kwargs,
    ) -> ToolResult:
        action = (action or "").lower()
        ps = ["$ErrorActionPreference='Stop'"]

        if action == "installed":
            ps.append(
                f"Get-HotFix | Select-Object -First {int(limit)} "
                "HotFixID, Description, InstalledOn, InstalledBy | "
                "Sort-Object InstalledOn -Descending | ConvertTo-Json -Depth 3"
            )
        elif action == "history":
            ps.append(
                "$session = New-Object -ComObject Microsoft.Update.Session; "
                "$searcher = $session.CreateUpdateSearcher(); "
                f"$history = $searcher.QueryHistory(0, {int(limit)}); "
                "$history | Select-Object Title, Date, "
                "@{Name='Result';Expression={switch($_.ResultCode){0{'NotStarted'}1{'InProgress'}2{'Succeeded'}3{'SucceededWithErrors'}4{'Failed'}5{'Aborted'}}}}, "
                "Description | ConvertTo-Json -Depth 3"
            )
        elif action == "pending":
            ps.append(
                "$session = New-Object -ComObject Microsoft.Update.Session; "
                "$searcher = $session.CreateUpdateSearcher(); "
                "$results = $searcher.Search('IsInstalled=0'); "
                "$results.Updates | Select-Object Title, "
                "@{Name='Size_MB';Expression={[math]::Round($_.MaxDownloadSize/1MB,1)}}, "
                "IsMandatory, IsDownloaded, Description | ConvertTo-Json -Depth 3"
            )
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        return ToolResult(success=True, data={"updates": parsed}, metadata={"action": action})


# =============================================================================
# System Performance Tool
# =============================================================================


class SystemPerformanceTool(BaseTool):
    """Monitor system performance: CPU, memory, disk I/O, GPU."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system_performance",
            description=(
                "Monitor system performance in real-time: CPU usage, memory usage, "
                "disk I/O, GPU info, system uptime, battery status, hardware info."
            ),
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["overview", "cpu", "memory", "gpu", "battery", "uptime", "hardware", "temperatures"],
                        "description": "Performance metric to check",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30)",
                        "default": 30,
                    },
                },
                "required": ["action"],
            },
            returns="dict with performance data",
        )

    async def execute(
        self,
        action: str,
        timeout: int = 30,
        **kwargs,
    ) -> ToolResult:
        action = (action or "").lower()
        ps = ["$ErrorActionPreference='Stop'"]

        if action == "overview":
            ps.append(
                "$cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average; "
                "$os = Get-CimInstance Win32_OperatingSystem; "
                "$mem_total = [math]::Round($os.TotalVisibleMemorySize/1MB, 1); "
                "$mem_free = [math]::Round($os.FreePhysicalMemory/1MB, 1); "
                "$mem_used = $mem_total - $mem_free; "
                "$uptime = (Get-Date) - $os.LastBootUpTime; "
                "@{CPU_Percent=$cpu; Memory_Total_GB=$mem_total; Memory_Used_GB=$mem_used; "
                "Memory_Free_GB=$mem_free; Memory_Percent=[math]::Round($mem_used/$mem_total*100,1); "
                "Uptime_Days=[math]::Round($uptime.TotalDays,1); "
                "Uptime_Hours=[math]::Round($uptime.TotalHours,1); "
                "ComputerName=$os.CSName; OS=$os.Caption; "
                "OS_Version=$os.Version} | ConvertTo-Json -Depth 3"
            )
        elif action == "cpu":
            ps.append(
                "Get-CimInstance Win32_Processor | Select-Object Name, "
                "NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed, "
                "LoadPercentage, CurrentClockSpeed, "
                "@{Name='Architecture';Expression={switch($_.Architecture){0{'x86'}9{'x64'}5{'ARM'}12{'ARM64'}}}} | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "memory":
            ps.append(
                "$os = Get-CimInstance Win32_OperatingSystem; "
                "$slots = Get-CimInstance Win32_PhysicalMemory; "
                "@{Total_GB=[math]::Round($os.TotalVisibleMemorySize/1MB,1); "
                "Free_GB=[math]::Round($os.FreePhysicalMemory/1MB,1); "
                "Used_GB=[math]::Round(($os.TotalVisibleMemorySize-$os.FreePhysicalMemory)/1MB,1); "
                "Virtual_Total_GB=[math]::Round($os.TotalVirtualMemorySize/1MB,1); "
                "Virtual_Free_GB=[math]::Round($os.FreeVirtualMemory/1MB,1); "
                "Slots=($slots | Select-Object DeviceLocator, "
                "@{Name='Size_GB';Expression={[math]::Round($_.Capacity/1GB,1)}}, Speed, Manufacturer)} | "
                "ConvertTo-Json -Depth 4"
            )
        elif action == "gpu":
            ps.append(
                "Get-CimInstance Win32_VideoController | Select-Object Name, "
                "DriverVersion, DriverDate, VideoProcessor, "
                "@{Name='VRAM_GB';Expression={[math]::Round($_.AdapterRAM/1GB,1)}}, "
                "CurrentRefreshRate, VideoModeDescription, Status | "
                "ConvertTo-Json -Depth 3"
            )
        elif action == "battery":
            ps.append(
                "$batt = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue; "
                "if ($batt) { $batt | Select-Object "
                "@{Name='Charge_Percent';Expression={$_.EstimatedChargeRemaining}}, "
                "@{Name='Status';Expression={switch($_.BatteryStatus){1{'Discharging'}2{'AC Power'}3{'Fully Charged'}4{'Low'}5{'Critical'}}}}, "
                "EstimatedRunTime, DesignCapacity | ConvertTo-Json -Depth 3 } "
                "else { @{battery='No battery detected (desktop)'} | ConvertTo-Json }"
            )
        elif action == "uptime":
            ps.append(
                "$os = Get-CimInstance Win32_OperatingSystem; "
                "$uptime = (Get-Date) - $os.LastBootUpTime; "
                "@{BootTime=$os.LastBootUpTime.ToString('o'); "
                "Uptime_Days=[math]::Round($uptime.TotalDays,2); "
                "Uptime_Hours=[math]::Round($uptime.TotalHours,1); "
                "Uptime_String=$uptime.ToString()} | ConvertTo-Json -Depth 3"
            )
        elif action == "hardware":
            ps.append(
                "$cs = Get-CimInstance Win32_ComputerSystem; "
                "$bios = Get-CimInstance Win32_BIOS; "
                "$mb = Get-CimInstance Win32_BaseBoard; "
                "@{Manufacturer=$cs.Manufacturer; Model=$cs.Model; "
                "SystemType=$cs.SystemType; "
                "TotalMemory_GB=[math]::Round($cs.TotalPhysicalMemory/1GB,1); "
                "BIOS_Version=$bios.SMBIOSBIOSVersion; "
                "BIOS_Date=$bios.ReleaseDate; "
                "Motherboard=$mb.Product; "
                "Motherboard_Manufacturer=$mb.Manufacturer} | ConvertTo-Json -Depth 3"
            )
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        return ToolResult(success=True, data={"performance": parsed}, metadata={"action": action})


# =============================================================================
# User and Security Tool
# =============================================================================


class UserSecurityTool(BaseTool):
    """Manage Windows users, groups, and security settings."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="user_security",
            description=(
                "Manage users and security: list local users, list groups, "
                "check current user, user sessions, audit policies."
            ),
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["current_user", "list_users", "list_groups", "user_groups", "sessions", "privileges"],
                        "description": "Security action to perform",
                    },
                    "username": {"type": "string", "description": "Username for user_groups"},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30)",
                        "default": 30,
                    },
                },
                "required": ["action"],
            },
            returns="dict with user/security data",
        )

    async def execute(
        self,
        action: str,
        username: Optional[str] = None,
        timeout: int = 30,
        **kwargs,
    ) -> ToolResult:
        action = (action or "").lower()
        ps = ["$ErrorActionPreference='Stop'"]

        if action == "current_user":
            ps.append(
                "@{Username=$env:USERNAME; Domain=$env:USERDOMAIN; "
                "ComputerName=$env:COMPUTERNAME; "
                "IsAdmin=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator); "
                "HomePath=$env:USERPROFILE} | ConvertTo-Json -Depth 3"
            )
        elif action == "list_users":
            ps.append(
                "Get-LocalUser | Select-Object Name, Enabled, LastLogon, "
                "PasswordLastSet, Description | ConvertTo-Json -Depth 3"
            )
        elif action == "list_groups":
            ps.append(
                "Get-LocalGroup | Select-Object Name, Description | ConvertTo-Json -Depth 3"
            )
        elif action == "user_groups":
            user = username or "$env:USERNAME"
            ps.append(
                f"Get-LocalGroup | Where-Object {{ (Get-LocalGroupMember $_.Name -ErrorAction SilentlyContinue).Name -like '*{user}*' }} | "
                "Select-Object Name, Description | ConvertTo-Json -Depth 3"
            )
        elif action == "sessions":
            ps.append(
                "query user 2>$null | ForEach-Object { $_ } | ConvertTo-Json"
            )
        elif action == "privileges":
            ps.append(
                "whoami /priv /fo csv | ConvertFrom-Csv | ConvertTo-Json -Depth 3"
            )
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        return ToolResult(success=True, data={"security": parsed}, metadata={"action": action})
