"""
Windows Native Tools - Services, Scheduled Tasks, Registry, Event Logs, UI Automation, CIM.

These tools provide first-class Windows automation capabilities using PowerShell
and Windows UI Automation (pywinauto).
"""

from __future__ import annotations

import asyncio
import json
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
        return ToolResult(
            success=success,
            data={
                "stdout": stdout_str.strip(),
                "stderr": stderr_str.strip(),
                "return_code": process.returncode,
            },
            metadata={"script": script[:200]},
        )
    except Exception as e:
        logger.error(f"PowerShell execution failed: {e}")
        return ToolResult(success=False, error=str(e))


def _parse_json_output(output: str) -> Any:
    if not output:
        return None
    try:
        return json.loads(output)
    except Exception:
        return output


class WindowsServicesTool(BaseTool):
    """Manage Windows services."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="windows_services",
            description="Manage Windows services: list, get, start, stop, restart, set_startup_type.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "get", "start", "stop", "restart", "set_startup_type"],
                        "description": "Service action to perform",
                    },
                    "name": {
                        "type": "string",
                        "description": "Service name (required for get/start/stop/restart/set_startup_type)",
                    },
                    "startup_type": {
                        "type": "string",
                        "enum": ["Automatic", "Manual", "Disabled"],
                        "description": "Startup type for set_startup_type",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of services to return for list",
                        "default": 200,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60)",
                        "default": 60,
                    },
                },
                "required": ["action"],
            },
            returns="dict with service data",
            requires_confirmation=True,
        )

    async def execute(
        self,
        action: str,
        name: Optional[str] = None,
        startup_type: Optional[str] = None,
        limit: int = 200,
        timeout: int = 60,
        **kwargs,
    ) -> ToolResult:
        action = (action or "").lower()
        if action in {"get", "start", "stop", "restart", "set_startup_type"} and not name:
            return ToolResult(success=False, error="Service name is required for this action")
        if action == "set_startup_type" and not startup_type:
            return ToolResult(success=False, error="startup_type is required for set_startup_type")

        ps = [
            "$ErrorActionPreference='Stop'",
        ]
        if action == "list":
            ps.append(
                f"Get-Service | Select-Object -First {int(limit)} Name, DisplayName, Status, StartType | ConvertTo-Json -Depth 3"
            )
        else:
            ps.append(f"$name = '{name}'")
            if action == "start":
                ps.append("Start-Service -Name $name")
            elif action == "stop":
                ps.append("Stop-Service -Name $name -Force")
            elif action == "restart":
                ps.append("Restart-Service -Name $name -Force")
            elif action == "set_startup_type":
                ps.append(f"Set-Service -Name $name -StartupType '{startup_type}'")
            ps.append(
                "Get-Service -Name $name | Select-Object Name, DisplayName, Status, StartType | ConvertTo-Json -Depth 3"
            )

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        return ToolResult(
            success=True,
            data={"services": parsed},
            metadata={"action": action},
        )


class WindowsTasksTool(BaseTool):
    """Manage Windows Scheduled Tasks."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="windows_tasks",
            description="Manage Windows Scheduled Tasks: list, get, create, enable, disable, delete, run.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "get", "create", "enable", "disable", "delete", "run"],
                        "description": "Task action to perform",
                    },
                    "name": {"type": "string", "description": "Task name"},
                    "action_path": {"type": "string", "description": "Executable path for create"},
                    "action_args": {"type": "string", "description": "Arguments for create"},
                    "schedule": {
                        "type": "string",
                        "enum": ["once", "daily", "at_logon", "on_startup"],
                        "description": "Schedule type for create",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start time for once/daily schedules (e.g., '09:00' or ISO)",
                    },
                    "description": {"type": "string", "description": "Task description"},
                    "limit": {
                        "type": "integer",
                        "description": "Max tasks to return for list",
                        "default": 200,
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Confirm destructive actions (delete)",
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
            returns="dict with task data",
            requires_confirmation=True,
        )

    async def execute(
        self,
        action: str,
        name: Optional[str] = None,
        action_path: Optional[str] = None,
        action_args: Optional[str] = None,
        schedule: Optional[str] = None,
        start_time: Optional[str] = None,
        description: Optional[str] = None,
        limit: int = 200,
        confirm: bool = False,
        timeout: int = 60,
        **kwargs,
    ) -> ToolResult:
        action = (action or "").lower()
        if action in {"get", "enable", "disable", "delete", "run"} and not name:
            return ToolResult(success=False, error="Task name is required for this action")
        if action == "delete" and not confirm:
            return ToolResult(success=False, error="Confirmation required to delete a task (confirm=true)")
        if action == "create":
            if not name or not action_path or not schedule:
                return ToolResult(success=False, error="name, action_path, and schedule are required for create")
            if schedule in {"once", "daily"} and not start_time:
                return ToolResult(success=False, error="start_time required for once/daily schedule")

        ps = ["$ErrorActionPreference='Stop'"]

        if action == "list":
            ps.append(
                f"Get-ScheduledTask | Select-Object -First {int(limit)} TaskName, State, TaskPath | ConvertTo-Json -Depth 3"
            )
        elif action == "get":
            ps.append(f"Get-ScheduledTask -TaskName '{name}' | Select-Object TaskName, State, TaskPath, Description | ConvertTo-Json -Depth 3")
        elif action == "enable":
            ps.append(f"Enable-ScheduledTask -TaskName '{name}'")
            ps.append(f"Get-ScheduledTask -TaskName '{name}' | Select-Object TaskName, State, TaskPath | ConvertTo-Json -Depth 3")
        elif action == "disable":
            ps.append(f"Disable-ScheduledTask -TaskName '{name}'")
            ps.append(f"Get-ScheduledTask -TaskName '{name}' | Select-Object TaskName, State, TaskPath | ConvertTo-Json -Depth 3")
        elif action == "run":
            ps.append(f"Start-ScheduledTask -TaskName '{name}'")
            ps.append(f"Get-ScheduledTask -TaskName '{name}' | Select-Object TaskName, State, TaskPath | ConvertTo-Json -Depth 3")
        elif action == "delete":
            ps.append(f"Unregister-ScheduledTask -TaskName '{name}' -Confirm:$false")
            ps.append("'{}' | ConvertTo-Json")
        elif action == "create":
            ps.append(f"$action = New-ScheduledTaskAction -Execute '{action_path}' -Argument '{action_args or ''}'")
            if schedule == "once":
                ps.append(f"$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date '{start_time}')")
            elif schedule == "daily":
                ps.append(f"$trigger = New-ScheduledTaskTrigger -Daily -At (Get-Date '{start_time}')")
            elif schedule == "at_logon":
                ps.append("$trigger = New-ScheduledTaskTrigger -AtLogOn")
            elif schedule == "on_startup":
                ps.append("$trigger = New-ScheduledTaskTrigger -AtStartup")
            if description:
                ps.append(f"$desc = '{description}'")
                ps.append("Register-ScheduledTask -TaskName '{0}' -Action $action -Trigger $trigger -Description $desc -Force".format(name))
            else:
                ps.append("Register-ScheduledTask -TaskName '{0}' -Action $action -Trigger $trigger -Force".format(name))
            ps.append(f"Get-ScheduledTask -TaskName '{name}' | Select-Object TaskName, State, TaskPath | ConvertTo-Json -Depth 3")

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        return ToolResult(success=True, data={"tasks": parsed}, metadata={"action": action})


class WindowsRegistryTool(BaseTool):
    """Read/write Windows Registry values."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="windows_registry",
            description="Access Windows Registry values and keys.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get", "set", "delete", "list_keys", "list_values"],
                        "description": "Registry action to perform",
                    },
                    "path": {"type": "string", "description": "Registry key path (e.g., HKLM:\\Software\\...)"},
                    "name": {"type": "string", "description": "Registry value name"},
                    "value": {"description": "Registry value to set"},
                    "value_type": {
                        "type": "string",
                        "enum": ["String", "DWord", "QWord", "MultiString", "ExpandString"],
                        "description": "Registry value type for set",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Confirm destructive actions (delete)",
                        "default": False,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60)",
                        "default": 60,
                    },
                },
                "required": ["action", "path"],
            },
            returns="dict with registry data",
            requires_confirmation=True,
        )

    async def execute(
        self,
        action: str,
        path: str,
        name: Optional[str] = None,
        value: Any = None,
        value_type: Optional[str] = None,
        confirm: bool = False,
        timeout: int = 60,
        **kwargs,
    ) -> ToolResult:
        action = (action or "").lower()
        if action in {"get", "set", "delete"} and not name:
            return ToolResult(success=False, error="Registry value name is required for this action")
        if action == "delete" and not confirm:
            return ToolResult(success=False, error="Confirmation required to delete a registry value (confirm=true)")
        if action == "set" and value_type is None:
            return ToolResult(success=False, error="value_type is required for set")

        value_json = json.dumps(value)
        ps = ["$ErrorActionPreference='Stop'"]
        ps.append(f"$path = '{path}'")

        if action == "get":
            ps.append(f"$value = Get-ItemProperty -Path $path -Name '{name}' | Select-Object -ExpandProperty '{name}'")
            ps.append("@{Name='{0}'; Value=$value} | ConvertTo-Json -Depth 3".format(name))
        elif action == "list_keys":
            ps.append("Get-ChildItem -Path $path | Select-Object Name | ConvertTo-Json -Depth 3")
        elif action == "list_values":
            ps.append("Get-ItemProperty -Path $path | Select-Object * -ExcludeProperty PSPath,PSParentPath,PSChildName,PSDrive,PSProvider | ConvertTo-Json -Depth 3")
        elif action == "set":
            ps.append(f"$value = ConvertFrom-Json -InputObject '{value_json}'")
            ps.append(
                "New-ItemProperty -Path $path -Name '{0}' -Value $value -PropertyType '{1}' -Force | Out-Null".format(
                    name, value_type
                )
            )
            ps.append("Get-ItemProperty -Path $path -Name '{0}' | Select-Object -ExpandProperty '{0}' | ConvertTo-Json -Depth 3".format(name))
        elif action == "delete":
            ps.append("Remove-ItemProperty -Path $path -Name '{0}' -Force".format(name))
            ps.append("'{}' | ConvertTo-Json")

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        return ToolResult(success=True, data={"registry": parsed}, metadata={"action": action})


class WindowsEventLogTool(BaseTool):
    """Query Windows Event Logs."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="windows_eventlog",
            description="Query Windows Event Logs (list logs, query events).",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list_logs", "query"],
                        "description": "Event log action",
                    },
                    "log_name": {"type": "string", "description": "Log name for query"},
                    "levels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Levels: Critical, Error, Warning, Information, Verbose",
                    },
                    "start_time": {"type": "string", "description": "Start time (e.g., ISO or '2025-01-01')"},
                    "end_time": {"type": "string", "description": "End time"},
                    "max_results": {
                        "type": "integer",
                        "description": "Max events to return",
                        "default": 50,
                    },
                    "truncate_message": {
                        "type": "integer",
                        "description": "Max message length (default: 2000)",
                        "default": 2000,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60)",
                        "default": 60,
                    },
                },
                "required": ["action"],
            },
            returns="dict with event log entries",
            requires_confirmation=True,
        )

    async def execute(
        self,
        action: str,
        log_name: Optional[str] = None,
        levels: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        max_results: int = 50,
        truncate_message: int = 2000,
        timeout: int = 60,
        **kwargs,
    ) -> ToolResult:
        action = (action or "").lower()
        if action == "query" and not log_name:
            return ToolResult(success=False, error="log_name is required for query")

        level_map = {
            "critical": 1,
            "error": 2,
            "warning": 3,
            "information": 4,
            "verbose": 5,
        }
        level_values = []
        if levels:
            for lvl in levels:
                val = level_map.get(str(lvl).lower())
                if val:
                    level_values.append(val)

        ps = ["$ErrorActionPreference='Stop'"]
        if action == "list_logs":
            ps.append(
                f"Get-WinEvent -ListLog * | Select-Object -First {int(max_results)} LogName, RecordCount, LastWriteTime | ConvertTo-Json -Depth 3"
            )
        else:
            ps.append("$filter = @{}")
            ps.append(f"$filter.LogName = '{log_name}'")
            if level_values:
                ps.append("$filter.Level = @({0})".format(",".join(str(v) for v in level_values)))
            if start_time:
                ps.append(f"$filter.StartTime = (Get-Date '{start_time}')")
            if end_time:
                ps.append(f"$filter.EndTime = (Get-Date '{end_time}')")
            ps.append(
                f"Get-WinEvent -FilterHashtable $filter -MaxEvents {int(max_results)} | "
                "Select-Object TimeCreated, Id, LevelDisplayName, ProviderName, Message | ConvertTo-Json -Depth 4"
            )

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        if isinstance(parsed, list):
            for item in parsed:
                msg = item.get("Message")
                if msg and truncate_message and len(msg) > truncate_message:
                    item["Message"] = msg[:truncate_message] + "..."
        return ToolResult(success=True, data={"events": parsed}, metadata={"action": action})


class WindowsUIAutomationTool(BaseTool):
    """Windows UI Automation via pywinauto."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="windows_ui_automation",
            description="Automate Windows UI: find_window, click, invoke, set_text, get_text, list_controls.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["find_window", "click", "invoke", "set_text", "get_text", "list_controls"],
                        "description": "UI action to perform",
                    },
                    "window_title": {"type": "string", "description": "Exact window title"},
                    "window_regex": {"type": "string", "description": "Regex for window title"},
                    "control_title": {"type": "string", "description": "Control title"},
                    "auto_id": {"type": "string", "description": "Automation ID"},
                    "control_type": {"type": "string", "description": "Control type (e.g., Button, Edit)"},
                    "text": {"type": "string", "description": "Text for set_text"},
                    "limit": {
                        "type": "integer",
                        "description": "Max controls to return for list_controls",
                        "default": 50,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 20)",
                        "default": 20,
                    },
                },
                "required": ["action"],
            },
            returns="dict with UI automation results",
            requires_confirmation=True,
        )

    async def execute(
        self,
        action: str,
        window_title: Optional[str] = None,
        window_regex: Optional[str] = None,
        control_title: Optional[str] = None,
        auto_id: Optional[str] = None,
        control_type: Optional[str] = None,
        text: Optional[str] = None,
        limit: int = 50,
        timeout: int = 20,
        **kwargs,
    ) -> ToolResult:
        if sys.platform != "win32":
            return ToolResult(success=False, error="UI automation is only available on Windows")

        try:
            from pywinauto import Desktop
        except Exception as e:
            return ToolResult(success=False, error=f"pywinauto not available: {e}")

        action = (action or "").lower()
        desktop = Desktop(backend="uia")

        def resolve_window():
            if window_title:
                return desktop.window(title=window_title)
            if window_regex:
                return desktop.window(title_re=window_regex)
            return None

        if action == "find_window":
            windows = []
            if window_title:
                matches = desktop.windows(title=window_title)
            elif window_regex:
                matches = desktop.windows(title_re=window_regex)
            else:
                return ToolResult(success=False, error="window_title or window_regex is required")
            for win in matches[:limit]:
                windows.append({
                    "title": win.window_text(),
                    "handle": int(win.handle),
                    "class_name": win.class_name(),
                    "pid": win.process_id(),
                })
            return ToolResult(success=True, data={"windows": windows})

        window = resolve_window()
        if not window:
            return ToolResult(success=False, error="window_title or window_regex is required")

        try:
            window.wait("exists enabled visible ready", timeout=timeout)
        except Exception as e:
            return ToolResult(success=False, error=f"Window not ready: {e}")

        if action == "list_controls":
            controls = []
            for ctrl in window.descendants()[:limit]:
                controls.append({
                    "title": ctrl.window_text(),
                    "control_type": ctrl.friendly_class_name(),
                    "auto_id": getattr(ctrl, "automation_id", lambda: None)(),
                })
            return ToolResult(success=True, data={"controls": controls})

        ctrl = window.child_window(
            title=control_title,
            auto_id=auto_id,
            control_type=control_type,
        )
        try:
            ctrl = ctrl.wrapper_object()
        except Exception as e:
            return ToolResult(success=False, error=f"Control not found: {e}")

        if action == "click":
            ctrl.click_input()
            return ToolResult(success=True, data={"clicked": True})
        if action == "invoke":
            if hasattr(ctrl, "invoke"):
                ctrl.invoke()
            else:
                ctrl.click_input()
            return ToolResult(success=True, data={"invoked": True})
        if action == "set_text":
            if text is None:
                return ToolResult(success=False, error="text is required for set_text")
            if hasattr(ctrl, "set_text"):
                ctrl.set_text(text)
            else:
                ctrl.type_keys(text, with_spaces=True, set_foreground=True)
            return ToolResult(success=True, data={"set_text": True})
        if action == "get_text":
            return ToolResult(success=True, data={"text": ctrl.window_text()})

        return ToolResult(success=False, error=f"Unsupported action: {action}")


class WindowsCIMTool(BaseTool):
    """Query Windows CIM/WMI classes."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="windows_cim",
            description="Query Windows CIM/WMI classes via PowerShell Get-CimInstance.",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.EXECUTE],
            parameters={
                "type": "object",
                "properties": {
                    "class_name": {"type": "string", "description": "CIM class name (e.g., Win32_OperatingSystem)"},
                    "filter": {"type": "string", "description": "Optional WMI filter (e.g., Name='svchost.exe')"},
                    "properties": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Properties to select (optional)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 100,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60)",
                        "default": 60,
                    },
                },
                "required": ["class_name"],
            },
            returns="list of CIM objects",
            requires_confirmation=True,
        )

    async def execute(
        self,
        class_name: str,
        filter: Optional[str] = None,
        properties: Optional[List[str]] = None,
        max_results: int = 100,
        timeout: int = 60,
        **kwargs,
    ) -> ToolResult:
        ps = ["$ErrorActionPreference='Stop'"]
        if filter:
            ps.append(
                f"$items = Get-CimInstance -ClassName '{class_name}' -Filter \"{filter}\""
            )
        else:
            ps.append(f"$items = Get-CimInstance -ClassName '{class_name}'")
        if properties:
            props = ", ".join(properties)
            ps.append(f"$items = $items | Select-Object -Property {props}")
        ps.append(f"$items | Select-Object -First {int(max_results)} | ConvertTo-Json -Depth 4")

        result = await _run_powershell("; ".join(ps), timeout=timeout)
        if not result.success:
            return result
        parsed = _parse_json_output(result.data.get("stdout", ""))
        return ToolResult(success=True, data={"items": parsed}, metadata={"class_name": class_name})

