# TOOLS.md - IntelCLaw Tool Capabilities

## Built-in Tools

### 🔍 Search Tools

#### Web Search (Tavily)
```
tavily_search(query: str, max_results: int = 5, search_depth: str = "basic") -> SearchResults
```
Search the internet for current information, news, and research.

#### File Search
```
file_search(directory: str, pattern: str) -> List[FileMatch]
```
Search for files matching a pattern in your file system.

### 📁 File Operations

#### Read File
```
file_read(path: str, encoding: str = "utf-8") -> FileContent
```
Read the contents of a text file.

#### Write File
```
file_write(path: str, content: str, mode: str = "write") -> Result
```
Write content to a file (creates if doesn't exist). Supports write and append modes.

#### Delete File
```
file_delete(path: str) -> Result
```
Delete a file from the filesystem.

#### Copy File
```
file_copy(source: str, destination: str) -> Result
```
Copy a file from source to destination.

#### Move/Rename File
```
file_move(source: str, destination: str) -> Result
```
Move or rename a file.

#### List Directory
```
list_directory(path: str) -> List[FileInfo]
```
List files and folders in a directory.

#### Get Current Directory
```
get_cwd() -> str
```
Get the current working directory.

### 💻 Shell & Code Execution

#### Shell Command
```
shell_command(command: str, working_dir: str = None, timeout: int = 60) -> CommandResult
```
Execute a shell command (PowerShell on Windows, bash on Linux/Mac).

#### PowerShell
```
powershell(script: str, working_dir: str = None, timeout: int = 60) -> CommandResult
```
Execute PowerShell-specific commands with proper integration.

#### Execute Code
```
execute_code(code: str, timeout: int = 30) -> ExecutionResult
```
Execute Python code in a sandboxed subprocess.

#### Pip Install
```
pip_install(packages: List[str], upgrade: bool = False) -> Result
```
Install Python packages via pip.

### 🖥️ System Tools

#### Screenshot
```
screenshot(active_window_only: bool = False, save_path: str = None) -> Image
```
Capture the screen or active window.

#### Clipboard
```
clipboard(action: str, content: str = None) -> Result
```
Read from or write to the system clipboard. Actions: read, write.

#### Launch App
```
launch_app(target: str, arguments: List[str] = None) -> Result
```
Launch an application or open a file with its default application.

#### System Info
```
system_info(info_type: str = "all") -> SystemInfo
```
Get system information: drives, memory, OS, environment. Types: drives, memory, os, env, all.

### 🌐 Web Tools

#### Web Scrape
```
web_scrape(url: str, extract_text: bool = True) -> WebContent
```
Fetch and extract content from a web page.

---

## 🪟 Windows Native Tools

### Process Management
```
process_management(action: str, name: str = None, pid: int = None, limit: int = 50, sort_by: str = "memory") -> ProcessData
```
**Actions:**
- `list` — List all running processes with CPU, memory, PID
- `find` — Search processes by name (supports wildcards)
- `details` — Full details for a specific process (by PID or name)
- `kill` — Terminate a process by PID or name
- `top_cpu` — Top CPU-consuming processes
- `top_memory` — Top memory-consuming processes
- `tree` — Process tree view with parent-child relationships

### Windows Services
```
windows_services(action: str, name: str = None, startup_type: str = None) -> ServiceData
```
**Actions:** list, get, start, stop, restart, set_startup_type

### Scheduled Tasks
```
windows_tasks(action: str, name: str = None, action_path: str = None, schedule: str = None) -> TaskData
```
**Actions:** list, get, create, enable, disable, delete, run

### Registry
```
windows_registry(action: str, path: str, name: str = None, value: Any = None) -> RegistryData
```
**Actions:** get, set, delete, list_keys, list_values

### Event Logs
```
windows_eventlog(action: str, log_name: str = None, levels: List[str] = None, max_results: int = 50) -> EventData
```
**Actions:** list_logs, query
**Levels:** Critical, Error, Warning, Information, Verbose

### Network Info
```
network_info(action: str, target: str = None, limit: int = 100) -> NetworkData
```
**Actions:**
- `adapters` — List network adapters with status and speed
- `connections` — Active TCP connections with process info
- `listening` — Listening ports with process info
- `dns_cache` — DNS client cache entries
- `ping` — Ping a target host
- `traceroute` — Trace route to a host
- `wifi_profiles` — Saved Wi-Fi profiles
- `public_ip` — Get public IP address
- `arp_table` — ARP neighbor table

### Disk Management
```
disk_management(action: str, path: str = None, min_size_mb: int = 100) -> DiskData
```
**Actions:**
- `disks` — Physical disk information
- `volumes` — Volume details with usage percentage
- `partitions` — Partition table
- `space` — Drive space summary
- `large_files` — Find large files on a drive
- `folder_size` — Analyze folder sizes

### Firewall
```
windows_firewall(action: str, name: str = None, direction: str = None, action_type: str = None) -> FirewallData
```
**Actions:** status, list_rules, get_rule, enable_rule, disable_rule, create_rule, delete_rule

### Installed Applications
```
installed_apps(action: str, name: str = None) -> AppData
```
**Actions:**
- `list` — All installed desktop applications
- `search` — Search by name
- `details` — Full app details
- `startup_apps` — Startup programs
- `store_apps` — Microsoft Store apps

### Environment Variables
```
environment_vars(action: str, name: str = None, value: str = None, scope: str = "User") -> EnvData
```
**Actions:** list, get, set, remove, path
**Scopes:** User, Machine, Process

### Windows Updates
```
windows_update(action: str) -> UpdateData
```
**Actions:** installed, history, pending

### System Performance
```
system_performance(action: str) -> PerformanceData
```
**Actions:**
- `overview` — CPU %, memory, uptime, OS info
- `cpu` — CPU cores, clock speed, architecture
- `memory` — RAM slots, speeds, virtual memory
- `gpu` — GPU name, VRAM, driver info
- `battery` — Charge %, status, runtime estimate
- `uptime` — System boot time and uptime
- `hardware` — Motherboard, BIOS, manufacturer

### User Security
```
user_security(action: str, username: str = None) -> SecurityData
```
**Actions:** current_user, list_users, list_groups, user_groups, sessions, privileges

### UI Automation
```
windows_ui_automation(action: str, window_title: str = None, control_title: str = None) -> UIData
```
**Actions:** find_window, click, invoke, set_text, get_text, list_controls

### WMI/CIM Queries
```
windows_cim(class_name: str, filter: str = None, properties: List[str] = None) -> CIMData
```
Query any Windows CIM/WMI class for system inventory, hardware, and OS state.

---

## Permission Levels

### 🟢 Read (No Confirmation)
- File reading
- Web searches
- Screen capture
- Clipboard reading
- System info
- Disk info
- Network info (read-only)
- Installed apps listing
- System performance monitoring
- Windows update status

### 🟡 Write (Confirmation Optional)
- File creation / writing
- Clipboard writing
- Settings changes
- Environment variable changes

### 🔴 Execute (Requires Confirmation)
- File deletion
- System commands (shell/PowerShell)
- Application control
- Service management
- Scheduled task creation/deletion
- Registry modifications
- Firewall rule changes
- Process termination
- UI automation actions

## MCP Server Extensions

IntelCLaw can connect to Model Context Protocol (MCP) servers for extended capabilities:

### Supported Protocols
- stdio (local subprocess)
- HTTP/SSE (remote servers)

### Configuration
Add MCP servers in `config.yaml` under the `mcp` section.

### Configuration
```yaml
mcp:
  servers:
    - name: filesystem
      command: uvx mcp-server-filesystem
      args: ["--allowed-dir", "C:/Users/"]
    
    - name: github
      command: uvx mcp-server-github
      env:
        GITHUB_TOKEN: $GITHUB_TOKEN
```

## Tool Selection

I automatically select tools based on your request:

| Request Type | Primary Tool |
|-------------|--------------|
| "Search for..." | Web Search |
| "Find files..." | File Search |
| "Read the file..." | Read File |
| "Create a file..." | Write File |
| "Screenshot..." | Screenshot |
| "Open..." | Launch App |

## Custom Tools

You can extend my capabilities by:
1. Adding MCP servers
2. Creating custom Python tools
3. Configuring external APIs

## Safety Features

All tools include:
- **Input Validation**: Sanitize all inputs
- **Permission Checks**: Verify user consent
- **Error Handling**: Graceful failure recovery
- **Audit Logging**: Track all tool usage
