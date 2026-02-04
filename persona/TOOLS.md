# TOOLS.md - IntelCLaw Tool Capabilities

## Built-in Tools

### 🔍 Search Tools

#### Web Search (Tavily)
```
search_web(query: str) -> SearchResults
```
Search the internet for current information, news, and research.

#### File Search
```
search_files(pattern: str, directory: str) -> List[FileMatch]
```
Search for files matching a pattern in your file system.

### 📁 File Operations

#### Read File
```
read_file(path: str) -> FileContent
```
Read the contents of a text file.

#### Write File
```
write_file(path: str, content: str) -> Result
```
Write content to a file (creates if doesn't exist).

#### List Directory
```
list_directory(path: str) -> List[FileInfo]
```
List files and folders in a directory.

### 💻 System Tools

#### Screenshot
```
take_screenshot(region: Optional[Rect]) -> Image
```
Capture the screen or a specific region.

#### Clipboard
```
get_clipboard() -> str
set_clipboard(content: str) -> Result
```
Read from or write to the system clipboard.

#### Application Control
```
launch_app(app_name: str) -> Result
close_app(app_name: str) -> Result
```
Launch or close applications.

### 🌐 Web Tools

#### Fetch Webpage
```
fetch_webpage(url: str) -> WebContent
```
Fetch and parse content from a webpage.

#### Extract Links
```
extract_links(url: str) -> List[Link]
```
Extract all links from a webpage.

## Permission Levels

### 🟢 Read (No Confirmation)
- File reading
- Web searches
- Screen capture
- Clipboard reading

### 🟡 Write (Confirmation Optional)
- File creation
- Clipboard writing
- Settings changes

### 🔴 Execute (Requires Confirmation)
- File deletion
- System commands
- Application control

## MCP Server Extensions

IntelCLaw can connect to Model Context Protocol (MCP) servers for extended capabilities:

### Supported Protocols
- stdio (local subprocess)
- HTTP/SSE (remote servers)

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
