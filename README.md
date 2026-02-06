# IntelCLaw 🦅

> **Autonomous AI Agent for Windows** - Your intelligent, always-on AI assistant with screen understanding and task automation.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://github.com/astral-sh/uv)

---

## 🌟 Features

- **🤖 REACT Agent Architecture**: Multi-agent system using LangChain/LangGraph with ReAct (Reasoning + Acting) pattern
- **👁️ Screen Understanding**: Real-time screen capture, OCR, and UI element recognition
- **🧠 Persistent Memory**: Local long-term memory (SQLite) with safe secret redaction (Mem0 optional)
- **📄 PDF RAG (PageIndex)**: Auto-ingest PDFs and retrieve relevant nodes from cached document trees
- **🎯 Task Automation**: Execute complex multi-step workflows autonomously
- **🔍 Intelligent Search**: Web search (Tavily), file search, and semantic RAG retrieval
- **📁 File Operations**: Read, write, search files with smart encoding detection and backup support
- **💻 PowerShell Integration**: Native PowerShell execution with system info tools for Windows
- **🖥️ Transparent Overlay**: Always-available chat interface with global hotkey (Ctrl+Shift+Space)
- **🌐 Web Gateway**: Real-time chat via WebSocket at `localhost:8765`
- **🔒 Privacy First**: Configurable privacy filters and secure credential storage
- **🔧 Extensible Tools**: MCP (Model Context Protocol) support for unlimited extensibility
- **🎭 Customizable Persona**: Edit `persona/SOUL.md` for personality and `persona/USER.md` for preferences

---

## 📋 Requirements

- **OS**: Windows 10/11
- **Python**: 3.11 or higher
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or pip
- **Tesseract OCR**: For screen text recognition
- **API Keys**: OpenAI (required), Tavily (optional), PageIndex (optional for PDF RAG)

---

## 🚀 Quick Start

### 1. Install uv (if not installed)

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the Repository

```powershell
git clone https://github.com/0x-Professor/IntelCLaw.git
cd IntelCLaw
```

### 3. Set Up Environment Variables

```powershell
# Create .env file
copy .env.example .env

# Edit .env with your API keys
notepad .env
```

Required:
```
OPENAI_API_KEY=sk-your-openai-key
```

Optional:
```
TAVILY_API_KEY=tvly-your-tavily-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GITHUB_TOKEN=ghp_your-github-token
PAGEINDEX_API_KEY=your-pageindex-key
```

> Security note: If you pasted an API key into chat/logs, treat it as compromised and rotate it in the provider dashboard.

### 4. Install Dependencies

```powershell
# Sync all dependencies
uv sync

# Or with specific groups
uv sync --group dev --group perception --group ui
```

### 5. Install Tesseract OCR

1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location or add to PATH

### 6. Run IntelCLaw

```powershell
# Using uv
uv run python main.py

# Or with options
uv run python main.py --debug
```

---

## 🎮 Usage

### Summon the Agent

Press **`Ctrl+Shift+Space`** to open the overlay interface.

### Quick Commands

| Command | Description |
|---------|-------------|
| `search: [query]` | Web search for information |
| `file: [action]` | File operations (read, write, search) |
| `code: [task]` | Coding assistance |
| `task: [description]` | Task management |
| `system: [command]` | System operations |

### Example Interactions

```
You: Search for the latest Python 3.13 features
IntelCLaw: [Searches web, summarizes key features]

You: Read the main.py file and explain what it does
IntelCLaw: [Reads file, provides explanation]

You: Create a new Python script that...
IntelCLaw: [Generates code, asks for confirmation]
```

---

## 🏗️ Architecture

```
IntelCLaw
├── 🧠 Agent System (LangChain/LangGraph)
│   ├── Orchestrator (REACT loop)
│   ├── Intent Router
│   └── Sub-Agents
│       ├── Research Agent
│       ├── Coding Agent
│       ├── Task Agent
│       └── System Agent
│
├── 👁️ Perception Layer
│   ├── Screen Capture (mss)
│   ├── OCR (pytesseract)
│   ├── UI Automation (pywinauto)
│   └── Activity Monitor
│
├── 🧠 Memory System
│   ├── Short-Term (conversation)
│   ├── Working (session/SQLite)
│   └── Long-Term (Mem0/ChromaDB)
│
├── 🔧 Tool System
│   ├── Built-in Tools
│   ├── MCP Servers
│   └── Tool Registry
│
└── 🖥️ User Interface
    ├── Transparent Overlay (PyQt6)
    └── System Tray (pystray)
```

---

## � Built-in Tools

IntelCLaw comes with a comprehensive set of built-in tools:

| Tool | Description |
|------|-------------|
| `file_read` | Read files with smart encoding detection (UTF-8, Latin-1, binary fallback) |
| `file_write` | Create/edit files with append mode and automatic backup (.bak) |
| `file_search` | Search for files by name pattern with recursive glob |
| `list_directory` | List directory contents with file sizes and types |
| `get_current_directory` | Get current working directory with metadata |
| `shell_command` | Execute shell commands (PowerShell on Windows) |
| `powershell` | Execute PowerShell scripts directly |
| `system_info` | Get drives, memory, OS info, and environment variables |
| `execute_code` | Run Python code in isolated subprocess |
| `pip_install` | Install Python packages via pip |
| `web_scrape` | Fetch and parse web page content |
| `tavily_search` | AI-powered web search via Tavily API |
| `screenshot` | Capture screen or window |
| `clipboard` | Read/write system clipboard |
| `launch_app` | Launch applications |

---

## �📁 Project Structure

```
IntelCLaw/
├── src/intelclaw/
│   ├── __init__.py
│   ├── core/               # App lifecycle, events
│   │   ├── app.py
│   │   └── events.py
│   ├── agent/              # Agent orchestration
│   │   ├── orchestrator.py
│   │   ├── router.py
│   │   ├── base.py
│   │   └── sub_agents/
│   ├── perception/         # Screen understanding
│   │   ├── manager.py
│   │   ├── screen_capture.py
│   │   ├── ocr.py
│   │   └── ui_automation.py
│   ├── memory/             # Memory systems
│   │   ├── manager.py
│   │   ├── short_term.py
│   │   ├── working_memory.py
│   │   └── long_term.py
│   ├── tools/              # Tool implementations
│   │   ├── registry.py
│   │   ├── base.py
│   │   └── builtin/
│   ├── ui/                 # User interface
│   │   ├── overlay.py
│   │   └── system_tray.py
│   ├── config/             # Configuration
│   │   └── manager.py
│   └── security/           # Security & auth
│       └── manager.py
├── persona/                # Agent personality & user preferences
│   ├── AGENT.md            # Agent behavior rules
│   ├── SOUL.md             # Personality, traits, communication style
│   ├── USER.md             # User preferences (coding style, paths, etc.)
│   ├── MEMORY.md           # Memory guidelines
│   ├── TOOLS.md            # Tool usage patterns
│   ├── SKILLS.md           # Learned skills
│   └── CONTACTS.md         # Known contacts
├── main.py                 # Entry point
├── config.yaml             # Configuration
├── pyproject.toml          # Dependencies
└── README.md
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize behavior:

```yaml
# Model Settings
models:
  primary: gpt-4o          # Main reasoning model
  fallback: gpt-4o-mini    # Fallback model
  temperature: 0.1         # Response randomness

# Privacy Settings
privacy:
  screen_capture: true
  track_keyboard: false    # Disabled by default
  excluded_windows:
    - "*password*"
    - "*bank*"

# Hotkeys
hotkeys:
  summon: ctrl+shift+space
  quick_action: ctrl+shift+q
```

---

## 📄 PageIndex PDF RAG (Auto-Watch)

If `PAGEINDEX_API_KEY` is set, IntelCLaw can ingest PDFs via PageIndex and cache a document tree locally for fast query-time retrieval.

- Drop PDFs into: `data/pageindex_inbox/` (configurable via `memory.pageindex.ingest_folder`)
- Cached trees + registry live under: `data/pageindex/`
- Index on-demand via the `rag_index_path` tool (see `persona/TOOLS.md`)

---

## 🌐 Web Gateway

IntelCLaw includes a WebSocket-based chat interface:

```powershell
# Start with web gateway enabled
uv run python main.py
```

Open `http://localhost:8765` in your browser for the web chat interface.

### WebSocket API

Connect to `ws://localhost:8765/ws` for real-time messaging:

```javascript
const ws = new WebSocket('ws://localhost:8765/ws');
ws.send(JSON.stringify({ type: 'message', content: 'Hello IntelCLaw!' }));
```

---

## 🎭 Persona System

Customize IntelCLaw's personality and behavior:

### SOUL.md - Agent Personality
Defines personality traits, communication style, and behavioral guidelines.

### USER.md - Your Preferences  
Store your coding preferences, project paths, and workflow settings. IntelCLaw will adapt to your style.

```markdown
# Example USER.md entries
**Primary Language**: Python
**Indentation**: 4 spaces
**Type Hints**: yes
**Projects Folder**: C:\Projects
```

---

## 🔧 Extending with MCP

Add MCP servers for additional capabilities:

```yaml
# In config.yaml
mcp:
  enabled: true
  servers:
    - name: filesystem
      command: uvx mcp-server-filesystem
      args: ["--allowed-dir", "C:/Users/"]
    
    - name: github
      command: uvx mcp-server-github
      env:
        GITHUB_TOKEN: ${GITHUB_TOKEN}
```

---

## 🛠️ Development

### Run Tests
```powershell
uv run pytest
```

### Run with Debug Logging
```powershell
uv run python main.py --debug
```

### Type Checking
```powershell
uv run mypy src/
```

### Formatting
```powershell
uv run ruff format .
uv run ruff check . --fix
```

---

## 🔒 Security

- **Credentials**: Stored in Windows Credential Manager
- **Audit Logging**: All operations logged to `data/audit.log`
- **Permission System**: Sensitive operations require confirmation
- **Privacy Filters**: Exclude sensitive windows from capture

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - Agent framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - State machine for agents
- [Mem0](https://mem0.ai/) - Long-term memory
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Tavily](https://tavily.com/) - AI-powered search

---

**Built with ❤️ for the future of human-AI collaboration**
