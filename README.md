# IntelCLaw 🦅

> **Autonomous AI Agent for Windows** - Your intelligent, always-on AI assistant with screen understanding and task automation.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://github.com/astral-sh/uv)

---

## 🌟 Features

- **🤖 REACT Agent Architecture**: Multi-agent system using LangChain/LangGraph with ReAct (Reasoning + Acting) pattern
- **👁️ Screen Understanding**: Real-time screen capture, OCR, and UI element recognition
- **🧠 Persistent Memory**: Multi-tier memory system with conversation history and long-term knowledge
- **🎯 Task Automation**: Execute complex multi-step workflows autonomously
- **🔍 Intelligent Search**: Web search, file search, and semantic retrieval
- **🖥️ Transparent Overlay**: Always-available chat interface with global hotkey (Ctrl+Shift+Space)
- **🔒 Privacy First**: Configurable privacy filters and secure credential storage
- **🔧 Extensible Tools**: MCP (Model Context Protocol) support for unlimited extensibility

---

## 📋 Requirements

- **OS**: Windows 10/11
- **Python**: 3.11 or higher
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or pip
- **Tesseract OCR**: For screen text recognition
- **API Keys**: OpenAI (required), Tavily (optional)

---

## 🚀 Quick Start

### 1. Install uv (if not installed)

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the Repository

```powershell
git clone https://github.com/yourusername/IntelCLaw.git
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
```

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

## 📁 Project Structure

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
├── persona/                # Agent personality files
│   ├── AGENT.md
│   ├── SOUL.md
│   ├── MEMORY.md
│   ├── TOOLS.md
│   ├── SKILLS.md
│   └── CONTACTS.md
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
