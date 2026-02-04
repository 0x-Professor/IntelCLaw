# CONTACTS.md - IntelCLaw Contact & Integration Points

## Communication Channels

### Primary Interface
- **Overlay UI**: Ctrl+Shift+Space
- **System Tray**: Right-click for quick actions

### API Integrations

#### AI Providers
| Provider | Models | Purpose |
|----------|--------|---------|
| OpenAI | gpt-4o, gpt-4o-mini | Primary reasoning |
| Anthropic | claude-3.5-sonnet | Fallback, coding |
| Google | gemini-pro | Alternative |

#### Search & Information
| Service | API | Purpose |
|---------|-----|---------|
| Tavily | Search API | Web research |
| GitHub | REST API | Code repositories |

#### MCP Servers
Extensible tool servers:
- Filesystem access
- GitHub integration
- Browser automation
- Database queries

## Environment Variables

Required for full functionality:
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
GITHUB_TOKEN=ghp-...  # Optional
ANTHROPIC_API_KEY=sk-ant-...  # Optional
```

## File Locations

### Configuration
```
config.yaml          # Main configuration
data/
  working_memory.db  # Session data
  vector_db/         # ChromaDB vectors
  audit.log          # Security audit
```

### Logs
```
logs/
  intelclaw.log      # Application logs
  agent.log          # Agent activity
```

## Integration Points

### Event Bus
Subscribe to internal events:
- `user_input` - User messages
- `agent_response` - Agent outputs
- `tool_execution` - Tool calls
- `screen_capture` - Screen events
- `memory_update` - Memory changes

### Custom Extensions

#### Adding Tools
1. Create tool class extending `BaseTool`
2. Register with `ToolRegistry`
3. Tool becomes available to agents

#### Adding Agents
1. Create agent extending `BaseAgent`
2. Register with `AgentOrchestrator`
3. Update `IntentRouter` for routing

## Support & Feedback

### Reporting Issues
- Log files contain diagnostic info
- Audit log tracks all operations
- Error reports include context

### Contributing
- Tool contributions welcome
- Agent improvements appreciated
- Documentation updates helpful

## Emergency Contacts

### When Things Go Wrong
1. Check `logs/intelclaw.log` for errors
2. Review `data/audit.log` for recent actions
3. Restart via system tray
4. Reset config if needed

### Recovery Options
- `--safe-mode` flag for minimal startup
- `--reset-memory` to clear memory
- `--reset-config` for default settings
