# MEMORY.md - IntelCLaw Memory Architecture

## Memory Tiers

### 1. Short-Term Memory (STM)
**Duration**: Current conversation session
**Capacity**: Last 50 messages
**Purpose**: Maintain conversation context

Contents:
- Recent messages and responses
- Current task state
- Immediate context

### 2. Working Memory
**Duration**: Persistent across sessions
**Storage**: SQLite database
**Purpose**: Track ongoing work and preferences

Contents:
- Active tasks and their status
- User preferences
- Session history
- Frequently used tools

### 3. Long-Term Memory (LTM)
**Duration**: Indefinite (with optional cleanup)
**Storage**: Mem0 + ChromaDB
**Purpose**: Store important facts and experiences

Contents:
- User facts and preferences
- Past conversation summaries
- Learned patterns
- Important events

## What I Remember

### Always Remember
- Your name and preferences
- Important facts you've shared
- Task completions and outcomes
- Tool usage patterns

### Contextual Memory
- Screen observations (summarized)
- Application usage patterns
- Workflow sequences

### Never Remember
- Passwords and credentials
- Financial information
- Content from privacy-filtered apps
- Sensitive personal data

## Memory Operations

### Storage
When you share important information, I store it appropriately:
- Facts → Long-term memory
- Preferences → Working memory
- Context → Short-term memory

### Retrieval
When responding, I recall relevant information:
- Search semantic memory for related facts
- Check recent conversation context
- Apply learned preferences

### Forgetting
You can ask me to forget:
- "Forget what I told you about X"
- "Clear our conversation history"
- "Reset all memories about Y"

## Privacy Controls

### Excluded Patterns
Applications matching these patterns are never captured:
- `*password*`
- `*bank*`
- `*1password*`
- `*keepass*`

### User Controls
- View stored memories
- Delete specific memories
- Adjust retention settings
- Export/import memory data

## Memory Quality

I maintain memory quality through:
- **Deduplication**: Avoiding redundant storage
- **Consolidation**: Summarizing old conversations
- **Pruning**: Removing outdated information
- **Validation**: Checking fact accuracy

## Your Data Rights

You always have the right to:
- Know what I've stored about you
- Correct inaccurate information
- Delete any or all memories
- Export your data
