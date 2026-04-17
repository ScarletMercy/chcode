# ChCode

```
 ██████╗  ██╗  ██╗   ██████╗   ██████╗   █████╗    ████████╗
██╔════╝  ██║  ██║  ██╔════╝  ██╔═══██╗  ██╔═██╗   ██╔═════╝
██║       ███████║  ██║       ██║   ██║  ██║  ██╗  ████████╗
██║       ██╔══██║  ██║       ██║   ██║  ██║ ██╔╝  ██╔═════╝
███████╗  ██║  ██║  ███████╗  ╚██████╔╝  █████╔╝   ████████╗
 ╚═════╝  ╚═╝  ╚═╝   ╚═════╝   ╚═════╝   ╚════╝     ╚══════╝
```

Terminal-based AI coding agent, built with LangChain + Typer + Rich.

> **Why "ChCode"?** The original prototype was a tkinter + LangChain app called **chat-agent** (chagent). When it evolved into a CLI tool, the name became **ChCode** — chat-agent, meet code.

<details>
<summary>📸 chagent — the original tkinter prototype</summary>
<img src="https://raw.githubusercontent.com/ScarletMercy/chcode/main/assets/chagent.png" alt="chagent prototype" width="600"/>
</details>

> 5000+ lines of Python, 13 built-in tools, full session persistence, git-aware workflow.

![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Demo

https://github.com/ScarletMercy/chcode/blob/main/assets/test.mp4

## Features

### Model Management

- Compatible with **all OpenAI-compatible APIs** (OpenAI, DeepSeek, Qwen, GLM, Claude via proxy, etc.)
- First-run wizard with **env auto-detection** (scans `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `ZHIPU_API_KEY`, etc.)
- Create / edit / switch models at runtime
- Per-model hyperparameter tuning (temperature, top_p, top_k, max_tokens, stop_sequences, etc.)

### Session & History

- **Persistent sessions** with SQLite-backed checkpoints (LangGraph)
- Session list, switch, delete
- **Context compression** — auto-summarize when approaching token limit
- Real-time **context usage display** in status bar

### Git Integration

- Working directory **rolls back with message edits**
- Create **branches from any message** (fork)
- Edit / fork / delete history messages via `/messages`
- Checkpoint counter in status bar

### Human-in-the-Loop

- **Common mode** — every tool call requires approval, with diff preview for edits
- **YOLO mode** — auto-approve everything
- Toggle with `Tab` key or `/mode` command

### Work Environment Isolation

- Per-project `.chat/` directory for sessions, skills, agents
- Global `~/.chat/` for shared skills and settings
- `/workdir` to switch project root

### Cross-Platform

- **Windows** — defaults to Git Bash, falls back to PowerShell
- **Linux / Mac** — native bash/zsh
- Persistent shell sessions with **automatic CWD tracking**

### Skill System

- Install / delete / manage skills via `/skill`
- Skills are injected into system prompt via LangChain middleware
- Supports project-level and global skill directories

## Built-in Tools (13)

| Tool | Description |
|------|-------------|
| `read` | Read file content with line numbers and offset |
| `write` | Create or overwrite files |
| `edit` | Surgical string replacement in existing files |
| `glob` | Find files by name pattern |
| `grep` | Search file contents with regex |
| `list_dir` | Browse directory structure |
| `bash` | Execute shell commands (Git Bash / PowerShell / bash) |
| `load_skill` | Dynamically load skill instructions via middleware |
| `web_fetch` | Fetch and convert URL content to markdown |
| `web_search` | Web search via [Tavily](https://tavily.com) |
| `ask_user` | Single-select, multi-select, batch questions for user interaction |
| `agent` | Launch sub-agents (explore, plan, general-purpose), supports parallel execution |
| `todo_write` | Structured task tracking for complex multi-step work |

## Quick Start

### Install

```bash
# Option 1: Install globally with uv (recommended)
uv tool install git+https://github.com/ScarletMercy/chcode.git

# Option 2: Clone and install with uv
git clone https://github.com/ScarletMercy/chcode.git
cd chcode
uv sync
uv run chcode

# Option 3: Install globally with pipx
pipx install git+https://github.com/ScarletMercy/chcode.git

# Option 4: Clone and install with pip
git clone https://github.com/ScarletMercy/chcode.git
cd chcode
pip install -e .
```

### Run

```bash
# Start interactive session
chcode

# Start in YOLO mode
chcode --yolo

# Model management
chcode config new    # add new model
chcode config edit   # edit current model
chcode config switch # switch model
```

### First Run

On first launch, ChCode will:

1. Scan environment variables for known API keys
2. Guide you through model configuration
3. Optionally configure Tavily for web search

## Commands

| Command | Description |
|---------|-------------|
| `/new` | Start new session |
| `/model` | Model management (new / edit / switch) |
| `/messages` | Edit / fork / delete history messages |
| `/history` | Browse and switch sessions |
| `/compress` | Compress current session |
| `/git` | Show git status |
| `/search` | Configure Tavily API key |
| `/mode` | Toggle Common / YOLO mode |
| `/workdir` | Switch working directory |
| `/tools` | List built-in tools |
| `/skill` | Manage skills |
| `/quit` | Exit |

## Keybindings

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Ctrl+Enter` | New line |
| `Tab` | Toggle Common/YOLO mode (when input empty) |
| `Ctrl+C` | Interrupt generation |

## Why No MCP?

ChCode intentionally does not integrate MCP (Model Context Protocol). The combination of **Skills + CLI tools** covers 95%+ of real-world coding agent scenarios. Skills provide structured, reusable instructions injected via middleware — simpler, faster, and more portable than MCP servers.

## Architecture

```
chcode/
├── cli.py                  # Typer CLI entry
├── chat.py                 # REPL main loop, slash commands, HITL
├── agent_setup.py          # Agent construction, middleware chain
├── config.py               # Model config, Tavily, env detection
├── display.py              # Rich rendering, streaming, status bar
├── prompts.py              # Interactive prompts (select/confirm/text)
├── session.py              # Session manager (SQLite)
├── skill_manager.py        # Skill install/delete UI
├── agents/
│   ├── definitions.py      # Agent types (explore, plan, general)
│   ├── loader.py           # Load custom agents from .chat/agents/
│   └── runner.py           # Sub-agent execution with middleware
└── utils/
    ├── tools.py            # 13 built-in tools
    ├── shell/              # Shell abstraction (Bash/PowerShell providers)
    ├── enhanced_chat_openai.py  # Extended ChatOpenAI with reasoning support
    ├── git_manager.py      # Git checkpoint management
    ├── skill_loader.py     # Skill discovery and loading
    └── tool_result_pipeline.py  # Output truncation and budget enforcement
```

## License

MIT
