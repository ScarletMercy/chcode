# ChCode

```
 ███████╗  ██╗   ██╗   ███████╗   ██████╗   █████╗     ████████╗
██╔═════╝  ██║   ██║  ██╔═════╝  ██╔═══██╗  ██╔══██╗   ██╔═════╝
██║        ████████║  ██║        ██║   ██║  ██║   ██╗  ████████╗
██║        ██╔═══██║  ██║        ██║   ██║  ██║  ██╔╝  ██╔═════╝
████████╗  ██║   ██║  ████████╗  ╚██████╔╝  █████╔═╝   ████████╗
 ╚══════╝  ╚═╝   ╚═╝   ╚══════╝   ╚═════╝   ╚════╝      ╚══════╝
```

Terminal-based AI coding agent, built with LangChain + Typer + Rich.

> **Why "ChCode"?** The original prototype was a tkinter + LangChain app called **chat-agent** (chagent). When it evolved into a CLI tool, the name became **ChCode** — chat-agent, meet code.

<details>
<summary>📸 chagent — the original tkinter prototype</summary>
<img src="https://raw.githubusercontent.com/ScarletMercy/chcode/main/assets/chagent.png" alt="chagent prototype" width="600"/>
</details>

> 15 built-in tools, full session persistence, git-aware workflow.

![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[中文文档](README.md)

<img src="https://raw.githubusercontent.com/ScarletMercy/chcode/main/assets/chcode.png" alt="ChCode main interface" width="800"/>

## Features

### Model Management

- Compatible with **all models supporting the OpenAI API** (OpenAI, DeepSeek, Qwen, GLM, Claude via proxy, etc.)
- Built-in quick setup for **ModelScope** (credits system, 200-250 Magicubes/day ≈ 100-500 model calls)
- Native **reasoning/thinking model** support — thinking tokens displayed in real time
- Create / edit / switch models at runtime
- Per-model hyperparameter tuning (temperature, top_p, top_k, max_completion_tokens, etc.)
- Automatic **retry with exponential backoff** (3/10/30/60s) and fallback model switching on persistent failure

### Vision & Multimodal

- Dedicated vision model configuration via `/vision` command (independent from main model)
- Vision understanding with **automatic media encoding** and base64 embedding
- Multimodal main models disable the vision tool and embed media files automatically; text main models understand media via the vision tool

### Session & History

- **Persistent sessions** with SQLite-backed checkpoints (LangGraph)
- Session list, switch, rename, delete
- **Context compression** — auto-summarize when approaching token limit
- Real-time **context usage display** in status bar

### Project Memory (CHCODE.md)

- **Auto-creates CHCODE.md** on first run in a project directory, accumulating cross-session project knowledge (common commands, package manager, coding standards, prohibitions, verification workflow, pitfalls, etc.)
- **Auto-migrates** a non-empty `CLAUDE.md` into an empty `CHCODE.md` (original file kept); afterwards `CHCODE.md` always takes precedence
- Injected into the conversation context as a `<system-reminder>` meta message every session; external edits reach the model next turn as an incremental diff reminder
- Save memories via the dedicated `update_memory` tool (approval required in Common mode); entries are enforced to be **brief, action-related, and constraint-phrased** — no API docs, changelogs, or empty slogans
- **Three-layer capacity control** — above 8K chars the AI is nudged to clean outdated entries; above 16K injection is truncated and appends are blocked (clean-before-add); the file itself is never silently modified

### Message Management & Shadow Repo (Git-aware)

- **Shadow repo isolation** — working-directory rollbacks caused by history edits and branches are performed in an isolated shadow repo, fully separated from your working repo and never polluting real code
- **Edit** — editing a history message automatically rolls the working directory back to that message's state
- **Branch** — fork a new session branch and a new directory branch from any message, then roll back the working directory the same way editing does
- **Delete** — delete history messages you no longer need; the working directory is unaffected
- All of the above via the `/messages` command

### Human-in-the-Loop

- **Common mode** — every risky tool call requires approval, with diff preview for edits. Only Explore and Plan sub-agents available.
- **YOLO mode** — auto-approve everything. All sub-agents available including General-purpose.
- Toggle with `Tab` key or `/mode` command
- Available sub-agents update dynamically when switching modes

### Work Environment Isolation

- Per-project `.chat/` directory for sessions, skills, agents
- Global `~/.chat/` for shared skills and settings
- `/workdir` to switch project root

### Cross-Platform

- **Windows** — defaults to Git Bash, falls back to PowerShell
- **Linux / Mac** — native bash/zsh
- Persistent shell sessions with **automatic CWD tracking**

### Rich Terminal UI

- Real-time **status bar** — context usage %, git status indicator
- **Streaming output** with token-by-token rendering
- Slash command auto-completion
- Color-coded tool approval UI with **inline diff preview** for file edits

### Observability

- **LangSmith tracing** — toggle on/off via `/langsmith` command
- Auto-disable tracing on 429 rate limit with user notification

### Sub-Agent System

- Three built-in agent types: **Explore** (codebase search, read-only), **Plan** (architecture design, read-only), **General-purpose** (full-capability coding)
- **Mode-aware availability** — Common mode: Explore + Plan only; YOLO mode: all three + custom agents
- **Parallel execution** — launch multiple agents concurrently for independent tasks, with live spinner progress display
- Sub-agents run with **isolated context**, protecting the main conversation from context pollution
- Read-only agents (Explore, Plan) have **bash command restrictions** to prevent accidental modifications
- **Custom agents** — define your own agent types in `.chat/agents/` with dedicated tools and instructions

### Skill System

- Install / delete / manage skills via `/skill`
- **Skill marketplace** — search and install skills from remote hubs (clawhub.ai / skills.sh), no LLM involved
- Skills are injected into system prompt via LangChain middleware
- Supports project-level and global skill directories

## Built-in Tools (15)

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
| `agent` | Launch sub-agents (Explore, Plan, General-purpose in YOLO mode, custom), supports parallel execution |
| `todo_write` | Structured task tracking for complex multi-step work |
| `update_memory` | Append / replace memory entries in CHCODE.md (approval + line-level diff preview in Common mode) |
| `vision` | Analyze images and videos via vision models |

## Quick Start

### Install

```bash
# Stable release (PyPI) — choose one
pip install chcode        # pip
uv tool install chcode    # uv (recommended)
pipx install chcode       # pipx

# Latest version (GitHub) — choose one
uv tool install git+https://github.com/ScarletMercy/chcode.git    # uv (recommended)
pipx install git+https://github.com/ScarletMercy/chcode.git       # pipx

# Development (from source)
git clone https://github.com/ScarletMercy/chcode.git
cd chcode
pip install -e .       # or: uv sync && uv run chcode
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
| `/history` | Browse and switch sessions |
| `/model` | Model management (new / edit / switch) |
| `/vision` | Visual model configuration |
| `/messages` | Edit / fork / delete history messages |
| `/compress` | Compress current session |
| `/skill` | Manage skills |
| `/search` | Configure Tavily API key |
| `/workdir` | Switch working directory |
| `/mode` | Toggle Common / YOLO mode |
| `/git` | Show git status |
| `/langsmith` | Toggle LangSmith tracing |
| `/tools` | List built-in tools |
| `/quit` | Exit |

## Keybindings

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Ctrl+Enter` | New line |
| `Tab` | Toggle Common/YOLO mode (when input empty) |
| `Ctrl+C` | Interrupt generation |

## Why No MCP?

ChCode intentionally does not integrate MCP (Model Context Protocol). The combination of **Skills + CLI tools** covers 95%+ of real-world coding agent scenarios. Skills provide structured, reusable instructions injected via middleware — simpler, faster, and more lightweight than MCP servers.

## Architecture

```
chcode/
├── cli.py                  # Typer CLI entry
├── chat.py                 # REPL main loop, slash commands, HITL
├── agent_setup.py          # Agent construction, middleware, model retry with fallback
├── config.py               # Model config, Tavily, env detection
├── display.py              # Rich rendering, streaming, status bar
├── i18n.py                 # i18n core
├── prompts.py              # Interactive prompts (select/confirm/text)
├── strings.py              # UI strings catalog (zh / en)
├── vision_config.py        # Vision model configuration
├── agents/
│   ├── definitions.py      # Agent types (explore, plan, general)
│   ├── loader.py           # Load custom agents from .chat/agents/
│   └── runner.py           # Sub-agent execution with middleware
└── utils/
    ├── tools.py            # Built-in tools
    ├── enhanced_chat_openai.py  # Extended ChatOpenAI with reasoning support
    ├── frontmatter.py      # YAML Frontmatter parser
    ├── git_checker.py      # Git availability check
    ├── git_manager.py      # Git checkpoint management
    ├── json_utils.py       # JSON atomic read/write + mtime cache
    ├── multimodal.py       # Multimodal model detection and media encoding
    ├── project_memory.py   # CHCODE.md project memory (migration/injection/capacity)
    ├── session.py          # Session manager (SQLite)
    ├── skill_hub.py        # Skill marketplace (search & install from remote hubs)
    ├── skill_loader.py     # Skill discovery and loading
    ├── skill_manager.py    # Skill install/delete UI
    ├── text_utils.py       # Message content text extraction
    ├── tool_result_pipeline.py  # Output truncation and budget enforcement
    └── shell/
        ├── provider.py     # Shell provider abstraction (Bash/PowerShell)
        ├── session.py      # Interactive shell session
        ├── output.py       # Output capture and temp files
        ├── result.py       # Execution result dataclass
        └── semantics.py    # Output semantic analysis (error detection etc.)
```

## License

MIT
