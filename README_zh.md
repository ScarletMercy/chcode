# ChCode

```
 ███████╗  ██╗   ██╗   ███████╗   ██████╗   █████╗     ████████╗
██╔═════╝  ██║   ██║  ██╔═════╝  ██╔═══██╗  ██╔══██╗   ██╔═════╝
██║        ████████║  ██║        ██║   ██║  ██║   ██╗  ████████╗
██║        ██╔═══██║  ██║        ██║   ██║  ██║  ██╔╝  ██╔═════╝
████████╗  ██║   ██║  ████████╗  ╚██████╔╝  █████╔═╝   ████████╗
 ╚══════╝  ╚═╝   ╚═╝   ╚══════╝   ╚═════╝   ╚════╝      ╚══════╝
```

基于终端的 AI 编程代理，使用 LangChain + Typer + Rich 构建。

> **为什么叫 "ChCode"？** 最初的原型是一个 tkinter + LangChain 应用，名为 **chat-agent**（chagent）。当它演变为 CLI 工具后，名字变成了 **ChCode** — chat-agent 遇上 code。

<details>
<summary>📸 chagent — 最初的 tkinter 原型</summary>
<img src="https://raw.githubusercontent.com/ScarletMercy/chcode/main/assets/chagent.png" alt="chagent prototype" width="600"/>
</details>

> 6000+ 行 Python 代码，14 个内置工具，完整会话持久化，Git 感知工作流。

![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[English](README.md)

<img src="https://raw.githubusercontent.com/ScarletMercy/chcode/main/assets/chcode.png" alt="ChCode 主界面" width="800"/>

## 功能特性

### 模型管理

- 兼容**所有 OpenAI 兼容 API**（OpenAI、DeepSeek、Qwen、GLM、Claude 代理等）
- 内置 **ModelScope**、**LongCat** 等主流平台快捷配置
- **ModelScope**：每天 2000 次免费模型调用
- **LongCat**：每天最低 5000 万+ 免费 token
- 首次运行向导，**自动检测环境变量**（扫描 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`ZHIPU_API_KEY`、`ModelScopeToken` 等）
- 原生支持 **reasoning/thinking 模型** — 实时显示思考过程
- 运行时创建 / 编辑 / 切换模型
- 每个模型独立调参（temperature、top_p、top_k、max_completion_tokens、stop_sequences 等）
- **指数退避自动重试**（3/10/30/60s），持续失败时自动切换备用模型

### 视觉与多模态

- 通过 `/vision` 命令独立配置视觉模型（与主模型分离）
- 图片分析，支持**自动媒体编码**和 base64 嵌入
- **视频支持** — 直接将视频发送给视觉模型进行分析（MP4、MOV、AVI、MKV、WebM）
- 大尺寸图片自动缩放
- 支持的图片格式：PNG、JPG、JPEG、GIF、BMP、WebP、TIFF

### 会话与历史

- **持久化会话**，基于 SQLite 的检查点存储（LangGraph）
- 会话列表、切换、重命名、删除
- **上下文压缩** — 接近 token 上限时自动摘要
- 状态栏实时显示**上下文使用量**

### Git 集成

- 编辑消息时工作目录**自动回滚**
- 从任意消息**创建分支**（fork）
- 通过 `/messages` 编辑 / fork / 删除历史消息
- 状态栏显示检查点计数

### 人工审核

- **Common 模式** — 每次工具调用需要确认，编辑操作显示 diff 预览
- **YOLO 模式** — 自动批准所有操作
- 通过 `Tab` 键或 `/mode` 命令切换

### 工作环境隔离

- 每个项目独立的 `.chat/` 目录存放会话、技能、代理
- 全局 `~/.chat/` 存放共享技能和设置
- `/workdir` 切换项目根目录

### 跨平台

- **Windows** — 默认使用 Git Bash，回退到 PowerShell
- **Linux / Mac** — 原生 bash/zsh
- 持久化 Shell 会话，**自动追踪 CWD**

### 终端 UI

- 实时**状态栏** — 上下文使用率、Git 检查点数、ModelScope API 配额
- **流式输出**，逐 token 渲染
- 斜杠命令自动补全
- 彩色工具审核界面，文件编辑显示**行内 diff 预览**

### 可观测性

- **LangSmith 追踪** — 通过 `/langsmith` 命令开关
- 遇到 429 限流时自动禁用追踪并通知用户

### 子代理系统

- 三种内置代理类型：**Explore**（代码库搜索，只读）、**Plan**（架构设计）、**General**（全能力编程）
- **并行执行** — 同时启动多个代理处理独立任务
- 子代理运行在**隔离上下文**中，保护主对话不被上下文污染
- **自定义代理** — 在 `.chat/agents/` 中定义自己的代理类型，配备专属工具和指令

### 技能系统

- 通过 `/skill` 安装 / 删除 / 管理技能
- 技能通过 LangChain 中间件注入系统提示
- 支持项目级和全局技能目录

### ModelScope 限额监控

- 状态栏实时显示 **API 配额**（每日剩余额度、每模型剩余额度）
- 使用 ModelScope 模型时自动启用

## 内置工具（14 个）

| 工具 | 说明 |
|------|------|
| `read` | 读取文件内容，支持行号和偏移量 |
| `write` | 创建或覆盖文件 |
| `edit` | 精准字符串替换编辑现有文件 |
| `glob` | 按文件名模式查找文件 |
| `grep` | 用正则搜索文件内容 |
| `list_dir` | 浏览目录结构 |
| `bash` | 执行 Shell 命令（Git Bash / PowerShell / bash） |
| `load_skill` | 通过中间件动态加载技能指令 |
| `web_fetch` | 抓取 URL 内容并转换为 Markdown |
| `web_search` | 通过 [Tavily](https://tavily.com) 进行网络搜索 |
| `ask_user` | 单选、多选、批量问题与用户交互 |
| `agent` | 启动子代理（explore、plan、general、custom），支持并行执行 |
| `todo_write` | 结构化任务追踪，适用于复杂多步骤工作 |
| `vision` | 通过视觉模型分析图片和视频 |

## 快速开始

### 安装

```bash
# 稳定版（PyPI）— 三选一
pip install chcode        # pip
uv tool install chcode    # uv（推荐）
pipx install chcode       # pipx

# 最新版（GitHub）— 二选一
uv tool install git+https://github.com/ScarletMercy/chcode.git    # uv（推荐）
pipx install git+https://github.com/ScarletMercy/chcode.git       # pipx

# 开发测试（从源码）
git clone https://github.com/ScarletMercy/chcode.git
cd chcode
pip install -e .       # 或: uv sync && uv run chcode
```

### 运行

```bash
# 启动交互式会话
chcode

# 以 YOLO 模式启动
chcode --yolo

# 模型管理
chcode config new    # 添加新模型
chcode config edit   # 编辑当前模型
chcode config switch # 切换模型
```

### 首次运行

首次启动时，ChCode 会：

1. 扫描环境变量中的已知 API Key
2. 引导你完成模型配置
3. 可选配置 Tavily 用于网络搜索

## 命令

| 命令 | 说明 |
|------|------|
| `/new` | 新建会话 |
| `/history` | 浏览和切换会话 |
| `/model` | 模型管理（新建 / 编辑 / 切换） |
| `/vision` | 视觉模型配置 |
| `/messages` | 编辑 / fork / 删除历史消息 |
| `/compress` | 压缩当前会话 |
| `/skill` | 管理技能 |
| `/search` | 配置 Tavily API Key |
| `/workdir` | 切换工作目录 |
| `/mode` | 切换 Common / YOLO 模式 |
| `/git` | 显示 Git 状态 |
| `/langsmith` | 开关 LangSmith 追踪 |
| `/tools` | 列出内置工具 |
| `/quit` | 退出 |

## 快捷键

| 按键 | 操作 |
|------|------|
| `Enter` | 发送消息 |
| `Ctrl+Enter` | 换行 |
| `Tab` | 切换 Common/YOLO 模式（输入为空时） |
| `Ctrl+C` | 中断生成 |

## 为什么不用 MCP？

ChCode 故意不集成 MCP（Model Context Protocol）。**技能 + CLI 工具**的组合覆盖了 95%+ 的真实编程代理场景。技能通过中间件注入结构化、可复用的指令 — 比 MCP 服务器更简单、更快、更轻量。

## 架构

```
chcode/
├── cli.py                  # Typer CLI 入口
├── chat.py                 # REPL 主循环、斜杠命令、人工审核
├── agent_setup.py          # 代理构建、中间件、模型重试与回退
├── config.py               # 模型配置、Tavily、环境变量检测
├── display.py              # Rich 渲染、流式输出、状态栏
├── prompts.py              # 交互式提示（选择/确认/文本）
├── session.py              # 会话管理器（SQLite）
├── skill_manager.py        # 技能安装/删除 UI
├── agents/
│   ├── definitions.py      # 代理类型（explore、plan、general）
│   ├── loader.py           # 从 .chat/agents/ 加载自定义代理
│   └── runner.py           # 子代理执行（含中间件）
└── utils/
    ├── tools.py            # 14 个内置工具
    ├── shell/              # Shell 抽象层（Bash/PowerShell 提供者）
    ├── enhanced_chat_openai.py  # 扩展 ChatOpenAI，支持 reasoning
    ├── git_manager.py      # Git 检查点管理
    ├── skill_loader.py     # 技能发现与加载
    ├── modelscope_ratelimit.py  # ModelScope API 限额监控
    └── tool_result_pipeline.py  # 输出截断与预算控制
```

## 许可证

MIT
