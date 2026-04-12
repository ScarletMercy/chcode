"""
Rich 输出渲染 — Markdown、流式输出、状态栏、消息样式
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich import box
from rich.table import Table
from rich.text import Text
from rich.markup import escape
from rich.live import Live
from rich.layout import Layout
from rich.rule import Rule

if TYPE_CHECKING:
    from langchain_core.messages import AIMessageChunk, ToolMessage, BaseMessage

console = Console()

# ─── 消息渲染 ──────────────────────────────────────────

def render_human(message: str) -> None:
    """渲染用户消息"""
    console.print(
        Panel(
            Markdown(message),
            border_style="blue",
            title="You",
            title_align="right",
            padding=(0, 1),
        )
    )


def render_ai_chunk(content: str) -> None:
    """渲染 AI 回复片段（流式）"""
    console.print(content, end="", style="white")


def render_ai_start() -> None:
    """AI 回复开始"""
    console.print()
    console.print("[bold cyan]AI[/bold cyan]", end="")


def render_ai_end() -> None:
    """AI 回复结束"""
    console.print()


def render_reasoning(reasoning: str) -> None:
    """渲染推理/思考内容（灰色斜体，折叠）"""
    console.print(
        Panel(
            Text(reasoning, style="dim italic"),
            border_style="dim",
            title="Thinking",
            title_align="left",
            padding=(0, 1),
        )
    )


def render_tool_call(name: str, summary: str) -> None:
    """渲染工具调用头部 — 类似 ask_user 的风格"""
    if len(summary) > 120:
        summary = summary[:117] + "..."
    console.print(Text(f"\n[{name}] {summary}", style="bold cyan"))


def render_tool(name: str, content: str) -> None:
    """渲染工具调用结果"""
    # 截断过长内容
    lines = content.split("\n")
    if len(lines) > 50:
        content = "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more lines)"
    console.print(
        Panel(
            Text(content, style="yellow"),
            border_style="yellow",
            title=f"Tool: {name}",
            title_align="left",
            padding=(0, 1),
        )
    )


def render_error(message: str) -> None:
    """渲染错误信息"""
    console.print(Text("Error: ", style="red bold"), Text(message, style="red bold"))


def render_info(message: str) -> None:
    """渲染信息"""
    console.print(f"[cyan]{message}[/cyan]")


def render_success(message: str) -> None:
    """渲染成功信息"""
    console.print(f"[green]{message}[/green]")


def render_warning(message: str) -> None:
    """渲染警告信息"""
    console.print(f"[yellow]{message}[/yellow]")


def render_separator() -> None:
    """渲染分隔线"""
    console.print(Rule(style="dim"))


def render_welcome() -> None:
    """渲染欢迎信息"""
    console.print()
    console.print(
        Panel(
            "[bold]ChCode[/bold] — Terminal-based AI Coding Agent\n"
            "Enter 发送 | Alt+Enter 换行 | /help 查看命令\n"
            "Ctrl+C 中断生成 | /quit 退出",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


# ─── 状态栏 ──────────────────────────────────────────

def render_status(
    workplace: str = "",
    model: str = "",
    tokens: str = "",
    git_status: str = "",
    mode: str = "Common",
) -> None:
    """渲染底部状态栏（带边框）"""
    parts = []
    if workplace:
        short = workplace.replace("\\", "/")
        if len(short) > 40:
            short = "..." + short[-37:]
        parts.append(f"[dim]{short}[/dim]")
    if model:
        parts.append(f"[cyan]{model}[/cyan]")
    if tokens:
        if "[" in tokens:
            parts.append(tokens)
        else:
            parts.append(f"[yellow]{tokens}[/yellow]")
    if git_status:
        parts.append(f"[green]{git_status}[/green]")
    if mode == "Yolo":
        parts.append("[bold red]YOLO[/bold red]")

    if parts:
        content = "  ".join(parts)
        console.print(
            Panel(
                content,
                box=box.ROUNDED,
                border_style="dim",
                padding=(0, 1),
                expand=False,
            )
        )


# ─── 消息列表渲染（加载历史） ─────────────────────────────

MAX_DISPLAY_LINES = 50


def render_conversation(messages: list) -> None:
    """渲染完整对话历史"""
    top_flag = True
    for i, message in enumerate(messages):
        if message.additional_kwargs.get("hide", ""):
            continue
        msg_type = message.type
        content = message.content

        if msg_type == "human":
            if top_flag:
                top_flag = False
            else:
                render_separator()
            render_human(content or "")

        elif msg_type == "ai":
            reasoning = message.additional_kwargs.get("reasoning")
            if reasoning:
                render_reasoning(reasoning)
            if content:
                render_ai_start()
                console.print(Markdown(content))
                render_ai_end()

        elif msg_type == "tool":
            if content:
                render_tool(message.name or "tool", content)

    console.print()


# ─── Token 统计 ──────────────────────────────────────────

def get_token_text(messages: list) -> str:
    """从消息列表提取最新 AI 消息的 token 统计"""
    total = input_t = output_t = 0
    for message in reversed(messages):
        from langchain_core.messages import AIMessage
        if isinstance(message, AIMessage):
            if message.additional_kwargs.get("error"):
                continue
            usage = message.usage_metadata
            if usage:
                input_t = usage.get("input_tokens", 0)
                output_t = usage.get("output_tokens", 0)
                total = usage.get("total_tokens", 0)
                break
    return f"总: {total} | 输入: {input_t} | 输出: {output_t}"


# ─── 上下文用量 ──────────────────────────────────────────

def _format_tokens(n: int) -> str:
    """格式化 token 数：123456 → 123.5K"""
    if n >= 1000:
        return f"{n / 1000:.1f}K"
    return str(n)


def get_context_usage_text(messages: list, max_context: int) -> str:
    """
    从消息列表计算上下文占用，返回带样式的文本。

    取最后一次 AIMessage 的 input_tokens 作为上下文快照
    （因为每次请求的 input_tokens 包含了完整上下文）。
    """
    input_tokens = 0
    for message in reversed(messages):
        from langchain_core.messages import AIMessage
        if isinstance(message, AIMessage):
            usage = message.usage_metadata
            if usage and usage.get("input_tokens"):
                input_tokens = usage["input_tokens"]
                break

    if input_tokens == 0:
        return ""

    pct = input_tokens / max_context
    used_str = _format_tokens(input_tokens)
    max_str = _format_tokens(max_context)
    pct_str = f"{pct * 100:.0f}%"

    if pct < 0.7:
        style = "yellow"
    elif pct < 0.9:
        style = "bold yellow"
    else:
        style = "bold red"

    return f"[{style}]{used_str}/{max_str} {pct_str}[/{style}]"
