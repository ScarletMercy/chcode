"""
Rich 输出渲染 — Markdown、流式输出、状态栏、消息样式
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.live import Live

import asyncio
import contextvars
import threading
import time

_subagent_count = 0
_subagent_count_lock = threading.Lock()
_subagent_parallel = False

_current_agent_tag: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_agent_tag", default=None
)
_agent_progress: dict[str, dict] = {}
_agent_progress_lock = threading.Lock()
_progress_live: Live | None = None
_progress_task: asyncio.Task | None = None

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
    if _subagent_parallel or _subagent_count > 0:
        return
    console.print(content, end="", style="white")


def render_ai_start():
    """AI 回复开始"""
    global _subagent_parallel
    # 先完成并清理之前的进度显示
    _finalize_progress()
    _subagent_parallel = False
    with _agent_progress_lock:
        _agent_progress.clear()
    if _subagent_count > 0:
        return
    console.print()


def render_ai_end() -> None:
    """AI 回复结束"""
    if _subagent_parallel or _subagent_count > 0:
        return
    console.print()


def render_reasoning(reasoning: str) -> None:
    """渲染推理/思考内容（灰色斜体，折叠）"""
    if _subagent_parallel or _subagent_count > 0:
        return
    console.print(
        Panel(
            Text(reasoning, style="dim italic"),
            border_style="dim",
            title="Thinking",
            title_align="left",
            padding=(0, 1),
        )
    )


def _start_progress():
    global _progress_live
    if _progress_live is None:
        _progress_live = Live("", transient=True, console=console, refresh_per_second=1)
        _progress_live.start()


def _update_progress():
    if not _progress_live:
        return
    with _agent_progress_lock:
        if not _agent_progress:
            _progress_live.update("")
            return
        now = time.time()
        lines = []
        for tag, info in _agent_progress.items():
            elapsed = int(now - info["start"])
            timeout = info.get("timeout", 300)

            if info.get("failed"):
                lines.append(f"  [red]\u274c {tag}: \u8d85\u65f6 ({timeout}s)[/red]")
            elif info.get("done"):
                lines.append(f"  [green]\u2705 {tag}: done ({elapsed}s)[/green]")
            else:
                remaining = max(0, timeout - elapsed)
                pct = min(elapsed / timeout, 1.0) if timeout > 0 else 1.0
                bar_len = int(pct * 16)
                bar = "\u2588" * bar_len + "\u2591" * (16 - bar_len)
                lines.append(f"  {tag}: {remaining}s \u5269\u4f59 [{bar}]")
    _progress_live.update("\n".join(lines))


async def _progress_updater():
    """定期更新进度显示的后台任务"""
    try:
        while True:
            await asyncio.sleep(1)
            if _progress_live is None:
                break
            _update_progress()
    except asyncio.CancelledError:
        pass  # 优雅处理取消


def _finalize_progress():
    """停止进度显示，打印最终状态并清理资源"""
    global _progress_live, _progress_task

    # 先取消更新任务
    if _progress_task is not None and not _progress_task.done():
        _progress_task.cancel()
        _progress_task = None

    # 停止 Live（transient=True 会自动清除显示内容）
    if _progress_live is not None:
        _progress_live.stop()
        _progress_live = None

    # 清空进度数据
    with _agent_progress_lock:
        _agent_progress.clear()


def render_tool_call(name: str, summary: str) -> None:
    tag = _current_agent_tag.get()
    if tag:
        with _agent_progress_lock:
            if tag in _agent_progress:
                _agent_progress[tag]["calls"] += 1
    if len(summary) > 120:
        summary = summary[:117] + "..."
    if name == "agent":
        console.print(Text(f"\n[{name}] {summary}", style="bold cyan"))
        return
    if _subagent_parallel or _subagent_count >= 2:
        return
    if _subagent_count == 1:
        console.print(Text(f"  [{name}] {summary}", style="dim cyan"))
        return
    console.print(Text(f"\n[{name}] {summary}", style="bold cyan"))


def render_tool(name: str, content: str) -> None:
    """渲染工具调用结果"""
    if _subagent_parallel or _subagent_count > 0:
        return
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
    if _subagent_parallel or _subagent_count > 0:
        return
    console.print(Text("Error: ", style="red bold"), Text(message, style="red bold"))


def render_info(message: str) -> None:
    """渲染信息"""
    if _subagent_parallel or _subagent_count > 0:
        return
    console.print(f"[cyan]{message}[/cyan]")


def render_success(message: str) -> None:
    """渲染成功信息"""
    if _subagent_parallel or _subagent_count > 0:
        return
    console.print(f"[green]{message}[/green]")


def render_warning(message: str) -> None:
    """渲染警告信息"""
    if _subagent_parallel or _subagent_count > 0:
        return
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
            "Enter 发送 | Ctrl+Enter 换行 | /help 查看命令\n"
            "Ctrl+C 中断生成 | Tab 切换模式 | /quit 退出",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


# ─── 消息列表渲染（加载历史） ─────────────────────────────


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
