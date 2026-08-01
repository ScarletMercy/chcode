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
from rich._spinners import SPINNERS

from chcode.i18n import t

import asyncio
import contextvars
import threading
import time

_subagent_count = 0
_subagent_count_lock = threading.Lock()
_subagent_parallel = False

# ─── AI 正文流式渲染（stable/unstable 分块） ──
# 原理：用 markdown-it 把累计文本切成块级 token。除"最后一个块"外都视作 stable
# (已完成)，用 console.print(Markdown) 落版进滚动历史，永不重绘。只有"最后一个
# 块"是 unstable(正在增长)，用 Live 原地重绘实时排版。单个 markdown block 通常
# 只有几行，Live 的光标重绘(CURSOR_UP)能正常回到段首，不超屏、不重复。
#
# ⚠️ 串行前提：以下全局状态只在"主 agent 流式渲染"这一单串行流程中读写
# （render_ai_chunk/_flush_unstable/render_ai_start/render_ai_end/force_reset）。
# 子 agent 走 ainvoke（非流式、独立 agent），不触碰这些状态。若未来引入并行
# 主 agent 流式输出，需把这些状态重做成实例/上下文局部，否则会竞态。
_committed_text: str = ""        # 已落版的 stable 文本（单调递增）
_unstable_text: str = ""         # 当前 unstable 块的文本
_full_text: str = ""             # 完整累计文本（只增不减），分块基于此
_live: Live | None = None        # 仅渲染 unstable 块的 Live
_md_parser = None                # 延迟初始化的 markdown-it 解析器


def _get_md_parser():
    global _md_parser
    if _md_parser is None:
        from markdown_it import MarkdownIt
        _md_parser = MarkdownIt("commonmark")
    return _md_parser


def _split_stable(text: str) -> tuple[str, str]:
    """把累计文本分成 (stable, unstable)。

    stable = 最后一个内容块之前的所有块（已完成，可安全落版）。
    unstable = 最后一个内容块及其之后（正在增长，用 Live 重绘）。
    边界只增不减（单调），stable 部分只会增长不会回退。
    未闭合代码块被 markdown-it 识别为单个 fence token，边界始终安全。
    """
    if not text:
        return "", ""
    tokens = _get_md_parser().parse(text, {})
    # 找最后一个"顶层"块级起始 token（level=0）。只认顶层块，跳过 list_item_open
    # 和列表项内的 paragraph_open 等嵌套子元素，否则列表会被拆成单项独立落版，
    # 导致每个项前后多出空行（Rich 把每个独立 Markdown 当孤立块渲染）。
    last_open = -1
    for idx, tok in enumerate(tokens):
        if tok.level != 0:
            continue
        if tok.type in ("fence", "hr", "html_block", "code_block") or tok.type.endswith("_open"):
            last_open = idx
    if last_open == -1 or tokens[last_open].map is None:
        # 还没形成任何完整块，全是 unstable
        return "", text
    boundary_line = tokens[last_open].map[0]
    # 用字符偏移切分（而非 split/join），完整保留段落间的 \n\n 空行分隔。
    # split/join 会丢失末尾空行，导致落版内容与原文换行不一致。
    boundary_offset = _line_start_offset(text, boundary_line)
    return text[:boundary_offset], text[boundary_offset:]


def _line_start_offset(text: str, line: int) -> int:
    """返回第 `line` 行（0-indexed）起始的字符偏移。"""
    if line <= 0:
        return 0
    current_line = 0
    for i, ch in enumerate(text):
        if ch == "\n":
            current_line += 1
            if current_line == line:
                return i + 1
    return len(text)


def _ends_with_code_block(text: str) -> bool:
    """判断 text 末尾是否是闭合的代码块（```...```）。

    代码块 Rich 渲染自带 1 个尾空行，落版后无需再补空行，否则会多出空行。
    """
    stripped = text.rstrip()
    if not stripped.endswith("```"):
        return False
    return stripped.count("```") % 2 == 0


def _starts_with_leading_blank_block(text: str) -> bool:
    """判断 text 开头是否是 Rich 渲染时自带前置空行的块。

    列表(-/*/+)、有序列表(数字.)、代码块(```)、引用(>)、表格(|)在 Rich 渲染时
    自带 1~2 个前置空行。若下一个块(unstable)是这些类型，当前落版无需补空行，
    否则会多出空行。
    """
    stripped = text.lstrip()
    if not stripped:
        return False
    first = stripped[0]
    if first in ("-", "*", "+", ">", "|"):
        return True
    # 有序列表：数字 + .
    if first.isdigit():
        rest = stripped.lstrip("0123456789")
        if rest.startswith("."):
            return True
    # 代码块：流式中 ``` 可能只到了 1~2 个反引号，都算代码块开始
    if stripped.startswith("`"):
        return True
    return False

_current_agent_tag: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_agent_tag", default=None
)
_agent_progress: dict[str, dict] = {}
_agent_progress_lock = threading.Lock()
_progress_live: Live | None = None
_progress_task: asyncio.Task | None = None

_DOTS = SPINNERS["dots"]["frames"]
_DOTS_MS = SPINNERS["dots"]["interval"]

if TYPE_CHECKING:
    pass

console = Console()


def _suppress_in_subagent(fn):
    """Decorator: suppress output when subagents are active (parallel or count > 0)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _subagent_parallel or _subagent_count > 0:
            return
        return fn(*args, **kwargs)

    return wrapper


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


def _commit_stable_increment(stable: str, unstable: str) -> None:
    """落版 stable 相对 _committed_text 的新增部分（含空行补偿）。

    调用方先 _split_stable 得到 (stable, unstable)，本函数把 stable 中比上次
    多出来的部分用 Markdown 落版进滚动历史。空行补偿规则：
      - rstrip 去掉末尾段落分隔 \\n\\n。Rich 把每个独立 Markdown 当孤立块渲染，
        末尾换行被吃掉，块间紧贴。rstrip 后段落/标题/列表尾空行均为 0，统一补
        1 个空行恢复间距。但两种情况不补，避免多空行：
          1) new_part 以代码块结尾：代码块自带 1 尾空行
          2) unstable 以列表/代码/引用/表格开头：这些块自带前置空行
    """
    global _committed_text
    if len(stable) > len(_committed_text):
        new_part = stable[len(_committed_text):]
        if new_part.strip():
            console.print(Markdown(new_part.rstrip()))
            if (
                unstable.strip()
                and not _ends_with_code_block(new_part)
                and not _starts_with_leading_blank_block(unstable)
            ):
                console.print()
        _committed_text = stable


@_suppress_in_subagent
def render_ai_chunk(content: str) -> None:
    """渲染 AI 回复片段（流式）— stable/unstable 分块。

    每收到一个 chunk 就重新分块：若 stable 增长了，把新增部分用 Markdown 落版
    （进滚动历史，永不重绘）；Live 只渲当前的 unstable 块（几行，不超屏）。
    """
    global _unstable_text, _full_text
    if _live is None:  # 子 agent 已被装饰器拦掉，此处防御
        console.print(content, end="", style="white")
        return
    # 累计完整文本（只增不减），分块基于它
    _full_text += content
    stable, unstable = _split_stable(_full_text)
    _commit_stable_increment(stable, unstable)
    _unstable_text = unstable
    # Live 只渲 unstable（几行），光标重绘不超屏
    _live.update(Markdown(_unstable_text))


def _flush_unstable() -> None:
    """reasoning 插入前，落版已完成的 stable 增量，unstable 留在 Live 继续渲染。

    reasoning 的 console.print 会输出在 Live 区域上方（Rich Live 的固有行为），
    无需清空 Live。关键是不能落版 unstable——若 unstable 是未闭合代码块，
    落版后代码块闭合时会被 _split_stable 重新切进 stable，导致同一代码块被拆成
    两次落版（首次未闭合、二次脱离代码块上下文），渲染错乱。
    """
    global _unstable_text, _full_text
    stable, unstable = _split_stable(_full_text)
    _commit_stable_increment(stable, unstable)
    _unstable_text = unstable


def render_ai_start():
    """AI 回复开始"""
    global _subagent_parallel, _committed_text, _unstable_text, _full_text, _live
    if _subagent_count == 0:
        _finalize_progress()
        with _agent_progress_lock:
            _agent_progress.clear()
    _subagent_parallel = False
    if _subagent_count > 0:
        return
    console.print()
    _committed_text = ""
    _unstable_text = ""
    _full_text = ""
    # Live 只渲 unstable 块（单个 markdown block，几行），ellipsis 兜底，
    # 绝不用 visible（超屏时光标回不到段首会崩成追加重复）。
    _live = Live(
        Markdown(""),
        console=console,
        refresh_per_second=12,
        transient=False,
        vertical_overflow="ellipsis",
    )
    _live.start()


@_suppress_in_subagent
def render_ai_end() -> None:
    """AI 回复结束 — 关 Live（unstable 的最终帧自然保留，无需额外落版）"""
    global _live, _committed_text, _unstable_text, _full_text
    if _live is not None:
        # transient=False 时 stop 保留最终帧。Live 的最后一帧已是 unstable 的完整
        # Markdown 渲染（内容不再变化），它就是最终态，留在屏幕上即可。
        # 切勿再 console.print(unstable)：那会与 Live 残留帧重复（最后一行重复）。
        _live.stop()
        _live = None
        _committed_text = ""
        _unstable_text = ""
        _full_text = ""
    console.print()


@_suppress_in_subagent
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


def _start_progress():
    global _progress_live
    if _progress_live is None:
        _live_console = Console(file=console.file)
        _progress_live = Live("", transient=False, console=_live_console, refresh_per_second=12)
        _progress_live.start()


def _update_progress():
    if not _progress_live:
        return
    with _agent_progress_lock:
        if not _agent_progress:
            _progress_live.update("")
            return
        frame = _DOTS[int(time.time() * 1000 / _DOTS_MS) % len(_DOTS)]
        lines = []
        for tag, info in _agent_progress.items():
            calls = info.get("calls", 0)
            calls_str = f" ({calls} calls)" if calls else ""
            if info.get("failed"):
                lines.append(f"  [red]✗ {tag}[/red]{calls_str}")
            elif info.get("done"):
                lines.append(f"  [green]✓ {tag}[/green]{calls_str}")
            else:
                lines.append(f"  [cyan]{frame}[/cyan] {tag}{calls_str}")
    _progress_live.update("\n".join(lines))


async def _progress_updater():
    try:
        while True:
            await asyncio.sleep(_DOTS_MS / 1000)
            if _progress_live is None:
                break
            _update_progress()
    except asyncio.CancelledError:
        pass


async def _result_spinner_updater():
    try:
        while True:
            await asyncio.sleep(_DOTS_MS / 1000)
            if _progress_live is None:
                break
            frame = _DOTS[int(time.time() * 1000 / _DOTS_MS) % len(_DOTS)]
            _progress_live.update(f"  [cyan]{frame}[/cyan] {t('display.organizing')}")
    except asyncio.CancelledError:
        pass


def _start_result_spinner():
    """单 agent 完成后，显示整理结果的加载圈"""
    global _progress_live, _progress_task
    if _progress_live is None:
        _live_console = Console(file=console.file)
        _progress_live = Live("", transient=False, console=_live_console, refresh_per_second=12)
        _progress_live.start()
    if _progress_task is None or _progress_task.done():
        _progress_task = asyncio.ensure_future(_result_spinner_updater())


def _finalize_progress():
    """停止进度显示并清理资源"""
    global _progress_live, _progress_task

    if _progress_task is not None and not _progress_task.done():
        _progress_task.cancel()
        _progress_task = None

    if _progress_live is not None:
        _update_progress()
        _progress_live.stop()
        _progress_live = None

    with _agent_progress_lock:
        _agent_progress.clear()


def force_reset_display() -> None:
    """异常退出时强制重置所有显示状态"""
    global _subagent_count, _subagent_parallel, _live, _committed_text, _unstable_text, _full_text
    _subagent_count = 0
    _subagent_parallel = False
    if _live is not None:
        try:
            _live.stop()
        except Exception:
            pass
        _live = None
    _committed_text = ""
    _unstable_text = ""
    _full_text = ""
    console.quiet = False
    _finalize_progress()


def render_tool_call(name: str, summary: str) -> None:
    tag = _current_agent_tag.get()
    if tag:
        with _agent_progress_lock:
            if tag in _agent_progress:
                _agent_progress[tag]["calls"] += 1
        return
    if _subagent_parallel:
        return
    if len(summary) > 120:
        summary = summary[:117] + "..."
    if _subagent_count == 1:
        console.print(Text(f"  [{name}] {summary}", style="dim cyan"))
        return
    console.print(Text(f"\n[{name}] {summary}", style="bold cyan"))


@_suppress_in_subagent
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


@_suppress_in_subagent
def render_error(message: str) -> None:
    """渲染错误信息"""
    console.print(Text("Error: ", style="red bold"), Text(message, style="red bold"))


@_suppress_in_subagent
def render_info(message: str) -> None:
    """渲染信息"""
    console.print(f"[cyan]{message}[/cyan]")


@_suppress_in_subagent
def render_success(message: str) -> None:
    """渲染成功信息"""
    console.print(f"[green]{message}[/green]")


@_suppress_in_subagent
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
            "[bold]ChCode[/bold] — " + t("display.welcome_title") + "\n"
            + t("display.welcome_hint"),
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
        from chcode.utils import get_text_content
        content = get_text_content(content)

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
                # 历史消息是完整内容，直接 Markdown 渲染进滚动历史，不走流式 Live。
                console.print()
                console.print(Markdown(content))
                console.print()

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
