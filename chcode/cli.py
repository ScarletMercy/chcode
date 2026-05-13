"""
CLI 入口 — Typer 应用
"""

from __future__ import annotations

import asyncio
import os
import sys
import warnings
from importlib.metadata import version as _pkg_version

warnings.filterwarnings("ignore", message="urllib3.*doesn't match a supported version")
warnings.filterwarnings("ignore", message="chardet.*doesn't match a supported version")


def _setup_langsmith_guard():
    """静默吞掉 LangSmith SDK 的 429/连接错误 stderr 输出，防止污染终端 UI"""
    _suppressed = False

    class _Guard:
        def __init__(self, original):
            self._original = original

        def write(self, data):
            nonlocal _suppressed
            if not data:
                return 0
            if _suppressed and ("LangSmith" in data or "langsmith" in data.lower()):
                return len(data)
            if "LangSmithRateLimitError" in data or (
                "langsmith" in data.lower() and "429" in data
            ):
                _suppressed = True
                return len(data)
            if "langsmith" in data.lower() and (
                "ConnectionError" in data
                or "MaxRetryError" in data
                or "ProtocolError" in data
                or "Failed to send" in data
                or "Connection aborted" in data
                or "ConnectionAbortedError" in data
                or "ConnectionResetError" in data
            ):
                _suppressed = True
                return len(data)
            return self._original.write(data)

        def flush(self):
            self._original.flush()

        def __getattr__(self, name):
            return getattr(self._original, name)

    _original = sys.stderr
    _guard = _Guard(_original)
    sys.stderr = _guard


_setup_langsmith_guard()

import typer  # noqa: E402
from chcode.display import console  # noqa: E402

app = typer.Typer(
    name="chcode",
    help="Terminal-based AI coding agent",
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    yolo: bool = typer.Option(
        False, "--yolo", "-y", help="启用 Yolo 模式（自动批准所有操作）"
    ),
    version: bool = typer.Option(False, "--version", "-v", help="显示版本"),
):
    """ChCode — 终端 AI 编程助手"""
    if version:
        console.print(f"chcode v{_pkg_version('chcode')}")
        raise typer.Exit()

    if ctx.invoked_subcommand is not None:
        return

    asyncio.run(_run_chat(yolo))


async def _run_chat(yolo: bool) -> None:
    from chcode.chat import ChatREPL

    repl = ChatREPL()
    repl.yolo = yolo

    try:
        ok = await repl.initialize()
    except Exception:
        console.print_exception()
        raise typer.Exit(1)

    if not ok:
        console.print("[red]初始化失败[/red]")
        raise typer.Exit(1)

    try:
        await repl.run()
    finally:
        await repl.close_checkpointer()


@app.command()
def config(
    action: str = typer.Argument("edit", help="edit | new | switch"),
):
    """模型配置管理"""
    asyncio.run(_run_config(action))


async def _run_config(action: str) -> None:
    from chcode.config import configure_new_model, edit_current_model, switch_model

    if action == "new":
        await configure_new_model()
    elif action == "edit":
        await edit_current_model()
    elif action == "switch":
        await switch_model()
    else:
        console.print(f"[yellow]未知操作: {action}[/yellow]")
        console.print("可用操作: new, edit, switch")


@app.command()
def homepage():
    """打开项目主页"""
    import webbrowser

    from chcode.config import HOMEPAGE_URL

    console.print(f"正在打开: {HOMEPAGE_URL}")
    webbrowser.open(HOMEPAGE_URL)


@app.command()
def version():
    """显示版本"""
    console.print(f"chcode v{_pkg_version('chcode')}")


if __name__ == "__main__":
    app()  # pragma: no cover
