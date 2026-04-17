"""
CLI 入口 — Typer 应用
"""

from __future__ import annotations

import asyncio
import typer
from rich.console import Console

from chcode.chat import ChatREPL

app = typer.Typer(
    name="chcode",
    help="Terminal-based AI coding agent",
    no_args_is_help=False,
)
console = Console()


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
        console.print("chcode v0.1.0")
        raise typer.Exit()

    if ctx.invoked_subcommand is not None:
        return

    asyncio.run(_run_chat(yolo))


async def _run_chat(yolo: bool) -> None:
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
        await repl.close()


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
def version():
    """显示版本"""
    console.print("chcode v0.1.0")


if __name__ == "__main__":
    app()
