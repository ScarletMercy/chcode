"""
CLI 入口 — Typer 应用
"""

from __future__ import annotations

import asyncio
import sys
import warnings
from importlib.metadata import version as _pkg_version

warnings.filterwarnings("ignore", message="urllib3.*doesn't match a supported version")
warnings.filterwarnings("ignore", message="chardet.*doesn't match a supported version")


def _app_version() -> str:
    """获取版本号；元数据缺失（未 pip 安装）时回退到 '0.0.0+unknown'。"""
    try:
        return _pkg_version("chcode")
    except Exception:
        return "0.0.0+unknown"


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
from chcode.i18n import t  # noqa: E402

app = typer.Typer(
    name="chcode",
    help="Terminal-based AI coding agent",
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
        yolo: bool = typer.Option(
            False, "--yolo", "-y", help="启用 Yolo 模式 / Enable Yolo mode (auto-approve all actions)"
        ),
        lang: str = typer.Option(
            None, "--lang", help="UI 语言 / language (zh | en)"
        ),
        version: bool = typer.Option(False, "--version", "-v", help="显示版本 / Show version"),
    ):
    """ChCode — 终端 AI 编程助手 / Terminal AI coding assistant"""
    if version:
        console.print(f"chcode v{_app_version()}")
        raise typer.Exit()

    # 尽早解析 UI 语言：--lang 标志 > 已保存配置 > 系统 locale 自动检测
    _resolve_language(lang)

    if ctx.invoked_subcommand is not None:
        return

    asyncio.run(_run_chat(yolo))


def _resolve_language(flag: str | None) -> str:
    """按优先级确定并设置 UI 语言：标志 > chagent.json > locale 自动检测。"""
    from chcode.config import load_language
    from chcode.i18n import set_language, detect_locale_language

    if flag:
        return set_language(flag)
    saved = load_language()
    if saved:
        return set_language(saved)
    return set_language(detect_locale_language())


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
        console.print(f"[red]{t('cli.init_failed')}[/red]")
        raise typer.Exit(1)

    try:
        await repl.run()
    finally:
        await repl.close_checkpointer()


@app.command()
def config(
    action: str = typer.Argument("edit", help="edit | new | switch"),
):
    """模型配置管理 / Model configuration management"""
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
        console.print(f"[yellow]{t('cli.unknown_action', action=action)}[/yellow]")
        console.print(t("cli.available_actions"))


@app.command()
def homepage():
    """打开项目主页 / Open project homepage"""
    import webbrowser

    from chcode.config import HOMEPAGE_URL

    console.print(t("cli.opening", url=HOMEPAGE_URL))
    webbrowser.open(HOMEPAGE_URL)


@app.command()
def version():
    """显示版本 / Show version"""
    console.print(f"chcode v{_app_version()}")


if __name__ == "__main__":
    app()  # pragma: no cover
