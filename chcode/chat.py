"""
主聊天 REPL — 类 Claude Code 终端体验

prompt_toolkit 多行输入 + rich 流式输出 + 斜杠命令 + HITL 审批
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from pathlib import Path

import openai
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ToolMessage,
    RemoveMessage,
    HumanMessage,
    BaseMessage,
)
from chcode.utils import get_text_content, mask_api_key
from chcode.i18n import t
from langgraph.types import Command

import chcode.display as _display
from chcode.display import (
    console,
    render_error,
    render_info,
    render_success,
    render_warning,
    render_welcome,
    render_conversation,
    render_ai_start,
    render_ai_chunk,
    render_ai_end,
    get_context_usage_text,
)
from chcode.prompts import select, confirm, select_or_custom, text, checkbox
from chcode.config import (
    get_default_model_config,
    load_workplace,
    save_workplace,
    configure_new_model,
    first_run_configure,
    edit_current_model,
    switch_model,
    ensure_config_dir,
    _DEFAULT_CONTEXT_WINDOW,
)
from chcode.utils.session import SessionManager
from chcode.utils.skill_loader import SkillAgentContext, SkillLoader
from chcode.agent_setup import (
    build_agent,
    create_checkpointer,
    INNER_MODEL_CONFIG,
    reset_budget_state,
    get_fallback_model,
    advance_fallback,
    ModelSwitchError,
)
from chcode.utils.skill_manager import manage_skills
from chcode.utils.git_checker import check_git_availability
from chcode.utils.git_manager import GitManager
from chcode.utils.modelscope_ratelimit import get_ratelimit, is_modelscope_model


# ─── 命令自动补全 ──────────────────────────────────────

SLASH_COMMANDS = {
    "/new": "cmd.new",
    "/history": "cmd.history",
    "/model": "cmd.model",
    "/vision": "cmd.vision",
    "/messages": "cmd.messages",
    "/compress": "cmd.compress",
    "/skill": "cmd.skill",
    "/search": "cmd.search",
    "/workdir": "cmd.workdir",
    "/mode": "cmd.mode",
    "/git": "cmd.git",
    "/langsmith": "cmd.langsmith",
    "/tools": "cmd.tools",
    "/lang": "cmd.lang",
    "/homepage": "cmd.homepage",
    "/help": "cmd.help",
    "/quit": "cmd.quit",
}


class SlashCommandCompleter(Completer):
    """斜杠命令自动补全器 - 输入 / 时触发下拉列表"""

    def get_completions(self, document, complete_event):
        # 获取光标前的完整文本
        text = document.text_before_cursor

        # 当输入 / 时触发补全
        if text.startswith("/"):
            # 把输入的文本中的字母转化成小写来处理（大小写不敏感）
            partial = text.lower()
            # 遍历预先定义的斜杠命令字典
            for cmd, desc_key in SLASH_COMMANDS.items():
                # 如果转化成小写的输入框中文本 被字典里 命令名 的 前缀匹配 到
                if cmd.startswith(partial):
                    # 生成命令
                    yield Completion(
                        cmd,  # 返回完整的命令
                        start_position=-len(partial),  # 返回前清空输入框已有输入
                        display=cmd,  # 下拉框显示的命令名
                        display_meta=t(desc_key),  # 下拉框显示的命令名的描述（按当前语言翻译）
                    )


# ─── 辅助函数 ──────────────────────────────────────────

# 简易的 BBCode 风格标记语言解析 （论坛或聊天软件）
_RE_TAG_SPLIT = re.compile(r"(\[/?[^\]]+\])")
_RE_TAG_OPEN = re.compile(r"^\[([^\]]+)\]$")
_RE_TAG_CLOSE = re.compile(r"^\[/([^\]]*)\]$")

_RICH_TAG_MAP = {
    "bold": "b",
    "italic": "i",
    "red": "fg:red",
    "green": "fg:green",
    "yellow": "fg:yellow",
    "blue": "fg:blue",
    "dim": "fg:#888888",
}


# 将BBCode 风格标记语言 渲染成 html 样式
def _rich_to_html(text: str) -> str:
    parts = _RE_TAG_SPLIT.split(text)
    opened: list[str] = []
    result: list[str] = []

    for part in parts:
        close_m = _RE_TAG_CLOSE.match(part)
        open_m = _RE_TAG_OPEN.match(part) if not close_m else None
        if close_m:
            while opened:
                tag = opened.pop()
                result.append(f"</{tag}>")
        elif open_m:
            tags = open_m.group(1).split()
            for t in tags:
                mapped = _RICH_TAG_MAP.get(t)
                if mapped:
                    if mapped.startswith("fg:"):
                        result.append(f'<style fg="{mapped[3:]}">')
                        opened.append("style")
                    else:
                        result.append(f"<{mapped}>")
                        opened.append(mapped)
        else:
            result.append(part)

    return "".join(result)


# 获取最近的几组消息
def find_and_slice_from_end(lst, x):
    """从后往前查找第一个 type==x 的元素，返回从该元素到末尾的切片"""
    for i in range(len(lst) - 1, -1, -1):
        if lst[i].type == x:
            return lst[i:]
    return []


# 消息分组
def _group_messages_by_turn(messages: list) -> list[list]:
    """
    将消息按轮次分组（参考 chagent 逻辑）
    从一个 HumanMessage 开始，到下一个 HumanMessage 之前为一组
    """
    groups = []
    current_group = []

    for msg in messages:
        if msg.type == "human":  # 下一组消息的第一个消息：HumanMessage
            if current_group:  # 当前消息组
                groups.append(current_group)
            current_group = [msg]  # 把下一组消息的第一个消息：HumanMessage，放入新的消息组
        else:
            current_group.append(msg)  # 把下一组消息的其余消息也放入新的消息组

    if current_group:  # 所有消息都遍历完 还没放入消息组
        groups.append(current_group)  # 所以需要放入消息组

    return groups


# 历史会话的会话名显示
def _get_group_display(group: list) -> str:
    """获取消息组的显示文本（以 HumanMessage 内容为代表）"""
    for msg in group:  # 遍历消息组
        if msg.type == "human":  # 遇到HumanMessage的话
            text_content = get_text_content(msg.content)  # 获取消息文本内容前60字当场会话名显示
            content = text_content[:60].replace("\n", " ")
            if len(text_content) > 60:
                content += "..."
            return content
    return t("chat.empty_group")  # zh：（空消息组）/ en: (empty message group)


# 收集即将被压缩的消息的消息id组
def _collect_ids_from_group(group_index: int, groups: list) -> tuple[list[str], list[str]]:
    all_ids = [m.id for group in groups for m in group]
    no_need_ids = []
    for i, group in enumerate(groups):
        if i >= group_index:
            no_need_ids.extend([m.id for m in group])
    return no_need_ids, all_ids


class _LimitedFileHistory(FileHistory):
    MAX_ENTRIES = 50

    def store_string(self, string):
        Path(self.filename).parent.mkdir(exist_ok=True)
        super().store_string(string)
        strings = list(self.load_history_strings())
        if len(strings) > self.MAX_ENTRIES:
            keep = strings[:self.MAX_ENTRIES]
            self._loaded_strings = keep
            self._rewrite(keep)

    def _rewrite(self, keep):
        import datetime as _dt
        Path(self.filename).parent.mkdir(exist_ok=True)
        with open(self.filename, "wb") as f:
            for s in reversed(keep):
                f.write(f"\n# {_dt.datetime.now()}\n".encode())
                for line in s.split("\n"):
                    f.write(f"+{line}\n".encode())


# ─── 主聊天类 ──────────────────────────────────────────


class ChatREPL:
    def __init__(self):
        self.workplace_path: Path | None = None  # 工作目录路径
        self.model_config: dict = {}  # 模型参数
        self.yolo = False  # Yolo模式
        self.agent = None  # agent实例
        self.checkpointer = None  # 检查点实例
        self.session_mgr: SessionManager | None = None  # 会话管理器
        self.git_manager: GitManager | None = None  # git管理器
        self.git = False  # git是否激活
        self._git_cp_count = 0  # git提交数
        self._stop_requested = False  # 暂停agent的flag
        self._processing = False
        self._prompt_session = None  # 初始化 prompt-toolkit 会话（用于命令自动补全）
        self._edit_buffer: str | None = None  # 编辑缓冲区（用于 /edit 命令）
        self._interrupt_buffer: str | None = None  # 中断恢复缓冲区（中断时将内容填回输入框，不进入编辑模式）
        self._skill_loader: SkillLoader | None = None  # SkillLoader 复用，避免每条消息重建
        self._context_text: str = ""  # 上下文用量缓存
        # LangSmith 配置
        self.langsmith_tracing = False
        self.langsmith_project = ""
        self.langsmith_api_key = ""
        # Windows 保留名（不能作为文件名）
        self.WINDOWS_RESERVED_NAMES = {
            "nul",
            "con",
            "aux",
            "prn",
            "com1",
            "com2",
            "com3",
            "com4",
            "lpt1",
            "lpt2",
            "lpt3",
        }

    # 确保配置文件存在
    @staticmethod
    def _ensure_chat_dir(workplace: Path) -> None:
        """确保工作目录下 .chat/sessions 和 .chat/skills 子目录存在。"""
        chat_dir = workplace / ".chat"
        chat_dir.mkdir(exist_ok=True)
        (chat_dir / "sessions").mkdir(exist_ok=True)
        (chat_dir / "skills").mkdir(exist_ok=True)

    # ─── 清理 ────────────────────────────────────────

    async def close_checkpointer(self) -> None:
        """安全关闭 checkpointer 连接"""
        if self.checkpointer is not None:
            try:
                await self.checkpointer.conn.close()
            except Exception:
                pass
            finally:
                self.checkpointer = None

    async def _rebuild_agent(self, *, rebuild_session: bool = False) -> None:
        """重建 agent（可选重建 session/checkpointer）"""
        if rebuild_session:
            await self.close_checkpointer()  # 关闭当前会话数据库连接
            self.session_mgr: SessionManager = SessionManager(self.workplace_path)  # 创建会话管理器
            db_path = self.session_mgr.sessions_dir / "checkpointer.db"  # 创建新的会话数据库（一般是进入新工作目录才这样）
            self.checkpointer = await create_checkpointer(db_path)  # 创建数据库连接
        self.agent = await asyncio.to_thread(  # 异步新线程构建agent
            build_agent,
            self.model_config,
            self.checkpointer,
            self.yolo,
        )

    # ─── 初始化 ────────────────────────────────────────

    async def initialize(self) -> bool:
        """初始化：加载配置、设置工作目录、构建 agent"""
        ensure_config_dir()  # 确保全局配置目录.chat存在

        self.workplace_path = Path.cwd()  # 获取当前目录路径

        self._ensure_chat_dir(self.workplace_path)  # 确保当前项目配置文件存在

        self.session_mgr: SessionManager = SessionManager(self.workplace_path)  # 初始化历史会话管理器

        self.model_config = get_default_model_config() or {}  # 尝试从model.json配置文件中获取默认模型配置，如果获取失败则返回空字典
        if not self.model_config:  # 如果默认模型配置不存在
            config = await first_run_configure()  # 进行初始化引导
            if config is None:  # 如果配置依旧为空
                return False  # 返回False
            self.model_config = config  # 否则缓存模型配置

        # 从环境变量恢复 LangSmith 配置
        from chcode.config import load_langsmith_config
        langsmith_cfg = load_langsmith_config()
        if langsmith_cfg["api_key"] or langsmith_cfg["tracing"]:
            self.langsmith_tracing = langsmith_cfg["tracing"]
            self.langsmith_project = langsmith_cfg["project"]
            self.langsmith_api_key = langsmith_cfg["api_key"]

        # 创建 checkpointer
        db_path = self.session_mgr.sessions_dir / "checkpointer.db"
        self.checkpointer = await create_checkpointer(db_path)

        # 构建 agent（可能较慢，放线程）
        console.print(
            "[dim cyan]"
            " ███████╗  ██╗   ██╗   ███████╗   ██████╗   █████╗     ████████╗\n"
            "██╔═════╝  ██║   ██║  ██╔═════╝  ██╔═══██╗  ██╔══██╗   ██╔═════╝\n"
            "██║        ████████║  ██║        ██║   ██║  ██║   ██╗  ████████╗\n"
            "██║        ██╔═══██║  ██║        ██║   ██║  ██║  ██╔╝  ██╔═════╝\n"
            "████████╗  ██║   ██║  ████████╗  ╚██████╔╝  █████╔═╝   ████████╗\n"
            " ╚══════╝  ╚═╝   ╚═╝   ╚══════╝   ╚═════╝   ╚════╝      ╚══════╝[/dim cyan]"
        )
        self.agent = await asyncio.to_thread(
            build_agent,
            self.model_config,
            self.checkpointer,
            self.yolo,
        )

        # 初始化 Git（subprocess.run 会阻塞事件循环）
        await self._init_git()

        return True

    async def _init_git(self) -> None:
        """初始化 Git（影子仓库）"""
        is_available, status, version = await asyncio.to_thread(check_git_availability)
        if is_available:
            self.git_manager = GitManager(str(self.workplace_path))
            shadow_ready = await asyncio.to_thread(self.git_manager.init_shadow)
            self.git = bool(shadow_ready)
            if self.git:
                self._git_cp_count = self.git_manager.count_checkpoints()

    # ─── 主循环 ────────────────────────────────────────

    async def run(self) -> None:
        """主聊天循环"""
        render_welcome()  # 渲染欢迎界面

        while True:
            try:
                user_input = await self._get_input()
                if user_input is None:
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                # 斜杠命令
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue

                # 正常对话
                await self._process_input(user_input)

            except KeyboardInterrupt:
                if self._processing:
                    self._stop_requested = True
                else:
                    console.print(Text(f"\n{t('chat.goodbye')}", style="dim"))
                    break
            except EOFError:
                break
            except Exception as e:
                render_error(f"Unexpected error: {e}")

    async def _get_input(self) -> str | None:
        """获取用户输入（使用 prompt-toolkit 实现命令自动补全）"""

        # 初始化 prompt session（带命令自动补全 + 底部状态栏）
        if self._prompt_session is None:
            completer = SlashCommandCompleter()

            # 自定义按键：Enter 提交，Ctrl+Enter 换行
            kb = KeyBindings()

            @kb.add("enter")
            def _submit(event):
                event.current_buffer.validate_and_handle()  # 验证并提交缓冲区内容

            @kb.add("c-j")  # Ctrl+Enter → 换行
            def _newline(event):
                event.current_buffer.insert_text("\n")  # 向缓冲区插入换行

            @kb.add("tab")
            def _tab_toggle_mode(event):
                if event.current_buffer.text:
                    return  # 有内容时走默认补全
                self.yolo = not self.yolo
                from chcode.agent_setup import update_hitl_config

                update_hitl_config(self.yolo)  # 构造agent前
                event.app.renderer._last_rendered_width = 0  # 强制刷新 toolbar

            _last_width = 0
            _last_width_time = 0.0

            def _bottom_toolbar():
                nonlocal _last_width, _last_width_time
                import time as _time
                now = _time.monotonic()
                if now - _last_width_time > 1.0:
                    _last_width = shutil.get_terminal_size().columns
                    _last_width_time = now
                width = _last_width or shutil.get_terminal_size().columns
                sep = "\u2500" * width
                parts = []
                model = self.model_config.get("model", t("chat.status.model_unset"))
                parts.append(model)
                if self._context_text:
                    styled = _rich_to_html(self._context_text)
                    parts.append(styled)
                parts.append(
                    t("chat.status.common_mode") if not self.yolo
                    else f"<ansired>{t('chat.status.yolo_mode')}</ansired>"
                )
                if self.git and self.git_manager:
                    parts.append(f"Git ({self._git_cp_count} cp)")
                wp = str(self.workplace_path) if self.workplace_path else ""
                if wp:
                    parts.append(f"cwd: {wp}")
                status = "  │  ".join(parts)
                ratelimit_line = ""
                if is_modelscope_model(self.model_config):
                    rl = get_ratelimit()
                    if rl:
                        total = f"{rl['total_remaining']}/{rl['total_limit']}"
                        model_name = self.model_config.get("model", "").split("/")[-1]
                        model_rl = f"{rl['model_remaining']}/{rl['model_limit']}"
                        ratelimit_line = t(
                            "chat.status.modelscope_quota",
                            total=total, model=model_name, model_rl=model_rl,
                        )
                return HTML(f"<ansiblue>{sep}</ansiblue>\n{status}{ratelimit_line}")

            self._prompt_session: PromptSession = PromptSession(
                history=_LimitedFileHistory(str(Path.home() / ".chat" / "history")),
                multiline=True,
                key_bindings=kb,
                completer=completer,
                complete_while_typing=True,
                reserve_space_for_menu=0,
                bottom_toolbar=_bottom_toolbar,
                refresh_interval=0.1,
                style=Style.from_dict(
                    {
                        "completion-menu.completion": "bg:#008888 #ffffff",
                        "completion-menu.completion.current": "bg:#00aaaa #000000",
                        "completion-menu.meta.completion": "bg:#008888 #ffffff",
                        "completion-menu.meta.completion.current": "bg:#00aaaa #000000",
                        "bottom-toolbar": "noreverse bg:#1a1a2e #aaaaaa",
                    }
                ),
            )

            # 动态缓存区高度
            def _dynamic_buffer_height():
                buff = self._prompt_session.default_buffer
                if buff.complete_state is not None:
                    n = len(buff.complete_state.completions)
                    needed = min(n + 2, 10)
                    return Dimension(min=needed, max=needed)
                line_count = buff.text.count("\n") + 1
                return Dimension(min=line_count, max=line_count)

            # 寻找缓存区窗口
            def _find_buffer_window(container):
                from prompt_toolkit.layout.containers import Window
                from prompt_toolkit.layout.controls import BufferControl

                if isinstance(container, Window):
                    if isinstance(getattr(container, "content", None), BufferControl):
                        return container
                for attr in ("content", "children", "alternative_content"):
                    child = getattr(container, attr, None)
                    if child is None:
                        continue
                    children = child if isinstance(child, list) else [child]
                    for c in children:
                        result = _find_buffer_window(c)
                        if result:
                            return result
                return None

            buffer_window = _find_buffer_window(
                self._prompt_session.app.layout.container
            )
            if buffer_window:
                buffer_window.height = _dynamic_buffer_height

        try:
            # 如果有编辑缓冲区，预填充到输入框
            if self._edit_buffer is not None:
                default_text = self._edit_buffer
                self._edit_buffer = None  # 清除缓冲区
            # 如果有中断恢复缓冲区，也预填充到输入框
            elif self._interrupt_buffer is not None:
                default_text = self._interrupt_buffer
                self._interrupt_buffer = None  # 清除缓冲区
            # 如果都没有，则不填充
            else:
                default_text = ""

            width = shutil.get_terminal_size().columns  # 获取终端大小的列数，确保分隔线始终覆盖整个宽度
            sep = "\u2500" * width  # 即为 width个 ─ , 效果：───────────────（这个是输入框的顶栏，由于prompt_toolkit不支持顶栏，所以需要自己构造）
            prompt_text = f"{sep}\n > "  # 构造顶栏和 > 提示符

            # 使用 prompt-toolkit 获取输入（支持命令自动补全）
            result = await self._prompt_session.prompt_async(  # 显示顶栏和 > 提示符，并等待用户输入，返回值也是用户的输入
                HTML(f"<ansiblue>{prompt_text}</ansiblue>"),
                default=default_text,  # 返回的默认值，代替用户输入或可能为空
            )
            return result
        except (EOFError, KeyboardInterrupt):
            return None

    # ─── 斜杠命令 ──────────────────────────────────────

    async def _handle_command(self, cmd: str) -> None:
        """处理斜杠命令"""
        parts = cmd.strip().split(maxsplit=1)  # 通过空格将命令分割放入列表，最大分割一次
        command = parts[0].lower()  # 第一个是命令
        arg = parts[1] if len(parts) > 1 else ""  # 如果有第二个则为参数

        handlers = {
            "/new": self._cmd_new,
            "/model": self._cmd_model,
            "/vision": self._cmd_vision,
            "/skill": self._cmd_skill,
            "/history": self._cmd_history,
            "/compress": self._cmd_compress,
            "/git": self._cmd_git,
            "/search": self._cmd_search,
            "/mode": self._cmd_mode,
            "/workdir": self._cmd_workdir,
            "/tools": self._cmd_tools,
            "/langsmith": self._cmd_langsmith,
            "/lang": self._cmd_lang,
            "/messages": self._cmd_messages,
            "/homepage": self._cmd_homepage,
            "/help": self._cmd_help,
            "/quit": self._cmd_quit,
        }

        handler = handlers.get(command)
        if handler:
            await handler(arg)
        else:
            render_warning(t("chat.unknown_command", command=command))

    # 创建新会话（使用新的thread_id,传入agent时，langchain会自动创建新会话）
    async def _cmd_new(self, _arg: str) -> None:
        reset_budget_state()
        self.session_mgr.new_session()
        render_success(t("chat.new_session_started"))

    # 模型配置
    async def _cmd_model(self, arg: str) -> None:
        if arg == "new":
            config = await configure_new_model()
        elif arg == "edit":
            config = await edit_current_model()
        elif arg == "switch":
            config = await switch_model()
        else:
            action = await select(
                t("chat.model.menu"),
                [
                    t("chat.model.new"),
                    t("chat.model.edit"),
                    t("chat.model.switch"),
                ],
            )
            if action is None:
                return
            # 按稳定子命令分发（选项文案中均含 /model new|edit|switch，语言无关）
            if "new" in action:
                config = await configure_new_model()
            elif "edit" in action:
                config = await edit_current_model()
            elif "switch" in action:
                config = await switch_model()
            else:
                return

        if config:
            self.model_config = config
            from chcode.agent_setup import update_summarization_model
            # 同步更新摘要模型
            update_summarization_model(config)

    def _sync_langsmith_env(self) -> None:
        from chcode.config import _apply_langsmith_env
        _apply_langsmith_env(self.langsmith_tracing, self.langsmith_project, self.langsmith_api_key)

    async def _cmd_langsmith(self, _arg: str) -> None:
        # 显示当前状态
        state = t("chat.langsmith.state_on") if self.langsmith_tracing else t("chat.langsmith.state_off")
        console.print(f"[bold]{t('chat.langsmith.tracing_line', state=state)}[/bold]")
        if self.langsmith_project:
            console.print(t("chat.langsmith.project_line", project=self.langsmith_project))
        if self.langsmith_api_key:
            masked = mask_api_key(self.langsmith_api_key)
            console.print(f"  Key:  {masked}")

        open_label = t("chat.langsmith.open_panel")
        enable_label = t("chat.langsmith.enable")
        disable_label = t("chat.langsmith.disable")
        rename_label = t("chat.langsmith.rename_project")
        changekey_label = t("chat.langsmith.change_key")
        match await select(  # match action
            t("chat.langsmith.operation"),
            [open_label, enable_label, disable_label, rename_label, changekey_label],
        ):
            case None:
                return  # 如果什么都不选，直接退出
            case action if action == open_label:
                import webbrowser

                webbrowser.open("https://smith.langchain.com")  # 如果是windows系统且有浏览器，会直接打开项目主页。如果不是则无效
                return
            case action if action == enable_label:
                if not self.langsmith_api_key:
                    console.print(f"[yellow]{t('chat.langsmith.set_key_first')}[/yellow]")
                    return  # 没有设置LangSmith API Key会提醒配置，然后直接退出
                self.langsmith_tracing = True
            case action if action == disable_label:
                self.langsmith_tracing = False
            case action if action == rename_label:
                new_name = await text(t("chat.langsmith.input_project"), default=self.langsmith_project or "chcode")
                if new_name is None:
                    return
                self.langsmith_project = new_name.strip() or "chcode"
            case action if action == changekey_label:
                new_key = await text(t("chat.langsmith.input_key"))
                if new_key:
                    self.langsmith_api_key = new_key
                else:
                    return

        self._sync_langsmith_env()
        render_success(t("chat.langsmith.config_updated"))

    # 列出所有可用工具及其描述
    async def _cmd_tools(self, _arg: str) -> None:
        from chcode.utils.tools import ALL_TOOLS  # 导入包含所有工具的列表
        from chcode.utils.multimodal import is_multimodal_model

        current_model = (self.model_config or {}).get("model", "")  # 安全获取模型配置
        native_vision = is_multimodal_model(current_model)  # 判断当前模型是否是视觉模型，如果是视觉模型会禁用视觉工具

        console.print(f"[bold]{t('chat.tools.title')}[/bold]")
        console.print()
        if native_vision:
            console.print(f"[dim]{t('chat.tools.native_vision')}[/dim]")
            console.print()
        for tool in ALL_TOOLS:
            # 优先取当前语言的翻译描述，缺失时回退到英文 docstring 首行
            _desc_key = f"tool_desc.{tool.name}"
            _translated = t(_desc_key)
            desc = _translated if _translated != _desc_key else (tool.description or "").split("\n")[0]
            is_disabled = tool.name == "vision" and native_vision
            style = "dim" if is_disabled else "cyan"  # 样式；dim：灰色（禁用时），cyan：青色（未禁用时）
            suffix = t("chat.tools.disabled") if is_disabled else ""  # 后缀
            console.print(f"  [{style}]{tool.name:<16}[/{style}] {desc}{suffix}")  # <：左对齐，16：补足空格至16个单位长度
        console.print()

    async def _cmd_skill(self, _arg: str) -> None:
        await manage_skills(self.workplace_path)

    async def _cmd_history(self, _arg: str) -> None:

        # ------------------- 1.选择会话--------------------------------------
        if not self.session_mgr or not self.checkpointer or not self.agent:
            return
        sessions = await self.session_mgr.list_sessions(self.checkpointer)  # 通过检查点从数据库（sqlite）中获取所有会话（实际为会话线程id）
        if not sessions:
            render_warning(t("chat.history.none"))
            return
        # ；把自动命名的会话名也缓存（可选）
        sessions = sessions[-50:]  # 取倒数50个会话并倒序排序（从新到旧）
        display_names = await self.session_mgr.get_display_names(sessions,
                                                                 self.agent)  # 渲染所有会话的名称，返回一个 {tid: display_name} 的字典
        label_to_tid: dict[str, str] = {}  # 初始化 <标签：会话线程id> 键值字典
        labels: list[str] = []  # 初始化标签列表（展示给用户的会话名）
        for tid in sessions:  # 遍历会话线程id
            name = display_names.get(tid, tid)  # 通过 会话线程id 获取渲染的 会话名 ，如果没有则直接用 线程id 代替 空的 会话名
            label = name if name == tid else f"{name}  ({tid})"  # 拼接 会话名 和 线程id 成 新的会话名，确保会话名 绝对的 唯一性
            label_to_tid[label] = tid  # 构建 <新的会话名：会话线程id>字典
            labels.append(label)  # 构建 会话名 列表（展示给用户）
        back_label = t("common.back")
        labels.append(back_label)  # 在 会话名 列表最后 加上 返回 选项

        action = await select(t("chat.history.select"), labels)  # 获取用户 选择的 会话名
        if action is None or action == back_label:  # 如果是返回直接退出
            return

        selected_tid = label_to_tid[action]  # 根据 用户选择 的会话名 在 之前构建好的<会话名：会话线程id>字典中 获取 会话名 对应的 会话线程id

        # ------------------- 2.操作选择的会话--------------------------------------
        load_label = t("chat.history.load")
        rename_label = t("chat.history.rename")
        delete_label = t("chat.history.delete")
        match await select(t("chat.history.operation"), [load_label, rename_label, delete_label, back_label]):  # 可以对会话进行 这4个操作
            case action if action == load_label:
                self.session_mgr.set_thread(selected_tid)  # 设置会话管理器 的 线程id 属性为 选中的 会话 对应的 线程id
                await self._load_conversation()  # 加载会话历史消息 （通过 线程id 从 agent 的 state 中取）
            case action if action == rename_label:
                try:
                    cur = self.session_mgr._load_names().get(selected_tid,
                                                             "")  # 尝试获取 已经可能被 更改过的 会话名  | names.json 通过 _save_names 保存。其在两个地方被调用：1. rename_session— 用户重命名会话时写入。  2. delete_session— 删除会话时从 names 里移除对应条目再写回
                except Exception:  # 获取失败（说明当前会话尚未被改过名）
                    cur = ""
                new_name = await text(t("chat.history.rename_prompt"), default=cur)
                if new_name is not None:
                    self.session_mgr.rename_session(selected_tid, new_name)  # 将 新会话名 和 对应的 线程id 持久化到 name.json中
                    render_success(t("chat.history.renamed"))
            case action if action == delete_label:
                ok = await confirm(t("chat.history.delete_confirm", tid=selected_tid), default=False)
                if ok:
                    await self.session_mgr.delete_session(selected_tid, self.checkpointer)  # 从数据库中删除 会话id 对应的 会话
                    render_success(t("chat.history.deleted"))
                    if selected_tid == self.session_mgr.thread_id:
                        await self._cmd_new("")  # 如删除的是 当前会话 就 原地开启 新会话
            case _:  # 返回 或 Ctrl C 都回到上一步（重新加载历史会话）
                await self._cmd_history(_arg)

    async def _cmd_compress(self, _arg: str) -> None:
        if not self.model_config:  # 压缩会话 的 模型 复用 主模型
            render_warning(t("chat.compress.no_model"))
            return

        if not await confirm(t("chat.compress.confirm"), default=True):
            return  # 如果拒绝直接退出

        render_info(t("chat.compress.working"))
        try:
            state = await self.agent.aget_state(self.session_mgr.config)  # 通过 config（会话线程id）取出 state （其中的 messages）
            messages: list[BaseMessage] = state.values["messages"]  # 从state取出 消息列表

            # 分离历史消息和最近消息
            recent_messages = []  # 最近2组消息 （保留的消息）
            recent_message_ids = []
            recent_count = 0
            for msg in reversed(messages):  # 倒着遍历 消息列表
                recent_messages.append(msg)
                recent_message_ids.append(msg.id)
                if isinstance(msg, HumanMessage):
                    recent_count += 1
                    if recent_count == 2:  # 只取最后两组消息
                        break

            pre_messages = []  # 最后2组消息之前的消息（要被压缩的消息）
            for msg in messages:
                if msg.id not in recent_message_ids:  # 只要除 最近消息（最后两组消息） 之外的消息
                    msg.additional_kwargs["composed"] = True  # 给需要被压缩的消息加上 压缩标记，由langchain中间件 识别 为已压缩过的消息，显示给用户，但不传给模型
                    # 压缩时去掉 base64 图片/视频，避免 payload 过大导致 API 返回空 choices
                    if isinstance(msg.content, list):  # 只有列表形式才可能包含图片/视频块
                        clean_blocks = [
                            b for b in msg.content
                            if not isinstance(b, dict)
                               or b.get("type") not in ("image_url", "video_url")
                        ]  # 提取 非字典 或 是字典 但类型不是 image-url和video-url 的内容
                        if clean_blocks != msg.content:
                            msg = msg.model_copy(update={"content": clean_blocks})  # 重新构造消息对象，已剔除 图片和视频 2进制数据
                    pre_messages.append(msg)

            from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI

            model = EnhancedChatOpenAI(**self.model_config)

            compact_prompt = """请为衔接我们上面的对话提供一个详细的总结。
重点关注那些对继续对话有帮助的信息，包括我们做了什么、正在做什么、正在处理哪些文件，以及接下来要做什么。
你构建的摘要将被另一个代理读取并继续工作。
不要调用任何工具。只回复摘要文本。
使用与对话中用户消息相同的语言回复。

在构建摘要时，请尽量遵循以下模板：
---
## 目标

[用户试图完成什么目标？]

## 指令

- [用户给出了哪些重要的相关指令]
- [如果有任何计划或规范，请包含相关信息，以便下一个代理可以继续使用]

## 发现

[在这次对话中学到了哪些值得注意的事情，这些信息对继续工作的下一个代理有用]

## 已完成的工作

[哪些工作已经完成，哪些工作正在进行中，还有哪些工作待完成？]

## 相关文件/目录

[构建一个相关文件的结构化列表，包括那些与手头任务相关的已被读取、编辑或创建的文件。如果某个目录中的所有文件都相关，请包含该目录的路径。]
---
            """
            human_msg = HumanMessage(
                content=f'{compact_prompt}，严格按以下JSON格式输出，不要使用markdown代码块：' + '\n{"summary": "总结内容"}',
                additional_kwargs={"hide": True, "composed": True},  # 隐藏显示，并且不传给agent
            )

            try:
                raw_resp = await asyncio.to_thread(
                    model.invoke, pre_messages + [human_msg]
                )  # 独立模型调用 压缩消息

                # -----------------------保证结构化输出--------------------------------------------------
                content = raw_resp.content.strip()  # 去除 压缩后的内容 （模型回复），并用 strip 清理空格
                ##-----------------------第一层处理（去除 markdown 代码块包裹）--------------------------------
                if content.startswith("```"):  # 模型可能以markdown的格式输出结构化json输出,所以需要清理 （```{}```）
                    content = re.sub(r"^```(?:json)?\s*\n?", "", content)  # 去掉开头的“```”或“```json”
                    content = re.sub(r"\n?```\s*$", "", content)  # 去掉末尾的“```”
                ##-----------------------第二层处理 （提取包含 "summary" 的 JSON 对象（模型可能在 JSON 前输出思考内容））-------------------------------
                # 针对"简单情况"——LLM 在 JSON 前说了一堆废话（如"好的，压缩后的内容如下："）——用非贪婪匹配从文本中抠出第一个包含 "summary" 键的 扁平 JSON 对象。
                json_match = re.search(r'\{[^{}]*"summary"[^{}]*\}', content)  #
                r"""
                正则表达式分解:
                \{：匹配左花括号 {（转义字符，因为 { 在正则中有特殊含义）。
                [^{}]*：匹配任意数量（包括零个）非花括号的字符（即 { 和 } 以外的字符）。
                "summary"：匹配字面字符串 "summary"。
                [^{}]*：继续匹配任意数量的非花括号字符。
                \}：匹配右花括号 }（转义字符）。
                """
                if json_match:  # 如果匹配到了直接取出来
                    content = json_match.group()
                ##-----------------------第三层处理（兜底）---------------------------------------------------
                else:  # 如果没有就 用 第3层 的 兜底操作
                    # 可能 summary 值中包含嵌套对象（例如{"summary": {"text": "..."}}），用逐字符括号匹配兜底
                    # NOTE: 不处理字符串内的 `}`，但模型 summary 含 `}` 的概率极低（例如 {"summary": "a}b"} ），暂不改
                    depth = 0
                    start = -1
                    for i, ch in enumerate(content):
                        if ch == '{':
                            if depth == 0:
                                start = i  # 如果不是嵌套 记录 “{” 位置
                            depth += 1
                        elif ch == '}':
                            depth -= 1  # 闭合一个“{}”，闭合标志：depth==0
                            if depth == 0 and start >= 0:  # 如果闭合至少一个非嵌套{}，
                                candidate = content[start:i + 1]  # 取出 {"xxx":"xxx"}
                                if '"summary"' in candidate:
                                    content = candidate  # 如果 summary 在 {"xxx":"xxx"} ，则代表匹配成功，直接取用
                                    break

                data = json.loads(content)  # 将json格式字符串 转成 json格式
                # -----------------------结构化输出保证工作自此 结束-------------------------------------

                ai_content = data.get("summary", "")
                if isinstance(ai_content, dict):  # summary 的值是 字典 也接受，把它转成json字符串
                    ai_content = json.dumps(ai_content, ensure_ascii=False)
                if not ai_content:  # summary值 为空 说明 模型输出错误，压缩失败
                    ai_content = t("chat.compress.failed_no_summary")
            except Exception as e:
                ai_content = t("chat.compress.failed_detail", error=e)

            if ai_content.startswith(t("chat.compress.failed_prefix")):
                ai_message = AIMessage(
                    ai_content,
                    additional_kwargs={"composed": True},
                    usage_metadata={
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                )
            else:
                ai_message = AIMessage(
                    t("chat.compress.done_prefix") + ai_content,
                    additional_kwargs={"hide": True},
                )

            await self.agent.aupdate_state(
                self.session_mgr.config,
                {"messages": pre_messages + [human_msg, ai_message] + recent_messages},
                as_node="model",
            )
            await self._load_conversation()  # 压缩后 重新加载一下会话
            render_success(t("chat.compress.complete"))
        except Exception as e:
            render_error(t("chat.compress.error", error=e))

    async def _cmd_git(self, _arg: str) -> None:
        if not self.git_manager: # 如果尚未初始化 git-manager，就先检查git是否可用
            is_available, status, version = await asyncio.to_thread(
                check_git_availability
            )
            if is_available:
                render_success(f"Git {version}")
                await self._init_git() # 初始化 git-manager
            else:
                render_error(t("chat.git.unavailable", status=status))
                return

        if self.git:
            count = self.git_manager.count_checkpoints()
            self._git_cp_count = count
            render_success(t("chat.git.repo_init", count=count))
        else:
            render_warning(t("chat.git.repo_not_init"))

    async def _cmd_vision(self, _arg: str) -> None:  # pragma: no cover
        """视觉模型配置命令"""  # pragma: no cover
        from chcode.vision_config import configure_vision_interactive  # pragma: no cover
        await configure_vision_interactive()  # pragma: no cover

    # Tavily搜索配置
    async def _cmd_search(self, _arg: str) -> None:
        from chcode.config import load_tavily_api_key, save_tavily_api_key
        from chcode.utils.tools import update_tavily_api_key

        current = load_tavily_api_key() # 加载当前tavily api key
        masked = (
            mask_api_key(current)
            if current and len(current) > 10
            else (current or t("chat.search.unset"))
        ) # mask api key
        render_info(t("chat.search.current_key", key=masked)) # 显示mask后的api key

        # 变量化选项标签
        back_label = t("common.back")
        configure_label = t("chat.search.configure")
        clear_label = t("chat.search.clear")
        action = await select(t("chat.search.operation"), [configure_label, clear_label, back_label])
        if action is None or action == back_label:
            return

        if action == clear_label:
            save_tavily_api_key("")
            update_tavily_api_key("")
            render_success(t("chat.search.cleared"))
            return

        new_key = await text(t("chat.search.input_key"))
        if new_key:
            save_tavily_api_key(new_key)  # 持久化到配置文件
            update_tavily_api_key(new_key) # 重建运行时全局 TavilyClient
            render_success(t("chat.search.saved"))
        else:
            render_warning(t("chat.search.cancelled"))

    # Common/Yolo 模式切换
    async def _cmd_mode(self, _arg: str) -> None:
        action = await select(
            t("chat.mode.select"),
            [t("chat.mode.common"), t("chat.mode.yolo")],
        )
        if action is None:
            return
        self.yolo = "Yolo" in action
        from chcode.agent_setup import update_hitl_config

        update_hitl_config(self.yolo) # 更新 人在闭环 中间配置
        mode_str = "Yolo" if self.yolo else "Common"
        render_success(t("chat.mode.switched", mode=mode_str))

    # 切换工作目录
    async def _cmd_workdir(self, _arg: str) -> None:
        saved = load_workplace() # 加载上次工作目录路径
        choices = [str(saved)] if saved else [] # 将其当做旧的工作目录选项

        result = await select_or_custom(  # 选择上一次的工作目录或 输入新的 工作目录路径
            t("chat.workdir.select"),
            choices,
            custom_label=t("chat.workdir.custom"),
            custom_prompt=t("chat.workdir.custom_prompt"),
        )
        if not result: # 不选择则直接退出此命令
            return

        new_path = Path(result) # 将选择的 工作目录路径 转化 成 Path对象
        if not new_path.exists(): # 判断路径是否存在
            render_error(t("chat.workdir.not_exist"))
            return

        self.workplace_path = new_path # 将实例路径 切换到 新选择的路径
        self._skill_loader = None  # 工作目录变了，失效Skill路径缓存
        os.chdir(self.workplace_path) # cd <new-path>
        save_workplace(self.workplace_path) # 缓存当前的新的工作目录路径，作为下次可快捷选择的路径

        # 重建子目录（skill和会话 等目录）
        self._ensure_chat_dir(self.workplace_path)

        # 关闭旧 checkpointer（会话） 连接，重建会话和 agent
        await self._rebuild_agent(rebuild_session=True)

        await self._init_git() # git init
        render_success(t("chat.workdir.current", path=self.workplace_path))

    # 打开项目github主页
    async def _cmd_homepage(self, _arg: str) -> None:
        import webbrowser

        from chcode.config import HOMEPAGE_URL

        render_success(t("chat.opening", url=HOMEPAGE_URL))
        webbrowser.open(HOMEPAGE_URL)

    # 中英切换
    async def _cmd_lang(self, _arg: str) -> None:
        """运行时切换 UI 语言并持久化。"""
        from chcode.i18n import set_language, get_language
        from chcode.config import save_language

        zh_label = "中文 (Chinese)"
        en_label = "English"
        current = get_language()
        from questionary import Style
        _no_bg = Style([("highlighted", "noinherit"), ("selected", "noinherit")])
        action = await select(
            t("cmd.lang"),
            [zh_label, en_label],
            default=zh_label if current == "zh" else en_label,
            style=_no_bg,
        )
        if action is None:
            return
        new_lang = "zh" if action == zh_label else "en"
        set_language(new_lang)
        save_language(new_lang)
        render_success(
            t("lang.saved_zh") if new_lang == "zh" else t("lang.saved_en")
        )

    async def _cmd_help(self, _arg: str) -> None:
        from rich.table import Table

        table = Table(title=t("chat.help.title")) # 创建表格实例
        table.add_column(t("chat.help.col_cmd"), style="cyan") # 添加命令列
        table.add_column(t("chat.help.col_desc")) # 添加命令描述列
        for cmd, desc_key in SLASH_COMMANDS.items(): # 添加 每个命令的 命令名 和 命令描述
            table.add_row(cmd, t(desc_key))
        console.print(table) # 打印出来

    async def _cmd_quit(self, _arg: str) -> None:
        raise EOFError()

    # ─── 消息管理命令 ──────────────────────────────────

    async def _cmd_messages(self, _arg: str) -> None:
        """管理历史消息：编辑、分叉、删除"""
        if not self.agent or not self.session_mgr: # 检查agent实例和 会话管理器 是否存在
            render_error(t("chat.messages.no_agent"))
            return

        state = await self.agent.aget_state(self.session_mgr.config) # 取出 agent 当前 state
        messages: list[BaseMessage] = state.values.get("messages", []) # 从state中获取消息列表

        groups = _group_messages_by_turn(messages) # 按照 从某个HumanMessage到下一个HumanMessage的上一个消息为一组 来对消息进行分组
        if not groups: # 消息列表中没有消息
            render_warning(t("chat.messages.none"))
            return

        # 变量化选项标签
        edit_label = t("chat.messages.edit")
        fork_label = t("chat.messages.fork")
        delete_label = t("chat.messages.delete")
        back_label = t("common.back")
        while True:
            # 第一步：选择操作类型
            action = await select(t("chat.messages.select_op"), [edit_label, fork_label, delete_label])
            if not action:
                return

            # 构建选项列表（带返回选项）
            options = []
            for idx, group in enumerate(groups): # 遍历消息分组
                display = _get_group_display(group) # 获取消息组中第一个HumanMessage的前60个字符
                options.append(f"[{idx + 1}] {display}") # [消息组索引] 第一条HumanMessage消息的前60个字符

            if action == delete_label:
                # 多选 消息组
                chosen_list = await checkbox(
                    t("chat.messages.select_delete"), options
                )
                if not chosen_list:
                    continue  # 没有选择消息就回车 则 返回操作选择

                ok = await confirm(
                    t("chat.messages.delete_confirm", count=len(chosen_list)), default=False
                ) # 是否确认删除
                if not ok:
                    continue

                delete_ids = [] # 初始化 要被删除的 消息 的 消息id 的列表
                for chosen in chosen_list: # 遍历消息组选项
                    try:
                        sel_idx = int(chosen.split("]")[0].replace("[", "")) - 1
                        if 0 <= sel_idx < len(groups):
                            delete_ids.extend([m.id for m in groups[sel_idx]])
                    except (ValueError, IndexError):
                        continue

                if not delete_ids:
                    render_error(t("chat.messages.no_valid"))
                    continue

                await self._delete_messages(delete_ids)
                render_success(t("chat.messages.deleted_groups", count=len(chosen_list)))
                return

            # 编辑 / 分叉：单选一条消息组
            if action == edit_label:
                hint = t("chat.messages.edit_hint")
            else:
                hint = t("chat.messages.fork_hint")

            select_options = options + [back_label]
            chosen = await select(hint, select_options)
            if not chosen:
                return
            if chosen == back_label:
                continue

            # 解析选择
            try:
                sel_idx = int(chosen.split("]")[0].replace("[", "")) - 1
                if sel_idx < 0 or sel_idx >= len(groups):
                    render_error(t("chat.messages.invalid"))
                    continue
            except (ValueError, IndexError):
                render_error(t("chat.messages.invalid"))
                continue

            if action == edit_label:
                target_group = groups[sel_idx]
                edit_msg = None
                for msg in target_group:
                    if msg.type == "human":
                        edit_msg = msg
                        break

                if not edit_msg:
                    render_warning(t("chat.messages.no_human"))
                    continue

                ok = await confirm(
                    t("chat.messages.edit_confirm"),
                    default=False,
                )
                if not ok:
                    continue

                no_need_ids, all_ids = _collect_ids_from_group(
                    sel_idx, groups
                )

                if self.git and self.git_manager:
                    try:
                        result = await asyncio.to_thread(
                            self.git_manager.rollback, no_need_ids, all_ids
                        )
                        if isinstance(result, int) and not isinstance(result, bool):
                            self._git_cp_count = result
                    except Exception as e:
                        render_warning(t("chat.git.rollback_failed", error=e))

                await self._delete_messages(no_need_ids)

                self._edit_buffer = get_text_content(edit_msg.content)
                render_success(t("chat.messages.loaded_to_input"))
                return

            elif action == fork_label:
                ok = await confirm(
                    t("chat.messages.fork_confirm", idx=sel_idx + 1), default=True
                )
                if not ok:
                    continue

                no_need_ids, all_ids = _collect_ids_from_group(
                    sel_idx, groups
                )

                saved = load_workplace()
                choices = [str(saved)] if saved else []

                new_path_str = await select_or_custom(
                    t("chat.messages.select_new_workdir"),
                    choices,
                    custom_label=t("chat.workdir.custom"),
                    custom_prompt=t("chat.workdir.custom_prompt"),
                )
                if not new_path_str:
                    continue

                new_path = Path(new_path_str)
                if not new_path.exists():
                    render_error(t("chat.workdir.not_exist"))
                    continue

                old_path = self.workplace_path

                self.workplace_path = new_path
                os.chdir(self.workplace_path)
                save_workplace(self.workplace_path)

                self._ensure_chat_dir(self.workplace_path)

                if old_path != new_path:
                    render_info(t("chat.messages.copying"))
                    try:
                        await asyncio.to_thread(self._copy_dir, old_path, new_path)
                        # 复制影子仓库以保留检查点数据
                        old_cp = old_path / ".chat" / "cp-repo"
                        new_cp = new_path / ".chat" / "cp-repo"
                        if old_cp.exists() and old_cp.is_dir():
                            await asyncio.to_thread(
                                shutil.copytree, old_cp, new_cp, dirs_exist_ok=True
                            )
                        sessions_path = self.workplace_path / ".chat" / "sessions"
                        if sessions_path.exists():
                            await asyncio.to_thread(shutil.rmtree, sessions_path)
                            sessions_path.mkdir(exist_ok=True)
                    except Exception:
                        import traceback

                        tb = traceback.format_exc()
                        render_error(t("chat.messages.copy_failed", tb=tb))
                        self.workplace_path = old_path
                        os.chdir(self.workplace_path)
                        return

                await self._rebuild_agent(rebuild_session=True)

                need_messages = []
                for i, group in enumerate(groups):
                    need_messages.extend(group)
                    if i == sel_idx:
                        break

                await self.agent.aupdate_state(
                    self.session_mgr.config,
                    {"messages": need_messages},
                )

                # 先初始化 git
                await self._init_git()

                # 回滚工作目录
                if self.git and self.git_manager:
                    try:
                        result = await asyncio.to_thread(
                            self.git_manager.rollback, no_need_ids, all_ids
                        )
                        if isinstance(result, int) and not isinstance(result, bool):
                            self._git_cp_count = result
                    except Exception as e:
                        render_warning(t("chat.git.rollback_failed", error=e))

                render_success(t("chat.messages.fork_done", path=self.workplace_path))
                await self._load_conversation()
                return

    async def _cleanup_last_turn(self, append_msg: str | None = None) -> list[BaseMessage] | None:
        """查找最后一组消息：若无 AIMessage 则删除整组并返回该组，否则追加错误消息返回 None

        用于统一 _handle_agent_error 和 _handle_cancel 的共同逻辑：
        找到最后一组消息（以最后一个 HumanMessage 开头），
        判断当前组是否有 AIMessage，分别处理。
        """
        try:
            state = await self.agent.aget_state(self.session_mgr.config)
            messages: list[BaseMessage] = state.values.get("messages", [])

            last_human_idx = -1
            for i, msg in enumerate(messages):
                if isinstance(msg, HumanMessage):
                    last_human_idx = i

            if last_human_idx >= 0:
                current_group = messages[last_human_idx:]
                has_ai = any(isinstance(m, AIMessage) for m in current_group)

                if not has_ai:
                    await self._delete_messages([m.id for m in current_group])
                    return current_group

            if append_msg:
                error_msg = AIMessage(
                    append_msg,
                    additional_kwargs={"composed": True},
                )
                await self.agent.aupdate_state(
                    self.session_mgr.config,
                    {"messages": [error_msg]},
                    as_node="model",
                )
        except Exception:
            pass
        return None

    async def _handle_agent_error(self, error: Exception) -> None:
        """Agent 出错时：当前组无 AIMessage 则删除整组，否则保存错误消息"""
        await self._cleanup_last_turn(t("chat.agent_error", error=error))
        # 如果没有删除整组（已有 AIMessage），错误消息已在 _cleanup_last_turn 中追加

    async def _handle_cancel(self, user_input: str) -> None:
        """取消时：当前组无 AIMessage 则删除整组并回填输入框，否则追加停止消息"""
        deleted = await self._cleanup_last_turn(t("chat.agent_stopped"))
        if deleted is not None:
            self._interrupt_buffer = user_input.strip()

    async def _delete_messages(self, message_ids: list[str]) -> None:
        """删除指定消息"""
        if not self.agent or not self.session_mgr:
            return

        # 使用 RemoveMessage 删除
        remove_messages = [RemoveMessage(id=mid) for mid in message_ids]
        await self.agent.aupdate_state(
            self.session_mgr.config,
            {"messages": remove_messages},
        )

    def _copy_dir(self, src: Path, dst: Path):
        """复制目录（同步版本）"""
        for item in src.iterdir():
            if item.name.startswith("."):
                continue
            if item.stem.lower() in self.WINDOWS_RESERVED_NAMES:
                print(t("chat.copy.skip_reserved", name=item.name))
                continue
            dest_item = dst / item.name
            if item.is_dir():
                try:
                    shutil.copytree(item, dest_item, dirs_exist_ok=True)
                except Exception as e:
                    print(t("chat.copy.dir_failed", name=item.name, error=e))
            else:
                try:
                    shutil.copy2(item, dest_item)
                except Exception as e:
                    print(t("chat.copy.file_failed", name=item.name, error=e))

    # ─── 对话处理 ──────────────────────────────────────

    async def _process_input(self, user_input: str) -> None:
        """处理用户输入并调用 agent"""
        self._processing = True
        self._stop_requested = False

        accumulated_content = ""
        ai_started = False

        try:
            # 多模态模型：检测图片/视频路径并嵌入消息
            from chcode.utils.multimodal import (
                is_multimodal_model,
                extract_media_paths,
                build_multimodal_message,
            )

            current_model = (self.model_config or {}).get("model", "")
            if self.workplace_path and is_multimodal_model(current_model):
                media_paths = extract_media_paths(user_input, self.workplace_path)
                if media_paths:
                    message = build_multimodal_message(user_input, media_paths)
                    input_data = {"messages": message}
                    render_info(t("chat.media_embedded", count=len(media_paths)))
                else:
                    input_data = {"messages": user_input}
            else:
                input_data = {"messages": user_input}

            # 保存原始输入，用于模型切换后重试时重置 input_data
            _original_input_data = input_data

            if self._skill_loader is None:
                from chcode.utils.skill_loader import SkillLoader

                self._skill_loader = SkillLoader(
                    [
                        self.workplace_path / ".chat/skills",
                        Path.home() / ".chat/skills",
                    ]
                )

            skill_agent_context = SkillAgentContext(
                skill_loader=self._skill_loader,
                working_directory=self.workplace_path,
                model_config=self.model_config or INNER_MODEL_CONFIG,
                thread_id=self.session_mgr.thread_id,
                yolo=self.yolo,
            )

            while True:
                interrupt_chunk = None

                try:
                    async for m, i in self.agent.astream(
                            input_data,
                            self.session_mgr.config,
                            stream_mode=["messages", "updates"],
                            context=skill_agent_context,
                    ):
                        if self._stop_requested:
                            raise asyncio.CancelledError()

                        if m == "messages":
                            content = get_text_content(i[0].content)
                            additional_kwargs = i[0].additional_kwargs

                            if additional_kwargs.get("hide", ""):
                                continue

                            if isinstance(i[0], AIMessageChunk):
                                reasoning = additional_kwargs.get("reasoning")
                                if reasoning:
                                    if (
                                            not _display._subagent_parallel
                                            and _display._subagent_count == 0
                                    ):
                                        console.print(reasoning, end="", style="dim")
                                if not ai_started:
                                    if not content:
                                        continue
                                    ai_started = True
                                    render_ai_start()
                                render_ai_chunk(content or "")
                                accumulated_content += content or ""

                            elif isinstance(i[0], ToolMessage):
                                ai_started = False

                        elif m == "updates" and "__interrupt__" in i:
                            interrupt_chunk = i

                except asyncio.CancelledError:
                    await self._handle_cancel(user_input)
                    _display.force_reset_display()
                    console.print(Text(f"\n{t('chat.interrupted')}", style="dim"), "\n")
                    break
                except ModelSwitchError:
                    # 需要切换到备用模型
                    fallback = get_fallback_model()
                    if fallback:
                        console.print(f"[yellow]{t('chat.switching_fallback', model=fallback.get('model', 'unknown'))}[/yellow]")
                        self.model_config = fallback
                        advance_fallback()
                        # 持久化到 model.json，确保模型列表显示一致
                        import copy
                        from chcode.config import load_model_json, save_model_json
                        _data = copy.deepcopy(load_model_json())
                        _old_default = _data.get("default", {})
                        _old_model = _old_default.get("model", "")
                        if _old_model and _old_model not in _data.get("fallback", {}):
                            _data.setdefault("fallback", {})[_old_model] = _old_default
                        _data["default"] = fallback
                        save_model_json(_data)
                        try:
                            await self._rebuild_agent()
                            # 重建 context 以使用新模型配置
                            skill_agent_context = SkillAgentContext(
                                skill_loader=self._skill_loader,
                                working_directory=self.workplace_path,
                                model_config=self.model_config or INNER_MODEL_CONFIG,
                                thread_id=self.session_mgr.thread_id,
                                yolo=self.yolo,
                            )
                            # 如果当前 input_data 是已消费的 Command(resume=...)，
                            # 重置为原始输入，避免复用已消费的 Command
                            if isinstance(input_data, Command):
                                input_data = _original_input_data
                            console.print(f"[green]{t('chat.switched_fallback_retry')}[/green]")
                            continue  # 用备用模型重试当前请求
                        except Exception as e:
                            render_error(t("chat.switch_failed", error=e))
                    else:
                        render_error(t("chat.no_more_fallback"))
                        await self._handle_agent_error(ModelSwitchError("所有模型均失败"))
                    break
                except openai.APIError as e:
                    render_error(t("chat.agent_error", error=e))
                    await self._handle_agent_error(e)
                    break
                except Exception as e:
                    render_error(t("chat.agent_error", error=e))
                    await self._handle_agent_error(e)
                    break

                if self._stop_requested:
                    break

                if interrupt_chunk is None:
                    break

                # HITL 审批
                decisions = await self._collect_decisions_async(interrupt_chunk)
                input_data = Command(resume={"decisions": decisions})

            if ai_started:
                render_ai_end()

            # 后处理（上下文更新 + Git 提交）放到后台，不阻塞输入框
            asyncio.create_task(self._post_process())

        finally:
            self._processing = False

    async def _post_process(self) -> None:
        """流式输出后的后台处理：更新上下文用量、Git 提交"""
        try:
            state = await self.agent.aget_state(self.session_mgr.config)
            messages = state.values.get("messages", [])
            max_ctx = (self.model_config.get("metadata") or {}).get(
                "context_length"
            ) or _DEFAULT_CONTEXT_WINDOW
            self._context_text = get_context_usage_text(messages, max_ctx)

            if self.git and self.git_manager:
                new_msgs = find_and_slice_from_end(messages, "human")
                ids = [m.id for m in new_msgs]
                result = await asyncio.to_thread(
                    self.git_manager.add_commit, "&".join(ids)
                )
                if isinstance(result, int) and not isinstance(result, bool):
                    self._git_cp_count = result
        except Exception:
            pass

    async def _collect_decisions_async(self, interrupt_chunk) -> list[dict]:
        """收集 HITL 决策"""
        console.print()  # 确保 AI 输出和 HITL 之间有换行
        decisions = []
        for interrupt in interrupt_chunk["__interrupt__"]:
            action_requests = interrupt.value["action_requests"]

            for action_request in action_requests:
                name = action_request["name"]
                args = action_request["args"]

                content = ""
                match name:
                    case "bash":
                        content = args.get("command", "")
                    case "write_file":
                        content = t("hitl.write_file", path=args.get('file_path'), content=args.get('content', '')[:200])
                    case "edit":
                        file_path = args.get("file_path", "")
                        old_str = args.get("old_string", "")
                        new_str = args.get("new_string", "")
                        render_warning(t("hitl.edit_modify", path=file_path))
                        import difflib
                        from rich.table import Table

                        # 查找 old_str 在文件中的起始行号
                        start_line = 1
                        try:
                            content = await asyncio.to_thread(
                                Path(file_path).read_text, encoding="utf-8"
                            )
                            for i, line in enumerate(content.splitlines(), 1):
                                if old_str.splitlines()[0] in line:
                                    start_line = i
                                    break
                        except Exception:
                            pass
                        old_lines = old_str.splitlines()
                        new_lines = new_str.splitlines()
                        table = Table(
                            show_header=False,
                            show_edge=False,
                            padding=(0, 1),
                            border_style="dim",
                        )
                        table.add_column("old", ratio=1)
                        table.add_column("new", ratio=1)
                        sm = difflib.SequenceMatcher(None, old_lines, new_lines)
                        old_num = start_line
                        new_num = start_line
                        for tag, i1, i2, j1, j2 in sm.get_opcodes():
                            if tag == "equal":
                                for k in range(i2 - i1):
                                    table.add_row(
                                        Text(
                                            f"  {old_num:>3}  {old_lines[i1 + k]}",
                                            style="dim",
                                        ),
                                        Text(
                                            f"  {new_num:>3}  {new_lines[j1 + k]}",
                                            style="dim",
                                        ),
                                    )
                                    old_num += 1
                                    new_num += 1
                            elif tag == "replace":
                                max_len = max(i2 - i1, j2 - j1)
                                for k in range(max_len):
                                    old_text = (
                                        Text(
                                            f"{old_num:>3} - {old_lines[i1 + k]}",
                                            style="red",
                                        )
                                        if k < i2 - i1
                                        else None
                                    )
                                    new_text = (
                                        Text(
                                            f"{new_num:>3} + {new_lines[j1 + k]}",
                                            style="green",
                                        )
                                        if k < j2 - j1
                                        else None
                                    )
                                    table.add_row(old_text, new_text)
                                    if k < i2 - i1:
                                        old_num += 1
                                    if k < j2 - j1:
                                        new_num += 1
                            elif tag == "delete":
                                for k in range(i2 - i1):
                                    table.add_row(
                                        Text(
                                            f"{old_num:>3} - {old_lines[i1 + k]}",
                                            style="red",
                                        )
                                    )
                                    old_num += 1
                            elif tag == "insert":
                                for k in range(j2 - j1):
                                    table.add_row(
                                        None,
                                        Text(
                                            f"{new_num:>3} + {new_lines[j1 + k]}",
                                            style="green",
                                        ),
                                    )
                                    new_num += 1
                        console.print(table)
                        content = None  # 已直接渲染，跳过通用渲染

                if self.yolo:
                    select_action = True
                else:
                    if content is not None:
                        render_warning(f"[HITL] {name}")
                        console.print(Text(f"  {content[:500]}", style="dim"))
                    result = await select(
                        t("hitl.action"),
                        [t("hitl.approve"), t("hitl.reject")],
                    )
                    select_action = result != t("hitl.reject") if result else False

                extra = {}
                if not select_action:
                    extra["message"] = t("hitl.user_rejected")
                decision = {"type": "approve" if select_action else "reject"}
                decision.update(extra)
                decisions.append(decision)

        return decisions

    async def _load_conversation(self) -> None:
        """加载当前会话的对话历史并渲染"""
        if not self.agent:
            return
        try:
            state = await self.agent.aget_state(self.session_mgr.config)
            messages = state.values.get("messages", [])
            render_conversation(messages)
        except Exception as e:
            render_error(t("chat.load_conv_failed", error=e))
