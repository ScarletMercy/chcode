"""
主聊天 REPL — 类 Claude Code 终端体验

prompt_toolkit 多行输入 + rich 流式输出 + 斜杠命令 + HITL 审批
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import AsyncIterator

import openai
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.shortcuts.prompt import CompleteStyle
from prompt_toolkit.styles import Style

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ToolMessage,
    RemoveMessage,
    HumanMessage,
    BaseMessage,
)
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
    render_status,
    render_ai_start,
    render_ai_chunk,
    render_ai_end,
    render_tool_call,
    get_context_usage_text,
    render_output_log_list,
    render_output_log_detail,
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
    get_context_window_size,
)
from chcode.session import SessionManager
from chcode.utils.skill_loader import SkillAgentContext
from chcode.agent_setup import (
    build_agent,
    create_checkpointer,
    INNER_MODEL_CONFIG,
    reset_budget_state,
)
from chcode.skill_manager import manage_skills
from chcode.utils.git_checker import check_git_availability
from chcode.utils.git_manager import GitManager
from chcode.utils.tools import _select_with_other_async


# ─── 命令自动补全 ──────────────────────────────────────

SLASH_COMMANDS = {
    "/new": "新会话",
    "/model": "模型管理",
    "/skill": "技能管理",
    "/history": "历史会话",
    "/compress": "压缩会话",
    "/git": "Git 状态",
    "/mode": "切换 Common/Yolo 模式",
    "/workdir": "切换工作目录",
    "/status": "显示状态栏",
    "/edit": "编辑历史消息",
    "/fork": "从某条消息创建分支",
    "/delete": "删除历史消息",
    "/output": "查看 Agent 输出日志",
    "/help": "显示帮助",
    "/quit": "退出",
}


class SlashCommandCompleter(Completer):
    """斜杠命令自动补全器 - 输入 / 时触发下拉列表"""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # 当输入 / 时触发补全
        if text.startswith("/"):
            partial = text[1:].lower()

            for cmd, desc in SLASH_COMMANDS.items():
                cmd_name = cmd[1:]  # 去掉 /
                if cmd_name.startswith(partial):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display=cmd,
                        display_meta=desc,
                    )


# ─── 辅助函数 ──────────────────────────────────────────


def find_and_slice_from_end(lst, x):
    """从后往前查找第一个 type==x 的元素，返回从该元素到末尾的切片"""
    for i in range(len(lst) - 1, -1, -1):
        if lst[i].type == x:
            return lst[i:]
    return []


def _group_messages_by_turn(messages: list) -> list[list]:
    """
    将消息按轮次分组（参考 chagent 逻辑）
    从一个 HumanMessage 开始，到下一个 HumanMessage 之前为一组
    """
    groups = []
    current_group = []

    for msg in messages:
        if msg.type == "human":
            if current_group:
                groups.append(current_group)
            current_group = [msg]
        else:
            current_group.append(msg)

    if current_group:
        groups.append(current_group)

    return groups


def _get_group_display(group: list) -> str:
    """获取消息组的显示文本（以 HumanMessage 内容为代表）"""
    for msg in group:
        if msg.type == "human":
            content = msg.content[:60].replace("\n", " ")
            if len(msg.content) > 60:
                content += "..."
            return content
    return "(空消息组)"


def _collect_ids_from_group(
    group_index: int, groups: list, mode: str = "edit"
) -> tuple[list[str], list[str]]:
    """
    收集要删除的消息 ID
    参考 chagent fork_message 逻辑：从目标 HumanMessage 开始，删除之后的所有消息

    Args:
        group_index: 目标组索引
        groups: 所有消息组
        mode: "edit" 删除目标组及之后, "fork" 只删除目标组之后（保留目标组）

    Returns:
        (no_need_ids, all_ids): 要删除的消息 ID 列表，所有消息 ID 列表
    """
    all_ids = [m.id for group in groups for m in group]
    no_need_ids = []

    for i, group in enumerate(groups):
        if mode == "edit":
            # edit: 从目标组开始删除
            if i >= group_index:
                no_need_ids.extend([m.id for m in group])
        elif mode == "fork":
            # fork: 从目标组之后开始删除（保留目标组）
            if i > group_index:
                no_need_ids.extend([m.id for m in group])

    return no_need_ids, all_ids


# ─── 主聊天类 ──────────────────────────────────────────


class ChatREPL:
    def __init__(self):
        self.workplace_path: Path | None = None
        self.model_config: dict = {}
        self.yolo = False
        self.agent = None
        self.checkpointer = None
        self.session_mgr: SessionManager | None = None
        self.git_manager: GitManager | None = None
        self.git = False
        self._stop_requested = False
        self._processing = False
        # 初始化 prompt-toolkit 会话（用于命令自动补全）
        self._prompt_session = None
        # 编辑缓冲区（用于 /edit 命令）
        self._edit_buffer: str | None = None
        # 中断恢复缓冲区（中断时将内容填回输入框，不进入编辑模式）
        self._interrupt_buffer: str | None = None
        # 上下文用量缓存
        self._context_text: str = ""
        # Agent 输出日志收集
        self._agent_output_logs: list[dict] = []
        self._current_log_turns: list[dict] = []
        self._current_log_index: int = 0

    # ─── 清理 ────────────────────────────────────────

    async def close(self) -> None:
        """关闭资源（aiosqlite 连接等）"""
        if self.checkpointer is not None:
            try:
                await self.checkpointer.conn.close()
            except Exception:
                pass
            self.checkpointer = None

    # ─── 初始化 ────────────────────────────────────────

    async def initialize(self) -> bool:
        """初始化：加载配置、设置工作目录、构建 agent"""
        ensure_config_dir()

        self.workplace_path = Path.cwd()
        os.chdir(self.workplace_path)

        chat_dir = self.workplace_path / ".chat"
        chat_dir.mkdir(exist_ok=True)
        (chat_dir / "sessions").mkdir(exist_ok=True)
        (chat_dir / "skills").mkdir(exist_ok=True)

        self.session_mgr = SessionManager(self.workplace_path)

        self.model_config = get_default_model_config() or {}
        if not self.model_config:
            config = await first_run_configure()
            if config is None:
                return False
            self.model_config = config

        # 创建 checkpointer
        db_path = self.workplace_path / ".chat" / "sessions" / "checkpointer.db"
        self.checkpointer = await create_checkpointer(db_path)

        # 构建 agent（可能较慢，放线程）
        console.print("[dim]构建 Agent...[/dim]")
        self.agent = await asyncio.to_thread(
            build_agent,
            self.model_config,
            self.checkpointer,
            None,
            self.yolo,
        )

        # 初始化 Git（subprocess.run 会阻塞事件循环）
        await self._init_git()

        # 初始化命令历史
        self._init_readline_history()

        return True

    def _init_readline_history(self):
        """初始化 readline 历史（跨会话保存）"""
        try:
            import readline

            history_path = Path.home() / ".chat" / "history"
            history_path.parent.mkdir(exist_ok=True)
            if history_path.exists():
                readline.read_history_file(str(history_path))
            readline.set_history_length(1000)
        except ImportError:
            pass

    def _save_readline_history(self):
        """保存 readline 历史"""
        try:
            import readline

            history_path = Path.home() / ".chat" / "history"
            history_path.parent.mkdir(exist_ok=True)
            readline.write_history_file(str(history_path))
        except ImportError:
            pass

        return True

    async def _init_git(self) -> None:
        """初始化 Git"""
        is_available, status, version = await asyncio.to_thread(check_git_availability)
        if is_available:
            self.git_manager = GitManager(str(self.workplace_path))
            if not self.git_manager.is_repo():
                await asyncio.to_thread(self.git_manager.init)
            self.git = True

    # ─── 主循环 ────────────────────────────────────────

    async def run(self) -> None:
        """主聊天循环"""
        render_welcome()

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
                    console.print(Text("\n再见！", style="dim"))
                    break
            except EOFError:
                break
            except Exception as e:
                render_error(f"Unexpected error: {e}")

    async def _get_input(self) -> str | None:
        """获取用户输入（使用 prompt-toolkit 实现命令自动补全）"""
        import re
        import shutil

        # 检查是否有编辑缓冲区或中断恢复缓冲区
        edit_mode = self._edit_buffer is not None
        interrupt_mode = self._interrupt_buffer is not None

        # 初始化 prompt session（带命令自动补全 + 底部状态栏）
        if self._prompt_session is None:
            completer = SlashCommandCompleter()

            # 自定义按键：Enter 提交，Alt+Enter 换行
            kb = KeyBindings()

            @kb.add("enter")
            def _submit(event):
                event.current_buffer.validate_and_handle()

            @kb.add("escape", "enter")  # Alt+Enter → 换行
            def _newline(event):
                event.current_buffer.insert_text("\n")

            def _bottom_toolbar():
                width = shutil.get_terminal_size().columns
                sep = "\u2500" * width
                parts = []
                model = self.model_config.get("model", "未设置")
                parts.append(model)
                if hasattr(self, "_context_text") and self._context_text:
                    clean = re.sub(r"\[/?\w+\]", "", self._context_text)
                    parts.append(clean)
                parts.append("普通模式" if not self.yolo else "YOLO 模式")
                if self.git and self.git_manager and self.git_manager.is_repo():
                    parts.append(f"Git ({self.git_manager.count_checkpoints()} cp)")
                wp = str(self.workplace_path) if self.workplace_path else ""
                if wp:
                    parts.append(f"cwd: {wp}")
                status = "  │  ".join(parts)
                return HTML(f"<ansiblue>{sep}</ansiblue>\n{status}")

            self._prompt_session = PromptSession(
                multiline=True,
                key_bindings=kb,
                completer=completer,
                complete_while_typing=True,
                reserve_space_for_menu=0,
                bottom_toolbar=_bottom_toolbar,
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

            def _dynamic_buffer_height():
                buff = self._prompt_session.default_buffer
                if buff.complete_state is not None:
                    n = len(buff.complete_state.completions)
                    needed = min(n + 2, 16)
                    return Dimension(min=needed, max=needed)
                return Dimension(min=1, max=1)

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
            # 如果有编辑缓冲区或中断恢复缓冲区，预填充到输入框
            if edit_mode:
                default_text = self._edit_buffer
                self._edit_buffer = None  # 清除缓冲区
            elif interrupt_mode:
                default_text = self._interrupt_buffer
                self._interrupt_buffer = None  # 清除缓冲区
            else:
                default_text = ""

            width = shutil.get_terminal_size().columns
            sep = "\u2500" * width
            prompt_text = (
                f"{sep}\n > " if not edit_mode else f"{sep}\n \u270f\ufe0f 编辑模式> "
            )

            # 使用 prompt-toolkit 获取输入（支持命令自动补全）
            result = await asyncio.to_thread(
                self._prompt_session.prompt,
                HTML(f"<ansiblue>{prompt_text}</ansiblue>"),
                default=default_text,
            )
            if result is not None:
                self._save_readline_history()
            return result
        except (EOFError, KeyboardInterrupt):
            return None

    # ─── 斜杠命令 ──────────────────────────────────────

    async def _handle_command(self, cmd: str) -> None:
        """处理斜杠命令"""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/new": self._cmd_new,
            "/model": self._cmd_model,
            "/skill": self._cmd_skill,
            "/history": self._cmd_history,
            "/compress": self._cmd_compress,
            "/git": self._cmd_git,
            "/mode": self._cmd_mode,
            "/workdir": self._cmd_workdir,
            "/edit": self._cmd_edit,
            "/fork": self._cmd_fork,
            "/delete": self._cmd_delete,
            "/help": self._cmd_help,
            "/quit": self._cmd_quit,
            "/status": self._cmd_status,
            "/output": self._cmd_output,
        }

        handler = handlers.get(command)
        if handler:
            await handler(arg)
        else:
            render_warning(f"未知命令: {command}，输入 /help 查看帮助")

    async def _cmd_new(self, _arg: str) -> None:
        reset_budget_state()
        self.session_mgr.new_session()
        render_success("新会话已开始")
        self._render_status_bar()

    async def _cmd_model(self, arg: str) -> None:
        if arg == "new":
            config = await configure_new_model()
        elif arg == "edit":
            config = await edit_current_model()
        elif arg == "switch":
            config = await switch_model()
        else:
            action = await select(
                "模型管理:",
                [
                    "新建模型 (/model new)",
                    "编辑当前模型 (/model edit)",
                    "切换模型 (/model switch)",
                ],
            )
            if action is None:
                return
            if "新建" in action:
                config = await configure_new_model()
            elif "编辑" in action:
                config = await edit_current_model()
            elif "切换" in action:
                config = await switch_model()
            else:
                return

        if config:
            self.model_config = config
            # 重建 agent
            self.agent = await asyncio.to_thread(
                build_agent,
                self.model_config,
                self.checkpointer,
                None,
                self.yolo,
            )
            self._render_status_bar()

    async def _cmd_skill(self, _arg: str) -> None:
        if not self.session_mgr:
            render_error("请先初始化工作目录")
            return
        await manage_skills(self.session_mgr)

    async def _cmd_history(self, _arg: str) -> None:
        if not self.session_mgr:
            return
        sessions = self.session_mgr.list_sessions()
        if not sessions:
            render_warning("没有历史会话")
            return

        sessions = sessions[-50:]  # 只显示最近 50 个
        action = await select_or_custom(
            "选择历史会话:",
            sessions,
            custom_label="返回",
        )
        if action is None or action == "返回":
            return

        op = await select("操作:", ["加载此会话", "删除此会话", "返回"])
        if op == "加载此会话":
            self.session_mgr.set_thread(action)
            await self._load_conversation()
            self._render_status_bar()
        elif op == "删除此会话":
            ok = await confirm(f"确定删除会话 {action}？", default=False)
            if ok:
                self.session_mgr.delete_session(action)
                if action == self.session_mgr.thread_id:
                    self._cmd_new("")
                render_success("会话已删除")

    async def _cmd_compress(self, _arg: str) -> None:
        if not self.model_config:
            render_warning("请先配置模型")
            return

        ok = await confirm("确定压缩当前会话？", default=True)
        if not ok:
            return

        render_info("压缩中...")
        try:
            state = await self.agent.aget_state(self.session_mgr.config)
            messages: list[BaseMessage] = state.values["messages"]

            # 分离历史消息和最近消息
            recent_messages = []
            recent_message_ids = []
            recent_count = 0
            for msg in reversed(messages):
                recent_messages.append(msg)
                recent_message_ids.append(msg.id)
                if isinstance(msg, HumanMessage):
                    recent_count += 1
                    if recent_count == 2:
                        break

            pre_messages = []
            for msg in messages:
                if msg.id not in recent_message_ids:
                    msg.additional_kwargs["composed"] = True
                    pre_messages.append(msg)

            from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI

            model = EnhancedChatOpenAI(**self.model_config)

            human_msg = HumanMessage(
                content='以你的角度用第二人称压缩会话，严格按以下JSON格式输出，不要使用markdown代码块：\n{{"summary": "压缩内容"}}',
                additional_kwargs={"hide": True, "composed": True},
            )

            try:
                raw_resp = await asyncio.to_thread(
                    model.invoke, pre_messages + [human_msg]
                )
                import re

                content = raw_resp.content.strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*\n?", "", content)
                    content = re.sub(r"\n?```\s*$", "", content)
                data = json.loads(content)
                ai_content = data.get("summary", "")
                if not ai_content:
                    ai_content = "会话压缩失败: LLM 返回结果缺少 summary 字段"
            except Exception as e:
                ai_content = f"会话压缩失败: {e}"
                human_msg.additional_kwargs["composed"] = True

            if ai_content.startswith("会话压缩失败"):
                ai_message = AIMessage(
                    ai_content,
                    additional_kwargs={"error": True, "composed": True},
                    usage_metadata={
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                )
            else:
                ai_message = AIMessage(
                    f"历史对话已压缩: {ai_content}",
                    additional_kwargs={"hide": True},
                )

            await self.agent.aupdate_state(
                self.session_mgr.config,
                {"messages": pre_messages + [human_msg, ai_message] + recent_messages},
                as_node="model",
            )
            await self._load_conversation()
            render_success("会话压缩完成")
        except Exception as e:
            render_error(f"压缩失败: {e}")

    async def _cmd_git(self, _arg: str) -> None:
        if not self.git_manager:
            is_available, status, version = await asyncio.to_thread(
                check_git_availability
            )
            if is_available:
                render_success(f"Git {version}")
                await self._init_git()
            else:
                render_error(f"Git 不可用: {status}")
                return

        if self.git_manager.is_repo():
            count = self.git_manager.count_checkpoints()
            render_success(f"Git 仓库已初始化 ({count} 个检查点)")
        else:
            render_warning("Git 仓库未初始化")

    async def _cmd_mode(self, _arg: str) -> None:
        action = await select(
            "选择模式:",
            ["Common (手动批准风险操作)", "Yolo (自动批准所有操作)"],
        )
        if action is None:
            return
        self.yolo = "Yolo" in action
        # 重建 agent 以更新 HITL 配置
        self.agent = await asyncio.to_thread(
            build_agent,
            self.model_config,
            self.checkpointer,
            None,
            self.yolo,
        )
        mode_str = "Yolo" if self.yolo else "Common"
        render_success(f"已切换到 {mode_str} 模式")

    async def _cmd_workdir(self, _arg: str) -> None:
        saved = load_workplace()
        if saved:
            choices = [str(saved), "自定义路径..."]
        else:
            choices = ["自定义路径..."]

        result = await select_or_custom("选择工作目录:", choices)
        if not result:
            return

        new_path = Path(result)
        if not new_path.exists():
            render_error("路径不存在")
            return

        self.workplace_path = new_path
        os.chdir(self.workplace_path)
        save_workplace(self.workplace_path)

        # 重建子目录
        chat_dir = self.workplace_path / ".chat"
        chat_dir.mkdir(exist_ok=True)
        (chat_dir / "sessions").mkdir(exist_ok=True)
        (chat_dir / "skills").mkdir(exist_ok=True)

        # 重建会话和 agent
        self.session_mgr = SessionManager(self.workplace_path)
        db_path = self.workplace_path / ".chat" / "sessions" / "checkpointer.db"
        self.checkpointer = await create_checkpointer(db_path)
        self.agent = await asyncio.to_thread(
            build_agent,
            self.model_config,
            self.checkpointer,
            None,
            self.yolo,
        )

        await self._init_git()
        render_success(f"工作目录: {self.workplace_path}")
        self._render_status_bar()

    async def _cmd_help(self, _arg: str) -> None:
        from rich.table import Table

        table = Table(title="命令列表")
        table.add_column("命令", style="cyan")
        table.add_column("说明")
        cmds = [
            ("/new", "新会话"),
            ("/model", "模型管理（新建/编辑/切换）"),
            ("/model new", "新建模型"),
            ("/model edit", "编辑当前模型"),
            ("/model switch", "切换模型"),
            ("/skill", "技能管理"),
            ("/history", "历史会话"),
            ("/compress", "压缩会话"),
            ("/git", "Git 状态"),
            ("/mode", "切换 Common/Yolo 模式"),
            ("/workdir", "切换工作目录"),
            ("/edit", "编辑历史消息"),
            ("/fork", "从某条消息创建分支"),
            ("/delete", "删除历史消息"),
            ("/output", "查看 Agent 输出日志"),
            ("/status", "显示状态栏"),
            ("/help", "显示此帮助"),
            ("/quit", "退出"),
        ]
        for cmd, desc in cmds:
            table.add_row(cmd, desc)
        console.print(table)

    async def _cmd_quit(self, _arg: str) -> None:
        raise EOFError()

    async def _cmd_status(self, _arg: str) -> None:
        self._render_status_bar()

    async def _cmd_output(self, arg: str) -> None:
        """查看 agent 输出日志"""
        if not self._agent_output_logs:
            render_warning("没有 agent 输出日志")
            return

        if not arg:
            # 显示日志列表
            log_entries = []
            for log in self._agent_output_logs:
                log_entries.append(
                    {
                        "index": log["index"],
                        "timestamp": log["timestamp"],
                        "turns": log["turns"],
                        "preview": log["preview"],
                    }
                )
            render_output_log_list(log_entries)
        else:
            # 查看具体日志
            if arg.lower() == "last":
                log = self._agent_output_logs[-1]
            else:
                try:
                    idx = int(arg) - 1
                    if idx < 0 or idx >= len(self._agent_output_logs):
                        render_error(f"日志索引 {arg} 无效")
                        return
                    log = self._agent_output_logs[idx]
                except ValueError:
                    render_error(f"无效的日志索引: {arg}")
                    return

            render_output_log_detail(
                {"user_input": log["user_input"], "turns": log["turns_data"]},
                log["index"],
            )

    # ─── 消息管理命令 ──────────────────────────────────

    async def _cmd_edit(self, _arg: str) -> None:
        """编辑历史消息"""
        if not self.agent or not self.session_mgr:
            render_error("Agent 未初始化")
            return

        state = await self.agent.aget_state(self.session_mgr.config)
        messages: list[BaseMessage] = state.values.get("messages", [])

        # 按轮次分组
        groups = _group_messages_by_turn(messages)
        if not groups:
            render_warning("没有可编辑的消息")
            return

        # 构建选项列表
        options = []
        for idx, group in enumerate(groups):
            display = _get_group_display(group)
            options.append(f"[{idx + 1}] {display}")

        # 用户选择
        chosen = await _select_with_other_async("选择要编辑的消息组:", options)
        if not chosen:
            return

        # 解析选择
        try:
            sel_idx = int(chosen.split("]")[0].replace("[", "")) - 1
            if sel_idx < 0 or sel_idx >= len(groups):
                render_error("无效的选择")
                return
        except (ValueError, IndexError):
            render_error("无效的选择")
            return

        target_group = groups[sel_idx]
        edit_msg = None
        for msg in target_group:
            if msg.type == "human":
                edit_msg = msg
                break

        if not edit_msg:
            render_warning("该组没有 HumanMessage")
            return

        # 确认
        ok = await confirm(
            f"确定编辑此消息组？编辑后将删除此消息组之后的所有内容。", default=False
        )
        if not ok:
            return

        # 收集要删除的消息 ID（从目标组开始）
        no_need_ids, all_ids = _collect_ids_from_group(sel_idx, groups, mode="edit")

        # Git rollback
        if self.git and self.git_manager:
            try:
                await asyncio.to_thread(self.git_manager.rollback, no_need_ids, all_ids)
            except Exception as e:
                render_warning(f"Git 回滚失败: {e}")

        # 删除消息
        await self._delete_messages(no_need_ids)

        # 将内容填入输入框
        self._edit_buffer = edit_msg.content
        render_success("消息已加载到输入框，修改后发送即可重新生成")

    async def _cmd_fork(self, _arg: str) -> None:
        """从某条消息创建分支"""
        if not self.agent or not self.session_mgr:
            render_error("Agent 未初始化")
            return

        state = await self.agent.aget_state(self.session_mgr.config)
        messages: list[BaseMessage] = state.values.get("messages", [])

        # 按轮次分组
        groups = _group_messages_by_turn(messages)
        if not groups:
            render_warning("没有可 Fork 的消息")
            return

        # 构建选项列表
        options = []
        for idx, group in enumerate(groups):
            display = _get_group_display(group)
            options.append(f"[{idx + 1}] {display}")

        # 用户选择 Fork 点
        chosen = await _select_with_other_async(
            "选择 Fork 点（此消息组将保留在分支中）:", options
        )
        if not chosen:
            return

        # 解析选择
        try:
            sel_idx = int(chosen.split("]")[0].replace("[", "")) - 1
            if sel_idx < 0 or sel_idx >= len(groups):
                render_error("无效的选择")
                return
        except (ValueError, IndexError):
            render_error("无效的选择")
            return

        fork_group = groups[sel_idx]

        # 确认
        ok = await confirm(f"确定从第 {sel_idx + 1} 条消息组创建分支？", default=True)
        if not ok:
            return

        # 收集要删除的消息 ID（只删除 Fork 点之后的组，保留 Fork 组）
        no_need_ids, all_ids = _collect_ids_from_group(sel_idx, groups, mode="fork")

        # 选择新工作目录
        saved = load_workplace()
        if saved:
            choices = [str(saved), "自定义路径..."]
        else:
            choices = ["自定义路径..."]

        new_path_str = await select_or_custom("选择新工作目录:", choices)
        if not new_path_str:
            return

        new_path = Path(new_path_str)
        if not new_path.exists():
            render_error("路径不存在")
            return

        # 保存旧路径
        old_path = self.workplace_path

        # 设置新工作目录
        self.workplace_path = new_path
        os.chdir(self.workplace_path)
        save_workplace(self.workplace_path)

        # 创建子目录
        chat_dir = self.workplace_path / ".chat"
        chat_dir.mkdir(exist_ok=True)
        (chat_dir / "sessions").mkdir(exist_ok=True)
        (chat_dir / "skills").mkdir(exist_ok=True)

        # 复制旧工作目录文件（如果不同）
        if old_path != new_path:
            render_info("复制工作目录文件...")
            try:
                await asyncio.to_thread(self._copy_dir, old_path, new_path)
                # 删除 sessions 目录（使用新的 checkpointer）
                sessions_path = self.workplace_path / ".chat" / "sessions"
                if sessions_path.exists():
                    import shutil

                    await asyncio.to_thread(shutil.rmtree, sessions_path)
                    sessions_path.mkdir(exist_ok=True)
            except Exception as e:
                render_warning(f"复制文件失败: {e}")

        # 重新初始化会话管理和 checkpointer
        self.session_mgr = SessionManager(self.workplace_path)
        db_path = self.workplace_path / ".chat" / "sessions" / "checkpointer.db"
        self.checkpointer = await create_checkpointer(db_path)

        # 重建 agent
        self.agent = await asyncio.to_thread(
            build_agent,
            self.model_config,
            self.checkpointer,
            None,
            self.yolo,
        )

        # 保留 Fork 点及之前的所有消息
        need_messages = []
        for i, group in enumerate(groups):
            need_messages.extend(group)
            if i == sel_idx:
                break

        # 更新状态
        await self.agent.aupdate_state(
            self.session_mgr.config,
            {"messages": need_messages},
        )

        # Git rollback（如果启用）
        if self.git and self.git_manager:
            try:
                await asyncio.to_thread(self.git_manager.rollback, no_need_ids, all_ids)
            except Exception:
                pass

        # 重新初始化 Git
        await self._init_git()

        render_success(f"分支已创建！工作目录: {self.workplace_path}")
        self._render_status_bar()

    async def _cmd_delete(self, _arg: str) -> None:
        """删除历史消息"""
        if not self.agent or not self.session_mgr:
            render_error("Agent 未初始化")
            return

        state = await self.agent.aget_state(self.session_mgr.config)
        messages: list[BaseMessage] = state.values.get("messages", [])

        # 按轮次分组
        groups = _group_messages_by_turn(messages)
        if not groups:
            render_warning("没有可删除的消息")
            return

        # 构建选项列表
        options = []
        for idx, group in enumerate(groups):
            display = _get_group_display(group)
            options.append(f"[{idx + 1}] {display}")

        # 多选
        chosen_list = await checkbox(
            "选择要删除的消息组（空格选择，回车确认）:", options
        )
        if not chosen_list:
            return

        # 确认
        ok = await confirm(f"确定删除 {len(chosen_list)} 个消息组？", default=False)
        if not ok:
            return

        # 解析选择并收集要删除的消息 ID
        delete_ids = []
        for chosen in chosen_list:
            try:
                sel_idx = int(chosen.split("]")[0].replace("[", "")) - 1
                if 0 <= sel_idx < len(groups):
                    delete_ids.extend([m.id for m in groups[sel_idx]])
            except (ValueError, IndexError):
                continue

        if not delete_ids:
            render_error("没有有效的选择")
            return

        # 删除消息
        await self._delete_messages(delete_ids)
        render_success(f"已删除 {len(chosen_list)} 个消息组")

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
        import shutil

        for item in src.iterdir():
            if item.name.startswith("."):
                continue  # 跳过隐藏文件/目录
            s = item
            d = dst / item.name
            if s.is_dir():
                if not d.exists():
                    d.mkdir(parents=True)
                self._copy_dir(s, d)
            else:
                shutil.copy2(str(s), str(d))

    def _render_status_bar(self) -> None:
        """状态栏由 bottom_toolbar 自动渲染，此方法仅用于触发刷新"""
        pass

    async def _update_context_usage(self) -> None:
        """从 agent state 更新上下文用量和 token 消耗缓存"""
        if not self.agent or not self.session_mgr:
            return
        try:
            state = await self.agent.aget_state(self.session_mgr.config)
            messages = state.values.get("messages", [])
            model_name = self.model_config.get("model", "")
            max_ctx = get_context_window_size(model_name)
            self._context_text = get_context_usage_text(messages, max_ctx)
        except Exception:
            pass

    # ─── 对话处理 ──────────────────────────────────────

    async def _process_input(self, user_input: str) -> None:
        """处理用户输入并调用 agent"""
        self._processing = True
        self._stop_requested = False

        accumulated_content = ""
        ai_started = False

        # 初始化日志收集
        self._current_log_turns = []
        self._current_log_index = len(self._agent_output_logs) + 1
        current_reasoning = ""
        current_response = ""

        try:
            input_data = {"messages": user_input}

            from chcode.utils.skill_loader import SkillLoader

            skill_agent_context = SkillAgentContext(
                skill_loader=SkillLoader(
                    [
                        self.workplace_path / ".chat/skills",
                        Path.home() / ".chat/skills",
                    ]
                ),
                working_directory=self.workplace_path,
                model_config=self.model_config or INNER_MODEL_CONFIG,
                thread_id=self.session_mgr.thread_id,
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
                            content = i[0].content
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
                                    current_reasoning += reasoning
                                if not ai_started:
                                    if not content:
                                        continue
                                    ai_started = True
                                    render_ai_start()
                                render_ai_chunk(content or "")
                                accumulated_content += content or ""
                                current_response += content or ""

                                # 收集工具调用信息
                                tool_calls = getattr(i[0], "tool_calls", None) or []
                                for tc in tool_calls:
                                    tc_name = tc.get("name", "unknown")
                                    tc_args = str(tc.get("args", {}))[:300]
                                    self._current_log_turns.append(
                                        {
                                            "type": "tool_call",
                                            "tool_name": tc_name,
                                            "content": f"参数: {tc_args}",
                                        }
                                    )

                            elif isinstance(i[0], ToolMessage):
                                ai_started = False
                                # 保存工具调用结果
                                tool_name = i[0].name or "unknown"
                                tool_result = str(content)[:2000]  # 限制长度
                                self._current_log_turns.append(
                                    {
                                        "type": "tool_result",
                                        "tool_name": tool_name,
                                        "content": tool_result,
                                    }
                                )

                        elif m == "updates" and "__interrupt__" in i:
                            interrupt_chunk = i

                except asyncio.CancelledError:
                    if user_input.strip():
                        self._interrupt_buffer = user_input.strip()
                    console.print(Text("\n[已中断]", style="dim"), "\n")
                    break
                except openai.APIError as e:
                    render_error(f"Agent 执行错误: {e}")
                    try:
                        error_msg = AIMessage(
                            f"Agent 执行错误: {e}",
                            additional_kwargs={"error": True, "composed": True},
                        )
                        await self.agent.aupdate_state(
                            self.session_mgr.config,
                            {"messages": [error_msg]},
                            as_node="model",
                        )
                    except Exception:
                        pass
                    break
                except Exception as e:
                    render_error(f"Agent 执行错误: {e}")
                    try:
                        error_msg = AIMessage(
                            f"Agent 执行错误: {e}",
                            additional_kwargs={"error": True, "composed": True},
                        )
                        await self.agent.aupdate_state(
                            self.session_mgr.config,
                            {"messages": [error_msg]},
                            as_node="model",
                        )
                    except Exception:
                        pass
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

                # 保存本轮日志
                if current_reasoning:
                    self._current_log_turns.append(
                        {
                            "type": "reasoning",
                            "content": current_reasoning,
                        }
                    )
                if current_response:
                    self._current_log_turns.append(
                        {
                            "type": "ai_response",
                            "content": current_response,
                        }
                    )

                if self._current_log_turns:
                    preview = (
                        current_response[:100] if current_response else "(无回复内容)"
                    )
                    from datetime import datetime

                    self._agent_output_logs.append(
                        {
                            "index": self._current_log_index,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "turns": len(self._current_log_turns),
                            "preview": preview,
                            "user_input": user_input[:200],
                            "turns_data": self._current_log_turns.copy(),
                        }
                    )
                    # 保存到文件
                    self._save_output_log()

            # 更新上下文用量并刷新状态栏
            await self._update_context_usage()
            self._render_status_bar()

            # Git 提交（静默）
            if self.git and self.git_manager:
                current_messages = (
                    await self.agent.aget_state(self.session_mgr.config)
                ).values.get("messages", [])
                new_msgs = find_and_slice_from_end(current_messages, "human")
                ids = [m.id for m in new_msgs]
                self.git_manager.add_commit("&".join(ids))

        finally:
            self._processing = False

    def _save_output_log(self) -> None:
        """将当前 agent 输出日志保存到文件"""
        if not self._current_log_turns or not self.workplace_path:
            return

        outputs_dir = self.workplace_path / ".chat" / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = outputs_dir / f"{timestamp}.log"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"# Agent Output Log #{self._current_log_index}\n")
            f.write(f"# Timestamp: {timestamp}\n")
            f.write(f"# User Input: {self._agent_output_logs[-1]['user_input']}\n")
            f.write("=" * 80 + "\n\n")

            for i, turn in enumerate(self._current_log_turns, 1):
                turn_type = turn.get("type", "unknown")
                content = turn.get("content", "")

                if turn_type == "ai_response":
                    f.write(f"## AI Response (Turn {i})\n\n")
                    f.write(content)
                    f.write("\n\n" + "-" * 60 + "\n\n")
                elif turn_type == "reasoning":
                    f.write(f"## Thinking (Turn {i})\n\n")
                    f.write(content)
                    f.write("\n\n" + "-" * 60 + "\n\n")
                elif turn_type == "tool_call":
                    tool_name = turn.get("tool_name", "unknown")
                    f.write(f"## Tool Call: {tool_name} (Turn {i})\n\n")
                    f.write(content)
                    f.write("\n\n" + "-" * 60 + "\n\n")
                elif turn_type == "tool_result":
                    tool_name = turn.get("tool_name", "unknown")
                    f.write(f"## Tool Result: {tool_name} (Turn {i})\n\n")
                    f.write(content[:5000])  # 限制长度
                    if len(content) > 5000:
                        f.write(f"\n... (truncated, {len(content) - 5000} more chars)")
                    f.write("\n\n" + "-" * 60 + "\n\n")

    async def _collect_decisions_async(self, interrupt_chunk) -> list[dict]:
        """收集 HITL 决策"""
        decisions = []
        for interrupt in interrupt_chunk["__interrupt__"]:
            action_requests = interrupt.value["action_requests"]
            review_configs = interrupt.value["review_configs"]
            review_dict = {
                i["action_name"]: i["allowed_decisions"] for i in review_configs
            }

            for action_request in action_requests:
                name = action_request["name"]
                args = action_request["args"]

                content = ""
                match name:
                    case "bash":
                        content = args.get("command", "")
                    case "write_file":
                        content = f"写入文件: {args.get('file_path')}\n内容: {args.get('content', '')[:200]}"
                    case "edit":
                        content = f"修改文件: {args.get('file_path')}\n{args.get('old_string', '')[:100]} -> {args.get('new_string', '')[:100]}"

                if self.yolo:
                    select_action = True
                else:
                    render_warning(f"[HITL] {name}")
                    console.print(Text(f"  {content[:500]}", style="dim"))
                    result = await select(
                        "操作:",
                        ["approve (批准)", "reject (拒绝)"],
                    )
                    select_action = result != "reject (拒绝)" if result else False

                extra = {}
                if not select_action:
                    extra["message"] = "用户已拒绝"
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
            render_error(f"加载对话失败: {e}")
