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

from chcode.display import (
    console,
    render_error,
    render_info,
    render_success,
    render_warning,
    render_welcome,
    render_conversation,
    render_status,
    render_ai_end,
    get_token_text,
)
from chcode.prompts import select, confirm, select_or_custom, text
from chcode.config import (
    get_default_model_config,
    load_workplace,
    save_workplace,
    configure_new_model,
    edit_current_model,
    switch_model,
    ensure_config_dir,
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
        console.print("[dim]初始化中...[/dim]")

        ensure_config_dir()

        # 加载工作目录
        wp = load_workplace()
        if wp:
            self.workplace_path = wp
        else:
            # 选择工作目录
            result = await text("输入工作目录路径 (留空使用当前目录):", default=str(Path.cwd()))
            if result:
                self.workplace_path = Path(result)
            else:
                self.workplace_path = Path.cwd()

        if not self.workplace_path or not self.workplace_path.exists():
            console.print("[red]工作目录无效[/red]")
            return False

        os.chdir(self.workplace_path)

        # 创建子目录
        chat_dir = self.workplace_path / ".chat"
        chat_dir.mkdir(exist_ok=True)
        (chat_dir / "sessions").mkdir(exist_ok=True)
        (chat_dir / "skills").mkdir(exist_ok=True)

        # 保存工作目录
        save_workplace(self.workplace_path)

        # 初始化会话管理
        self.session_mgr = SessionManager(self.workplace_path)

        # 加载模型配置
        self.model_config = get_default_model_config() or {}
        if not self.model_config:
            console.print("[yellow]未检测到模型配置[/yellow]")
            config = await configure_new_model()
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
            self.model_config, self.checkpointer, None, self.yolo,
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
        mode = "YOLO" if self.yolo else "Common"
        model = self.model_config.get("model", "")[:30]
        wp = str(self.workplace_path) if self.workplace_path else ""
        if len(wp) > 30:
            wp = "..." + wp[-27:]
        prompt_text = f"{mode} {model} {wp}\n> "

        # 初始化 prompt session（带命令自动补全）
        if self._prompt_session is None:
            completer = SlashCommandCompleter()
            self._prompt_session = PromptSession(
                completer=completer,
                complete_while_typing=True,  # 实时触发补全
                style=Style.from_dict({
                    "completion-menu.completion": "bg:#008888 #ffffff",
                    "completion-menu.completion.current": "bg:#00aaaa #000000",
                    "completion-menu.meta.completion": "bg:#008888 #ffffff",
                    "completion-menu.meta.completion.current": "bg:#00aaaa #000000",
                }),
            )

        try:
            # 使用 prompt-toolkit 获取输入（支持命令自动补全）
            result = await asyncio.to_thread(
                self._prompt_session.prompt,
                HTML(f"<ansiblue>{prompt_text}</ansiblue>")
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
            "/help": self._cmd_help,
            "/quit": self._cmd_quit,
            "/status": self._cmd_status,
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
                ["新建模型 (/model new)", "编辑当前模型 (/model edit)", "切换模型 (/model switch)"],
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
                self.model_config, self.checkpointer, None, self.yolo,
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
                raw_resp = await asyncio.to_thread(model.invoke, pre_messages + [human_msg])
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
                    usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
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
            is_available, status, version = await asyncio.to_thread(check_git_availability)
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
            self.model_config, self.checkpointer, None, self.yolo,
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
            self.model_config, self.checkpointer, None, self.yolo,
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

    def _render_status_bar(self) -> None:
        wp = str(self.workplace_path) if self.workplace_path else ""
        model = self.model_config.get("model", "未设置")
        git_str = ""
        if self.git and self.git_manager and self.git_manager.is_repo():
            git_str = f"Git ({self.git_manager.count_checkpoints()} cp)"
        render_status(
            workplace=wp,
            model=model,
            git_status=git_str,
            mode="Yolo" if self.yolo else "Common",
        )

    # ─── 对话处理 ──────────────────────────────────────

    async def _process_input(self, user_input: str) -> None:
        """处理用户输入并调用 agent"""
        self._processing = True
        self._stop_requested = False
        console.print(Text(f"\n> {user_input}", style="bold cyan"))

        accumulated_content = ""
        ai_started = False

        try:
            input_data = {"messages": user_input}
            pre_messages = (await self.agent.aget_state(self.session_mgr.config)).values.get("messages", [])

            from chcode.utils.skill_loader import SkillLoader
            skill_agent_context = SkillAgentContext(
                skill_loader=SkillLoader([
                    self.workplace_path / ".chat/skills",
                    Path.home() / ".chat/skills",
                ]),
                working_directory=self.workplace_path,
                model_config=self.model_config or INNER_MODEL_CONFIG,
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
                                    console.print(reasoning, end="", style="dim")
                                if not ai_started:
                                    if not content:
                                        continue
                                    ai_started = True
                                console.print(content, end="")
                                accumulated_content += content or ""

                            elif isinstance(i[0], ToolMessage):
                                ai_started = False
                                if content:
                                    name = i[0].name or "tool"
                                    console.print(Text(f"  {name}", style="dim"))

                        elif m == "updates" and "__interrupt__" in i:
                            interrupt_chunk = i

                except asyncio.CancelledError:
                    await self._write_interrupted_message(accumulated_content)
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

            # Git 提交（静默）
            if self.git and self.git_manager:
                current_messages = (await self.agent.aget_state(self.session_mgr.config)).values.get("messages", [])
                new_msgs = find_and_slice_from_end(current_messages, "human")
                ids = [m.id for m in new_msgs]
                self.git_manager.add_commit("&".join(ids))

        finally:
            self._processing = False

    async def _collect_decisions_async(self, interrupt_chunk) -> list[dict]:
        """收集 HITL 决策"""
        decisions = []
        for interrupt in interrupt_chunk["__interrupt__"]:
            action_requests = interrupt.value["action_requests"]
            review_configs = interrupt.value["review_configs"]
            review_dict = {i["action_name"]: i["allowed_decisions"] for i in review_configs}

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

    async def _write_interrupted_message(self, accumulated_content: str) -> None:
        """写入中断消息到 checkpoint"""
        try:
            state = await self.agent.aget_state(self.session_mgr.config)
            messages = state.values.get("messages", [])
            if messages and isinstance(messages[-1], AIMessage):
                return
            content = accumulated_content.strip()
            if not content:
                content = "[这条消息已被用户中断]"
            else:
                content += "\n\n[这条消息已被用户中断]"
            msg = AIMessage(
                content,
                additional_kwargs={"interrupted": True, "composed": True},
            )
            await self.agent.aupdate_state(
                self.session_mgr.config,
                {"messages": [msg]},
                as_node="model",
            )
        except Exception:
            pass

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