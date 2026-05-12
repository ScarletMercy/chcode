"""
Agent 构建 — 中间件注册、checkpointer 初始化
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import time
from pathlib import Path
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    dynamic_prompt,
    wrap_tool_call,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    HumanInTheLoopMiddleware,
)
from langchain.agents.middleware.context_editing import (
    ContextEditingMiddleware,
    ClearToolUsesEdit,
)
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_core.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI
from chcode.utils.modelscope_ratelimit import is_modelscope_model, get_modelscope_clients
from chcode.utils.multimodal import is_multimodal_model
from chcode.utils.skill_loader import SkillAgentContext
from chcode.display import console
from chcode.utils.tool_result_pipeline import (
    clean_tool_output,
    truncate_large_result,
    enforce_per_turn_budget,
    reset_budget_state,  # noqa: F401  # 重新导出供其他模块使用
)

import aiosqlite


# ─── 内置默认模型配置 ──────────────────────────────────

import os

INNER_MODEL_CONFIG = {
    "model": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "base_url": "https://api-inference.modelscope.cn/v1",
    "api_key": os.getenv("ModelScopeToken"),
    "temperature": 1,
    "top_p": 1,
    "stream_usage": True,
    "extra_body": {"stream": True},
}


# ─── 重试配置 ──────────────────────────────────────────

RETRY_DELAYS = [3, 10, 30, 60]
_fallback_models: list[dict] = []
_fallback_index: int = 0


def set_fallback_models(models: list[dict]) -> None:
    global _fallback_models, _fallback_index
    _fallback_models = models
    _fallback_index = 0


def get_fallback_model() -> dict | None:
    if _fallback_index < len(_fallback_models):
        return _fallback_models[_fallback_index]
    return None


def advance_fallback() -> None:
    global _fallback_index
    _fallback_index += 1


def _load_fallback_config() -> dict | None:
    """获取当前备用模型"""
    global _fallback_models
    if not _fallback_models:
        from chcode.config import load_model_json

        data = load_model_json()
        fallback = data.get("fallback", {})
        if not fallback:
            return None
        _fallback_models = list(fallback.values())

    return get_fallback_model()


# ─── 中间件 ──────────────────────────────────────────


_IPC_SOCK: socket.socket | None = None
_IPC_ADDR = ("127.0.0.1", 19876)


def _ipc_send(event: dict) -> None:
    global _IPC_SOCK
    try:
        if _IPC_SOCK is None:
            _IPC_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = json.dumps(event, ensure_ascii=False).encode("utf-8")
        _IPC_SOCK.sendto(data, _IPC_ADDR)
    except Exception:
        pass


@wrap_tool_call
async def restrict_agent_type(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], Command]
) -> Command | ToolMessage:
    if request.tool_call.get("name") == "agent":
        args = request.tool_call.get("args", {})
        if args.get("subagent_type") == "general-purpose":
            if _hitl_middleware is not None and _hitl_middleware.interrupt_on:
                args["subagent_type"] = "Explore"
    return await handler(request)


@wrap_tool_call
async def emit_tool_events(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], Command]
) -> Command | ToolMessage:
    tool_name = request.tool_call.get("name", "")
    args = request.tool_call.get("args", {})
    summary = ""
    for key in ("command", "file_path", "pattern", "query", "url", "question",
                "task", "filePath", "skill_name", "path", "prompt", "image_path"):
        if key in args:
            summary = str(args[key])[:80]
            break
    if not summary and "todos" in args:
        todos = args["todos"]
        if isinstance(todos, list) and todos:
            first = todos[0]
            if isinstance(first, dict):
                summary = first.get("content", str(first))[:80]
            else:
                summary = str(first)[:80]

    start_evt: dict = {"type": "tool_start", "tool": tool_name, "summary": summary, "ts": time.time()}
    if tool_name == "agent":
        sa_type = args.get("subagent_type", "general-purpose")
        sa_desc = args.get("description", "")[:30]
        start_evt["subagent_type"] = sa_type
        start_evt["subagent_tag"] = f"{sa_type}: {sa_desc}"
    try:
        from chcode.display import _current_agent_tag
        tag = _current_agent_tag.get(None)
    except Exception:
        tag = None
    if tag:
        start_evt["subagent"] = tag

    _ipc_send(start_evt)
    try:
        result = await handler(request)
        ok = not (isinstance(result, ToolMessage) and getattr(result, "status", None) == "error")
        end_evt: dict = {"type": "tool_end", "tool": tool_name, "success": ok, "ts": time.time()}
        if tool_name == "agent":
            end_evt["subagent_type"] = args.get("subagent_type", "general-purpose")
            end_evt["subagent_tag"] = start_evt.get("subagent_tag", "")
        if tag:
            end_evt["subagent"] = tag
        _ipc_send(end_evt)
        return result
    except Exception:
        end_evt = {"type": "tool_end", "tool": tool_name, "success": False, "ts": time.time()}
        if tool_name == "agent":
            end_evt["subagent_type"] = args.get("subagent_type", "general-purpose")
            end_evt["subagent_tag"] = start_evt.get("subagent_tag", "")
        _ipc_send(end_evt)
        raise


@wrap_model_call
async def emit_thinking_events(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    _ipc_send({"type": "thinking_start", "ts": time.time()})
    try:
        result = await handler(request)
        _ipc_send({"type": "thinking_end", "ts": time.time()})
        return result
    except Exception:
        _ipc_send({"type": "thinking_end", "ts": time.time()})
        raise


@wrap_model_call
async def detect_parallel_agents(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    result = await handler(request)
    if not result.result:
        return result
    ai_msg = result.result[0]
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        from chcode import display as _d
        agent_count = sum(1 for tc in ai_msg.tool_calls if tc.get("name") == "agent")
        if agent_count >= 2:
            _d._subagent_parallel = True
    return result


@wrap_tool_call
async def handle_tool_errors(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], Command]
) -> Command | ToolMessage:
    try:
        return await handler(request)
    except Exception as e:
        return ToolMessage(
            f"Tool error: Please check your input and try again ({e})",
            tool_call_id=request.tool_call["id"],
            status="error",
        )


class ModelSwitchError(Exception):
    """标记需要切换模型的异常"""
    pass


@wrap_tool_call
async def filter_vision_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Command],
) -> Command | ToolMessage:
    """多模态模型时屏蔽 vision 工具 — 模型自带视觉能力"""
    tool_name = request.tool_call.get("name", "")
    if tool_name == "vision":
        model_config = request.runtime.context.model_config
        model_name = model_config.get("model", "")

        if is_multimodal_model(model_name):
            return ToolMessage(
                content="当前模型支持原生视觉，图片/视频已直接嵌入消息，无需调用 vision 工具。请直接分析消息中的图片/视频内容。",
                tool_call_id=request.tool_call["id"],
                status="error",
            )
    return await handler(request)


@wrap_model_call
async def model_retry_with_backoff(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """指数级退避重试中间件 — 每次调用独立计数"""
    max_retries = 4

    retry_count = 0

    while True:
        try:
            return await handler(request)
        except Exception as e:
            retry_count += 1

            if retry_count >= max_retries:
                fallback = _load_fallback_config()
                if fallback:
                    console.print(f"[yellow]主模型重试{retry_count}次失败，切换到备用模型...[/yellow]")
                    raise ModelSwitchError("切换到备用模型")
                console.print(f"[red]请求失败，无备用模型可用，放弃请求\n  {e}[/red]")
                raise

            delay_idx = min(retry_count - 1, len(RETRY_DELAYS) - 1)
            delay = RETRY_DELAYS[delay_idx]

            console.print(f"[yellow]请求失败 ({retry_count}/{max_retries}), {delay}秒后重试...\n  {e}[/yellow]")

            await asyncio.sleep(delay)


@dynamic_prompt
async def load_skills(request: ModelRequest) -> str:
    """构建 system prompt — Level 1: 注入所有 Skills 元数据"""
    skill_loader = request.runtime.context.skill_loader
    os_name = sys.platform
    model_config = request.runtime.context.model_config
    model_name = model_config.get("model", "")

    native_vision = is_multimodal_model(model_name)

    if native_vision:
        base_prompt = f"""You are a coding assistant. OS: {os_name}. CWD: {request.runtime.context.working_directory}.

Tools:
- bash: execute shell commands and scripts. Stop immediately if the user refuses.
- read_file: view file content; write_file: create or save files; edit: modify existing files. Always read before write, prefer edit over write_file.
- glob: find files by name pattern; grep: search file contents with regex; list_dir: browse directory structure.
- web_search: search the Internet; web_fetch: fetch and read a URL's content.
- ask_user: present choices to the user and collect their input or confirmation.
- todo_write: create and manage a task list for complex multi-step work.
- load_skill: when a request matches a skill's description, load it first to get detailed instructions.

 Guidelines:
- Never create .md/README files unless explicitly asked.
- You have native vision capability. When the user sends an image or video file path, the image/video is already embedded in the message — analyze it directly. Do NOT call the vision tool."""
    else:
        base_prompt = f"""You are a coding assistant. OS: {os_name}. CWD: {request.runtime.context.working_directory}.

Tools:
- bash: execute shell commands and scripts. Stop immediately if the user refuses.
- read_file: view file content; write_file: create or save files; edit: modify existing files. Always read before write, prefer edit over write_file.
- glob: find files by name pattern; grep: search file contents with regex; list_dir: browse directory structure.
- web_search: search the Internet; web_fetch: fetch and read a URL's content.
- ask_user: present choices to the user and collect their input or confirmation.
- todo_write: create and manage a task list for complex multi-step work.
- load_skill: when a request matches a skill's description, load it first to get detailed instructions.
- vision: analyze an image or video file using a vision model. Use when the user provides an image/video path or asks about visual content. Supports PNG, JPG, GIF, BMP, WebP, TIFF, MP4, MOV, AVI, MKV, WebM. The user can paste file paths directly in chat.

 Guidelines:
- Never create .md/README files unless explicitly asked.
- When the user sends an image or video file path, use vision to understand it before responding."""

    # 动态注入可用子 agent 列表
    yolo = request.runtime.context.yolo
    agents_section = "\n\nSub-agents:\n- Explore: codebase exploration and search\n- Plan: design implementation plans"
    if yolo:
        agents_section += "\n- general-purpose: full-capability tasks including reading, writing, and executing code"
    base_prompt += agents_section

    return await asyncio.to_thread(skill_loader.build_system_prompt, base_prompt)


@wrap_model_call
async def load_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """动态加载模型"""
    model_config = request.runtime.context.model_config
    kwargs = dict(model_config)
    if is_modelscope_model(model_config):
        sync_client, async_client = get_modelscope_clients()
        kwargs["http_client"] = sync_client
        kwargs["http_async_client"] = async_client
    return await handler(request.override(model=EnhancedChatOpenAI(**kwargs)))


@wrap_model_call
async def fix_messages(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """过滤隐藏消息"""
    messages = request.messages
    real_messages = [m for m in messages if not m.additional_kwargs.get("composed", "")]
    if len(real_messages) == len(messages):
        return await handler(request)
    return await handler(request.override(messages=real_messages))


@wrap_model_call
async def tool_result_budget(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """工具结果截断和 token 预算控制"""
    workplace = request.runtime.context.working_directory
    messages = list(request.messages)
    changed = False
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and msg.content:
            if msg.additional_kwargs.get("_budget_ok"):
                continue
            cleaned = clean_tool_output(msg.content)
            truncated = truncate_large_result(
                cleaned,
                msg.name or "",
                msg.tool_call_id,
                workplace=workplace,
            )
            new_kwargs = {**msg.additional_kwargs, "_budget_ok": True}
            messages[i] = msg.model_copy(update={"content": truncated, "additional_kwargs": new_kwargs})
            changed = True
    if changed:
        messages = enforce_per_turn_budget(messages, budget=200_000, workplace=workplace)
        return await handler(request.override(messages=messages))
    return await handler(request)


# ─── Agent 构建 ──────────────────────────────────────────


class AsyncHITL(HumanInTheLoopMiddleware):
    """异步 HITL 中间件 — 审批在 chat loop 中处理"""

    async def awrap_model_call(self, request, handler):
        return await handler(request)


_hitl_middleware: AsyncHITL | None = None
_summarization_model: EnhancedChatOpenAI | None = None


def _build_interrupt_on(yolo: bool) -> dict:
    return (
        {}
        if yolo
        else {
            "bash": {"allowed_decisions": ["approve", "reject"]},
            "edit": {"allowed_decisions": ["approve", "reject"]},
            "write_file": {"allowed_decisions": ["approve", "reject"]},
        }
    )


def _dummy_model():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="placeholder", api_key="sk-placeholder", max_retries=0)


def build_agent(
    model_config: dict | None = None,
    checkpointer: AsyncSqliteSaver | None = None,
    yolo: bool = False,
) -> object:
    """构建 agent 实例"""
    global _hitl_middleware, _summarization_model

    cfg = model_config or INNER_MODEL_CONFIG
    model = _dummy_model()

    _hitl_middleware = AsyncHITL(interrupt_on=_build_interrupt_on(yolo))
    _summarization_model = EnhancedChatOpenAI(**cfg)

    # 加载 fallback 模型配置
    from chcode.config import load_model_json, get_context_window_size

    data = load_model_json()
    fallback = data.get("fallback", {})
    if fallback:
        current_model = cfg.get("model", "")
        filtered = [v for k, v in fallback.items() if v.get("model") != current_model]
        set_fallback_models(filtered)

    # 摘要触发阈值 = 上下文窗口的 90%
    model_name = cfg.get("model", "")
    ctx_window = get_context_window_size(model_name)
    summary_trigger = int(ctx_window * 0.9)

    agent = create_agent(
        model,
        _get_all_tools(),
        middleware=[
            restrict_agent_type,
            emit_tool_events,
            handle_tool_errors,
            filter_vision_tool,
            emit_thinking_events,
            detect_parallel_agents,
            tool_result_budget,
            load_skills,
            load_model,
            model_retry_with_backoff,
            fix_messages,
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=100_000,
                        keep=3,
                        exclude_tools=["read_file"],
                        placeholder="[Old tool result content cleared]",
                    )
                ]
            ),
            SummarizationMiddleware(
                model=_summarization_model,
                trigger=("tokens", summary_trigger),
                keep=("messages", 20),
            ),
            _hitl_middleware,
        ],
        context_schema=SkillAgentContext,
        checkpointer=checkpointer,
    )
    return agent


def update_hitl_config(yolo: bool) -> None:
    """运行时更新 HITL interrupt_on 配置，无需重建 agent"""
    if _hitl_middleware is not None:
        _hitl_middleware.interrupt_on = _build_interrupt_on(yolo)
    from chcode.utils.tools import update_agent_tool_desc
    update_agent_tool_desc(yolo)


def update_summarization_model(model_config: dict) -> None:
    """运行时更新 SummarizationMiddleware 的模型"""
    if _summarization_model is not None:
        new_model = EnhancedChatOpenAI(**model_config)
        for key in new_model.model_fields_set:
            try:
                if key in new_model.__dict__:
                    setattr(_summarization_model, key, new_model.__dict__[key])
            except (AttributeError, TypeError):
                pass


async def create_checkpointer(db_path: Path) -> AsyncSqliteSaver:
    """创建异步 SQLite checkpointer"""
    conn = await aiosqlite.connect(str(db_path))
    return AsyncSqliteSaver(conn)


def _get_all_tools() -> list:
    """获取所有工具（延迟导入避免循环依赖）"""
    from chcode.utils.tools import ALL_TOOLS

    return ALL_TOOLS
