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
from typing import Annotated, Any, Awaitable, Callable, NotRequired, TypedDict

import httpx
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    before_agent,
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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI
from chcode.utils.multimodal import is_multimodal_model
from chcode.utils.skill_loader import SkillAgentContext
from chcode.utils.project_memory import build_memory_reminder, get_memory_enabled
from chcode.display import console
from chcode.i18n import t
from chcode.utils.tool_result_pipeline import (
    clean_tool_output,
    truncate_large_result,
    enforce_per_turn_budget,
    reset_budget_state,  # noqa: F401  # 重新导出供其他模块使用
)

import aiosqlite


# ─── 内置默认模型配置 ──────────────────────────────────

import os


class ToolCallSnapshot(TypedDict):
    name: str
    args_key: str
    result_key: str


class ToolStormTracker(TypedDict):
    previous: ToolCallSnapshot
    exact_streak: int
    varying_result_streak: int
    varying_args_streak: int


class State(AgentState):
    tool_storm: NotRequired[Annotated[ToolStormTracker | None, lambda _old, new: new]]
    # 会话（线程）级记忆开关：线程首次运行由 seed_memory_flag 播种，
    # 之后不可变（LastValue 合并，随 checkpointer 持久化）。缺省视为
    # 开启以兼容无此键的旧 checkpoint。
    memory_enabled: NotRequired[bool]


INNER_MODEL_CONFIG = {
    "model": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "base_url": "https://api-inference.modelscope.cn/v1",
    "api_key": os.getenv("ModelScopeToken"),
    "temperature": 1,
    "top_p": 1,
    "stream_usage": True,
    "extra_body": {"stream": True},
    "metadata": {"context_length": 256000},
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
    for key in (
        "command",
        "file_path",
        "pattern",
        "query",
        "url",
        "question",
        "task",
        "filePath",
        "skill_name",
        "path",
        "prompt",
        "image_path",
        "section",
    ):
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

    start_evt: dict = {
        "type": "tool_start",
        "tool": tool_name,
        "summary": summary,
        "ts": time.time(),
    }
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
        # 判断工具是否失败：直接 ToolMessage 看 status，Command 则穿透 update.messages
        if isinstance(result, ToolMessage):
            failed = result.status == "error"
        else:
            msgs = (
                result.update.get("messages", [])
                if isinstance(result.update, dict)
                else []
            )
            if isinstance(msgs, ToolMessage):
                msgs = [msgs]
            failed = isinstance(msgs, (list, tuple)) and any(
                isinstance(m, ToolMessage) and m.status == "error" for m in msgs
            )
        end_evt: dict = {
            "type": "tool_end",
            "tool": tool_name,
            "success": not failed,
            "ts": time.time(),
        }
        if tool_name == "agent":
            end_evt["subagent_type"] = args.get("subagent_type", "general-purpose")
            end_evt["subagent_tag"] = start_evt.get("subagent_tag", "")
        if tag:
            end_evt["subagent"] = tag
        _ipc_send(end_evt)
        return result
    except Exception:
        end_evt = {
            "type": "tool_end",
            "tool": tool_name,
            "success": False,
            "ts": time.time(),
        }
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
                content=t("agent.vision_native_filter"),
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
            # retry_count = 已重试次数（不含首次失败），仅在进入重试时递增
            if retry_count >= max_retries:
                fallback = _load_fallback_config()
                if fallback:
                    console.print(
                        f"[yellow]{t('agent.switch_to_fallback', count=retry_count)}[/yellow]"
                    )
                    raise ModelSwitchError(t("agent.switch_error"))
                console.print(f"[red]{t('agent.no_fallback_giveup', error=e)}[/red]")
                raise

            delay_idx = min(retry_count, len(RETRY_DELAYS) - 1)
            delay = RETRY_DELAYS[delay_idx]
            retry_count += 1

            console.print(
                f"[yellow]{t('agent.retry_in', count=retry_count, max=max_retries, delay=delay, error=e)}[/yellow]"
            )

            for _ in range(int(delay)):
                await asyncio.sleep(1)


# 工具清单提示（vision 两分支共用 — 文本单一事实源，字节稳定保前缀缓存）。
# update_memory 行按会话开关条件注入：开关会话内固定，故会话内字节仍稳定。
_TOOLS_PROMPT_HEAD = """Tools:
- bash: execute shell commands and scripts. Stop immediately if the user refuses.
- read_file: view file content; write_file: create or save files; edit: modify existing files. Always read before write, prefer edit over write_file.
"""

_UPDATE_MEMORY_PROMPT_LINE = "- update_memory: save durable project knowledge (commands, conventions, prohibitions, pitfalls) to CHCODE.md; keep entries brief and constraint-style.\n"

_TOOLS_PROMPT_TAIL = """- glob: find files by name pattern; grep: search file contents with regex; list_dir: browse directory structure.
- web_search: search the Internet; web_fetch: fetch and read a URL's content.
- ask_user: present choices to the user and collect their input or confirmation.
- todo_write: create and manage a task list for complex multi-step work.
- load_skill: when a request matches a skill's description, load it first to get detailed instructions."""

# 无原生视觉能力的模型追加的 vision 工具行
_VISION_TOOL_PROMPT = "- vision: analyze an image or video file using a vision model. Use when the user provides an image/video path or asks about visual content. Supports PNG, JPG, GIF, BMP, WebP, TIFF, MP4, MOV, AVI, MKV, WebM. The user can paste file paths directly in chat."

# CHCODE.md 维护指引（记忆开启时注入；记忆内容由 inject_project_memory
# 以 <system-reminder> 元消息前置注入，会话内字节稳定以保前缀缓存）
_MEMORY_GUIDE_PROMPT = """

Project Memory (CHCODE.md):
- CHCODE.md is the persistent project memory; its contents are provided in the conversation context. Whenever you learn a durable, reusable fact about this project — common commands, package manager type, build/test/verify steps, coding conventions, prohibitions, pitfalls encountered — save it immediately with the update_memory tool.
- Every entry MUST be brief, action-related, and phrased as a constraint (MUST / NEVER / ALWAYS), never a vague suggestion.
- When an entry becomes outdated or wrong, fix it with mode="replace"; remove entries that no longer apply.
- NEVER save: complete API documentation, session history or changelogs, empty slogans, expired rules, or temporary task state."""


@before_agent
def seed_memory_flag(state: State, runtime) -> dict[str, Any] | None:
    """线程首次运行把配置值播种为会话记忆开关；已有值不动（不可变）。

    /memory 只改配置值并提示新会话生效：/new、/workdir 等产生的新线程
    首次运行经此处取当前配置；同线程后续运行返回 None，开关随
    checkpoint 持久化保持不变。
    """
    if state.get("memory_enabled") is None:
        return {"memory_enabled": get_memory_enabled()}
    return None


@wrap_model_call
async def filter_memory_tools(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """记忆关闭的会话按请求过滤掉 update_memory（工具始终全量绑定）。

    与提示词工具行、记忆注入读同一个 state 键，三方同源；按请求
    过滤无需重建 agent，新旧会话形状自然一致。
    """
    if request.state.get("memory_enabled", True):
        return await handler(request)
    return await handler(
        request.override(
            tools=[
                t for t in request.tools if getattr(t, "name", "") != "update_memory"
            ]
        )
    )


@dynamic_prompt
async def load_skills(request: ModelRequest) -> str:
    """构建 system prompt — Level 1: 注入所有 Skills 元数据"""
    skill_loader = request.runtime.context.skill_loader
    os_name = sys.platform
    model_config = request.runtime.context.model_config
    model_name = model_config.get("model", "")

    native_vision = is_multimodal_model(model_name)

    memory_enabled = request.state.get("memory_enabled", True)
    tools_prompt = (
        _TOOLS_PROMPT_HEAD
        + (_UPDATE_MEMORY_PROMPT_LINE if memory_enabled else "")
        + _TOOLS_PROMPT_TAIL
    )
    base_prompt = (
        f"You are a coding assistant. OS: {os_name}. CWD: {request.runtime.context.working_directory}.\n\n"
        f"{tools_prompt}"
    )
    if native_vision:
        base_prompt += (
            "\n\n Guidelines:\n"
            "- Never create .md/README files unless explicitly asked.\n"
            "- You have native vision capability. When the user sends an image or video file path, the "
            "image/video is already embedded in the message — analyze it directly. Do NOT call the vision tool."
        )
    else:
        base_prompt += (
            f"\n{_VISION_TOOL_PROMPT}\n\n"
            " Guidelines:\n"
            "- Never create .md/README files unless explicitly asked.\n"
            "- When the user sends an image or video file path, use vision to understand it before responding."
        )

    # 动态注入可用子 agent 列表
    yolo = request.runtime.context.yolo
    agents_section = "\n\nSub-agents:\n- Explore: codebase exploration and search\n- Plan: design implementation plans"
    if yolo:
        agents_section += "\n- general-purpose: full-capability tasks including reading, writing, and executing code"
    base_prompt += agents_section

    # CHCODE.md 维护指引 — 记忆关闭的会话整体省略（工具行同上条件注入）
    if memory_enabled:
        base_prompt += _MEMORY_GUIDE_PROMPT

    return await asyncio.to_thread(skill_loader.build_system_prompt, base_prompt)


@wrap_model_call
async def inject_project_memory(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    每次模型调用注入项目记忆（注入/剥离式），主/子代理共用：
    - CHCODE.md 冻结块包成 <system-reminder> 元消息前置到消息流最前
      （会话内字节稳定以保前缀缓存）
    - 挂在用户消息 metadata（memory_note）上的外部变更提醒，展开拼进
      该消息 content 尾部 — 状态里只存 metadata，展示层天然不可见；
      发送副本同步剥离该 metadata，避免随 additional_kwargs 序列化
      进 API payload 造成内容重复传输
    - 子代理消息不携带 memory_note（轮询只在主循环发生），展开对其
      天然不生效
    - 会话（线程）记忆开关关闭（state memory_enabled=False）时不做
      任何注入，请求原样放行；子代理图经输入播种同一键
    """
    if not request.state.get("memory_enabled", True):
        return await handler(request)

    workdir = request.runtime.context.working_directory
    reminder = await asyncio.to_thread(build_memory_reminder, workdir)

    messages = [HumanMessage(content=reminder)]
    for m in request.messages:
        note = getattr(m, "additional_kwargs", {}).get("memory_note")
        if isinstance(m, HumanMessage) and note:
            if isinstance(m.content, str):
                new_content = f"{m.content}\n\n{note}"
            else:  # 多模态列表 content：追加文本块
                new_content = list(m.content) + [{"type": "text", "text": note}]
            kw = {k: v for k, v in m.additional_kwargs.items() if k != "memory_note"}
            messages.append(
                m.model_copy(update={"content": new_content, "additional_kwargs": kw})
            )
        else:
            messages.append(m)
    return await handler(request.override(messages=messages))


@wrap_model_call
async def load_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """动态加载模型"""
    model_config = request.runtime.context.model_config
    kwargs = dict(model_config)
    kwargs.setdefault("timeout", httpx.Timeout(connect=10, read=10, write=60, pool=10))
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
            messages[i] = msg.model_copy(
                update={"content": truncated, "additional_kwargs": new_kwargs}
            )
            changed = True
    if changed:
        messages = enforce_per_turn_budget(
            messages, budget=200_000, workplace=workplace
        )
        return await handler(request.override(messages=messages))
    return await handler(request)


# 工具风暴阻断提示：按模式描述具体行为
_STORM_DETAIL = {
    "exact": "调用同一工具，且工具参数和返回结果完全相同",
    "varying_result": "使用相同工具和参数，但返回结果持续波动",
    "varying_args": "调用同一工具并频繁更换参数，但问题仍未收敛",
}


@wrap_tool_call
async def tool_call_storm_block(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:
    name = request.tool_call.get("name", "")
    args_key = json.dumps(
        request.tool_call.get("args", {}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    tracker = request.state.get("tool_storm")

    # 并行批量工具调用：streak 不确定，执行但重置 tracker
    last_ai_msg = next(
        (
            m
            for m in reversed(request.state.get("messages", []))
            if isinstance(m, AIMessage)
        ),
        None,
    )
    if last_ai_msg and len(last_ai_msg.tool_calls) > 1:
        result = await handler(request)
        if isinstance(result, Command):
            return result
        return Command(update={"tool_storm": None, "messages": [result]})

    # 判断是否触发阻断（阈值：完全重复 3、同参异果 5、同工具异参 8）
    reason: str | None = None
    if tracker:
        previous = tracker["previous"]
        same_name = name == previous["name"]
        same_args = args_key == previous["args_key"]
        if same_name and same_args and tracker["exact_streak"] >= 3:
            reason = "exact"
        elif same_name and same_args and tracker["varying_result_streak"] >= 5:
            reason = "varying_result"
        elif same_name and not same_args and tracker["varying_args_streak"] >= 8:
            reason = "varying_args"

    if reason is not None:
        message = ToolMessage(
            content=(
                "【工具调用风暴限制触发】\n"
                f"你已连续多次{_STORM_DETAIL[reason]}，已阻断本次调用。\n"
                "请复盘当前解题思路：调整查询参数、更换其他可用工具，或换全新推理策略，"
                "不要重复无意义工具调用，调用任意其他类型的工具 1 次，即可重置限制并继续使用本工具。"
            ),
            tool_call_id=request.tool_call["id"],
            status="error",
        )
        return Command(update={"tool_storm": None, "messages": [message]})

    result = await handler(request)
    if isinstance(result, Command):
        # 控制流指令（如 native command）：原样透传，不参与风暴计数。
        # 注意：tracker 在此既不更新也不重置，理论上若模型连续 N 次同工具同参后
        # 插入一个返回 Command 的操作、再重复同一调用，会基于过期 streak 误阻断。
        # 实际不触发：当前所有工具都返回 ToolMessage，无工具返回 Command。
        return result

    # 更新 storm tracker：与上次快照比对，累加命中的 streak（其余重置为 1）
    current: ToolCallSnapshot = {
        "name": name,
        "args_key": args_key,
        "result_key": json.dumps(
            {"content": result.content, "status": result.status},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    }
    next_tracker: ToolStormTracker = {
        "previous": current,
        "exact_streak": 1,
        "varying_result_streak": 1,
        "varying_args_streak": 1,
    }
    if tracker:
        previous = tracker["previous"]
        same_name = current["name"] == previous["name"]
        same_args = current["args_key"] == previous["args_key"]
        same_result = current["result_key"] == previous["result_key"]

        if same_name and same_args and same_result:
            next_tracker["exact_streak"] = tracker["exact_streak"] + 1
        elif same_name and same_args:
            next_tracker["varying_result_streak"] = tracker["varying_result_streak"] + 1
        elif same_name:
            next_tracker["varying_args_streak"] = tracker["varying_args_streak"] + 1

    return Command(update={"tool_storm": next_tracker, "messages": [result]})


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
            "update_memory": {"allowed_decisions": ["approve", "reject"]},
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
    from chcode.config import load_model_json, _DEFAULT_CONTEXT_WINDOW

    data = load_model_json()
    fallback = data.get("fallback", {})
    if fallback:
        current_model = cfg.get("model", "")
        filtered = [v for k, v in fallback.items() if v.get("model") != current_model]
        set_fallback_models(filtered)

    # 摘要触发阈值 = 上下文窗口的 90%（自定义 context_length 优先，缺失回退默认）
    ctx_window = (cfg.get("metadata") or {}).get(
        "context_length"
    ) or _DEFAULT_CONTEXT_WINDOW
    summary_trigger = int(ctx_window * 0.9)

    agent = create_agent(
        model,
        _get_all_tools(),
        middleware=[
            seed_memory_flag,
            restrict_agent_type,
            emit_tool_events,
            tool_call_storm_block,
            handle_tool_errors,
            filter_vision_tool,
            emit_thinking_events,
            detect_parallel_agents,
            tool_result_budget,
            load_skills,
            filter_memory_tools,
            inject_project_memory,
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
        state_schema=State,
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
    """获取所有工具（延迟导入避免循环依赖）。

    update_memory 始终全量绑定；记忆关闭的会话由 filter_memory_tools
    中间件按线程 state 过滤，无需重建 agent。
    """
    from chcode.utils.tools import ALL_TOOLS

    return ALL_TOOLS
