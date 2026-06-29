from __future__ import annotations

import asyncio
from pathlib import Path

from langchain.agents import create_agent
from typing import Callable

from langchain.agents.middleware import (
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from rich.text import Text

from chcode.agents.definitions import AgentDefinition
from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI
from chcode.utils.skill_loader import SkillLoader, SkillAgentContext
from chcode.utils.tool_result_pipeline import (
    clean_tool_output,
    truncate_large_result,
    enforce_per_turn_budget,
)
from chcode.i18n import t


@wrap_tool_call
async def _display_subagent_tools(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], Command]
) -> Command | ToolMessage:
    """单 agent 模式下缩进打印子 agent 的工具调用"""
    import chcode.display as _d

    if _d._subagent_count == 1 and not _d._subagent_parallel:
        tool_name = request.tool_call.get("name", "")
        args = request.tool_call.get("args", {})
        summary = ""
        for key in ("command", "file_path", "pattern", "query", "url", "question",
                    "task", "filePath", "skill_name", "path", "prompt", "image_path"):
            if key in args:
                summary = str(args[key])[:80]
                break
        if summary:
            _d.console.print(Text(f"    [{tool_name}] {summary}", style="dim cyan"))

    return await handler(request)


_BLOCKED_PREFIXES = (
    "mkdir", "touch", "rm ", "rm -", "cp ", "mv ", "rmdir",
    "chmod", "chown", "dd ", "mkfs", "format ", "del ",
    "git add", "git commit", "git push", "git checkout",
    "npm install", "pip install", "pip3 install",
)

_BLOCKED_TOKENS = (
    "remove-item",
    "format-volume",
    "reg delete",
    "reg add",
    "stop-process -force",
    "set-executionpolicy",
)


@wrap_tool_call
async def _restrict_bash(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], Command]
) -> Command | ToolMessage:
    if request.tool_call.get("name") == "bash":
        command = request.tool_call.get("args", {}).get("command", "")
        stripped = command.strip().lower()
        for prefix in _BLOCKED_PREFIXES:
            if stripped.startswith(prefix):
                return ToolMessage(
                    content=f"Blocked: '{prefix.strip()}' is not allowed in read-only mode.",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
        for token in _BLOCKED_TOKENS:
            if token in stripped:
                return ToolMessage(
                    content=f"Blocked: '{token}' is not allowed in read-only mode.",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
    return await handler(request)


@wrap_model_call
async def _tool_result_budget(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
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


@dynamic_prompt
async def _subagent_system_prompt(request: ModelRequest) -> str:
    return request.runtime.context.extra.get("system_prompt", "")


def _resolve_tools(
    agent_def: AgentDefinition,
    all_tools: list,
) -> list:
    result = []
    for tool in all_tools:
        name = getattr(tool, "name", None) or getattr(getattr(tool, "func", None), "__name__", "")
        if name == "agent":
            continue
        if name in agent_def.disallowed_tools:
            continue
        if agent_def.tools is not None and name not in agent_def.tools:
            continue
        result.append(tool)
    return result


async def run_subagent(
    prompt: str,
    agent_def: AgentDefinition,
    model_config: dict,
    working_directory: Path,
    skill_loader: SkillLoader,
    timeout_seconds: int = 300,
    description: str = "",
    yolo: bool = False,
) -> tuple[str, bool]:
    timeout_seconds = max(timeout_seconds, 300)
    from chcode.utils.tools import ALL_TOOLS

    filtered_tools = _resolve_tools(agent_def, ALL_TOOLS)

    cfg = dict(model_config)
    if agent_def.model:
        cfg = {**cfg, "model": agent_def.model}

    model = EnhancedChatOpenAI(**cfg)

    subagent_context = SkillAgentContext(
        skill_loader=skill_loader,
        working_directory=working_directory,
        model_config=cfg,
        yolo=yolo,
        extra={"system_prompt": agent_def.system_prompt},
    )

    from chcode.agent_setup import handle_tool_errors, emit_tool_events

    middleware = [
        emit_tool_events,
        _display_subagent_tools,
        handle_tool_errors,
        _tool_result_budget,
        _subagent_system_prompt,
    ]

    if agent_def.read_only:
        middleware.insert(1, _restrict_bash)

    from chcode.agent_setup import model_retry_with_backoff, ModelSwitchError

    middleware.append(model_retry_with_backoff)

    subagent = create_agent(
        model,
        filtered_tools,
        middleware=middleware,
        context_schema=SkillAgentContext,
    )

    try:
        result = await asyncio.wait_for(
            subagent.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"configurable": {"thread_id": f"subagent_{id(subagent)}"}},
                context=subagent_context,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return f"Agent {agent_def.agent_type} timed out after {timeout_seconds}s.", True
    except ModelSwitchError:
        return t("runner.fallback_switched", type=agent_def.agent_type), True
    except Exception as e:
        return f"Agent {agent_def.agent_type} error: {e}", True

    from chcode.utils import get_text_content
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if msg.type == "ai" and msg.content:
            content = get_text_content(msg.content)
            if content.strip():
                return content.strip(), False

    return "(Agent completed with no text output)", False
