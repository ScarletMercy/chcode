from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    dynamic_prompt,
    wrap_tool_call,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest

from chcode.agents.definitions import AgentDefinition
from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI
from chcode.utils.skill_loader import SkillLoader, SkillAgentContext
from chcode.utils.tool_result_pipeline import (
    clean_tool_output,
    truncate_large_result,
    enforce_per_turn_budget,
)


@wrap_tool_call
async def _handle_tool_errors(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], object]
) -> object:
    try:
        return await handler(request)
    except Exception as e:
        return ToolMessage(
            f"Tool error: Please check your input and try again ({e})",
            tool_call_id=request.tool_call["id"],
            status="error",
        )


@dynamic_prompt
async def _subagent_system_prompt(request: ModelRequest) -> str:
    return request.runtime.context.extra.get("system_prompt", "")


@wrap_model_call
async def _tool_result_budget(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    workplace = request.runtime.context.working_directory
    messages = list(request.messages)
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and msg.content:
            cleaned = clean_tool_output(msg.content)
            truncated = truncate_large_result(
                cleaned,
                msg.name or "",
                msg.tool_call_id,
                workplace=workplace,
            )
            messages[i] = msg.model_copy(update={"content": truncated})
    messages = enforce_per_turn_budget(messages, budget=200_000, workplace=workplace)
    return await handler(request.override(messages=messages))


def _resolve_tools(
    agent_def: AgentDefinition,
    all_tools: list,
) -> list:
    result = []
    for t in all_tools:
        name = getattr(t, "name", None) or getattr(t, "func", {}).get("__name__", "")
        if name == "agent":
            continue
        if name in agent_def.disallowed_tools:
            continue
        if agent_def.tools is not None and name not in agent_def.tools:
            continue
        result.append(t)
    return result


async def run_subagent(
    prompt: str,
    agent_def: AgentDefinition,
    model_config: dict,
    working_directory: Path,
    skill_loader: SkillLoader,
    timeout_seconds: int = 300,
    description: str = "",
) -> str:
    # 守卫：超时最小 300s
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
        extra={"system_prompt": agent_def.system_prompt},
    )

    middleware = [
        _handle_tool_errors,
        _tool_result_budget,
        _subagent_system_prompt,
    ]
    # 非 read-only 子代理继承主 agent 的 HITL 配置
    if not agent_def.read_only:
        from chcode.agent_setup import _hitl_middleware

        if _hitl_middleware is not None:
            middleware.append(_hitl_middleware)

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
        return f"Agent {agent_def.agent_type} timed out after {timeout_seconds}s."
    except Exception as e:
        return f"Agent {agent_def.agent_type} error: {e}"

    messages = result.get("messages", [])
    for msg in reversed(messages):
        if msg.type == "ai" and msg.content:
            content = msg.content
            if isinstance(content, list):
                parts = [b.get("text", "") for b in content if b.get("type") == "text"]
                content = "\n".join(parts)
            if content.strip():
                return content.strip()

    return "(Agent completed with no text output)"
