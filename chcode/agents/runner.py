from __future__ import annotations

import asyncio
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    dynamic_prompt,
    ModelRequest,
)
from langchain_core.messages import HumanMessage

from chcode.agents.definitions import AgentDefinition
from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI
from chcode.utils.skill_loader import SkillLoader, SkillAgentContext
from chcode.agent_setup import handle_tool_errors, tool_result_budget
from chcode.agent_setup import emit_tool_events


@dynamic_prompt
async def _subagent_system_prompt(request: ModelRequest) -> str:
    return request.runtime.context.extra.get("system_prompt", "")


def _resolve_tools(
    agent_def: AgentDefinition,
    all_tools: list,
) -> list:
    result = []
    for t in all_tools:
        name = getattr(t, "name", None) or getattr(getattr(t, "func", None), "__name__", "")
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
        extra={"system_prompt": agent_def.system_prompt},
    )

    middleware = [
        emit_tool_events,
        handle_tool_errors,
        tool_result_budget,
        _subagent_system_prompt,
    ]

    from chcode.agent_setup import model_retry_with_backoff, ModelSwitchError

    middleware.append(model_retry_with_backoff)
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
        return f"Agent {agent_def.agent_type} timed out after {timeout_seconds}s.", True
    except ModelSwitchError:
        return f"Agent {agent_def.agent_type} 主模型失败，已切换备用模型，请重试", True
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
