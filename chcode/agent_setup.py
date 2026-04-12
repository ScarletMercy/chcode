"""
Agent 构建 — 中间件注册、checkpointer 初始化
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Callable

from langchain.agents import create_agent, AgentState
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
from langchain_core.messages import AIMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI
from chcode.utils.skill_loader import SkillLoader, SkillAgentContext
from chcode.utils.tool_result_pipeline import (
    clean_tool_output,
    truncate_large_result,
    enforce_per_turn_budget,
    reset_budget_state,
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


# ─── 中间件 ──────────────────────────────────────────

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


@dynamic_prompt
async def load_skills(request: ModelRequest) -> str:
    """构建 system prompt — Level 1: 注入所有 Skills 元数据"""
    skill_loader = request.runtime.context.skill_loader
    os_name = sys.platform
    base_prompt = f"""You are a helpful coding assistant with access to specialized skills.

Your capabilities include:
- Loading and using specialized skills for specific tasks
- Executing bash commands and scripts (current os: {os_name}). If user refuses, stop!
- When working with files: use `read_file` to view content, `write_file` to create or save, `edit` to modify existing files, `glob` to find files by name pattern, `grep` to search content within files, and `list_dir` to browse directory structure.
- When you need information from the Internet, use `web_search` or `web_fetch`.
- When you need to ask the user about choices, preferences, or options, please use the `ask_user` tool.

Current working directory: {request.runtime.context.working_directory}.
Skills Path: {request.runtime.context.skill_loader.skill_paths}

When a user request matches a skill's description, use the load_skill tool to get detailed instructions before proceeding."""

    return skill_loader.build_system_prompt(base_prompt)


@wrap_model_call
async def load_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """动态加载模型"""
    model_config = request.runtime.context.model_config
    return await handler(request.override(model=EnhancedChatOpenAI(**model_config)))


@wrap_model_call
async def fix_messages(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """过滤隐藏消息，统计 tokens"""
    messages = request.messages
    real_messages = []
    tokens = 0
    for message in messages:
        if not message.additional_kwargs.get("composed", ""):
            real_messages.append(message)
            if isinstance(message, AIMessage):
                usage = message.usage_metadata
                if usage:
                    tokens += usage.get("total_tokens", 0)
    print(f"tokens: {tokens}")
    return await handler(request.override(messages=real_messages))


@wrap_model_call
async def tool_result_budget(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """工具结果截断和 token 预算控制"""
    workplace = request.runtime.context.working_directory
    messages = list(request.messages)
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and msg.content:
            cleaned = clean_tool_output(msg.content)
            truncated = truncate_large_result(
                cleaned, msg.name or "", msg.tool_call_id, workplace=workplace,
            )
            messages[i] = msg.model_copy(update={"content": truncated})
    messages = enforce_per_turn_budget(messages, budget=200_000, workplace=workplace)
    return await handler(request.override(messages=messages))


# ─── Agent 构建 ──────────────────────────────────────────

class AsyncHITL(HumanInTheLoopMiddleware):
    """异步 HITL 中间件 — 审批在 chat loop 中处理"""

    async def awrap_model_call(self, request, handler):
        return await handler(request)


def build_agent(
    model_config: dict | None = None,
    checkpointer: AsyncSqliteSaver | None = None,
    mcp_tools: list | None = None,
    yolo: bool = False,
) -> object:
    """构建 agent 实例"""
    cfg = model_config or INNER_MODEL_CONFIG
    model = EnhancedChatOpenAI(**cfg)

    interrupt_on = (
        {}
        if yolo
        else {
            "bash": {"allowed_decisions": ["approve", "reject"]},
            "edit": {"allowed_decisions": ["approve", "reject"]},
            "write_file": {"allowed_decisions": ["approve", "reject"]},
        }
    )

    agent = create_agent(
        model,
        _get_all_tools() + (mcp_tools or []),
        middleware=[
            handle_tool_errors,
            tool_result_budget,
            load_skills,
            load_model,
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
                model=EnhancedChatOpenAI(**cfg),
                trigger=("tokens", 170_000),
                keep=("messages", 20),
            ),
            AsyncHITL(interrupt_on=interrupt_on),
        ],
        context_schema=SkillAgentContext,
        checkpointer=checkpointer,
    )
    return agent


async def create_checkpointer(db_path: Path) -> AsyncSqliteSaver:
    """创建异步 SQLite checkpointer"""
    conn = await aiosqlite.connect(str(db_path))
    return AsyncSqliteSaver(conn)


def _get_all_tools() -> list:
    """获取所有工具（延迟导入避免循环依赖）"""
    from chcode.utils.tools import ALL_TOOLS
    return ALL_TOOLS
