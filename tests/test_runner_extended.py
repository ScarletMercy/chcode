"""Extended tests for chcode/agents/runner.py"""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chcode.agents.definitions import AgentDefinition
from chcode.agents.runner import _resolve_tools, _subagent_system_prompt
from chcode.agent_setup import handle_tool_errors, tool_result_budget


def _tool(name):
    t = MagicMock()
    t.name = name
    return t


class TestHandleToolErrors:
    async def test_passthrough_runner(self):
        mock_handler = AsyncMock(return_value="ok")
        mock_request = MagicMock()
        mock_request.tool_call = {"id": "123"}
        result = await handle_tool_errors.awrap_tool_call(mock_request, mock_handler)
        assert result == "ok"

    async def test_exception_returns_tool_message(self):
        mock_handler = AsyncMock(side_effect=ValueError("err"))
        mock_request = MagicMock()
        mock_request.tool_call = {"id": "123"}
        from langchain_core.messages import ToolMessage
        result = await handle_tool_errors.awrap_tool_call(mock_request, mock_handler)
        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "123"
        assert "err" in result.content


class TestSubagentSystemPrompt:
    async def test_returns_system_prompt(self):
        from langchain_core.messages import SystemMessage
        mock_request = MagicMock()
        mock_request.runtime.context.extra = {"system_prompt": "You are a planner."}
        mock_request.override = MagicMock(return_value=mock_request)
        mock_handler = AsyncMock(return_value="response")
        result = await _subagent_system_prompt.awrap_model_call(mock_request, mock_handler)
        mock_request.override.assert_called_once()
        call_kwargs = mock_request.override.call_args
        assert call_kwargs[1]["system_message"].content == "You are a planner."
        mock_handler.assert_called_once()

    async def test_returns_empty_when_no_key(self):
        from langchain_core.messages import SystemMessage
        mock_request = MagicMock()
        mock_request.runtime.context.extra = {}
        mock_request.override = MagicMock(return_value=mock_request)
        mock_handler = AsyncMock(return_value="response")
        result = await _subagent_system_prompt.awrap_model_call(mock_request, mock_handler)
        mock_request.override.assert_called_once()
        call_kwargs = mock_request.override.call_args
        assert call_kwargs[1]["system_message"].content == ""
        mock_handler.assert_called_once()


class TestToolResultBudget:
    async def test_no_tool_messages_passthrough(self):
        from langchain_core.messages import HumanMessage
        mock_request = MagicMock()
        mock_request.messages = [HumanMessage(content="hi")]
        mock_request.runtime.context.working_directory = Path("/w")
        mock_handler = AsyncMock(return_value="response")
        with patch("chcode.agent_setup.enforce_per_turn_budget", return_value=[HumanMessage(content="hi")]):
            result = await tool_result_budget.awrap_model_call(mock_request, mock_handler)
        mock_handler.assert_called_once()

    async def test_processes_tool_messages(self):
        from langchain_core.messages import ToolMessage
        mock_msg = MagicMock(spec=ToolMessage)
        mock_msg.content = "output"
        mock_msg.name = "bash"
        mock_msg.tool_call_id = "t1"
        mock_msg.additional_kwargs = {}
        mock_msg.model_copy = MagicMock(return_value=mock_msg)
        mock_request = MagicMock()
        mock_request.messages = [mock_msg]
        mock_request.runtime.context.working_directory = Path("/w")
        mock_handler = AsyncMock(return_value="response")
        with patch("chcode.agent_setup.clean_tool_output", return_value="cleaned"), \
             patch("chcode.agent_setup.truncate_large_result", return_value="truncated"), \
             patch("chcode.agent_setup.enforce_per_turn_budget", return_value=[mock_msg]):
            await tool_result_budget.awrap_model_call(mock_request, mock_handler)
        assert mock_handler.called


class TestRunSubagent:
    async def test_success(self):
        from chcode.agents.runner import run_subagent
        mock_def = AgentDefinition(
            agent_type="Explore", when_to_use="explore",
            system_prompt="prompt", read_only=True, tools=None,
            disallowed_tools=[], model=None,
        )
        mock_msg = MagicMock()
        mock_msg.type = "ai"
        mock_msg.content = "result text"
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_msg]})
        with patch("chcode.agents.runner.create_agent", return_value=mock_agent), \
             patch("chcode.agents.runner._resolve_tools", return_value=[]), \
             patch("chcode.agents.runner.EnhancedChatOpenAI", MagicMock()):
            result, is_error = await run_subagent("task", mock_def, {"model": "gpt-4"}, Path("/w"), MagicMock())
        assert result == "result text"
        assert is_error is False

    async def test_timeout(self):
        import asyncio
        from chcode.agents.runner import run_subagent
        mock_def = AgentDefinition(
            agent_type="Explore", when_to_use="explore",
            system_prompt="prompt", read_only=True, tools=None,
            disallowed_tools=[], model=None,
        )
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError())
        with patch("chcode.agents.runner.create_agent", return_value=mock_agent), \
             patch("chcode.agents.runner._resolve_tools", return_value=[]), \
             patch("chcode.agents.runner.EnhancedChatOpenAI", MagicMock()):
            result, is_error = await run_subagent("task", mock_def, {"model": "gpt-4"}, Path("/w"), MagicMock(), timeout_seconds=300)
        assert "timed out" in result
        assert is_error is True

    async def test_model_switch_error_runner(self):
        from chcode.agents.runner import run_subagent
        from chcode.agent_setup import ModelSwitchError
        mock_def = AgentDefinition(
            agent_type="Explore", when_to_use="explore",
            system_prompt="prompt", read_only=True, tools=None,
            disallowed_tools=[], model=None,
        )
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(side_effect=ModelSwitchError("switch"))
        with patch("chcode.agents.runner.create_agent", return_value=mock_agent), \
             patch("chcode.agents.runner._resolve_tools", return_value=[]), \
             patch("chcode.agents.runner.EnhancedChatOpenAI", MagicMock()):
            result, is_error = await run_subagent("task", mock_def, {"model": "gpt-4"}, Path("/w"), MagicMock(), timeout_seconds=300)
        assert "备用模型" in result
        assert is_error is True

    async def test_generic_error(self):
        from chcode.agents.runner import run_subagent
        mock_def = AgentDefinition(
            agent_type="Explore", when_to_use="explore",
            system_prompt="prompt", read_only=True, tools=None,
            disallowed_tools=[], model=None,
        )
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(side_effect=RuntimeError("crash"))
        with patch("chcode.agents.runner.create_agent", return_value=mock_agent), \
             patch("chcode.agents.runner._resolve_tools", return_value=[]), \
             patch("chcode.agents.runner.EnhancedChatOpenAI", MagicMock()):
            result, is_error = await run_subagent("task", mock_def, {"model": "gpt-4"}, Path("/w"), MagicMock(), timeout_seconds=300)
        assert "error" in result.lower()
        assert is_error is True

    async def test_list_content_output(self):
        from chcode.agents.runner import run_subagent
        mock_def = AgentDefinition(
            agent_type="Explore", when_to_use="explore",
            system_prompt="prompt", read_only=True, tools=None,
            disallowed_tools=[], model=None,
        )
        mock_msg = MagicMock()
        mock_msg.type = "ai"
        mock_msg.content = [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_msg]})
        with patch("chcode.agents.runner.create_agent", return_value=mock_agent), \
             patch("chcode.agents.runner._resolve_tools", return_value=[]), \
             patch("chcode.agents.runner.EnhancedChatOpenAI", MagicMock()):
            result, is_error = await run_subagent("task", mock_def, {"model": "gpt-4"}, Path("/w"), MagicMock(), timeout_seconds=300)
        assert "part1" in result and "part2" in result
        assert is_error is False
