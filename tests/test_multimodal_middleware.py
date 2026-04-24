"""Tests for filter_vision_tool middleware in chcode/agent_setup.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.messages import ToolMessage


def _make_tool_request(tool_name="vision", tool_id="tc-123"):
    """Create a mock ToolCallRequest."""
    request = MagicMock()
    request.tool_call = {"name": tool_name, "id": tool_id}
    request.runtime = MagicMock()
    request.runtime.context.model_config = {}
    return request


class TestFilterVisionTool:

    async def test_blocks_vision_for_multimodal_model(self):
        """When model is multimodal, vision tool call should be blocked."""
        from chcode.agent_setup import filter_vision_tool

        request = _make_tool_request("vision")
        request.runtime.context.model_config = {"model": "moonshotai/Kimi-K2.5"}

        handler = AsyncMock(return_value="should not reach")

        result = await filter_vision_tool.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert "原生视觉" in result.content
        assert result.tool_call_id == "tc-123"
        handler.assert_not_called()

    async def test_blocks_vision_for_qwen_multimodal(self):
        """Qwen3.5-397B should also block vision tool."""
        from chcode.agent_setup import filter_vision_tool

        request = _make_tool_request("vision")
        request.runtime.context.model_config = {"model": "Qwen/Qwen3.5-397B-A17B"}

        handler = AsyncMock(return_value="should not reach")

        result = await filter_vision_tool.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        handler.assert_not_called()

    async def test_allows_vision_for_non_multimodal_model(self):
        """When model is NOT multimodal, vision tool should pass through."""
        from chcode.agent_setup import filter_vision_tool

        request = _make_tool_request("vision")
        request.runtime.context.model_config = {"model": "glm-5"}

        handler = AsyncMock(return_value="tool executed")

        result = await filter_vision_tool.awrap_tool_call(request, handler)

        assert result == "tool executed"
        handler.assert_called_once()

    async def test_allows_other_tools_for_multimodal_model(self):
        """Non-vision tools should pass through even for multimodal models."""
        from chcode.agent_setup import filter_vision_tool

        request = _make_tool_request("bash")
        request.runtime.context.model_config = {"model": "moonshotai/Kimi-K2.5"}

        handler = AsyncMock(return_value="bash executed")

        result = await filter_vision_tool.awrap_tool_call(request, handler)

        assert result == "bash executed"
        handler.assert_called_once()

    async def test_allows_vision_when_model_name_empty(self):
        """Empty model name should not block vision."""
        from chcode.agent_setup import filter_vision_tool

        request = _make_tool_request("vision")
        request.runtime.context.model_config = {"model": ""}

        handler = AsyncMock(return_value="tool executed")

        result = await filter_vision_tool.awrap_tool_call(request, handler)

        assert result == "tool executed"
        handler.assert_called_once()


class TestLoadSkillsSystemPrompt:
    async def test_multimodal_prompt_omits_vision(self):
        """System prompt for multimodal model should not mention vision tool."""
        from chcode.agent_setup import load_skills

        mock_loader = MagicMock()
        mock_loader.build_system_prompt = MagicMock(return_value="prompt")

        mock_request = MagicMock()
        mock_request.runtime.context.skill_loader = mock_loader
        mock_request.runtime.context.working_directory = "/w"
        mock_request.runtime.context.model_config = {"model": "moonshotai/Kimi-K2.5"}

        handler = AsyncMock(return_value="model response")

        with patch("chcode.agent_setup.sys.platform", "linux"):
            await load_skills.awrap_model_call(mock_request, handler)

        # Verify build_system_prompt was called
        mock_loader.build_system_prompt.assert_called_once()
        base_prompt = mock_loader.build_system_prompt.call_args[0][0]

        # Vision tool should NOT be listed in the Tools section
        tools_section = base_prompt.split("Guidelines")[0]
        assert "vision" not in tools_section.lower()
        # But should mention native vision capability in Guidelines
        assert "native vision" in base_prompt

    async def test_non_multimodal_prompt_includes_vision(self):
        """System prompt for non-multimodal model should mention vision tool."""
        from chcode.agent_setup import load_skills

        mock_loader = MagicMock()
        mock_loader.build_system_prompt = MagicMock(return_value="prompt")

        mock_request = MagicMock()
        mock_request.runtime.context.skill_loader = mock_loader
        mock_request.runtime.context.working_directory = "/w"
        mock_request.runtime.context.model_config = {"model": "glm-5"}

        handler = AsyncMock(return_value="model response")

        with patch("chcode.agent_setup.sys.platform", "linux"):
            await load_skills.awrap_model_call(mock_request, handler)

        base_prompt = mock_loader.build_system_prompt.call_args[0][0]

        assert "vision" in base_prompt.lower()
