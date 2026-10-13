"""Tests for multimodal (vision) guidance in chcode/agent_setup.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _preset_vision_models(tmp_path, monkeypatch):
    """隔离并预置 vision_model.json,让 is_multimodal_model 识别测试用的多模态模型。

    is_multimodal_model 现从 vision_model.json 读取模型清单,需预置测试模型
    (Kimi-K2.5/Qwen3.5-397B)并隔离,避免读写真实 home(CI 干净环境会 otherwise 失败)。
    """
    import chcode.vision_config as vc

    monkeypatch.setattr(vc, "VISION_JSON", tmp_path / "vision_model.json")
    vc._vision_json.invalidate()
    vc.save_vision_json(
        {
            "default": {"model": "moonshotai/Kimi-K2.5", "api_key": "k"},
            "fallback": {
                "Qwen/Qwen3.5-397B-A17B": {
                    "model": "Qwen/Qwen3.5-397B-A17B",
                    "api_key": "k",
                },
            },
        }
    )


class TestLoadSkillsSystemPrompt:
    async def test_multimodal_prompt_guides_vision_usage(self):
        """Multimodal prompt: analyze embedded media directly, vision only for
        media not yet in the conversation (proactive viewing)."""
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

        # Native vision guidance: embedded media analyzed directly
        assert "native vision" in base_prompt
        assert "analyze it directly" in base_prompt
        # vision tool still mentioned — for proactively viewing non-embedded media
        assert "vision tool" in base_prompt
        assert "not yet in the conversation" in base_prompt

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
