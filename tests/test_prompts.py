"""Tests for chcode/prompts.py — select, confirm, checkbox, text, password, select_or_custom, model_config_form."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from chcode.prompts import (
    select,
    confirm,
    checkbox,
    text,
    password,
    select_or_custom,
    model_config_form,
    _SkipSentinel,
    _SKIP,
    SKIP_LABEL,
    BASE_URL_PRESETS,
    TEMPERATURE_PRESETS,
    API_KEY_ENV_VARS,
)


# ── select ────────────────────────────────────────────────────


class TestSelect:
    async def test_returns_selection(self):
        with patch("chcode.prompts.asyncio.to_thread", new_callable=AsyncMock, return_value="B"):
            result = await select("Pick?", ["A", "B"])
        assert result == "B"

    async def test_returns_none_on_cancel(self):
        with patch("chcode.prompts.asyncio.to_thread", new_callable=AsyncMock, return_value=None):
            result = await select("Pick?", ["A"])
        assert result is None


# ── confirm ────────────────────────────────────────────────────


class TestConfirm:
    async def test_returns_true(self):
        with patch("chcode.prompts.asyncio.to_thread", new_callable=AsyncMock, return_value=True):
            result = await confirm("Sure?")
        assert result is True

    async def test_returns_false(self):
        with patch("chcode.prompts.asyncio.to_thread", new_callable=AsyncMock, return_value=False):
            result = await confirm("Sure?", default=False)
        assert result is False


# ── checkbox ───────────────────────────────────────────────────


class TestCheckbox:
    async def test_returns_selections(self):
        with patch("chcode.prompts.asyncio.to_thread", new_callable=AsyncMock, return_value=["A", "C"]):
            result = await checkbox("Pick:", ["A", "B", "C"])
        assert result == ["A", "C"]

    async def test_empty_returns_empty_list(self):
        with patch("chcode.prompts.asyncio.to_thread", new_callable=AsyncMock, return_value=[]):
            result = await checkbox("Pick:", ["A"])
        assert result == []


# ── text ───────────────────────────────────────────────────────


class TestText:
    async def test_returns_input(self):
        with patch("chcode.prompts.asyncio.to_thread", new_callable=AsyncMock, return_value="hello"):
            result = await text("Name:")
        assert result == "hello"

    async def test_returns_empty(self):
        with patch("chcode.prompts.asyncio.to_thread", new_callable=AsyncMock, return_value=""):
            result = await text("Name:")
        assert result == ""


# ── password ────────────────────────────────────────────────────


class TestPassword:
    async def test_returns_password(self):
        with patch("chcode.prompts.asyncio.to_thread", new_callable=AsyncMock, return_value="secret"):
            result = await password("Enter key:")
        assert result == "secret"


# ── select_or_custom ───────────────────────────────────────────


class TestSelectOrCustom:
    async def test_selects_preset(self):
        async def _mock_select(msg, choices, default=None, **kwargs):
            return "https://api.openai.com/v1"
        async def _mock_text(msg, default="", **kwargs):
            return "custom"
        with patch("chcode.prompts.select", _mock_select), \
             patch("chcode.prompts.text", _mock_text):
            result = await select_or_custom("URL:", BASE_URL_PRESETS[:2])
        assert result == "https://api.openai.com/v1"

    async def test_selects_custom(self):
        async def _mock_select(msg, choices, default=None, **kwargs):
            return "自定义输入..."
        async def _mock_text(msg, default="", **kwargs):
            return "https://custom.api/v1"
        with patch("chcode.prompts.select", _mock_select), \
             patch("chcode.prompts.text", _mock_text):
            result = await select_or_custom("URL:", BASE_URL_PRESETS[:2])
        assert result == "https://custom.api/v1"

    async def test_cancel_returns_none(self):
        async def _mock_select(msg, choices, default=None, **kwargs):
            return None
        with patch("chcode.prompts.select", _mock_select):
            result = await select_or_custom("URL:", BASE_URL_PRESETS[:2])
        assert result is None


# ── _SkipSentinel ───────────────────────────────────────────────


class TestSkipSentinel:
    def test_singleton(self):
        assert _SKIP is _SkipSentinel()

    def test_repr(self):
        assert repr(_SKIP) == "SKIP"


# ── model_config_form helpers ─────────────────────────────────


def _mock_select_async(return_value):
    async def _s(msg, choices, default=None, **kwargs):
        return return_value
    return _s


def _mock_confirm_async(return_value):
    async def _c(msg, default=True, **kwargs):
        return return_value
    return _c


def _mock_text_async(return_value):
    async def _t(msg, default="", **kwargs):
        return return_value
    return _t


def _mock_password_async(return_value):
    async def _p(msg, **kwargs):
        return return_value
    return _p


class TestModelConfigForm:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Remove all API key env vars for model_config_form tests."""
        for var, _ in API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)

    async def test_new_config_no_hyperparams(self):
        async def _select_route(msg, choices, default=None, **kwargs):
            if "Base URL" in msg:
                return "https://api.openai.com/v1"
            return "手动输入 API Key..."
        with patch("chcode.prompts.text", _mock_text_async("gpt-4")), \
             patch("chcode.prompts.password", _mock_password_async("sk-123")), \
             patch("chcode.prompts.confirm", _mock_confirm_async(False)), \
             patch("chcode.prompts.select", _select_route):
            result = await model_config_form(None)
        assert result is not None
        assert result["model"] == "gpt-4"
        assert result["base_url"] == "https://api.openai.com/v1"
        assert result["api_key"] == "sk-123"
        assert result["stream_usage"] is True

    async def test_empty_model_name_returns_none(self):
        with patch("chcode.prompts.text", _mock_text_async("")):
            result = await model_config_form(None)
        assert result is None

    async def test_empty_api_key_returns_none(self):
        async def _select_route(msg, choices, default=None, **kwargs):
            if "Base URL" in msg:
                return "https://api.openai.com/v1"
            return "手动输入 API Key..."
        with patch("chcode.prompts.text", _mock_text_async("gpt-4")), \
             patch("chcode.prompts.password", _mock_password_async("")), \
             patch("chcode.prompts.confirm", _mock_confirm_async(False)), \
             patch("chcode.prompts.select", _select_route):
            result = await model_config_form(None)
        assert result is None

    async def test_with_hyperparams(self):
        async def _select_route(msg, choices, default=None, **kwargs):
            if "Base URL" in msg:
                return "https://api.openai.com/v1"
            if "API Key" in msg:
                return "手动输入 API Key..."
            if "Temperature" in msg:
                return "0.7"
            # Top P, Top K, Max Completion Tokens all need numeric values
            return "4096"
        with patch("chcode.prompts.text", _mock_text_async("gpt-4")), \
             patch("chcode.prompts.password", _mock_password_async("sk-123")), \
             patch("chcode.prompts.confirm", _mock_confirm_async(True)), \
             patch("chcode.prompts.select", _select_route):
            result = await model_config_form(None)
        assert result is not None
        assert result["temperature"] == 0.7
        assert result["extra_body"]["top_k"] == 4096

    async def test_skip_hyperparam(self):
        async def _select_skip(msg, choices, default=None, **kwargs):
            if "Base URL" in msg:
                return "https://api.openai.com/v1"
            if "API Key" in msg:
                return "手动输入 API Key..."
            return SKIP_LABEL
        with patch("chcode.prompts.text", _mock_text_async("gpt-4")), \
             patch("chcode.prompts.password", _mock_password_async("sk-123")), \
             patch("chcode.prompts.confirm", _mock_confirm_async(True)), \
             patch("chcode.prompts.select", _select_skip):
            result = await model_config_form(None)
        assert result is not None
        assert "temperature" not in result or result.get("temperature") is None

    async def test_new_config_modelscope_url(self):
        """New config selects ModelScope shortcut URL."""
        async def _select_route(msg, choices, default=None, **kwargs):
            if "Base URL" in msg:
                return "魔搭 (ModelScope)"
            return "手动输入 API Key..."
        with patch("chcode.prompts.text", _mock_text_async("gpt-4")), \
             patch("chcode.prompts.password", _mock_password_async("ms-key")), \
             patch("chcode.prompts.confirm", _mock_confirm_async(False)), \
             patch("chcode.prompts.select", _select_route):
            result = await model_config_form(None)
        assert result is not None
        assert result["base_url"] == "https://api-inference.modelscope.cn/v1"
        assert result["api_key"] == "ms-key"

    async def test_new_config_custom_url_empty(self):
        """New config custom URL empty input returns None."""
        async def _text_route(msg, default="", **kwargs):
            if "Base URL" in msg or "base" in msg.lower():
                return ""
            return "gpt-4"
        async def _select_route(msg, choices, default=None, **kwargs):
            if "Base URL" in msg:
                return "自定义输入..."
            return "手动输入 API Key..."
        with patch("chcode.prompts.text", _text_route), \
             patch("chcode.prompts.select", _select_route), \
             patch("chcode.prompts.password", _mock_password_async("sk-123")):
            result = await model_config_form(None)
        assert result is None

    async def test_edit_config_custom_url_empty(self):
        """Edit mode custom URL empty input returns None."""
        existing = {
            "model": "gpt-4",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-123",
        }
        async def _select_route(msg, choices, default=None, **kwargs):
            if "Base URL" in msg:
                return "自定义输入..."
            return "保持当前 Key (****)"
        with patch("chcode.prompts.text", _mock_text_async("")), \
             patch("chcode.prompts.select", _select_route):
            result = await model_config_form(existing)
        assert result is None

    async def test_edit_mode_keeps_base_url(self):
        async def _select_route(msg, choices, default=None, **kwargs):
            if "API Key" in msg:
                return "保持当前 Key (****)"
            if "Base URL" in msg:
                return "保持当前值 (https://old.com)"
            return "0.7"
        with patch("chcode.prompts.text", _mock_text_async("gpt-4")), \
             patch("chcode.prompts.select", _select_route), \
             patch("chcode.prompts.password", _mock_password_async("sk-old")), \
             patch("chcode.prompts.confirm", _mock_confirm_async(False)):
            result = await model_config_form({"model": "gpt-4", "base_url": "https://old.com", "api_key": "sk-old"})
        assert result["base_url"] == "https://old.com"

    async def test_env_key_detected(self):
        async def _select_env(msg, choices, default=None, **kwargs):
            # First call: API key source select; rest: hyperparams
            if "API Key" in msg:
                return "OPENAI_API_KEY (OpenAI)"
            return "0.7"
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-123"}), \
             patch("chcode.prompts.text", _mock_text_async("gpt-4")), \
             patch("chcode.prompts.select_or_custom", _mock_select_async("https://api.openai.com/v1")), \
             patch("chcode.prompts.password", _mock_password_async("sk-123")), \
             patch("chcode.prompts.confirm", _mock_confirm_async(False)), \
             patch("chcode.prompts.select", _select_env):
            result = await model_config_form(None)
        assert result["api_key"] == "env-key-123"

    async def test_model_form_cancel_returns_none(self):
        with patch("chcode.prompts.text", _mock_text_async(None)):
            result = await model_config_form(None)
        assert result is None


# ── Inner _ask function coverage (lines 40, 52, 61) ────────────
# These lines are inside _ask() which is passed to asyncio.to_thread.
# To cover them without a real terminal, we mock questionary at the
# module level so _ask() runs successfully inside to_thread.


class TestInnerAskFunctions:
    """Cover lines 40, 52, 61: inner _ask functions for confirm, checkbox, text.

    By mocking questionary.X().ask() to return immediately, the _ask inner
    function executes fully when called via asyncio.to_thread (real, not mocked).
    """

    async def test_confirm_inner_ask(self):
        """Line 40: questionary.confirm().ask() executes."""
        mock_confirm = MagicMock()
        mock_confirm.return_value.ask.return_value = True
        with patch("chcode.prompts.questionary.confirm", mock_confirm):
            result = await confirm("Sure?", default=True)
        assert result is True

    async def test_checkbox_inner_ask(self):
        """Line 52: questionary.checkbox().ask() executes."""
        mock_checkbox = MagicMock()
        mock_checkbox.return_value.ask.return_value = ["A", "B"]
        with patch("chcode.prompts.questionary.checkbox", mock_checkbox):
            result = await checkbox("Pick:", ["A", "B", "C"])
        assert result == ["A", "B"]

    async def test_text_inner_ask(self):
        """Line 61: questionary.text().ask() executes."""
        mock_text = MagicMock()
        mock_text.return_value.ask.return_value = "hello world"
        with patch("chcode.prompts.questionary.text", mock_text):
            result = await text("Name:", default="default_val")
        assert result == "hello world"

    async def test_select_inner_ask(self):
        """Line 27 (parallel): questionary.select().ask() executes."""
        mock_select = MagicMock()
        mock_select.return_value.ask.return_value = "B"
        with patch("chcode.prompts.questionary.select", mock_select):
            result = await select("Pick?", ["A", "B", "C"], default="A")
        assert result == "B"

    async def test_confirm_inner_ask_false(self):
        """Line 40: questionary.confirm with default=False returns False."""
        mock_confirm = MagicMock()
        mock_confirm.return_value.ask.return_value = False
        with patch("chcode.prompts.questionary.confirm", mock_confirm):
            result = await confirm("Sure?", default=False)
        assert result is False

    async def test_checkbox_inner_ask_empty(self):
        """Line 52: questionary.checkbox returns empty list."""
        mock_checkbox = MagicMock()
        mock_checkbox.return_value.ask.return_value = []
        with patch("chcode.prompts.questionary.checkbox", mock_checkbox):
            result = await checkbox("Pick:", ["A"])
        assert result == []

    async def test_text_inner_ask_empty(self):
        """Line 61: questionary.text returns empty string."""
        mock_text = MagicMock()
        mock_text.return_value.ask.return_value = ""
        with patch("chcode.prompts.questionary.text", mock_text):
            result = await text("Name:")
        assert result == ""
