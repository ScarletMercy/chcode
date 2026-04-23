"""Tests for chcode/vision_config.py"""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


@pytest.fixture
def mock_config_dir(tmp_path: Path, monkeypatch):
    """Setup mock config directory for vision config tests."""
    import chcode.vision_config as mod

    config_dir = tmp_path / ".chat"
    config_dir.mkdir()
    monkeypatch.setattr(mod, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(mod, "VISION_JSON", config_dir / "vision_model.json")
    mod._vision_json_cache = None
    return config_dir


class TestEnsureConfigDir:
    """Tests for ensure_config_dir()."""

    def test_creates_dir_if_not_exists(self, tmp_path: Path, monkeypatch):
        """Directory should be created if it doesn't exist."""
        import chcode.vision_config as mod

        config_dir = tmp_path / ".chat"
        monkeypatch.setattr(mod, "CONFIG_DIR", config_dir)
        assert not config_dir.exists()

        result = mod.ensure_config_dir()

        assert config_dir.exists()
        assert result == config_dir

    def test_returns_config_dir_if_exists(self, tmp_path: Path, monkeypatch):
        """Should return CONFIG_DIR if it already exists."""
        import chcode.vision_config as mod

        config_dir = tmp_path / ".chat"
        config_dir.mkdir()
        monkeypatch.setattr(mod, "CONFIG_DIR", config_dir)

        result = mod.ensure_config_dir()

        assert result == config_dir


class TestLoadVisionJson:
    """Tests for load_vision_json()."""

    def test_returns_empty_dict_if_file_missing(self, mock_config_dir):
        """Missing vision_model.json should return empty dict."""
        import chcode.vision_config as mod

        result = mod.load_vision_json()
        assert result == {}

    def test_loads_valid_json(self, mock_config_dir):
        """Should parse and return valid JSON content."""
        import chcode.vision_config as mod

        data = {"default": {"model": "test-model", "api_key": "key123"}, "fallback": {}}
        mod.VISION_JSON.write_text(json.dumps(data), encoding="utf-8")

        result = mod.load_vision_json()

        assert result == data
        assert result["default"]["model"] == "test-model"

    def test_uses_cache_on_same_mtime(self, mock_config_dir):
        """Second call should return cached data without re-reading."""
        import chcode.vision_config as mod

        data = {"default": {"model": "cached-model"}, "fallback": {}}
        mod.VISION_JSON.write_text(json.dumps(data), encoding="utf-8")

        result1 = mod.load_vision_json()
        result2 = mod.load_vision_json()

        assert result1 == result2
        assert mod._vision_json_cache is not None
        assert mod._vision_json_cache[1] == data

    def test_returns_empty_dict_on_invalid_json(self, mock_config_dir):
        """Invalid JSON should return empty dict."""
        import chcode.vision_config as mod

        mod.VISION_JSON.write_text("not valid json {", encoding="utf-8")

        result = mod.load_vision_json()

        assert result == {}


class TestSaveVisionJson:
    """Tests for save_vision_json()."""

    def test_saves_json_to_file(self, mock_config_dir):
        """Should write dict as formatted JSON to vision_model.json."""
        import chcode.vision_config as mod

        data = {"default": {"model": "save-test"}, "fallback": {"fb1": {"model": "fb1"}}}
        mod.save_vision_json(data)

        assert mod.VISION_JSON.exists()
        loaded = json.loads(mod.VISION_JSON.read_text(encoding="utf-8"))
        assert loaded == data

    def test_invalidates_cache(self, mock_config_dir):
        """save_vision_json should clear the cache."""
        import chcode.vision_config as mod

        mod.VISION_JSON.write_text(json.dumps({"test": True}), encoding="utf-8")
        mod.load_vision_json()
        assert mod._vision_json_cache is not None

        mod.save_vision_json({"new": True})

        assert mod._vision_json_cache is None


class TestGetVisionDefaultModel:
    """Tests for get_vision_default_model()."""

    def test_returns_none_when_no_file(self, mock_config_dir):
        """Should return None if vision_model.json doesn't exist."""
        import chcode.vision_config as mod

        result = mod.get_vision_default_model()
        assert result is None

    def test_returns_none_when_default_missing_api_key(self, mock_config_dir):
        """Should return None if default exists but api_key is empty."""
        import chcode.vision_config as mod

        mod.save_vision_json({"default": {"model": "test", "api_key": ""}, "fallback": {}})

        result = mod.get_vision_default_model()
        assert result is None

    def test_returns_default_with_api_key(self, mock_config_dir):
        """Should return default model when api_key is present."""
        import chcode.vision_config as mod

        expected = {"model": "moonshotai/Kimi-K2.5", "api_key": "secret-key", "base_url": "https://x.com"}
        mod.save_vision_json({"default": expected, "fallback": {}})

        result = mod.get_vision_default_model()

        assert result == expected
        assert result["api_key"] == "secret-key"


class TestGetVisionFallbackModels:
    """Tests for get_vision_fallback_models()."""

    def test_returns_empty_list_when_no_file(self, mock_config_dir):
        """Should return empty list if vision_model.json doesn't exist."""
        import chcode.vision_config as mod

        result = mod.get_vision_fallback_models()
        assert result == []

    def test_returns_models_with_api_key(self, mock_config_dir):
        """Should return fallback models that have api_key."""
        import chcode.vision_config as mod

        mod.save_vision_json({
            "default": {"model": "default", "api_key": "k1"},
            "fallback": {
                "fb1": {"model": "fb1", "api_key": "k2"},
                "fb2": {"model": "fb2", "api_key": ""},
                "fb3": {"model": "fb3", "api_key": "k3"},
            }
        })

        result = mod.get_vision_fallback_models()

        assert len(result) == 2
        models = [m["model"] for m in result]
        assert "fb1" in models
        assert "fb3" in models
        assert "fb2" not in models


class TestDetectModelscopeApiKey:
    """Tests for _detect_modelscope_api_key()."""

    def test_prefers_env_var(self, mock_config_dir, monkeypatch):
        """Should return ModelScopeToken env var if present."""
        import chcode.vision_config as mod

        monkeypatch.setenv("ModelScopeToken", "env-ms-key")

        result = mod._detect_modelscope_api_key()

        assert result == "env-ms-key"

    def test_falls_back_to_model_json_default(self, mock_config_dir, monkeypatch):
        """Should check model.json default if no env var."""
        import chcode.vision_config as mod

        monkeypatch.delenv("ModelScopeToken", raising=False)
        model_json = mock_config_dir / "model.json"
        model_json.write_text(json.dumps({
            "default": {
                "model": "test",
                "api_key": "json-key",
                "base_url": "https://api-inference.modelscope.cn/v1"
            }
        }), encoding="utf-8")

        result = mod._detect_modelscope_api_key()

        assert result == "json-key"

    def test_checks_model_json_fallback(self, mock_config_dir, monkeypatch):
        """Should check model.json fallback models."""
        import chcode.vision_config as mod

        monkeypatch.delenv("ModelScopeToken", raising=False)
        model_json = mock_config_dir / "model.json"
        model_json.write_text(json.dumps({
            "default": {
                "model": "other",
                "api_key": "other-key",
                "base_url": "https://other.com"
            },
            "fallback": {
                "ms-model": {
                    "model": "ms-model",
                    "api_key": "fb-key",
                    "base_url": "https://api-inference.modelscope.cn/v1"
                }
            }
        }), encoding="utf-8")

        result = mod._detect_modelscope_api_key()

        assert result == "fb-key"

    def test_returns_none_when_no_key(self, mock_config_dir, monkeypatch):
        """Should return None if no API key found anywhere."""
        import chcode.vision_config as mod

        monkeypatch.delenv("ModelScopeToken", raising=False)

        result = mod._detect_modelscope_api_key()

        assert result is None


class TestBuildVisionConfig:
    """Tests for _build_vision_config()."""

    def test_uses_first_preset_as_default(self, mock_config_dir):
        """First VISION_MODEL_PRESETS entry should become default."""
        import chcode.vision_config as mod

        result = mod._build_vision_config("test-key")

        assert result["default"]["model"] == "moonshotai/Kimi-K2.5"
        assert result["default"]["api_key"] == "test-key"
        assert "moonshotai/Kimi-K2.5" not in result["fallback"]

    def test_remaining_presets_as_fallback(self, mock_config_dir):
        """Remaining presets should become fallback models."""
        import chcode.vision_config as mod

        result = mod._build_vision_config("test-key")

        assert len(result["fallback"]) == len(mod.VISION_MODEL_PRESETS) - 1
        for cfg in result["fallback"].values():
            assert cfg["api_key"] == "test-key"
            assert cfg["model"] != "moonshotai/Kimi-K2.5"


class TestAutoConfigureVision:
    """Tests for auto_configure_vision()."""

    def test_returns_none_when_no_api_key(self, mock_config_dir, monkeypatch):
        """Should return None if no ModelScope key is available."""
        import chcode.vision_config as mod

        monkeypatch.delenv("ModelScopeToken", raising=False)

        result = mod.auto_configure_vision()

        assert result is None

    def test_creates_config_with_env_key(self, mock_config_dir, monkeypatch):
        """Should create config from env var."""
        import chcode.vision_config as mod

        monkeypatch.setenv("ModelScopeToken", "ms-env-key")

        result = mod.auto_configure_vision()

        assert result is not None
        assert result["api_key"] == "ms-env-key"
        assert mod.VISION_JSON.exists()

    def test_does_not_overwrite_same_key(self, mock_config_dir, monkeypatch):
        """Should not write if existing key matches env key."""
        import chcode.vision_config as mod

        monkeypatch.setenv("ModelScopeToken", "same-key")
        mod.save_vision_json({
            "default": {"model": "keep-this", "api_key": "same-key"},
            "fallback": {}
        })
        old_mtime = mod.VISION_JSON.stat().st_mtime

        result = mod.auto_configure_vision()

        assert result["model"] == "keep-this"
        assert mod.VISION_JSON.stat().st_mtime == old_mtime

    def test_overwrites_if_key_differs(self, mock_config_dir, monkeypatch):
        """Should overwrite if existing key differs from env key."""
        import chcode.vision_config as mod

        monkeypatch.setenv("ModelScopeToken", "new-key")
        mod.save_vision_json({
            "default": {"model": "old-model", "api_key": "old-key"},
            "fallback": {}
        })

        result = mod.auto_configure_vision()

        assert result["api_key"] == "new-key"
        assert result["model"] == "moonshotai/Kimi-K2.5"


class TestConfigureVisionInteractive:
    """Tests for configure_vision_interactive()."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_config_and_cancel(self, mock_config_dir):
        """User cancels on unconfigured state."""
        import chcode.vision_config as mod

        with patch("chcode.vision_config.select", new_callable=AsyncMock, return_value="返回"):
            result = await mod.configure_vision_interactive()
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_configured_and_cancel(self, mock_config_dir):
        """User cancels on configured state."""
        import chcode.vision_config as mod

        mod.save_vision_json({"default": {"model": "m", "api_key": "k"}, "fallback": {}})

        with patch("chcode.vision_config.select", new_callable=AsyncMock, return_value="返回"):
            result = await mod.configure_vision_interactive()
            assert result is None

    @pytest.mark.asyncio
    async def test_displays_config(self, mock_config_dir):
        """User selects 查看当前配置."""
        import chcode.vision_config as mod

        mod.save_vision_json({"default": {"model": "display-test"}, "fallback": {}})

        with patch("chcode.vision_config.select", new_callable=AsyncMock, return_value="查看当前配置"), \
             patch("chcode.vision_config.console"):
            result = await mod.configure_vision_interactive()
            assert result is None

    @pytest.mark.asyncio
    async def test_clears_config(self, mock_config_dir):
        """User clears config."""
        import chcode.vision_config as mod

        mod.save_vision_json({"default": {"model": "m", "api_key": "k"}, "fallback": {}})

        with patch("chcode.vision_config.select", new_callable=AsyncMock, return_value="清除配置"), \
             patch("chcode.vision_config.confirm", new_callable=AsyncMock, return_value=True), \
             patch("chcode.vision_config.console"):
            result = await mod.configure_vision_interactive()

            assert result is None
            assert not mod.VISION_JSON.exists()
            assert mod._vision_json_cache is None

    @pytest.mark.asyncio
    async def test_clears_config_declined(self, mock_config_dir):
        """User declines clear confirmation."""
        import chcode.vision_config as mod

        mod.save_vision_json({"default": {"model": "m", "api_key": "k"}, "fallback": {}})

        with patch("chcode.vision_config.select", new_callable=AsyncMock, return_value="清除配置"), \
             patch("chcode.vision_config.confirm", new_callable=AsyncMock, return_value=False):
            result = await mod.configure_vision_interactive()

            assert result is None
            assert mod.VISION_JSON.exists()

    @pytest.mark.asyncio
    async def test_returns_wizard_result(self, mock_config_dir):
        """configure_vision_interactive returns wizard result on configure."""
        import chcode.vision_config as mod

        async def select_route(msg, choices, **kw):
            if "未配置" in msg or "视觉模型配置:" in msg:
                return "配置视觉模型"
            if "API Key" in msg:
                return "手动输入 API Key"
            if "默认视觉模型" in msg:
                return mod.VISION_MODEL_PRESETS[0]["model"]
            return choices[0]

        with patch("chcode.vision_config.select", new_callable=AsyncMock, side_effect=select_route), \
             patch("chcode.vision_config.password", new_callable=AsyncMock, return_value="wizard-key"), \
             patch("chcode.vision_config.console"):
            result = await mod.configure_vision_interactive()

            assert result is not None
            assert result["model"] == "moonshotai/Kimi-K2.5"
            assert result["api_key"] == "wizard-key"


class TestConfigureVisionWizard:
    """Tests for _configure_vision_wizard()."""

    @pytest.mark.asyncio
    async def test_cancel_key_source(self, mock_config_dir):
        """User cancels at API key source selection."""
        import chcode.vision_config as mod

        with patch("chcode.vision_config.select", new_callable=AsyncMock, return_value=None):
            result = await mod._configure_vision_wizard()
            assert result is None

    @pytest.mark.asyncio
    async def test_empty_manual_key_returns_none(self, mock_config_dir):
        """User enters empty API key."""
        import chcode.vision_config as mod

        async def select_route(msg, choices, **kw):
            if "API Key" in msg:
                return "手动输入 API Key"
            if "默认视觉模型" in msg:
                return mod.VISION_MODEL_PRESETS[0]["model"]
            return choices[0]

        with patch("chcode.vision_config.select", new_callable=AsyncMock, side_effect=select_route), \
             patch("chcode.vision_config.password", new_callable=AsyncMock, return_value=""):
            result = await mod._configure_vision_wizard()
            assert result is None

    @pytest.mark.asyncio
    async def test_successful_wizard_with_env_key(self, mock_config_dir, monkeypatch):
        """User completes wizard with env var key."""
        import chcode.vision_config as mod

        monkeypatch.setenv("ModelScopeToken", "wizard-key")
        chosen_model = "Qwen/Qwen3-VL-235B-A22B-Instruct"

        async def select_route(msg, choices, **kw):
            if "API Key" in msg:
                return f"使用环境变量 ModelScopeToken (wizard...key)"
            if "默认视觉模型" in msg:
                return chosen_model
            return choices[0]

        with patch("chcode.vision_config.select", new_callable=AsyncMock, side_effect=select_route), \
             patch("chcode.vision_config.console"):
            result = await mod._configure_vision_wizard()

            assert result is not None
            assert result["model"] == chosen_model
            assert result["api_key"] == "wizard-key"
            assert len(mod.load_vision_json()["fallback"]) == len(mod.VISION_MODEL_PRESETS) - 1

    @pytest.mark.asyncio
    async def test_successful_wizard_with_manual_key(self, mock_config_dir):
        """User completes wizard with manual key input."""
        import chcode.vision_config as mod

        async def select_route(msg, choices, **kw):
            if "API Key" in msg:
                return "手动输入 API Key"
            if "默认视觉模型" in msg:
                return mod.VISION_MODEL_PRESETS[0]["model"]
            return choices[0]

        with patch("chcode.vision_config.select", new_callable=AsyncMock, side_effect=select_route), \
             patch("chcode.vision_config.password", new_callable=AsyncMock, return_value="manual-key"), \
             patch("chcode.vision_config.console"):
            result = await mod._configure_vision_wizard()

            assert result is not None
            assert result["api_key"] == "manual-key"

    @pytest.mark.asyncio
    async def test_wizard_cancel_model_selection(self, mock_config_dir):
        """User cancels during model selection step."""
        import chcode.vision_config as mod

        async def select_route(msg, choices, **kw):
            if "API Key" in msg:
                return "手动输入 API Key"
            if "默认视觉模型" in msg:
                return None
            return choices[0]

        with patch("chcode.vision_config.select", new_callable=AsyncMock, side_effect=select_route), \
             patch("chcode.vision_config.password", new_callable=AsyncMock, return_value="key"):
            result = await mod._configure_vision_wizard()
            assert result is None


class TestDisplayVisionConfig:
    """Tests for _display_vision_config()."""

    def test_empty_config(self):
        """Should print warning for empty config."""
        import chcode.vision_config as mod

        with patch("chcode.vision_config.console") as mock_console:
            mod._display_vision_config({})
            mock_console.print.assert_any_call("[yellow]未配置视觉模型[/yellow]")

    def test_shows_default_model(self):
        """Should display default model name."""
        import chcode.vision_config as mod

        with patch("chcode.vision_config.console") as mock_console:
            mod._display_vision_config({
                "default": {"model": "Qwen/Qwen3-VL-235B-A22B-Instruct"},
                "fallback": {}
            })
            mock_console.print.assert_any_call("[bold]默认视觉模型:[/bold] Qwen/Qwen3-VL-235B-A22B-Instruct")

    def test_shows_fallback_table(self):
        """Should show table when fallback models exist."""
        import chcode.vision_config as mod

        with patch("chcode.vision_config.console") as mock_console:
            mod._display_vision_config({
                "default": {"model": "m1"},
                "fallback": {"fb1": {"model": "fb1"}, "fb2": {"model": "fb2"}}
            })
            mock_table = mock_console.print.call_args_list[-1].args[0]
            assert hasattr(mock_table, "title")
            assert "备用视觉模型" in str(mock_table.title)