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
    import chcode.config as cfgmod

    config_dir = tmp_path / ".chat"
    config_dir.mkdir(exist_ok=True)
    # 隔离视觉配置文件
    monkeypatch.setattr(mod, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(mod, "VISION_JSON", config_dir / "vision_model.json")
    mod._vision_json.invalidate()
    # 同时隔离主模型配置文件（_add_to_model_fallback 等会写 model.json，
    # 不隔离会污染真实的 ~/.chat/model.json）
    monkeypatch.setattr(cfgmod, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cfgmod, "MODEL_JSON", config_dir / "model.json")
    monkeypatch.setattr(cfgmod, "SETTING_JSON", config_dir / "chagent.json")
    cfgmod._model_json.invalidate()
    return config_dir


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
        assert mod._vision_json._cache is not None
        assert mod._vision_json._cache[1] == data

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

        data = {
            "default": {"model": "save-test"},
            "fallback": {"fb1": {"model": "fb1"}},
        }
        mod.save_vision_json(data)

        assert mod.VISION_JSON.exists()
        loaded = json.loads(mod.VISION_JSON.read_text(encoding="utf-8"))
        assert loaded == data

    def test_invalidates_cache(self, mock_config_dir):
        """save_vision_json should clear the cache."""
        import chcode.vision_config as mod

        mod.VISION_JSON.write_text(json.dumps({"test": True}), encoding="utf-8")
        mod.load_vision_json()
        assert mod._vision_json._cache is not None

        mod.save_vision_json({"new": True})

        assert mod._vision_json._cache is None


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

        mod.save_vision_json(
            {"default": {"model": "test", "api_key": ""}, "fallback": {}}
        )

        result = mod.get_vision_default_model()
        assert result is None

    def test_returns_default_with_api_key(self, mock_config_dir):
        """Should return default model when api_key is present."""
        import chcode.vision_config as mod

        expected = {
            "model": "moonshotai/Kimi-K2.5",
            "api_key": "secret-key",
            "base_url": "https://x.com",
        }
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

        mod.save_vision_json(
            {
                "default": {"model": "default", "api_key": "k1"},
                "fallback": {
                    "fb1": {"model": "fb1", "api_key": "k2"},
                    "fb2": {"model": "fb2", "api_key": ""},
                    "fb3": {"model": "fb3", "api_key": "k3"},
                },
            }
        )

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

        result = mod._detect_modelscope_api_key(mod.MODELSCOPE_BASE_URL)

        assert result == "env-ms-key"

    def test_env_var_does_not_apply_to_intl_family(self, mock_config_dir, monkeypatch):
        """Env var is scoped to the cn family; intl family needs model.json .ai config."""
        import chcode.vision_config as mod

        monkeypatch.setenv("ModelScopeToken", "env-ms-key")

        result = mod._detect_modelscope_api_key(mod.MODELSCOPE_INTL_BASE_URL)

        assert result is None

    def test_falls_back_to_model_json_default(self, mock_config_dir, monkeypatch):
        """Should check model.json default if no env var."""
        import chcode.vision_config as mod

        monkeypatch.delenv("ModelScopeToken", raising=False)
        model_json = mock_config_dir / "model.json"
        model_json.write_text(
            json.dumps(
                {
                    "default": {
                        "model": "test",
                        "api_key": "json-key",
                        "base_url": "https://api-inference.modelscope.cn/v1",
                    }
                }
            ),
            encoding="utf-8",
        )

        result = mod._detect_modelscope_api_key(mod.MODELSCOPE_BASE_URL)

        assert result == "json-key"

    def test_matches_intl_family_from_model_json(self, mock_config_dir, monkeypatch):
        """Should detect key from model.json .ai base_url for the intl family."""
        import chcode.vision_config as mod

        monkeypatch.delenv("ModelScopeToken", raising=False)
        model_json = mock_config_dir / "model.json"
        model_json.write_text(
            json.dumps(
                {
                    "default": {
                        "model": "test",
                        "api_key": "intl-key",
                        "base_url": "https://api-inference.modelscope.ai/v1",
                    }
                }
            ),
            encoding="utf-8",
        )

        assert (
            mod._detect_modelscope_api_key(mod.MODELSCOPE_INTL_BASE_URL) == "intl-key"
        )
        assert mod._detect_modelscope_api_key(mod.MODELSCOPE_BASE_URL) is None

    def test_checks_model_json_fallback(self, mock_config_dir, monkeypatch):
        """Should check model.json fallback models."""
        import chcode.vision_config as mod

        monkeypatch.delenv("ModelScopeToken", raising=False)
        model_json = mock_config_dir / "model.json"
        model_json.write_text(
            json.dumps(
                {
                    "default": {
                        "model": "other",
                        "api_key": "other-key",
                        "base_url": "https://other.com",
                    },
                    "fallback": {
                        "ms-model": {
                            "model": "ms-model",
                            "api_key": "fb-key",
                            "base_url": "https://api-inference.modelscope.cn/v1",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        result = mod._detect_modelscope_api_key(mod.MODELSCOPE_BASE_URL)

        assert result == "fb-key"

    def test_returns_none_when_no_key(self, mock_config_dir, monkeypatch):
        """Should return None if no API key found anywhere."""
        import chcode.vision_config as mod

        monkeypatch.delenv("ModelScopeToken", raising=False)

        result = mod._detect_modelscope_api_key(mod.MODELSCOPE_BASE_URL)

        assert result is None


class TestBuildVisionConfig:
    """Tests for _build_vision_config()."""

    def test_uses_first_preset_as_default(self, mock_config_dir):
        """First VISION_MODEL_PRESETS entry should become default."""
        import chcode.vision_config as mod

        result = mod._build_vision_config("test-key")

        first = mod.VISION_MODEL_PRESETS[0]["model"]
        assert result["default"]["model"] == first
        assert result["default"]["api_key"] == "test-key"
        assert first not in result["fallback"]

    def test_remaining_presets_as_fallback(self, mock_config_dir):
        """Remaining presets should become fallback models."""
        import chcode.vision_config as mod

        result = mod._build_vision_config("test-key")

        first = mod.VISION_MODEL_PRESETS[0]["model"]
        assert len(result["fallback"]) == len(mod.VISION_MODEL_PRESETS) - 1
        for cfg in result["fallback"].values():
            assert cfg["api_key"] == "test-key"
            assert cfg["model"] != first


class TestAutoConfigureVision:
    """Tests for auto_configure_vision()."""

    def test_returns_none_when_no_api_key(self, mock_config_dir, monkeypatch):
        """Should return None if no API key is available."""
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
        """Should not write if existing key and base_url match."""
        import chcode.vision_config as mod

        monkeypatch.setenv("ModelScopeToken", "same-key")
        mod.save_vision_json(
            {
                "default": {
                    "model": "keep-this",
                    "api_key": "same-key",
                    "base_url": mod.MODELSCOPE_BASE_URL,
                },
                "fallback": {},
            }
        )
        old_mtime = mod.VISION_JSON.stat().st_mtime

        result = mod.auto_configure_vision()

        assert result["model"] == "keep-this"
        assert mod.VISION_JSON.stat().st_mtime == old_mtime

    def test_overwrites_if_key_differs(self, mock_config_dir, monkeypatch):
        """When existing key differs, should NOT overwrite default, only add to fallback."""
        import chcode.vision_config as mod

        monkeypatch.setenv("ModelScopeToken", "new-key")
        mod.save_vision_json(
            {
                "default": {
                    "model": "old-model",
                    "api_key": "old-key",
                    "base_url": "other",
                },
                "fallback": {},
            }
        )

        result = mod.auto_configure_vision()

        # 旧默认保留不覆盖
        assert result["api_key"] == "old-key"
        assert result["model"] == "old-model"
        # ModelScope 预设模型应加入 fallback
        data = mod.load_vision_json()
        assert mod.VISION_MODEL_PRESETS[0]["model"] in data["fallback"]


class TestConfigureVisionInteractive:
    """Tests for configure_vision_interactive()."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_config_and_cancel(self, mock_config_dir):
        """User cancels on unconfigured state."""
        import chcode.vision_config as mod

        with patch(
            "chcode.vision_config.select", new_callable=AsyncMock, return_value="返回"
        ):
            result = await mod.configure_vision_interactive()
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_configured_and_cancel(self, mock_config_dir):
        """User cancels on configured state."""
        import chcode.vision_config as mod

        mod.save_vision_json(
            {"default": {"model": "m", "api_key": "k"}, "fallback": {}}
        )

        with patch(
            "chcode.vision_config.select", new_callable=AsyncMock, return_value="返回"
        ):
            result = await mod.configure_vision_interactive()
            assert result is None

    @pytest.mark.asyncio
    async def test_displays_config(self, mock_config_dir):
        """User selects 查看当前配置."""
        import chcode.vision_config as mod

        mod.save_vision_json({"default": {"model": "display-test"}, "fallback": {}})

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value="查看当前配置",
            ),
            patch("chcode.vision_config.console"),
        ):
            result = await mod.configure_vision_interactive()
            assert result is None

    @pytest.mark.asyncio
    async def test_switch_model(self, mock_config_dir):
        """User switches to another model."""
        import chcode.vision_config as mod

        mod.save_vision_json(
            {
                "default": {"model": "model_a", "api_key": "k", "base_url": "url"},
                "fallback": {
                    "model_b": {"model": "model_b", "api_key": "k", "base_url": "url"},
                    "model_c": {"model": "model_c", "api_key": "k", "base_url": "url"},
                },
            }
        )

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=[
                    "切换模型",  # 菜单选择
                    "model_c (当前默认)"
                    if "fallback" in mod.get_vision_fallback_models().__str__()
                    else "model_c",  # 选择模型
                ],
            ),
            patch(
                "chcode.vision_config.confirm",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            result = await mod.configure_vision_interactive()

            assert result is not None
            assert result["model"] == "model_c"

            data = mod.load_vision_json()
            assert data["default"]["model"] == "model_c"
            assert "model_a" in data["fallback"]

    @pytest.mark.asyncio
    async def test_switch_model_declined(self, mock_config_dir):
        """User declines switch confirmation."""
        import chcode.vision_config as mod

        mod.save_vision_json(
            {
                "default": {"model": "model_a", "api_key": "k", "base_url": "url"},
                "fallback": {
                    "model_b": {"model": "model_b", "api_key": "k", "base_url": "url"}
                },
            }
        )

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=[
                    "切换模型",
                    "model_b",
                ],
            ),
            patch(
                "chcode.vision_config.confirm",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("chcode.vision_config.console"),
        ):
            result = await mod.configure_vision_interactive()

            assert result is None
            data = mod.load_vision_json()
            assert data["default"]["model"] == "model_a"

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

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=select_route,
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="wizard-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            result = await mod.configure_vision_interactive()

            assert result is not None
            assert result["model"] == mod.VISION_MODEL_PRESETS[0]["model"]
            assert result["api_key"] == "wizard-key"


class TestConfigureVisionCustom:
    """Tests for _configure_vision_custom() and _test_vision_connection()."""

    @pytest.mark.asyncio
    async def test_test_vision_connection_success(self, mock_config_dir):
        """_test_vision_connection returns True when model returns non-empty content."""
        import chcode.vision_config as mod

        with patch(
            "chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI"
        ) as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm_cls.return_value = mock_llm
            mock_result = MagicMock()
            mock_result.content = "红色"
            mock_llm.invoke.return_value = mock_result

            result = await mod._test_vision_connection(
                {"model": "vl", "base_url": "http://x", "api_key": "k"}, quiet=True
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_test_vision_connection_empty_content_still_passes(
        self, mock_config_dir
    ):
        """不报错即通过：即使响应 content 为空也算成功（不检查响应内容）。"""
        import chcode.vision_config as mod

        with patch(
            "chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI"
        ) as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm_cls.return_value = mock_llm
            mock_result = MagicMock()
            mock_result.content = ""
            mock_llm.invoke.return_value = mock_result

            result = await mod._test_vision_connection(
                {"model": "vl", "base_url": "http://x", "api_key": "k"}, quiet=True
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_test_vision_connection_exception_returns_error(
        self, mock_config_dir
    ):
        """_test_vision_connection returns error string on exception."""
        import chcode.vision_config as mod

        with patch(
            "chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI"
        ) as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm_cls.return_value = mock_llm
            mock_llm.invoke.side_effect = Exception("boom")

            result = await mod._test_vision_connection(
                {"model": "vl", "base_url": "http://x", "api_key": "k"},
                quiet=True,
                return_error=True,
            )
            assert result == "boom"

    @pytest.mark.asyncio
    async def test_test_vision_connection_null_choices_is_success(
        self, mock_config_dir
    ):
        """null value for 'choices' 视为连接通过（与文本侧 _test_connection 对齐，#5）。"""
        import chcode.vision_config as mod

        with patch(
            "chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI"
        ) as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm_cls.return_value = mock_llm
            mock_llm.invoke.side_effect = Exception("null value for 'choices'")

            result = await mod._test_vision_connection(
                {"model": "vl", "base_url": "http://x", "api_key": "k"}, quiet=True
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_test_vision_connection_null_other_is_failure(self, mock_config_dir):
        """null value 但不含 'choices'（如 'model'）仍判失败（#5）。"""
        import chcode.vision_config as mod

        with patch(
            "chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI"
        ) as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm_cls.return_value = mock_llm
            mock_llm.invoke.side_effect = Exception("null value for 'model'")

            result = await mod._test_vision_connection(
                {"model": "vl", "base_url": "http://x", "api_key": "k"}, quiet=True
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_custom_success_saves_as_default(self, mock_config_dir):
        """Filled form + passing test -> add_vision_model called, returns config."""
        import chcode.vision_config as mod
        from chcode.i18n import t

        with (
            patch(
                "chcode.vision_config.text",
                new_callable=AsyncMock,
                side_effect=[
                    "my-vl",
                    "http://api.x/v1",
                ],
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="key123",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "chcode.vision_config.add_vision_model", return_value="default"
            ) as mock_add,
            patch("chcode.config.text", new_callable=AsyncMock, return_value=""),
            patch("chcode.config._add_to_model_fallback") as mock_sync,
            patch("chcode.vision_config.console"),
            patch(
                "chcode.vision_config.confirm",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await mod._configure_vision_custom()

            assert result is not None
            assert result["model"] == "my-vl"
            assert result["api_key"] == "key123"
            mock_add.assert_called_once()
            # 同步到主模型 fallback
            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_config_carries_temperature_top_p(self, mock_config_dir):
        """自定义视觉条目带 temperature/top_p，与预设视觉模型结构一致（#7）。"""
        import chcode.vision_config as mod

        with (
            patch(
                "chcode.vision_config.text",
                new_callable=AsyncMock,
                side_effect=[
                    "my-vl",
                    "http://api.x/v1",
                ],
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="key123",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "chcode.vision_config.add_vision_model", return_value="default"
            ) as mock_add,
            patch("chcode.config.text", new_callable=AsyncMock, return_value=""),
            patch("chcode.config._add_to_model_fallback"),
            patch("chcode.vision_config.console"),
            patch(
                "chcode.vision_config.confirm",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await mod._configure_vision_custom()

            passed_config = mock_add.call_args.args[0]
            assert passed_config["temperature"] == 1.0
            assert passed_config["top_p"] == 0.95

    @pytest.mark.asyncio
    async def test_custom_user_configures_custom_hyperparams(self, mock_config_dir):
        """用户选"配超参"并填自定义值 → temperature/top_p 用用户输入（#7 配置入口）。"""
        import chcode.vision_config as mod

        # _ask_hyperparam 在函数内 from prompts import，故 patch prompts 模块
        with (
            patch(
                "chcode.vision_config.text",
                new_callable=AsyncMock,
                side_effect=[
                    "my-vl",
                    "http://api.x/v1",
                ],
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="key123",
            ),
            patch(
                "chcode.vision_config.confirm",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "chcode.vision_config.add_vision_model", return_value="default"
            ) as mock_add,
            patch("chcode.config.text", new_callable=AsyncMock, return_value=""),
            patch("chcode.config._add_to_model_fallback"),
            patch(
                "chcode.prompts._ask_hyperparam",
                new_callable=AsyncMock,
                side_effect=["0.3", "0.8"],
            ),
            patch("chcode.vision_config.console"),
        ):
            await mod._configure_vision_custom()

            passed_config = mock_add.call_args.args[0]
            assert passed_config["temperature"] == 0.3
            assert passed_config["top_p"] == 0.8

    @pytest.mark.asyncio
    async def test_custom_user_cancels_at_model_name(self, mock_config_dir):
        """Empty model name -> returns None without testing."""
        import chcode.vision_config as mod

        with (
            patch("chcode.vision_config.text", new_callable=AsyncMock, return_value=""),
            patch(
                "chcode.vision_config._test_vision_connection", new_callable=AsyncMock
            ) as mock_test,
        ):
            result = await mod._configure_vision_custom()
            assert result is None
            mock_test.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_test_fails_then_retry_then_success(self, mock_config_dir):
        """Test fails once, user retries, second attempt succeeds."""
        import chcode.vision_config as mod
        from chcode.i18n import t

        with (
            patch(
                "chcode.vision_config.text",
                new_callable=AsyncMock,
                side_effect=[
                    "my-vl",
                    "http://api.x/v1",
                ],
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="key123",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                side_effect=[
                    "connection refused",  # first test fails
                    True,  # retry succeeds
                ],
            ),
            patch(
                "chcode.config.select",
                new_callable=AsyncMock,
                return_value=t("connection.retry"),
            ),
            patch("chcode.vision_config.add_vision_model", return_value="default"),
            patch("chcode.config.text", new_callable=AsyncMock, return_value=""),
            patch("chcode.config._add_to_model_fallback"),
            patch("chcode.vision_config.console"),
            patch("chcode.config.console"),
            patch(
                "chcode.vision_config.confirm",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await mod._configure_vision_custom()
            assert result is not None
            assert result["model"] == "my-vl"

    @pytest.mark.asyncio
    async def test_custom_test_fails_then_reinput_then_success(self, mock_config_dir):
        """测试失败 -> 用户选"重新输入配置" -> 重填 -> 第二次测试成功（#6 循环 reinput 路径）。"""
        import chcode.vision_config as mod
        from chcode.i18n import t

        with (
            patch(
                "chcode.vision_config.text",
                new_callable=AsyncMock,
                side_effect=[
                    "bad-vl",
                    "http://bad/v1",  # 第一次收集
                    "good-vl",
                    "http://good/v1",  # 重新输入后的收集
                ],
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="key123",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                side_effect=[
                    "connection refused",  # 第一次测试失败
                    True,  # 重新输入后测试成功
                ],
            ),
            patch(
                "chcode.config.select",
                new_callable=AsyncMock,
                return_value=t("connection.reinput"),
            ),
            patch(
                "chcode.vision_config.add_vision_model", return_value="default"
            ) as mock_add,
            patch("chcode.config.text", new_callable=AsyncMock, return_value=""),
            patch("chcode.config._add_to_model_fallback"),
            patch("chcode.vision_config.console"),
            patch("chcode.config.console"),
            patch(
                "chcode.vision_config.confirm",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await mod._configure_vision_custom()
            assert result is not None
            # 最终保存的是重新输入后的模型
            assert result["model"] == "good-vl"
            assert result["base_url"] == "http://good/v1"
            mock_add.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_test_fails_user_aborts(self, mock_config_dir):
        """Test fails, user chooses abort -> returns None, nothing saved."""
        import chcode.vision_config as mod
        from chcode.i18n import t

        with (
            patch(
                "chcode.vision_config.text",
                new_callable=AsyncMock,
                side_effect=[
                    "my-vl",
                    "http://api.x/v1",
                ],
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="key123",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value="boom",
            ),
            patch(
                "chcode.config.select",
                new_callable=AsyncMock,
                return_value=t("connection.abort"),
            ),
            patch("chcode.vision_config.add_vision_model") as mock_add,
            patch("chcode.vision_config.console"),
            patch("chcode.config.console"),
            patch(
                "chcode.vision_config.confirm",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await mod._configure_vision_custom()
            assert result is None
            mock_add.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_syncs_to_main_model_fallback(self, mock_config_dir):
        """通过测试后，模型被加入 model.json 的 fallback，且不动 default；补问 context_length。"""
        import chcode.vision_config as mod
        import chcode.config as cfgmod

        # 预置一个主模型默认
        cfgmod.save_model_json(
            {
                "default": {
                    "model": "text-main",
                    "api_key": "tk",
                    "base_url": "http://t/v1",
                },
                "fallback": {},
            }
        )

        with (
            patch(
                "chcode.vision_config.text",
                new_callable=AsyncMock,
                side_effect=[
                    "vl-model",
                    "http://vl/v1",
                ],
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="vk",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.add_vision_model", return_value="fallback"),
            patch("chcode.config.text", new_callable=AsyncMock, return_value="200000"),
            patch("chcode.vision_config.console"),
            patch(
                "chcode.vision_config.confirm",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await mod._configure_vision_custom()

        # 验证 model.json：default 不变，fallback 含新模型且带 context_length
        data = cfgmod.load_model_json()
        assert data["default"]["model"] == "text-main"
        assert "vl-model" in data["fallback"]
        assert data["fallback"]["vl-model"]["metadata"]["context_length"] == 200000

    @pytest.mark.asyncio
    async def test_menu_new_model_routes_to_custom(self, mock_config_dir):
        """Configured menu: selecting 新建自定义模型 routes to _configure_vision_custom."""
        import chcode.vision_config as mod
        from chcode.i18n import t

        mod.save_vision_json(
            {"default": {"model": "m", "api_key": "k"}, "fallback": {}}
        )

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=t("vision.new_model"),
            ),
            patch(
                "chcode.vision_config._configure_vision_custom",
                new_callable=AsyncMock,
                return_value={"model": "custom"},
            ) as mock_custom,
        ):
            result = await mod.configure_vision_interactive()
            assert result is not None
            assert result["model"] == "custom"
            mock_custom.assert_called_once()

    @pytest.mark.asyncio
    async def test_menu_unconfigured_new_model_routes_to_custom(self, mock_config_dir):
        """Unconfigured menu: 新建自定义模型 also routes to _configure_vision_custom."""
        import chcode.vision_config as mod
        from chcode.i18n import t

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=t("vision.new_model"),
            ),
            patch(
                "chcode.vision_config._configure_vision_custom",
                new_callable=AsyncMock,
                return_value={"model": "custom"},
            ) as mock_custom,
        ):
            result = await mod.configure_vision_interactive()
            assert result is not None
            mock_custom.assert_called_once()


class TestConfigureVisionModelscope:
    """Tests for _configure_vision_modelscope() and its menu routing."""

    @pytest.mark.asyncio
    async def test_menu_modelscope_quick_routes_to_modelscope(self, mock_config_dir):
        """Configured menu: selecting 魔搭快捷配置 routes to _configure_vision_modelscope."""
        import chcode.vision_config as mod
        from chcode.i18n import t

        mod.save_vision_json(
            {"default": {"model": "m", "api_key": "k"}, "fallback": {}}
        )

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=t("vision.modelscope_quick"),
            ),
            patch(
                "chcode.vision_config._configure_vision_modelscope",
                new_callable=AsyncMock,
                return_value={"model": "ms"},
            ) as mock_ms,
        ):
            result = await mod.configure_vision_interactive()
            assert result is not None
            assert result["model"] == "ms"
            mock_ms.assert_called_once()

    @pytest.mark.asyncio
    async def test_menu_modelscope_quick_intl_routes_with_intl_flag(
        self, mock_config_dir
    ):
        """Configured menu: selecting 魔搭快捷配置（国际版）routes to _configure_vision_modelscope(intl=True)."""
        import chcode.vision_config as mod
        from chcode.i18n import t

        mod.save_vision_json(
            {"default": {"model": "m", "api_key": "k"}, "fallback": {}}
        )

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=t("vision.modelscope_quick_intl"),
            ),
            patch(
                "chcode.vision_config._configure_vision_modelscope",
                new_callable=AsyncMock,
                return_value={"model": "ms-intl"},
            ) as mock_ms,
        ):
            result = await mod.configure_vision_interactive()
            assert result is not None
            assert result["model"] == "ms-intl"
            mock_ms.assert_called_once_with(intl=True)

    @pytest.mark.asyncio
    async def test_menu_unconfigured_configure_intl_routes_to_wizard(
        self, mock_config_dir
    ):
        """Unconfigured menu: 配置视觉模型（国际版）routes to _configure_vision_wizard(intl=True)."""
        import chcode.vision_config as mod
        from chcode.i18n import t

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=t("vision.configure_intl"),
            ),
            patch(
                "chcode.vision_config._configure_vision_wizard",
                new_callable=AsyncMock,
                return_value={"model": "w-intl"},
            ) as mock_wizard,
        ):
            result = await mod.configure_vision_interactive()
            assert result is not None
            assert result["model"] == "w-intl"
            mock_wizard.assert_called_once_with(intl=True)

    @pytest.mark.asyncio
    async def test_modelscope_appends_presets_without_changing_default(
        self, mock_config_dir
    ):
        """给 API Key → 8 个预设补进 fallback，default 完全不变，model.json 不被写入。"""
        import chcode.vision_config as mod
        from chcode.i18n import t

        hand_filled_default = {
            "model": "my-handfilled-vl",
            "base_url": "http://custom/v1",
            "api_key": "orig-key",
        }
        mod.save_vision_json({"default": hand_filled_default, "fallback": {}})

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=t("vision.manual_key"),
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="ms-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            await mod._configure_vision_modelscope()

        data = mod.load_vision_json()
        # default 完全不变（手填值原样保留）
        assert data["default"] == hand_filled_default
        # 8 个预设全部进了 fallback（default 模型名不与任何预设重名）
        assert len(data["fallback"]) == len(mod.VISION_MODEL_PRESETS)
        for preset in mod.VISION_MODEL_PRESETS:
            assert preset["model"] in data["fallback"]
            assert data["fallback"][preset["model"]]["api_key"] == "ms-key"
        # model.json 不应被写入（不同步到文本侧）
        from chcode.config import MODEL_JSON

        assert not MODEL_JSON.exists()

    @pytest.mark.asyncio
    async def test_modelscope_intl_writes_ai_base_url(self, mock_config_dir):
        """国际版快捷配置：fallback 预设全部使用 .ai base_url，模型与参数同国内版。"""
        import chcode.vision_config as mod
        from chcode.i18n import t

        hand_filled_default = {
            "model": "my-handfilled-vl",
            "base_url": "http://custom/v1",
            "api_key": "orig-key",
        }
        mod.save_vision_json({"default": hand_filled_default, "fallback": {}})

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=t("vision.manual_key"),
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="ms-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            await mod._configure_vision_modelscope(intl=True)

        data = mod.load_vision_json()
        assert data["default"] == hand_filled_default
        assert len(data["fallback"]) == len(mod.VISION_MODEL_INTL_PRESETS)
        from chcode.utils.json_utils import region_key

        for preset in mod.VISION_MODEL_INTL_PRESETS:
            # 国际版预设打 region="intl" 标记，fallback key 带 (国际版) 后缀
            expected_key = region_key(preset)
            assert expected_key == f"{preset['model']} (国际版)"
            assert (
                data["fallback"][expected_key]["base_url"]
                == mod.MODELSCOPE_INTL_BASE_URL
            )
            assert data["fallback"][expected_key]["api_key"] == "ms-key"

    @pytest.mark.asyncio
    async def test_modelscope_skips_preset_same_as_default(self, mock_config_dir):
        """B2 回归：default 模型名与某预设相同时，跳过它——default 不被预设覆盖，该模型不进 fallback。"""
        import chcode.vision_config as mod
        from chcode.i18n import t

        # default 用一个预设模型名，但手填值（无 temperature/top_p）与预设不一致
        preset_model = mod.VISION_MODEL_PRESETS[0]["model"]
        hand_filled_default = {
            "model": preset_model,
            "base_url": "http://custom/v1",
            "api_key": "orig-key",
        }
        mod.save_vision_json({"default": hand_filled_default, "fallback": {}})

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=t("vision.manual_key"),
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="ms-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            await mod._configure_vision_modelscope()

        data = mod.load_vision_json()
        # default 未被预设覆盖（仍是手填值，没有 temperature/top_p）
        assert data["default"] == hand_filled_default
        assert "temperature" not in data["default"]
        # 同名预设未进 fallback（其余 7 个进了）
        assert preset_model not in data["fallback"]
        assert len(data["fallback"]) == len(mod.VISION_MODEL_PRESETS) - 1

    @pytest.mark.asyncio
    async def test_modelscope_cross_region_same_name_not_skipped(self, mock_config_dir):
        """跨 region 同名不误跳：default 是国际版 235B，走国内版预设时，国内版 235B
        仍应进 fallback（region_key 区分，不再被纯名比较误杀）。"""
        import chcode.vision_config as mod
        from chcode.i18n import t

        # default = 国际版 235B（与国内版预设同名，但 region 不同）
        same_name = mod.VISION_MODEL_PRESETS[0]["model"]
        intl_default = {
            "model": same_name,
            "base_url": mod.MODELSCOPE_INTL_BASE_URL,
            "api_key": "intl-key",
            "metadata": {"region": "intl"},
        }
        mod.save_vision_json({"default": intl_default, "fallback": {}})

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=t("vision.manual_key"),
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="ms-cn",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            await mod._configure_vision_modelscope()  # 走国内版预设

        data = mod.load_vision_json()
        # 国内版 235B 进了 fallback（未被纯名比较误跳）
        assert same_name in data["fallback"], (
            "国内版 235B 应进 fallback，未被国际版 default 误杀"
        )
        assert data["fallback"][same_name]["base_url"] == mod.MODELSCOPE_BASE_URL
        # default 仍是国际版，未被覆盖
        assert (data["default"].get("metadata") or {}).get("region") == "intl"
        # 国内版全部 7 个预设都进了 fallback
        assert len(data["fallback"]) == len(mod.VISION_MODEL_PRESETS)

    @pytest.mark.asyncio
    async def test_modelscope_preserves_existing_fallback(self, mock_config_dir):
        """已有 fallback 时跑快捷配置：旧的手填 fallback 保留，预设补充进来，default 不变。"""
        import chcode.vision_config as mod
        from chcode.i18n import t

        hand_filled_default = {
            "model": "my-handfilled-vl",
            "base_url": "http://custom/v1",
            "api_key": "orig-key",
        }
        existing_fallback = {
            "someone-else-vl": {
                "model": "someone-else-vl",
                "api_key": "old",
                "base_url": "http://x/v1",
            },
        }
        mod.save_vision_json(
            {"default": hand_filled_default, "fallback": existing_fallback}
        )

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=t("vision.manual_key"),
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="ms-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            await mod._configure_vision_modelscope()

        data = mod.load_vision_json()
        # default 不变
        assert data["default"] == hand_filled_default
        # 旧的 fallback 模型保留（键还在，值未被覆盖）
        assert "someone-else-vl" in data["fallback"]
        assert data["fallback"]["someone-else-vl"]["api_key"] == "old"
        # 8 个预设全部补充进来
        for preset in mod.VISION_MODEL_PRESETS:
            assert preset["model"] in data["fallback"]
            assert data["fallback"][preset["model"]]["api_key"] == "ms-key"
        # 总数 = 旧 1 + 预设 8
        assert len(data["fallback"]) == 1 + len(mod.VISION_MODEL_PRESETS)

    @pytest.mark.asyncio
    async def test_modelscope_test_fails_then_retry_then_success(self, mock_config_dir):
        """连接测试全失败 -> 选重试 -> 第二次测试通过 -> 落盘。"""
        import chcode.vision_config as mod
        from chcode.i18n import t

        mod.save_vision_json(
            {"default": {"model": "m", "api_key": "k"}, "fallback": {}}
        )

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=t("vision.manual_key"),
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="ms-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                side_effect=[
                    "connection refused",
                    "connection refused",
                    "connection refused",  # 首轮 3 个全失败
                    True,  # 重试后通过
                ],
            ),
            patch(
                "chcode.config.select",
                new_callable=AsyncMock,
                return_value=t("connection.retry"),
            ),
            patch("chcode.vision_config.console"),
            patch("chcode.config.console"),
        ):
            result = await mod._configure_vision_modelscope()

        # 测试通过后才落盘
        data = mod.load_vision_json()
        assert len(data["fallback"]) == len(mod.VISION_MODEL_PRESETS)
        # default 未被改动
        assert data["default"] == {"model": "m", "api_key": "k"}

    @pytest.mark.asyncio
    async def test_modelscope_test_fails_then_reinput_then_success(
        self, mock_config_dir
    ):
        """连接测试全失败 -> 选重新输入配置 -> 重填 Key -> 测试通过。"""
        import chcode.vision_config as mod
        from chcode.i18n import t

        mod.save_vision_json(
            {"default": {"model": "m", "api_key": "k"}, "fallback": {}}
        )

        # vision_config.select: 两次 _collect_key 各消耗一次（都选 manual_key）
        # password: 首次 bad-key, 重输后 good-key
        # _test_vision_connection: 首轮 3 个预设全失败, 重输后第一个就通过
        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=[
                    t("vision.manual_key"),  # 首次收集 Key 来源
                    t("vision.manual_key"),  # 重新输入后的 Key 来源
                ],
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                side_effect=[
                    "bad-key",
                    "good-key",
                ],
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                side_effect=[
                    "auth failed",
                    "auth failed",
                    "auth failed",  # 首轮 3 个代表预设全失败
                    True,  # 重输 good-key 后通过
                ],
            ),
            patch(
                "chcode.config.select",
                new_callable=AsyncMock,
                return_value=t("connection.reinput"),
            ),
            patch("chcode.vision_config.console"),
            patch("chcode.config.console"),
        ):
            await mod._configure_vision_modelscope()

        data = mod.load_vision_json()
        # 落盘用的是重输后的 good-key
        for preset in mod.VISION_MODEL_PRESETS:
            assert data["fallback"][preset["model"]]["api_key"] == "good-key"

    @pytest.mark.asyncio
    async def test_modelscope_test_fails_user_aborts(self, mock_config_dir):
        """连接测试全失败 -> 选放弃 -> 返回 None，不落盘。"""
        import chcode.vision_config as mod
        from chcode.i18n import t

        mod.save_vision_json(
            {"default": {"model": "m", "api_key": "k"}, "fallback": {}}
        )

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=t("vision.manual_key"),
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="ms-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value="boom",
            ),
            patch(
                "chcode.config.select",
                new_callable=AsyncMock,
                return_value=t("connection.abort"),
            ),
            patch("chcode.vision_config.console"),
            patch("chcode.config.console"),
        ):
            result = await mod._configure_vision_modelscope()

        assert result is None
        # 未落盘：fallback 仍为空
        data = mod.load_vision_json()
        assert data["fallback"] == {}

    @pytest.mark.asyncio
    async def test_modelscope_tests_first_passing_preset(self, mock_config_dir):
        """任一代表模型通过即成功：第一个失败、第二个通过，不再测第三个。"""
        import chcode.vision_config as mod
        from chcode.i18n import t

        mod.save_vision_json(
            {"default": {"model": "m", "api_key": "k"}, "fallback": {}}
        )

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=t("vision.manual_key"),
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="ms-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                side_effect=[
                    "err",  # 第一个预设失败
                    True,  # 第二个预设通过 -> break
                ],
            ) as mock_test,
            patch("chcode.vision_config.console"),
        ):
            await mod._configure_vision_modelscope()

        # 只测了 2 个（第三个因第二个通过而不再测）
        assert mock_test.call_count == 2


class TestConfigureVisionWizard:
    """Tests for _configure_vision_wizard()."""

    @pytest.mark.asyncio
    async def test_cancel_key_source(self, mock_config_dir):
        """User cancels at API key source selection."""
        import chcode.vision_config as mod

        with patch(
            "chcode.vision_config.select", new_callable=AsyncMock, return_value=None
        ):
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

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=select_route,
            ),
            patch(
                "chcode.vision_config.password", new_callable=AsyncMock, return_value=""
            ),
        ):
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

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=select_route,
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            result = await mod._configure_vision_wizard()

            assert result is not None
            assert result["model"] == chosen_model
            assert result["api_key"] == "wizard-key"
            assert (
                len(mod.load_vision_json()["fallback"])
                == len(mod.VISION_MODEL_PRESETS) - 1
            )

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

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=select_route,
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="manual-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            result = await mod._configure_vision_wizard()

            assert result is not None
            assert result["api_key"] == "manual-key"

    @pytest.mark.asyncio
    async def test_intl_wizard_uses_ai_base_url(self, mock_config_dir):
        """国际版向导：default/fallback 全部使用 .ai base_url，模型与参数同国内版。"""
        import chcode.vision_config as mod

        async def select_route(msg, choices, **kw):
            if "API Key" in msg:
                return "手动输入 API Key"
            if "默认视觉模型" in msg:
                return mod.VISION_MODEL_INTL_PRESETS[0]["model"]
            return choices[0]

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=select_route,
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="intl-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            result = await mod._configure_vision_wizard(intl=True)

        assert result is not None
        assert result["model"] == mod.VISION_MODEL_INTL_PRESETS[0]["model"]
        assert result["base_url"] == mod.MODELSCOPE_INTL_BASE_URL
        assert result["api_key"] == "intl-key"
        assert (
            len(mod.load_vision_json()["fallback"])
            == len(mod.VISION_MODEL_INTL_PRESETS) - 1
        )
        for cfg in mod.load_vision_json()["fallback"].values():
            assert cfg["base_url"] == mod.MODELSCOPE_INTL_BASE_URL

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

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=select_route,
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="key",
            ),
        ):
            result = await mod._configure_vision_wizard()
            assert result is None

    @pytest.mark.asyncio
    async def test_wizard_test_fails_then_retry_then_success(self, mock_config_dir):
        """连接测试失败 -> 选重试 -> 第二次通过 -> 落盘。wizard 只测用户选的 default。"""
        import chcode.vision_config as mod
        from chcode.i18n import t

        async def select_route(msg, choices, **kw):
            if "API Key" in msg:
                return "手动输入 API Key"
            if "默认视觉模型" in msg:
                return mod.VISION_MODEL_PRESETS[0]["model"]
            return choices[0]

        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=select_route,
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="ms-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                side_effect=[
                    "auth failed",  # 首轮失败
                    True,  # 重试通过
                ],
            ),
            patch(
                "chcode.config.select",
                new_callable=AsyncMock,
                return_value=t("connection.retry"),
            ),
            patch("chcode.vision_config.console"),
            patch("chcode.config.console"),
        ):
            result = await mod._configure_vision_wizard()

        assert result is not None
        assert result["api_key"] == "ms-key"
        # 测试通过后才落盘
        data = mod.load_vision_json()
        assert data["default"]["model"] == mod.VISION_MODEL_PRESETS[0]["model"]

    @pytest.mark.asyncio
    async def test_wizard_test_fails_then_reinput_then_success(self, mock_config_dir):
        """连接测试失败 -> 选重新输入 -> 重填 Key（不重选模型）-> 测试通过。"""
        import chcode.vision_config as mod
        from chcode.i18n import t

        chosen = mod.VISION_MODEL_PRESETS[0]["model"]
        # select 顺序: 首次 key 来源 → 选模型 → 重新输入时 key 来源（不重选模型）
        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=[
                    "手动输入 API Key",  # 首次 key 来源
                    chosen,  # 选模型
                    "手动输入 API Key",  # 重新输入 key 来源
                ],
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                side_effect=[
                    "bad-key",
                    "good-key",
                ],
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                side_effect=[
                    "auth failed",  # bad-key 失败
                    True,  # good-key 通过
                ],
            ),
            patch(
                "chcode.config.select",
                new_callable=AsyncMock,
                return_value=t("connection.reinput"),
            ),
            patch("chcode.vision_config.console"),
            patch("chcode.config.console"),
        ):
            result = await mod._configure_vision_wizard()

        assert result is not None
        assert result["api_key"] == "good-key", "落盘的应是重新输入后的 good-key"

    @pytest.mark.asyncio
    async def test_wizard_test_fails_user_aborts(self, mock_config_dir):
        """连接测试失败 -> 选放弃 -> 不落盘，返回 None。"""
        import chcode.vision_config as mod
        from chcode.i18n import t

        async def select_route(msg, choices, **kw):
            if "API Key" in msg:
                return "手动输入 API Key"
            if "默认视觉模型" in msg:
                return mod.VISION_MODEL_PRESETS[0]["model"]
            return choices[0]

        mod.save_vision_json({"default": {}, "fallback": {}})
        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=select_route,
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="bad-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value="auth failed",
            ),
            patch(
                "chcode.config.select",
                new_callable=AsyncMock,
                return_value=t("connection.abort"),
            ),
            patch("chcode.vision_config.console"),
            patch("chcode.config.console"),
        ):
            result = await mod._configure_vision_wizard()

        assert result is None
        # 未落盘
        data = mod.load_vision_json()
        assert data.get("default") in (None, {})


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
            mod._display_vision_config(
                {
                    "default": {"model": "Qwen/Qwen3-VL-235B-A22B-Instruct"},
                    "fallback": {},
                }
            )
            mock_console.print.assert_any_call(
                "[bold]默认视觉模型:[/bold] Qwen/Qwen3-VL-235B-A22B-Instruct"
            )

    def test_shows_fallback_table(self):
        """Should show table when fallback models exist."""
        import chcode.vision_config as mod

        with patch("chcode.vision_config.console") as mock_console:
            mod._display_vision_config(
                {
                    "default": {"model": "m1"},
                    "fallback": {"fb1": {"model": "fb1"}, "fb2": {"model": "fb2"}},
                }
            )
            mock_table = mock_console.print.call_args_list[-1].args[0]
            assert hasattr(mock_table, "title")
            assert "备用视觉模型" in str(mock_table.title)


class TestAddVisionModel:
    """Tests for add_vision_model()."""

    def test_no_default_becomes_default(self, mock_config_dir):
        """无 default 时新模型设为 default。"""
        import chcode.vision_config as mod

        config = {
            "model": "mm-1",
            "base_url": "https://x/v1",
            "api_key": "k1",
            "stream_usage": True,
        }
        role = mod.add_vision_model(config)

        assert role == "default"
        data = mod.load_vision_json()
        assert data["default"]["model"] == "mm-1"
        assert data["default"]["api_key"] == "k1"

    def test_existing_default_goes_fallback(self, mock_config_dir):
        """已有有效 default（api_key 非空）时加入 fallback，default 不变。"""
        import chcode.vision_config as mod

        mod.save_vision_json(
            {"default": {"model": "old", "api_key": "ko"}, "fallback": {}}
        )
        role = mod.add_vision_model({"model": "new", "api_key": "kn"})

        assert role == "fallback"
        data = mod.load_vision_json()
        assert data["default"]["model"] == "old"
        assert "new" in data["fallback"]

    def test_empty_api_key_default_treated_as_no_default(self, mock_config_dir):
        """default 的 api_key 为空时视同无 default，新模型设为 default。"""
        import chcode.vision_config as mod

        mod.save_vision_json(
            {"default": {"model": "old", "api_key": ""}, "fallback": {}}
        )
        role = mod.add_vision_model({"model": "new", "api_key": "kn"})

        assert role == "default"
        assert mod.load_vision_json()["default"]["model"] == "new"

    def test_same_name_same_key_idempotent(self, mock_config_dir):
        """同名同 key 已是 default 时幂等返回 None。"""
        import chcode.vision_config as mod

        mod.save_vision_json(
            {"default": {"model": "m", "api_key": "k"}, "fallback": {}}
        )
        assert mod.add_vision_model({"model": "m", "api_key": "k"}) is None

    def test_same_name_in_fallback_overwrites(self, mock_config_dir):
        """同名模型已在 fallback 时覆盖更新该条目。"""
        import chcode.vision_config as mod

        mod.save_vision_json(
            {
                "default": {"model": "other", "api_key": "ko"},
                "fallback": {"m": {"model": "m", "api_key": "old-key"}},
            }
        )
        role = mod.add_vision_model({"model": "m", "api_key": "new-key"})

        assert role == "fallback"
        assert mod.load_vision_json()["fallback"]["m"]["api_key"] == "new-key"

    def test_same_name_in_default_updates_key(self, mock_config_dir):
        """同名模型已是 default 且改了 api_key → 就地更新 default，不新增 fallback。"""
        import chcode.vision_config as mod

        mod.save_vision_json(
            {"default": {"model": "m", "api_key": "old"}, "fallback": {}}
        )
        role = mod.add_vision_model({"model": "m", "api_key": "new", "base_url": "x"})

        assert role == "default"
        data = mod.load_vision_json()
        assert data["default"]["api_key"] == "new"
        assert "m" not in data["fallback"]

    def test_same_name_in_default_updates_params(self, mock_config_dir):
        """同名模型已是 default 且改了超参（key 不变）→ 更新 default 超参。"""
        import chcode.vision_config as mod

        mod.save_vision_json(
            {
                "default": {"model": "m", "api_key": "k", "temperature": 0.5},
                "fallback": {},
            }
        )
        role = mod.add_vision_model({"model": "m", "api_key": "k", "temperature": 0.9})

        assert role == "default"
        assert mod.load_vision_json()["default"]["temperature"] == 0.9

    def test_same_name_in_fallback_idempotent(self, mock_config_dir):
        """同名模型在 fallback 且内容完全一致 → 幂等返回 None。"""
        import chcode.vision_config as mod

        mod.save_vision_json(
            {
                "default": {"model": "other", "api_key": "ko"},
                "fallback": {"m": {"model": "m", "api_key": "k"}},
            }
        )
        assert mod.add_vision_model({"model": "m", "api_key": "k"}) is None

    def test_missing_model_returns_none(self, mock_config_dir):
        """缺 model 返回 None 且不写入。"""
        import chcode.vision_config as mod

        assert mod.add_vision_model({"api_key": "k"}) is None
        assert mod.load_vision_json() == {}

    def test_missing_api_key_returns_none(self, mock_config_dir):
        """缺 api_key 返回 None。"""
        import chcode.vision_config as mod

        assert mod.add_vision_model({"model": "m"}) is None

    def test_filters_non_vision_fields(self, mock_config_dir):
        """只保留视觉白名单字段，丢弃 extra_body/stop_sequences/max_retries。"""
        import chcode.vision_config as mod

        config = {
            "model": "m",
            "base_url": "https://x/v1",
            "api_key": "k",
            "temperature": 0.7,
            "top_p": 0.9,
            "stream_usage": True,
            "extra_body": {"top_k": 20},
            "stop_sequences": ["x"],
            "max_retries": 4,
        }
        mod.add_vision_model(config)
        entry = mod.load_vision_json()["default"]

        assert set(entry.keys()) == {
            "model",
            "base_url",
            "api_key",
            "temperature",
            "top_p",
            "stream_usage",
        }

    def test_preserves_existing_fallback_when_setting_default(self, mock_config_dir):
        """设为 default 时保留已存在的 fallback。"""
        import chcode.vision_config as mod

        mod.save_vision_json(
            {
                "default": {},
                "fallback": {"fb": {"model": "fb", "api_key": "kf"}},
            }
        )
        mod.add_vision_model({"model": "new", "api_key": "kn"})
        data = mod.load_vision_json()

        assert data["default"]["model"] == "new"
        assert "fb" in data["fallback"]

    def test_does_not_mutate_input(self, mock_config_dir):
        """不修改入参 dict。"""
        import chcode.vision_config as mod

        config = {"model": "m", "api_key": "k", "extra_body": {"top_k": 20}}
        snapshot = {"model": "m", "api_key": "k", "extra_body": {"top_k": 20}}
        mod.add_vision_model(config)

        assert config == snapshot


class TestVisionRegionAwareFallback:
    """视觉 fallback region 对齐文本侧：同名国内/国际版模型能共存、key 区分。"""

    def test_add_vision_model_cross_region_same_name_not_overwrite_default(
        self, mock_config_dir
    ):
        """跨 region 同名不误覆盖 default：default 是国内版 235B，加入国际版 235B（同名
        不同 region）时，应进 fallback 而非覆盖 default——否则旧国内版 default 丢失。"""
        import chcode.vision_config as mod

        same_name = "Qwen/Qwen3-VL-235B-A22B-Instruct"
        mod.save_vision_json(
            {
                "default": {
                    "model": same_name,
                    "base_url": mod.MODELSCOPE_BASE_URL,
                    "api_key": "cn-key",
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "stream_usage": True,
                },
                "fallback": {},
            }
        )
        # 加入国际版同名模型
        role = mod.add_vision_model(
            {
                "model": same_name,
                "base_url": mod.MODELSCOPE_INTL_BASE_URL,
                "api_key": "intl-key",
                "metadata": {"region": "intl"},
                "temperature": 1.0,
                "top_p": 0.95,
                "stream_usage": True,
            }
        )
        data = mod.load_vision_json()
        # 国际版进 fallback，default 保持国内版不被覆盖
        assert role == "fallback", "跨 region 同名应进 fallback，而非覆盖 default"
        assert data["default"]["base_url"] == mod.MODELSCOPE_BASE_URL, (
            "default 应保持国内版"
        )
        assert f"{same_name} (国际版)" in data["fallback"], "国际版 235B 应进 fallback"

    def test_intl_presets_carry_region_marker(self):
        """国际版预设打 region=intl 标记，国内版预设不带。"""
        import chcode.vision_config as mod

        for p in mod.VISION_MODEL_INTL_PRESETS:
            assert (p.get("metadata") or {}).get("region") == "intl", p["model"]
        for p in mod.VISION_MODEL_PRESETS:
            assert "region" not in (p.get("metadata") or {}), p["model"]

    def test_whitelist_keeps_metadata(self, mock_config_dir):
        """_VISION_FIELDS 含 metadata，add_vision_model 不再剥离 region 标记。"""
        import chcode.vision_config as mod

        assert "metadata" in mod._VISION_FIELDS
        # 落盘的视觉条目保留 metadata.region
        mod.save_vision_json(
            {"default": {"model": "d", "api_key": "kd"}, "fallback": {}}
        )
        mod.add_vision_model(
            {
                "model": "Qwen/Qwen3-VL-8B-Instruct",
                "api_key": "k",
                "base_url": mod.MODELSCOPE_INTL_BASE_URL,
                "metadata": {"region": "intl"},
            }
        )
        data = mod.load_vision_json()
        # 国际版 key 带 (国际版) 后缀
        key = "Qwen/Qwen3-VL-8B-Instruct (国际版)"
        assert key in data["fallback"]
        assert data["fallback"][key]["metadata"]["region"] == "intl"

    def test_same_name_cn_and_intl_coexist_in_fallback(self, mock_config_dir):
        """同名国内/国际版视觉模型经 add_vision_model 后在 fallback 共存，互不覆盖。"""
        import chcode.vision_config as mod

        mod.save_vision_json(
            {"default": {"model": "d", "api_key": "kd"}, "fallback": {}}
        )
        # 国内版
        mod.add_vision_model(
            {"model": "VL", "api_key": "cn-key", "base_url": mod.MODELSCOPE_BASE_URL}
        )
        # 国际版同名
        mod.add_vision_model(
            {
                "model": "VL",
                "api_key": "intl-key",
                "base_url": mod.MODELSCOPE_INTL_BASE_URL,
                "metadata": {"region": "intl"},
            }
        )
        data = mod.load_vision_json()
        # 两个 key 都在，各自独立
        assert "VL" in data["fallback"]
        assert "VL (国际版)" in data["fallback"]
        assert data["fallback"]["VL"]["api_key"] == "cn-key"
        assert data["fallback"]["VL (国际版)"]["api_key"] == "intl-key"

    def test_auto_configure_follows_main_default_family(
        self, mock_config_dir, monkeypatch
    ):
        """auto_configure 只跟随主模型 model.json default 所属家族，不把两家族都放入。

        主模型 default 是国内版 → 只配国内版视觉；是国际版 → 只配国际版视觉。
        回归：此前 auto_configure 会无差别地把检测到的两家族全塞进 fallback。
        """
        import chcode.vision_config as mod

        # 两家族都检测到（model.json 里两个 key 都在）
        cn_preset = {
            "model": "VL",
            "base_url": mod.MODELSCOPE_BASE_URL,
            "temperature": 1.0,
            "top_p": 0.95,
            "stream_usage": True,
        }
        intl_preset = {
            "model": "VL",
            "base_url": mod.MODELSCOPE_INTL_BASE_URL,
            "temperature": 1.0,
            "top_p": 0.95,
            "stream_usage": True,
            "metadata": {"region": "intl"},
        }
        monkeypatch.setattr(
            mod,
            "_detect_api_keys",
            lambda: [("cn-key", [cn_preset]), ("intl-key", [intl_preset])],
        )

        # 主模型 default = 国内版 → 视觉只配国内版
        monkeypatch.setattr(
            "chcode.config.load_model_json",
            lambda: {"default": {"base_url": mod.MODELSCOPE_BASE_URL}},
        )
        mod.save_vision_json({"default": {}, "fallback": {}})
        mod.auto_configure_vision()
        data = mod.load_vision_json()
        assert "VL" in data["fallback"]
        assert "VL (国际版)" not in data["fallback"], "国内版 default 不应配国际版视觉"
        assert len(data["fallback"]) == 1

        # 主模型 default = 国际版 → 视觉只配国际版
        monkeypatch.setattr(
            "chcode.config.load_model_json",
            lambda: {"default": {"base_url": mod.MODELSCOPE_INTL_BASE_URL}},
        )
        mod.save_vision_json({"default": {}, "fallback": {}})
        mod.auto_configure_vision()
        data = mod.load_vision_json()
        assert "VL (国际版)" in data["fallback"]
        assert "VL" not in data["fallback"], "国际版 default 不应配国内版视觉"
        assert len(data["fallback"]) == 1

    def test_auto_configure_non_modelscope_default_takes_one_family(
        self, mock_config_dir, monkeypatch
    ):
        """主模型 default 非魔搭家族（如 OpenAI）时，只取检测到的第一个家族，不全放入。"""
        import chcode.vision_config as mod

        cn_preset = {
            "model": "VL",
            "base_url": mod.MODELSCOPE_BASE_URL,
            "temperature": 1.0,
            "top_p": 0.95,
            "stream_usage": True,
        }
        intl_preset = {
            "model": "VL",
            "base_url": mod.MODELSCOPE_INTL_BASE_URL,
            "temperature": 1.0,
            "top_p": 0.95,
            "stream_usage": True,
            "metadata": {"region": "intl"},
        }
        monkeypatch.setattr(
            mod,
            "_detect_api_keys",
            lambda: [("cn-key", [cn_preset]), ("intl-key", [intl_preset])],
        )
        monkeypatch.setattr(
            "chcode.config.load_model_json",
            lambda: {"default": {"base_url": "https://api.openai.com/v1"}},
        )

        mod.save_vision_json({"default": {}, "fallback": {}})
        mod.auto_configure_vision()
        data = mod.load_vision_json()
        # 只有一个家族（首个），不是两个都进
        assert len(data["fallback"]) == 1

    @pytest.mark.asyncio
    async def test_switch_to_intl_model_across_region(self, mock_config_dir):
        """切换到同名国际版模型：choice_names 平行列表正确取回带后缀的 key，
        旧国内版 default 转入 fallback（纯名 key），国际版条目从 fallback 移除。"""
        import chcode.vision_config as mod

        same_name = "VL"
        mod.save_vision_json(
            {
                "default": {
                    "model": same_name,
                    "api_key": "cn-key",
                    "base_url": mod.MODELSCOPE_BASE_URL,
                },
                "fallback": {
                    # 同名国际版（key 带后缀）+ 另一个国内版模型
                    f"{same_name} (国际版)": {
                        "model": same_name,
                        "api_key": "intl-key",
                        "base_url": mod.MODELSCOPE_INTL_BASE_URL,
                        "metadata": {"region": "intl"},
                    },
                    "other": {
                        "model": "other",
                        "api_key": "k",
                        "base_url": "http://x/v1",
                    },
                },
            }
        )

        # select 返回国际版那项的显示文本（key 本身，无"当前默认"标记——因为
        # 当前 default 是国内版同名，不在 fallback 列表里）
        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                return_value=f"{same_name} (国际版)",
            ),
            patch(
                "chcode.vision_config.confirm",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            result = await mod._switch_vision_model()

        data = mod.load_vision_json()
        # 新 default 是国际版
        assert data["default"]["model"] == same_name
        assert (data["default"].get("metadata") or {}).get("region") == "intl"
        # 旧国内版 default 转入 fallback（纯名 key，无后缀）
        assert same_name in data["fallback"]
        assert data["fallback"][same_name]["base_url"] == mod.MODELSCOPE_BASE_URL
        # 国际版条目已从 fallback 移除（成为 default）
        assert f"{same_name} (国际版)" not in data["fallback"]


class TestVisionWizardPreservesOldDefault:
    """重新配置向导时，旧 default 必须转入 fallback（对齐文本侧 _merge_and_save_config），
    不能被同名新 default 直接覆盖丢失——这是 cn/intl 同名互覆盖的根因。"""

    @pytest.mark.asyncio
    async def test_cn_default_preserved_when_reconfiguring_intl(self, mock_config_dir):
        """先配国内版 default，再用国际版向导配同名 default → 旧国内版保留进 fallback。"""
        import chcode.vision_config as mod

        chosen = mod.VISION_MODEL_PRESETS[2]["model"]  # Qwen3-VL-8B（国内/国际同名）

        async def select_route(msg, choices, **kw):
            if "API Key" in msg:
                return "手动输入 API Key"
            if "默认视觉模型" in msg:
                return chosen
            return choices[0]

        mod.save_vision_json({"default": {}, "fallback": {}})
        # 第一次：国内版向导
        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=select_route,
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="cn-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            await mod._configure_vision_wizard(intl=False)
        # 第二次：国际版向导，选同名 default
        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=select_route,
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="intl-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            await mod._configure_vision_wizard(intl=True)

        data = mod.load_vision_json()
        # 新 default 是国际版（同名）
        assert data["default"]["model"] == chosen
        assert (data["default"].get("metadata") or {}).get("region") == "intl"
        # 旧国内版 default 保留进 fallback（未被覆盖丢失）
        assert chosen in data["fallback"], "旧国内版 default 应转入 fallback"
        assert data["fallback"][chosen]["api_key"] == "cn-key"
        # 国际版其余预设也在 fallback（带后缀），数量 = 6 国内预设 + 6 国际预设 + 1 旧 default
        assert (
            len(data["fallback"])
            == len(mod.VISION_MODEL_PRESETS)
            - 1
            + len(mod.VISION_MODEL_INTL_PRESETS)
            - 1
            + 1
        )

    @pytest.mark.asyncio
    async def test_intl_default_preserved_when_reconfiguring_cn(self, mock_config_dir):
        """反向：先配国际版 default，再配国内版同名 default → 旧国际版保留进 fallback。"""
        import chcode.vision_config as mod

        chosen = mod.VISION_MODEL_PRESETS[2]["model"]

        async def select_route(msg, choices, **kw):
            if "API Key" in msg:
                return "手动输入 API Key"
            if "默认视觉模型" in msg:
                return chosen
            return choices[0]

        mod.save_vision_json({"default": {}, "fallback": {}})
        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=select_route,
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="intl-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            await mod._configure_vision_wizard(intl=True)
        with (
            patch(
                "chcode.vision_config.select",
                new_callable=AsyncMock,
                side_effect=select_route,
            ),
            patch(
                "chcode.vision_config.password",
                new_callable=AsyncMock,
                return_value="cn-key",
            ),
            patch(
                "chcode.vision_config._test_vision_connection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("chcode.vision_config.console"),
        ):
            await mod._configure_vision_wizard(intl=False)

        data = mod.load_vision_json()
        assert data["default"]["model"] == chosen
        assert (data["default"].get("metadata") or {}).get(
            "region"
        ) != "intl"  # 新 default 是国内版
        # 旧国际版 default 保留进 fallback（带国际版后缀）
        intl_key = f"{chosen} (国际版)"
        assert intl_key in data["fallback"], "旧国际版 default 应转入 fallback"
        assert data["fallback"][intl_key]["api_key"] == "intl-key"
