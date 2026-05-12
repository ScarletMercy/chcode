import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from chcode.config import (
    load_model_json,
    save_model_json,
    load_workplace,
    save_workplace,
    load_tavily_api_key,
    save_tavily_api_key,
    ensure_config_dir,
    load_langsmith_config,
    _apply_langsmith_env,
    _test_connection,
)


class TestLoadModelJson:
    def test_no_file(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        mod._model_json.invalidate()
        monkeypatch.setattr(mod, "MODEL_JSON", tmp_path / "model.json")
        assert load_model_json() == {}

    def test_valid_file(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        mod._model_json.invalidate()
        f = tmp_path / "model.json"
        f.write_text(json.dumps({"default": {"model": "gpt-4o"}}))
        monkeypatch.setattr(mod, "MODEL_JSON", f)
        data = load_model_json()
        assert data["default"]["model"] == "gpt-4o"

    def test_mtime_cache(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        mod._model_json.invalidate()
        f = tmp_path / "model.json"
        f.write_text(json.dumps({"default": {"model": "a"}}))
        monkeypatch.setattr(mod, "MODEL_JSON", f)
        d1 = load_model_json()
        d2 = load_model_json()
        assert d1 is d2

    def test_invalid_json(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        mod._model_json.invalidate()
        f = tmp_path / "model.json"
        f.write_text("not json{{{")
        monkeypatch.setattr(mod, "MODEL_JSON", f)
        assert load_model_json() == {}


class TestSaveModelJson:
    def test_saves_and_invalidates_cache(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        mod._model_json.invalidate()
        f = tmp_path / "model.json"
        monkeypatch.setattr(mod, "MODEL_JSON", f)
        save_model_json({"default": {"model": "test"}})
        assert f.exists()
        data = json.loads(f.read_text())
        assert data["default"]["model"] == "test"
        assert mod._model_json._cache is None


class TestLoadWorkplace:
    def test_no_file_v2(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        monkeypatch.setattr(mod, "SETTING_JSON", tmp_path / "nope.json")
        assert load_workplace() is None

    def test_valid(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"workplace_path": str(tmp_path)}))
        monkeypatch.setattr(mod, "SETTING_JSON", f)
        assert load_workplace() == tmp_path


class TestSaveWorkplace:
    def test_saves(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        f = tmp_path / "settings.json"
        monkeypatch.setattr(mod, "SETTING_JSON", f)
        monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
        save_workplace(tmp_path / "myproject")
        data = json.loads(f.read_text())
        assert "myproject" in data["workplace_path"]


class TestLoadSaveTavilyApiKey:
    def test_no_file_v3(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setattr(mod, "SETTING_JSON", tmp_path / "nope.json")
        assert load_tavily_api_key() == ""

    def test_save_and_load(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        f = tmp_path / "settings.json"
        f.write_text("{}")
        monkeypatch.setattr(mod, "SETTING_JSON", f)
        monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
        save_tavily_api_key("tvly-test123")
        key = load_tavily_api_key()
        assert key == "tvly-test123"


class TestEnsureConfigDir:
    def test_creates_dir(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        d = tmp_path / "newconfig"
        monkeypatch.setattr(mod, "CONFIG_DIR", d)
        result = ensure_config_dir()
        assert d.exists()
        assert result == d


# ============================================================================
# Test LangSmith config functions
# ============================================================================


@pytest.mark.skipif(sys.platform != "win32", reason="LangSmith env persistence requires Windows")
class TestLoadLangsmithConfig:
    def test_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "my-proj")
        monkeypatch.setenv("LANGSMITH_TRACING", "true")

        cfg = load_langsmith_config()
        assert cfg["api_key"] == "lsv2_test_key"
        assert cfg["project"] == "my-proj"
        assert cfg["tracing"] is True

    def test_from_env_key_only(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        cfg = load_langsmith_config()
        assert cfg["api_key"] == "lsv2_test_key"
        assert cfg["project"] == "chcode"
        assert cfg["tracing"] is False

    def test_no_config(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        cfg = load_langsmith_config()
        assert cfg["api_key"] == ""
        assert cfg["project"] == ""
        assert cfg["tracing"] is False


@pytest.mark.skipif(sys.platform != "win32", reason="LangSmith env persistence requires Windows")
class TestApplyLangsmithEnv:
    def test_sets_all_env_vars(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

        with patch("chcode.config._persist_env"):
            _apply_langsmith_env(True, "my-proj", "lsv2_key")

        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_PROJECT"] == "my-proj"
        assert os.environ["LANGSMITH_API_KEY"] == "lsv2_key"

    def test_disabled(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        with patch("chcode.config._persist_env"):
            _apply_langsmith_env(False, "", "")

        assert os.environ["LANGSMITH_TRACING"] == "false"


class TestTestConnectionNullValue:
    @pytest.mark.asyncio
    async def test_null_value_with_choices_is_success(self):
        config = {"model": "test", "base_url": "http://x", "api_key": "k"}
        with patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_model:
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.side_effect = Exception("null value for 'choices'")
            result = await _test_connection(config, quiet=True)
            assert result is True

    @pytest.mark.asyncio
    async def test_null_value_without_choices_is_failure(self):
        config = {"model": "test", "base_url": "http://x", "api_key": "k"}
        with patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_model:
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.side_effect = Exception("null value for 'model'")
            result = await _test_connection(config, quiet=True)
            assert result is False


@pytest.mark.skipif(sys.platform != "win32", reason="LangSmith env persistence requires Windows")
class TestConfigureLangsmithBehavior:
    @pytest.mark.asyncio
    async def test_env_key_no_tracing_defaults_false(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        from chcode.config import configure_langsmith
        with patch("chcode.config._persist_env"):
            cfg = await configure_langsmith()
        assert cfg["tracing"] is False

    @pytest.mark.asyncio
    async def test_env_key_with_tracing_true(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "my-proj")
        monkeypatch.setenv("LANGSMITH_TRACING", "true")

        from chcode.config import configure_langsmith
        with patch("chcode.config._persist_env"):
            cfg = await configure_langsmith()
        assert cfg["tracing"] is True

    @pytest.mark.asyncio
    async def test_interactive_full_config(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        from chcode.config import configure_langsmith
        with patch("chcode.config.select", new_callable=AsyncMock, return_value="是"), \
             patch("chcode.config.text", new_callable=AsyncMock, side_effect=["my-proj", "lsv2_key"]), \
             patch("chcode.config._persist_env"):
            cfg = await configure_langsmith()

        assert cfg["tracing"] is True
        assert cfg["project"] == "my-proj"
        assert cfg["api_key"] == "lsv2_key"
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == "lsv2_key"

    @pytest.mark.asyncio
    async def test_interactive_skip(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        from chcode.config import configure_langsmith
        with patch("chcode.config.select", new_callable=AsyncMock, return_value="否"):
            cfg = await configure_langsmith()

        assert cfg["tracing"] is False
        assert cfg["api_key"] == ""

    @pytest.mark.asyncio
    async def test_interactive_no_key(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        from chcode.config import configure_langsmith
        with patch("chcode.config.select", new_callable=AsyncMock, return_value="是"), \
             patch("chcode.config.text", new_callable=AsyncMock, side_effect=["my-proj", ""]):
            cfg = await configure_langsmith()

        assert cfg["tracing"] is False
        assert cfg["api_key"] == ""
