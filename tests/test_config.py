import json
import os
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
    save_langsmith_config,
    _apply_langsmith_env,
    _test_connection,
    LANGSMITH_ENDPOINT,
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


class TestLoadLangsmithConfig:
    def test_from_env_vars(self, monkeypatch):
        import chcode.config as mod

        monkeypatch.setattr(mod, "SETTING_JSON", Path("/nonexistent"))
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
        monkeypatch.setenv("LANGCHAIN_PROJECT", "my-proj")
        monkeypatch.setenv("LANGCHAIN_TRACING", "true")
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

        cfg = load_langsmith_config()
        assert cfg["api_key"] == "lsv2_test_key"
        assert cfg["project"] == "my-proj"
        assert cfg["tracing"] is True

    def test_from_env_key_only_defaults_project(self, monkeypatch):
        import chcode.config as mod

        monkeypatch.setattr(mod, "SETTING_JSON", Path("/nonexistent"))
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
        monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

        cfg = load_langsmith_config()
        assert cfg["api_key"] == "lsv2_test_key"
        assert cfg["project"] == "chcode"
        assert cfg["tracing"] is False

    def test_from_setting_json(self, tmp_path, monkeypatch):
        import chcode.config as mod

        f = tmp_path / "settings.json"
        f.write_text(json.dumps({
            "langsmith_api_key": "lsv2_saved_key",
            "langsmith_project": "saved-proj",
            "langsmith_tracing": True,
        }))
        monkeypatch.setattr(mod, "SETTING_JSON", f)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

        cfg = load_langsmith_config()
        assert cfg["api_key"] == "lsv2_saved_key"
        assert cfg["project"] == "saved-proj"
        assert cfg["tracing"] is True

    def test_env_vars_take_priority(self, tmp_path, monkeypatch):
        import chcode.config as mod

        f = tmp_path / "settings.json"
        f.write_text(json.dumps({
            "langsmith_api_key": "old_key",
            "langsmith_project": "old-proj",
            "langsmith_tracing": False,
        }))
        monkeypatch.setattr(mod, "SETTING_JSON", f)
        monkeypatch.setenv("LANGSMITH_API_KEY", "new_key")
        monkeypatch.setenv("LANGCHAIN_PROJECT", "new-proj")
        monkeypatch.setenv("LANGCHAIN_TRACING", "true")
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

        cfg = load_langsmith_config()
        assert cfg["api_key"] == "new_key"
        assert cfg["project"] == "new-proj"
        assert cfg["tracing"] is True

    def test_env_tracing_false_not_overridden_by_file(self, tmp_path, monkeypatch):
        import chcode.config as mod

        f = tmp_path / "settings.json"
        f.write_text(json.dumps({
            "langsmith_api_key": "lsv2_saved_key",
            "langsmith_project": "saved-proj",
            "langsmith_tracing": True,
        }))
        monkeypatch.setattr(mod, "SETTING_JSON", f)
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_saved_key")
        monkeypatch.setenv("LANGCHAIN_PROJECT", "saved-proj")
        monkeypatch.setenv("LANGCHAIN_TRACING", "false")
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

        cfg = load_langsmith_config()
        assert cfg["tracing"] is False

    def test_no_config(self, monkeypatch):
        import chcode.config as mod

        monkeypatch.setattr(mod, "SETTING_JSON", Path("/nonexistent"))
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

        cfg = load_langsmith_config()
        assert cfg["api_key"] == ""
        assert cfg["project"] == ""
        assert cfg["tracing"] is False


class TestSaveLangsmithConfig:
    def test_save_new(self, tmp_path, monkeypatch):
        import chcode.config as mod

        f = tmp_path / "settings.json"
        monkeypatch.setattr(mod, "SETTING_JSON", f)
        monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)

        save_langsmith_config(True, "my-proj", "lsv2_key")
        data = json.loads(f.read_text())
        assert data["langsmith_tracing"] is True
        assert data["langsmith_project"] == "my-proj"
        assert data["langsmith_api_key"] == "lsv2_key"

    def test_save_preserves_other_keys(self, tmp_path, monkeypatch):
        import chcode.config as mod

        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"tavily_api_key": "tvly_old"}))
        monkeypatch.setattr(mod, "SETTING_JSON", f)
        monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)

        save_langsmith_config(False, "proj", "key")
        data = json.loads(f.read_text())
        assert data["tavily_api_key"] == "tvly_old"
        assert data["langsmith_tracing"] is False


class TestApplyLangsmithEnv:
    def test_sets_all_env_vars(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_TRACING", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)

        _apply_langsmith_env(True, "my-proj", "lsv2_key")

        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_PROJECT"] == "my-proj"
        assert os.environ["LANGSMITH_API_KEY"] == "lsv2_key"
        assert os.environ["LANGCHAIN_ENDPOINT"] == LANGSMITH_ENDPOINT

    def test_disabled(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_TRACING", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

        _apply_langsmith_env(False, "", "")

        assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


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


class TestConfigureLangsmithBehavior:
    @pytest.mark.asyncio
    async def test_env_key_no_tracing_env_defaults_false(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
        monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING", raising=False)

        from chcode.config import configure_langsmith
        cfg = await configure_langsmith()
        assert cfg["tracing"] is False

    @pytest.mark.asyncio
    async def test_env_key_with_tracing_v2_true(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
        monkeypatch.setenv("LANGCHAIN_PROJECT", "my-proj")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.delenv("LANGCHAIN_TRACING", raising=False)

        from chcode.config import configure_langsmith
        cfg = await configure_langsmith()
        assert cfg["tracing"] is True
