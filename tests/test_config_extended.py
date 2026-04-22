"""
Extended tests for chcode/config.py - interactive functions

Tests for:
- first_run_configure()
- configure_new_model()
- edit_current_model()
- switch_model()
- get_context_window_size(model)
- configure_tavily()
- get_default_model_config()
"""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


@pytest.fixture
def mock_config_dir(tmp_path: Path, monkeypatch):
    """Setup mock config directory"""
    import chcode.config as mod

    config_dir = tmp_path / ".chat"
    config_dir.mkdir()
    monkeypatch.setattr(mod, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(mod, "MODEL_JSON", config_dir / "model.json")
    monkeypatch.setattr(mod, "SETTING_JSON", config_dir / "chagent.json")
    return config_dir


class TestFirstRunConfigure:
    """Tests for first_run_configure()"""

    @pytest.mark.asyncio
    async def test_with_detected_env_key_and_success(self, mock_config_dir, monkeypatch):
        """Test first run with detected env key and successful connection"""
        import chcode.config as mod

        # Mock env key detection
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")

        # Mock prompts
        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select, patch(
            "chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI"
        ) as mock_model, patch("chcode.config.configure_tavily", new_callable=AsyncMock):

            mock_select.side_effect = [
                "OpenAI (检测到 OPENAI_API_KEY)",  # Config choice
                "gpt-4o",  # Model choice
            ]
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.return_value = MagicMock()

            result = await mod.first_run_configure()

            assert result is not None
            assert result["model"] == "gpt-4o"
            assert result["api_key"] == "sk-test123"
            assert result["base_url"] == "https://api.openai.com/v1"

    @pytest.mark.asyncio
    async def test_with_detected_env_key_connection_fails(self, mock_config_dir, monkeypatch):
        """Test first run with detected env key but connection fails"""
        import chcode.config as mod

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select, patch(
            "chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI"
        ) as mock_model:
            mock_select.side_effect = [
                "OpenAI (检测到 OPENAI_API_KEY)",
                "gpt-4o",
            ]
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.side_effect = Exception("Connection failed")

            result = await mod.first_run_configure()

            assert result is None

    @pytest.mark.asyncio
    async def test_no_detected_keys_choose_exit(self, mock_config_dir, monkeypatch):
        """Test first run with no detected keys, user exits"""
        import chcode.config as mod

        for var in mod.ENV_TO_CONFIG:
            monkeypatch.delenv(var, raising=False)

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select:
            mock_select.return_value = "退出"
            result = await mod.first_run_configure()

            assert result is None

    @pytest.mark.asyncio
    async def test_no_detected_keys_choose_manual(self, mock_config_dir, monkeypatch):
        """Test first run with no detected keys, user chooses manual"""
        import chcode.config as mod

        for var in mod.ENV_TO_CONFIG:
            monkeypatch.delenv(var, raising=False)

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select, patch(
            "chcode.config.configure_new_model", new_callable=AsyncMock
        ) as mock_configure:
            mock_select.return_value = "手动配置..."
            mock_configure.return_value = {"model": "test-model", "api_key": "key"}

            result = await mod.first_run_configure()

            assert result is not None
            mock_configure.assert_called_once()

    @pytest.mark.asyncio
    async def test_detected_keys_choose_manual(self, mock_config_dir, monkeypatch):
        """Test first run with detected keys but user chooses manual config"""
        import chcode.config as mod

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select, patch(
            "chcode.config.configure_new_model", new_callable=AsyncMock
        ) as mock_configure:
            mock_select.return_value = "手动配置..."
            mock_configure.return_value = {"model": "test-model", "api_key": "key"}

            result = await mod.first_run_configure()

            assert result is not None
            mock_configure.assert_called_once()

    @pytest.mark.asyncio
    async def test_detected_keys_user_cancels_model_select(self, mock_config_dir, monkeypatch):
        """Test first run with detected keys but user cancels model selection"""
        import chcode.config as mod

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select:
            mock_select.side_effect = [
                "OpenAI (检测到 OPENAI_API_KEY)",
                None,  # User cancels model selection
            ]

            result = await mod.first_run_configure()

            assert result is None


class TestConfigureNewModel:
    """Tests for configure_new_model()"""

    @pytest.mark.asyncio
    async def test_first_config_becomes_default(self, mock_config_dir):
        """Test first configuration becomes default"""
        import chcode.config as mod

        config = {
            "model": "test-model",
            "base_url": "https://api.test.com/v1",
            "api_key": "sk-test",
            "stream_usage": True,
        }

        with patch("chcode.config.model_config_form", new_callable=AsyncMock) as mock_form, patch(
            "chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI"
        ) as mock_model, patch("chcode.config.configure_tavily", new_callable=AsyncMock), \
            patch("chcode.config.select", new_callable=AsyncMock, return_value="手动配置..."):
            mock_form.return_value = config
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.return_value = MagicMock()

            result = await mod.configure_new_model()

            assert result is not None
            assert result["model"] == "test-model"

            # Verify it was saved as default
            data = mod.load_model_json()
            assert data["default"]["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_second_config_goes_to_fallback(self, mock_config_dir):
        """Test second configuration goes to fallback"""
        import chcode.config as mod

        # Setup existing default
        mod.save_model_json(
            {
                "default": {"model": "default-model", "api_key": "key1", "base_url": "https://api1.com"},
                "fallback": {},
            }
        )

        new_config = {
            "model": "fallback-model",
            "base_url": "https://api.test.com/v1",
            "api_key": "sk-test",
            "stream_usage": True,
        }

        with patch("chcode.config.model_config_form", new_callable=AsyncMock) as mock_form, patch(
            "chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI"
        ) as mock_model, patch("chcode.config.configure_tavily", new_callable=AsyncMock), \
            patch("chcode.config.select", new_callable=AsyncMock, return_value="手动配置..."):
            mock_form.return_value = new_config
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.return_value = MagicMock()

            result = await mod.configure_new_model()

            assert result is not None
            assert result["model"] == "fallback-model"

            # Verify it went to fallback
            data = mod.load_model_json()
            assert data["default"]["model"] == "default-model"
            assert "fallback-model" in data["fallback"]

    @pytest.mark.asyncio
    async def test_user_cancels_form(self, mock_config_dir):
        """Test user cancels configuration form"""
        import chcode.config as mod

        with patch("chcode.config.model_config_form", new_callable=AsyncMock) as mock_form, \
            patch("chcode.config.select", new_callable=AsyncMock, return_value=None):
            mock_form.return_value = None

            result = await mod.configure_new_model()

            assert result is None

    @pytest.mark.asyncio
    async def test_connection_test_fails(self, mock_config_dir):
        """Test configuration fails when connection test fails"""
        import chcode.config as mod

        config = {
            "model": "test-model",
            "base_url": "https://api.test.com/v1",
            "api_key": "sk-test",
            "stream_usage": True,
        }

        with patch("chcode.config.model_config_form", new_callable=AsyncMock) as mock_form, patch(
            "chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI"
        ) as mock_model, \
            patch("chcode.config.select", new_callable=AsyncMock, return_value="手动配置..."):
            mock_form.return_value = config
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.side_effect = Exception("Connection failed")

            result = await mod.configure_new_model()

            assert result is None


class TestEditCurrentModel:
    """Tests for edit_current_model()"""

    @pytest.mark.asyncio
    async def test_edit_existing_model(self, mock_config_dir):
        """Test editing existing model configuration"""
        import chcode.config as mod

        existing = {
            "model": "old-model",
            "base_url": "https://api.old.com/v1",
            "api_key": "sk-old",
            "stream_usage": True,
        }
        mod.save_model_json({"default": existing, "fallback": {}})

        updated = {
            "model": "new-model",
            "base_url": "https://api.new.com/v1",
            "api_key": "sk-new",
            "stream_usage": True,
        }

        with patch("chcode.config.model_config_form", new_callable=AsyncMock) as mock_form, patch(
            "chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI"
        ) as mock_model:
            mock_form.return_value = updated
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.return_value = MagicMock()

            result = await mod.edit_current_model()

            assert result is not None
            assert result["model"] == "new-model"

    @pytest.mark.asyncio
    async def test_no_current_model_creates_new(self, mock_config_dir):
        """Test edit when no current model exists triggers new config"""
        import chcode.config as mod

        mod.save_model_json({})

        with patch("chcode.config.configure_new_model", new_callable=AsyncMock) as mock_configure:
            mock_configure.return_value = {"model": "new-model", "api_key": "key"}

            result = await mod.edit_current_model()

            assert result is not None
            mock_configure.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_cancels_edit(self, mock_config_dir):
        """Test user cancels edit form"""
        import chcode.config as mod

        existing = {
            "model": "old-model",
            "base_url": "https://api.old.com/v1",
            "api_key": "sk-old",
            "stream_usage": True,
        }
        mod.save_model_json({"default": existing, "fallback": {}})

        with patch("chcode.config.model_config_form", new_callable=AsyncMock) as mock_form:
            mock_form.return_value = None

            result = await mod.edit_current_model()

            assert result is None


class TestSwitchModel:
    """Tests for switch_model()"""

    @pytest.mark.asyncio
    async def test_switch_to_fallback_model(self, mock_config_dir):
        """Test switching from default to a fallback model"""
        import chcode.config as mod

        default = {"model": "default-model", "api_key": "key1", "base_url": "https://api1.com"}
        fallback = {
            "fallback-model": {"model": "fallback-model", "api_key": "key2", "base_url": "https://api2.com"}
        }
        mod.save_model_json({"default": default, "fallback": fallback})

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select, patch(
            "chcode.config.confirm", new_callable=AsyncMock
        ) as mock_confirm:
            mock_select.return_value = "fallback-model"
            mock_confirm.return_value = True

            result = await mod.switch_model()

            assert result is not None
            assert result["model"] == "fallback-model"

            # Verify switch happened
            data = mod.load_model_json()
            assert data["default"]["model"] == "fallback-model"
            assert "default-model" in data["fallback"]

    @pytest.mark.asyncio
    async def test_switch_with_current_tag(self, mock_config_dir):
        """Test switching when default is also in fallback with tag"""
        import chcode.config as mod

        default = {"model": "model-a", "api_key": "key1", "base_url": "https://api1.com"}
        fallback = {
            "model-b": {"model": "model-b", "api_key": "key2", "base_url": "https://api2.com"}
        }
        mod.save_model_json({"default": default, "fallback": fallback})

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select, patch(
            "chcode.config.confirm", new_callable=AsyncMock
        ) as mock_confirm:
            mock_select.return_value = "model-b"
            mock_confirm.return_value = True

            result = await mod.switch_model()

            assert result is not None
            assert result["model"] == "model-b"

    @pytest.mark.asyncio
    async def test_no_default_model(self, mock_config_dir):
        """Test switch when no default model exists"""
        import chcode.config as mod

        mod.save_model_json({})

        with patch("chcode.config.configure_new_model", new_callable=AsyncMock) as mock_configure:
            mock_configure.return_value = {"model": "new-model", "api_key": "key"}

            result = await mod.switch_model()

            assert result is not None
            mock_configure.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_fallback_models(self, mock_config_dir):
        """Test switch when no fallback models available"""
        import chcode.config as mod

        default = {"model": "default-model", "api_key": "key1", "base_url": "https://api1.com"}
        mod.save_model_json({"default": default, "fallback": {}})

        result = await mod.switch_model()

        assert result is None

    @pytest.mark.asyncio
    async def test_user_cancels_selection(self, mock_config_dir):
        """Test user cancels model selection"""
        import chcode.config as mod

        default = {"model": "default-model", "api_key": "key1", "base_url": "https://api1.com"}
        fallback = {
            "fallback-model": {"model": "fallback-model", "api_key": "key2", "base_url": "https://api2.com"}
        }
        mod.save_model_json({"default": default, "fallback": fallback})

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select:
            mock_select.return_value = None

            result = await mod.switch_model()

            assert result is None

    @pytest.mark.asyncio
    async def test_user_rejects_confirmation(self, mock_config_dir):
        """Test user rejects confirmation dialog"""
        import chcode.config as mod

        default = {"model": "default-model", "api_key": "key1", "base_url": "https://api1.com"}
        fallback = {
            "fallback-model": {"model": "fallback-model", "api_key": "key2", "base_url": "https://api2.com"}
        }
        mod.save_model_json({"default": default, "fallback": fallback})

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select, patch(
            "chcode.config.confirm", new_callable=AsyncMock
        ) as mock_confirm:
            mock_select.return_value = "fallback-model"
            mock_confirm.return_value = False

            result = await mod.switch_model()

            assert result is None


class TestGetDefaultModelConfig:
    """Tests for get_default_model_config()"""

    def test_returns_default_when_exists(self, mock_config_dir):
        """Test returns default config when it exists"""
        import chcode.config as mod

        expected = {"model": "test-model", "api_key": "key"}
        mod.save_model_json({"default": expected, "fallback": {}})

        result = mod.get_default_model_config()

        assert result == expected

    def test_returns_none_when_no_default(self, mock_config_dir):
        """Test returns None when no default config"""
        import chcode.config as mod

        mod.save_model_json({"fallback": {}})

        result = mod.get_default_model_config()

        assert result is None

    def test_returns_none_when_empty_json(self, mock_config_dir):
        """Test returns None when model.json is empty"""
        import chcode.config as mod

        mod.save_model_json({})

        result = mod.get_default_model_config()

        assert result is None


class TestGetContextWindowSize:
    """Tests for get_context_window_size() - additional coverage"""

    def test_case_insensitive_match(self):
        """Test case-insensitive model name matching"""
        import chcode.config as mod

        assert mod.get_context_window_size("GPT-4O") == 128000

    def test_slash_prefix_handling(self):
        """Test handling of org/ prefixed models"""
        import chcode.config as mod

        assert mod.get_context_window_size("org/gpt-4o-mini") == 128000

    def test_substring_in_long_name(self):
        """Test substring matching in longer model names"""
        import chcode.config as mod

        assert mod.get_context_window_size("my-custom-deepseek-chat-v2") == 65536

    def test_unknown_model_returns_default(self):
        """Test unknown model returns default context window"""
        import chcode.config as mod

        assert mod.get_context_window_size("totally-unknown-model-x") == mod._DEFAULT_CONTEXT_WINDOW

    def test_glm_models(self):
        """Test GLM model context windows"""
        import chcode.config as mod

        assert mod.get_context_window_size("glm-5.1") == 200000
        assert mod.get_context_window_size("glm-4.7") == 200000

    def test_qwen_models(self):
        """Test Qwen model context windows"""
        import chcode.config as mod

        assert mod.get_context_window_size("qwen3.5-plus") == 1000000
        assert mod.get_context_window_size("qwen") == 256000


class TestConfigureTavily:
    """Tests for configure_tavily()"""

    @pytest.mark.asyncio
    async def test_with_env_var(self, mock_config_dir, monkeypatch):
        """Test Tavily configuration with environment variable"""
        import chcode.config as mod

        monkeypatch.setenv("TAVILY_API_KEY", "tvly-env-key")

        with patch("chcode.utils.tools.update_tavily_api_key") as mock_update:
            await mod.configure_tavily()

            mock_update.assert_called_once_with("tvly-env-key")

    @pytest.mark.asyncio
    async def test_with_existing_saved_key(self, mock_config_dir, monkeypatch):
        """Test Tavily configuration with existing saved key"""
        import chcode.config as mod

        for var in ["TAVILY_API_KEY"] + list(mod.ENV_TO_CONFIG.keys()):
            monkeypatch.delenv(var, raising=False)

        # Save existing key
        mod.save_tavily_api_key("tvly-saved-key")

        with patch("chcode.utils.tools.update_tavily_api_key") as mock_update:
            await mod.configure_tavily()

            mock_update.assert_called_once_with("tvly-saved-key")

    @pytest.mark.asyncio
    async def test_user_declines_configuration(self, mock_config_dir, monkeypatch):
        """Test user declines Tavily configuration"""
        import chcode.config as mod

        for var in ["TAVILY_API_KEY"] + list(mod.ENV_TO_CONFIG.keys()):
            monkeypatch.delenv(var, raising=False)

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select:
            mock_select.return_value = "否"

            await mod.configure_tavily()

        # Verify user was asked and declined
        mock_select.assert_called_once()
        assert mock_select.return_value == "否"

    @pytest.mark.asyncio
    async def test_user_provides_new_key(self, mock_config_dir, monkeypatch):
        """Test user provides new Tavily API key"""
        import chcode.config as mod

        for var in ["TAVILY_API_KEY"] + list(mod.ENV_TO_CONFIG.keys()):
            monkeypatch.delenv(var, raising=False)

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select, patch(
            "chcode.config.text", new_callable=AsyncMock
        ) as mock_text, patch("chcode.utils.tools.update_tavily_api_key") as mock_update:
            mock_select.return_value = "是"
            mock_text.return_value = "tvly-new-key"

            await mod.configure_tavily()

            mock_update.assert_called_once_with("tvly-new-key")
            assert mod.load_tavily_api_key() == "tvly-new-key"

    @pytest.mark.asyncio
    async def test_user_cancels_key_input(self, mock_config_dir, monkeypatch):
        """Test user cancels Tavily key input"""
        import chcode.config as mod

        for var in ["TAVILY_API_KEY"] + list(mod.ENV_TO_CONFIG.keys()):
            monkeypatch.delenv(var, raising=False)

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select, patch(
            "chcode.config.text", new_callable=AsyncMock
        ) as mock_text:
            mock_select.return_value = "是"
            mock_text.return_value = ""

            await mod.configure_tavily()

        # Verify user was asked for input and cancelled
        mock_text.assert_called_once()
        assert mock_text.return_value == ""

    @pytest.mark.asyncio
    async def test_user_cancels_initial_prompt(self, mock_config_dir, monkeypatch):
        """Test user cancels initial Tavily prompt"""
        import chcode.config as mod

        for var in ["TAVILY_API_KEY"] + list(mod.ENV_TO_CONFIG.keys()):
            monkeypatch.delenv(var, raising=False)

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select:
            mock_select.return_value = None

            await mod.configure_tavily()

        # Verify user cancelled the initial prompt
        mock_select.assert_called_once()
        assert mock_select.return_value is None


class TestFirstRunConfigureExit:
    """Cover lines 130-133: user selects exit from first_run_configure."""

    @pytest.mark.asyncio
    async def test_detected_keys_select_none_exits(self, mock_config_dir, monkeypatch):
        """Lines 130-133: detected keys present, user cancels first select (returns None)."""
        import chcode.config as mod

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select:
            mock_select.return_value = None  # User cancels the first select
            result = await mod.first_run_configure()
            assert result is None

    @pytest.mark.asyncio
    async def test_detected_keys_select_exit(self, mock_config_dir, monkeypatch):
        """Lines 130-133: detected keys present, user selects 退出."""
        import chcode.config as mod

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select:
            mock_select.return_value = "退出"
            result = await mod.first_run_configure()
            assert result is None


class TestConfigureNewModelNullChoices:
    """Cover lines 244-251: configure_new_model connection test with 'null value for choices'."""

    @pytest.mark.asyncio
    async def test_null_choices_error_continues(self, mock_config_dir):
        """Lines 244-251: 'null value for choices' in error is treated as success."""
        import chcode.config as mod

        config = {
            "model": "test-model",
            "base_url": "https://api.test.com/v1",
            "api_key": "sk-test",
            "stream_usage": True,
        }

        with patch("chcode.config.model_config_form", new_callable=AsyncMock) as mock_form, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_model, \
             patch("chcode.config.configure_tavily", new_callable=AsyncMock), \
             patch("chcode.config.select", new_callable=AsyncMock, return_value="手动配置..."):
            mock_form.return_value = config
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            # The "null value for 'choices'" error should be silently ignored
            mock_model_inst.invoke.side_effect = Exception("null value for 'choices'")

            result = await mod.configure_new_model()

            # Should succeed because the error contains 'null value for choices'
            assert result is not None
            assert result["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_other_error_in_connection_test(self, mock_config_dir):
        """Lines 248-251: other exception shows error message."""
        import chcode.config as mod

        config = {
            "model": "test-model",
            "base_url": "https://api.test.com/v1",
            "api_key": "sk-test",
            "stream_usage": True,
        }

        with patch("chcode.config.model_config_form", new_callable=AsyncMock) as mock_form, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_model, \
             patch("chcode.config.console") as mock_console, \
             patch("chcode.config.select", new_callable=AsyncMock, return_value="手动配置..."):
            mock_form.return_value = config
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.side_effect = Exception("Connection refused")

            result = await mod.configure_new_model()

            assert result is None
            # Should have printed the error
            assert mock_console.print.called


class TestEditCurrentModelConnectionErrors:
    """Cover lines 244-251: edit_current_model connection test error paths."""

    @pytest.mark.asyncio
    async def test_edit_null_choices_error_continues(self, mock_config_dir):
        """Lines 244-251: 'null value for choices' error in edit is silently ignored."""
        import chcode.config as mod

        existing = {
            "model": "old-model",
            "base_url": "https://api.old.com/v1",
            "api_key": "sk-old",
            "stream_usage": True,
        }
        mod.save_model_json({"default": existing, "fallback": {}})

        updated = {
            "model": "new-model",
            "base_url": "https://api.new.com/v1",
            "api_key": "sk-new",
            "stream_usage": True,
        }

        with patch("chcode.config.model_config_form", new_callable=AsyncMock) as mock_form, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_model:
            mock_form.return_value = updated
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.side_effect = Exception("null value for 'choices'")

            result = await mod.edit_current_model()
            # Should succeed despite the error
            assert result is not None
            assert result["model"] == "new-model"

    @pytest.mark.asyncio
    async def test_edit_connection_error_returns_none(self, mock_config_dir):
        """Lines 248-251: other error in edit connection test."""
        import chcode.config as mod

        existing = {
            "model": "old-model",
            "base_url": "https://api.old.com/v1",
            "api_key": "sk-old",
            "stream_usage": True,
        }
        mod.save_model_json({"default": existing, "fallback": {}})

        updated = {
            "model": "new-model",
            "base_url": "https://api.new.com/v1",
            "api_key": "sk-new",
            "stream_usage": True,
        }

        with patch("chcode.config.model_config_form", new_callable=AsyncMock) as mock_form, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_model, \
             patch("chcode.config.console") as mock_console:
            mock_form.return_value = updated
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.side_effect = Exception("Connection refused")

            result = await mod.edit_current_model()
            assert result is None
            assert mock_console.print.called


class TestLoadWorkplaceException:
    """Cover line 308: load_workplace exception path."""

    def test_load_workplace_json_parse_error(self, mock_config_dir):
        """Line 308: JSON parse error in load_workplace."""
        import chcode.config as mod

        # Write invalid JSON to setting file
        mod.SETTING_JSON.write_text("not valid json{{{", encoding="utf-8")

        result = mod.load_workplace()
        assert result is None


class TestSaveWorkplaceException:
    """Cover lines 317-319: save_workplace exception path."""

    def test_save_workplace_json_parse_error(self, mock_config_dir):
        """Lines 317-319: existing setting file has invalid JSON."""
        import chcode.config as mod

        # Write invalid JSON
        mod.SETTING_JSON.write_text("bad json {{{", encoding="utf-8")

        # Should not raise, should create fresh data
        from pathlib import Path
        mod.save_workplace(Path("/tmp/test_workplace"))

        # Verify it saved correctly
        data = json.loads(mod.SETTING_JSON.read_text(encoding="utf-8"))
        assert "workplace_path" in data


class TestLoadTavilyApiKeyException:
    """Cover line 333: load_tavily_api_key exception path."""

    def test_load_tavily_json_parse_error(self, mock_config_dir, monkeypatch):
        """Line 333: JSON parse error in load_tavily_api_key."""
        import chcode.config as mod

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        mod.SETTING_JSON.write_text("bad json {{{", encoding="utf-8")

        result = mod.load_tavily_api_key()
        assert result == ""


class TestSaveTavilyApiKeyException:
    """Cover line 345: save_tavily_api_key exception path."""

    def test_save_tavily_json_parse_error(self, mock_config_dir):
        """Line 345: existing setting file has invalid JSON."""
        import chcode.config as mod

        mod.SETTING_JSON.write_text("bad json {{{", encoding="utf-8")

        # Should not raise
        mod.save_tavily_api_key("tvly-test-key")

        # Verify saved correctly
        data = json.loads(mod.SETTING_JSON.read_text(encoding="utf-8"))
        assert data.get("tavily_api_key") == "tvly-test-key"


class TestConfigureTavilySavedKeyException:
    """Cover line 414: configure_tavily exception when reading saved key."""

    @pytest.mark.asyncio
    async def test_configure_tavily_saved_key_read_error(self, mock_config_dir, monkeypatch):
        """Line 414: exception when reading tavily key from settings."""
        import chcode.config as mod

        for var in ["TAVILY_API_KEY"] + list(mod.ENV_TO_CONFIG.keys()):
            monkeypatch.delenv(var, raising=False)

        # Write invalid JSON
        mod.SETTING_JSON.write_text("bad json {{{", encoding="utf-8")

        with patch("chcode.config.select", new_callable=AsyncMock) as mock_select:
            mock_select.return_value = "否"
            await mod.configure_tavily()

        # Should not raise, should proceed to ask user
        mock_select.assert_called_once()
        assert mock_select.return_value == "否"


class TestConfigureModelscope:
    """Tests for _configure_modelscope_with_test() and configure_modelscope()."""

    @pytest.mark.asyncio
    async def test_modelscope_first_time(self, mock_config_dir):
        """First-time config: no existing model.json -> full ModelScope config saved."""
        import chcode.config as mod
        from chcode.prompts import MODELSCOPE_PRESETS

        with patch("chcode.prompts.configure_modelscope", new_callable=AsyncMock) as mock_ms, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_model, \
             patch("chcode.config.configure_tavily", new_callable=AsyncMock):
            ms_result = {
                "default": {**MODELSCOPE_PRESETS[0], "api_key": "ms-key"},
                "fallback": {
                    m["model"]: {**m, "api_key": "ms-key"}
                    for m in MODELSCOPE_PRESETS[1:]
                },
            }
            mock_ms.return_value = ms_result
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.return_value = MagicMock()

            result = await mod._configure_modelscope_with_test()

            assert result is not None
            assert result["model"] == "ZhipuAI/GLM-5"
            assert result["api_key"] == "ms-key"

            data = mod.load_model_json()
            assert data["default"]["model"] == "ZhipuAI/GLM-5"
            assert len(data["fallback"]) == 11

    @pytest.mark.asyncio
    async def test_modelscope_merge_existing(self, mock_config_dir):
        """Existing config: old default moves to fallback, ModelScope presets merged in."""
        import chcode.config as mod
        from chcode.prompts import MODELSCOPE_PRESETS

        mod.save_model_json({
            "default": {"model": "old-model", "api_key": "k", "base_url": "https://x.com/v1", "stream_usage": True},
            "fallback": {"existing-fb": {"model": "existing-fb", "api_key": "k2"}},
        })

        with patch("chcode.prompts.configure_modelscope", new_callable=AsyncMock) as mock_ms, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_model, \
             patch("chcode.config.configure_tavily", new_callable=AsyncMock):
            ms_result = {
                "default": {**MODELSCOPE_PRESETS[0], "api_key": "ms-key"},
                "fallback": {
                    m["model"]: {**m, "api_key": "ms-key"}
                    for m in MODELSCOPE_PRESETS[1:]
                },
            }
            mock_ms.return_value = ms_result
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.return_value = MagicMock()

            result = await mod._configure_modelscope_with_test()

            data = mod.load_model_json()
            assert data["default"]["model"] == "ZhipuAI/GLM-5"
            assert "old-model" in data["fallback"]
            assert "existing-fb" in data["fallback"]

    @pytest.mark.asyncio
    async def test_modelscope_connection_fails(self, mock_config_dir):
        """Connection test fails -> returns None without saving."""
        import chcode.config as mod
        from chcode.prompts import MODELSCOPE_PRESETS

        with patch("chcode.prompts.configure_modelscope", new_callable=AsyncMock) as mock_ms, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_model, \
             patch("chcode.config.console") as mock_console:
            mock_ms.return_value = {
                "default": {**MODELSCOPE_PRESETS[0], "api_key": "ms-key"},
                "fallback": {m["model"]: {**m, "api_key": "ms-key"} for m in MODELSCOPE_PRESETS[1:]},
            }
            mock_model_inst = MagicMock()
            mock_model.return_value = mock_model_inst
            mock_model_inst.invoke.side_effect = Exception("Connection refused")

            result = await mod._configure_modelscope_with_test()

            assert result is None
            assert not mod.load_model_json().get("default")

    @pytest.mark.asyncio
    async def test_configure_modelscope_manual_key(self):
        """configure_modelscope() manual key input."""
        from chcode.prompts import configure_modelscope

        with patch("chcode.prompts.select", new_callable=AsyncMock, return_value="手动输入..."), \
             patch("chcode.prompts.password", new_callable=AsyncMock, return_value="ms-test-key"):
            result = await configure_modelscope()

        assert result is not None
        assert result["default"]["api_key"] == "ms-test-key"
        assert len(result["fallback"]) == 11

    @pytest.mark.asyncio
    async def test_configure_modelscope_cancel(self):
        """configure_modelscope() returns None when user cancels select."""
        from chcode.prompts import configure_modelscope

        with patch("chcode.prompts.select", new_callable=AsyncMock, return_value=None):
            result = await configure_modelscope()

        assert result is None

    @pytest.mark.asyncio
    async def test_configure_modelscope_empty_key(self):
        """configure_modelscope() returns None for empty API key."""
        from chcode.prompts import configure_modelscope

        with patch("chcode.prompts.select", new_callable=AsyncMock, return_value="手动输入..."), \
             patch("chcode.prompts.password", new_callable=AsyncMock, return_value=""):
            result = await configure_modelscope()

        assert result is None

    @pytest.mark.asyncio
    async def test_configure_modelscope_with_env_token(self):
        """configure_modelscope() detects ModelScopeToken env var."""
        import os as _os
        from chcode.prompts import configure_modelscope

        with patch.dict(_os.environ, {"ModelScopeToken": "env-ms-key"}), \
             patch("chcode.prompts.select", new_callable=AsyncMock, return_value="ModelScopeToken (ModelScope)"):
            result = await configure_modelscope()

        assert result is not None
        assert result["default"]["api_key"] == "env-ms-key"

    @pytest.mark.asyncio
    async def test_modelscope_configure_returns_none(self, mock_config_dir):
        """configure_modelscope returns None -> _configure_modelscope_with_test returns None."""
        import chcode.config as mod

        with patch("chcode.prompts.configure_modelscope", new_callable=AsyncMock, return_value=None):
            result = await mod._configure_modelscope_with_test()

        assert result is None

    @pytest.mark.asyncio
    async def test_first_run_detected_modelscope(self, mock_config_dir):
        """first_run_configure with detected key chooses 魔搭 path."""
        import chcode.config as mod

        with patch("chcode.config.detect_env_api_keys", return_value=[
            {"name": "TestProvider", "env_var": "TEST_KEY", "base_url": "https://x.com", "models": ["m1"], "api_key": "k"}
        ]), \
             patch("chcode.config.select", new_callable=AsyncMock, return_value="魔搭快捷配置..."), \
             patch("chcode.prompts.configure_modelscope", new_callable=AsyncMock, return_value=None):
            result = await mod.first_run_configure()

        # Returns None because configure_modelscope returns None
        assert result is None

    @pytest.mark.asyncio
    async def test_first_run_no_env_modelscope(self, mock_config_dir):
        """first_run_configure no env vars, user picks 魔搭."""
        import chcode.config as mod

        with patch("chcode.config.detect_env_api_keys", return_value=[]), \
             patch("chcode.config.select", new_callable=AsyncMock, return_value="魔搭快捷配置..."), \
             patch("chcode.prompts.configure_modelscope", new_callable=AsyncMock, return_value=None):
            result = await mod.first_run_configure()

        assert result is None

    @pytest.mark.asyncio
    async def test_configure_new_model_selects_modelscope(self, mock_config_dir):
        """configure_new_model user selects 魔搭."""
        import chcode.config as mod

        with patch("chcode.config.select", new_callable=AsyncMock, return_value="魔搭快捷配置..."), \
             patch("chcode.prompts.configure_modelscope", new_callable=AsyncMock, return_value=None):
            result = await mod.configure_new_model()

        assert result is None

    @pytest.mark.asyncio
    async def test_configure_new_model_form_returns_none(self, mock_config_dir):
        """configure_new_model: user selects manual, model_config_form returns None."""
        import chcode.config as mod

        with patch("chcode.config.select", new_callable=AsyncMock, return_value="手动配置..."), \
             patch("chcode.config.model_config_form", new_callable=AsyncMock, return_value=None):
            result = await mod.configure_new_model()

        assert result is None
