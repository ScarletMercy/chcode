"""
Tests for chcode/i18n.py — t(), set_language, detect_locale_language, os_env_lang,
load_language, save_language, and ChatREPL._cmd_lang.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from chcode.i18n import (
    t,
    set_language,
    get_language,
    detect_locale_language,
    os_env_lang,
    DEFAULT_LANG,
    SUPPORTED_LANGS,
)


# ============================================================================
# TestTFunction
# ============================================================================


class TestTFunction:
    """Tests for the t() translation function."""

    def test_existing_key_zh(self):
        set_language("zh")
        assert t("chat.goodbye") == "再见！"

    def test_existing_key_en(self):
        set_language("en")
        assert t("chat.goodbye") == "Goodbye!"

    def test_missing_key_fallback_to_key(self):
        """Missing key falls back to the key string itself."""
        set_language("zh")
        assert t("nonexistent.key.xyz") == "nonexistent.key.xyz"

    def test_missing_key_en_fallback_zh(self):
        """In en mode, missing key falls back to zh then to key."""
        set_language("en")
        assert t("nonexistent.key.xyz") == "nonexistent.key.xyz"

    def test_format_placeholders(self):
        set_language("zh")
        result = t("chat.unknown_command", command="/foo")
        assert "/foo" in result

    def test_format_placeholders_en(self):
        set_language("en")
        result = t("chat.unknown_command", command="/bar")
        assert "/bar" in result

    def test_positional_only_key(self):
        """key is positional-only (/); passing a keyword named 'key' goes into **fmt
        without conflicting with the positional parameter."""
        set_language("zh")
        # 'key' as keyword goes into **fmt; since the template has no {key} placeholder,
        # .format(key=...) is a no-op and returns the normal translation.
        result = t("chat.goodbye", key="value")
        assert "再见" in result

    def test_no_format_no_braces(self):
        """Without fmt kwargs, .format() is not called — key with braces returned as-is."""
        set_language("zh")
        # A key that exists but has no placeholders; calling without fmt should work
        result = t("chat.goodbye")
        assert "再见" in result

    def test_format_multiple_placeholders(self):
        set_language("zh")
        result = t("agent.retry_in", count=1, max=3, delay=2, error="boom")
        assert "1" in result
        assert "3" in result
        assert "boom" in result


# ============================================================================
# TestSetGetLanguage
# ============================================================================


class TestSetGetLanguage:
    """Tests for set_language() / get_language()."""

    def test_set_valid_zh(self):
        set_language("zh")
        assert get_language() == "zh"

    def test_set_valid_en(self):
        set_language("en")
        assert get_language() == "en"

    def test_set_case_insensitive(self):
        assert set_language("ZH") == "zh"
        assert get_language() == "zh"
        assert set_language("EN") == "en"
        assert get_language() == "en"

    def test_set_invalid_falls_back(self):
        result = set_language("fr")
        assert result == DEFAULT_LANG
        assert get_language() == DEFAULT_LANG

    def test_set_none_falls_back(self):
        result = set_language(None)
        assert result == DEFAULT_LANG
        assert get_language() == DEFAULT_LANG

    def test_set_empty_falls_back(self):
        result = set_language("")
        assert result == DEFAULT_LANG

    def test_returns_set_language(self):
        assert set_language("en") == "en"
        assert set_language("zh") == "zh"

    def test_supported_langs(self):
        assert "zh" in SUPPORTED_LANGS
        assert "en" in SUPPORTED_LANGS


# ============================================================================
# TestDetectLocaleLanguage
# ============================================================================


class TestDetectLocaleLanguage:
    """Tests for detect_locale_language()."""

    def test_zh_locale_code(self):
        """Unix-style locale code 'zh_CN' → 'zh'."""
        with patch("chcode.i18n._locale.getlocale", return_value=("zh_CN", "UTF-8")):
            with patch("chcode.i18n.os_env_lang", return_value=""):
                assert detect_locale_language() == "zh"

    def test_en_locale_code(self):
        """Unix-style locale code 'en_US' → 'en'."""
        with patch("chcode.i18n._locale.getlocale", return_value=("en_US", "UTF-8")):
            with patch("chcode.i18n.os_env_lang", return_value=""):
                assert detect_locale_language() == "en"

    def test_chinese_locale_name(self):
        """Windows-style full name 'Chinese (Simplified)_China' → 'zh'."""
        with patch("chcode.i18n._locale.getlocale", return_value=("Chinese (Simplified)_China", "936")):
            with patch("chcode.i18n.os_env_lang", return_value=""):
                assert detect_locale_language() == "zh"

    def test_empty_locale_fallback_env(self):
        """getlocale returns (None, None) → fallback to os_env_lang."""
        with patch("chcode.i18n._locale.getlocale", return_value=(None, None)):
            with patch("chcode.i18n.os_env_lang", return_value="zh_CN.UTF-8"):
                assert detect_locale_language() == "zh"

    def test_empty_locale_empty_env(self):
        """getlocale returns (None, None) and no env → 'en'."""
        with patch("chcode.i18n._locale.getlocale", return_value=(None, None)):
            with patch("chcode.i18n.os_env_lang", return_value=""):
                assert detect_locale_language() == "en"

    def test_exception_fallback(self):
        """getlocale raises → fallback to os_env_lang, no crash."""
        with patch("chcode.i18n._locale.getlocale", side_effect=Exception("boom")):
            with patch("chcode.i18n.os_env_lang", return_value="en_US.UTF-8"):
                assert detect_locale_language() == "en"

    def test_exception_empty_env(self):
        """getlocale raises and no env → 'en'."""
        with patch("chcode.i18n._locale.getlocale", side_effect=Exception("boom")):
            with patch("chcode.i18n.os_env_lang", return_value=""):
                assert detect_locale_language() == "en"


# ============================================================================
# TestOsEnvLang
# ============================================================================


class TestOsEnvLang:
    """Tests for os_env_lang()."""

    def test_reads_LANG(self, monkeypatch):
        monkeypatch.setenv("LANG", "zh_CN.UTF-8")
        monkeypatch.delenv("LC_ALL", raising=False)
        monkeypatch.delenv("LC_MESSAGES", raising=False)
        assert os_env_lang() == "zh_CN.UTF-8"

    def test_reads_LC_ALL(self, monkeypatch):
        """When LANG is unset, LC_ALL is read."""
        monkeypatch.delenv("LANG", raising=False)
        monkeypatch.setenv("LC_ALL", "zh_CN.UTF-8")
        monkeypatch.delenv("LC_MESSAGES", raising=False)
        assert os_env_lang() == "zh_CN.UTF-8"

    def test_lang_takes_priority(self, monkeypatch):
        """LANG is checked first (current implementation order: LANG → LC_ALL → LC_MESSAGES)."""
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        monkeypatch.setenv("LC_ALL", "zh_CN.UTF-8")
        monkeypatch.delenv("LC_MESSAGES", raising=False)
        assert os_env_lang() == "en_US.UTF-8"

    def test_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("LANG", raising=False)
        monkeypatch.delenv("LC_ALL", raising=False)
        monkeypatch.delenv("LC_MESSAGES", raising=False)
        assert os_env_lang() == ""

    def test_LC_MESSAGES(self, monkeypatch):
        monkeypatch.delenv("LANG", raising=False)
        monkeypatch.delenv("LC_ALL", raising=False)
        monkeypatch.setenv("LC_MESSAGES", "en_US.UTF-8")
        assert os_env_lang() == "en_US.UTF-8"


# ============================================================================
# TestLoadSaveLanguage
# ============================================================================


@pytest.fixture
def mock_config_dir(tmp_path: Path, monkeypatch):
    """Setup mock config directory for language persistence tests."""
    import chcode.config as mod

    config_dir = tmp_path / ".chat"
    config_dir.mkdir()
    monkeypatch.setattr(mod, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(mod, "MODEL_JSON", config_dir / "model.json")
    monkeypatch.setattr(mod, "SETTING_JSON", config_dir / "chagent.json")
    return config_dir


class TestLoadSaveLanguage:
    """Tests for load_language() / save_language() in config.py."""

    def test_save_then_load(self, mock_config_dir):
        from chcode.config import save_language, load_language

        save_language("en")
        assert load_language() == "en"

    def test_save_zh(self, mock_config_dir):
        from chcode.config import save_language, load_language

        save_language("zh")
        assert load_language() == "zh"

    def test_save_invalid_ignored(self, mock_config_dir):
        from chcode.config import save_language, load_language

        save_language("fr")
        assert load_language() is None

    def test_load_none_when_unset(self, mock_config_dir):
        from chcode.config import load_language

        assert load_language() is None

    def test_load_invalid_returns_none(self, mock_config_dir):
        """Corrupted language value in json → returns None."""
        import json
        import chcode.config as mod

        setting_path = mod.SETTING_JSON
        setting_path.write_text(
            json.dumps({"language": "fr"}), encoding="utf-8"
        )
        from chcode.config import load_language

        assert load_language() is None

    def test_overwrite_existing(self, mock_config_dir):
        from chcode.config import save_language, load_language

        save_language("en")
        assert load_language() == "en"
        save_language("zh")
        assert load_language() == "zh"


# ============================================================================
# TestCmdLang
# ============================================================================


class TestCmdLang:
    """Tests for ChatREPL._cmd_lang."""

    @pytest.mark.asyncio
    async def test_switch_to_en(self):
        from chcode.chat import ChatREPL
        from chcode.i18n import get_language

        set_language("zh")
        repl = ChatREPL()

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel, \
             patch("chcode.chat.render_success"), \
             patch("chcode.config.save_language") as mock_save:
            mock_sel.return_value = "English"
            await repl._cmd_lang("")

            assert get_language() == "en"
            mock_save.assert_called_once_with("en")

    @pytest.mark.asyncio
    async def test_switch_to_zh(self):
        from chcode.chat import ChatREPL
        from chcode.i18n import get_language

        set_language("en")
        repl = ChatREPL()

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel, \
             patch("chcode.chat.render_success"), \
             patch("chcode.config.save_language") as mock_save:
            mock_sel.return_value = "中文 (Chinese)"
            await repl._cmd_lang("")

            assert get_language() == "zh"
            mock_save.assert_called_once_with("zh")

    @pytest.mark.asyncio
    async def test_cancel_does_nothing(self):
        from chcode.chat import ChatREPL
        from chcode.i18n import get_language

        set_language("en")
        repl = ChatREPL()

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel, \
             patch("chcode.chat.render_success"), \
             patch("chcode.config.save_language") as mock_save:
            mock_sel.return_value = None
            await repl._cmd_lang("")

            assert get_language() == "en"
            mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_highlights_current_zh(self):
        from chcode.chat import ChatREPL

        set_language("zh")
        repl = ChatREPL()

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel, \
             patch("chcode.chat.render_success"), \
             patch("chcode.config.save_language"):
            mock_sel.return_value = "中文 (Chinese)"
            await repl._cmd_lang("")

            call_kwargs = mock_sel.call_args
            default = call_kwargs[1].get("default") or call_kwargs.kwargs.get("default")
            assert default == "中文 (Chinese)"

    @pytest.mark.asyncio
    async def test_default_highlights_current_en(self):
        from chcode.chat import ChatREPL

        set_language("en")
        repl = ChatREPL()

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel, \
             patch("chcode.chat.render_success"), \
             patch("chcode.config.save_language"):
            mock_sel.return_value = "English"
            await repl._cmd_lang("")

            call_kwargs = mock_sel.call_args
            default = call_kwargs[1].get("default") or call_kwargs.kwargs.get("default")
            assert default == "English"
