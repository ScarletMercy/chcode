"""
Tests for chcode/chat.py - ChatREPL class and related functionality
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from unittest.mock import PropertyMock

import pytest
from prompt_toolkit.completion import Completion
from rich.text import Text

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage, RemoveMessage, BaseMessage
from langgraph.types import Command

from chcode.chat import (
    ChatREPL,
    SlashCommandCompleter,
    SLASH_COMMANDS,
    _rich_to_html,
    find_and_slice_from_end,
    _group_messages_by_turn,
    _get_group_display,
    _collect_ids_from_group,
)


# ============================================================================
# Test SlashCommandCompleter
# ============================================================================

class TestSlashCommandCompleter:
    def test_get_completions_with_slash(self):
        completer = SlashCommandCompleter()
        document = Mock()
        document.text_before_cursor = "/mo"

        completions = list(completer.get_completions(document, Mock()))
        assert len(completions) > 0
        assert any(c.text == "/model" for c in completions)

    def test_get_completions_full_match(self):
        completer = SlashCommandCompleter()
        document = Mock()
        document.text_before_cursor = "/new"

        completions = list(completer.get_completions(document, Mock()))
        assert any(c.text == "/new" for c in completions)

    def test_get_completions_no_match(self):
        completer = SlashCommandCompleter()
        document = Mock()
        document.text_before_cursor = "/xyz"

        completions = list(completer.get_completions(document, Mock()))
        assert len(completions) == 0

    def test_get_completions_without_slash(self):
        completer = SlashCommandCompleter()
        document = Mock()
        document.text_before_cursor = "regular text"

        completions = list(completer.get_completions(document, Mock()))
        assert len(completions) == 0

    def test_get_completions_case_insensitive(self):
        completer = SlashCommandCompleter()
        document = Mock()
        document.text_before_cursor = "/Mo"

        completions = list(completer.get_completions(document, Mock()))
        assert any(c.text == "/model" for c in completions)

    def test_get_completions_partial_multiple(self):
        completer = SlashCommandCompleter()
        document = Mock()
        document.text_before_cursor = "/"

        completions = list(completer.get_completions(document, Mock()))
        assert len(completions) == len(SLASH_COMMANDS)


# ============================================================================
# Test ChatREPL.__init__
# ============================================================================

class TestChatREPLInit:
    def test_init_default_values(self):
        repl = ChatREPL()
        assert repl.workplace_path is None
        assert repl.model_config == {}
        assert repl.yolo is False
        assert repl.agent is None
        assert repl.checkpointer is None
        assert repl.session_mgr is None
        assert repl.git_manager is None
        assert repl.git is False
        assert repl._git_cp_count == 0
        assert repl._stop_requested is False
        assert repl._processing is False
        assert repl._prompt_session is None
        assert repl._edit_buffer is None
        assert repl._interrupt_buffer is None
        assert repl._skill_loader is None
        assert repl._context_text == ""
        assert "con" in repl.WINDOWS_RESERVED_NAMES
        assert "nul" in repl.WINDOWS_RESERVED_NAMES


# ============================================================================
# Test ChatREPL.close
# ============================================================================

class TestChatREPLClose:
    @pytest.mark.asyncio
    async def test_close_with_checkpointer(self):
        repl = ChatREPL()
        mock_conn = AsyncMock()
        repl.checkpointer = Mock()
        repl.checkpointer.conn = mock_conn

        await repl.close_checkpointer()

        mock_conn.close.assert_called_once()
        assert repl.checkpointer is None

    @pytest.mark.asyncio
    async def test_close_without_checkpointer(self):
        repl = ChatREPL()
        await repl.close_checkpointer()
        assert repl.checkpointer is None

    @pytest.mark.asyncio
    async def test_close_with_exception(self):
        repl = ChatREPL()
        mock_conn = AsyncMock()
        mock_conn.close.side_effect = Exception("Close error")
        repl.checkpointer = Mock()
        repl.checkpointer.conn = mock_conn

        await repl.close_checkpointer()  # Should not raise
        assert repl.checkpointer is None


# ============================================================================
# Test ChatREPL.initialize
# ============================================================================

class TestChatREPLInitialize:
    @pytest.mark.asyncio
    async def test_initialize_success(self, tmp_path):
        repl = ChatREPL()

        with patch("chcode.chat.ensure_config_dir"):
            with patch("chcode.chat.Path.cwd", return_value=tmp_path):
                with patch("chcode.chat.os.chdir"):
                    with patch("chcode.chat.get_default_model_config", return_value={"model": "gpt-4"}):
                        with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock) as mock_cp:
                            mock_cp.return_value = Mock()
                            with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                                mock_thread.return_value = Mock()
                                with patch.object(repl, "_init_git", new_callable=AsyncMock):
                                    with patch("chcode.config.configure_langsmith", new_callable=AsyncMock, return_value={"tracing": False, "project": "", "api_key": ""}):
                                        result = await repl.initialize()

                                        assert result is True
                                        assert repl.workplace_path == tmp_path
                                        assert repl.model_config == {"model": "gpt-4"}
                                        mock_cp.assert_called_once()
                                        mock_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_first_run(self, tmp_path):
        repl = ChatREPL()

        with patch("chcode.chat.ensure_config_dir"):
            with patch("chcode.chat.Path.cwd", return_value=tmp_path):
                with patch("chcode.chat.os.chdir"):
                    with patch("chcode.chat.get_default_model_config", return_value=None):
                        with patch("chcode.chat.first_run_configure", new_callable=AsyncMock) as mock_frc:
                            mock_frc.return_value = {"model": "gpt-3.5"}
                            with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock) as mock_cp:
                                mock_cp.return_value = Mock()
                                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock):
                                    with patch.object(repl, "_init_git", new_callable=AsyncMock):
                                        with patch("chcode.config.configure_langsmith", new_callable=AsyncMock, return_value={"tracing": False, "project": "", "api_key": ""}):
                                            result = await repl.initialize()

                                            assert result is True
                                            assert repl.model_config == {"model": "gpt-3.5"}
                                            mock_frc.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_first_run_cancelled(self, tmp_path):
        repl = ChatREPL()

        with patch("chcode.chat.ensure_config_dir"):
            with patch("chcode.chat.Path.cwd", return_value=tmp_path):
                with patch("chcode.chat.os.chdir"):
                    with patch("chcode.chat.get_default_model_config", return_value=None):
                        with patch("chcode.chat.first_run_configure", new_callable=AsyncMock) as mock_frc:
                            mock_frc.return_value = None
                            result = await repl.initialize()

                            assert result is False
                            mock_frc.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_creates_directories(self, tmp_path):
        repl = ChatREPL()

        with patch("chcode.chat.ensure_config_dir"):
            with patch("chcode.chat.Path.cwd", return_value=tmp_path):
                with patch("chcode.chat.os.chdir"):
                    with patch("chcode.chat.get_default_model_config", return_value={"model": "gpt-4"}):
                        with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock):
                            with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock):
                                with patch.object(repl, "_init_git", new_callable=AsyncMock):
                                    with patch("chcode.config.configure_langsmith", new_callable=AsyncMock, return_value={"tracing": False, "project": "", "api_key": ""}):
                                        await repl.initialize()

                                        assert (tmp_path / ".chat").exists()
                                        assert (tmp_path / ".chat" / "sessions").exists()
                                        assert (tmp_path / ".chat" / "skills").exists()


# ============================================================================
# Test ChatREPL._init_git
# ============================================================================

class TestChatREPLInitGit:
    @pytest.mark.asyncio
    async def test_init_git_available(self, tmp_path):
        repl = ChatREPL()
        repl.workplace_path = tmp_path

        with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = (True, "ok", "2.0.0")
            with patch("chcode.chat.GitManager") as mock_gm:
                mock_repo = Mock()
                mock_repo.is_repo.return_value = True
                mock_repo.count_checkpoints.return_value = 5
                mock_gm.return_value = mock_repo

                await repl._init_git()

                assert repl.git is True
                assert repl.git_manager == mock_repo
                assert repl._git_cp_count == 5

    @pytest.mark.asyncio
    async def test_init_git_not_available(self, tmp_path):
        repl = ChatREPL()
        repl.workplace_path = tmp_path

        with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = (False, "not found", None)

            await repl._init_git()

            assert repl.git is False
            assert repl.git_manager is None

    @pytest.mark.asyncio
    async def test_init_git_init_repo(self, tmp_path):
        repl = ChatREPL()
        repl.workplace_path = tmp_path

        with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = [(True, "ok", "2.0.0"), None]
            with patch("chcode.chat.GitManager") as mock_gm:
                mock_repo = Mock()
                mock_repo.is_repo.return_value = False
                mock_repo.init = Mock()
                mock_repo.count_checkpoints.return_value = 0
                mock_gm.return_value = mock_repo

                await repl._init_git()

                # init is passed to to_thread, not called directly
                assert mock_thread.call_count == 2
                # Second to_thread call should be for git_manager.init
                second_call_args = mock_thread.call_args_list[1]
                assert second_call_args[0][0] is mock_repo.init


# ============================================================================
# Test ChatREPL._get_input
# ============================================================================

def _make_repl_for_input(tmp_path, **overrides):
    """Create a ChatREPL with a pre-set mock PromptSession to skip real terminal init."""
    repl = ChatREPL()
    repl.workplace_path = tmp_path
    repl.model_config = {"model": "gpt-4"}
    repl.git = False
    repl._prompt_session = Mock()  # skip real PromptSession creation
    for k, v in overrides.items():
        setattr(repl, k, v)
    return repl


class TestChatREPLGetInput:
    @pytest.mark.asyncio
    async def test_get_input_normal(self, tmp_path):
        repl = _make_repl_for_input(tmp_path)

        with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = "hello world"
            result = await repl._get_input()

            assert result == "hello world"

    @pytest.mark.asyncio
    async def test_get_input_with_edit_buffer(self, tmp_path):
        repl = _make_repl_for_input(tmp_path, _edit_buffer="previous input")

        with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            def side_effect(*args, **kwargs):
                assert "default" in kwargs
                assert kwargs["default"] == "previous input"
                return "new input"
            mock_thread.side_effect = side_effect

            result = await repl._get_input()

            assert result == "new input"
            assert repl._edit_buffer is None

    @pytest.mark.asyncio
    async def test_get_input_with_interrupt_buffer(self, tmp_path):
        repl = _make_repl_for_input(tmp_path, _interrupt_buffer="interrupted")

        with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            def side_effect(*args, **kwargs):
                assert kwargs["default"] == "interrupted"
                return "resumed"
            mock_thread.side_effect = side_effect

            result = await repl._get_input()

            assert result == "resumed"
            assert repl._interrupt_buffer is None

    @pytest.mark.asyncio
    async def test_get_input_eof_error(self, tmp_path):
        repl = _make_repl_for_input(tmp_path)

        with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = EOFError()

            result = await repl._get_input()

            assert result is None

    @pytest.mark.asyncio
    async def test_get_input_keyboard_interrupt(self, tmp_path):
        repl = _make_repl_for_input(tmp_path)

        with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = KeyboardInterrupt()

            result = await repl._get_input()

            assert result is None

    @pytest.mark.asyncio
    async def test_get_input_prompt_session_created_once(self, tmp_path):
        repl = _make_repl_for_input(tmp_path)

        with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = "test"

            await repl._get_input()
            session1 = repl._prompt_session

            await repl._get_input()
            session2 = repl._prompt_session

            assert session1 is session2


# ============================================================================
# Test ChatREPL._handle_command
# ============================================================================

class TestChatREPLHandleCommand:
    @pytest.mark.asyncio
    async def test_handle_command_new(self):
        repl = ChatREPL()
        repl.session_mgr = Mock()
        repl.session_mgr.new_session = Mock()

        with patch("chcode.chat.reset_budget_state"):
            with patch("chcode.chat.render_success"):
                with patch.object(repl, "_render_status_bar"):
                    await repl._handle_command("/new")

                    repl.session_mgr.new_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_command_help(self):
        repl = ChatREPL()

        with patch("chcode.chat.console.print") as mock_print:
            await repl._handle_command("/help")
            assert mock_print.called  # Console should have been used

    @pytest.mark.asyncio
    async def test_handle_command_quit(self):
        repl = ChatREPL()

        with pytest.raises(EOFError):
            await repl._handle_command("/quit")

    @pytest.mark.asyncio
    async def test_handle_command_unknown(self):
        repl = ChatREPL()

        with patch("chcode.chat.render_warning") as mock_warn:
            await repl._handle_command("/unknown")

            mock_warn.assert_called_once()
            assert "未知命令" in mock_warn.call_args[0][0]

    @pytest.mark.asyncio
    async def test_handle_command_with_argument(self):
        repl = ChatREPL()

        with patch.object(repl, "_cmd_model", new_callable=AsyncMock) as mock_cmd:
            await repl._handle_command("/model new")

            mock_cmd.assert_called_once_with("new")

    @pytest.mark.asyncio
    async def test_handle_command_case_insensitive(self):
        repl = ChatREPL()

        with patch.object(repl, "_cmd_new", new_callable=AsyncMock) as mock_cmd:
            await repl._handle_command("/NEW")

            mock_cmd.assert_called_once_with("")


# ============================================================================
# Test ChatREPL slash command handlers
# ============================================================================

class TestChatREPLSlashCommands:
    @pytest.mark.asyncio
    async def test_cmd_new(self):
        repl = ChatREPL()
        repl.session_mgr = Mock()
        repl.session_mgr.new_session = Mock()

        with patch("chcode.chat.reset_budget_state"):
            with patch("chcode.chat.render_success") as mock_success:
                with patch.object(repl, "_render_status_bar"):
                    await repl._cmd_new("")

                    mock_success.assert_called_once()
                    assert "新会话" in mock_success.call_args[0][0]

    @pytest.mark.asyncio
    async def test_cmd_model_with_arg_new(self):
        repl = ChatREPL()
        repl.model_config = {}

        with patch("chcode.chat.configure_new_model", new_callable=AsyncMock) as mock_cfg:
            mock_cfg.return_value = {"model": "gpt-4"}
            with patch("chcode.agent_setup.update_summarization_model"):
                with patch.object(repl, "_render_status_bar"):
                    await repl._cmd_model("new")

                    assert repl.model_config == {"model": "gpt-4"}

    @pytest.mark.asyncio
    async def test_cmd_model_with_arg_edit(self):
        repl = ChatREPL()
        repl.model_config = {}

        with patch("chcode.chat.edit_current_model", new_callable=AsyncMock) as mock_edit:
            mock_edit.return_value = {"model": "gpt-3.5"}
            with patch("chcode.agent_setup.update_summarization_model"):
                with patch.object(repl, "_render_status_bar"):
                    await repl._cmd_model("edit")

                    assert repl.model_config == {"model": "gpt-3.5"}

    @pytest.mark.asyncio
    async def test_cmd_model_with_arg_switch(self):
        repl = ChatREPL()
        repl.model_config = {}

        with patch("chcode.chat.switch_model", new_callable=AsyncMock) as mock_sw:
            mock_sw.return_value = {"model": "claude"}
            with patch("chcode.agent_setup.update_summarization_model"):
                with patch.object(repl, "_render_status_bar"):
                    await repl._cmd_model("switch")

                    assert repl.model_config == {"model": "claude"}

    @pytest.mark.asyncio
    async def test_cmd_model_no_arg_select_new(self):
        repl = ChatREPL()
        repl.model_config = {}

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = "新建模型"
            with patch("chcode.chat.configure_new_model", new_callable=AsyncMock) as mock_cfg:
                mock_cfg.return_value = {"model": "new"}
                with patch("chcode.agent_setup.update_summarization_model"):
                    with patch.object(repl, "_render_status_bar"):
                        await repl._cmd_model("")

                        assert repl.model_config == {"model": "new"}

    @pytest.mark.asyncio
    async def test_cmd_model_no_arg_cancel(self):
        repl = ChatREPL()
        repl.model_config = {}

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = None
            await repl._cmd_model("")

            assert repl.model_config == {}

    @pytest.mark.asyncio
    async def test_cmd_model_config_none(self):
        repl = ChatREPL()
        repl.model_config = {}

        with patch("chcode.chat.configure_new_model", new_callable=AsyncMock) as mock_cfg:
            mock_cfg.return_value = None
            with patch("chcode.agent_setup.update_summarization_model"):
                with patch.object(repl, "_render_status_bar"):
                    await repl._cmd_model("new")

                    assert repl.model_config == {}

    @pytest.mark.asyncio
    async def test_cmd_langsmith_enable_no_key(self):
        repl = ChatREPL()

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel, \
             patch("chcode.chat.console.print") as mock_print:
            mock_sel.return_value = "开启追踪"
            await repl._cmd_langsmith("")

            assert repl.langsmith_tracing is False
            printed = [str(c) for c in mock_print.call_args_list]
            assert any("请先设置" in p for p in printed)

    @pytest.mark.asyncio
    async def test_cmd_langsmith_enable(self):
        repl = ChatREPL()
        repl.langsmith_api_key = "lsv2_test_key"

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = "开启追踪"
            with patch("chcode.chat.render_success") as mock_success, \
                 patch("chcode.config._persist_env"), \
                 patch("chcode.config._update_setting"):
                await repl._cmd_langsmith("")

                assert repl.langsmith_tracing is True
                assert os.environ.get("LANGSMITH_TRACING") == "true"
                mock_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_langsmith_disable(self):
        repl = ChatREPL()
        repl.langsmith_tracing = True

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = "关闭追踪"
            with patch("chcode.chat.render_success") as mock_success, \
                 patch("chcode.config._persist_env"), \
                 patch("chcode.config._update_setting"):
                await repl._cmd_langsmith("")

                assert repl.langsmith_tracing is False
                assert os.environ.get("LANGSMITH_TRACING") == "false"
                mock_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_langsmith_cancel(self):
        repl = ChatREPL()
        original = os.environ.get("LANGSMITH_TRACING", "false")

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = None
            await repl._cmd_langsmith("")

            assert os.environ.get("LANGSMITH_TRACING") == original

    @pytest.mark.asyncio
    async def test_cmd_langsmith_edit_project(self):
        repl = ChatREPL()
        repl.langsmith_project = "old-project"

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel, \
             patch("chcode.chat.text", new_callable=AsyncMock) as mock_txt:
            mock_sel.return_value = "修改项目名称"
            mock_txt.return_value = "new-project"
            with patch("chcode.chat.render_success"), \
                 patch("chcode.config._persist_env"), \
                 patch("chcode.config._update_setting"):
                await repl._cmd_langsmith("")
                assert repl.langsmith_project == "new-project"

    @pytest.mark.asyncio
    async def test_cmd_langsmith_edit_key(self):
        repl = ChatREPL()

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel, \
             patch("chcode.chat.text", new_callable=AsyncMock) as mock_txt:
            mock_sel.return_value = "修改 API Key"
            mock_txt.return_value = "lsv2_new_key_here"
            with patch("chcode.chat.render_success"), \
                 patch("chcode.config._persist_env"), \
                 patch("chcode.config._update_setting"):
                await repl._cmd_langsmith("")
                assert repl.langsmith_api_key == "lsv2_new_key_here"

    @pytest.mark.asyncio
    async def test_cmd_langsmith_edit_key_cancel(self):
        repl = ChatREPL()
        repl.langsmith_api_key = "old_key"

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel, \
             patch("chcode.chat.text", new_callable=AsyncMock) as mock_txt:
            mock_sel.return_value = "修改 API Key"
            mock_txt.return_value = ""
            await repl._cmd_langsmith("")
            assert repl.langsmith_api_key == "old_key"

    @pytest.mark.asyncio
    async def test_cmd_langsmith_short_key_masked(self):
        """API Key 短于 10 字符时脱敏显示为 ***"""
        repl = ChatREPL()
        repl.langsmith_api_key = "short"

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel, \
             patch("chcode.chat.text", new_callable=AsyncMock) as mock_txt, \
             patch("chcode.chat.console.print") as mock_print:
            mock_sel.return_value = "修改项目名称"
            mock_txt.return_value = "proj"
            with patch("chcode.chat.render_success"), \
                 patch("chcode.config._persist_env"), \
                 patch("chcode.config._update_setting"):
                await repl._cmd_langsmith("")
                # 验证状态显示时 Key 脱敏为 ***
                printed = [str(c) for c in mock_print.call_args_list]
                assert any("***" in p for p in printed)

    @pytest.mark.asyncio
    async def test_sync_langsmith_env(self):
        repl = ChatREPL()
        repl.langsmith_tracing = True
        repl.langsmith_project = "test-proj"
        repl.langsmith_api_key = "test-key"

        with patch.dict(os.environ, {}, clear=True), \
             patch("chcode.config._persist_env"), \
             patch("chcode.config._update_setting"):
            repl._sync_langsmith_env()
            assert os.environ["LANGSMITH_TRACING"] == "true"
            assert os.environ["LANGSMITH_PROJECT"] == "test-proj"
            assert os.environ["LANGSMITH_API_KEY"] == "test-key"

    @pytest.mark.asyncio
    async def test_sync_langsmith_env_disabled(self):
        repl = ChatREPL()
        repl.langsmith_tracing = False

        with patch.dict(os.environ, {}, clear=True), \
             patch("chcode.config._persist_env"), \
             patch("chcode.config._update_setting"):
            repl._sync_langsmith_env()
            assert os.environ["LANGSMITH_TRACING"] == "false"

        with patch("chcode.chat.console.print") as mock_print:
            with patch("chcode.utils.tools.ALL_TOOLS", []):
                await repl._cmd_tools("")
                assert mock_print.called  # Console should have been used

    @pytest.mark.asyncio
    async def test_cmd_skill_no_session_mgr(self):
        repl = ChatREPL()
        repl.session_mgr = None

        with patch("chcode.chat.render_error") as mock_err:
            await repl._cmd_skill("")

            mock_err.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_skill_with_session_mgr(self):
        repl = ChatREPL()
        repl.session_mgr = Mock()

        with patch("chcode.chat.manage_skills", new_callable=AsyncMock) as mock_ms:
            await repl._cmd_skill("")

            mock_ms.assert_called_once_with(repl.session_mgr)

    @pytest.mark.asyncio
    async def test_cmd_history_no_session(self):
        repl = ChatREPL()
        repl.session_mgr = None
        repl.checkpointer = None
        repl.agent = None

        result = await repl._cmd_history("")
        assert result is None  # Should return early when no session manager

    @pytest.mark.asyncio
    async def test_cmd_history_no_sessions(self):
        repl = ChatREPL()
        repl.session_mgr = Mock()
        repl.checkpointer = Mock()
        repl.agent = Mock()

        with patch.object(repl.session_mgr, "list_sessions", new_callable=AsyncMock) as mock_ls:
            mock_ls.return_value = []
            with patch("chcode.chat.render_warning") as mock_warn:
                await repl._cmd_history("")

                mock_warn.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_mode_common(self):
        repl = ChatREPL()
        repl.yolo = True

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = "Common (手动批准风险操作)"
            with patch("chcode.agent_setup.update_hitl_config"):
                with patch("chcode.chat.render_success") as mock_success:
                    await repl._cmd_mode("")

                    assert repl.yolo is False
                    mock_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_mode_yolo(self):
        repl = ChatREPL()
        repl.yolo = False

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = "Yolo (自动批准所有操作)"
            with patch("chcode.agent_setup.update_hitl_config"):
                with patch("chcode.chat.render_success") as mock_success:
                    await repl._cmd_mode("")

                    assert repl.yolo is True
                    mock_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_help(self):
        repl = ChatREPL()

        with patch("chcode.chat.console.print") as mock_print:
            await repl._cmd_help("")
            assert mock_print.called  # Console should have been used

    @pytest.mark.asyncio
    async def test_cmd_homepage(self):
        repl = ChatREPL()

        with patch("webbrowser.open") as mock_open, \
             patch("chcode.chat.render_success") as mock_success:
            await repl._cmd_homepage("")
            mock_success.assert_called_once_with("正在打开: https://github.com/ScarletMercy/chcode")
            mock_open.assert_called_once_with("https://github.com/ScarletMercy/chcode")

    @pytest.mark.asyncio
    async def test_cmd_quit_raises_eof(self):
        repl = ChatREPL()

        with pytest.raises(EOFError):
            await repl._cmd_quit("")


# ============================================================================
# Test ChatREPL._cmd_git
# ============================================================================

class TestChatREPLCmdGit:
    @pytest.mark.asyncio
    async def test_cmd_git_no_manager_available(self):
        repl = ChatREPL()
        repl.git_manager = None

        async def mock_init_git():
            repl.git_manager = Mock()
            repl.git_manager.is_repo.return_value = True
            repl.git_manager.count_checkpoints.return_value = 0

        with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value=(True, "ok", "2.0.0")), \
             patch.object(repl, "_init_git", side_effect=mock_init_git), \
             patch("chcode.chat.render_success") as mock_success:
            await repl._cmd_git("")
            mock_success.assert_called()

    @pytest.mark.asyncio
    async def test_cmd_git_no_manager_unavailable(self):
        repl = ChatREPL()
        repl.git_manager = None

        with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = (False, "not installed", None)
            with patch("chcode.chat.render_error") as mock_err:
                await repl._cmd_git("")

                mock_err.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_git_is_repo(self):
        repl = ChatREPL()
        mock_gm = Mock()
        mock_gm.is_repo.return_value = True
        mock_gm.count_checkpoints.return_value = 3
        repl.git_manager = mock_gm

        with patch("chcode.chat.render_success") as mock_success:
            await repl._cmd_git("")

            assert repl._git_cp_count == 3
            mock_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_git_not_repo(self):
        repl = ChatREPL()
        mock_gm = Mock()
        mock_gm.is_repo.return_value = False
        repl.git_manager = mock_gm

        with patch("chcode.chat.render_warning") as mock_warn:
            await repl._cmd_git("")

            mock_warn.assert_called_once()


# ============================================================================
# Test ChatREPL._cmd_search
# ============================================================================

class TestChatREPLCmdSearch:
    @pytest.mark.asyncio
    async def test_cmd_search_show_current(self):
        repl = ChatREPL()

        with patch("chcode.config.load_tavily_api_key") as mock_load:
            mock_load.return_value = "tvly-1234567890abcdef"
            with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
                mock_sel.return_value = "返回"
                with patch("chcode.chat.render_info"):
                    await repl._cmd_search("")

                    mock_sel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_search_clear_key(self):
        repl = ChatREPL()

        with patch("chcode.config.load_tavily_api_key", return_value="some-key"):
            with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
                mock_sel.return_value = "清除 API Key"
                with patch("chcode.config.save_tavily_api_key") as mock_save:
                    with patch("chcode.utils.tools.update_tavily_api_key") as mock_upd:
                        with patch("chcode.chat.render_success"):
                            await repl._cmd_search("")

                            mock_save.assert_called_once_with("")
                            mock_upd.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_cmd_search_set_new_key(self):
        repl = ChatREPL()

        with patch("chcode.config.load_tavily_api_key", return_value=None):
            with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
                mock_sel.return_value = "配置 API Key"
                with patch("chcode.chat.text", new_callable=AsyncMock) as mock_text:
                    mock_text.return_value = "new-key-123"
                    with patch("chcode.config.save_tavily_api_key") as mock_save:
                        with patch("chcode.utils.tools.update_tavily_api_key") as mock_upd:
                            with patch("chcode.chat.render_success"):
                                await repl._cmd_search("")

                                mock_save.assert_called_once_with("new-key-123")
                                mock_upd.assert_called_once_with("new-key-123")

    @pytest.mark.asyncio
    async def test_cmd_search_empty_key(self):
        repl = ChatREPL()

        with patch("chcode.config.load_tavily_api_key", return_value=None):
            with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
                mock_sel.return_value = "配置 API Key"
                with patch("chcode.chat.text", new_callable=AsyncMock) as mock_text:
                    mock_text.return_value = ""
                    with patch("chcode.chat.render_warning") as mock_warn:
                        await repl._cmd_search("")
                        mock_warn.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_search_cancel(self):
        repl = ChatREPL()

        with patch("chcode.config.load_tavily_api_key", return_value=None):
            with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
                mock_sel.return_value = None
                with patch("chcode.chat.text") as mock_text:
                    result = await repl._cmd_search("")

                    # Should return None when cancelled
                    assert result is None
                    mock_text.assert_not_called()


# ============================================================================
# Test ChatREPL._cmd_workdir
# ============================================================================

class TestChatREPLCmdWorkdir:
    @pytest.mark.asyncio
    async def test_cmd_workdir_no_selection(self):
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}

        with patch("chcode.chat.load_workplace", return_value=None):
            with patch("chcode.chat.select_or_custom", new_callable=AsyncMock) as mock_sel:
                mock_sel.return_value = ""
                result = await repl._cmd_workdir("")
                assert result is None  # Should return early when cancelled

    @pytest.mark.asyncio
    async def test_cmd_workdir_path_not_exists(self):
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}

        with patch("chcode.chat.load_workplace", return_value=None):
            with patch("chcode.chat.select_or_custom", new_callable=AsyncMock) as mock_sel:
                mock_sel.return_value = "/nonexistent/path"
                with patch("chcode.chat.render_error") as mock_err:
                    await repl._cmd_workdir("")
                    mock_err.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_workdir_success(self, tmp_path):
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}
        repl.yolo = False

        with patch("chcode.chat.load_workplace", return_value=None):
            with patch("chcode.chat.select_or_custom", new_callable=AsyncMock) as mock_sel:
                mock_sel.return_value = str(tmp_path)
                with patch("chcode.chat.os.chdir"):
                    with patch("chcode.chat.save_workplace"):
                        with patch("chcode.chat.SessionManager") as mock_sm:
                            mock_sm_inst = Mock()
                            mock_sm_inst.sessions_dir = tmp_path / ".chat" / "sessions"
                            mock_sm.return_value = mock_sm_inst
                            with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock) as mock_cp:
                                mock_cp.return_value = Mock()
                                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                                    mock_thread.return_value = Mock()
                                    with patch.object(repl, "_init_git", new_callable=AsyncMock):
                                        with patch("chcode.chat.render_success"):
                                            with patch.object(repl, "_render_status_bar"):
                                                await repl._cmd_workdir("")

                                                assert repl.workplace_path == tmp_path
                                                assert repl._skill_loader is None

    @pytest.mark.asyncio
    async def test_cmd_workdir_saved_workplace(self, tmp_path):
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}
        repl.yolo = False

        with patch("chcode.chat.load_workplace", return_value=tmp_path):
            with patch("chcode.chat.select_or_custom", new_callable=AsyncMock) as mock_sel:
                mock_sel.return_value = str(tmp_path)
                with patch("chcode.chat.os.chdir"):
                    with patch("chcode.chat.save_workplace"):
                        with patch("chcode.chat.SessionManager") as mock_sm:
                            mock_sm_inst = Mock()
                            mock_sm_inst.sessions_dir = tmp_path / ".chat" / "sessions"
                            mock_sm.return_value = mock_sm_inst
                            with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock):
                                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock):
                                    with patch.object(repl, "_init_git", new_callable=AsyncMock):
                                        with patch("chcode.chat.render_success") as mock_ok:
                                            with patch.object(repl, "_render_status_bar"):
                                                await repl._cmd_workdir("")
                                                mock_ok.assert_called_once()


# ============================================================================
# Test ChatREPL._cmd_compress
# ============================================================================

class TestChatREPLCmdCompress:
    @pytest.mark.asyncio
    async def test_cmd_compress_no_config(self):
        repl = ChatREPL()
        repl.model_config = {}

        with patch("chcode.chat.render_warning") as mock_warn:
            await repl._cmd_compress("")

            mock_warn.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_compress_user_cancel(self):
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        with patch("chcode.chat.confirm", new_callable=AsyncMock) as mock_conf:
            mock_conf.return_value = False
            result = await repl._cmd_compress("")
            assert result is None  # Should return early when user cancels

    @pytest.mark.asyncio
    async def test_cmd_compress_success(self):
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        state = Mock()
        state.values = {"messages": [HumanMessage("test", id="1")]}
        repl.agent.aget_state.return_value = state
        repl.agent.aupdate_state = AsyncMock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        with patch("chcode.chat.confirm", new_callable=AsyncMock) as mock_conf:
            mock_conf.return_value = True
            with patch("chcode.chat.render_info"):
                with patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm:
                    mock_inst = Mock()
                    mock_inst.invoke = Mock()
                    mock_resp = Mock()
                    mock_resp.content = '{"summary": "compressed"}'
                    mock_resp.content = mock_resp.content.strip()
                    mock_inst.invoke.return_value = mock_resp
                    mock_llm.return_value = mock_inst
                    with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                        mock_thread.return_value = mock_resp
                        with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                            with patch("chcode.chat.render_success"):
                                await repl._cmd_compress("")

                                repl.agent.aupdate_state.assert_called()

    @pytest.mark.asyncio
    async def test_cmd_compress_error(self):
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock(side_effect=Exception("test error"))
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        with patch("chcode.chat.confirm", new_callable=AsyncMock) as mock_conf:
            mock_conf.return_value = True
            with patch("chcode.chat.render_info"):
                with patch("chcode.chat.render_error") as mock_err:
                    await repl._cmd_compress("")

                    mock_err.assert_called_once()

    @pytest.mark.asyncio
    async def test_compress_strips_multimodal_content(self):
        """多模态消息中的 base64 图片/视频块应在压缩时被过滤，只保留文本块。"""
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        # 构造包含多模态 content 的消息
        multimodal_msg = HumanMessage(
            content=[
                {"type": "text", "text": "[image: test.png] 描述这张图"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            id="1",
        )
        text_msg = HumanMessage("普通文本消息", id="2")
        state = Mock()
        state.values = {"messages": [multimodal_msg, text_msg]}
        repl.agent.aget_state.return_value = state
        repl.agent.aupdate_state = AsyncMock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        captured_pre_messages = None

        def capture_invoke(func, messages, *args):
            nonlocal captured_pre_messages
            captured_pre_messages = messages
            mock_resp = Mock()
            mock_resp.content = '{"summary": "ok"}'
            return mock_resp

        with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
            with patch("chcode.chat.render_info"):
                with patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm:
                    mock_inst = Mock()
                    mock_inst.invoke = Mock(side_effect=capture_invoke)
                    mock_llm.return_value = mock_inst
                    with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, side_effect=capture_invoke):
                        with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                            with patch("chcode.chat.render_success"):
                                await repl._cmd_compress("")

        # 验证发给模型的消息中，多模态块已被过滤
        for msg in captured_pre_messages:
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        assert block["type"] not in ("image_url", "video_url"), \
                            f"多模态块未被过滤: {block['type']}"
        # 验证原始消息未被修改（content 仍包含 image_url）
        assert multimodal_msg.content[1]["type"] == "image_url"


# ============================================================================
# Test ChatREPL._cmd_messages
# ============================================================================

class TestChatREPLCmdMessages:
    @pytest.mark.asyncio
    async def test_cmd_messages_no_agent(self):
        repl = ChatREPL()
        repl.agent = None

        with patch("chcode.chat.render_error") as mock_err:
            await repl._cmd_messages("")

            mock_err.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_messages_no_messages(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        state = Mock()
        state.values = {"messages": []}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        with patch("chcode.chat.render_warning") as mock_warn:
            await repl._cmd_messages("")

            mock_warn.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_messages_cancel_action(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        state = Mock()
        msg = HumanMessage("test", id="1")
        state.values = {"messages": [msg]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        with patch("chcode.chat.select", new_callable=AsyncMock, return_value=None):
            result = await repl._cmd_messages("")
            assert result is None  # Should return immediately when cancelled

    @pytest.mark.asyncio
    async def test_cmd_messages_delete_success(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg = HumanMessage("test", id="1")
        state = Mock()
        state.values = {"messages": [msg]}
        repl.agent.aget_state.return_value = state
        repl.agent.aupdate_state = AsyncMock()
        repl.session_mgr = Mock()

        with patch("chcode.chat.select", new_callable=AsyncMock, return_value="删除消息"), \
             patch("chcode.chat.checkbox", new_callable=AsyncMock, return_value=["[1] test"]), \
             patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True), \
             patch.object(repl, "_delete_messages", new_callable=AsyncMock) as mock_del, \
             patch("chcode.chat.render_success"):
            await repl._cmd_messages("")
            mock_del.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_messages_delete_cancel_confirm(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg = HumanMessage("test", id="1")
        state = Mock()
        state.values = {"messages": [msg]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        # select("删除消息") -> checkbox -> confirm(False) -> loops to select(None) -> return
        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["删除消息", None]), \
             patch("chcode.chat.checkbox", new_callable=AsyncMock, return_value=["[1] test"]), \
             patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=False):
            result = await repl._cmd_messages("")
            assert result is None  # Should return via cancel on second select

    @pytest.mark.asyncio
    async def test_cmd_messages_edit_no_human_msg(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg = AIMessage("test", id="1")
        state = Mock()
        state.values = {"messages": [msg]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        # select("编辑消息") -> select("[1] test") -> render_warning -> continue -> select(None) -> return
        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["编辑消息", "[1] test", None]) as mock_sel, \
             patch("chcode.chat.render_warning") as mock_warn:
            await repl._cmd_messages("")
            mock_warn.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_messages_edit_success(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg = HumanMessage("test", id="1")
        state = Mock()
        state.values = {"messages": [msg]}
        repl.agent.aget_state.return_value = state
        repl.agent.aupdate_state = AsyncMock()
        repl.session_mgr = Mock()

        # select("编辑消息") -> select("[1] test") -> confirm(True) -> delete + return
        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["编辑消息", "[1] test"]), \
             patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True), \
             patch("chcode.chat.load_workplace", return_value=None), \
             patch.object(repl, "_delete_messages", new_callable=AsyncMock), \
             patch("chcode.chat.render_success"):
            await repl._cmd_messages("")
            assert repl._edit_buffer == "test"

    @pytest.mark.asyncio
    async def test_cmd_messages_fork_cancel(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg = HumanMessage("test", id="1")
        state = Mock()
        state.values = {"messages": [msg]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        # select("分叉消息") -> select("[1] test") -> confirm(False) -> continue -> select(None) -> return
        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test", None]), \
             patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=False):
            result = await repl._cmd_messages("")
            assert result is None  # Should return via cancel on third select

    @pytest.mark.asyncio
    async def test_cmd_messages_fork_no_path(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg = HumanMessage("test", id="1")
        state = Mock()
        state.values = {"messages": [msg]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        # select("分叉消息") -> select("[1] test") -> confirm(True) -> select_or_custom("") -> continue -> select(None) -> return
        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test", None]), \
             patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True), \
             patch("chcode.chat.load_workplace", return_value=None), \
             patch("chcode.chat.select_or_custom", new_callable=AsyncMock, return_value=""):
            result = await repl._cmd_messages("")
            assert result is None  # Should return via cancel on third select


# ============================================================================
# Test ChatREPL._cmd_history
# ============================================================================

class TestChatREPLCmdHistory:
    @pytest.mark.asyncio
    async def test_cmd_history_load_session(self):
        repl = ChatREPL()
        repl.session_mgr = Mock()
        repl.checkpointer = Mock()
        repl.agent = Mock()

        with patch.object(repl.session_mgr, "list_sessions", new_callable=AsyncMock, return_value=["thread1"]), \
             patch.object(repl.session_mgr, "get_display_names", new_callable=AsyncMock, return_value={"thread1": "Session 1"}), \
             patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["Session 1  (thread1)", "加载此会话"]), \
             patch.object(repl.session_mgr, "set_thread"), \
             patch.object(repl, "_load_conversation", new_callable=AsyncMock), \
             patch.object(repl, "_render_status_bar"):
            await repl._cmd_history("")
            repl.session_mgr.set_thread.assert_called_once_with("thread1")

    @pytest.mark.asyncio
    async def test_cmd_history_rename_session(self):
        repl = ChatREPL()
        repl.session_mgr = Mock()
        repl.checkpointer = Mock()
        repl.agent = Mock()
        repl.session_mgr._load_names = Mock(return_value={"thread1": "Old Name"})

        with patch.object(repl.session_mgr, "list_sessions", new_callable=AsyncMock, return_value=["thread1"]), \
             patch.object(repl.session_mgr, "get_display_names", new_callable=AsyncMock, return_value={"thread1": "Session 1"}), \
             patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["Session 1  (thread1)", "重命名此会话"]), \
             patch("chcode.chat.text", new_callable=AsyncMock, return_value="New Name"), \
             patch.object(repl.session_mgr, "rename_session"), \
             patch("chcode.chat.render_success"):
            await repl._cmd_history("")
            repl.session_mgr.rename_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_history_delete_session(self):
        repl = ChatREPL()
        repl.session_mgr = Mock()
        repl.checkpointer = Mock()
        repl.agent = Mock()
        repl.session_mgr.thread_id = "thread1"

        with patch.object(repl.session_mgr, "list_sessions", new_callable=AsyncMock, return_value=["thread1"]), \
             patch.object(repl.session_mgr, "get_display_names", new_callable=AsyncMock, return_value={"thread1": "Session 1"}), \
             patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["Session 1  (thread1)", "删除此会话"]), \
             patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True), \
             patch.object(repl.session_mgr, "delete_session", new_callable=AsyncMock) as mock_del, \
             patch.object(repl, "_cmd_new", new_callable=AsyncMock), \
             patch("chcode.chat.render_success"):
            await repl._cmd_history("")
            mock_del.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_history_cancel_select(self):
        repl = ChatREPL()
        repl.session_mgr = Mock()
        repl.checkpointer = Mock()
        repl.agent = Mock()

        with patch.object(repl.session_mgr, "list_sessions", new_callable=AsyncMock) as mock_ls:
            mock_ls.return_value = ["thread1"]
            with patch.object(repl.session_mgr, "get_display_names", new_callable=AsyncMock) as mock_gdn:
                mock_gdn.return_value = {"thread1": "Session 1"}
                with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
                    mock_sel.return_value = "返回"
                    result = await repl._cmd_history("")
                    assert result is None  # Should return

    @pytest.mark.asyncio
    async def test_cmd_history_cancel_operation(self):
        repl = ChatREPL()
        repl.session_mgr = Mock()
        repl.checkpointer = Mock()
        repl.agent = Mock()

        with patch.object(repl.session_mgr, "list_sessions", new_callable=AsyncMock) as mock_ls:
            mock_ls.return_value = ["thread1"]
            with patch.object(repl.session_mgr, "get_display_names", new_callable=AsyncMock) as mock_gdn:
                mock_gdn.return_value = {"thread1": "Session 1"}
                with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
                    mock_sel.return_value = "Session 1  (thread1)"
                    with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel2:
                        mock_sel2.return_value = "返回"
                        result = await repl._cmd_history("")
                        assert result is None  # Should return


# ============================================================================
# Test ChatREPL._delete_messages
# ============================================================================

class TestChatREPLDeleteMessages:
    @pytest.mark.asyncio
    async def test_delete_messages(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aupdate_state = AsyncMock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        await repl._delete_messages(["msg1", "msg2"])

        repl.agent.aupdate_state.assert_called_once()
        call_args = repl.agent.aupdate_state.call_args
        assert call_args[0][0] == repl.session_mgr.config
        messages = call_args[0][1]["messages"]
        assert len(messages) == 2
        assert all(isinstance(m, RemoveMessage) for m in messages)

    @pytest.mark.asyncio
    async def test_delete_messages_no_agent(self):
        repl = ChatREPL()
        repl.agent = None
        repl.session_mgr = Mock()

        await repl._delete_messages(["msg1"])
        # Should return early without error when agent is None
        assert repl.agent is None

    @pytest.mark.asyncio
    async def test_delete_messages_no_session_mgr(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = None

        await repl._delete_messages(["msg1"])
        # Should return early without error when session_mgr is None
        assert repl.session_mgr is None


# ============================================================================
# Test ChatREPL._copy_dir
# ============================================================================

class TestChatREPLCopyDir:
    def test_copy_dir_files(self, tmp_path):
        repl = ChatREPL()
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()

        (src / "file1.txt").write_text("content1")
        (src / "file2.txt").write_text("content2")

        repl._copy_dir(src, dst)

        assert (dst / "file1.txt").exists()
        assert (dst / "file2.txt").exists()
        assert (dst / "file1.txt").read_text() == "content1"

    def test_copy_dir_subdirs(self, tmp_path):
        repl = ChatREPL()
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()

        (src / "subdir").mkdir()
        (src / "subdir" / "file.txt").write_text("content")

        repl._copy_dir(src, dst)

        assert (dst / "subdir" / "file.txt").exists()

    def test_copy_dir_skips_dot_files(self, tmp_path):
        repl = ChatREPL()
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()

        (src / ".hidden").write_text("hidden")
        (src / "normal.txt").write_text("normal")

        repl._copy_dir(src, dst)

        assert not (dst / ".hidden").exists()
        assert (dst / "normal.txt").exists()

    def test_copy_dir_skips_windows_reserved(self, tmp_path):
        repl = ChatREPL()
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()

        # Create files with Windows reserved names (case-insensitive check)
        (src / "con").mkdir(exist_ok=True)
        (src / "normal.txt").write_text("normal")

        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            repl._copy_dir(src, dst)
        finally:
            sys.stdout = old_stdout

        assert (dst / "normal.txt").exists()


# ============================================================================
# Test ChatREPL._load_conversation
# ============================================================================

class TestChatREPLLoadConversation:
    @pytest.mark.asyncio
    async def test_load_conversation_no_agent(self):
        repl = ChatREPL()
        repl.agent = None

        await repl._load_conversation()
        # Should return early without error when agent is None
        assert repl.agent is None

    @pytest.mark.asyncio
    async def test_load_conversation_success(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        state = Mock()
        state.values = {"messages": [HumanMessage("test", id="1")]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        with patch("chcode.chat.render_conversation"):
            await repl._load_conversation()

            repl.agent.aget_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_conversation_error(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock(side_effect=Exception("load error"))
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        with patch("chcode.chat.render_error") as mock_err:
            await repl._load_conversation()

            mock_err.assert_called_once()


# ============================================================================
# Test ChatREPL._post_process
# ============================================================================

class TestChatREPLPostProcess:
    @pytest.mark.asyncio
    async def test_post_process_success(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        state = Mock()
        msg = HumanMessage("test", id="1")
        state.values = {"messages": [msg]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.model_config = {"model": "gpt-4"}
        repl.git = True
        repl.git_manager = Mock()
        repl.git_manager.add_commit = Mock(return_value=5)

        with patch("chcode.chat.get_context_window_size", return_value=128000):
            with patch("chcode.chat.get_context_usage_text", return_value="1000/128000"):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                    mock_thread.return_value = 5
                    await repl._post_process()

                    assert repl._context_text == "1000/128000"
                    assert repl._git_cp_count == 5

    @pytest.mark.asyncio
    async def test_post_process_no_git(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        state = Mock()
        state.values = {"messages": []}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.model_config = {"model": "gpt-4"}
        repl.git = False

        with patch("chcode.chat.get_context_window_size", return_value=128000):
            with patch("chcode.chat.get_context_usage_text", return_value="0/128000"):
                await repl._post_process()

                assert repl._context_text == "0/128000"

    @pytest.mark.asyncio
    async def test_post_process_exception(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock(side_effect=Exception("test"))
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        # Should not raise exception
        await repl._post_process()
        # Verify that get_state was called (exception is caught internally)
        repl.agent.aget_state.assert_called_once()


# ============================================================================
# Test ChatREPL._collect_decisions_async
# ============================================================================

class TestChatREPLCollectDecisions:
    @pytest.mark.asyncio
    async def test_collect_decisions_approve(self):
        repl = ChatREPL()
        repl.yolo = False

        interrupt_chunk = {
            "__interrupt__": [
                Mock(value={
                    "action_requests": [{"name": "bash", "args": {"command": "ls"}}],
                    "review_configs": []
                })
            ]
        }

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = "approve (批准)"
            decisions = await repl._collect_decisions_async(interrupt_chunk)

            assert len(decisions) == 1
            assert decisions[0]["type"] == "approve"

    @pytest.mark.asyncio
    async def test_collect_decisions_reject(self):
        repl = ChatREPL()
        repl.yolo = False

        interrupt_chunk = {
            "__interrupt__": [
                Mock(value={
                    "action_requests": [{"name": "bash", "args": {"command": "ls"}}],
                    "review_configs": []
                })
            ]
        }

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = "reject (拒绝)"
            decisions = await repl._collect_decisions_async(interrupt_chunk)

            assert len(decisions) == 1
            assert decisions[0]["type"] == "reject"
            assert "message" in decisions[0]

    @pytest.mark.asyncio
    async def test_collect_decisions_yolo_mode(self):
        repl = ChatREPL()
        repl.yolo = True

        interrupt_chunk = {
            "__interrupt__": [
                Mock(value={
                    "action_requests": [{"name": "bash", "args": {"command": "ls"}}],
                    "review_configs": []
                })
            ]
        }

        decisions = await repl._collect_decisions_async(interrupt_chunk)

        assert len(decisions) == 1
        assert decisions[0]["type"] == "approve"

    @pytest.mark.asyncio
    async def test_collect_decisions_cancel(self):
        repl = ChatREPL()
        repl.yolo = False

        interrupt_chunk = {
            "__interrupt__": [
                Mock(value={
                    "action_requests": [{"name": "bash", "args": {"command": "ls"}}],
                    "review_configs": []
                })
            ]
        }

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = None
            decisions = await repl._collect_decisions_async(interrupt_chunk)

            assert len(decisions) == 1
            assert decisions[0]["type"] == "reject"

    @pytest.mark.asyncio
    async def test_collect_decisions_write_file(self):
        repl = ChatREPL()
        repl.yolo = False

        interrupt_chunk = {
            "__interrupt__": [
                Mock(value={
                    "action_requests": [{
                        "name": "write_file",
                        "args": {"file_path": "/path/to/file", "content": "content"}
                    }],
                    "review_configs": []
                })
            ]
        }

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = "approve (批准)"
            with patch("chcode.chat.render_warning"):
                decisions = await repl._collect_decisions_async(interrupt_chunk)

                assert len(decisions) == 1

    @pytest.mark.asyncio
    async def test_collect_decisions_edit(self, tmp_path):
        repl = ChatREPL()
        repl.yolo = False

        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3")

        interrupt_chunk = {
            "__interrupt__": [
                Mock(value={
                    "action_requests": [{
                        "name": "edit",
                        "args": {
                            "file_path": str(test_file),
                            "old_string": "line2",
                            "new_string": "new_line"
                        }
                    }],
                    "review_configs": []
                })
            ]
        }

        with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = "approve (批准)"
            with patch("chcode.chat.render_warning"):
                with patch("chcode.display.console.print"):
                    decisions = await repl._collect_decisions_async(interrupt_chunk)

                    assert len(decisions) == 1


# ============================================================================
# Test ChatREPL.run
# ============================================================================

class TestChatREPLRun:
    @pytest.mark.asyncio
    async def test_run_normal_input(self):
        repl = ChatREPL()

        with patch("chcode.chat.render_welcome"):
            with patch.object(repl, "_get_input", new_callable=AsyncMock) as mock_input:
                mock_input.return_value = "hello"
                with patch.object(repl, "_process_input", new_callable=AsyncMock) as mock_proc:
                    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
                        # Should run once then loop
                        mock_input.side_effect = ["hello", EOFError()]
                        with patch.object(repl, "_handle_command", new_callable=AsyncMock):
                            try:
                                await repl.run()
                            except EOFError:
                                pass

                            mock_proc.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_run_empty_input(self):
        repl = ChatREPL()

        with patch("chcode.chat.render_welcome"):
            with patch.object(repl, "_get_input", new_callable=AsyncMock) as mock_input:
                mock_input.side_effect = ["", "   ", EOFError()]
                with patch.object(repl, "_process_input", new_callable=AsyncMock) as mock_proc:
                    try:
                        await repl.run()
                    except EOFError:
                        pass

                    # Empty/whitespace input should not trigger processing
                    mock_proc.assert_not_called()
                    # Verify _get_input was called multiple times before EOF
                    assert mock_input.call_count == 3

    @pytest.mark.asyncio
    async def test_run_slash_command(self):
        repl = ChatREPL()

        with patch("chcode.chat.render_welcome"):
            with patch.object(repl, "_get_input", new_callable=AsyncMock) as mock_input:
                mock_input.side_effect = ["/help", EOFError()]
                with patch.object(repl, "_handle_command", new_callable=AsyncMock) as mock_cmd:
                    try:
                        await repl.run()
                    except EOFError:
                        pass

                    mock_cmd.assert_called_once_with("/help")

    @pytest.mark.asyncio
    async def test_run_keyboard_interrupt_not_processing(self):
        repl = ChatREPL()
        repl._processing = False

        with patch("chcode.chat.render_welcome"):
            with patch("chcode.chat.console.print"):
                with patch.object(repl, "_get_input", new_callable=AsyncMock) as mock_input:
                    mock_input.side_effect = [KeyboardInterrupt()]
                    await repl.run()
                    # Should exit loop cleanly when not processing
                    assert repl._processing is False

    @pytest.mark.asyncio
    async def test_run_keyboard_interrupt_while_processing(self):
        repl = ChatREPL()
        repl._processing = True

        with patch("chcode.chat.render_welcome"):
            with patch("chcode.chat.console.print"):
                with patch.object(repl, "_get_input", new_callable=AsyncMock) as mock_input:
                    mock_input.side_effect = [KeyboardInterrupt(), EOFError()]
                    await repl.run()

                    assert repl._stop_requested is True

    @pytest.mark.asyncio
    async def test_run_unexpected_error(self):
        repl = ChatREPL()

        with patch("chcode.chat.render_welcome"):
            with patch.object(repl, "_get_input", new_callable=AsyncMock) as mock_input:
                mock_input.side_effect = [RuntimeError("test error"), EOFError()]
                with patch("chcode.chat.render_error") as mock_err:
                    try:
                        await repl.run()
                    except RuntimeError:
                        pass
                    mock_err.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_langsmith_env_not_mutated_by_guard(self):
        repl = ChatREPL()
        repl.langsmith_tracing = True

        with patch("chcode.chat.render_welcome"):
            with patch.object(repl, "_get_input", new_callable=AsyncMock) as mock_input:
                mock_input.side_effect = ["test", EOFError()]
                with patch.object(repl, "_process_input", new_callable=AsyncMock) as mock_proc:
                    with patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}, clear=False):
                        try:
                            await repl.run()
                        except EOFError:
                            pass

                        assert repl.langsmith_tracing is True


# ============================================================================
# Test ChatREPL._process_input
# ============================================================================

class TestChatREPLProcessInput:
    @pytest.mark.asyncio
    async def test_process_input_simple_stream(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        async def mock_astream(*args, **kwargs):
            yield "messages", [AIMessageChunk(content="Hello")]

        repl.agent.astream = mock_astream
        repl.agent.aget_state = AsyncMock()

        with patch("chcode.chat.render_ai_start"):
            with patch("chcode.chat.render_ai_chunk"):
                with patch("chcode.chat.render_ai_end"):
                    with patch("chcode.chat.asyncio.create_task"):
                        await repl._process_input("test")

                        assert repl._processing is False

    @pytest.mark.asyncio
    async def test_process_input_with_interrupt(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        call_count = [0]

        async def mock_astream(*args, **kwargs):
            if call_count[0] == 0:
                yield "messages", [AIMessageChunk(content="Hello")]
                yield "updates", {"__interrupt__": [Mock(value={"action_requests": [{"name": "bash", "args": {"command": "ls"}}], "review_configs": []})]}
            call_count[0] += 1

        repl.agent.astream = mock_astream
        repl.agent.aupdate_state = AsyncMock()

        with patch("chcode.chat.render_ai_start"):
            with patch("chcode.chat.render_ai_chunk"):
                with patch("chcode.chat.render_ai_end"):
                    with patch("chcode.chat.select", new_callable=AsyncMock) as mock_sel:
                        mock_sel.return_value = "approve (批准)"
                        with patch("chcode.chat.asyncio.create_task"):
                            await repl._process_input("test")
                            mock_sel.assert_called_once()
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        async def mock_astream(*args, **kwargs):
            repl._stop_requested = True
            yield "messages", [AIMessageChunk(content="Hello")]

        repl.agent.astream = mock_astream
        # _handle_cancel 需要 aget_state，当前组有 HumanMessage 无 AIMessage 会回填输入框
        from langchain_core.messages import HumanMessage
        repl.agent.aget_state = AsyncMock(
            return_value=Mock(values={"messages": [HumanMessage("test", id="h1")]})
        )
        repl.agent.aupdate_state = AsyncMock()

        with patch("chcode.chat.console.print"):
            with patch("chcode.chat.asyncio.create_task"):
                await repl._process_input("test")

                assert repl._interrupt_buffer == "test"

    @pytest.mark.asyncio
    async def test_process_input_api_error(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        async def mock_astream(*args, **kwargs):
            import openai
            raise openai.APIError("API error")

        repl.agent.astream = mock_astream
        repl.agent.aupdate_state = AsyncMock()

        with patch("chcode.chat.render_error") as mock_err:
            with patch("chcode.chat.asyncio.create_task"):
                await repl._process_input("test")
                mock_err.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_input_general_error(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        async def mock_astream(*args, **kwargs):
            raise ValueError("General error")

        repl.agent.astream = mock_astream
        repl.agent.aupdate_state = AsyncMock()

        with patch("chcode.chat.render_error") as mock_err:
            with patch("chcode.chat.asyncio.create_task"):
                await repl._process_input("test")
                mock_err.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_input_skill_loader_init(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}
        repl._skill_loader = None

        async def mock_astream(*args, **kwargs):
            yield "messages", [AIMessageChunk(content="Hello")]

        repl.agent.astream = mock_astream
        repl.agent.aget_state = AsyncMock()

        with patch("chcode.chat.SkillLoader"):
            with patch("chcode.chat.render_ai_start"):
                with patch("chcode.chat.render_ai_chunk"):
                    with patch("chcode.chat.render_ai_end"):
                        with patch("chcode.chat.asyncio.create_task"):
                            await repl._process_input("test")

                            assert repl._skill_loader is not None

    @pytest.mark.asyncio
    async def test_process_input_hide_message(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        chunk = AIMessageChunk(content="hidden")
        chunk.additional_kwargs = {"hide": True}

        async def mock_astream(*args, **kwargs):
            yield "messages", [chunk]

        repl.agent.astream = mock_astream
        repl.agent.aget_state = AsyncMock()

        with patch("chcode.chat.asyncio.create_task"):
            await repl._process_input("test")
            assert repl._processing is False  # Processing should complete normally

    @pytest.mark.asyncio
    async def test_process_input_tool_message(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        async def mock_astream(*args, **kwargs):
            yield "messages", [ToolMessage(content="result", tool_call_id="123")]

        repl.agent.astream = mock_astream
        repl.agent.aget_state = AsyncMock()

        with patch("chcode.chat.asyncio.create_task"):
            await repl._process_input("test")
            assert repl._processing is False  # Processing should complete normally

    @pytest.mark.asyncio
    async def test_process_input_reasoning(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        chunk = AIMessageChunk(content="Hello")
        chunk.additional_kwargs = {"reasoning": "Thinking..."}

        async def mock_astream(*args, **kwargs):
            yield "messages", [chunk]

        repl.agent.astream = mock_astream
        repl.agent.aget_state = AsyncMock()

        with patch("chcode.chat.console.print"):
            with patch("chcode.chat.render_ai_start"):
                with patch("chcode.chat.render_ai_chunk"):
                    with patch("chcode.chat.render_ai_end"):
                        with patch("chcode.chat.asyncio.create_task"):
                            await repl._process_input("test")
                            assert repl._processing is False  # Processing should complete

    @pytest.mark.asyncio
    async def test_process_input_model_switch_error(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        from chcode.agent_setup import ModelSwitchError

        # Must be an async generator so `async for` can iterate it
        call_count = 0

        async def mock_astream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ModelSwitchError("Switch needed")
            yield  # make it a generator
            return

        repl.agent.astream = mock_astream

        with patch("chcode.chat.get_fallback_model", return_value={"model": "fallback"}):
            with patch("chcode.chat.advance_fallback"):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                    mock_thread.return_value = Mock()
                    with patch("chcode.display.console.print"):
                        with patch("chcode.chat.asyncio.create_task"):
                            await repl._process_input("test")

                            assert repl.model_config["model"] == "fallback"
                            assert call_count >= 1  # 至少调用了一次，触发切换

    @pytest.mark.asyncio
    async def test_process_input_model_switch_no_fallback(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        from chcode.agent_setup import ModelSwitchError

        async def mock_astream(*args, **kwargs):
            raise ModelSwitchError("Switch needed")

        repl.agent.astream = mock_astream
        # _handle_agent_error 需要 aget_state
        repl.agent.aget_state = AsyncMock(return_value=Mock(values={"messages": []}))

        with patch("chcode.chat.get_fallback_model", return_value=None):
            with patch("chcode.chat.render_error") as mock_err:
                with patch("chcode.chat.asyncio.create_task"):
                    await repl._process_input("test")
                    mock_err.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_input_sets_processing_flag(self):
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        async def mock_astream(*args, **kwargs):
            assert repl._processing is True
            yield "messages", [AIMessageChunk(content="Hello")]

        repl.agent.astream = mock_astream
        repl.agent.aget_state = AsyncMock()

        with patch("chcode.chat.render_ai_start"):
            with patch("chcode.chat.render_ai_chunk"):
                with patch("chcode.chat.render_ai_end"):
                    with patch("chcode.chat.asyncio.create_task"):
                        await repl._process_input("test")

                        assert repl._processing is False


# ============================================================================
# Test _rich_to_html helper
# ============================================================================

class TestRichToHtml:
    def test_plain_text(self):
        assert _rich_to_html("hello world") == "hello world"

    def test_bold_tag(self):
        assert _rich_to_html("[bold]text[/bold]") == "<b>text</b>"

    def test_italic_tag(self):
        assert _rich_to_html("[italic]text[/italic]") == "<i>text</i>"

    def test_red_tag(self):
        result = _rich_to_html("[red]text[/red]")
        assert '<style fg="red">' in result
        assert "text" in result

    def test_green_tag(self):
        result = _rich_to_html("[green]text[/green]")
        assert '<style fg="green">' in result

    def test_multiple_tags(self):
        result = _rich_to_html("[bold][red]text[/red][/bold]")
        assert "<b>" in result
        assert '<style fg="red">' in result

    def test_unmatched_close(self):
        result = _rich_to_html("[bold]text[/bold][red]extra[/red]")
        assert "</style>" in result

    def test_empty_text(self):
        assert _rich_to_html("") == ""

    def test_no_tags(self):
        assert _rich_to_html("plain text") == "plain text"

    def test_unknown_tag(self):
        assert _rich_to_html("[unknown]text[/unknown]") == "text"

    def test_nested_different_tags(self):
        result = _rich_to_html("[bold]hello [italic]world[/italic][/bold]")
        assert "<b>hello <i>world</i></b>" in result


# ============================================================================
# Test find_and_slice_from_end helper
# ============================================================================

class TestFindAndSliceFromEnd:
    def test_found_at_end(self):
        items = [Mock(type="a"), Mock(type="b"), Mock(type="c")]
        result = find_and_slice_from_end(items, "b")
        assert len(result) == 2
        assert result[0].type == "b"

    def test_found_at_start(self):
        items = [Mock(type="x"), Mock(type="y"), Mock(type="z")]
        result = find_and_slice_from_end(items, "x")
        assert len(result) == 3
        assert result[0].type == "x"

    def test_not_found(self):
        items = [Mock(type="a"), Mock(type="b")]
        result = find_and_slice_from_end(items, "c")
        assert result == []

    def test_empty_list(self):
        result = find_and_slice_from_end([], "a")
        assert result == []

    def test_multiple_matches(self):
        items = [Mock(type="a"), Mock(type="b"), Mock(type="a")]
        result = find_and_slice_from_end(items, "a")
        assert len(result) == 1
        assert result[0].type == "a"


# ============================================================================
# Test _group_messages_by_turn helper
# ============================================================================

class TestGroupMessagesByTurn:
    def test_single_human_message(self):
        msgs = [Mock(type="human", content="hi")]
        result = _group_messages_by_turn(msgs)
        assert len(result) == 1
        assert result[0][0].type == "human"

    def test_human_ai_pair(self):
        msgs = [
            Mock(type="human", content="q1"),
            Mock(type="ai", content="a1"),
        ]
        result = _group_messages_by_turn(msgs)
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_two_turns(self):
        msgs = [
            Mock(type="human", content="q1"),
            Mock(type="ai", content="a1"),
            Mock(type="human", content="q2"),
            Mock(type="ai", content="a2"),
        ]
        result = _group_messages_by_turn(msgs)
        assert len(result) == 2
        assert result[0][0].content == "q1"
        assert result[1][0].content == "q2"

    def test_empty_list_v2(self):
        result = _group_messages_by_turn([])
        assert result == []

    def test_multiple_ai_messages(self):
        msgs = [
            Mock(type="human", content="q"),
            Mock(type="ai", content="a1"),
            Mock(type="ai", content="a2"),
        ]
        result = _group_messages_by_turn(msgs)
        assert len(result) == 1
        assert len(result[0]) == 3


# ============================================================================
# Test _get_group_display helper
# ============================================================================

class TestGetGroupDisplay:
    def test_with_human_message(self):
        group = [Mock(type="human", content="Hello world")]
        result = _get_group_display(group)
        assert "Hello world" in result

    def test_long_content_truncated(self):
        long_text = "x" * 100
        group = [Mock(type="human", content=long_text)]
        result = _get_group_display(group)
        assert "..." in result
        assert len(result) <= 70

    def test_newline_replaced(self):
        group = [Mock(type="human", content="line1\nline2\nline3")]
        result = _get_group_display(group)
        assert "\n" not in result

    def test_no_human_message(self):
        group = [Mock(type="ai", content="response")]
        result = _get_group_display(group)
        assert result == "(空消息组)"

    def test_empty_group(self):
        result = _get_group_display([])
        assert result == "(空消息组)"

    def test_human_in_middle_of_group(self):
        group = [
            Mock(type="ai", content="a1"),
            Mock(type="human", content="question"),
            Mock(type="ai", content="a2"),
        ]
        result = _get_group_display(group)
        assert "question" in result


# ============================================================================
# Test _collect_ids_from_group helper
# ============================================================================

class TestCollectIdsFromGroup:
    def test_collect_from_first_group(self):
        groups = [
            [Mock(id="a1"), Mock(id="a2")],
            [Mock(id="b1"), Mock(id="b2")],
        ]
        no_need, all_ids = _collect_ids_from_group(0, groups)
        assert "a1" in no_need
        assert "a2" in no_need
        assert "b1" in no_need
        assert "b2" in no_need
        assert len(all_ids) == 4

    def test_collect_from_middle_group(self):
        groups = [
            [Mock(id="a1")],
            [Mock(id="b1"), Mock(id="b2")],
            [Mock(id="c1")],
        ]
        no_need, all_ids = _collect_ids_from_group(1, groups)
        assert "a1" not in no_need
        assert "b1" in no_need
        assert "b2" in no_need
        assert "c1" in no_need

    def test_collect_from_last_group(self):
        groups = [
            [Mock(id="a1")],
            [Mock(id="b1")],
            [Mock(id="c1")],
        ]
        no_need, all_ids = _collect_ids_from_group(2, groups)
        assert "a1" not in no_need
        assert "b1" not in no_need
        assert "c1" in no_need

    def test_all_ids_collected(self):
        groups = [
            [Mock(id="x1"), Mock(id="x2")],
            [Mock(id="y1"), Mock(id="y2")],
        ]
        no_need, all_ids = _collect_ids_from_group(0, groups)
        assert len(all_ids) == 4
        assert "x1" in all_ids
        assert "x2" in all_ids
        assert "y1" in all_ids
        assert "y2" in all_ids
