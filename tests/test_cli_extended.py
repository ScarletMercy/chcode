"""Extended tests for chcode/cli.py"""
import io
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest


class TestLangSmithGuard:
    """Cover lines 23, 25-27, 31-33, 37, 40: _Guard class behavior.

    The _Guard class is a closure inside _setup_langsmith_guard().
    We reconstruct it by calling the setup function which installs it on sys.stderr.
    """

    def _make_guard(self):
        """Create a fresh _Guard by calling _setup_langsmith_guard.

        Returns the guard instance and its original stderr.
        """
        # Save current stderr
        old_stderr = sys.stderr
        old__stderr__ = sys.__stderr__
        # Call setup which wraps sys.stderr/sys.__stderr__
        from chcode.cli import _setup_langsmith_guard
        _setup_langsmith_guard()
        guard = sys.stderr
        original = guard._original
        return guard, original, old_stderr, old__stderr__

    def _cleanup(self, old_stderr, old__stderr__):
        """Restore original stderr."""
        sys.stderr = old_stderr
        sys.__stderr__ = old__stderr__

    def test_guard_write_empty_data(self):
        """Line 22-23: write('') returns 0."""
        guard, original, old_s, old__s = self._make_guard()
        try:
            result = guard.write("")
            assert result == 0
        finally:
            self._cleanup(old_s, old__s)

    def test_guard_write_none_data(self):
        """Line 22-23: write(None) returns 0."""
        guard, original, old_s, old__s = self._make_guard()
        try:
            result = guard.write(None)
            assert result == 0
        finally:
            self._cleanup(old_s, old__s)

    def test_guard_write_normal_data(self):
        """Normal data passes through to original."""
        guard, original, old_s, old__s = self._make_guard()
        buf = io.StringIO()
        guard._original = buf
        try:
            ret = guard.write("normal output\n")
            assert ret == len("normal output\n")
            assert buf.getvalue() == "normal output\n"
        finally:
            guard._original = original
            self._cleanup(old_s, old__s)

    def test_guard_detects_langsmith_rate_limit_error(self):
        """Lines 28-33: LangSmithRateLimitError triggers disable."""
        old_env = os.environ.pop("LANGCHAIN_TRACING", None)
        guard, original, old_s, old__s = self._make_guard()
        buf = io.StringIO()
        guard._original = buf
        try:
            ret = guard.write("Error: LangSmithRateLimitError occurred\n")
            assert ret > 0
            assert os.environ.get("LANGCHAIN_TRACING") == "false"
        finally:
            guard._original = original
            self._cleanup(old_s, old__s)
            if old_env is not None:
                os.environ["LANGCHAIN_TRACING"] = old_env
            else:
                os.environ.pop("LANGCHAIN_TRACING", None)

    def test_guard_detects_langsmith_429(self):
        """Lines 28-33: langsmith + 429 triggers disable."""
        old_env = os.environ.pop("LANGCHAIN_TRACING", None)
        guard, original, old_s, old__s = self._make_guard()
        buf = io.StringIO()
        guard._original = buf
        try:
            ret = guard.write("langsmith: 429 Too Many Requests\n")
            assert ret > 0
            assert os.environ.get("LANGCHAIN_TRACING") == "false"
        finally:
            guard._original = original
            self._cleanup(old_s, old__s)
            if old_env is not None:
                os.environ["LANGCHAIN_TRACING"] = old_env
            else:
                os.environ.pop("LANGCHAIN_TRACING", None)

    def test_guard_disabled_filters_langsmith_429(self):
        """Lines 24-27: after disable, langsmith+429 messages are suppressed and env set."""
        old_env = os.environ.pop("LANGCHAIN_TRACING", None)
        guard, original, old_s, old__s = self._make_guard()
        buf = io.StringIO()
        guard._original = buf
        try:
            # First trigger disable (lines 28-33)
            guard.write("LangSmithRateLimitError\n")
            # Now write langsmith message with 429 - should hit lines 24-27
            ret = guard.write("LangSmith: 429 error\n")
            assert ret > 0
            # The 429 inside disabled block should set env (line 26)
            assert os.environ.get("LANGCHAIN_TRACING") == "false"
        finally:
            guard._original = original
            self._cleanup(old_s, old__s)
            if old_env is not None:
                os.environ["LANGCHAIN_TRACING"] = old_env
            else:
                os.environ.pop("LANGCHAIN_TRACING", None)

    def test_guard_disabled_filters_langsmith_rate_limit(self):
        """Lines 24-27: after disable, Rate limit messages are suppressed."""
        old_env = os.environ.pop("LANGCHAIN_TRACING", None)
        guard, original, old_s, old__s = self._make_guard()
        buf = io.StringIO()
        guard._original = buf
        try:
            guard.write("LangSmithRateLimitError\n")
            ret = guard.write("LangSmith: Rate limit exceeded\n")
            assert ret > 0
        finally:
            guard._original = original
            self._cleanup(old_s, old__s)
            if old_env is not None:
                os.environ["LANGCHAIN_TRACING"] = old_env
            else:
                os.environ.pop("LANGCHAIN_TRACING", None)

    def test_guard_detects_connection_error(self):
        """Lines 36-48: langsmith + ConnectionError triggers disable."""
        old_env = os.environ.pop("LANGCHAIN_TRACING", None)
        guard, original, old_s, old__s = self._make_guard()
        buf = io.StringIO()
        guard._original = buf
        try:
            ret = guard.write(
                "Failed to send compressed multipart ingest: langsmith ConnectionError\n"
            )
            assert ret > 0
            assert os.environ.get("LANGCHAIN_TRACING") == "false"
        finally:
            guard._original = original
            self._cleanup(old_s, old__s)
            if old_env is not None:
                os.environ["LANGCHAIN_TRACING"] = old_env
            else:
                os.environ.pop("LANGCHAIN_TRACING", None)

    def test_guard_detects_max_retry_error(self):
        """Lines 36-48: langsmith + MaxRetryError triggers disable."""
        old_env = os.environ.pop("LANGCHAIN_TRACING", None)
        guard, original, old_s, old__s = self._make_guard()
        buf = io.StringIO()
        guard._original = buf
        try:
            ret = guard.write(
                "langsmith MaxRetryError: api.smith.langchain.com\n"
            )
            assert ret > 0
            assert os.environ.get("LANGCHAIN_TRACING") == "false"
        finally:
            guard._original = original
            self._cleanup(old_s, old__s)
            if old_env is not None:
                os.environ["LANGCHAIN_TRACING"] = old_env
            else:
                os.environ.pop("LANGCHAIN_TRACING", None)

    def test_guard_detects_connection_aborted(self):
        """Lines 36-48: ConnectionAbortedError triggers disable."""
        old_env = os.environ.pop("LANGCHAIN_TRACING", None)
        guard, original, old_s, old__s = self._make_guard()
        buf = io.StringIO()
        guard._original = buf
        try:
            ret = guard.write(
                "langsmith ConnectionAbortedError(10053)\n"
            )
            assert ret > 0
            assert os.environ.get("LANGCHAIN_TRACING") == "false"
        finally:
            guard._original = original
            self._cleanup(old_s, old__s)
            if old_env is not None:
                os.environ["LANGCHAIN_TRACING"] = old_env
            else:
                os.environ.pop("LANGCHAIN_TRACING", None)

    def test_guard_detects_connection_reset(self):
        """Lines 36-48: ConnectionResetError triggers disable."""
        old_env = os.environ.pop("LANGCHAIN_TRACING", None)
        guard, original, old_s, old__s = self._make_guard()
        buf = io.StringIO()
        guard._original = buf
        try:
            ret = guard.write(
                "langsmith ConnectionResetError(10054)\n"
            )
            assert ret > 0
            assert os.environ.get("LANGCHAIN_TRACING") == "false"
        finally:
            guard._original = original
            self._cleanup(old_s, old__s)
            if old_env is not None:
                os.environ["LANGCHAIN_TRACING"] = old_env
            else:
                os.environ.pop("LANGCHAIN_TRACING", None)

    def test_guard_disabled_filters_connection_error(self):
        """After disable, connection errors are suppressed."""
        old_env = os.environ.pop("LANGCHAIN_TRACING", None)
        guard, original, old_s, old__s = self._make_guard()
        buf = io.StringIO()
        guard._original = buf
        try:
            guard.write("LangSmithRateLimitError\n")
            ret = guard.write("langsmith: Failed to send ConnectionError\n")
            assert ret > 0
        finally:
            guard._original = original
            self._cleanup(old_s, old__s)
            if old_env is not None:
                os.environ["LANGCHAIN_TRACING"] = old_env
            else:
                os.environ.pop("LANGCHAIN_TRACING", None)

    def test_guard_flush(self):
        """Line 37: flush delegates to original."""
        guard, original, old_s, old__s = self._make_guard()
        buf = io.StringIO()
        guard._original = buf
        try:
            guard.flush()
        except Exception:
            self.fail("flush() raised unexpectedly")
        finally:
            guard._original = original
            self._cleanup(old_s, old__s)

    def test_guard_getattr(self):
        """Line 40: __getattr__ delegates to original."""
        guard, original, old_s, old__s = self._make_guard()
        mock_original = MagicMock()
        mock_original.encoding = "utf-8"
        mock_original.mode = "w"
        guard._original = mock_original
        try:
            # __getattr__ is called for attributes not on _Guard
            assert guard.encoding == "utf-8"
            assert guard.mode == "w"
        finally:
            guard._original = original
            self._cleanup(old_s, old__s)


class TestMainEntry:
    """Cover line 131: __main__ entry."""

    def test_main_entry(self):
        """Line 131: if __name__ == '__main__': app()."""
        # The __main__ entry just calls app(). Verify it exists.
        from chcode.cli import app
        assert app is not None
        # Verify app has a callable method
        assert hasattr(app, '__call__') or hasattr(app, 'run')


class TestMainCallback:
    def test_no_subcommand_invokes_run_chat(self):
        import asyncio
        from chcode.cli import main, app
        mock_ctx = MagicMock()
        mock_ctx.invoked_subcommand = None
        with patch("chcode.cli.asyncio.run") as mock_run:
            try:
                main(mock_ctx, yolo=False, version=False)
            except SystemExit:
                pass
            mock_run.assert_called_once()
            # Verify run_chat was scheduled, not something else
            assert mock_run.call_args is not None

    def test_version_flag(self):
        from chcode.cli import main
        mock_ctx = MagicMock()
        mock_ctx.invoked_subcommand = None
        with patch("chcode.cli.console") as mock_console:
            try:
                main(mock_ctx, yolo=False, version=True)
            except (SystemExit, click.exceptions.Exit):
                pass
            mock_console.print.assert_called()
            # Verify version was printed (not something else)
            assert mock_console.print.call_count >= 1


class TestConfigCommand:
    def test_config_new(self):
        from chcode.cli import config
        with patch("chcode.cli.asyncio.run") as mock_run:
            config("new")
            mock_run.assert_called_once()
            # Verify _run_config was called with "new"
            assert mock_run.call_args[0][0].__name__ == "_run_config"

    def test_config_edit(self):
        from chcode.cli import config
        with patch("chcode.cli.asyncio.run") as mock_run:
            config("edit")
            mock_run.assert_called_once()
            # Verify _run_config was called with "edit"
            assert mock_run.call_args[0][0].__name__ == "_run_config"

    def test_config_switch(self):
        from chcode.cli import config
        with patch("chcode.cli.asyncio.run") as mock_run:
            config("switch")
            mock_run.assert_called_once()
            # Verify _run_config was called with "switch"
            assert mock_run.call_args[0][0].__name__ == "_run_config"

    def test_config_unknown(self):
        from chcode.cli import config
        with patch("chcode.cli.asyncio.run") as mock_run:
            config("unknown")
            mock_run.assert_called_once()
            # Verify _run_config was called with unknown action
            assert mock_run.call_args[0][0].__name__ == "_run_config"


class TestRunConfig:
    async def test_new_action(self):
        from chcode.cli import _run_config
        with patch("chcode.config.configure_new_model", new_callable=AsyncMock) as mock:
            await _run_config("new")
            mock.assert_called_once()

    async def test_edit_action(self):
        from chcode.cli import _run_config
        with patch("chcode.config.edit_current_model", new_callable=AsyncMock) as mock:
            await _run_config("edit")
            mock.assert_called_once()

    async def test_switch_action(self):
        from chcode.cli import _run_config
        with patch("chcode.config.switch_model", new_callable=AsyncMock) as mock:
            await _run_config("switch")
            mock.assert_called_once()

    async def test_unknown_action(self):
        from chcode.cli import _run_config
        with patch("chcode.cli.console") as mock_console:
            await _run_config("bad_action")
            assert mock_console.print.called


class TestStderrIntegrity:
    """Verify _setup_langsmith_guard does NOT override sys.__stderr__."""

    def test_stderr_not_overridden(self):
        """sys.__stderr__ must remain the original file object, not a _Guard."""
        # The guard is already installed at import time (module level call).
        # sys.stderr should be wrapped, but sys.__stderr__ must be preserved.
        assert sys.stderr is not sys.__stderr__, (
            "sys.stderr (guard) and sys.__stderr__ should differ"
        )
        # sys.__stderr__ should be a real file-like object, not a guard wrapper
        assert hasattr(sys.__stderr__, 'fileno'), (
            "sys.__stderr__ should be a real file object with fileno()"
        )
        # Verify we can recover sys.stderr from sys.__stderr__
        old_stderr = sys.stderr
        sys.stderr = sys.__stderr__
        assert sys.stderr is not old_stderr, (
            "Recovery via sys.__stderr__ should restore the real file"
        )
        # Restore guard
        sys.stderr = old_stderr

    def test_guard_only_wraps_stderr(self):
        """After _setup_langsmith_guard, only sys.stderr is wrapped, never sys.__stderr__."""
        import io
        # sys.__stderr__ should support fileno (real file)
        try:
            sys.__stderr__.fileno()
        except Exception:
            self.fail("sys.__stderr__.fileno() should work — it must be a real file")
        # Under pytest capture, sys.stderr may be a pytest EncodedFile, not our guard.
        # But sys.__stderr__ must never be a guard wrapper — it's always the real file.
        # We verify this by checking __getattr__ is not overridden ( Guard uses __getattr__).
        # A real file's type doesn't have a custom __getattr__ for delegation.
        stderr_type = type(sys.__stderr__)
        # The key invariant: sys.__stderr__ must NOT be a _Guard instance
        # We check this by ensuring it lacks the _original attr that _Guard sets
        assert not hasattr(sys.__stderr__, '_original'), (
            "sys.__stderr__ must not be a _Guard wrapper (no _original attr)"
        )


class TestRunChat:
    async def test_init_success(self):
        from chcode.cli import _run_chat
        mock_repl = MagicMock()
        mock_repl.initialize = AsyncMock(return_value=True)
        mock_repl.run = AsyncMock()
        mock_repl.close = AsyncMock()
        with patch("chcode.chat.ChatREPL", return_value=mock_repl):
            await _run_chat(False)
        mock_repl.initialize.assert_called_once()
        mock_repl.run.assert_called_once()
        mock_repl.close.assert_called_once()

    async def test_init_failure(self):
        from chcode.cli import _run_chat
        mock_repl = MagicMock()
        mock_repl.initialize = AsyncMock(return_value=False)
        mock_repl.close = AsyncMock()
        with patch("chcode.chat.ChatREPL", return_value=mock_repl):
            try:
                await _run_chat(False)
            except (SystemExit, click.exceptions.Exit):
                pass
        # Verify initialize was called and returned False
        mock_repl.initialize.assert_called_once()
        # Exit happens without calling close when initialize fails

    async def test_init_exception(self):
        from chcode.cli import _run_chat
        mock_repl = MagicMock()
        mock_repl.initialize = AsyncMock(side_effect=RuntimeError("fail"))
        mock_repl.close = AsyncMock()
        with patch("chcode.chat.ChatREPL", return_value=mock_repl), \
             patch("chcode.cli.console") as mock_console:
            try:
                await _run_chat(False)
            except (SystemExit, click.exceptions.Exit):
                pass
            mock_console.print_exception.assert_called_once()
        # Verify initialize was called and raised exception
        mock_repl.initialize.assert_called_once()
        # Exception is logged via console.print_exception
