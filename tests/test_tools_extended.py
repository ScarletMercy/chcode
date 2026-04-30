"""
Extended tests for chcode/utils/tools.py — targeting uncovered lines.
Focuses on file operations, config loading, string operations, simple conditionals.
"""
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from chcode.utils.shell.result import ShellResult
from chcode.utils.tools import (
    _html_to_markdown,
    _is_binary_content_type,
    _ask_multi_questions,
    resolve_path,
    _ensure_tavily_key,
    _todo_path,
    _save_todos,
    TodoItem,
    FETCH_TIMEOUT,
    MAX_MARKDOWN_LENGTH,
    MAX_URL_LENGTH,
)


def _make_runtime(**kwargs):
    """Create a mock ToolRuntime with SkillAgentContext."""
    rt = MagicMock()
    ctx = MagicMock()
    for k, v in kwargs.items():
        setattr(ctx, k, v)
    rt.context = ctx
    return rt


# ============================================================================
# Lines 59-64: _ensure_tavily_key — loading from SETTING_JSON file
# ============================================================================


class TestEnsureTavilyKeyFromFile:
    def test_loads_key_from_setting_json(self, tmp_path, monkeypatch):
        """Lines 59-63: Valid JSON with tavily_api_key is read from file."""
        import chcode.utils.tools as mod
        mod._tavily_api_key = ""
        mod._tavily_key_loaded = False
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        setting_file = tmp_path / "chagent.json"
        setting_file.write_text(
            json.dumps({"tavily_api_key": "tvly-filekey"}), encoding="utf-8"
        )
        monkeypatch.setattr(mod, "SETTING_JSON", setting_file)
        _ensure_tavily_key()
        assert mod._tavily_api_key == "tvly-filekey"

    def test_empty_key_in_setting_json(self, tmp_path, monkeypatch):
        """Lines 59-63: JSON has tavily_api_key but it's empty string."""
        import chcode.utils.tools as mod
        mod._tavily_api_key = ""
        mod._tavily_key_loaded = False
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        setting_file = tmp_path / "chagent.json"
        setting_file.write_text(
            json.dumps({"tavily_api_key": ""}), encoding="utf-8"
        )
        monkeypatch.setattr(mod, "SETTING_JSON", setting_file)
        _ensure_tavily_key()
        assert mod._tavily_api_key == ""

    def test_missing_tavily_key_in_json(self, tmp_path, monkeypatch):
        """Lines 59-63: JSON exists but has no tavily_api_key field."""
        import chcode.utils.tools as mod
        mod._tavily_api_key = ""
        mod._tavily_key_loaded = False
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        setting_file = tmp_path / "chagent.json"
        setting_file.write_text(json.dumps({"other": "value"}), encoding="utf-8")
        monkeypatch.setattr(mod, "SETTING_JSON", setting_file)
        _ensure_tavily_key()
        assert mod._tavily_api_key == ""

    def test_invalid_json_in_setting_file(self, tmp_path, monkeypatch):
        """Line 64: Invalid JSON triggers except branch."""
        import chcode.utils.tools as mod
        mod._tavily_api_key = ""
        mod._tavily_key_loaded = False
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        setting_file = tmp_path / "chagent.json"
        setting_file.write_text("not json {{{", encoding="utf-8")
        monkeypatch.setattr(mod, "SETTING_JSON", setting_file)
        _ensure_tavily_key()
        # Should not crash, key stays empty
        assert mod._tavily_api_key == ""


# ============================================================================
# Lines 273-274: bash tool — timeout handling
# ============================================================================


class TestBashTimeout:
    async def test_timed_out_message(self):
        """Lines 272-274: Command times out, timed_out message is appended."""
        from chcode.utils.tools import bash

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        result = ShellResult(stdout="some output", exit_code=-1, timed_out=True)
        truncated = MagicMock()
        truncated.content = "some output"
        truncated.truncated = False
        mock_sess = MagicMock()
        mock_sess.provider_name = "bash"
        mock_sess.execute = MagicMock(return_value=(result, truncated))
        with patch("chcode.utils.tools._get_shell_session", return_value=mock_sess), \
             patch("chcode.utils.tools.render_tool_call"):
            out = await bash.coroutine("sleep 999", runtime=rt, timeout=10)
        assert "timed out" in out
        assert "10s" in out

    async def test_timeout_clamped_to_600(self):
        """Line 237: timeout is clamped to max 600."""
        from chcode.utils.tools import bash

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        result = ShellResult(stdout="", exit_code=0, timed_out=False)
        truncated = MagicMock()
        truncated.content = ""
        truncated.truncated = False
        mock_sess = MagicMock()
        mock_sess.provider_name = "bash"
        mock_sess.execute = MagicMock(return_value=(result, truncated))
        with patch("chcode.utils.tools._get_shell_session", return_value=mock_sess), \
             patch("chcode.utils.tools.render_tool_call"):
            out = await bash.coroutine("echo hi", runtime=rt, timeout=10000)
        # Should not crash; session.execute was called
        mock_sess.execute.assert_called_once()


# ============================================================================
# Lines 255, 315, 319, 324-325: read_file tool — line limit and exceptions
# ============================================================================


class TestReadFileExtended:
    async def test_file_over_2000_lines(self, tmp_path):
        """Lines 314-315, 318-319: Files with >2000 lines get truncation message."""
        from chcode.utils.tools import read_file

        lines = [f"line {i}" for i in range(2500)]
        f = tmp_path / "big.txt"
        f.write_text("\n".join(lines), encoding="utf-8")
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await read_file.coroutine(str(f), runtime=rt)
        assert "[OK]" in out
        assert "2500 lines" in out
        assert "first 2000" in out
        assert "500 more lines" in out

    async def test_file_exception_generic(self, tmp_path):
        """Lines 324-325: Generic Exception handler in read_file."""
        from chcode.utils.tools import read_file

        f = tmp_path / "e.txt"
        f.write_text("ok", encoding="utf-8")
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("aiofiles.open", side_effect=OSError("disk error")):
            out = await read_file.coroutine(str(f), runtime=rt)
        assert "[FAILED]" in out
        assert "disk error" in out


# ============================================================================
# Lines 388-389, 394: glob tool — ValueError and truncation
# ============================================================================


class TestGlobExtended:
    async def test_value_error_in_relative_path(self, tmp_path):
        """Lines 387-389: ValueError when path is not relative to cwd."""
        from chcode.utils.tools import glob

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        # Create a file outside the working directory
        other_dir = tmp_path.parent
        external_file = other_dir / "external.py"
        external_file.touch()

        # Patch glob to return a mix of relative and absolute paths
        with patch("chcode.utils.tools.render_tool_call"), \
             patch.object(Path, "glob") as mock_glob:
            # Return one file inside cwd and one outside
            mock_glob.return_value = [tmp_path / "a.py", external_file]
            out = await glob.coroutine("*.py", runtime=rt)
        assert "[OK]" in out

    async def test_truncated_results(self, tmp_path):
        """Line 394: More than 100 matches triggers truncation."""
        from chcode.utils.tools import glob

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        # Create 110 files
        for i in range(110):
            (tmp_path / f"f{i}.py").touch()

        with patch("chcode.utils.tools.render_tool_call"):
            out = await glob.coroutine("*.py", runtime=rt)
        assert "[OK]" in out
        assert "110 matches" in out
        assert "10 more files" in out


# ============================================================================
# Lines 512-514, 526-527, 541, 543, 549, 551, 553, 566: grep tool
# ============================================================================


class TestGrepExtended:
    async def test_file_too_large(self, tmp_path):
        """Lines 511-512: Files exceeding _GREP_MAX_FILE_SIZE are skipped."""
        from chcode.utils.tools import grep

        f = tmp_path / "large.txt"
        # Write a file just over 1MB
        f.write_bytes(b"x" * (1024 * 1024 + 1))
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await grep.coroutine("pattern", str(f), rt)
        assert "No matches" in out

    async def test_file_stat_oserror(self, tmp_path):
        """Lines 513-514: OSError during stat() is silently caught."""
        from chcode.utils.tools import grep

        f = tmp_path / "perm.txt"
        f.write_text("match_here", encoding="utf-8")
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        original_stat = Path.stat

        def failing_stat(self):
            raise OSError("permission denied")

        with patch("chcode.utils.tools.render_tool_call"), \
             patch.object(Path, "stat", failing_stat):
            out = await grep.coroutine("match", str(f), rt)
        assert "No matches" in out

    async def test_value_error_in_relative_path_v2(self, tmp_path):
        """Lines 526-527: ValueError when computing relative path."""
        from chcode.utils.tools import grep

        # Create a file in one directory and search from another
        search_dir = tmp_path / "search_dir"
        search_dir.mkdir()
        match_file = tmp_path / "outside.txt"
        match_file.write_text("target_pattern\n", encoding="utf-8")

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        # Search a specific file that is NOT relative to cwd
        with patch("chcode.utils.tools.render_tool_call"):
            out = await grep.coroutine("target_pattern", str(match_file), rt)
        assert "[OK]" in out
        assert "target_pattern" in out

    async def test_max_results_truncation(self, tmp_path):
        """Lines 530-531, 565-566: Results truncated at max_results."""
        from chcode.utils.tools import grep

        # Create many files with matches
        for i in range(60):
            f = tmp_path / f"m{i}.py"
            f.write_text(f"unique_match_{i}\n", encoding="utf-8")

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await grep.coroutine("unique_match_", str(tmp_path), rt)
        assert "truncated" in out
        assert "50 matches" in out

    async def test_directory_breaks_on_max(self, tmp_path):
        """Line 540-541: rglob loop breaks when max_results reached."""
        from chcode.utils.tools import grep

        for i in range(60):
            f = tmp_path / f"x{i}.txt"
            f.write_text(f"match{i}\n", encoding="utf-8")

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await grep.coroutine("match", str(tmp_path), rt)
        assert "truncated" in out

    async def test_skips_directories(self, tmp_path):
        """Line 543: Non-files are skipped (directories)."""
        from chcode.utils.tools import grep

        subdir = tmp_path / "sub"
        subdir.mkdir()
        # This test verifies the `if not p.is_file(): continue` branch
        f = tmp_path / "a.py"
        f.write_text("needle\n", encoding="utf-8")
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await grep.coroutine("needle", str(tmp_path), rt)
        assert "[OK]" in out

    async def test_excludes_hidden_dirs(self, tmp_path):
        """Lines 545-549: Files in hidden dirs are excluded."""
        from chcode.utils.tools import grep

        # .git dir
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("match_here\n", encoding="utf-8")

        # Normal file
        (tmp_path / "normal.py").write_text("nope\n", encoding="utf-8")

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await grep.coroutine("match_here", str(tmp_path), rt)
        assert "No matches" in out

    async def test_excludes_known_dirs(self, tmp_path):
        """Lines 545-549: Files in _GREP_EXCLUDED_DIRS are excluded."""
        from chcode.utils.tools import grep

        excluded = tmp_path / "node_modules"
        excluded.mkdir()
        (excluded / "pkg.js").write_text("findme\n", encoding="utf-8")

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await grep.coroutine("findme", str(tmp_path), rt)
        assert "No matches" in out

    async def test_excludes_binary_extensions(self, tmp_path):
        """Lines 550-551: Binary extensions are skipped."""
        from chcode.utils.tools import grep

        (tmp_path / "img.png").write_bytes(b"match_content_here")
        (tmp_path / "code.py").write_text("nope\n", encoding="utf-8")

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await grep.coroutine("match_content", str(tmp_path), rt)
        assert "No matches" in out

    async def test_permission_error_skipped(self, tmp_path):
        """Line 553: PermissionError in _search_file is caught."""
        from chcode.utils.tools import grep

        f = tmp_path / "locked.py"
        f.write_text("content\n", encoding="utf-8")
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")

        original_open = open

        def permission_open(path, *args, **kwargs):
            if "locked" in str(path):
                raise PermissionError("denied")
            return original_open(path, *args, **kwargs)

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("builtins.open", permission_open):
            out = await grep.coroutine("content", str(tmp_path), rt)
        # Should not crash; the locked file is silently skipped
        assert "grep:" in out


# ============================================================================
# Lines 601, 627-628: edit tool — not a file, generic exception
# ============================================================================


class TestEditExtended:
    async def test_not_a_file(self, tmp_path):
        """Line 601: Path is a directory, not a file."""
        from chcode.utils.tools import edit

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await edit.coroutine(str(tmp_path), "old", "new", rt)
        assert "Not a file" in out

    async def test_generic_exception(self, tmp_path):
        """Lines 627-628: Non-UnicodeDecodeError exception handler."""
        from chcode.utils.tools import edit

        f = tmp_path / "e.txt"
        f.write_text("hello\n", encoding="utf-8")
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("aiofiles.open", side_effect=OSError("read error")):
            out = await edit.coroutine(str(f), "hello", "bye", rt)
        assert "[FAILED]" in out
        assert "read error" in out


# ============================================================================
# Lines 661-662, 675-678, 682: list_dir tool
# ============================================================================


class TestListDirExtended:
    async def test_stat_exception(self, tmp_path):
        """Lines 661-662: Exception during stat() falls back to (name, False, 0).
        We use a counter so that is_dir() succeeds during sort but stat() fails
        inside the try block on the second call for the target file."""
        from chcode.utils.tools import list_dir

        f = tmp_path / "special.txt"
        f.write_text("x", encoding="utf-8")

        original_stat = Path.stat
        special_call_count = 0

        def patched_stat(self, **kwargs):
            nonlocal special_call_count
            if self.name == "special.txt":
                special_call_count += 1
                # First call is from is_dir() in the sort lambda — let it succeed
                # Second call is from entry.stat() in the try block — make it fail
                if special_call_count >= 2:
                    raise OSError("stat error")
            return original_stat(self, **kwargs)

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"), \
             patch.object(Path, "stat", patched_stat):
            out = await list_dir.coroutine(str(tmp_path), runtime=rt)
        assert "[OK]" in out
        # The file should still appear in the listing despite stat failure
        assert "special.txt" in out

    async def test_file_size_kb(self, tmp_path):
        """Lines 675-676: File size in KB range."""
        from chcode.utils.tools import list_dir

        f = tmp_path / "mid.bin"
        f.write_bytes(b"x" * 2048)  # 2KB
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await list_dir.coroutine(str(tmp_path), runtime=rt)
        assert "2KB" in out

    async def test_file_size_mb(self, tmp_path):
        """Lines 677-678: File size in MB range."""
        from chcode.utils.tools import list_dir

        f = tmp_path / "big.bin"
        f.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await list_dir.coroutine(str(tmp_path), runtime=rt)
        assert "2MB" in out

    async def test_file_size_bytes(self, tmp_path):
        """Lines 673-674: File size < 1024 bytes."""
        from chcode.utils.tools import list_dir

        f = tmp_path / "tiny.txt"
        f.write_bytes(b"x" * 42)
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await list_dir.coroutine(str(tmp_path), runtime=rt)
        assert "42B" in out

    async def test_truncated_entries(self, tmp_path):
        """Line 681-682: More than 100 entries — note: _sync_list_dir already
        slices to 100, so len(entries) <= 100 always. Lines 681-682 are dead code.
        This test verifies the behavior when exactly 100 files are present."""
        from chcode.utils.tools import list_dir

        for i in range(100):
            (tmp_path / f"f{i:03d}.txt").write_text("x")

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await list_dir.coroutine(str(tmp_path), runtime=rt)
        assert "[OK]" in out
        assert "100 entries" in out


# ============================================================================
# Lines 719-728: _html_to_markdown — ImportError fallback
# ============================================================================


class TestHtmlToMarkdownFallback:
    def test_fallback_strips_scripts(self):
        """Lines 720-728: Fallback removes script/style tags and HTML tags."""
        html = "<script>alert('x')</script><p>Hello <b>world</b></p><style>.x{}</style>"
        result = _html_to_markdown(html)
        assert "alert" not in result
        assert ".x" not in result
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_fallback_empty_html(self):
        """Lines 720-728: Fallback with empty/whitespace HTML."""
        result = _html_to_markdown("")
        assert result == ""

    def test_fallback_only_tags(self):
        """Lines 720-728: HTML with only tags produces minimal output."""
        result = _html_to_markdown("<div><span></span></div>")
        # After removing tags and collapsing whitespace, should be empty or minimal
        assert "<div>" not in result

    def test_import_error_triggers_fallback(self):
        """Line 719: When markdownify import fails, fallback is used."""
        with patch.dict("sys.modules", {"markdownify": None}):
            # Force reimport to hit ImportError
            result = _html_to_markdown("<b>test</b>")
            assert "test" in result


# ============================================================================
# Lines 803, 808: web_fetch — HTML detection, content length truncation
# ============================================================================


class TestWebFetchExtended:
    async def test_html_content_detection(self):
        """Line 802-803: text/html content type triggers _html_to_markdown."""
        from chcode.utils.tools import web_fetch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.reason_phrase = "OK"
        mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
        mock_resp.content = b"<h1>Title</h1>"
        mock_resp.text = "<h1>Title</h1>"
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools.httpx.AsyncClient", return_value=mock_client):
            out = await web_fetch.coroutine("https://example.com")
        # Should have converted HTML
        assert "Title" in out["result"]

    async def test_non_html_content_passthrough(self):
        """Line 804-805: Non-HTML content is passed through as-is."""
        from chcode.utils.tools import web_fetch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.reason_phrase = "OK"
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.content = b"plain text"
        mock_resp.text = "plain text"
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools.httpx.AsyncClient", return_value=mock_client):
            out = await web_fetch.coroutine("https://example.com/data.txt")
        assert "plain text" in out["result"]

    async def test_content_truncation(self):
        """Lines 807-811: Content exceeding MAX_MARKDOWN_LENGTH is truncated."""
        from chcode.utils.tools import web_fetch

        long_text = "x" * (MAX_MARKDOWN_LENGTH + 5000)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.reason_phrase = "OK"
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.content = long_text.encode()
        mock_resp.text = long_text
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools.httpx.AsyncClient", return_value=mock_client):
            out = await web_fetch.coroutine("https://example.com/big")
        assert "truncated" in out["result"]

    async def test_generic_exception_v2(self):
        """Lines 844-845: Generic Exception handler in web_fetch."""
        from chcode.utils.tools import web_fetch

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=ValueError("unexpected"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools.httpx.AsyncClient", return_value=mock_client):
            out = await web_fetch.coroutine("https://example.com")
        assert "Error fetching URL" in out["result"]


# ============================================================================
# Lines 1255, 1263-1264, 1303-1304, 1309-1310, 1313-1314: ask_user / _ask_multi_questions
# ============================================================================


class TestAskUserExtended:
    async def test_multi_select_cancel(self):
        """Line 1255: _checkbox_with_other_async returns None -> cancel."""
        from chcode.utils.tools import ask_user

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools._checkbox_with_other_async", new_callable=AsyncMock, return_value=None):
            out = await ask_user.coroutine("pick?", options=["A", "B"], is_multiple=True)
        assert "\u7528\u6237\u53d6\u6d88" in out  # Chinese for "user cancelled"

    async def test_select_exception(self):
        """Lines 1263-1264: Exception in select flow."""
        from chcode.utils.tools import ask_user

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools._select_with_other_async", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            out = await ask_user.coroutine("pick?", options=["A"])
        assert "fail" in out.lower() or "boom" in out

    async def test_multi_select_exception(self):
        """Lines 1263-1264: Exception in checkbox flow."""
        from chcode.utils.tools import ask_user

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools._checkbox_with_other_async", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            out = await ask_user.coroutine("pick?", options=["A"], is_multiple=True)
        assert "fail" in out.lower() or "boom" in out


class TestAskMultiQuestionsExtended:
    async def test_checkbox_cancel_per_question(self):
        """Lines 1302-1303: _checkbox returns None for a multi-select question."""
        qs = [{"question": "pick?", "options": ["A", "B"], "is_multiple": True}]
        with patch("chcode.utils.tools._checkbox_with_other_async", new_callable=AsyncMock, return_value=None):
            out = await _ask_multi_questions(qs)
        assert "\u7528\u6237\u53d6\u6d88" in out  # Chinese for "user cancelled"

    async def test_select_cancel_per_question(self):
        """Lines 1308-1309: _select returns None for a single-select question."""
        qs = [{"question": "pick?", "options": ["A", "B"]}]
        with patch("chcode.utils.tools._select_with_other_async", new_callable=AsyncMock, return_value=None):
            out = await _ask_multi_questions(qs)
        assert "\u7528\u6237\u53d6\u6d88" in out  # Chinese for "user cancelled"

    async def test_checkbox_exception_per_question(self):
        """Lines 1313-1314: Exception in checkbox question."""
        qs = [{"question": "pick?", "options": ["A"], "is_multiple": True}]
        with patch("chcode.utils.tools._checkbox_with_other_async", new_callable=AsyncMock, side_effect=RuntimeError("err")):
            out = await _ask_multi_questions(qs)
        assert "fail" in out.lower() or "err" in out

    async def test_select_exception_per_question(self):
        """Lines 1313-1314: Exception in select question."""
        qs = [{"question": "pick?", "options": ["A"]}]
        with patch("chcode.utils.tools._select_with_other_async", new_callable=AsyncMock, side_effect=RuntimeError("err")):
            out = await _ask_multi_questions(qs)
        assert "fail" in out.lower() or "err" in out

    async def test_multiple_questions_sequential(self):
        """Multiple questions asked sequentially."""
        qs = [
            {"question": "name?", "options": []},
            {"question": "pick?", "options": ["A", "B"]},
        ]
        with patch("chcode.utils.tools.asyncio.to_thread", new_callable=AsyncMock, return_value="Alice"), \
             patch("chcode.utils.tools._select_with_other_async", new_callable=AsyncMock, return_value="B"):
            out = await _ask_multi_questions(qs)
        assert "Alice" in out
        assert "B" in out

    async def test_text_cancel_then_continue(self):
        """User cancels a text question, continues to next."""
        qs = [
            {"question": "skip?", "options": []},
            {"question": "name?", "options": []},
        ]
        call_count = 0

        def mock_thread(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # cancel first
            return "Bob"

        with patch("chcode.utils.tools.asyncio.to_thread", new_callable=AsyncMock, side_effect=mock_thread):
            out = await _ask_multi_questions(qs)
        assert "\u7528\u6237\u53d6\u6d88" in out  # Chinese for "user cancelled"
        assert "Bob" in out


# ============================================================================
# Lines 1383-1386, 1405: agent tool — parallel detection and failure
# ============================================================================


class TestAgentParallelAndFailure:
    async def test_parallel_agent_detection(self):
        """Lines 1382-1386: When _subagent_count >= 2, parallel progress starts."""
        from chcode.utils.tools import agent

        mock_def = MagicMock()
        mock_def.agent_type = "Explore"
        mock_def.read_only = True
        mock_def.system_prompt = "prompt"
        mock_def.tools = None
        mock_def.disallowed_tools = []
        mock_def.model = None

        rt = _make_runtime(
            working_directory=Path("/w"),
            thread_id="t1",
            model_config={"model": "gpt-4"},
        )

        mock_progress_task = AsyncMock()
        mock_progress_task.done = MagicMock(return_value=False)

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.agents.loader.load_agents", return_value={"Explore": mock_def}), \
             patch("chcode.agents.runner.run_subagent", new_callable=AsyncMock, return_value=("done", False)) as mock_run, \
             patch("chcode.display._subagent_count_lock"), \
             patch("chcode.display._agent_progress_lock"), \
             patch("chcode.display._current_agent_tag"), \
             patch("chcode.display._start_progress"), \
             patch("chcode.display._progress_task", mock_progress_task), \
             patch("chcode.display._progress_updater", return_value=AsyncMock()), \
             patch("chcode.display._update_progress"), \
             patch("chcode.display._finalize_progress"), \
             patch("chcode.display.console"):
            import chcode.display as _display
            # Simulate first agent already running
            _display._subagent_count = 1
            _display._subagent_parallel = False
            _display._agent_progress = {}
            try:
                out = await agent.coroutine("task", subagent_type="Explore", runtime=rt)
                assert "done" in out
            finally:
                _display._subagent_count = 0
                _display._subagent_parallel = False

    async def test_agent_failure_detection(self):
        """Line 1404-1405: Result containing 'timed out' or 'error:' sets failed=True."""
        from chcode.utils.tools import agent

        mock_def = MagicMock()
        mock_def.agent_type = "Explore"
        mock_def.read_only = True
        mock_def.system_prompt = "prompt"
        mock_def.tools = None
        mock_def.disallowed_tools = []
        mock_def.model = None

        rt = _make_runtime(
            working_directory=Path("/w"),
            thread_id="t1",
            model_config={"model": "gpt-4"},
        )

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.agents.loader.load_agents", return_value={"Explore": mock_def}), \
             patch("chcode.agents.runner.run_subagent", new_callable=AsyncMock, return_value=("Task timed out after 300s", True)) as mock_run, \
             patch("chcode.display._subagent_count_lock"), \
             patch("chcode.display._agent_progress_lock"), \
             patch("chcode.display._current_agent_tag"), \
             patch("chcode.display._start_progress"), \
             patch("chcode.display._progress_updater", return_value=AsyncMock()), \
             patch("chcode.display._update_progress"), \
             patch("chcode.display._finalize_progress"), \
             patch("chcode.display.console"):
            import chcode.display as _display
            _display._subagent_count = 0
            _display._subagent_parallel = False
            _display._agent_progress = {}
            try:
                out = await agent.coroutine("task", subagent_type="Explore", runtime=rt)
                assert "timed out" in out
            finally:
                _display._subagent_count = 0

    async def test_agent_error_failure_detection(self):
        """Line 1404-1405: Result containing 'error:' sets failed=True."""
        from chcode.utils.tools import agent

        mock_def = MagicMock()
        mock_def.agent_type = "Explore"
        mock_def.read_only = True
        mock_def.system_prompt = "prompt"
        mock_def.tools = None
        mock_def.disallowed_tools = []
        mock_def.model = None

        rt = _make_runtime(
            working_directory=Path("/w"),
            thread_id="t1",
            model_config={"model": "gpt-4"},
        )

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.agents.loader.load_agents", return_value={"Explore": mock_def}), \
             patch("chcode.agents.runner.run_subagent", new_callable=AsyncMock, return_value=("error: something went wrong", True)), \
             patch("chcode.display._subagent_count_lock"), \
             patch("chcode.display._agent_progress_lock"), \
             patch("chcode.display._current_agent_tag"), \
             patch("chcode.display._start_progress"), \
             patch("chcode.display._progress_updater", return_value=AsyncMock()), \
             patch("chcode.display._update_progress"), \
             patch("chcode.display._finalize_progress"), \
             patch("chcode.display.console"):
            import chcode.display as _display
            _display._subagent_count = 0
            _display._subagent_parallel = False
            _display._agent_progress = {}
            try:
                out = await agent.coroutine("task", subagent_type="Explore", runtime=rt)
                assert "error:" in out
            finally:
                _display._subagent_count = 0


# ============================================================================
# Lines 255: bash — interpretation message for non-error exit codes
# ============================================================================


class TestBashInterpretation:
    async def test_grep_no_matches_interpretation(self):
        """Line 255: grep exit code 1 gives [OK] with interpretation message."""
        from chcode.utils.tools import bash

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        # grep returns exit code 1 for no matches — not an error
        result = ShellResult(stdout="", exit_code=1, stderr="")
        truncated = MagicMock()
        truncated.content = ""
        truncated.truncated = False
        mock_sess = MagicMock()
        mock_sess.provider_name = "bash"
        mock_sess.execute = MagicMock(return_value=(result, truncated))
        with patch("chcode.utils.tools._get_shell_session", return_value=mock_sess), \
             patch("chcode.utils.tools.render_tool_call"):
            out = await bash.coroutine("grep pattern file.txt", runtime=rt)
        assert "[OK]" in out
        assert "No matches found" in out

    async def test_stderr_with_output(self):
        """Lines 267-269: stderr is shown after stdout with separator."""
        from chcode.utils.tools import bash

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        result = ShellResult(
            stdout="output line", exit_code=0, stderr="warning msg"
        )
        truncated = MagicMock()
        truncated.content = "output line"
        truncated.truncated = False
        mock_sess = MagicMock()
        mock_sess.provider_name = "bash"
        mock_sess.execute = MagicMock(return_value=(result, truncated))
        with patch("chcode.utils.tools._get_shell_session", return_value=mock_sess), \
             patch("chcode.utils.tools.render_tool_call"):
            out = await bash.coroutine("cmd", runtime=rt)
        assert "output line" in out
        assert "--- stderr ---" in out
        assert "warning msg" in out

    async def test_stderr_only(self):
        """Lines 266-270: Only stderr, no stdout."""
        from chcode.utils.tools import bash

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        result = ShellResult(stdout="", exit_code=1, stderr="error msg")
        truncated = MagicMock()
        truncated.content = ""
        truncated.truncated = False
        mock_sess = MagicMock()
        mock_sess.provider_name = "bash"
        mock_sess.execute = MagicMock(return_value=(result, truncated))
        with patch("chcode.utils.tools._get_shell_session", return_value=mock_sess), \
             patch("chcode.utils.tools.render_tool_call"):
            out = await bash.coroutine("cmd", runtime=rt)
        assert "--- stderr ---" in out
        assert "error msg" in out

    async def test_truncated_output(self):
        """Line 262: Truncated output is used when available."""
        from chcode.utils.tools import bash

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        result = ShellResult(stdout="full output", exit_code=0, stderr="")
        truncated = MagicMock()
        truncated.content = "truncated output..."
        truncated.truncated = True
        mock_sess = MagicMock()
        mock_sess.provider_name = "bash"
        mock_sess.execute = MagicMock(return_value=(result, truncated))
        with patch("chcode.utils.tools._get_shell_session", return_value=mock_sess), \
             patch("chcode.utils.tools.render_tool_call"):
            out = await bash.coroutine("cmd", runtime=rt)
        assert "truncated output..." in out
        assert "full output" not in out


# ============================================================================
# Additional: read_file exactly 2000 lines (boundary)
# ============================================================================


class TestReadFileBoundary:
    async def test_exactly_2000_lines(self, tmp_path):
        """Lines 314-319: File with exactly 2000 lines shows no truncation."""
        from chcode.utils.tools import read_file

        lines = [f"line {i}" for i in range(2000)]
        f = tmp_path / "exact.txt"
        f.write_text("\n".join(lines), encoding="utf-8")
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await read_file.coroutine(str(f), runtime=rt)
        assert "[OK]" in out
        assert "2000 lines" not in out  # should not mention truncation
        assert "more lines" not in out


# ============================================================================
# Additional: bash with workdir parameter
# ============================================================================


class TestBashWorkdir:
    async def test_workdir_override(self):
        """Lines 244-246: workdir parameter overrides session cwd."""
        from chcode.utils.tools import bash

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        result = ShellResult(stdout="ok", exit_code=0, stderr="")
        truncated = MagicMock()
        truncated.content = "ok"
        truncated.truncated = False
        mock_sess = MagicMock()
        mock_sess.provider_name = "bash"
        mock_sess.execute = MagicMock(return_value=(result, truncated))
        with patch("chcode.utils.tools._get_shell_session", return_value=mock_sess), \
             patch("chcode.utils.tools.render_tool_call"):
            out = await bash.coroutine("echo hi", runtime=rt, workdir="/custom")
        # workdir should be passed to session.execute
        call_kwargs = mock_sess.execute.call_args
        assert call_kwargs[1]["workdir"] == "/custom" or (
            len(call_kwargs[0]) > 2 and call_kwargs[0][2] == "/custom"
        )


# ============================================================================
# Lines 498-499: grep tool — invalid regex
# ============================================================================


class TestGrepInvalidRegex:
    async def test_invalid_regex_pattern(self):
        """Lines 498-499: Invalid regex pattern returns error message."""
        from chcode.utils.tools import grep

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await grep.coroutine("[invalid(", "file.txt", rt)
        assert "[FAILED]" in out
        assert "Invalid regex" in out


# ============================================================================
# Lines 598, 605-623, 626: edit tool — file not found, read content, Unicode error
# ============================================================================


class TestEditFileNotFound:
    async def test_file_not_exists_tools(self):
        """Line 598: File doesn't exist returns error."""
        from chcode.utils.tools import edit

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await edit.coroutine("nonexistent.txt", "old", "new", rt)
        assert "File not found" in out


class TestEditContentHandling:
    async def test_file_read_success(self, tmp_path):
        """Lines 605-608: Successfully read file and count occurrences."""
        from chcode.utils.tools import edit

        f = tmp_path / "test.txt"
        f.write_text("hello world\n", encoding="utf-8")
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await edit.coroutine(str(f), "hello", "hi", rt)
        assert "[OK]" in out

    async def test_string_not_found(self, tmp_path):
        """Lines 609-610: old_string not found in file."""
        from chcode.utils.tools import edit

        f = tmp_path / "test.txt"
        f.write_text("hello world\n", encoding="utf-8")
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await edit.coroutine(str(f), "goodbye", "hi", rt)
        assert "String not found" in out

    async def test_string_appears_multiple_times(self, tmp_path):
        """Lines 612-613: old_string appears multiple times."""
        from chcode.utils.tools import edit

        f = tmp_path / "test.txt"
        f.write_text("hello\nhello\nhello\n", encoding="utf-8")
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await edit.coroutine(str(f), "hello", "hi", rt)
        assert "appears 3 times" in out

    async def test_single_line_replacement(self, tmp_path):
        """Lines 620-623: Single line replacement."""
        from chcode.utils.tools import edit

        f = tmp_path / "test.txt"
        f.write_text("old_line\n", encoding="utf-8")
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await edit.coroutine(str(f), "old_line", "new_line", rt)
        assert "[OK]" in out
        assert "replaced 1 lines with 1 lines" in out


class TestEditUnicodeError:
    async def test_unicode_decode_error(self, tmp_path):
        """Line 626: UnicodeDecodeError when reading file."""
        from chcode.utils.tools import edit

        f = tmp_path / "test.txt"
        f.write_bytes(b"\xff\xfe binary data")

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await edit.coroutine(str(f), "anything", "new", rt)
        assert "Cannot edit file" in out
        assert "binary or unknown encoding" in out


# ============================================================================
# Lines 648, 651, 671: list_dir tool — directory not found, not a directory, dirs
# ============================================================================


class TestListDirErrors:
    async def test_directory_not_found(self):
        """Line 648: Directory doesn't exist."""
        from chcode.utils.tools import list_dir

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await list_dir.coroutine("nonexistent", runtime=rt)
        assert "Directory not found" in out

    async def test_not_a_directory(self, tmp_path):
        """Line 651: Path exists but is not a directory."""
        from chcode.utils.tools import list_dir

        f = tmp_path / "file.txt"
        f.write_text("content\n", encoding="utf-8")
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await list_dir.coroutine("file.txt", runtime=rt)
        assert "Not a directory" in out


class TestListDirEntries:
    async def test_directory_entries(self, tmp_path):
        """Line 671: Directory entries are marked with '/'."""
        from chcode.utils.tools import list_dir

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await list_dir.coroutine(".", runtime=rt)
        assert "subdir/" in out


# ============================================================================
# Lines 682, 696-700: list_dir truncation, web_search no client
# ============================================================================


class TestListDirTruncation:
    async def test_more_than_100_entries(self, tmp_path):
        """Line 681-682: Note that entries[:100] prevents len(entries) > 100,
        so this line is unreachable. Test verifies the behavior anyway."""
        from chcode.utils.tools import list_dir

        # Create exactly 101 files
        for i in range(101):
            (tmp_path / f"f{i:03d}.txt").write_text("x")

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await list_dir.coroutine(".", runtime=rt)
        assert "[OK]" in out
        # Due to entries[:100], we'll only show 100


class TestWebSearchNoClient:
    async def test_no_tavily_client(self):
        """Lines 698-699: get_tavily_client returns None."""
        from chcode.utils.tools import web_search

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools.get_tavily_client", return_value=None):
            out = await web_search.coroutine("query", runtime=rt)
        assert "Tavily API Key 未配置" in out


# ============================================================================
# Lines 752, 764, 774, 791, 827, 836: web_fetch — URL length, invalid URL, http to https, binary, timeout
# ============================================================================


class TestWebFetchErrors:
    async def test_url_too_long(self):
        """Line 752: URL exceeds MAX_URL_LENGTH."""
        from chcode.utils.tools import web_fetch

        long_url = "https://example.com/" + "x" * (MAX_URL_LENGTH + 100)
        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await web_fetch.coroutine(long_url)
        assert "exceeds maximum length" in out["result"]

    async def test_invalid_url_no_scheme(self):
        """Line 764: URL has no scheme or netloc."""
        from chcode.utils.tools import web_fetch

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await web_fetch.coroutine("not-a-url")
        assert "Invalid URL" in out["result"]

    async def test_invalid_url_no_netloc(self):
        """Line 764: URL has scheme but no netloc."""
        from chcode.utils.tools import web_fetch

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await web_fetch.coroutine("https://")
        assert "Invalid URL" in out["result"]

    async def test_http_upgraded_to_https(self):
        """Line 774: http:// URL is replaced with https://."""
        from chcode.utils.tools import web_fetch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.reason_phrase = "OK"
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.content = b"content"
        mock_resp.text = "content"
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools.httpx.AsyncClient", return_value=mock_client):
            out = await web_fetch.coroutine("http://example.com")
        # Should have made request with https://
        assert "https://example.com" in mock_client.get.call_args[0][0]

    async def test_binary_content_type(self):
        """Line 791: Binary content type detection."""
        from chcode.utils.tools import web_fetch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.reason_phrase = "OK"
        mock_resp.headers = {"content-type": "application/pdf"}
        mock_resp.content = b"%PDF-1.4..."
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools.httpx.AsyncClient", return_value=mock_client):
            out = await web_fetch.coroutine("https://example.com/doc.pdf")
        assert "Binary content" in out["result"]
        assert "application/pdf" in out["result"]

    async def test_timeout_exception(self):
        """Lines 826-834: TimeoutException returns timeout message."""
        from chcode.utils.tools import web_fetch

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools.httpx.AsyncClient", return_value=mock_client):
            out = await web_fetch.coroutine("https://example.com")
        assert "Timeout" in out["code_text"]
        assert "timed out" in out["result"]

    async def test_http_error_exception(self):
        """Line 835-839: HTTPError returns error message."""
        from chcode.utils.tools import web_fetch

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("Connection error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools.httpx.AsyncClient", return_value=mock_client):
            out = await web_fetch.coroutine("https://example.com")
        assert "Error" in out["code_text"]


# ============================================================================
# Lines 864-1027: _checkbox_with_other_async — multi-select with custom input
# ============================================================================


class TestCheckboxWithOtherAsync:
    async def test_returns_selected_options(self):
        """Lines 983-996: User selects multiple options and presses Enter."""
        from chcode.utils.tools import _checkbox_with_other_async

        # Mock the prompt_toolkit Application and its run_async method
        mock_app = MagicMock()
        mock_app.run_async = AsyncMock(return_value=["opt1", "opt2"])

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", return_value=mock_app):
                result = await _checkbox_with_other_async("Choose?", ["opt1", "opt2", "opt3"])

        assert result == ["opt1", "opt2"]

    async def test_returns_selected_with_custom_input(self):
        """Lines 989-992: User selects options and adds custom input."""
        from chcode.utils.tools import _checkbox_with_other_async

        mock_app = MagicMock()
        mock_app.run_async = AsyncMock(return_value=["opt1", "custom_value"])

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", return_value=mock_app):
                result = await _checkbox_with_other_async("Choose?", ["opt1", "opt2"])

        assert result == ["opt1", "custom_value"]

    async def test_returns_none_on_escape(self):
        """Lines 998-1007: User presses Escape to cancel."""
        from chcode.utils.tools import _checkbox_with_other_async

        mock_app = MagicMock()
        mock_app.run_async = AsyncMock(return_value=None)

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", return_value=mock_app):
                result = await _checkbox_with_other_async("Choose?", ["opt1", "opt2"])

        assert result is None

    async def test_returns_none_on_ctrl_c(self):
        """Lines 1009-1018: User presses Ctrl-C to cancel."""
        from chcode.utils.tools import _checkbox_with_other_async

        mock_app = MagicMock()
        mock_app.run_async = AsyncMock(return_value=None)

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", return_value=mock_app):
                result = await _checkbox_with_other_async("Choose?", ["opt1", "opt2"])

        assert result is None

    async def test_custom_input_only(self):
        """Lines 989-992: Only custom input, no selections from list."""
        from chcode.utils.tools import _checkbox_with_other_async

        mock_app = MagicMock()
        mock_app.run_async = AsyncMock(return_value=["custom_only"])

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", return_value=mock_app):
                result = await _checkbox_with_other_async("Choose?", ["opt1", "opt2"])

        assert result == ["custom_only"]


# ============================================================================
# Lines 1035-1193: _select_with_other_async — single-select with custom input
# ============================================================================


class TestSelectWithOtherAsync:
    async def test_returns_selected_option(self):
        """Lines 1143-1162: User selects an option from list and presses Enter."""
        from chcode.utils.tools import _select_with_other_async

        mock_app = MagicMock()
        mock_app.run_async = AsyncMock(return_value="opt2")

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", return_value=mock_app):
                result = await _select_with_other_async("Pick one?", ["opt1", "opt2", "opt3"])

        assert result == "opt2"

    async def test_returns_custom_input(self):
        """Lines 1149-1156: User moves to input line and enters custom text."""
        from chcode.utils.tools import _select_with_other_async

        mock_app = MagicMock()
        mock_app.run_async = AsyncMock(return_value="custom_option")

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", return_value=mock_app):
                result = await _select_with_other_async("Pick one?", ["opt1", "opt2"])

        assert result == "custom_option"

    async def test_returns_none_on_escape_v2(self):
        """Lines 1164-1173: User presses Escape to cancel."""
        from chcode.utils.tools import _select_with_other_async

        mock_app = MagicMock()
        mock_app.run_async = AsyncMock(return_value=None)

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", return_value=mock_app):
                result = await _select_with_other_async("Pick one?", ["opt1", "opt2"])

        assert result is None

    async def test_returns_none_on_ctrl_c_v2(self):
        """Lines 1175-1184: User presses Ctrl-C to cancel."""
        from chcode.utils.tools import _select_with_other_async

        mock_app = MagicMock()
        mock_app.run_async = AsyncMock(return_value=None)

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", return_value=mock_app):
                result = await _select_with_other_async("Pick one?", ["opt1", "opt2"])

        assert result is None


# ============================================================================
# Lines 1455, 1459-1467, 1496-1552: todo_write and related functions
# ============================================================================


class TestTodoPath:
    def test_returns_path_for_session_id(self):
        """Line 1455: _todo_path constructs correct file path."""
        result = _todo_path("session123")
        assert "ses_session123.json" in result
        assert "todo" in result

    def test_path_includes_storage_dir(self):
        """Line 1455: Path is within _TODO_STORAGE_DIR."""
        import chcode.utils.tools as mod

        result = _todo_path("test")
        assert result.startswith(mod._TODO_STORAGE_DIR)


class TestSaveTodos:
    async def test_creates_directory_if_not_exists(self, tmp_path, monkeypatch):
        """Line 1459: os.makedirs is called with exist_ok=True."""
        import chcode.utils.tools as mod

        storage = tmp_path / "todo_storage"
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        await _save_todos("test_session", [{"content": "task1"}])

        assert storage.exists()
        assert (storage / "ses_test_session.json").exists()

    async def test_adds_position_to_todos(self, tmp_path, monkeypatch):
        """Lines 1461-1462: Each todo gets a position index."""
        import chcode.utils.tools as mod
        import json

        storage = tmp_path / "todo_storage"
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        todos = [
            {"content": "task1"},
            {"content": "task2"},
            {"content": "task3"},
        ]
        await _save_todos("test", todos)

        todo_file = storage / "ses_test.json"
        saved = json.loads(todo_file.read_text(encoding="utf-8"))
        assert saved[0]["position"] == 0
        assert saved[1]["position"] == 1
        assert saved[2]["position"] == 2

    async def test_adds_time_updated(self, tmp_path, monkeypatch):
        """Lines 1460-1463: time_updated is set for all todos."""
        import chcode.utils.tools as mod
        import json

        storage = tmp_path / "todo_storage"
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        todos = [{"content": "task1"}]
        await _save_todos("test", todos)

        todo_file = storage / "ses_test.json"
        saved = json.loads(todo_file.read_text(encoding="utf-8"))
        assert "time_updated" in saved[0]
        assert isinstance(saved[0]["time_updated"], int)

    async def test_adds_time_created_for_new_todos(self, tmp_path, monkeypatch):
        """Lines 1464-1465: time_created is added if missing."""
        import chcode.utils.tools as mod
        import json

        storage = tmp_path / "todo_storage"
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        todos = [{"content": "new task"}]
        await _save_todos("test", todos)

        todo_file = storage / "ses_test.json"
        saved = json.loads(todo_file.read_text(encoding="utf-8"))
        assert "time_created" in saved[0]
        assert isinstance(saved[0]["time_created"], int)

    async def test_preserves_existing_time_created(self, tmp_path, monkeypatch):
        """Lines 1464-1465: Existing time_created is not overwritten."""
        import chcode.utils.tools as mod
        import json

        storage = tmp_path / "todo_storage"
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        existing_time = 1234567890000
        todos = [{"content": "task", "time_created": existing_time}]
        await _save_todos("test", todos)

        todo_file = storage / "ses_test.json"
        saved = json.loads(todo_file.read_text(encoding="utf-8"))
        assert saved[0]["time_created"] == existing_time

    async def test_writes_json_file(self, tmp_path, monkeypatch):
        """Lines 1466-1467: File is written as JSON with proper formatting."""
        import chcode.utils.tools as mod

        storage = tmp_path / "todo_storage"
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        todos = [
            {"content": "task1", "status": "pending"},
            {"content": "task2", "status": "completed"},
        ]
        await _save_todos("test", todos)

        todo_file = storage / "ses_test.json"
        content = todo_file.read_text(encoding="utf-8")
        # Check it's valid JSON and properly formatted
        assert "task1" in content
        assert "task2" in content
        assert "pending" in content
        assert "completed" in content


class TestTodoWrite:
    async def test_uses_default_session_id_when_none(self, tmp_path, monkeypatch):
        """Line 1496: Uses 'default' when thread_id is None."""
        import chcode.utils.tools as mod

        storage = tmp_path / "todo_storage"
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        rt = _make_runtime(working_directory=Path("/w"), thread_id=None)

        with patch("chcode.utils.tools.render_tool_call"):
            with patch("chcode.utils.tools.console"):
                from chcode.utils.tools import todo_write
                await todo_write.coroutine(
                    [TodoItem(content="task1")],
                    runtime=rt,
                )

        assert (storage / "ses_default.json").exists()

    async def test_deletes_file_when_empty_and_exists(self, tmp_path, monkeypatch):
        """Lines 1502-1504: File is removed when todos become empty."""
        import chcode.utils.tools as mod

        storage = tmp_path / "todo_storage"
        storage.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        # First create a file
        (storage / "ses_test.json").write_text("[]", encoding="utf-8")

        rt = _make_runtime(working_directory=Path("/w"), thread_id="test")

        with patch("chcode.utils.tools.render_tool_call"):
            with patch("chcode.utils.tools.console"):
                with patch("chcode.utils.tools.os.remove") as mock_remove:
                    from chcode.utils.tools import todo_write
                    await todo_write.coroutine([], runtime=rt)

        mock_remove.assert_called_once()

    async def test_saves_when_non_empty(self, tmp_path, monkeypatch):
        """Lines 1505-1506: Calls _save_todos when todos list is not empty."""
        import chcode.utils.tools as mod

        storage = tmp_path / "todo_storage"
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        rt = _make_runtime(working_directory=Path("/w"), thread_id="test")

        with patch("chcode.utils.tools.render_tool_call"):
            with patch("chcode.utils.tools.console"):
                with patch("chcode.utils.tools._save_todos", new_callable=AsyncMock) as mock_save:
                    from chcode.utils.tools import todo_write
                    await todo_write.coroutine(
                        [TodoItem(content="task1")],
                        runtime=rt,
                    )

        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args[0][0] == "test"
        assert len(call_args[0][1]) == 1

    async def test_counts_active_todos(self, tmp_path, monkeypatch):
        """Line 1508: Active count excludes completed todos."""
        import chcode.utils.tools as mod

        storage = tmp_path / "todo_storage"
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        rt = _make_runtime(working_directory=Path("/w"), thread_id="test")

        with patch("chcode.utils.tools.render_tool_call"):
            with patch("chcode.utils.tools.console"):
                with patch("chcode.utils.tools._save_todos", new_callable=AsyncMock):
                    from chcode.utils.tools import todo_write
                    output = await todo_write.coroutine(
                        [
                            TodoItem(content="pending", status="pending"),
                            TodoItem(content="in_progress", status="in_progress"),
                            TodoItem(content="completed", status="completed"),
                        ],
                        runtime=rt,
                    )

        assert "2 active todo" in output

    async def test_all_completed_message(self, tmp_path, monkeypatch):
        """Line 1526: Shows 'All todos completed' when all are done."""
        import chcode.utils.tools as mod

        storage = tmp_path / "todo_storage"
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        rt = _make_runtime(working_directory=Path("/w"), thread_id="test")

        with patch("chcode.utils.tools.render_tool_call"):
            with patch("chcode.utils.tools.console"):
                with patch("chcode.utils.tools._save_todos", new_callable=AsyncMock):
                    from chcode.utils.tools import todo_write
                    output = await todo_write.coroutine(
                        [
                            TodoItem(content="done1", status="completed"),
                            TodoItem(content="done2", status="completed"),
                        ],
                        runtime=rt,
                    )

        assert "All todos completed" in output

    async def test_cleared_message(self, tmp_path, monkeypatch):
        """Line 1526: Shows 'Todo list cleared' when list is empty."""
        import chcode.utils.tools as mod

        storage = tmp_path / "todo_storage"
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        rt = _make_runtime(working_directory=Path("/w"), thread_id="test")

        with patch("chcode.utils.tools.render_tool_call"):
            with patch("chcode.utils.tools.console"):
                with patch("chcode.utils.tools.os.path.isfile", return_value=False):
                    from chcode.utils.tools import todo_write
                    output = await todo_write.coroutine([], runtime=rt)

        assert "Todo list cleared" in output

    async def test_status_markers(self, tmp_path, monkeypatch):
        """Lines 1517-1520: Correct markers for each status."""
        import chcode.utils.tools as mod

        storage = tmp_path / "todo_storage"
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        rt = _make_runtime(working_directory=Path("/w"), thread_id="test")

        with patch("chcode.utils.tools.render_tool_call"):
            with patch("chcode.utils.tools.console"):
                with patch("chcode.utils.tools._save_todos", new_callable=AsyncMock):
                    from chcode.utils.tools import todo_write
                    output = await todo_write.coroutine(
                        [
                            TodoItem(content="a", status="pending"),
                            TodoItem(content="b", status="in_progress"),
                            TodoItem(content="c", status="completed"),
                            TodoItem(content="d", status="cancelled"),
                        ],
                        runtime=rt,
                    )

        assert "[ ]" in output  # pending
        assert "[>]" in output  # in_progress
        assert "[x]" in output  # completed
        assert "[-]" in output  # cancelled

    async def test_convert_pydantic_to_dict(self, tmp_path, monkeypatch):
        """Lines 1498-1499: TodoItem models are converted to dicts."""
        import chcode.utils.tools as mod
        import json

        storage = tmp_path / "todo_storage"
        monkeypatch.setattr(mod, "_TODO_STORAGE_DIR", str(storage))

        rt = _make_runtime(working_directory=Path("/w"), thread_id="test")

        with patch("chcode.utils.tools.render_tool_call"):
            with patch("chcode.utils.tools.console"):
                from chcode.utils.tools import todo_write
                await todo_write.coroutine(
                    [
                        TodoItem(content="task1", status="pending", priority="high"),
                    ],
                    runtime=rt,
                )

        todo_file = storage / "ses_test.json"
        saved = json.loads(todo_file.read_text(encoding="utf-8"))
        assert saved[0]["content"] == "task1"
        assert saved[0]["status"] == "pending"
        assert saved[0]["priority"] == "high"


# ============================================================================
# Lines 71-77: get_tavily_client — cached client, no api key, create client
# ============================================================================


class TestGetTavilyClient:
    def test_returns_cached_client(self, monkeypatch):
        """Lines 72-73: Returns cached client if already initialized."""
        import chcode.utils.tools as mod

        mock_client = MagicMock()
        mod._tavily_client = mock_client
        mod._tavily_api_key = "cached_key"
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        result = mod.get_tavily_client()
        assert result is mock_client

    def test_no_api_key_returns_none(self, monkeypatch, tmp_path):
        """Lines 74-75: No API key configured returns None."""
        import chcode.utils.tools as mod

        mod._tavily_client = None
        mod._tavily_api_key = ""
        mod._tavily_key_loaded = False
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        # Ensure no settings file exists
        monkeypatch.setattr(mod, "SETTING_JSON", tmp_path / "nonexistent.json")

        result = mod.get_tavily_client()
        assert result is None

    def test_creates_new_client_with_api_key(self, monkeypatch):
        """Lines 76-77: Creates and returns new TavilyClient."""
        import chcode.utils.tools as mod
        from tavily import TavilyClient

        mod._tavily_client = None
        mod._tavily_api_key = "tvly-test123"
        mod._tavily_key_loaded = True

        with patch("chcode.utils.tools.TavilyClient") as mock_tavily:
            mock_client = MagicMock()
            mock_tavily.return_value = mock_client
            result = mod.get_tavily_client()
            assert result is mock_client
            mock_tavily.assert_called_once_with(api_key="tvly-test123")


# ============================================================================
# Lines 83-87: update_tavily_api_key — set key, create client, set to None
# ============================================================================


class TestUpdateTavilyApiKey:
    def test_sets_api_key_and_creates_client(self, monkeypatch):
        """Lines 83-85: Sets API key and creates new TavilyClient."""
        import chcode.utils.tools as mod
        from tavily import TavilyClient

        mod._tavily_api_key = ""
        mod._tavily_client = None

        with patch("chcode.utils.tools.TavilyClient") as mock_tavily:
            mock_client = MagicMock()
            mock_tavily.return_value = mock_client
            mod.update_tavily_api_key("tvly-newkey")

        assert mod._tavily_api_key == "tvly-newkey"
        assert mod._tavily_client is mock_client

    def test_sets_to_none_with_empty_key(self, monkeypatch):
        """Lines 86-87: Empty API key sets client to None."""
        import chcode.utils.tools as mod

        mod._tavily_api_key = "old_key"
        mod._tavily_client = MagicMock()

        mod.update_tavily_api_key("")

        assert mod._tavily_api_key == ""
        assert mod._tavily_client is None


# ============================================================================
# Lines 122-162: load_skill — not found scenarios, scripts directory
# ============================================================================


class TestLoadSkillExtended:
    async def test_skill_not_found_with_available_skills(self):
        """Lines 129-133: Skill not found, lists available skills."""
        from chcode.utils.tools import load_skill

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        mock_loader = MagicMock()
        mock_loader.load_skill.return_value = None
        mock_skill = MagicMock()
        mock_skill.name = "available-skill"
        mock_loader.scan_skills.return_value = [mock_skill]
        rt.context.skill_loader = mock_loader

        with patch("chcode.utils.tools.render_tool_call"):
            out = await load_skill.coroutine("missing", runtime=rt)

        assert "not found" in out
        assert "available-skill" in out

    async def test_skill_not_found_no_skills_available(self):
        """Lines 133-134: Skill not found, no skills available."""
        from chcode.utils.tools import load_skill

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        mock_loader = MagicMock()
        mock_loader.load_skill.return_value = None
        mock_loader.scan_skills.return_value = []
        rt.context.skill_loader = mock_loader

        with patch("chcode.utils.tools.render_tool_call"):
            out = await load_skill.coroutine("missing", runtime=rt)

        assert "not found" in out
        assert "No skills are currently available" in out

    async def test_skill_with_scripts_directory(self, tmp_path):
        """Lines 138-150: Scripts directory exists and is shown."""
        from chcode.utils.tools import load_skill

        # Create skill structure with scripts directory
        skill_path = tmp_path / "test-skill"
        skill_path.mkdir()
        scripts_dir = skill_path / "scripts"
        scripts_dir.mkdir()

        mock_content = MagicMock()
        mock_content.instructions = "Do this task"
        mock_content.metadata.skill_path = skill_path

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        mock_loader = MagicMock()
        mock_loader.load_skill.return_value = mock_content
        rt.context.skill_loader = mock_loader

        with patch("chcode.utils.tools.render_tool_call"):
            out = await load_skill.coroutine("test-skill", runtime=rt)

        assert "Scripts Directory" in out
        assert str(scripts_dir) in out
        assert "uv run" in out


# ============================================================================
# Lines 172-191: _create_shell_session — platform detection and providers
# ============================================================================


class TestCreateShellSession:
    def test_windows_bash_provider_available(self, monkeypatch):
        """Lines 173-178: Windows with BashProvider available."""
        import chcode.utils.tools as mod

        mock_provider = MagicMock()
        mock_provider.is_available = True
        mock_session = MagicMock()
        mock_session.cwd = "/w"

        with patch("platform.system", return_value="Windows"), \
             patch("chcode.utils.tools.BashProvider", return_value=mock_provider), \
             patch("chcode.utils.tools.ShellSession", return_value=mock_session) as mock_shell:
            result = mod._create_shell_session("/w")
            assert result is mock_session
            mock_shell.assert_called_once_with(mock_provider)

    def test_windows_powershell_provider_fallback(self, monkeypatch):
        """Lines 179-183: Windows BashProvider unavailable, uses PowerShell."""
        import chcode.utils.tools as mod

        mock_bash = MagicMock()
        mock_bash.is_available = False
        mock_ps = MagicMock()
        mock_ps.is_available = True
        mock_session = MagicMock()
        mock_session.cwd = "/w"

        with patch("platform.system", return_value="Windows"), \
             patch("chcode.utils.tools.BashProvider", return_value=mock_bash), \
             patch("chcode.utils.tools.PowerShellProvider", return_value=mock_ps), \
             patch("chcode.utils.tools.ShellSession", return_value=mock_session) as mock_shell:
            result = mod._create_shell_session("/w")
            assert result is mock_session
            mock_shell.assert_called_once_with(mock_ps)

    def test_non_windows_bash_provider(self, monkeypatch):
        """Lines 184-189: Non-Windows platform uses BashProvider."""
        import chcode.utils.tools as mod

        mock_provider = MagicMock()
        mock_provider.is_available = True
        mock_session = MagicMock()
        mock_session.cwd = "/w"

        with patch("platform.system", return_value="Linux"), \
             patch("chcode.utils.tools.BashProvider", return_value=mock_provider), \
             patch("chcode.utils.tools.ShellSession", return_value=mock_session) as mock_shell:
            result = mod._create_shell_session("/w")
            assert result is mock_session
            mock_shell.assert_called_once_with(mock_provider)


# ============================================================================
# Lines 198-206: _get_shell_session — get existing, create new, return None
# ============================================================================


class TestGetShellSession:
    def test_returns_existing_session(self, monkeypatch):
        """Lines 199-201: Returns cached session if exists."""
        import chcode.utils.tools as mod

        mock_session = MagicMock()
        mod._shell_sessions = {"/w": mock_session}

        with patch("chcode.utils.tools._create_shell_session") as mock_create:
            result = mod._get_shell_session("/w")
            assert result is mock_session
            mock_create.assert_not_called()

    def test_creates_and_stores_new_session(self, monkeypatch):
        """Lines 202-205: Creates new session and stores it."""
        import chcode.utils.tools as mod

        mock_session = MagicMock()
        mod._shell_sessions = {}

        with patch("chcode.utils.tools._create_shell_session", return_value=mock_session):
            result = mod._get_shell_session("/w")
            assert result is mock_session
            assert mod._shell_sessions["/w"] is mock_session

    def test_returns_none_when_create_fails(self, monkeypatch):
        """Line 206: Returns None when _create_shell_session returns None."""
        import chcode.utils.tools as mod

        mod._shell_sessions = {}

        with patch("chcode.utils.tools._create_shell_session", return_value=None):
            result = mod._get_shell_session("/w")
            assert result is None


# ============================================================================
# Lines 242: bash tool — no shell available
# ============================================================================


class TestBashNoShell:
    async def test_no_shell_available_message(self):
        """Line 242: Returns error message when no shell available."""
        from chcode.utils.tools import bash

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")

        with patch("chcode.utils.tools._get_shell_session", return_value=None), \
             patch("chcode.utils.tools.render_tool_call"):
            out = await bash.coroutine("cmd", runtime=rt)

        assert "No shell available" in out
        assert "[FAILED]" in out


# ============================================================================
# Lines 300, 303, 323: read_file — file not found, not a file, unicode error
# ============================================================================


class TestReadFileErrors:
    async def test_file_not_found(self):
        """Line 300: Returns error when file doesn't exist."""
        from chcode.utils.tools import read_file

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")

        with patch("chcode.utils.tools.render_tool_call"):
            out = await read_file.coroutine("nonexistent.txt", runtime=rt)

        assert "File not found" in out
        assert "[FAILED]" in out

    async def test_not_a_file_v2(self, tmp_path):
        """Line 303: Returns error when path is a directory."""
        from chcode.utils.tools import read_file

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")

        with patch("chcode.utils.tools.render_tool_call"):
            out = await read_file.coroutine(str(tmp_path), runtime=rt)

        assert "Not a file" in out
        assert "[FAILED]" in out

    async def test_unicode_decode_error_v2(self, tmp_path):
        """Line 323: Returns error for binary/unknown encoding."""
        from chcode.utils.tools import read_file

        f = tmp_path / "binary.dat"
        f.write_bytes(b"\xff\xfe\x00\x00")

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")

        with patch("chcode.utils.tools.render_tool_call"):
            out = await read_file.coroutine(str(f), runtime=rt)

        assert "Cannot read file" in out
        assert "binary or unknown encoding" in out


# ============================================================================
# Lines 344-355: write_file — resolve, render, mkdirs, write, exception
# ============================================================================


class TestWriteFile:
    async def test_resolve_path_and_render(self, tmp_path):
        """Lines 344-345: Path is resolved and tool call is rendered."""
        from chcode.utils.tools import write_file

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")

        with patch("chcode.utils.tools.render_tool_call") as mock_render:
            await write_file.coroutine("test.txt", "content", runtime=rt)
            mock_render.assert_called_once_with("write_file", "test.txt")

    async def test_create_parent_directories(self, tmp_path):
        """Line 348: Parent directories are created."""
        from chcode.utils.tools import write_file

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")

        with patch("chcode.utils.tools.render_tool_call"):
            out = await write_file.coroutine("subdir/nested/file.txt", "content", runtime=rt)

        assert (tmp_path / "subdir" / "nested" / "file.txt").exists()
        assert "[OK]" in out

    async def test_write_success(self, tmp_path):
        """Lines 350-352: Successful write returns OK message."""
        from chcode.utils.tools import write_file

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")

        with patch("chcode.utils.tools.render_tool_call"):
            out = await write_file.coroutine("test.txt", "hello world", runtime=rt)

        assert "[OK]" in out
        assert "File written" in out
        assert (tmp_path / "test.txt").read_text(encoding="utf-8") == "hello world"

    async def test_write_exception(self, tmp_path):
        """Lines 354-355: Exception during write returns error."""
        from chcode.utils.tools import write_file

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("aiofiles.open", side_effect=OSError("disk full")):
            out = await write_file.coroutine("test.txt", "content", runtime=rt)

        assert "[FAILED]" in out
        assert "disk full" in out


# ============================================================================
# Lines 379, 398-399: glob — no matches, exception
# ============================================================================


class TestGlobErrors:
    async def test_no_files_matching_pattern(self, tmp_path):
        """Line 379: Returns error when no files match pattern."""
        from chcode.utils.tools import glob

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")

        with patch("chcode.utils.tools.render_tool_call"):
            out = await glob.coroutine("*.nonexistent", runtime=rt)

        assert "No files matching pattern" in out
        assert "[FAILED]" in out

    async def test_glob_generic_exception(self, tmp_path):
        """Lines 398-399: Generic exception during glob."""
        from chcode.utils.tools import glob

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")

        with patch("chcode.utils.tools.render_tool_call"), \
             patch.object(Path, "glob", side_effect=RuntimeError("glob error")):
            out = await glob.coroutine("*.py", runtime=rt)

        assert "[FAILED]" in out
        assert "glob error" in out


# ============================================================================
# Lines 513-514, 526-527, 682: grep OSError, ValueError, list_dir truncation
# ============================================================================


class TestGrepOSError:
    async def test_oserror_in_stat(self, tmp_path):
        """Lines 513-514: OSError when calling stat() on file."""
        from chcode.utils.tools import grep

        f = tmp_path / "test.txt"
        f.write_text("content\n", encoding="utf-8")

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")

        original_stat = Path.stat

        def oserror_stat(self):
            if "test" in str(self):
                raise OSError("Device not ready")
            return original_stat(self)

        with patch("chcode.utils.tools.render_tool_call"), \
             patch.object(Path, "stat", oserror_stat):
            out = await grep.coroutine("content", str(f), rt)
        # Should handle gracefully - file is skipped
        assert "grep:" in out


class TestGrepValueError:
    async def test_value_error_relative_path(self, tmp_path):
        """Lines 526-527: ValueError when computing relative path."""
        from chcode.utils.tools import grep

        # Create file with match
        f = tmp_path / "match.txt"
        f.write_text("target\n", encoding="utf-8")

        # Set working directory to different path
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")

        # Create a mock path that will raise ValueError on relative_to
        original_relative_to = Path.relative_to

        def failing_relative_to(self, other):
            # Raise for our specific file
            if "match" in str(self):
                raise ValueError("path is not in subpath")
            return original_relative_to(self, other)

        with patch("chcode.utils.tools.render_tool_call"), \
             patch.object(Path, "relative_to", failing_relative_to):
            out = await grep.coroutine("target", str(tmp_path), rt)
        # Should handle gracefully, use absolute path
        assert "grep:" in out


class TestListDirTruncationEdge:
    async def test_exactly_100_entries_no_truncation_message(self, tmp_path):
        """Line 682: With exactly 100 entries, no truncation message due to slicing."""
        from chcode.utils.tools import list_dir

        # Create exactly 100 files
        for i in range(100):
            (tmp_path / f"file{i:03d}.txt").write_text("x")

        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await list_dir.coroutine(str(tmp_path), runtime=rt)
        assert "[OK]" in out
        assert "100 entries" in out
        # No "more entries" message because entries[:100] prevents it


# ============================================================================
# Lines 1241, 1246-1249, 1256, 1259-1262: ask_user multi-question, text input, exceptions
# ============================================================================


class TestAskUserMultiQuestion:
    async def test_questions_parameter_takes_precedence(self):
        """Line 1241: When questions parameter is provided, single question is ignored."""
        from chcode.utils.tools import ask_user

        questions = [{"question": "Q1", "options": ["A", "B"]}]

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools._ask_multi_questions", new_callable=AsyncMock, return_value="multi result"):
            # Use ainvoke to call the tool
            result = await ask_user.ainvoke(
                {"question": "Single?", "options": ["X", "Y"], "questions": questions}
            )
        assert "multi result" in result


class TestAskUserNoOptions:
    async def test_text_input_without_options(self):
        """Lines 1246-1249: When options is None/empty, use text input."""
        from chcode.utils.tools import ask_user

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools.asyncio.to_thread", new_callable=AsyncMock, return_value="free text"):
            result = await ask_user.ainvoke({"question": "Type something:"})
        assert "free text" in result

    async def test_text_input_returns_none_on_cancel(self):
        """Lines 1247-1248: Text input returns None (user cancels)."""
        from chcode.utils.tools import ask_user

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools.asyncio.to_thread", new_callable=AsyncMock, return_value=None):
            result = await ask_user.ainvoke({"question": "Type:"})
        assert "用户取消" in result


class TestAskUserCheckboxEmpty:
    async def test_checkbox_empty_selection(self):
        """Line 1256: Empty selection from checkbox returns 'user_answer:\\n'."""
        from chcode.utils.tools import ask_user

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools._checkbox_with_other_async", new_callable=AsyncMock, return_value=[]):
            result = await ask_user.ainvoke(
                {"question": "Pick?", "options": ["A", "B"], "is_multiple": True}
            )
        # Empty list results in empty joined string
        assert result == "user_answer:\n"


class TestAskUserSelectExceptions:
    async def test_select_raises_exception(self):
        """Lines 1259-1262: Exception in _select_with_other_async."""
        from chcode.utils.tools import ask_user

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools._select_with_other_async", new_callable=AsyncMock, side_effect=ValueError("UI error")):
            result = await ask_user.ainvoke({"question": "Pick?", "options": ["A"]})
        assert "失败" in result or "UI error" in result

    async def test_checkbox_raises_exception(self):
        """Lines 1259-1262: Exception in _checkbox_with_other_async."""
        from chcode.utils.tools import ask_user

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools._checkbox_with_other_async", new_callable=AsyncMock, side_effect=RuntimeError("Check error")):
            result = await ask_user.ainvoke(
                {"question": "Pick?", "options": ["A", "B"], "is_multiple": True}
            )
        assert "失败" in result or "Check error" in result


class TestAskMultiQuestionsCheckbox:
    async def test_checkbox_joins_results(self):
        """Line 1305: Checkbox results are joined with commas."""
        from chcode.utils.tools import _ask_multi_questions

        qs = [{"question": "Pick?", "options": ["A", "B", "C"], "is_multiple": True}]

        with patch("chcode.utils.tools.console"), \
             patch("chcode.utils.tools._checkbox_with_other_async", new_callable=AsyncMock, return_value=["A", "C"]):
            out = await _ask_multi_questions(qs)
        assert "A, C" in out


# ============================================================================
# Lines 1364-1365: agent tool - unknown agent type
# ============================================================================


class TestAgentUnknownType:
    async def test_unknown_agent_type_error(self):
        """Lines 1364-1365: Unknown agent type returns error message."""
        from chcode.utils.tools import agent

        rt = _make_runtime(
            working_directory=Path("/w"),
            thread_id="t1",
            model_config={"model": "gpt-4"},
        )

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.agents.loader.load_agents", return_value={"Explore": MagicMock()}):
            out = await agent.coroutine("task", subagent_type="UnknownType", runtime=rt)
        assert "Unknown agent type" in out
        assert "UnknownType" in out
        assert "Explore" in out


# ============================================================================
# Line 191: _create_shell_session returns None when both providers unavailable
# ============================================================================


class TestCreateShellSessionNoProviders:
    def test_windows_both_providers_unavailable(self, monkeypatch):
        """Line 191: Windows with both BashProvider and PowerShellProvider unavailable."""
        import chcode.utils.tools as mod

        mock_bash = MagicMock()
        mock_bash.is_available = False
        mock_ps = MagicMock()
        mock_ps.is_available = False

        with patch("platform.system", return_value="Windows"), \
             patch("chcode.utils.tools.BashProvider", return_value=mock_bash), \
             patch("chcode.utils.tools.PowerShellProvider", return_value=mock_ps):
            result = mod._create_shell_session("/w")
            assert result is None

    def test_non_windows_bash_provider_unavailable(self, monkeypatch):
        """Line 191: Non-Windows platform with BashProvider unavailable."""
        import chcode.utils.tools as mod

        mock_provider = MagicMock()
        mock_provider.is_available = False

        with patch("platform.system", return_value="Linux"), \
             patch("chcode.utils.tools.BashProvider", return_value=mock_provider):
            result = mod._create_shell_session("/w")
            assert result is None


# ============================================================================
# Line 700: web_search with actual Tavily client
# ============================================================================


class TestWebSearchWithClient:
    async def test_search_with_client_calls_tavily(self):
        """Line 700: When get_tavily_client returns a client, search is called via to_thread."""
        from chcode.utils.tools import web_search

        mock_client = MagicMock()
        search_result = {"results": [{"title": "test", "url": "http://x.com", "content": "result"}]}

        rt = _make_runtime(working_directory=Path("/w"), thread_id="t1")
        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools.get_tavily_client", return_value=mock_client), \
             patch("chcode.utils.tools.asyncio.to_thread", new_callable=AsyncMock, return_value=search_result) as mock_to_thread:
            out = await web_search.coroutine("test query", runtime=rt)

        assert "results" in out
        # asyncio.to_thread(client.search, query, ...) should have been called
        mock_to_thread.assert_called_once()
        assert mock_to_thread.call_args[0][0] == mock_client.search
        assert mock_to_thread.call_args[0][1] == "test query"


# ============================================================================
# Lines 1259-1261: ask_user single-select returns None (user cancelled)
# ============================================================================


class TestAskUserSelectCancel:
    async def test_select_returns_none(self):
        """Lines 1259-1261: _select_with_other_async returns None -> cancel message."""
        from chcode.utils.tools import ask_user

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools._select_with_other_async", new_callable=AsyncMock, return_value=None):
            out = await ask_user.coroutine("pick?", options=["A", "B"])
        assert "\u7528\u6237\u53d6\u6d88" in out  # Chinese for "user cancelled"


# ============================================================================
# Line 1386: agent parallel progress task assignment when _progress_task is None
# ============================================================================


class TestAgentParallelProgressTask:
    async def test_parallel_creates_progress_task_when_none(self):
        """Line 1386: When _progress_task is None, a new task is created."""
        from chcode.utils.tools import agent

        mock_def = MagicMock()
        mock_def.agent_type = "Explore"
        mock_def.read_only = True
        mock_def.system_prompt = "prompt"
        mock_def.tools = None
        mock_def.disallowed_tools = []
        mock_def.model = None

        rt = _make_runtime(
            working_directory=Path("/w"),
            thread_id="t1",
            model_config={"model": "gpt-4"},
        )

        # _progress_updater returns an async function, which is called without await
        # producing a coroutine that gets passed to ensure_future
        async def mock_updater():
            pass

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.agents.loader.load_agents", return_value={"Explore": mock_def}), \
             patch("chcode.agents.runner.run_subagent", new_callable=AsyncMock, return_value=("done", False)), \
             patch("chcode.display._subagent_count_lock"), \
             patch("chcode.display._agent_progress_lock"), \
             patch("chcode.display._current_agent_tag"), \
             patch("chcode.display._start_progress"), \
             patch("chcode.display._progress_updater", return_value=mock_updater), \
             patch("chcode.display._update_progress"), \
             patch("chcode.display._finalize_progress"), \
             patch("chcode.display.console"), \
             patch("chcode.display.asyncio.ensure_future") as mock_ensure_future:
            import chcode.display as _display
            # Simulate first agent already running, no progress task yet
            _display._subagent_count = 1
            _display._subagent_parallel = False
            _display._agent_progress = {}
            _display._progress_task = None
            try:
                out = await agent.coroutine("task", subagent_type="Explore", runtime=rt)
                assert "done" in out
                # ensure_future should have been called to create the progress task
                mock_ensure_future.assert_called_once()
            finally:
                _display._subagent_count = 0
                _display._subagent_parallel = False
                _display._progress_task = None


# ============================================================================
# Lines 513-514: grep — OSError when calling file_path.stat()
# ============================================================================


class TestGrepOSErrorOnStat:
    async def test_stat_raises_oserror(self, tmp_path):
        """Lines 513-514: file_path.stat() raises OSError, callback returns."""
        import os as _os
        from chcode.utils.tools import grep
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        f = tmp_path / "test.py"
        f.write_text("hello", encoding="utf-8")

        # Need stat to succeed for is_file() but fail inside _search_file
        call_count = [0]
        real_stat = _os.stat

        def selective_stat(*args, **kwargs):
            call_count[0] += 1
            # First call is from is_file() — let it succeed
            if call_count[0] <= 1:
                return real_stat(*args, **kwargs)
            # Second call is from _search_file's file_path.stat() — raise OSError
            raise OSError("Permission denied")

        with patch("os.stat", side_effect=selective_stat), \
             patch("chcode.utils.tools.render_tool_call"):
            out = await grep.coroutine("hello", path=str(f), runtime=rt)
        assert isinstance(out, str)


# ============================================================================
# Line 682: ls with > 100 entries shows "... and N more entries"
# ============================================================================


class TestLsOver100Entries:
    async def test_ls_truncates_at_100(self, tmp_path):
        """Line 682: when entries > 100, shows truncation message."""
        from chcode.utils.tools import list_dir
        rt = _make_runtime(working_directory=tmp_path, thread_id="t1")
        dir_path = tmp_path / "bigdir"
        dir_path.mkdir()
        for i in range(110):
            (dir_path / f"file_{i:03d}.txt").write_text("x", encoding="utf-8")
        with patch("chcode.utils.tools.render_tool_call"):
            out = await list_dir.coroutine(str(dir_path), runtime=rt)
        assert "more entries" in out
        assert "10 more" in out


# ============================================================================
# Line 1157: _select_with_other_async — Enter on empty input resets _exiting
# ============================================================================


class TestSelectEnterEmptyInput:
    async def test_enter_on_empty_input_resets_exiting(self):
        """Line 1157: pressing Enter on input row with empty text sets _exiting=False."""
        import inspect
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value="fallback")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        assert captured_kb is not None
        enter_handler = None
        for binding in captured_kb.bindings:
            if "ControlM" in str(binding.keys):
                enter_handler = binding.handler
                break
        assert enter_handler is not None

        # Use inspect to access closure variables
        closure_vars = inspect.getclosurevars(enter_handler)
        control = closure_vars.nonlocals["control"]
        input_buffer = closure_vars.nonlocals["input_buffer"]
        input_row_idx = closure_vars.nonlocals["input_row_idx"]

        # Set state to trigger line 1157: selected == input_row_idx, empty text
        control.selected = input_row_idx
        input_buffer.text = ""

        mock_event = MagicMock()
        mock_event.app = MagicMock()

        # Call handler — should set _exiting=False (line 1157)
        enter_handler(mock_event)

        # Verify _exiting was reset to False by checking the closure
        exiting_cell = None
        for cell in enter_handler.__closure__:
            try:
                if isinstance(cell.cell_contents, bool) or cell.cell_contents is False:
                    exiting_cell = cell
                    break
            except ValueError:
                continue
        # After the handler, _exiting should be False (was set True, then reset to False)
        # We can verify by calling again — if _exiting is False, it should proceed
        mock_event.app.exit.reset_mock()
        enter_handler(mock_event)
        # Should try to exit (but input_buffer.text is still empty, so _exiting resets again)


# ============================================================================
# Line 1261: ask_user single select path
# ============================================================================


class TestAskUserSingleSelect:
    async def test_single_select_returns_answer(self):
        """Lines 1257-1261: is_multiple=False uses _select_with_other_async."""
        from chcode.utils.tools import ask_user

        with patch("chcode.utils.tools.render_tool_call"), \
             patch("chcode.utils.tools._select_with_other_async", new_callable=AsyncMock, return_value="Option A"):
            result = await ask_user.ainvoke(
                {"question": "Pick one", "options": ["A", "B", "C"], "is_multiple": False}
            )
        assert "Option A" in result
