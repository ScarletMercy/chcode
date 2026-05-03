import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chcode.utils.tools import (
    _coerce_json_list,
    _html_to_markdown,
    _is_binary_content_type,
    _load_tavily_key,
    resolve_path,
)


class TestCoerceJsonList:
    def test_json_string(self):
        assert _coerce_json_list('["a", "b"]') == ["a", "b"]

    def test_invalid_json_tools(self):
        assert _coerce_json_list("not json") == "not json"

    def test_non_string(self):
        assert _coerce_json_list(["a"]) == ["a"]

    def test_number(self):
        assert _coerce_json_list(42) == 42


class TestHtmlToMarkdown:
    def test_basic_html(self):
        result = _html_to_markdown("<b>bold</b>")
        assert "bold" in result

    def test_empty_tools(self):
        result = _html_to_markdown("")
        assert result == ""


class TestIsBinaryContentType:
    def test_pdf(self):
        assert _is_binary_content_type("application/pdf") is True

    def test_image(self):
        assert _is_binary_content_type("image/png") is True

    def test_text(self):
        assert _is_binary_content_type("text/html") is False

    def test_json(self):
        assert _is_binary_content_type("application/json") is False


class TestResolvePath:
    def test_relative_path(self, tmp_path: Path):
        result = resolve_path("test.txt", tmp_path)
        assert result == tmp_path / "test.txt"

    def test_absolute_path(self, tmp_path: Path):
        import os
        drive = os.path.splitdrive(os.getcwd())[0]
        abs_path = Path(f"{drive}/some/absolute/path.txt")
        result = resolve_path(str(abs_path), tmp_path)
        assert result == abs_path


class TestLoadTavilyKey:
    def test_no_env_no_file(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with patch("chcode.config.load_tavily_api_key", return_value=""):
            assert _load_tavily_key() == ""

    def test_from_config(self, monkeypatch):
        with patch("chcode.config.load_tavily_api_key", return_value="tvly-test"):
            assert _load_tavily_key() == "tvly-test"
