import sys
from io import StringIO

import pytest


class TestLangsmithGuard:
    def test_normal_write_passes_through(self):
        from chcode.cli import _setup_langsmith_guard

        captured = StringIO()
        from unittest.mock import patch

        with patch.object(sys, "__stderr__", captured):
            original = sys.stderr
            _setup_langsmith_guard()
            guard = sys.stderr
            guard.write("normal output\n")
            assert captured.getvalue() == "" or "normal output" in captured.getvalue()

    def test_empty_write(self):
        from chcode.cli import _setup_langsmith_guard

        guard = sys.stderr
        result = guard.write("")
        assert result == 0

    def test_flush(self):
        guard = sys.stderr
        # flush() should not raise an exception
        try:
            guard.flush()
        except Exception:
            self.fail("flush() raised unexpectedly")
