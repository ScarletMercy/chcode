import asyncio
import os
import subprocess

import pytest

from chcode.utils.shell.provider import BashProvider
from chcode.utils.shell.session import ShellSession


def _make_session():
    provider = BashProvider()
    if not provider.is_available:
        pytest.skip("No shell available")
    return ShellSession(provider)


class TestShellSessionIntegration:
    def test_echo_hello(self):
        session = _make_session()
        result, output = session.execute("echo hello", timeout=5000)
        assert "hello" in result.stdout
        assert result.exit_code == 0
        assert output.truncated is False

    def test_exit_code_nonzero(self):
        session = _make_session()
        result, _ = session.execute("exit 1", timeout=5000)
        assert result.exit_code != 0

    def test_stdin_is_closed_cat_exits_immediately(self):
        session = _make_session()
        result, _ = session.execute("cat", timeout=8000)
        assert result.timed_out is False
        assert result.stdout == ""
        assert result.exit_code == 0

    def test_interactive_prompt_fails_fast_without_echo(self):
        session = _make_session()
        result, _ = session.execute(
            "echo '口令：' >&2; read -r pw && echo \"got:$pw\"", timeout=8000
        )
        assert result.timed_out is False
        assert result.exit_code != 0
        assert "got:" not in result.stdout


class TestSpawnHardening:
    class _FakeProc:
        pid = 1234
        returncode = 0

        def communicate(self, timeout=None):
            return b"", b""

        def kill(self): ...

    def test_spawn_never_exposes_terminal_input(self, monkeypatch):
        captured = {}

        def fake_popen(args, **kwargs):
            captured.update(kwargs)
            return TestSpawnHardening._FakeProc()

        monkeypatch.setattr(
            "chcode.utils.shell.session.subprocess.Popen", fake_popen
        )
        session = ShellSession(BashProvider())
        session.execute("echo ok", timeout=5000)

        assert captured["stdin"] == subprocess.DEVNULL
        if os.name == "nt":
            assert captured.get("creationflags", 0) & subprocess.DETACHED_PROCESS

    def test_cwd_tracking(self):
        session = _make_session()
        target = os.path.dirname(os.path.realpath(__file__))
        session.execute(f"cd '{target}'", timeout=5000)
        assert os.path.normcase(session.cwd) == os.path.normcase(target)
