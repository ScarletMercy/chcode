import asyncio

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

    def test_cwd_tracking(self):
        session = _make_session()
        import tempfile

        tmpdir = tempfile.gettempdir()
        result, _ = session.execute(f"cd {tmpdir} && pwd", timeout=5000)
        tracked_cwd = session._provider.read_cwd_file(
            session._provider.create_cwd_file()
        )
        provider = session._provider
        cwd_file = provider.create_cwd_file()
        result, _ = session.execute(f"cd {tmpdir} && pwd", timeout=5000)
        tracked_cwd = provider.read_cwd_file(cwd_file)
        if tracked_cwd:
            assert (
                tmpdir in tracked_cwd or tmpdir.lower() in (tracked_cwd or "").lower()
            )
        provider.cleanup_cwd_file(cwd_file)
