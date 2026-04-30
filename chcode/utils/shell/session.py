from __future__ import annotations

import contextlib
import locale
import os
import re
import signal
import subprocess

from charset_normalizer import from_bytes

from chcode.utils.shell.output import TruncatedOutput, truncate_output
from chcode.utils.shell.provider import ShellProvider
from chcode.utils.shell.result import ShellResult


class ShellSession:
    def __init__(self, provider: ShellProvider) -> None:
        self._provider = provider
        self._cwd: str = os.getcwd()

    @property
    def cwd(self) -> str:
        return self._cwd

    @cwd.setter
    def cwd(self, value: str) -> None:
        if os.path.isdir(value):
            self._cwd = value

    @property
    def provider_name(self) -> str:
        return self._provider.display_name

    def execute(
        self,
        command: str,
        timeout: int | None = 120000,
        workdir: str | None = None,
    ) -> tuple[ShellResult, TruncatedOutput]:
        cwd_file = self._provider.create_cwd_file()
        full_command = self._provider.build_command(command, cwd_file)

        exec_cwd = workdir or self._cwd
        if not os.path.isdir(exec_cwd):
            exec_cwd = self._cwd

        spawn_args = self._provider.get_spawn_args(full_command)
        env = {**os.environ, **self._provider.get_env_overrides()}

        timeout_sec = (timeout / 1000) if timeout else None
        timed_out = False

        try:
            proc = subprocess.Popen(
                [self._provider.shell_path, *spawn_args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=exec_cwd,
                env=env,
            )
        except FileNotFoundError:
            return (
                ShellResult(exit_code=127, stderr=f"Shell not found: {self._provider.shell_path}"),
                truncate_output(""),
            )
        except OSError as e:
            return (
                ShellResult(exit_code=126, stderr=f"Failed to execute: {e}"),
                truncate_output(""),
            )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(OSError):
                _kill_proc_tree(proc)
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout_bytes, stderr_bytes = b"", b""

        stdout = _robust_decode(stdout_bytes) if stdout_bytes else ""
        stderr = _robust_decode(stderr_bytes) if stderr_bytes else ""

        new_cwd = self._provider.read_cwd_file(cwd_file)
        if new_cwd and not workdir:
            if os.name == "nt" and new_cwd.startswith("/"):
                match = re.match(r"^/([a-zA-Z])(/.*)?$", new_cwd)
                if match:
                    drive = match.group(1).upper()
                    rest = match.group(2) or "\\"
                    new_cwd = f"{drive}:{rest.replace('/', chr(92))}"
            if os.path.isdir(new_cwd):
                self._cwd = new_cwd

        self._provider.cleanup_cwd_file(cwd_file)

        result = ShellResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode if proc.returncode is not None else 1,
            timed_out=timed_out,
            interrupted=timed_out,
        )

        truncated = truncate_output(result.stdout)
        if truncated.truncated:
            result.output_file_path = truncated.persisted_path
            result.output_file_size = truncated.total_bytes

        return result, truncated


def _robust_decode(data: bytes) -> str:
    if not data:
        return ""
    system_encoding = locale.getpreferredencoding() or "utf-8"
    if len(data) >= 4:
        bom = data[:4]
        if bom[:3] == b"\xef\xbb\xbf":
            return data[3:].decode("utf-8", errors="replace")
        if bom[:2] in (b"\xff\xfe", b"\xfe\xff"):
            return data.decode("utf-16", errors="replace")
    result = from_bytes(data)
    best = result.best() if result else None
    if best and best.coherence > 0.5:
        return str(best)
    for enc in ["utf-8", "gb18030", system_encoding, "latin-1"]:
        try:
            return data.decode(enc, errors="strict")
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode(system_encoding, errors="replace")  # pragma: no cover


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    pid = proc.pid
    if pid is None:
        return

    try:
        import psutil

        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
        parent.kill()
    except ImportError:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
            )
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.kill()
        else:
            with contextlib.suppress(OSError, ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
                return
            proc.kill()
