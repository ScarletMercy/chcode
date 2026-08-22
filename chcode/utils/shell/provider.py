from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod


class ShellProvider(ABC):
    @property
    @abstractmethod
    def shell_path(self) -> str: ...

    @property
    @abstractmethod
    def is_available(self) -> bool: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @abstractmethod
    def build_command(self, command: str, cwd_file: str) -> str: ...

    @abstractmethod
    def get_spawn_args(self, command_string: str) -> list[str]: ...

    def get_env_overrides(self) -> dict[str, str]:
        return {}

    def create_cwd_file(self) -> str:
        tmpdir = tempfile.gettempdir()
        file_id = uuid.uuid4().hex[:8]
        return os.path.join(tmpdir, f"chcode-cwd-{file_id}")

    def read_cwd_file(self, cwd_file: str) -> str | None:
        try:
            with open(cwd_file, encoding="utf-8-sig") as f:
                return f.read().strip()
        except OSError:
            return None

    def cleanup_cwd_file(self, cwd_file: str) -> None:
        with contextlib.suppress(OSError):
            os.unlink(cwd_file)


class BashProvider(ShellProvider):
    def __init__(self) -> None:
        self._shell = self._detect_shell()

    @property
    def shell_path(self) -> str:
        return self._shell

    @property
    def is_available(self) -> bool:
        return self._shell != ""

    @property
    def display_name(self) -> str:
        return "bash"

    def _detect_shell(self) -> str:
        if os.name == "nt":
            git_path = shutil.which("git")
            if git_path:
                git_bin = os.path.dirname(git_path)
                candidate = os.path.join(git_bin, "bash.exe")
                if os.path.isfile(candidate):
                    return candidate
                candidate = os.path.join(
                    git_bin, "..", "bin", "bash.exe"
                )  # pragma: no cover
                if os.path.isfile(candidate):  # pragma: no cover
                    return os.path.normpath(candidate)  # pragma: no cover
            bash_path = shutil.which("bash")  # pragma: no cover
            if bash_path and os.path.isfile(bash_path):  # pragma: no cover
                return bash_path  # pragma: no cover
            return ""
        env_shell = os.environ.get("SHELL", "")
        if env_shell and os.path.isfile(env_shell):
            return env_shell
        for candidate in ["/bin/bash", "/usr/bin/bash", "/bin/zsh", "/usr/bin/zsh"]:
            if os.path.isfile(candidate):
                return candidate
        return shutil.which("bash") or shutil.which("zsh") or ""

    def build_command(self, command: str, cwd_file: str) -> str:
        escaped_cwd = cwd_file.replace("'", "'\\''")
        escaped_cmd = command.replace("'", "'\\''")
        return f"eval '{escaped_cmd}' && pwd -P >| '{escaped_cwd}'"

    def get_spawn_args(self, command_string: str) -> list[str]:
        return ["-c", command_string]


class PowerShellProvider(ShellProvider):
    @property
    def shell_path(self) -> str:
        return "powershell"

    @property
    def is_available(self) -> bool:
        import platform

        return platform.system() == "Windows" and shutil.which("powershell") is not None

    @property
    def display_name(self) -> str:
        return "powershell"

    def build_command(self, command: str, cwd_file: str) -> str:
        escaped = cwd_file.replace("'", "''")
        return (
            f"{command}\n"
            f"; $_ec = if ($null -ne $LASTEXITCODE) {{ $LASTEXITCODE }} "
            f"elseif ($?) {{ 0 }} else {{ 1 }}\n"
            f"; (Get-Location).Path | Out-File -FilePath '{escaped}' "
            f"-Encoding utf8 -NoNewline\n"
            f"; exit $_ec"
        )

    def get_spawn_args(self, command_string: str) -> list[str]:
        return ["-NoProfile", "-NonInteractive", "-Command", command_string]

    def get_env_overrides(self) -> dict[str, str]:
        return {"PSMODULEPATH": ""}
