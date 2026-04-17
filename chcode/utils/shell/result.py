from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShellResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    interrupted: bool = False
    timed_out: bool = False
    output_file_path: str | None = None
    output_file_size: int | None = None
