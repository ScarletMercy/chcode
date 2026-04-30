from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Interpretation:
    is_error: bool
    message: str | None = None


_RULES: list[tuple[list[str], dict[int, str]]] = [
    (["grep", "rg", "ag", "ack", "findstr", "select-string"], {1: "No matches found"}),
    (["diff", "compare-object", "fc"], {1: "Files or inputs differ"}),
    (["test", "["], {1: "Test condition evaluated to false"}),
    (["ping"], {1: "Host unreachable or no response"}),
    (["which", "where", "where.exe", "command", "get-command"], {1: "Command not found"}),
    (["type", "cat", "get-content"], {1: "File not found or unreadable"}),
    (["mkdir"], {1: "Directory creation failed"}),
    (["robocopy"], {1: "Files copied successfully (robocopy exit 1 = success)"}),
]


def _get_base_command(command: str) -> str:
    segments = re.split(r"[|;&]", command)
    last = segments[-1].strip()
    for op in ["&&", "||"]:
        parts = last.split(op)
        last = parts[-1].strip()
    tokens = last.split()
    if not tokens:
        return ""
    base = tokens[0]
    if "/" in base or "\\" in base:
        base = base.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return base.lower()


def interpret_command_result(command: str, exit_code: int) -> Interpretation:
    if exit_code == 0:
        return Interpretation(is_error=False)

    base = _get_base_command(command)
    if not base:
        return Interpretation(is_error=True, message=f"Exit code {exit_code}")

    for commands, exit_map in _RULES:
        if base in commands or base.removesuffix(".exe") in commands:
            if exit_code in exit_map:
                return Interpretation(is_error=False, message=exit_map[exit_code])
            if 1 in exit_map and exit_code > 1:
                return Interpretation(is_error=True, message=f"Exit code {exit_code}")

    return Interpretation(is_error=True, message=f"Exit code {exit_code}")
