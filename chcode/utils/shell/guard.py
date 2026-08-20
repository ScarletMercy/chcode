"""危险命令拦截中间件 — 在命令执行前扫描原始命令字符串。

覆盖 bash / powershell / cmd 三种 shell 方言：拦截器对整条命令做匹配，
因此无论是 bash 直接执行、powershell 执行，还是 `cmd /c ...` 嵌套调用，
只要危险片段出现在命令字符串里就会被拦下。

两类规则：
- 命令位置规则：裸命令词（shutdown / halt / reboot / mkfs 等）必须出现在
  某段命令的首个词位置（允许 sudo/doas 启动器前缀），才会命中。这样
  `cat shutdown.log`、`grep reboot /var/log/syslog` 这类把危险词当作
  文件名或参数的情况不会误报。systemctl 子命令形式（systemctl poweroff）
  作为该规则的补充一并覆盖。
- 上下文/标志规则：命令词 + 特定标志的组合（rm -rf、taskkill /f、dd if= 等）
  按子串匹配——这类组合不会出现在普通文件名或参数中。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GuardResult:
    blocked: bool
    category: str = ""
    pattern_id: str = ""


# 危险命令拦截总开关（默认开启）。可通过 /danger 斜杠命令切换。
# 持久化到 ~/.chat/chagent.json 的 "guard_enabled" 字段，跨会话保留。
_guard_enabled: bool = True

# 被禁用的拦截类别（这些类别的命令不会被拦截）。可通过 /danger 斜杠命令编辑。
# 持久化到 ~/.chat/chagent.json 的 "disabled_categories" 字段。
_disabled_categories: set[str] = set()

# 所有可拦截的类别
ALL_CATEGORIES = ["recursive_delete", "force_kill", "system_damage", "shutdown"]


def _load_persisted_state() -> None:
    """模块加载时从 chagent.json 读取持久化的开关与禁用类别（失败则保持默认值）。"""
    global _guard_enabled, _disabled_categories
    try:
        from chcode.config import _load_setting

        data = _load_setting()
        guard_value = data.get("guard_enabled")
        if isinstance(guard_value, bool):
            _guard_enabled = guard_value
        categories = data.get("disabled_categories")
        if isinstance(categories, list):
            _disabled_categories = {c for c in categories if c in ALL_CATEGORIES}
    except Exception:
        pass


_load_persisted_state()


def _persist() -> None:
    """将开关状态和禁用类别持久化到 chagent.json。"""
    try:
        from chcode.config import _update_setting

        _update_setting(
            guard_enabled=_guard_enabled,
            disabled_categories=sorted(_disabled_categories),
        )
    except Exception:
        pass


def set_guard_enabled(enabled: bool) -> None:
    """设置危险命令拦截总开关，并持久化到 ~/.chat/chagent.json。"""
    global _guard_enabled
    _guard_enabled = enabled
    _persist()


def is_guard_enabled() -> bool:
    """查询危险命令拦截总开关状态。"""
    return _guard_enabled


def set_category_enabled(category: str, enabled: bool) -> None:
    """启用或禁用某个拦截类别，并持久化。无效类别静默忽略。"""
    global _disabled_categories
    if category not in ALL_CATEGORIES:
        return
    if enabled:
        _disabled_categories.discard(category)
    else:
        _disabled_categories.add(category)
    _persist()


def is_category_enabled(category: str) -> bool:
    """查询某个拦截类别是否启用（未被禁用）。"""
    return category not in _disabled_categories


def get_disabled_categories() -> set[str]:
    """获取所有被禁用的拦截类别。"""
    return set(_disabled_categories)


def ensure_guard_config_written() -> None:
    """确保 chagent.json 中存在 guard_enabled 和 disabled_categories 字段。

    首次初始化时若字段缺失，写入默认值（拦截开启、无禁用类别），使用户配置可见；
    已存在则不做任何操作，避免覆盖用户已保存的偏好。
    应在 ChatREPL.initialize() 中、ensure_config_dir() 之后调用。
    """
    try:
        from chcode.config import _load_setting, _update_setting

        data = _load_setting()
        updated: dict[str, object] = {}
        if "guard_enabled" not in data:
            updated["guard_enabled"] = True
        if "disabled_categories" not in data:
            updated["disabled_categories"] = []
        if updated:
            _update_setting(**updated)
    except Exception:
        pass


# 启动器前缀：透明传递，使 sudo halt / doas reboot 仍按命令位置匹配
_LAUNCHERS = {"sudo", "doas"}

# systemctl 关机/重启子命令（危险词作为 systemctl 参数出现）
_SYSTEMCTL_DANGER = {"poweroff", "reboot", "halt"}


def _basename(token: str) -> str:
    """剥离路径前缀与 .exe 后缀，返回命令名小写形式（与 semantics 一致）。"""
    base = token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if base.lower().endswith(".exe"):
        base = base[:-4]
    return base


def _is_relative_path(token: str) -> bool:
    """判断 token 是否为相对路径调用（含路径分隔符且非绝对路径）。

    相对路径（./reboot、subdir/reboot、.\\script）视为用户项目内的
    同名文件/工具，豁免裸命令词拦截；绝对路径（/sbin/reboot）与裸词
    仍按系统命令处理。
    """
    if "/" not in token and "\\" not in token:
        return False
    if token.startswith("/") or token.startswith("\\"):
        return False
    if len(token) >= 2 and token[1] == ":" and token[0].isalpha():
        return False
    return True


def _segment_leaders(command: str) -> list[tuple[str, str, str]]:
    """按 ; & | && || 切段，返回每段的（原始首词, 首词基名, 次词基名），剥离启动器前缀。

    原始首词保留路径信息，供调用方区分相对路径调用（用户同名文件）与
    系统命令；基名用于命令位置规则匹配。
    """
    leaders: list[tuple[str, str, str]] = []
    for seg in re.split(r"&&|\|\||[;&|]", command):
        toks = seg.split()
        while toks and toks[0].lower() in _LAUNCHERS:
            toks = toks[1:]
        if not toks:
            continue
        raw_first = toks[0]
        first = _basename(raw_first).lower()
        second = _basename(toks[1]).lower() if len(toks) > 1 else ""
        leaders.append((raw_first, first, second))
    return leaders


def _match_position(first: str, second: str) -> tuple[str, str] | None:
    """命令位置规则：裸命令词需出现在段首（含 systemctl 子命令形式）。

    命中返回 (category, pattern_id)，否则 None。
    """
    if first in {
        "shutdown",
        "halt",
        "poweroff",
        "reboot",
        "stop-computer",
        "restart-computer",
    }:
        return ("shutdown", first)
    if first.startswith("mkfs"):
        return ("system_damage", "mkfs")
    if first == "init" and second in {"0", "6"}:
        return ("shutdown", "init_runlevel")
    if first == "systemctl" and second in _SYSTEMCTL_DANGER:
        return ("shutdown", "systemctl")
    return None


# 上下文/标志规则：命中特定命令+标志组合即拦截。
# 这些组合不会出现在普通文件名或参数中，故按子串匹配即可。
_CONTEXT_RULES: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "recursive_delete": [
        # bash: rm 带含 r 的短标志（-r / -rf / -fr）或 --recursive 长标志。
        # 长标志显式锚定到 rm，避免误伤 grep --recursive / cp --recursive 等。
        (
            "rm_recursive",
            re.compile(
                r"\brm\b[^|;&\n]*(?:\s-[a-zA-Z]*r[a-zA-Z]*\b|\s--recursive\b)",
                re.IGNORECASE,
            ),
        ),
        # powershell: Remove-Item / ri 带 -Recurse
        (
            "remove_item_recursive",
            re.compile(r"\b(?:remove-item|ri)\b[^|;&\n]*-recurse\b", re.IGNORECASE),
        ),
        # cmd: rmdir /s 、rd /s 、del /s 、erase /s （/s = 含子目录）
        (
            "cmd_rmdir_s",
            re.compile(r"\b(?:rmdir|rd)\s+/[a-z]*s[a-z]*\b", re.IGNORECASE),
        ),
        (
            "cmd_del_s",
            re.compile(r"\b(?:del|erase)\s+/[a-z]*s[a-z]*\b", re.IGNORECASE),
        ),
    ],
    "force_kill": [
        # cmd: taskkill 带 /f（强制）或 /t（结束进程树）
        (
            "taskkill_force",
            re.compile(r"\btaskkill\b[^|;&\n]*\/[a-z]*[ft][a-z]*\b", re.IGNORECASE),
        ),
        # bash: kill -9 / -KILL / -SIGKILL（精确匹配信号，避免 -19 等含 9 的信号误伤）
        (
            "kill_force",
            re.compile(
                r"\b(?:kill|killall|pkill)\b\s+(?:-\w+\s+)*-(?:9|kill|sigkill)\b",
                re.IGNORECASE,
            ),
        ),
        # powershell: Stop-Process 带 -Force
        (
            "stop_process_force",
            re.compile(r"\bstop-process\b[^|;&\n]*-force\b", re.IGNORECASE),
        ),
    ],
    "system_damage": [
        # 磁盘格式化
        ("format_drive", re.compile(r"\bformat\b\s+[a-z]:", re.IGNORECASE)),
        # dd 写入裸盘（of=/dev/sd* 等）才危险；读 if= 写到普通文件是安全操作
        (
            "dd_raw_disk",
            re.compile(
                r"\bdd\b\s+[^|;&\n]*\bof=/dev/(?:sd|nvme|hd|mmcblk)", re.IGNORECASE
            ),
        ),
        ("write_raw_disk", re.compile(r">\s*/dev/(?:sd|nvme|hd)", re.IGNORECASE)),
        ("fork_bomb", re.compile(r":\(\)\s*\{", re.IGNORECASE)),
    ],
}


def check_command(command: str) -> GuardResult:
    """检查命令字符串是否命中危险规则。

    命中任意一条即返回 blocked=True 及其类别；未命中返回 blocked=False。
    对空命令直接放行。总开关关闭时一律放行。被禁用的类别也放行。
    """
    if not _guard_enabled:
        return GuardResult(blocked=False)

    if not command or not command.strip():
        return GuardResult(blocked=False)

    # 命令位置规则（裸命令词须在段首；相对路径调用的同名文件视为用户工具，豁免）
    for raw_first, first, second in _segment_leaders(command):
        if _is_relative_path(raw_first):
            continue
        match = _match_position(first, second)
        if match is not None:
            category, pattern_id = match
            if category not in _disabled_categories:
                return GuardResult(blocked=True, category=category, pattern_id=pattern_id)

    # 上下文/标志规则
    text = command.strip()
    for category, rules in _CONTEXT_RULES.items():
        if category in _disabled_categories:
            continue
        for pattern_id, pattern in rules:
            if pattern.search(text):
                return GuardResult(blocked=True, category=category, pattern_id=pattern_id)

    return GuardResult(blocked=False)
