"""
CHCODE.md 项目记忆 — 跨会话的持久化项目知识库

机制：
- 首次在目录运行 chcode 时自动创建 CHCODE.md（分节模板）
- CHCODE.md 缺失或有效内容为空时，若存在非空 CLAUDE.md 则自动迁移其内容
  （原样保留源文件，此后始终以 CHCODE.md 为准）
- 注入方式：内容不进 system prompt，而是包成 <system-reminder> 元消息
  前置到消息流最前（每次 API 请求携带）；会话内冻结不重读（保提示词
  前缀缓存），仅在失效事件（新会话/压缩/切目录）后全量重读
- 外部修改：每轮用户输入前 mtime 轮询检测（用户回合边界，由
  chat._process_input 调用），变化时以增量 diff 提醒注入
  （不重写冻结块）；工具自身写入刷新基线、不产生提醒
- 通过 update_memory 工具写入：节内追加条目 / 整节重写

记忆条目要求：简短、与行动相关、以约束（必须/禁止/一律）表述。
禁止保存：完整接口文档、历史流水账、空泛口号、过期规则、临时任务。
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from chcode.i18n import get_language

MEMORY_FILENAME = "CHCODE.md"
LEGACY_FILENAME = "CLAUDE.md"
MAX_ENTRY_CHARS = 500  # 单条记忆长度上限，强制简要
MEMORY_SOFT_LIMIT = 8_000  # 超过即提醒模型自清理
MEMORY_HARD_LIMIT = 16_000  # 注入截断 + 禁止新增（replace 清理不受限）

# 节定义 (zh 名, en 名, zh 提示, en 提示) — 模板生成与节名匹配的单一事实源
_SECTIONS: list[tuple[str, str, str, str]] = [
    (
        "项目概览",
        "Project Overview",
        "项目主要功能：一到两句话",
        "What the project does: one or two sentences",
    ),
    (
        "开发风格",
        "Development Style",
        "命名、注释、模块划分等团队风格",
        "Naming, comments, module layout conventions",
    ),
    (
        "项目结构",
        "Project Structure",
        "关键目录与入口文件的用途",
        "Key directories and entry points",
    ),
    (
        "常用命令",
        "Common Commands",
        "构建/运行/测试/安装命令，注明包管理器类型",
        "Build / run / test / install commands; note the package manager",
    ),
    (
        "编码规范",
        "Coding Standards",
        "强制执行的编码约定：格式、类型、错误处理等",
        "Mandatory conventions: formatting, typing, error handling",
    ),
    ("禁止事项", "Prohibitions", "绝对不允许的操作", "Things that must never be done"),
    (
        "验证流程",
        "Verification Workflow",
        "改动后如何验证：测试命令、手动检查步骤",
        "How to verify changes: test commands, manual checks",
    ),
    (
        "踩过的坑",
        "Pitfalls",
        "已解决过的坑及规避方法",
        "Solved pitfalls and how to avoid them",
    ),
]

# 规范节名对（zh, en）— 工具的节名匹配用
CANONICAL_SECTIONS: list[tuple[str, str]] = [(zh, en) for zh, en, _, _ in _SECTIONS]

_HEADER_LINE_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


# ─── 模板 ────────────────────────────────────────────

_TEMPLATE_HEADERS = {
    "zh": (
        "<!-- chcode:template v1 — 本文件由 chcode 自动创建并维护，\n"
        "     是跨会话的项目长期记忆。请保持每条记录简短、与行动相关、\n"
        "     以约束方式表述（必须/禁止/一律）。 -->"
    ),
    "en": (
        "<!-- chcode:template v1 — auto-created and maintained by chcode.\n"
        "     Persistent project memory. Keep entries brief, action-related,\n"
        "     and phrased as constraints (MUST / NEVER / ALWAYS). -->"
    ),
}


def get_template(lang: str | None = None) -> str:
    """
    生成空白 CHCODE.md 模板（节提示用 HTML 注释承载，不占用有效内容）。

    Args:
        lang: 语言（zh/en），默认取当前 i18n 设置
    """
    zh = (lang or get_language()) != "en"
    parts = [_TEMPLATE_HEADERS["zh" if zh else "en"], "", "# CHCODE.md"]
    for zh_name, en_name, zh_hint, en_hint in _SECTIONS:
        parts += [
            "",
            f"## {zh_name if zh else en_name}",
            f"<!-- {zh_hint if zh else en_hint} -->",
        ]
    return "\n".join(parts) + "\n"


# ─── 内容处理 ─────────────────────────────────────────


def _strip_html_comments(text: str) -> str:
    """剥离 HTML 注释（模板提示/迁移标记不进入提示词）"""
    return _COMMENT_RE.sub("", text)


def _is_effectively_empty(text: str) -> bool:
    """剥离注释、标题行与空白后无有效内容即视为空（纯模板 = 空）"""
    stripped = _COMMENT_RE.sub("", text)
    lines = [
        ln
        for ln in stripped.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    return not lines


def _collapse_blank_lines(text: str) -> str:
    """压缩 3 个以上连续空行为 2 个（注释剥离后的整理）"""
    return re.sub(r"\n{3,}", "\n\n", text)


def _visible_content(text: str) -> str:
    """原始文本 → 注入用文本（剥离注释、压缩空行）"""
    return _collapse_blank_lines(_strip_html_comments(text)).strip()


def _read_nonempty(path: Path) -> str | None:
    """读取文件，存在且有非空白内容时返回 strip 后的文本，否则 None"""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    text = text.strip()
    return text if text else None


# ─── 发现/迁移/读取 ────────────────────────────────────


def _plan_memory_text(workdir: Path) -> tuple[str, str]:
    """
    计算 ensure 之后文件应有的文本与状态（纯读，不落盘）。

    Returns:
        (文件文本, 状态) — 状态: "created" | "migrated" | "exists"
        （"exists" 时文本为当前内容或空串，调用方不应写盘）
    """
    memory_path = workdir / MEMORY_FILENAME
    legacy_path = workdir / LEGACY_FILENAME

    if memory_path.exists():
        try:
            current = memory_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return "", "exists"  # 读失败（含非 UTF-8 编码）：不覆盖写
        if not _is_effectively_empty(current):
            return current, "exists"  # 已有内容：CHCODE.md 始终优先
        legacy = _read_nonempty(legacy_path)
        if legacy is not None:
            return f"<!-- migrated from CLAUDE.md -->\n\n{legacy}\n", "migrated"
        return current, "exists"  # 空且无 CLAUDE.md：保持原样（可能是用户主动清空）

    legacy = _read_nonempty(legacy_path)
    if legacy is not None:
        return f"<!-- migrated from CLAUDE.md -->\n\n{legacy}\n", "migrated"
    return get_template(), "created"


def ensure_project_memory(workdir: Path) -> str:
    """
    确保项目根存在 CHCODE.md，实现首次创建与 CLAUDE.md 迁移。

    规则：
    - CHCODE.md 有有效内容 → "exists"（不读 CLAUDE.md）
    - CHCODE.md 缺失或有效内容为空 + 存在非空 CLAUDE.md → 内容原样迁移
      （顶部加迁移标记，CLAUDE.md 文件保留不动）→ "migrated"
    - CHCODE.md 缺失且无 CLAUDE.md → 写入分节模板 → "created"

    Args:
        workdir: 项目根目录

    Returns:
        "created" | "migrated" | "exists"
    """
    workdir = Path(workdir)
    memory_path = workdir / MEMORY_FILENAME
    text, status = _plan_memory_text(workdir)
    if status != "exists":
        memory_path.write_text(text, encoding="utf-8", newline="\n")
    return status


def load_memory_content(workdir: Path) -> str | None:
    """
    读取 CHCODE.md 的注入用内容（剥离 HTML 注释）。

    有效内容为空（如全新模板）返回 None；文件不存在或不可读返回 None。
    不做缓存 — 调用方 get_session_memory 的会话冻结块就是缓存层。
    """
    workdir = Path(workdir)
    path = workdir / MEMORY_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    visible = _visible_content(text)
    return None if _is_effectively_empty(text) else visible


def _apply_capacity_directives(content: str) -> str:
    """
    容量控制（只影响注入文本，磁盘文件永不修改）：
    - 超软上限：追加自清理提醒，引导模型用 replace 清理过期条目
    - 超硬上限：截断注入（优先在换行处切）并附醒目警告
    """
    if len(content) <= MEMORY_SOFT_LIMIT:
        return content
    if len(content) <= MEMORY_HARD_LIMIT:
        return (
            f"{content}\n\n"
            f"> NOTE: CHCODE.md is approaching the capacity limit "
            f"({len(content)}/{MEMORY_HARD_LIMIT} chars). Actively clean up "
            f'outdated or redundant entries with update_memory (mode="replace").'
        )
    cut = content.rfind("\n", 0, MEMORY_HARD_LIMIT)
    if cut < MEMORY_SOFT_LIMIT:  # 找不到合理换行位置则硬切
        cut = MEMORY_HARD_LIMIT
    omitted = len(content) - cut
    return (
        f"{content[:cut]}\n\n"
        f"> WARNING: TRUNCATED — {omitted} characters omitted "
        f"(CHCODE.md is {len(content)} chars, injection capped at {MEMORY_HARD_LIMIT}). "
        f'New entries are blocked until it is cleaned up with update_memory (mode="replace").'
    )


# ─── 会话级注入（会话内冻结 + mtime 轮询增量提醒）───

# 会话缓存: {workdir: 冻结的注入内容（含容量指令）或 None}
_session_memory_blocks: dict[str, str | None] = {}
# 文件状态基线: {workdir: (mtime_ns, size, 磁盘原文)}
_memory_file_state: dict[str, tuple[int, int, str]] = {}

MEMORY_REMINDER_HEADER = "The following project context is available to you:"


def get_session_memory(workdir: Path) -> str | None:
    """
    会话级冻结的记忆注入内容。

    首次访问读取并冻结整个会话；外部修改不刷新本块（变更由
    check_memory_changed 以增量 diff 提醒传达），仅在
    reset_memory_cache() 失效事件后重读 — 保证注入字节稳定以命中
    提示词前缀缓存。
    """
    workdir = Path(workdir) if workdir else None
    if workdir is None:
        return None
    key = str(workdir)
    if key not in _session_memory_blocks:
        content = load_memory_content(workdir)
        _session_memory_blocks[key] = (
            None if content is None else _apply_capacity_directives(content)
        )
    return _session_memory_blocks[key]


def build_memory_reminder(workdir: Path) -> str:
    """
    构建前置到消息流最前的 <system-reminder> 元消息内容（每次 API 请求携带）。
    """
    block = get_session_memory(workdir)
    body = block or (
        "(Currently empty — populate it via the update_memory tool "
        "as you learn about the project.)"
    )
    return (
        "<system-reminder>\n"
        f"{MEMORY_REMINDER_HEADER}\n"
        "# project_memory\n"
        f"{body}\n\n"
        "This context may or may not apply to the current task; do not "
        "respond to or mention it unless it is directly relevant.\n"
        "</system-reminder>"
    )


def _diff_snippet(old: str, new: str, max_lines: int = 40) -> str:
    """新旧文本的紧凑 diff 片段，超限截断"""
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile="before",
        tofile="after",
        lineterm="",
        n=1,
    )
    lines = [ln for ln in diff][2:]  # 去掉 ---/+++ 文件头
    snippet = "\n".join(lines[:max_lines])
    return snippet + ("\n..." if len(lines) > max_lines else "")


def _seed_file_state(workdir: Path) -> None:
    """建立/刷新文件基线（mtime、size、原文）"""
    if not workdir:
        return
    workdir = Path(workdir)
    path = workdir / MEMORY_FILENAME
    try:
        st = path.stat()
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    _memory_file_state[str(workdir)] = (st.st_mtime_ns, st.st_size, text)


def refresh_memory_state(workdir: Path) -> None:
    """工具自身写入后刷新基线，避免把自己的写入误判为外部修改"""
    _seed_file_state(workdir)


def check_memory_changed(workdir: Path) -> str | None:
    """
    mtime 轮询检测 CHCODE.md 外部修改。

    每轮用户输入前执行（用户回合边界，由 chat._process_input 调用）：
    - 无基线 → 静默建立（会话启动种子）
    - mtime/size 未变 → None
    - 变了但内容一致（touched）→ 更新基线，返回 None
    - 内容有实际变化 → 更新基线，返回 diff 提醒文本

    Returns:
        提醒文本（<system-reminder>Note: ... was modified ...</system-reminder>）
        或 None
    """
    if not workdir:
        return None
    workdir = Path(workdir)
    path = workdir / MEMORY_FILENAME
    key = str(workdir)
    try:
        st = path.stat()
    except OSError:
        return None
    sig = (st.st_mtime_ns, st.st_size)
    state = _memory_file_state.get(key)
    if state is None:
        _seed_file_state(workdir)
        return None
    if (state[0], state[1]) == sig:
        return None

    try:
        new_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    old_text = state[2]
    _memory_file_state[key] = (sig[0], sig[1], new_text)

    snippet = _diff_snippet(old_text, new_text)
    if not snippet.strip():
        return None  # touched 但内容未变
    return (
        "<system-reminder>\n"
        f"Note: {path} was changed outside chcode (by the user or a linter). "
        "Work from the new version and do not revert it unless the user asks. "
        "No need to mention this change. Relevant changes:\n"
        f"{snippet}\n"
        "</system-reminder>"
    )


def reset_memory_cache() -> None:
    """
    缓存失效事件（新会话/压缩/切目录）后调用：
    清空会话冻结块与文件基线，下次访问全量重读。
    """
    _session_memory_blocks.clear()
    _memory_file_state.clear()


# ─── 生命周期门面 ─────────────────────────────────────


def begin_memory_session(workdir: Path) -> str:
    """
    会话启动/切换目录的记忆生命周期入口：
    确保文件存在（首次创建 / CLAUDE.md 迁移）→ 冻结块失效 → 重建轮询基线。

    Args:
        workdir: 项目根目录

    Returns:
        ensure_project_memory 的状态："created" | "migrated" | "exists"
    """
    workdir = Path(workdir)
    status = ensure_project_memory(workdir)
    reset_memory_cache()
    refresh_memory_state(workdir)
    return status


def reset_memory_session(workdir: Path) -> None:
    """
    会话内上下文重建（新会话/压缩）的记忆生命周期入口：
    冻结块失效重读 + 重建轮询基线。不做 ensure —— 用户会话中
    删掉 CHCODE.md 后不应被静默重建模板。
    """
    reset_memory_cache()
    refresh_memory_state(workdir)


# ─── 写入 ─────────────────────────────────────────────


def _resolve_section(section: str) -> tuple[list[str], str]:
    """
    解析节名：规范节返回 (zh/en 两个可匹配名, 展示名)，
    其他名字原样接受（自定义节）。

    展示名始终用请求原名 — 新建节的标题在写入那一刻定死，
    不随运行时语言转换（节内写入靠双语匹配归位，不依赖此名）。
    """
    name = section.strip()
    low = name.lower()
    for zh, en in CANONICAL_SECTIONS:
        if low in (zh.lower(), en.lower()):
            return [zh.lower(), en.lower()], name
    return [low], name


def _find_section_range(
    lines: list[str], accepted: list[str]
) -> tuple[int, int] | None:
    """
    在行列表中定位节：返回 (header 行号, 节末行号)。
    节在下一个任意级别 header 前结束；未命中返回 None。
    """
    start = None
    for i, line in enumerate(lines):
        m = _HEADER_LINE_RE.match(line)
        if not m:
            continue
        if start is None:
            if m.group(1).lower() in accepted:
                start = i
        else:
            return (start, i)  # 下一个 header（任意级别）即节边界
    if start is not None:
        return (start, len(lines))
    return None


def _format_entry(content: str) -> list[str]:
    """条目转 bullet 行：首行 '- ' 前缀，续行两空格缩进"""
    lines = [ln.rstrip() for ln in content.strip().split("\n")]
    return [f"- {lines[0]}"] + [f"  {ln}" if ln else "" for ln in lines[1:]]


def _check_append_capacity(text: str, mode: str) -> None:
    """容量节流：超硬上限后禁止新增，倒逼先清理（replace 清理不受限）"""
    if mode == "append":
        visible_len = len(_visible_content(text))
        if visible_len > MEMORY_HARD_LIMIT:
            raise ValueError(
                f"CHCODE.md is over capacity ({visible_len}/{MEMORY_HARD_LIMIT} chars). "
                'Clean up outdated entries with mode="replace" before adding new ones'
            )


def _compute_entry_text(
    text: str, section: str, content: str, mode: str
) -> tuple[str, str]:
    """
    在给定文件文本上计算写入结果（纯函数，不落盘）。

    Args:
        text: 当前文件文本
        section: 目标节名
        content: 条目/新节正文
        mode: "append" | "replace"

    Returns:
        (新文本, 动作描述)

    Raises:
        ValueError: 参数不合法（节名/内容为空、条目超长、mode 非法）
    """
    if mode not in ("append", "replace"):
        raise ValueError(f"Invalid mode '{mode}': use 'append' or 'replace'")
    if not section.strip():
        raise ValueError("Section must not be empty")
    content = content.strip()
    if not content:
        raise ValueError("Content must not be empty")
    if mode == "append" and len(content) > MAX_ENTRY_CHARS:
        raise ValueError(
            f"Entry too long ({len(content)} chars, max {MAX_ENTRY_CHARS}): "
            "keep each memory entry brief and action-related"
        )

    accepted, display = _resolve_section(section)
    lines = text.split("\n")
    rng = _find_section_range(lines, accepted)

    if mode == "replace":
        body = [ln.rstrip() for ln in content.split("\n")]
    else:
        body = _format_entry(content)

    if rng is None:
        # 新建节（规范节用展示名，自定义节用原名）
        if text and not text.endswith("\n"):
            text += "\n"
        text += f"\n## {display}\n" + "\n".join(body) + "\n"
        action = f"Created section '{display}'"
    else:
        start, end = rng
        header_title = _HEADER_LINE_RE.match(lines[start]).group(1)
        if mode == "append":
            # 插到节内最后一个非空行之后；节内全空则紧跟 header
            insert_at = start + 1
            for i in range(end - 1, start, -1):
                if lines[i].strip():
                    insert_at = i + 1
                    break
            new_lines = lines[:insert_at] + body + lines[insert_at:]
        else:
            new_lines = lines[: start + 1] + body + lines[end:]
        text = "\n".join(new_lines)
        if not text.endswith("\n"):
            text += "\n"
        action = (
            f"Appended entry to section '{header_title}'"
            if mode == "append"
            else f"Replaced section '{header_title}'"
        )

    return text, action


def save_memory_entry(
    workdir: Path, section: str, content: str, mode: str = "append"
) -> str:
    """
    向 CHCODE.md 写入一条记忆或重写某节。

    Args:
        workdir: 项目根目录
        section: 目标节名（规范节支持 zh/en，未命中现有节时新建）
        content: append 模式为一条简短约束式条目；replace 模式为整节新正文
        mode: "append" 节内追加条目 | "replace" 重写节正文（清理/更新用）

    Returns:
        成功描述信息

    Raises:
        ValueError: 参数不合法（节名/内容为空、条目超长、mode 非法、超容量追加）
    """
    # 确保文件存在（首条记忆可能在模板生成前写入）
    workdir = Path(workdir)
    ensure_project_memory(workdir)
    memory_path = workdir / MEMORY_FILENAME
    try:
        text = memory_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"Cannot read {memory_path}: {e}") from e

    _check_append_capacity(text, mode)
    new_text, action = _compute_entry_text(text, section, content, mode)

    memory_path.write_text(new_text, encoding="utf-8", newline="\n")
    # 刷新轮询基线：自己的写入不算外部修改，变更经由工具结果传达
    refresh_memory_state(workdir)
    return f"{action} in {memory_path}"


def preview_memory_entry(
    workdir: Path, section: str, content: str, mode: str = "append"
) -> tuple[str, str]:
    """
    预演 save_memory_entry 的写入效果（只读，绝不落盘）。

    供 HITL 审批界面展示写入前后差异使用；参数校验与容量节流
    行为与 save_memory_entry 完全一致。

    Args:
        workdir: 项目根目录
        section: 目标节名
        content: 条目/新节正文
        mode: "append" | "replace"

    Returns:
        (写入前文本, 写入后文本)

    Raises:
        ValueError: 与 save_memory_entry 相同的校验失败
    """
    workdir = Path(workdir)
    old_text, _status = _plan_memory_text(workdir)
    _check_append_capacity(old_text, mode)
    new_text, _action = _compute_entry_text(old_text, section, content, mode)
    return old_text, new_text


def _content_lines(text: str) -> list[str]:
    """提取有效内容行（剥 HTML 注释与空行）— diff 展示的基础"""
    visible = _strip_html_comments(text)
    return [ln.strip() for ln in visible.splitlines() if ln.strip()]


def diff_memory_lines(old_text: str, new_text: str) -> tuple[list[str], list[str]]:
    """
    比较写入前后文本的有效内容行，供审批界面渲染红删绿增预览。

    基于 _content_lines 比较（HTML 注释与空行不计入，与注入逻辑一致），
    模板提示注释被清理不影响 diff。

    Returns:
        (新增行列表, 删除行列表)
    """
    old_lines = _content_lines(old_text)
    new_lines = _content_lines(new_text)
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)

    added: list[str] = []
    removed: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            removed.extend(old_lines[i1:i2])
            added.extend(new_lines[j1:j2])
        elif tag == "delete":
            removed.extend(old_lines[i1:i2])
        elif tag == "insert":
            added.extend(new_lines[j1:j2])
    return added, removed
