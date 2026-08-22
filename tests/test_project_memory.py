"""
Tests for chcode/utils/project_memory.py — CHCODE.md 项目记忆系统。

覆盖：模板生成、首次创建、CLAUDE.md 迁移、内容读取与缓存失效、
条目写入（append/replace/新建节/超长拒绝）、update_memory 工具、
HITL 注册与 load_skills 提示词注入。
"""

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from chcode.utils.project_memory import (
    MAX_ENTRY_CHARS,
    begin_memory_session,
    build_memory_reminder,
    check_memory_changed,
    diff_memory_lines,
    ensure_project_memory,
    get_session_memory,
    get_template,
    load_memory_content,
    preview_memory_entry,
    reset_memory_cache,
    reset_memory_session,
    save_memory_entry,
)


@pytest.fixture(autouse=True)
def _reset_memory_session_state():
    """会话冻结块/轮询基线是模块级全局状态，测试间必须清零"""
    reset_memory_cache()
    yield
    reset_memory_cache()


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


# ============================================================================
# get_template
# ============================================================================


class TestGetTemplate:
    def test_zh_template_sections(self):
        tpl = get_template("zh")
        assert "## 项目概览" in tpl
        assert "## 禁止事项" in tpl
        assert "## 踩过的坑" in tpl

    def test_en_template_sections(self):
        tpl = get_template("en")
        assert "## Project Overview" in tpl
        assert "## Prohibitions" in tpl
        assert "## Pitfalls" in tpl

    def test_template_is_effectively_empty(self):
        """模板只有注释和标题，无有效内容，load 应返回 None"""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write(d / "CHCODE.md", get_template("zh"))
            assert load_memory_content(d) is None


# ============================================================================
# ensure_project_memory — 创建/迁移/保持
# ============================================================================


class TestEnsureProjectMemory:
    def test_created_fresh_dir(self, tmp_path):
        assert ensure_project_memory(tmp_path) == "created"
        f = tmp_path / "CHCODE.md"
        assert f.exists()
        assert "## 项目概览" in f.read_text(encoding="utf-8")

    def test_migrated_when_missing(self, tmp_path):
        _write(tmp_path / "CLAUDE.md", "# Legacy\n\n- old rule\n")
        assert ensure_project_memory(tmp_path) == "migrated"
        text = (tmp_path / "CHCODE.md").read_text(encoding="utf-8")
        assert "<!-- migrated from CLAUDE.md -->" in text
        assert "# Legacy" in text
        assert "- old rule" in text

    def test_migrated_when_effectively_empty(self, tmp_path):
        _write(tmp_path / "CHCODE.md", "<!-- chcode:template v1 -->\n\n# CHCODE.md\n")
        _write(tmp_path / "CLAUDE.md", "# Legacy\n\nreal content\n")
        assert ensure_project_memory(tmp_path) == "migrated"
        text = (tmp_path / "CHCODE.md").read_text(encoding="utf-8")
        assert "real content" in text

    def test_keeps_claude_md_file(self, tmp_path):
        legacy_text = "# Legacy\n\n- keep me\n"
        _write(tmp_path / "CLAUDE.md", legacy_text)
        ensure_project_memory(tmp_path)
        # 原文件保留且内容不变
        assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == legacy_text

    def test_exists_when_has_content(self, tmp_path):
        _write(tmp_path / "CHCODE.md", "# Mine\n\n- my rule\n")
        _write(tmp_path / "CLAUDE.md", "# Legacy\n")
        assert ensure_project_memory(tmp_path) == "exists"
        # CHCODE.md 优先：不迁移 CLAUDE.md 内容
        text = (tmp_path / "CHCODE.md").read_text(encoding="utf-8")
        assert "Legacy" not in text

    def test_exists_when_empty_and_no_legacy(self, tmp_path):
        """用户主动清空的文件保持原样，不重建模板"""
        _write(tmp_path / "CHCODE.md", "")
        assert ensure_project_memory(tmp_path) == "exists"
        assert (tmp_path / "CHCODE.md").read_text(encoding="utf-8") == ""

    def test_gbk_encoded_memory_does_not_crash(self, tmp_path):
        """非 UTF-8 编码的 CHCODE.md：不崩溃、不覆盖写（启动路径防崩）"""
        gbk = "项目概览：中文内容".encode("gbk")
        (tmp_path / "CHCODE.md").write_bytes(gbk)
        assert ensure_project_memory(tmp_path) == "exists"
        # 原字节原样保留，未被覆盖为模板
        assert (tmp_path / "CHCODE.md").read_bytes() == gbk

    def test_gbk_encoded_legacy_not_migrated(self, tmp_path):
        """非 UTF-8 编码的 CLAUDE.md 视为不可读：跳过迁移、走模板创建"""
        (tmp_path / "CLAUDE.md").write_bytes("项目规则".encode("gbk"))
        assert ensure_project_memory(tmp_path) == "created"
        # 原文件保留不动
        assert (tmp_path / "CLAUDE.md").read_bytes() == "项目规则".encode("gbk")


# ============================================================================
# load_memory_content — 读取
# ============================================================================


class TestLoadMemoryContent:
    def test_missing_returns_none(self, tmp_path):
        assert load_memory_content(tmp_path) is None

    def test_content_with_comments_stripped(self, tmp_path):
        _write(
            tmp_path / "CHCODE.md",
            "<!-- comment -->\n\n# Title\n\n## 节\n\n- entry\n<!-- hint -->\n",
        )
        content = load_memory_content(tmp_path)
        assert content is not None
        assert "comment" not in content
        assert "hint" not in content
        assert "- entry" in content


# ============================================================================
# save_memory_entry — 写入
# ============================================================================


class TestSaveMemoryEntry:
    def test_append_to_canonical_zh_section(self, tmp_path):
        ensure_project_memory(tmp_path)
        save_memory_entry(
            tmp_path, "常用命令", "NEVER use pip; MUST use uv to add deps."
        )
        text = (tmp_path / "CHCODE.md").read_text(encoding="utf-8")
        idx_cmd = text.index("## 常用命令")
        idx_next = text.index("## 编码规范")
        assert "- NEVER use pip; MUST use uv to add deps." in text[idx_cmd:idx_next]

    def test_append_en_name_maps_to_zh_section(self, tmp_path):
        ensure_project_memory(tmp_path)
        save_memory_entry(
            tmp_path, "Common Commands", "ALWAYS run tests before commit."
        )
        text = (tmp_path / "CHCODE.md").read_text(encoding="utf-8")
        # 不应新建英文节，条目落入中文节
        assert "## Common Commands" not in text
        assert "- ALWAYS run tests before commit." in text

    def test_append_creates_custom_section(self, tmp_path):
        ensure_project_memory(tmp_path)
        save_memory_entry(tmp_path, "部署注意", "MUST stop service before upgrade.")
        text = (tmp_path / "CHCODE.md").read_text(encoding="utf-8")
        assert "## 部署注意" in text
        assert "- MUST stop service before upgrade." in text

    def test_new_canonical_section_uses_requested_name(self, tmp_path):
        """新建规范节用请求原名（写入即定死），不随运行时语言转换"""
        from chcode.i18n import set_language

        _write(tmp_path / "CHCODE.md", "# T\n\n## 项目概览\n\n- a\n")
        set_language("en")  # 运行时英文也不把节名转成 Pitfalls
        save_memory_entry(tmp_path, "踩过的坑", "rule")
        text = (tmp_path / "CHCODE.md").read_text(encoding="utf-8")
        assert "## 踩过的坑" in text
        assert "## Pitfalls" not in text

    def test_append_to_existing_custom_header(self, tmp_path):
        """迁移自 CLAUDE.md 的自定义节头也能命中"""
        _write(
            tmp_path / "CHCODE.md", "# Proj\n\n## Guidelines\n\n- rule A\n\n## Other\n"
        )
        save_memory_entry(tmp_path, "Guidelines", "NEVER push to main directly.")
        text = (tmp_path / "CHCODE.md").read_text(encoding="utf-8")
        idx = text.index("## Guidelines")
        nxt = text.index("## Other")
        assert "- NEVER push to main directly." in text[idx:nxt]

    def test_append_multiple_entries_order(self, tmp_path):
        ensure_project_memory(tmp_path)
        save_memory_entry(tmp_path, "禁止事项", "NEVER edit generated files.")
        save_memory_entry(tmp_path, "禁止事项", "NEVER commit secrets.")
        text = (tmp_path / "CHCODE.md").read_text(encoding="utf-8")
        assert text.index("NEVER edit generated files.") < text.index(
            "NEVER commit secrets."
        )

    def test_replace_section_body(self, tmp_path):
        ensure_project_memory(tmp_path)
        save_memory_entry(tmp_path, "验证流程", "old workflow")
        save_memory_entry(
            tmp_path, "验证流程", "MUST run: uv run pytest", mode="replace"
        )
        content = load_memory_content(tmp_path)
        assert "old workflow" not in content
        assert "MUST run: uv run pytest" in content

    def test_append_multiline_entry_indented(self, tmp_path):
        ensure_project_memory(tmp_path)
        save_memory_entry(tmp_path, "踩过的坑", "line one\nline two")
        text = (tmp_path / "CHCODE.md").read_text(encoding="utf-8")
        assert "- line one\n  line two" in text

    def test_rejects_long_entry(self, tmp_path):
        with pytest.raises(ValueError, match="too long"):
            save_memory_entry(tmp_path, "禁止事项", "x" * (MAX_ENTRY_CHARS + 1))

    def test_rejects_empty_section_or_content(self, tmp_path):
        with pytest.raises(ValueError, match="Section"):
            save_memory_entry(tmp_path, "  ", "content")
        with pytest.raises(ValueError, match="Content"):
            save_memory_entry(tmp_path, "禁止事项", "  ")

    def test_rejects_invalid_mode(self, tmp_path):
        with pytest.raises(ValueError, match="mode"):
            save_memory_entry(tmp_path, "禁止事项", "content", mode="delete")

    def test_creates_file_when_missing(self, tmp_path):
        """未调用 ensure 直接写入时自动建文件"""
        save_memory_entry(tmp_path, "项目概览", "A terminal AI coding agent.")
        assert (tmp_path / "CHCODE.md").exists()
        assert "- A terminal AI coding agent." in (tmp_path / "CHCODE.md").read_text(
            encoding="utf-8"
        )

    def test_gbk_encoded_memory_raises_value_error(self, tmp_path):
        """非 UTF-8 编码的 CHCODE.md：工具层拿到 ValueError 而非裸解码异常"""
        (tmp_path / "CHCODE.md").write_bytes("旧内容".encode("gbk"))
        with pytest.raises(ValueError, match="Cannot read"):
            save_memory_entry(tmp_path, "常用命令", "NEVER use pip.")


# ============================================================================
# 元消息注入（build_memory_reminder）与会话冻结（get_session_memory）
# ============================================================================


class TestBuildMemoryReminder:
    def test_reminder_wraps_content_in_system_reminder(self, tmp_path):
        save_memory_entry(tmp_path, "禁止事项", "NEVER use pip; use uv.")
        reminder = build_memory_reminder(tmp_path)
        assert reminder.startswith("<system-reminder>")
        assert reminder.rstrip().endswith("</system-reminder>")
        assert "# project_memory" in reminder
        assert "NEVER use pip; use uv." in reminder
        assert "may or may not apply" in reminder

    def test_empty_memory_hint(self, tmp_path):
        ensure_project_memory(tmp_path)
        assert "Currently empty" in build_memory_reminder(tmp_path)


class TestSessionFreeze:
    def test_frozen_within_session_despite_external_edit(self, tmp_path):
        """会话内冻结：外部编辑后冻结块不变（保提示词前缀缓存）"""
        save_memory_entry(tmp_path, "常用命令", "old entry")
        frozen = get_session_memory(tmp_path)
        assert "old entry" in frozen
        # 外部编辑
        _write(
            tmp_path / "CHCODE.md",
            (tmp_path / "CHCODE.md").read_text(encoding="utf-8") + "- new entry\n",
        )
        assert get_session_memory(tmp_path) == frozen  # 仍是旧内容

    def test_reset_memory_cache_rereads(self, tmp_path):
        """失效事件后全量重读"""
        save_memory_entry(tmp_path, "常用命令", "old entry")
        get_session_memory(tmp_path)
        save_memory_entry(tmp_path, "常用命令", "new entry")
        reset_memory_cache()
        assert "new entry" in get_session_memory(tmp_path)


# ============================================================================
# 生命周期门面（begin_memory_session / reset_memory_session）
# ============================================================================


class TestBeginMemorySession:
    def test_created_seeds_freeze_and_baseline(self, tmp_path):
        """全新目录：创建模板、冻结块可读、轮询基线已建立"""
        status = begin_memory_session(tmp_path)
        assert status == "created"
        assert (tmp_path / "CHCODE.md").exists()
        assert get_session_memory(tmp_path) is None  # 纯模板 = 空
        assert check_memory_changed(tmp_path) is None  # 基线已建立，无提醒

    def test_migrated_from_claude_md(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("- 使用 uv", encoding="utf-8")
        assert begin_memory_session(tmp_path) == "migrated"
        assert "使用 uv" in get_session_memory(tmp_path)

    def test_exists_rereads_external_change(self, tmp_path):
        """外部改写后重启会话：状态 exists，冻结块反映新内容"""
        begin_memory_session(tmp_path)
        assert get_session_memory(tmp_path) is None  # 冻结空模板
        _write(
            tmp_path / "CHCODE.md", "# CHCODE.md\n\n## 常用命令\n\n- pytest tests/\n"
        )
        assert begin_memory_session(tmp_path) == "exists"
        assert "pytest tests/" in get_session_memory(tmp_path)


class TestResetMemorySession:
    def test_clears_frozen_block(self, tmp_path):
        """冻结块失效后重读磁盘新内容"""
        begin_memory_session(tmp_path)
        assert get_session_memory(tmp_path) is None  # 首次访问冻结空模板
        _write(
            tmp_path / "CHCODE.md", "# CHCODE.md\n\n## 常用命令\n\n- pytest tests/\n"
        )
        assert get_session_memory(tmp_path) is None  # 会话内冻结：仍是旧内容
        reset_memory_session(tmp_path)
        assert "pytest tests/" in get_session_memory(tmp_path)

    def test_does_not_recreate_deleted_file(self, tmp_path):
        """不 ensure：用户删掉 CHCODE.md 后不被静默重建模板"""
        begin_memory_session(tmp_path)
        (tmp_path / "CHCODE.md").unlink()
        reset_memory_session(tmp_path)
        assert not (tmp_path / "CHCODE.md").exists()

    def test_reseeds_polling_baseline(self, tmp_path):
        """基线重建后，外部变更不再被报告为提醒"""
        begin_memory_session(tmp_path)
        _write(tmp_path / "CHCODE.md", "# CHCODE.md\n\n## 常用命令\n\n- changed\n")
        reset_memory_session(tmp_path)
        assert check_memory_changed(tmp_path) is None


# ============================================================================
# 外部修改轮询检测（check_memory_changed）
# ============================================================================


class TestCheckMemoryChanged:
    def test_first_call_seeds_silently(self, tmp_path):
        """无基线时静默建立，不产生提醒"""
        save_memory_entry(tmp_path, "常用命令", "entry")
        assert check_memory_changed(tmp_path) is None

    def test_external_edit_produces_diff_reminder(self, tmp_path):
        save_memory_entry(tmp_path, "常用命令", "old rule")
        check_memory_changed(tmp_path)  # 建立基线
        time.sleep(0.01)  # 确保外部写入落在新的 mtime 刻度
        # 外部编辑
        _write(
            tmp_path / "CHCODE.md",
            (tmp_path / "CHCODE.md")
            .read_text(encoding="utf-8")
            .replace("- old rule", "- new rule"),
        )
        note = check_memory_changed(tmp_path)
        assert note is not None
        assert "was changed outside chcode" in note
        # 统一 diff 行："-- old rule"（删除）与 "+- new rule"（新增）
        assert "-- old rule" in note
        assert "+- new rule" in note
        # 基线已更新：再次检查无提醒
        assert check_memory_changed(tmp_path) is None

    def test_touched_but_unchanged_no_reminder(self, tmp_path):
        save_memory_entry(tmp_path, "常用命令", "stable rule")
        check_memory_changed(tmp_path)
        text = (tmp_path / "CHCODE.md").read_text(encoding="utf-8")
        _write(tmp_path / "CHCODE.md", text)  # 重写同样内容（mtime 变了）
        assert check_memory_changed(tmp_path) is None

    def test_tool_write_does_not_trigger_reminder(self, tmp_path):
        """update_memory 自己的写入刷新基线，不算外部修改"""
        save_memory_entry(tmp_path, "常用命令", "first")
        check_memory_changed(tmp_path)  # 建立基线
        save_memory_entry(tmp_path, "常用命令", "second")  # 工具写入
        assert check_memory_changed(tmp_path) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert check_memory_changed(tmp_path) is None


# ============================================================================
# update_memory 工具
# ============================================================================


def _make_runtime(**kwargs):
    rt = MagicMock()
    ctx = MagicMock()
    for k, v in kwargs.items():
        setattr(ctx, k, v)
    rt.context = ctx
    return rt


class TestUpdateMemoryTool:
    async def test_ok_write(self, tmp_path):
        from chcode.utils.tools import update_memory

        rt = _make_runtime(working_directory=tmp_path)
        with patch("chcode.utils.tools.render_tool_call"):
            out = await update_memory.coroutine(
                "常用命令", "NEVER use pip; use uv.", runtime=rt
            )
        assert out.startswith("update_memory:\n[OK]")
        assert "- NEVER use pip; use uv." in (tmp_path / "CHCODE.md").read_text(
            encoding="utf-8"
        )

    async def test_failed_on_long_entry(self, tmp_path):
        from chcode.utils.tools import update_memory

        rt = _make_runtime(working_directory=tmp_path)
        with patch("chcode.utils.tools.render_tool_call"):
            out = await update_memory.coroutine(
                "禁止事项", "x" * (MAX_ENTRY_CHARS + 1), runtime=rt
            )
        assert out.startswith("update_memory:\n[FAILED]")
        assert "too long" in out


# ============================================================================
# 容量控制 — 软提醒 / 硬截断注入 / append 节流
# ============================================================================


class TestCapacityControl:
    def _fill(self, tmp_path: Path, line: str, count: int) -> None:
        _write(tmp_path / "CHCODE.md", "# T\n\n## 节\n\n" + line * count)

    def test_under_limits_no_directive(self, tmp_path):
        self._fill(tmp_path, "- x\n", 100)  # ~400 chars
        block = get_session_memory(tmp_path)
        assert "capacity limit" not in block
        assert "TRUNCATED" not in block

    def test_soft_limit_appends_cleanup_hint(self, tmp_path):
        self._fill(tmp_path, "- x\n", 2600)  # ~10K：介于软/硬之间
        block = get_session_memory(tmp_path)
        assert "approaching the capacity limit" in block
        assert "TRUNCATED" not in block

    def test_hard_limit_truncates_injection_only(self, tmp_path):
        self._fill(tmp_path, "- " + "y" * 40 + "\n", 500)  # ~21.5K
        original = (tmp_path / "CHCODE.md").read_text(encoding="utf-8")
        block = get_session_memory(tmp_path)
        assert "TRUNCATED" in block
        # 只截断注入，磁盘文件不动
        assert (tmp_path / "CHCODE.md").read_text(encoding="utf-8") == original

    def test_append_blocked_over_hard_limit(self, tmp_path):
        self._fill(tmp_path, "- " + "y" * 80 + "\n", 300)  # ~25K
        with pytest.raises(ValueError, match="capacity"):
            save_memory_entry(tmp_path, "踩过的坑", "new entry")

    def test_replace_still_allowed_over_hard_limit(self, tmp_path):
        """replace 是清理通道，超限时也不受限"""
        self._fill(tmp_path, "- " + "y" * 80 + "\n", 300)  # ~25K
        save_memory_entry(tmp_path, "节", "MUST keep it short.", mode="replace")
        text = (tmp_path / "CHCODE.md").read_text(encoding="utf-8")
        assert "MUST keep it short." in text
        assert "yyyy" not in text  # 旧冗长正文已被整体替换

    async def test_tool_reports_capacity_failure(self, tmp_path):
        from chcode.utils.tools import update_memory

        self._fill(tmp_path, "- " + "y" * 80 + "\n", 300)  # ~25K
        rt = _make_runtime(working_directory=tmp_path)
        with patch("chcode.utils.tools.render_tool_call"):
            out = await update_memory.coroutine("踩过的坑", "new", runtime=rt)
        assert out.startswith("update_memory:\n[FAILED]")
        assert "capacity" in out


# ============================================================================
# 预演（preview_memory_entry）与行级 diff（diff_memory_lines）
# ============================================================================


class TestPreviewMemoryEntry:
    def test_preview_never_writes(self, tmp_path):
        ensure_project_memory(tmp_path)
        f = tmp_path / "CHCODE.md"
        before = f.read_text(encoding="utf-8")
        old, new = preview_memory_entry(tmp_path, "常用命令", "NEVER use pip.")
        assert f.read_text(encoding="utf-8") == before  # 磁盘未动
        assert "- NEVER use pip." not in old
        assert "- NEVER use pip." in new

    def test_preview_matches_actual_save(self, tmp_path):
        ensure_project_memory(tmp_path)
        _, previewed = preview_memory_entry(tmp_path, "踩过的坑", "MUST handle CRLF.")
        save_memory_entry(tmp_path, "踩过的坑", "MUST handle CRLF.")
        assert (tmp_path / "CHCODE.md").read_text(encoding="utf-8") == previewed

    def test_preview_on_missing_file_mirrors_ensure(self, tmp_path):
        """文件不存在时，预演基于 ensure 将生成的文本（模板）计算"""
        _, new = preview_memory_entry(tmp_path, "项目概览", "A CLI tool.")
        assert "## 项目概览" in new  # 模板被纳入预演
        assert "- A CLI tool." in new
        assert not (tmp_path / "CHCODE.md").exists()  # 仍未落盘

    def test_preview_validates_like_save(self, tmp_path):
        with pytest.raises(ValueError, match="too long"):
            preview_memory_entry(tmp_path, "禁止事项", "x" * (MAX_ENTRY_CHARS + 1))

    def test_preview_capacity_throttle(self, tmp_path):
        _write(
            tmp_path / "CHCODE.md", "# T\n\n## 节\n\n" + ("- " + "y" * 80 + "\n") * 300
        )
        with pytest.raises(ValueError, match="capacity"):
            preview_memory_entry(tmp_path, "踩过的坑", "new entry")


class TestDiffMemoryLines:
    def test_pure_addition(self):
        added, removed = diff_memory_lines(
            "# T\n\n## 节\n\n- a\n", "# T\n\n## 节\n\n- a\n- b\n"
        )
        assert added == ["- b"]
        assert removed == []

    def test_modified_line(self):
        added, removed = diff_memory_lines("- run pytest\n", "- run uv run pytest\n")
        assert added == ["- run uv run pytest"]
        assert removed == ["- run pytest"]

    def test_pure_removal(self):
        added, removed = diff_memory_lines("- a\n- b\n", "- a\n")
        assert added == []
        assert removed == ["- b"]

    def test_identical(self):
        assert diff_memory_lines("# T\n\n- a\n", "# T\n\n- a\n") == ([], [])

    def test_html_comments_ignored(self):
        """模板提示注释（HTML 注释）不计入 diff"""
        added, removed = diff_memory_lines(
            "<!-- hint -->\n- a\n", "<!-- other hint -->\n- a\n"
        )
        assert (added, removed) == ([], [])

    def test_append_flow_via_preview(self, tmp_path):
        """端到端：模板上 append 的预演结果只有新增行"""
        ensure_project_memory(tmp_path)
        old, new = preview_memory_entry(tmp_path, "常用命令", "NEVER use pip.")
        assert diff_memory_lines(old, new) == (["- NEVER use pip."], [])

    def test_replace_flow_via_preview(self, tmp_path):
        """端到端：replace 为不同内容 → 旧行进删除列、新行进新增列"""
        save_memory_entry(tmp_path, "验证流程", "rule one")
        old, new = preview_memory_entry(
            tmp_path, "验证流程", "- MUST run uv run pytest", mode="replace"
        )
        assert diff_memory_lines(old, new) == (
            ["- MUST run uv run pytest"],
            ["- rule one"],
        )


# ============================================================================
# HITL 注册与提示词注入
# ============================================================================


class TestInterruptRegistration:
    def test_update_memory_needs_approval(self):
        from chcode.agent_setup import _build_interrupt_on

        result = _build_interrupt_on(False)
        assert "update_memory" in result

    def test_yolo_still_empty(self):
        from chcode.agent_setup import _build_interrupt_on

        assert _build_interrupt_on(True) == {}

    def test_readonly_subagents_cannot_update_memory(self):
        """只读子代理（Explore/Plan）禁用 update_memory — 遵守其
        READ-ONLY 契约（子代理链无 HITL，不禁用即可无审批写入）"""
        from chcode.agents.definitions import BUILT_IN_AGENTS
        from chcode.agents.runner import _resolve_tools
        from chcode.utils.tools import ALL_TOOLS

        for name in ("Explore", "Plan"):
            tools = _resolve_tools(BUILT_IN_AGENTS[name], ALL_TOOLS)
            names = [getattr(t, "name", "") for t in tools]
            assert "update_memory" not in names
            # 写工具同被禁用，读工具保留（回归既有契约）
            assert "write_file" not in names
            assert "edit" not in names
            assert "read_file" in names


class TestLoadSkillsPrompt:
    async def test_system_prompt_keeps_only_static_guidance(self, tmp_path):
        """记忆内容不进 system prompt，只留静态维护指引"""
        from chcode.agent_setup import load_skills

        save_memory_entry(tmp_path, "常用命令", "NEVER use pip; use uv.")
        mock_loader = MagicMock()
        mock_loader.build_system_prompt = MagicMock(return_value="prompt")

        mock_request = MagicMock()
        mock_request.runtime.context.skill_loader = mock_loader
        mock_request.runtime.context.working_directory = tmp_path
        mock_request.runtime.context.model_config = {"model": "glm-5"}
        mock_request.runtime.context.yolo = False

        handler = AsyncMock(return_value="model response")

        with patch("chcode.agent_setup.sys.platform", "linux"):
            await load_skills.awrap_model_call(mock_request, handler)

        base_prompt = mock_loader.build_system_prompt.call_args[0][0]
        # 静态维护指引与工具登记仍在
        assert "Project Memory (CHCODE.md)" in base_prompt
        assert "update_memory:" in base_prompt
        # 记忆内容本身不在 system prompt
        assert "NEVER use pip; use uv." not in base_prompt
        assert "# project_memory" not in base_prompt


class TestInjectProjectMemoryMiddleware:
    async def _make_request(self, tmp_path):
        request = MagicMock()
        request.runtime.context.working_directory = tmp_path
        request.messages = [HumanMessage(content="hi")]
        return request

    async def test_prepends_reminder_message(self, tmp_path):
        """每次模型调用：记忆块作为 <system-reminder> 元消息前置到最前"""
        from chcode.agent_setup import inject_project_memory

        save_memory_entry(tmp_path, "禁止事项", "NEVER commit secrets.")
        request = await self._make_request(tmp_path)
        handler = AsyncMock(return_value="resp")

        await inject_project_memory.awrap_model_call(request, handler)

        request.override.assert_called_once()
        messages = request.override.call_args.kwargs["messages"]
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content.startswith("<system-reminder>")
        assert "# project_memory" in messages[0].content
        assert "NEVER commit secrets." in messages[0].content
        # 原消息保留在后
        assert messages[-1].content == "hi"
        handler.assert_awaited_once()

    async def test_no_change_note_even_after_external_edit(self, tmp_path):
        """diff 提醒已移交回合边界落库（chat._process_input），中间件只前置冻结块"""
        from chcode.agent_setup import inject_project_memory

        save_memory_entry(tmp_path, "常用命令", "old rule")
        check_memory_changed(tmp_path)  # 建立基线
        build_memory_reminder(tmp_path)  # 冻结会话块
        time.sleep(0.01)
        # 外部编辑
        _write(
            tmp_path / "CHCODE.md",
            (tmp_path / "CHCODE.md")
            .read_text(encoding="utf-8")
            .replace("- old rule", "- new rule"),
        )
        request = await self._make_request(tmp_path)
        handler = AsyncMock(return_value="resp")

        await inject_project_memory.awrap_model_call(request, handler)

        messages = request.override.call_args.kwargs["messages"]
        assert len(messages) == 2  # 只有前置块 + 原消息，无变更提醒
        assert messages[-1].content == "hi"

    async def test_no_change_note_without_external_edit(self, tmp_path):
        from chcode.agent_setup import inject_project_memory

        save_memory_entry(tmp_path, "常用命令", "rule")
        check_memory_changed(tmp_path)  # 建立基线
        request = await self._make_request(tmp_path)
        handler = AsyncMock(return_value="resp")

        await inject_project_memory.awrap_model_call(request, handler)

        messages = request.override.call_args.kwargs["messages"]
        assert len(messages) == 2  # 只有前置块 + 原消息，无变更提醒


class TestSubagentMemoryInjection:
    async def test_subagent_gets_frozen_block_only(self, tmp_path):
        """子代理注入冻结块；其消息流不带 memory_note，展开天然不生效"""
        from chcode.agent_setup import inject_project_memory

        save_memory_entry(tmp_path, "禁止事项", "NEVER commit secrets.")
        check_memory_changed(tmp_path)  # 主代理视角的基线
        # 外部编辑 —— 轮询只在主循环发生，子代理消息不携带变更提醒
        _write(
            tmp_path / "CHCODE.md",
            (tmp_path / "CHCODE.md").read_text(encoding="utf-8") + "- extra\n",
        )
        request = MagicMock()
        request.runtime.context.working_directory = tmp_path
        request.messages = [HumanMessage(content="sub task")]
        handler = AsyncMock(return_value="resp")

        await inject_project_memory.awrap_model_call(request, handler)

        messages = request.override.call_args.kwargs["messages"]
        assert len(messages) == 2  # 前置块 + 原消息，无变更提醒
        assert "# project_memory" in messages[0].content

    async def test_subagent_update_memory_summary_shown(self):
        """单 agent 模式下子代理的 update_memory 调用显示缩进摘要（section 键）"""
        from langchain_core.messages import ToolMessage

        from chcode.agents.runner import _display_subagent_tools
        import chcode.display as display_mod

        request = MagicMock()
        request.tool_call = {
            "name": "update_memory",
            "args": {"section": "常用命令", "content": "x", "mode": "append"},
            "id": "call_1",
        }
        handler = AsyncMock(
            return_value=ToolMessage(content="ok", tool_call_id="call_1")
        )

        with (
            patch.object(display_mod, "_subagent_count", 1),
            patch.object(display_mod, "_subagent_parallel", False),
            patch.object(display_mod, "console") as mock_console,
        ):
            await _display_subagent_tools.awrap_tool_call(request, handler)

        printed = "".join(str(c) for c in mock_console.print.call_args_list)
        assert "update_memory" in printed
        assert "常用命令" in printed
        handler.assert_awaited_once()


class TestSubagentMemoryNotesPassing:
    """主代理调用子代理时显式传入变更提醒（收集 + 系统提示词拼接）"""

    def test_collect_memory_notes_chronological(self):
        from chcode.utils.tools import _collect_memory_notes

        note1 = "<system-reminder>Note: first change</system-reminder>"
        note2 = "<system-reminder>Note: second change</system-reminder>"
        msgs = [
            HumanMessage(content="q1", additional_kwargs={"memory_note": note1}),
            HumanMessage(content="a1"),  # 无 note 的消息原样跳过
            HumanMessage(content="q2", additional_kwargs={"memory_note": note2}),
        ]
        assert _collect_memory_notes(msgs) == [note1, note2]

    def test_collect_memory_notes_empty_safe(self):
        from chcode.utils.tools import _collect_memory_notes

        assert _collect_memory_notes([]) == []
        assert _collect_memory_notes([HumanMessage(content="hi")]) == []

    def test_with_memory_notes_appends_in_order(self):
        from chcode.agents.runner import _with_memory_notes

        base = "You are Explore."
        assert _with_memory_notes(base, None) == base
        assert _with_memory_notes(base, []) == base
        combined = _with_memory_notes(base, ["n1", "n2"])
        assert combined.startswith(base)
        assert combined.index("n1") < combined.index("n2")


class TestSubagentEndToEndMemory:
    """端到端：create_agent 真实装配后，子代理模型实际收到记忆上下文。

    既有测试都是直接调中间件函数；本类走完整装配链路（create_agent +
    全部中间件），只替换模型为记录消息的假实现，防中间件注册/装配
    改动悄悄断掉注入。
    """

    async def test_subagent_receives_memory_and_notes(self, tmp_path):
        from chcode.agents import runner as runner_mod
        from chcode.agents.definitions import BUILT_IN_AGENTS
        from chcode.utils.skill_loader import SkillLoader

        save_memory_entry(tmp_path, "禁止事项", "VERIF-MARKER-42")

        captured: list[list] = []

        class FakeModel:
            def __init__(self, **kwargs):
                pass

            def bind_tools(self, tools, **kw):
                return self

            async def ainvoke(self, messages, config=None, **kw):
                captured.append(list(messages))
                return AIMessage(content="done")

            async def astream(self, messages, config=None, **kw):
                captured.append(list(messages))
                yield AIMessage(content="done")

        agent_def = BUILT_IN_AGENTS["Explore"]
        with patch.object(runner_mod, "EnhancedChatOpenAI", FakeModel):
            result, is_error = await runner_mod.run_subagent(
                prompt="just acknowledge",
                agent_def=agent_def,
                model_config={"model": "fake"},
                working_directory=tmp_path,
                skill_loader=SkillLoader([tmp_path]),
                timeout_seconds=300,
                description="verify",
                yolo=False,
                memory_notes=["<system-reminder>VERIF-NOTE-7</system-reminder>"],
            )

        assert is_error is False
        assert len(captured) == 1  # 假模型直接终答，单次调用
        msgs = captured[0]
        # 系统提示 = agent 定义提示词 + 变更提醒拼在末尾
        assert msgs[0].content.startswith(agent_def.system_prompt)
        assert "VERIF-NOTE-7" in msgs[0].content
        # 冻结块作为会话首条消息（<system-reminder> 包裹，含记忆内容）
        assert msgs[1].content.startswith("<system-reminder>")
        assert "# project_memory" in msgs[1].content
        assert "VERIF-MARKER-42" in msgs[1].content
        # 任务 prompt 原样透传
        assert msgs[2].content == "just acknowledge"


class TestRenderMemoryPreview:
    """审批界面的写入预览渲染：省略行数 = 两列表各自超出部分之和"""

    def test_omitted_counts_both_lists(self):
        from chcode.chat import ChatREPL

        repl = ChatREPL()
        with patch("chcode.chat.console.print") as mock_print:
            repl._render_memory_preview(
                added=[f"+ entry {i}" for i in range(60)],
                removed=[f"- entry {i}" for i in range(60)],
            )

        texts = [str(c.args[0]) for c in mock_print.call_args_list if c.args]
        # 50 + 50 行已显示，各自超出 10 行 → 省略 20 行（而非旧算法的 10）
        assert any("20" in t and "未显示" in t for t in texts)

    def test_no_omitted_line_when_under_limit(self):
        from chcode.chat import ChatREPL

        repl = ChatREPL()
        with patch("chcode.chat.console.print") as mock_print:
            repl._render_memory_preview(added=["- a", "- b"], removed=[])

        texts = [str(c.args[0]) for c in mock_print.call_args_list if c.args]
        assert not any("未显示" in t for t in texts)


class TestAttachMemoryNote:
    """回合边界检测的变更提醒：挂用户消息 metadata（渲染天然不可见）"""

    def test_string_input_attaches_note_to_user_message(self):
        from chcode.chat import _attach_memory_note

        result = _attach_memory_note(
            {"messages": "hello"}, "<system-reminder>Note...</system-reminder>"
        )
        msgs = result["messages"]
        assert len(msgs) == 1  # 不追加新消息，只挂 metadata
        assert msgs[0].content == "hello"  # 原文不动
        assert (
            msgs[0].additional_kwargs.get("memory_note")
            == "<system-reminder>Note...</system-reminder>"
        )

    def test_single_message_form_attaches_note(self):
        """多模态输入是单个 HumanMessage（非列表），也要能挂上备注"""
        from chcode.chat import _attach_memory_note

        mm = HumanMessage(content=[{"type": "text", "text": "look at this"}])
        result = _attach_memory_note({"messages": mm}, "note text")
        msgs = result["messages"]
        assert isinstance(msgs, HumanMessage)
        assert msgs.additional_kwargs.get("memory_note") == "note text"
        assert msgs.content == [{"type": "text", "text": "look at this"}]

    def test_list_input_attaches_to_last_human(self):
        from chcode.chat import _attach_memory_note

        q = HumanMessage(content="q")
        media = HumanMessage(content=[{"type": "text", "text": "img desc"}])
        result = _attach_memory_note({"messages": [q, media]}, "note text")
        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[0].additional_kwargs == {}  # 最早的消息不动
        assert msgs[1].additional_kwargs.get("memory_note") == "note text"
        assert msgs[1].content == [{"type": "text", "text": "img desc"}]

    def test_note_invisible_to_render(self, capsys):
        """渲染只读 content：metadata 里的备注天然不显示"""
        from chcode.display import render_conversation

        render_conversation(
            [
                HumanMessage(
                    content="VISIBLE-Msg",
                    additional_kwargs={"memory_note": "SECRET-NOTE"},
                )
            ]
        )
        out = capsys.readouterr().out
        assert "VISIBLE-Msg" in out
        assert "SECRET-NOTE" not in out


class TestMemoryNoteExpansion:
    async def test_metadata_note_expanded_into_content(self):
        """中间件把 memory_note metadata 展开拼进该消息 content（注入），
        原消息对象不受影响（剥离）"""
        from chcode.agent_setup import inject_project_memory

        note = "<system-reminder>Note: file was modified...</system-reminder>"
        original = HumanMessage(
            content="hello", additional_kwargs={"memory_note": note}
        )
        plain = HumanMessage(content="no note here")

        request = MagicMock()
        request.runtime.context.working_directory = None  # 无 CHCODE.md 也应正常
        request.messages = [original, plain]
        handler = AsyncMock(return_value="resp")

        await inject_project_memory.awrap_model_call(request, handler)

        messages = request.override.call_args.kwargs["messages"]
        # 前置冻结块 + 两条原消息，数量不变
        assert len(messages) == 3
        # 展开进 content
        assert messages[1].content == f"hello\n\n{note}"
        # 发送副本剥离 memory_note，不随 additional_kwargs 序列化进 API payload
        assert "memory_note" not in messages[1].additional_kwargs
        # 无备注的消息原样透传
        assert messages[2].content == "no note here"
        # 原消息对象未被修改（状态保持 metadata 形态）
        assert original.content == "hello"
        assert original.additional_kwargs.get("memory_note") == note
