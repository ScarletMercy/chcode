from pathlib import Path
from unittest.mock import patch

import pytest

from chcode.utils.tool_result_pipeline import (
    PREVIEW_MAX_CHARS,
    BudgetState,
    clean_tool_output,
    enforce_per_turn_budget,
    get_budget_state,
    reset_budget_state,
    _generate_preview,
    _persist_to_disk,
    truncate_large_result,
    _collect_tool_messages_by_turn,
    _select_to_replace,
)


class TestCleanToolOutput:
    def test_empty_pipeline(self):
        assert clean_tool_output("") == ""

    def test_strips_ansi(self):
        text = "\x1b[31mred\x1b[0m text"
        assert clean_tool_output(text) == "red text"

    def test_strips_osc(self):
        text = "\x1b]0;title\x07rest"
        assert clean_tool_output(text) == "rest"

    def test_preserves_angle_brackets(self):
        assert clean_tool_output("#include <stdio.h>") == "#include <stdio.h>"
        assert clean_tool_output("std::vector<int>") == "std::vector<int>"

    def test_list_input(self):
        assert clean_tool_output(["a", "b"]) == "a\nb"

    def test_non_string_input(self):
        assert clean_tool_output(42) == "42"


class TestGeneratePreview:
    def test_short_content(self):
        content, truncated = _generate_preview("short")
        assert content == "short"
        assert truncated is False

    def test_long_content(self):
        text = "a" * (PREVIEW_MAX_CHARS + 1000)
        content, truncated = _generate_preview(text)
        assert truncated is True
        assert len(content) <= PREVIEW_MAX_CHARS

    def test_long_content_with_newline(self):
        text = "line\n" * 1000
        content, truncated = _generate_preview(text)
        assert truncated is True


class TestTruncateLargeResult:
    def test_empty_string_pipeline(self):
        assert truncate_large_result("") == ""

    def test_whitespace_only(self):
        result = truncate_large_result("   \n  ", tool_name="test")
        assert "no output" in result

    def test_small_content_unchanged(self):
        assert truncate_large_result("hello") == "hello"

    def test_large_content_truncated(self, tmp_path: Path):
        content = "x" * 60_000
        result = truncate_large_result(content, "tool", "id1", workplace=tmp_path)
        assert "too large" in result.lower() or "truncated" in result.lower()

    def test_large_content_no_workplace(self):
        content = "x" * 60_000
        result = truncate_large_result(content, "tool", "id1", workplace=None)
        assert "too large" in result.lower() or "truncated" in result.lower()


class TestPersistToDisk:
    def test_creates_file_pipeline(self, tmp_path: Path):
        path = _persist_to_disk("content", "tool_1", tmp_path)
        assert path is not None
        assert Path(path).exists()

    def test_no_workplace(self):
        assert _persist_to_disk("content", "tool_1", None) is None

    def test_sanitizes_tool_id(self, tmp_path: Path):
        path = _persist_to_disk("content", "a/b:c", tmp_path)
        assert path is not None
        assert "_" in Path(path).name


class TestBudgetState:
    def test_reset(self):
        state = BudgetState()
        state.seen_ids.add("abc")
        state.replacements["abc"] = "replaced"
        state.reset()
        assert len(state.seen_ids) == 0
        assert len(state.replacements) == 0


class TestGetBudgetState:
    def test_singleton_pipeline(self):
        reset_budget_state()
        s1 = get_budget_state()
        s2 = get_budget_state()
        assert s1 is s2

    def test_reset_creates_new(self):
        s1 = get_budget_state()
        reset_budget_state()
        s2 = get_budget_state()
        assert s1 is not s2


class TestEnforcePerTurnBudget:
    def test_no_tool_messages_pipeline(self):
        from langchain_core.messages import HumanMessage

        msgs = [HumanMessage(content="hello")]
        result = enforce_per_turn_budget(msgs)
        assert result is msgs

    def test_with_tool_message(self, tmp_path: Path):
        from langchain_core.messages import AIMessage, ToolMessage

        reset_budget_state()
        tool_msg = ToolMessage(
            content="x" * 300_000,
            tool_call_id="tc1",
            name="bash",
        )
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "bash", "args": {}}],
        )
        msgs = [ai_msg, tool_msg]
        result = enforce_per_turn_budget(msgs, workplace=tmp_path)
        assert result[1].content != tool_msg.content


class TestPersistToDiskException:
    def test_exception_returns_none_pipeline(self, tmp_path: Path):
        """Lines 58-59: Exception in _persist_to_disk returns None."""
        # Create a directory that will fail on write
        result_dir = tmp_path / "readonly"
        result_dir.mkdir()

        with patch("pathlib.Path.write_text", side_effect=OSError("Permission denied")):
            result = _persist_to_disk("content", "tool_1", tmp_path)
            assert result is None


class TestCollectToolMessagesByTurn:
    def test_non_tool_message_ends_turn(self):
        """Lines 140, 145-147: Non-tool messages end current turn."""
        from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "bash", "args": {}}],
        )
        tool_msg = ToolMessage(content="output", tool_call_id="tc1", name="bash")
        human_msg = HumanMessage(content="next")

        turns = _collect_tool_messages_by_turn([ai_msg, tool_msg, human_msg])
        # Human message ends the current turn but doesn't start a new one
        # Only the AI+Tool turn is collected
        assert len(turns) == 1
        assert len(turns[0]) == 2
        assert isinstance(turns[0][0][1], AIMessage)
        assert isinstance(turns[0][1][1], ToolMessage)

    def test_only_human_messages(self):
        """Lines 145-147: Only human messages create no turns (no AI with tool_calls)."""
        from langchain_core.messages import HumanMessage

        msgs = [HumanMessage(content="one"), HumanMessage(content="two")]
        turns = _collect_tool_messages_by_turn(msgs)
        # No turns are collected because there's no AIMessage with tool_calls
        assert len(turns) == 0

    def test_consecutive_ai_tool_pairs(self):
        """Lines 138-142: Multiple AI+Tool pairs create separate turns."""
        from langchain_core.messages import AIMessage, ToolMessage

        ai1 = AIMessage(content="", tool_calls=[{"id": "tc1", "name": "bash", "args": {}}])
        tool1 = ToolMessage(content="out1", tool_call_id="tc1", name="bash")
        ai2 = AIMessage(content="", tool_calls=[{"id": "tc2", "name": "grep", "args": {}}])
        tool2 = ToolMessage(content="out2", tool_call_id="tc2", name="grep")

        turns = _collect_tool_messages_by_turn([ai1, tool1, ai2, tool2])
        # Should have two separate turns
        assert len(turns) == 2
        assert len(turns[0]) == 2  # ai1 + tool1
        assert len(turns[1]) == 2  # ai2 + tool2


class TestSelectToReplace:
    def test_no_deficit_returns_empty(self):
        """Line 162: When frozen_size + fresh_total <= limit, return empty list."""
        from langchain_core.messages import ToolMessage

        fresh = [(0, ToolMessage(content="small", tool_call_id="tc1"))]
        result = _select_to_replace(fresh, frozen_size=1000, limit=200000)
        assert result == []

    def test_all_content_fits(self):
        """Line 162: Content fits within budget."""
        from langchain_core.messages import ToolMessage

        fresh = [
            (0, ToolMessage(content="a" * 100, tool_call_id="tc1")),
            (1, ToolMessage(content="b" * 100, tool_call_id="tc2")),
        ]
        result = _select_to_replace(fresh, frozen_size=1000, limit=200000)
        assert result == []


class TestEnforceBudgetFrozenSize:
    def test_frozen_size_without_replacement(self, tmp_path: Path):
        """Lines 203-206: Frozen messages add to size without replacement."""
        from langchain_core.messages import AIMessage, ToolMessage

        reset_budget_state()
        state = BudgetState()
        state.seen_ids.add("tc1")

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "bash", "args": {}}],
        )
        frozen_msg = ToolMessage(content="x" * 1000, tool_call_id="tc1", name="bash")
        fresh_msg = ToolMessage(content="small", tool_call_id="tc2", name="bash")
        ai_msg2 = AIMessage(
            content="",
            tool_calls=[{"id": "tc2", "name": "bash", "args": {}}],
        )

        # High budget so fresh won't be replaced
        msgs = [ai_msg, frozen_msg, ai_msg2, fresh_msg]
        result = enforce_per_turn_budget(
            msgs, budget=50000, workplace=tmp_path, state=state
        )
        # Frozen message content should be preserved
        assert "x" * 1000 in result[1].content

    def test_no_fresh_messages_continues(self, tmp_path: Path):
        """Line 211: Continue when no fresh messages in turn."""
        from langchain_core.messages import AIMessage, ToolMessage

        reset_budget_state()
        state = BudgetState()
        state.seen_ids.add("tc1")
        state.replacements["tc1"] = "replaced"

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "bash", "args": {}}],
        )
        frozen_msg = ToolMessage(content="x" * 1000, tool_call_id="tc1", name="bash")

        msgs = [ai_msg, frozen_msg]
        result = enforce_per_turn_budget(
            msgs, budget=50000, workplace=tmp_path, state=state
        )
        assert result[1].content == "replaced"

    def test_non_selected_mark_as_seen(self, tmp_path: Path):
        """Line 217: Non-selected fresh items are marked as seen."""
        from langchain_core.messages import AIMessage, ToolMessage

        reset_budget_state()
        state = BudgetState()

        # Create messages where some will be selected and some won't
        small_msg = ToolMessage(content="x", tool_call_id="tc_small", name="bash")
        large_msg = ToolMessage(content="x" * 300000, tool_call_id="tc_large", name="bash")

        ai1 = AIMessage(content="", tool_calls=[{"id": "tc_small", "name": "bash", "args": {}}])
        ai2 = AIMessage(content="", tool_calls=[{"id": "tc_large", "name": "bash", "args": {}}])

        msgs = [ai1, small_msg, ai2, large_msg]
        result = enforce_per_turn_budget(
            msgs, budget=50000, workplace=tmp_path, state=state
        )
        # Small one should be marked as seen, large one replaced
        assert "tc_small" in state.seen_ids
        assert "tc_large" in state.seen_ids

    def test_no_replacement_returns_original(self, tmp_path: Path):
        """Line 233: Return original messages when no replacement_map."""
        from langchain_core.messages import HumanMessage

        reset_budget_state()
        msgs = [HumanMessage(content="hello")]
        result = enforce_per_turn_budget(msgs, workplace=tmp_path)
        assert result is msgs


class TestGetBudgetStateAfterReset:
    """Cover line 120: get_budget_state creates new BudgetState when _budget_state is None."""

    def test_creates_new_state_when_none(self):
        """Line 120: after setting _budget_state to None, get_budget_state creates a new one."""
        import chcode.utils.tool_result_pipeline as pipe_mod

        reset_budget_state()
        # Directly set the module-level variable to None to hit line 120
        pipe_mod._budget_state = None
        state = get_budget_state()
        assert state is not None
        assert isinstance(state, BudgetState)
