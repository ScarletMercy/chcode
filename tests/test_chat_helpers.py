from unittest.mock import MagicMock

from chcode.chat import (
    _collect_ids_from_group,
    _get_group_display,
    _group_messages_by_turn,
    _rich_to_html,
    find_and_slice_from_end,
)


class TestRichToHtml:
    def test_plain_text_helpers(self):
        assert _rich_to_html("hello") == "hello"

    def test_bold(self):
        assert _rich_to_html("[bold]text[/bold]") == "<b>text</b>"

    def test_italic(self):
        assert _rich_to_html("[italic]text[/italic]") == "<i>text</i>"

    def test_color(self):
        result = _rich_to_html("[red]error[/red]")
        assert '<style fg="red">' in result
        assert "</style>" in result

    def test_dim(self):
        result = _rich_to_html("[dim]faded[/dim]")
        assert '<style fg="#888888">' in result

    def test_nested(self):
        result = _rich_to_html("[bold][red]text[/red][/bold]")
        assert "<b>" in result
        assert '<style fg="red">' in result

    def test_no_tags_helpers(self):
        assert _rich_to_html("plain text") == "plain text"

    def test_unmapped_tag(self):
        assert _rich_to_html("[unknown]text[/unknown]") == "text"


class TestFindAndSliceFromEnd:
    def test_found(self):
        items = [MagicMock(type="a"), MagicMock(type="b"), MagicMock(type="a")]
        result = find_and_slice_from_end(items, "a")
        assert len(result) == 1
        assert result[0].type == "a"

    def test_not_found_helpers(self):
        items = [MagicMock(type="a"), MagicMock(type="b")]
        assert find_and_slice_from_end(items, "c") == []

    def test_empty_input_for_find_and_slice(self):
        assert find_and_slice_from_end([], "a") == []


class TestGroupMessagesByTurn:
    def _msg(self, msg_type, content="hi"):
        m = MagicMock()
        m.type = msg_type
        m.content = content
        m.id = f"id_{msg_type}_{content}"
        return m

    def test_single_human(self):
        msgs = [self._msg("human")]
        groups = _group_messages_by_turn(msgs)
        assert len(groups) == 1

    def test_two_turns_helpers(self):
        msgs = [
            self._msg("human", "q1"),
            self._msg("ai", "a1"),
            self._msg("human", "q2"),
            self._msg("ai", "a2"),
        ]
        groups = _group_messages_by_turn(msgs)
        assert len(groups) == 2
        assert groups[0][0].content == "q1"
        assert groups[1][0].content == "q2"

    def test_empty_helpers(self):
        assert _group_messages_by_turn([]) == []

    def test_injected_media_does_not_start_new_group(self):
        """vision 原生注入的媒体消息不是用户发言：不开启新组，归入当前组。"""
        from langchain_core.messages import AIMessage, HumanMessage

        from chcode.utils.multimodal import INJECTED_MEDIA_FLAG

        user = HumanMessage(content="看图", id="m-user")
        injected = HumanMessage(
            content=[{"type": "text", "text": "[vision] a.png"}],
            additional_kwargs={INJECTED_MEDIA_FLAG: True},
            id="m-inject",
        )
        ai = AIMessage(content="结论", id="m-ai")

        groups = _group_messages_by_turn([user, injected, ai])

        assert len(groups) == 1
        assert [m.id for m in groups[0]] == ["m-user", "m-inject", "m-ai"]


class TestGetGroupDisplay:
    def _msg(self, msg_type, content="hi"):
        m = MagicMock()
        m.type = msg_type
        m.content = content
        return m

    def test_with_human(self):
        group = [self._msg("human", "what is python?")]
        assert "what is python?" in _get_group_display(group)

    def test_long_content_truncated_helpers(self):
        group = [self._msg("human", "x" * 100)]
        display = _get_group_display(group)
        assert "..." in display
        assert len(display) <= 70

    def test_no_human(self):
        group = [self._msg("ai", "response")]
        assert _get_group_display(group) == "(空消息组)"


class TestCollectIdsFromGroup:
    def _msg(self, mid):
        m = MagicMock()
        m.id = mid
        return m

    def test_collects_from_index(self):
        groups = [
            [self._msg("a"), self._msg("b")],
            [self._msg("c"), self._msg("d")],
            [self._msg("e")],
        ]
        no_need, all_ids = _collect_ids_from_group(1, groups)
        assert "a" not in no_need
        assert "c" in no_need
        assert "e" in no_need
        assert len(all_ids) == 5

    def test_first_group(self):
        groups = [[self._msg("x")]]
        no_need, all_ids = _collect_ids_from_group(0, groups)
        assert "x" in no_need
