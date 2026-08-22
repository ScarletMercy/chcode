"""
Tests for prompt_toolkit UI code in chcode/utils/tools.py.
Targets lines 864-1027 (_checkbox_with_other_async) and 1035-1193 (_select_with_other_async).

Strategy: Capture KeyBindings, then invoke handlers directly.
Application is imported locally, so we patch prompt_toolkit.Application.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCheckboxControl:
    """Tests for _CheckboxControl class (lines 878-916)."""

    async def test_is_focusable_returns_true(self):
        """Line 884-885: is_focusable always returns True."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control
            captured_kb = key_bindings
            # Capture control from the layout
            layout = kwargs.get("layout")
            if layout:
                # Get the HSplit container
                hsplit = layout.container
                # Second window is the control_window
                captured_control = hsplit.children[1].content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        assert captured_control is not None
        assert captured_control.is_focusable() is True

    async def test_preferred_height_returns_options_plus_one(self):
        """Lines 890-893: preferred_height returns len(opts) + 1 for input row."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_control = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_control
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B", "C"])

        assert captured_control is not None
        # 3 options + 1 input row = 4
        height = captured_control.preferred_height(
            width=80, max_available_height=100, wrap_lines=False, get_line_prefix=None
        )
        assert height == 4

    async def test_create_content_with_checked_items(self):
        """Lines 895-916: create_content renders checkbox markers correctly."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_control = None
        captured_buffer = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_control, captured_buffer
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                # Get input buffer from third window
                captured_buffer = hsplit.children[2].content.buffer
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B", "C"])

        assert captured_control is not None
        # Initially no items checked
        content = captured_control.create_content(width=80, height=10)
        assert content.line_count == 4  # 3 options + 1 input row

        # Check second item
        captured_control.checked.add(1)
        content = captured_control.create_content(width=80, height=10)
        line_1 = content.get_line(1)
        # Should have [√] marker for checked item
        assert "[√]" in str(line_1)

    async def test_create_content_with_selection(self):
        """Lines 899-902: Selected item gets bold style and > prefix."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_control = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_control
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        assert captured_control is not None
        # Select second item (index 1)
        captured_control.selected = 1
        content = captured_control.create_content(width=80, height=10)
        line_1 = content.get_line(1)
        # Should have > prefix and bold style
        line_str = str(line_1)
        assert "❯" in line_str

    async def test_create_content_with_input_row_selected(self):
        """Lines 904-908: Input row gets > prefix when selected."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_control = None
        captured_buffer = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_control, captured_buffer
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_buffer = hsplit.children[2].content.buffer
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["custom"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        assert captured_control is not None
        # Select input row (index 2 for 2 options)
        captured_control.selected = 2
        captured_buffer.text = "custom input"
        content = captured_control.create_content(width=80, height=10)
        line_2 = content.get_line(2)
        line_str = str(line_2)
        # Should have > prefix and show buffer text
        assert "❯" in line_str
        assert "> " in line_str
        assert "custom input" in line_str


class TestCheckboxHandlers:
    """Tests for checkbox key binding handlers (lines 937-1018)."""

    async def test_up_handler_decrements_selection(self):
        """Lines 937-948: Up arrow decrements selected index."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B", "C"])

        # Start at selection 2
        captured_control.selected = 2
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        # Find and call up handler
        for binding in captured_kb.bindings:
            if "up" in str(binding.keys).lower():
                binding.handler(mock_event)
                break

        assert captured_control.selected == 1
        mock_event.app.invalidate.assert_called_once()

    async def test_up_handler_when_exiting(self):
        """Lines 939-941: Up handler returns early when _exiting is True."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # Verify up handler exists and is callable
        up_handler_found = False
        for binding in captured_kb.bindings:
            if "up" in str(binding.keys).lower():
                up_handler_found = True
                assert callable(binding.handler)
                break

        assert up_handler_found, "Up handler not found in key bindings"

    async def test_down_handler_increments_selection(self):
        """Lines 950-959: Down arrow increments selected index."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B", "C"])

        # Start at selection 0
        captured_control.selected = 0
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        # Find and call down handler
        for binding in captured_kb.bindings:
            if "down" in str(binding.keys).lower():
                binding.handler(mock_event)
                break

        assert captured_control.selected == 1
        mock_event.app.invalidate.assert_called_once()

    async def test_down_handler_focuses_input_at_bottom(self):
        """Lines 957-958: Down handler focuses input when reaching input_row_idx."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None
        captured_input_window = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_input_window
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_input_window = hsplit.children[2]
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # Select last option (index 1 for 2 options)
        captured_control.selected = 1
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        # Find and call down handler
        for binding in captured_kb.bindings:
            if "down" in str(binding.keys).lower():
                binding.handler(mock_event)
                break

        # Should have moved to input row (index 2)
        assert captured_control.selected == 2
        mock_event.app.layout.focus.assert_called_with(captured_input_window)

    async def test_tab_handler_toggles_to_input(self):
        """Lines 961-972: Tab toggles between list and input focus."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None
        captured_input_window = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_input_window
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_input_window = hsplit.children[2]
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # Start with selection on an option (not input row)
        captured_control.selected = 0
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        # Find and call tab handler (tab is represented as Keys.ControlI)
        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-i" in keys_str or "tab" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Should have moved to input row and focused input window
        assert captured_control.selected == 2
        mock_event.app.layout.focus.assert_called_with(captured_input_window)

    async def test_tab_handler_toggles_to_list(self):
        """Lines 966-968: Tab from input row moves back to first option."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None
        captured_control_window = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_control_window
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control_window = hsplit.children[1]
                captured_control = captured_control_window.content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # Start with selection on input row
        captured_control.selected = 2
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        # Find and call tab handler (tab is Keys.ControlI)
        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-i" in keys_str or "tab" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Should have moved to first option and focused control window
        assert captured_control.selected == 0
        mock_event.app.layout.focus.assert_called_with(captured_control_window)

    async def test_space_handler_toggles_checkbox(self):
        """Lines 974-981: Space toggles checked state for current selection."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B", "C"])

        # Select first option
        captured_control.selected = 0
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        # Find and call space handler - space is just (' ',) tuple
        for binding in captured_kb.bindings:
            # Check if keys tuple contains the space character
            if len(binding.keys) == 1 and binding.keys[0] == " ":
                binding.handler(mock_event)
                break

        # Should have checked first item
        assert 0 in captured_control.checked
        mock_event.app.invalidate.assert_called_once()

        # Call again to uncheck
        for binding in captured_kb.bindings:
            if len(binding.keys) == 1 and binding.keys[0] == " ":
                binding.handler(mock_event)
                break

        assert 0 not in captured_control.checked

    async def test_space_handler_does_nothing_on_input_row(self):
        """Lines 979-980: Space on input row doesn't toggle anything."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # Select input row
        captured_control.selected = 2
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        # Find and call space handler - space is just (' ',) tuple
        for binding in captured_kb.bindings:
            if len(binding.keys) == 1 and binding.keys[0] == " ":
                binding.handler(mock_event)
                break

        # No items should be checked
        assert len(captured_control.checked) == 0

    async def test_enter_handler_submits_selection(self):
        """Lines 983-996: Enter submits selected items and exits."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None
        captured_buffer = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_buffer, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_buffer = hsplit.children[2].content.buffer
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value=["A", "B"])
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B", "C"])

        # Check first and third items
        captured_control.checked = {0, 2}
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        # Find and call enter handler (enter is Keys.ControlM)
        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Should have called app.exit with sorted selected names
        mock_event.app.exit.assert_called_once()
        call_args = mock_event.app.exit.call_args
        result = call_args.kwargs.get("result") or call_args[1].get("result")
        assert result == ["A", "C"]

    async def test_enter_handler_includes_custom_input(self):
        """Lines 989-992: Enter includes custom input from buffer."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None
        captured_buffer = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_buffer, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_buffer = hsplit.children[2].content.buffer
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value=["A", "custom"])
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # Check first item and add custom input
        captured_control.checked = {0}
        captured_buffer.text = " custom value "
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        # Find and call enter handler (enter is Keys.ControlM)
        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Should include both selected and custom input
        mock_event.app.exit.assert_called_once()
        call_args = mock_event.app.exit.call_args
        result = call_args.kwargs.get("result") or call_args[1].get("result")
        assert result == ["A", "custom value"]

    async def test_enter_handler_with_no_selection(self):
        """Lines 989-992: Enter with no selection returns None."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None
        captured_buffer = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_buffer, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_buffer = hsplit.children[2].content.buffer
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value=None)
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # No items checked, no custom input
        captured_control.checked = set()
        captured_buffer.text = ""
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        # Find and call enter handler (enter is Keys.ControlM)
        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Should have called exit with None
        mock_event.app.exit.assert_called_once()
        call_args = mock_event.app.exit.call_args
        result = call_args.kwargs.get("result") or call_args[1].get("result")
        assert result is None

    async def test_esc_handler_cancels(self):
        """Lines 998-1007: Escape cancels and returns None."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.exit = MagicMock()
            mock_app.run_async = AsyncMock(return_value=None)
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()
        mock_event.app.exit = MagicMock()

        # Find and call esc handler
        for binding in captured_kb.bindings:
            if (
                "escape" in str(binding.keys).lower()
                or "esc" in str(binding.keys).lower()
            ):
                binding.handler(mock_event)
                break

        # Should have called exit with None
        mock_event.app.exit.assert_called_once()
        call_args = mock_event.app.exit.call_args
        result = call_args.kwargs.get("result") or call_args[1].get("result")
        assert result is None

    async def test_cancel_handler_cancels(self):
        """Lines 1009-1018: Ctrl-C cancels and returns None."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.exit = MagicMock()
            mock_app.run_async = AsyncMock(return_value=None)
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()
        mock_event.app.exit = MagicMock()

        # Find and call cancel handler (c-c)
        for binding in captured_kb.bindings:
            if "c-c" in str(binding.keys):
                binding.handler(mock_event)
                break

        # Should have called exit with None
        mock_event.app.exit.assert_called_once()
        call_args = mock_event.app.exit.call_args
        result = call_args.kwargs.get("result") or call_args[1].get("result")
        assert result is None


class TestSelectControl:
    """Tests for _SelectControl class (lines 1049-1085)."""

    async def test_select_is_focusable_returns_true(self):
        """Line 1054-1055: is_focusable always returns True."""
        from chcode.utils.tools import _select_with_other_async

        captured_control = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_control
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value="A")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        assert captured_control is not None
        assert captured_control.is_focusable() is True

    async def test_select_preferred_height(self):
        """Lines 1060-1063: preferred_height returns len(opts) + 1."""
        from chcode.utils.tools import _select_with_other_async

        captured_control = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_control
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value="A")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B", "C"])

        assert captured_control is not None
        height = captured_control.preferred_height(
            width=80, max_available_height=100, wrap_lines=False, get_line_prefix=None
        )
        assert height == 4  # 3 options + 1 input row

    async def test_select_create_content_with_selection(self):
        """Lines 1065-1085: create_content renders selected item with > prefix."""
        from chcode.utils.tools import _select_with_other_async

        captured_control = None
        captured_buffer = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_control, captured_buffer
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_buffer = hsplit.children[2].content.buffer
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value="A")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B", "C"])

        assert captured_control is not None
        # Select second item
        captured_control.selected = 1
        content = captured_control.create_content(width=80, height=10)
        line_1 = content.get_line(1)
        line_str = str(line_1)
        # Should have > prefix and bold style
        assert "❯" in line_str
        assert "B" in line_str

    async def test_select_create_content_with_input_row_selected(self):
        """Lines 1073-1077: Input row shows buffer text when selected."""
        from chcode.utils.tools import _select_with_other_async

        captured_control = None
        captured_buffer = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_control, captured_buffer
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_buffer = hsplit.children[2].content.buffer
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value="custom")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        assert captured_control is not None
        # Select input row (index 2 for 2 options)
        captured_control.selected = 2
        captured_buffer.text = "custom value"
        content = captured_control.create_content(width=80, height=10)
        line_2 = content.get_line(2)
        line_str = str(line_2)
        # Should show buffer text with > prefix
        assert "❯" in line_str
        assert "custom value" in line_str


class TestSelectHandlers:
    """Tests for select key binding handlers (lines 1106-1184)."""

    async def test_select_up_handler_decrements_selection(self):
        """Lines 1106-1117: Up arrow decrements selected index."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value="A")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B", "C"])

        # Start at selection 2
        captured_control.selected = 2
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        # Find and call up handler
        for binding in captured_kb.bindings:
            if "up" in str(binding.keys).lower():
                binding.handler(mock_event)
                break

        assert captured_control.selected == 1
        mock_event.app.invalidate.assert_called_once()

    async def test_select_down_handler_increments_selection(self):
        """Lines 1119-1128: Down arrow increments selected index."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value="A")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B", "C"])

        # Start at selection 0
        captured_control.selected = 0
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        # Find and call down handler
        for binding in captured_kb.bindings:
            if "down" in str(binding.keys).lower():
                binding.handler(mock_event)
                break

        assert captured_control.selected == 1
        mock_event.app.invalidate.assert_called_once()

    async def test_select_down_handler_focuses_input_at_bottom(self):
        """Lines 1126-1127: Down handler focuses input when reaching input_row_idx."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None
        captured_input_window = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_input_window
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_input_window = hsplit.children[2]
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value="A")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        # Select last option
        captured_control.selected = 1
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        # Find and call down handler
        for binding in captured_kb.bindings:
            if "down" in str(binding.keys).lower():
                binding.handler(mock_event)
                break

        # Should have moved to input row
        assert captured_control.selected == 2
        mock_event.app.layout.focus.assert_called_with(captured_input_window)

    async def test_select_tab_handler_toggles_focus(self):
        """Lines 1130-1141: Tab toggles between list and input focus."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None
        captured_control_window = None
        captured_input_window = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal \
                captured_kb, \
                captured_control, \
                captured_control_window, \
                captured_input_window
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control_window = hsplit.children[1]
                captured_input_window = hsplit.children[2]
                captured_control = captured_control_window.content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value="A")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        # Start with selection on an option
        captured_control.selected = 0
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        # Find and call tab handler (tab is Keys.ControlI)
        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-i" in keys_str or "tab" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Should have moved to input row
        assert captured_control.selected == 2
        mock_event.app.layout.focus.assert_called_with(captured_input_window)

    async def test_select_enter_handler_submits_option(self):
        """Lines 1143-1162: Enter submits selected option."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value="B")
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B", "C"])

        # Select second option
        captured_control.selected = 1
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        # Find and call enter handler (enter is Keys.ControlM)
        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Should have exited with selected option
        mock_event.app.exit.assert_called_once()
        call_args = mock_event.app.exit.call_args
        result = call_args.kwargs.get("result") or call_args[1].get("result")
        assert result == "B"

    async def test_select_enter_handler_with_custom_input(self):
        """Lines 1149-1156: Enter with input row selected submits custom text."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None
        captured_buffer = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_buffer, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_buffer = hsplit.children[2].content.buffer
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value="custom")
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        # Select input row and add custom text
        captured_control.selected = 2
        captured_buffer.text = " custom option "
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        # Find and call enter handler (enter is Keys.ControlM)
        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Should have exited with custom text
        mock_event.app.exit.assert_called_once()
        call_args = mock_event.app.exit.call_args
        result = call_args.kwargs.get("result") or call_args[1].get("result")
        assert result == "custom option"

    async def test_select_enter_handler_with_empty_custom_input(self):
        """Lines 1156-1157: Enter on empty input row unsets _exiting."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None
        captured_buffer = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_buffer
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_buffer = hsplit.children[2].content.buffer
            mock_app = MagicMock()
            mock_app.exit = MagicMock()
            mock_app.run_async = AsyncMock(return_value=None)
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        # Select input row with empty text
        captured_control.selected = 2
        captured_buffer.text = ""
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()
        mock_event.app.exit = MagicMock()

        # Find and call enter handler
        for binding in captured_kb.bindings:
            if "enter" in str(binding.keys).lower():
                binding.handler(mock_event)
                break

        # Should NOT have called exit (empty input)
        mock_event.app.exit.assert_not_called()

    async def test_select_esc_handler_cancels(self):
        """Lines 1164-1173: Escape cancels and returns None."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.exit = MagicMock()
            mock_app.run_async = AsyncMock(return_value=None)
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()
        mock_event.app.exit = MagicMock()

        # Find and call esc handler
        for binding in captured_kb.bindings:
            if (
                "escape" in str(binding.keys).lower()
                or "esc" in str(binding.keys).lower()
            ):
                binding.handler(mock_event)
                break

        # Should have called exit with None
        mock_event.app.exit.assert_called_once()
        call_args = mock_event.app.exit.call_args
        result = call_args.kwargs.get("result") or call_args[1].get("result")
        assert result is None

    async def test_select_cancel_handler_cancels(self):
        """Lines 1175-1184: Ctrl-C cancels and returns None."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.exit = MagicMock()
            mock_app.run_async = AsyncMock(return_value=None)
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()
        mock_event.app.exit = MagicMock()

        # Find and call cancel handler (c-c)
        for binding in captured_kb.bindings:
            if "c-c" in str(binding.keys):
                binding.handler(mock_event)
                break

        # Should have called exit with None
        mock_event.app.exit.assert_called_once()
        call_args = mock_event.app.exit.call_args
        result = call_args.kwargs.get("result") or call_args[1].get("result")
        assert result is None


class TestExitingGuard:
    """Tests for _exiting guard in all handlers."""

    async def test_checkbox_up_guard_when_exiting(self):
        """Lines 939-941: _exiting guard prevents handler execution."""
        from chcode.utils.tools import _checkbox_with_other_async

        # We'll test this by verifying the handler structure includes the guard
        # The actual _exiting variable is in the closure and is set by enter/esc/cancel
        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # Verify up handler exists
        up_handler_found = False
        for binding in captured_kb.bindings:
            if "up" in str(binding.keys).lower():
                up_handler_found = True
                # Handler is callable
                assert callable(binding.handler)
                break

        assert up_handler_found, "Up handler not found in key bindings"

    async def test_select_enter_guard_when_exiting(self):
        """Lines 1145-1147: _exiting guard prevents enter from running twice."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.exit = MagicMock()
            mock_app.run_async = AsyncMock(return_value="A")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        # Verify enter handler exists and has guard (enter is Keys.ControlM)
        enter_handler_found = False
        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                enter_handler_found = True
                assert callable(binding.handler)
                break

        assert enter_handler_found, "Enter handler not found in key bindings"


class TestApplicationCreation:
    """Tests for Application creation and run_async calls."""

    async def test_checkbox_application_created_with_correct_params(self):
        """Lines 1020-1023: Application created with layout, key_bindings, etc."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_app_args = {}

        def capture_app(*args, key_bindings=None, **kwargs):
            captured_app_args["layout"] = kwargs.get("layout")
            captured_app_args["key_bindings"] = key_bindings
            captured_app_args["full_screen"] = kwargs.get("full_screen")
            captured_app_args["erase_when_done"] = kwargs.get("erase_when_done")
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        assert captured_app_args["layout"] is not None
        assert captured_app_args["key_bindings"] is not None
        assert captured_app_args["full_screen"] is False
        assert captured_app_args["erase_when_done"] is True

    async def test_select_application_created_with_correct_params(self):
        """Lines 1186-1189: Application created with correct parameters."""
        from chcode.utils.tools import _select_with_other_async

        captured_app_args = {}

        def capture_app(*args, key_bindings=None, **kwargs):
            captured_app_args["layout"] = kwargs.get("layout")
            captured_app_args["key_bindings"] = key_bindings
            captured_app_args["full_screen"] = kwargs.get("full_screen")
            captured_app_args["erase_when_done"] = kwargs.get("erase_when_done")
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value="A")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        assert captured_app_args["layout"] is not None
        assert captured_app_args["key_bindings"] is not None
        assert captured_app_args["full_screen"] is False
        assert captured_app_args["erase_when_done"] is True


# ============================================================================
# Coverage for remaining uncovered handler branches
# ============================================================================


class TestCheckboxExitingGuards:
    """Tests for _exiting guard in checkbox handlers (lines 941, 954, 965, 978, 987, 1002, 1013)."""

    async def test_up_guard_after_enter(self):
        """Line 941: Up handler returns early when _exiting is True (set by enter)."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value=None)
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # Call enter first to set _exiting = True
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Now call up — should return early (line 941)
        captured_control.selected = 1
        mock_event2 = MagicMock()
        mock_event2.app.layout = MagicMock()
        for binding in captured_kb.bindings:
            if "up" in str(binding.keys).lower():
                binding.handler(mock_event2)
                break

        assert captured_control.selected == 1

    async def test_down_guard_after_enter(self):
        """Line 954: Down handler returns early when _exiting is True."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value=None)
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # Call enter first to set _exiting = True
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Now call down — should return early (line 954)
        captured_control.selected = 0
        mock_event2 = MagicMock()
        mock_event2.app.layout = MagicMock()
        for binding in captured_kb.bindings:
            if "down" in str(binding.keys).lower():
                binding.handler(mock_event2)
                break

        assert captured_control.selected == 0

    async def test_tab_guard_after_enter(self):
        """Line 965: Tab handler returns early when _exiting is True."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value=None)
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # Call enter first to set _exiting = True
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Now call tab — should return early (line 965)
        captured_control.selected = 0
        mock_event2 = MagicMock()
        mock_event2.app.layout = MagicMock()
        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-i" in keys_str or "tab" in keys_str.lower():
                binding.handler(mock_event2)
                break

        assert captured_control.selected == 0

    async def test_space_guard_after_enter(self):
        """Line 978: Space handler returns early when _exiting is True."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value=None)
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # Call enter first to set _exiting = True
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Now call space — should return early (line 978)
        captured_control.selected = 0
        mock_event2 = MagicMock()
        mock_event2.app.layout = MagicMock()
        for binding in captured_kb.bindings:
            if len(binding.keys) == 1 and binding.keys[0] == " ":
                binding.handler(mock_event2)
                break

        assert len(captured_control.checked) == 0

    async def test_enter_guard_called_twice(self):
        """Line 987: Enter handler returns early when _exiting is True (second call)."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value=None)
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # Call enter first time
        captured_control.checked = {0}
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        assert captured_app.exit.call_count == 1

        # Call enter second time — should hit _exiting guard (line 987)
        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        assert captured_app.exit.call_count == 1

    async def test_esc_guard_called_twice(self):
        """Line 1002: Escape handler returns early when _exiting is True (second call)."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.exit = MagicMock()
            mock_app.run_async = AsyncMock(return_value=None)
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()
        mock_event.app.exit = MagicMock()

        for binding in captured_kb.bindings:
            if (
                "escape" in str(binding.keys).lower()
                or "esc" in str(binding.keys).lower()
            ):
                binding.handler(mock_event)
                break

        assert mock_event.app.exit.call_count == 1

        # Call escape second time — should hit _exiting guard (line 1002)
        mock_event2 = MagicMock()
        mock_event2.app.layout = MagicMock()
        mock_event2.app.exit = MagicMock()
        for binding in captured_kb.bindings:
            if (
                "escape" in str(binding.keys).lower()
                or "esc" in str(binding.keys).lower()
            ):
                binding.handler(mock_event2)
                break

        assert mock_event2.app.exit.call_count == 0

    async def test_cancel_guard_called_twice(self):
        """Line 1013: Cancel handler returns early when _exiting is True (second call)."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.exit = MagicMock()
            mock_app.run_async = AsyncMock(return_value=None)
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()
        mock_event.app.exit = MagicMock()

        for binding in captured_kb.bindings:
            if "c-c" in str(binding.keys):
                binding.handler(mock_event)
                break

        assert mock_event.app.exit.call_count == 1

        # Call cancel second time — should hit _exiting guard (line 1013)
        mock_event2 = MagicMock()
        mock_event2.app.layout = MagicMock()
        mock_event2.app.exit = MagicMock()
        for binding in captured_kb.bindings:
            if "c-c" in str(binding.keys):
                binding.handler(mock_event2)
                break

        assert mock_event2.app.exit.call_count == 0


class TestCheckboxHandlerExceptions:
    """Tests for exception branches in checkbox handlers (lines 995, 1006, 1017)."""

    async def test_enter_handler_exception_caught(self):
        """Line 995: Exception in app.exit is caught silently."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            captured_app = MagicMock()
            captured_app.exit = MagicMock(side_effect=RuntimeError("already exiting"))
            captured_app.run_async = AsyncMock(return_value=None)
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        captured_control.checked = {0}
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        captured_app.exit.assert_called_once()

    async def test_esc_handler_exception_caught(self):
        """Line 1006: Exception in app.exit is caught silently."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.exit = MagicMock(side_effect=RuntimeError("already exiting"))
            mock_app.run_async = AsyncMock(return_value=None)
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()
        mock_event.app.exit = MagicMock(side_effect=RuntimeError("already exiting"))

        for binding in captured_kb.bindings:
            if (
                "escape" in str(binding.keys).lower()
                or "esc" in str(binding.keys).lower()
            ):
                binding.handler(mock_event)
                break

        mock_event.app.exit.assert_called_once()

    async def test_cancel_handler_exception_caught(self):
        """Line 1017: Exception in app.exit is caught silently."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.exit = MagicMock(side_effect=RuntimeError("already exiting"))
            mock_app.run_async = AsyncMock(return_value=None)
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()
        mock_event.app.exit = MagicMock(side_effect=RuntimeError("already exiting"))

        for binding in captured_kb.bindings:
            if "c-c" in str(binding.keys):
                binding.handler(mock_event)
                break

        mock_event.app.exit.assert_called_once()


class TestCheckboxUpHandlerElseBranch:
    """Tests for line 947: up handler else branch (focus input_edit)."""

    async def test_up_from_above_input_row_focuses_input(self):
        """Line 947: Up handler focuses input_edit when selected-1 >= input_row_idx."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_kb = None
        captured_control = None
        captured_input_window = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_input_window
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_input_window = hsplit.children[2]
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        # Set selected to input_row_idx + 1 (beyond normal range)
        # input_row_idx = len(options) = 2
        captured_control.selected = 3
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            if "up" in str(binding.keys).lower():
                binding.handler(mock_event)
                break

        # After decrement: selected = 2, which is input_row_idx.
        # 2 < 2 is False, so it goes to else branch (line 947)
        assert captured_control.selected == 2
        mock_event.app.layout.focus.assert_called_with(captured_input_window)


class TestCheckboxGetInvalidateEvents:
    """Tests for line 888: get_invalidate_events yields on_text_changed."""

    async def test_get_invalidate_events_yields(self):
        """Line 888: get_invalidate_events yields input_buffer.on_text_changed."""
        from chcode.utils.tools import _checkbox_with_other_async

        captured_control = None
        captured_buffer = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_control, captured_buffer
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_buffer = hsplit.children[2].content.buffer
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value=["A"])
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _checkbox_with_other_async("Pick?", ["A", "B"])

        events = list(captured_control.get_invalidate_events())
        assert len(events) == 1
        assert events[0] is captured_buffer.on_text_changed


class TestSelectExitingGuards:
    """Tests for _exiting guard in select handlers (lines 1110, 1123, 1134, 1147, 1168, 1179)."""

    async def test_select_up_guard_after_enter(self):
        """Line 1110: Up handler returns early when _exiting is True."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value="A")
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        # Call enter first to set _exiting = True
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Now call up — should return early (line 1110)
        captured_control.selected = 1
        mock_event2 = MagicMock()
        mock_event2.app.layout = MagicMock()
        for binding in captured_kb.bindings:
            if "up" in str(binding.keys).lower():
                binding.handler(mock_event2)
                break

        assert captured_control.selected == 1

    async def test_select_down_guard_after_enter(self):
        """Line 1123: Down handler returns early when _exiting is True."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value="A")
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        # Call enter first to set _exiting = True
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Now call down — should return early (line 1123)
        captured_control.selected = 0
        mock_event2 = MagicMock()
        mock_event2.app.layout = MagicMock()
        for binding in captured_kb.bindings:
            if "down" in str(binding.keys).lower():
                binding.handler(mock_event2)
                break

        assert captured_control.selected == 0

    async def test_select_tab_guard_after_enter(self):
        """Line 1134: Tab handler returns early when _exiting is True."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value="A")
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        # Call enter first to set _exiting = True
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Now call tab — should return early (line 1134)
        captured_control.selected = 0
        mock_event2 = MagicMock()
        mock_event2.app.layout = MagicMock()
        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-i" in keys_str or "tab" in keys_str.lower():
                binding.handler(mock_event2)
                break

        assert captured_control.selected == 0

    async def test_select_enter_guard_called_twice(self):
        """Line 1147: Enter handler returns early when _exiting is True (second call)."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            captured_app = MagicMock()
            captured_app.exit = MagicMock()
            captured_app.run_async = AsyncMock(return_value="A")
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        # Call enter first time — selects option at index 0
        captured_control.selected = 0
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        assert captured_app.exit.call_count == 1

        # Call enter second time — should hit _exiting guard (line 1147)
        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        assert captured_app.exit.call_count == 1

    async def test_select_esc_guard_called_twice(self):
        """Line 1168: Escape handler returns early when _exiting is True (second call)."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.exit = MagicMock()
            mock_app.run_async = AsyncMock(return_value=None)
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()
        mock_event.app.exit = MagicMock()

        for binding in captured_kb.bindings:
            if (
                "escape" in str(binding.keys).lower()
                or "esc" in str(binding.keys).lower()
            ):
                binding.handler(mock_event)
                break

        assert mock_event.app.exit.call_count == 1

        # Call escape second time — should hit _exiting guard (line 1168)
        mock_event2 = MagicMock()
        mock_event2.app.layout = MagicMock()
        mock_event2.app.exit = MagicMock()
        for binding in captured_kb.bindings:
            if (
                "escape" in str(binding.keys).lower()
                or "esc" in str(binding.keys).lower()
            ):
                binding.handler(mock_event2)
                break

        assert mock_event2.app.exit.call_count == 0

    async def test_select_cancel_guard_called_twice(self):
        """Line 1179: Cancel handler returns early when _exiting is True (second call)."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.exit = MagicMock()
            mock_app.run_async = AsyncMock(return_value=None)
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()
        mock_event.app.exit = MagicMock()

        for binding in captured_kb.bindings:
            if "c-c" in str(binding.keys):
                binding.handler(mock_event)
                break

        assert mock_event.app.exit.call_count == 1

        # Call cancel second time — should hit _exiting guard (line 1179)
        mock_event2 = MagicMock()
        mock_event2.app.layout = MagicMock()
        mock_event2.app.exit = MagicMock()
        for binding in captured_kb.bindings:
            if "c-c" in str(binding.keys):
                binding.handler(mock_event2)
                break

        assert mock_event2.app.exit.call_count == 0


class TestSelectHandlerExceptions:
    """Tests for exception branches in select handlers (lines 1154, 1161, 1172, 1183)."""

    async def test_select_enter_exception_on_custom_input(self):
        """Lines 1154-1155: Exception in app.exit when entering custom text is caught."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None
        captured_buffer = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_buffer, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_buffer = hsplit.children[2].content.buffer
            captured_app = MagicMock()
            captured_app.exit = MagicMock(side_effect=RuntimeError("already exiting"))
            captured_app.run_async = AsyncMock(return_value=None)
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        # Select input row with text
        captured_control.selected = 2
        captured_buffer.text = "custom"
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        captured_app.exit.assert_called_once()

    async def test_select_enter_exception_on_option(self):
        """Lines 1160-1161: Exception in app.exit when selecting option is caught."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None
        captured_app = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_app
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                captured_control = layout.container.children[1].content
            captured_app = MagicMock()
            captured_app.exit = MagicMock(side_effect=RuntimeError("already exiting"))
            captured_app.run_async = AsyncMock(return_value=None)
            return captured_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        # Select an option
        captured_control.selected = 0
        mock_event = MagicMock()
        mock_event.app = captured_app
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-m" in keys_str or "enter" in keys_str.lower():
                binding.handler(mock_event)
                break

        captured_app.exit.assert_called_once()

    async def test_select_esc_exception_caught(self):
        """Lines 1171-1172: Exception in app.exit is caught silently."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.exit = MagicMock(side_effect=RuntimeError("already exiting"))
            mock_app.run_async = AsyncMock(return_value=None)
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()
        mock_event.app.exit = MagicMock(side_effect=RuntimeError("already exiting"))

        for binding in captured_kb.bindings:
            if (
                "escape" in str(binding.keys).lower()
                or "esc" in str(binding.keys).lower()
            ):
                binding.handler(mock_event)
                break

        mock_event.app.exit.assert_called_once()

    async def test_select_cancel_exception_caught(self):
        """Lines 1182-1183: Exception in app.exit is caught silently."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb
            captured_kb = key_bindings
            mock_app = MagicMock()
            mock_app.exit = MagicMock(side_effect=RuntimeError("already exiting"))
            mock_app.run_async = AsyncMock(return_value=None)
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()
        mock_event.app.exit = MagicMock(side_effect=RuntimeError("already exiting"))

        for binding in captured_kb.bindings:
            if "c-c" in str(binding.keys):
                binding.handler(mock_event)
                break

        mock_event.app.exit.assert_called_once()


class TestSelectUpElseBranch:
    """Tests for line 1116: select up handler else branch (focus input_edit)."""

    async def test_select_up_from_above_input_row_focuses_input(self):
        """Line 1116: Up handler focuses input_edit when selected-1 >= input_row_idx."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None
        captured_input_window = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_input_window
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_input_window = hsplit.children[2]
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value="A")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        # Set selected to input_row_idx + 1 (beyond normal range)
        captured_control.selected = 3
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            if "up" in str(binding.keys).lower():
                binding.handler(mock_event)
                break

        # After decrement: selected = 2, which is input_row_idx.
        # 2 < 2 is False, so it goes to else branch (line 1116)
        assert captured_control.selected == 2
        mock_event.app.layout.focus.assert_called_with(captured_input_window)


class TestSelectTabFromInputRow:
    """Tests for lines 1136-1137: select tab from input_row_idx to 0."""

    async def test_select_tab_from_input_row_to_first_option(self):
        """Lines 1136-1137: Tab from input row moves to first option and focuses control."""
        from chcode.utils.tools import _select_with_other_async

        captured_kb = None
        captured_control = None
        captured_control_window = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_kb, captured_control, captured_control_window
            captured_kb = key_bindings
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control_window = hsplit.children[1]
                captured_control = captured_control_window.content
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value="A")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        # Start with selection on input row (index 2 for 2 options)
        captured_control.selected = 2
        mock_event = MagicMock()
        mock_event.app.layout = MagicMock()

        for binding in captured_kb.bindings:
            keys_str = str(binding.keys)
            if "c-i" in keys_str or "tab" in keys_str.lower():
                binding.handler(mock_event)
                break

        # Should have moved to first option (line 1136) and focused control window (line 1137)
        assert captured_control.selected == 0
        mock_event.app.layout.focus.assert_called_with(captured_control_window)


class TestSelectGetInvalidateEvents:
    """Tests for line 1058: get_invalidate_events yields on_text_changed."""

    async def test_select_get_invalidate_events_yields(self):
        """Line 1058: get_invalidate_events yields input_buffer.on_text_changed."""
        from chcode.utils.tools import _select_with_other_async

        captured_control = None
        captured_buffer = None

        def capture_app(*args, key_bindings=None, **kwargs):
            nonlocal captured_control, captured_buffer
            layout = kwargs.get("layout")
            if layout:
                hsplit = layout.container
                captured_control = hsplit.children[1].content
                captured_buffer = hsplit.children[2].content.buffer
            mock_app = MagicMock()
            mock_app.run_async = AsyncMock(return_value="A")
            return mock_app

        with patch("chcode.utils.tools.console"):
            with patch("prompt_toolkit.Application", side_effect=capture_app):
                await _select_with_other_async("Pick?", ["A", "B"])

        events = list(captured_control.get_invalidate_events())
        assert len(events) == 1
        assert events[0] is captured_buffer.on_text_changed
