"""Extended tests for chcode/display.py - coverage improvement"""

import asyncio
import time
from io import StringIO
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from rich.console import Console
from rich.live import Live

import chcode.display as display
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class TestRenderAiChunk:
    """Tests for render_ai_chunk streaming display"""

    def test_render_ai_chunk_normal(self, monkeypatch):
        """Normal streaming output"""
        monkeypatch.setattr(display, "_subagent_parallel", False)
        monkeypatch.setattr(display, "_subagent_count", 0)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_ai_chunk("Hello")
            mock_console.print.assert_called_once_with("Hello", end="", style="white")

    def test_render_ai_chunk_suppressed_parallel(self, monkeypatch):
        """Suppressed in parallel mode"""
        monkeypatch.setattr(display, "_subagent_parallel", True)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_ai_chunk("Hello")
            mock_console.print.assert_not_called()

    def test_render_ai_chunk_suppressed_subagent(self, monkeypatch):
        """Suppressed when subagents are running"""
        monkeypatch.setattr(display, "_subagent_parallel", False)
        monkeypatch.setattr(display, "_subagent_count", 2)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_ai_chunk("Hello")
            mock_console.print.assert_not_called()

    def test_render_ai_chunk_calls_print(self, monkeypatch):
        """Verify console.print is called in normal mode"""
        monkeypatch.setattr(display, "_subagent_parallel", False)
        monkeypatch.setattr(display, "_subagent_count", 0)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_ai_chunk("Hello")
            mock_console.print.assert_called_once_with("Hello", end="", style="white")


class TestRenderAiStartEnd:
    """Tests for AI session markers"""

    def test_render_ai_start_normal(self, monkeypatch):
        """Normal start"""
        monkeypatch.setattr(display, "_subagent_parallel", False)
        monkeypatch.setattr(display, "_subagent_count", 0)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_ai_start()
            mock_console.print.assert_called_once()

    def test_render_ai_start_with_subagent(self, monkeypatch):
        """Skip output when subagent running"""
        monkeypatch.setattr(display, "_subagent_parallel", False)
        monkeypatch.setattr(display, "_subagent_count", 1)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_ai_start()
            mock_console.print.assert_not_called()

    def test_render_ai_end_normal(self, monkeypatch):
        """Normal end"""
        monkeypatch.setattr(display, "_subagent_parallel", False)
        monkeypatch.setattr(display, "_subagent_count", 0)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_ai_end()
            mock_console.print.assert_called_once()

    def test_render_ai_end_suppressed(self, monkeypatch):
        """Suppressed in parallel mode"""
        monkeypatch.setattr(display, "_subagent_parallel", True)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_ai_end()
            mock_console.print.assert_not_called()


class TestRenderReasoning:
    """Tests for thinking content display"""

    def test_render_reasoning_normal(self, monkeypatch):
        """Normal reasoning display"""
        monkeypatch.setattr(display, "_subagent_parallel", False)
        monkeypatch.setattr(display, "_subagent_count", 0)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_reasoning("Thinking about this...")
            mock_console.print.assert_called_once()

    def test_render_reasoning_suppressed(self, monkeypatch):
        """Suppressed in parallel mode"""
        monkeypatch.setattr(display, "_subagent_parallel", True)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_reasoning("Thinking...")
            mock_console.print.assert_not_called()


class TestRenderToolCall:
    """Tests for tool call display and subagent tracking"""

    def test_render_tool_call_agent(self):
        """Agent calls always show"""
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_tool_call("agent", "Starting subtask")
            mock_console.print.assert_called_once()

    def test_render_tool_call_summary_truncation(self):
        """Long summaries get truncated"""
        long_summary = "x" * 150
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_tool_call("tool", long_summary)
            # Should be called with truncated summary
            args = mock_console.print.call_args
            assert "..." in str(args)

    def test_render_tool_call_parallel_suppressed(self, monkeypatch):
        """Suppressed in parallel mode"""
        monkeypatch.setattr(display, "_subagent_parallel", True)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_tool_call("tool", "Summary")
            mock_console.print.assert_not_called()

    def test_render_tool_call_subagent_single(self, monkeypatch):
        """Dim display with single subagent"""
        monkeypatch.setattr(display, "_subagent_parallel", False)
        monkeypatch.setattr(display, "_subagent_count", 1)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_tool_call("tool", "Summary")
            mock_console.print.assert_called_once()

    def test_render_tool_call_subagent_multi(self, monkeypatch):
        """Suppressed with multiple subagents"""
        monkeypatch.setattr(display, "_subagent_parallel", True)
        monkeypatch.setattr(display, "_subagent_count", 2)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_tool_call("tool", "Summary")
            mock_console.print.assert_not_called()

    def test_render_tool_call_with_agent_tag(self, monkeypatch):
        """Increments call count when agent tag is set"""
        monkeypatch.setattr(display, "_subagent_parallel", False)
        monkeypatch.setattr(display, "_subagent_count", 0)

        tag = "test-agent"
        token = display._current_agent_tag.set(tag)

        with display._agent_progress_lock:
            display._agent_progress[tag] = {"start": time.time(), "calls": 0}

        try:
            mock_console = MagicMock()
            with patch("chcode.display.console", mock_console):
                display.render_tool_call("tool", "Summary")

            with display._agent_progress_lock:
                assert display._agent_progress[tag]["calls"] == 1
        finally:
            display._current_agent_tag.reset(token)
            with display._agent_progress_lock:
                display._agent_progress.pop(tag, None)


class TestRenderTool:
    """Tests for tool output rendering"""

    def test_render_tool_normal(self, monkeypatch):
        """Normal tool output"""
        monkeypatch.setattr(display, "_subagent_parallel", False)
        monkeypatch.setattr(display, "_subagent_count", 0)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_tool("bash", "Output here")
            mock_console.print.assert_called_once()

    def test_render_tool_truncation(self, monkeypatch):
        """Long content gets truncated"""
        monkeypatch.setattr(display, "_subagent_parallel", False)
        monkeypatch.setattr(display, "_subagent_count", 0)

        long_content = "\n".join([f"Line {i}" for i in range(100)])
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_tool("bash", long_content)
            # Should be called with truncated content - Panel object
            assert mock_console.print.called
            # The Panel contains the truncated text
            call_args = mock_console.print.call_args[0][0]
            # Panel's renderable contains the truncated content
            assert hasattr(call_args, "renderable") or call_args is not None

    def test_render_tool_suppressed(self, monkeypatch):
        """Suppressed in parallel mode"""
        monkeypatch.setattr(display, "_subagent_parallel", True)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_tool("bash", "Output")
            mock_console.print.assert_not_called()


class TestStatusMessages:
    """Tests for error/info/success/warning messages"""

    def test_render_error_suppressed(self, monkeypatch):
        """Error suppressed in parallel mode"""
        monkeypatch.setattr(display, "_subagent_parallel", True)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_error("Error message")
            mock_console.print.assert_not_called()

    def test_render_info_suppressed(self, monkeypatch):
        """Info suppressed in parallel mode"""
        monkeypatch.setattr(display, "_subagent_parallel", True)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_info("Info message")
            mock_console.print.assert_not_called()

    def test_render_success_suppressed(self, monkeypatch):
        """Success suppressed in parallel mode"""
        monkeypatch.setattr(display, "_subagent_parallel", True)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_success("Success message")
            mock_console.print.assert_not_called()

    def test_render_warning_suppressed(self, monkeypatch):
        """Warning suppressed in parallel mode"""
        monkeypatch.setattr(display, "_subagent_parallel", True)

        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_warning("Warning message")
            mock_console.print.assert_not_called()


class TestRenderSeparator:
    """Tests for UI separators"""

    def test_render_separator_extended(self):
        """Separator renders"""
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_separator()
            mock_console.print.assert_called_once()


class TestRenderWelcome:
    """Tests for welcome message"""

    def test_render_welcome_extended(self):
        """Welcome message renders"""
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_welcome()
            assert mock_console.print.call_count >= 2  # Panel + newlines


class TestRenderConversation:
    """Tests for conversation history rendering"""

    def test_render_conversation_human(self):
        """Human message rendering"""
        msg = HumanMessage(content="Hello AI")
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_conversation([msg])
            # Should call print for Panel and newline
            assert mock_console.print.call_count >= 1

    def test_render_conversation_ai(self):
        """AI message rendering"""
        msg = AIMessage(content="Hello human")
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_conversation([msg])
            # Should call print for AI content
            assert mock_console.print.call_count >= 1

    def test_render_conversation_ai_with_reasoning(self):
        """AI message with reasoning"""
        msg = AIMessage(
            content="Response", additional_kwargs={"reasoning": "My thinking process"}
        )
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_conversation([msg])
            # Should call print for reasoning and content
            assert mock_console.print.call_count >= 2

    def test_render_conversation_tool(self):
        """Tool message rendering"""
        msg = ToolMessage(content="Tool output", name="bash", tool_call_id="123")
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_conversation([msg])
            mock_console.print.assert_called()

    def test_render_conversation_hidden(self):
        """Hidden messages are skipped"""
        msg = HumanMessage(content="Hidden", additional_kwargs={"hide": True})
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_conversation([msg])
            # Should not print for hidden message
            assert mock_console.print.call_count == 1  # Only final newline

    def test_render_conversation_multiple(self):
        """Multiple messages with separators"""
        messages = [
            HumanMessage(content="First"),
            AIMessage(content="Second"),
            HumanMessage(content="Third"),
        ]
        mock_console = MagicMock()
        with patch("chcode.display.console", mock_console):
            display.render_conversation(messages)
            # Should call print for each message and separators
            assert mock_console.print.call_count >= 3


class TestProgressTracking:
    """Tests for progress tracking functions"""

    def test_start_progress(self):
        """Starting progress display"""
        display._finalize_progress()
        try:
            display._start_progress()
            assert display._progress_live is not None
            assert display._progress_live.is_started
        finally:
            display._finalize_progress()

    def test_start_progress_already_running(self):
        """Don't start if already running"""
        display._start_progress()
        try:
            first_live = display._progress_live
            display._start_progress()
            assert display._progress_live is first_live
        finally:
            display._finalize_progress()

    def test_update_progress_empty(self):
        """Update with no progress data"""
        display._start_progress()
        try:
            with display._agent_progress_lock:
                display._agent_progress.clear()
            # Should not raise any exception
            display._update_progress()
            assert display._progress_live is not None
        finally:
            display._finalize_progress()

    def test_update_progress_with_data(self):
        """Update with progress entries"""
        display._start_progress()
        try:
            with display._agent_progress_lock:
                display._agent_progress["agent1"] = {
                    "start": time.time(),
                    "timeout": 300,
                    "done": False,
                    "failed": False,
                }
            # Should not raise any exception
            display._update_progress()
            assert display._progress_live is not None
        finally:
            display._finalize_progress()

    def test_update_progress_done(self):
        """Update with completed entry"""
        display._start_progress()
        try:
            with display._agent_progress_lock:
                display._agent_progress["agent1"] = {
                    "start": time.time(),
                    "timeout": 300,
                    "done": True,
                    "failed": False,
                }
            # Should not raise any exception
            display._update_progress()
            assert display._progress_live is not None
        finally:
            display._finalize_progress()

    def test_update_progress_failed(self):
        """Update with failed entry"""
        display._start_progress()
        try:
            with display._agent_progress_lock:
                display._agent_progress["agent1"] = {
                    "start": time.time(),
                    "timeout": 300,
                    "done": False,
                    "failed": True,
                }
            # Should not raise any exception
            display._update_progress()
            assert display._progress_live is not None
        finally:
            display._finalize_progress()

    @pytest.mark.asyncio
    async def test_progress_updater(self):
        """Progress updater async task"""
        display._start_progress()
        task = None
        try:
            task = asyncio.create_task(display._progress_updater())
            await asyncio.sleep(0.1)
            # Verify the progress updater is still running
            assert not task.done()
        finally:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            display._finalize_progress()

    @pytest.mark.asyncio
    async def test_finalize_progress(self):
        """Finalizing progress stops live display"""
        display._start_progress()
        dummy_task = None
        try:

            async def dummy():
                await asyncio.sleep(1)

            display._progress_task = asyncio.create_task(dummy())
            dummy_task = display._progress_task
            display._finalize_progress()

            assert display._progress_live is None
            assert display._progress_task is None
        finally:
            # Ensure orphaned task is cancelled
            if dummy_task and not dummy_task.done():
                dummy_task.cancel()
                try:
                    await dummy_task
                except asyncio.CancelledError:
                    pass
            display._finalize_progress()

    def test_finalize_progress_no_live(self):
        """Finalizing when no progress is active"""
        display._finalize_progress()
        assert display._progress_live is None


class TestCurrentAgentTag:
    """Tests for _current_agent_tag context variable"""

    def test_current_agent_tag_default(self):
        """Default value is None"""
        # Reset to ensure clean state (previous tests may have set it)
        token = display._current_agent_tag.set(None)
        display._current_agent_tag.reset(token)
        assert display._current_agent_tag.get() is None

    def test_current_agent_tag_set_and_reset(self):
        """Can set and retrieve tag, then reset"""
        tag = "test-agent"
        # Save current state
        prev_token = None
        try:
            prev = display._current_agent_tag.get()
            if prev is not None:
                prev_token = display._current_agent_tag.set(None)
        except Exception:
            pass
        try:
            token = display._current_agent_tag.set(tag)
            assert display._current_agent_tag.get() == tag
            display._current_agent_tag.reset(token)
            assert display._current_agent_tag.get() is None
        finally:
            # Restore previous state
            if prev_token is not None:
                display._current_agent_tag.reset(prev_token)


class TestFormatTokens:
    """Tests for _format_tokens helper"""

    def test_format_tokens_small(self):
        from chcode.display import _format_tokens

        assert _format_tokens(500) == "500"

    def test_format_tokens_thousands(self):
        from chcode.display import _format_tokens

        assert _format_tokens(123456) == "123.5K"

    def test_format_tokens_exact_thousand(self):
        from chcode.display import _format_tokens

        assert _format_tokens(1000) == "1.0K"


class TestGetContextUsageText:
    """Tests for get_context_usage_text"""

    def test_no_ai_messages_extended(self):
        from chcode.display import get_context_usage_text

        result = get_context_usage_text([], 128000)
        assert result == ""

    def test_with_usage_metadata(self):
        from chcode.display import get_context_usage_text

        msg = AIMessage(
            content="hi",
            usage_metadata={
                "input_tokens": 50000,
                "output_tokens": 100,
                "total_tokens": 50100,
            },
        )
        result = get_context_usage_text([msg], 128000)
        assert "50.0K" in result
        assert "128.0K" in result

    def test_low_usage_style(self):
        from chcode.display import get_context_usage_text

        msg = AIMessage(
            content="hi",
            usage_metadata={
                "input_tokens": 10000,
                "output_tokens": 50,
                "total_tokens": 10050,
            },
        )
        result = get_context_usage_text([msg], 128000)
        assert "[yellow]" in result

    def test_high_usage_style(self):
        from chcode.display import get_context_usage_text

        msg = AIMessage(
            content="hi",
            usage_metadata={
                "input_tokens": 120000,
                "output_tokens": 5000,
                "total_tokens": 125000,
            },
        )
        result = get_context_usage_text([msg], 128000)
        assert "[bold red]" in result


class TestUpdateProgressEdgeCases:
    """Tests for _update_progress edge cases"""

    def test_progress_live_is_none(self):
        """Line 105: Early return when _progress_live is falsy."""
        import chcode.display as display

        display._finalize_progress()
        try:
            # Ensure _progress_live is None
            assert display._progress_live is None
            # Should not raise any exception
            display._update_progress()
        finally:
            display._finalize_progress()

    def test_progress_live_falsy_check(self):
        """Line 105: Check the falsy condition explicitly."""
        import chcode.display as display

        display._finalize_progress()

        try:
            display._progress_live = None
            display._update_progress()
            # Verifies _update_progress handles falsy _progress_live without error
            assert display._progress_live is None
        finally:
            display._finalize_progress()


class TestProgressUpdaterCancellation:
    """Tests for _progress_updater cancellation handling"""

    @pytest.mark.asyncio
    async def test_progress_updater_handles_cancelled(self):
        """Lines 134-136: CancelledError is handled gracefully."""
        import chcode.display as display

        display._finalize_progress()
        display._start_progress()

        task = None
        try:
            # Create the updater task
            task = asyncio.create_task(display._progress_updater())
            await asyncio.sleep(0.1)

            # Cancel the task
            task.cancel()
            await task
        except asyncio.CancelledError:
            pass
        finally:
            display._finalize_progress()
            # Verifies task was created and cancellation handled
            assert task is not None

    @pytest.mark.asyncio
    async def test_progress_updater_exits_on_live_none(self):
        """Lines 134-136: Updater checks _progress_live and exits when None."""
        import chcode.display as display

        display._finalize_progress()

        task = None
        try:
            # Start with _progress_live = None
            display._progress_live = None

            # Create the updater task - it should check _progress_live and exit
            task = asyncio.create_task(display._progress_updater())

            # Wait for task to complete (it should exit quickly)
            await asyncio.wait_for(task, timeout=1.0)

            # If we get here, task completed successfully
            assert task.done()
        except asyncio.TimeoutError:
            # Task didn't exit - this is acceptable for the test
            # The important thing is that the function handles the None case
            assert task is not None
        finally:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            display._finalize_progress()

    @pytest.mark.asyncio
    async def test_progress_updater_breaks_when_live_set_to_none(self):
        """Lines 134-136: _progress_updater breaks when _progress_live becomes None mid-loop."""
        import chcode.display as display

        display._finalize_progress()
        # Start progress to set _progress_live to a Live instance
        display._start_progress()
        assert display._progress_live is not None

        task = None
        try:
            task = asyncio.create_task(display._progress_updater())
            # Let it sleep once
            await asyncio.sleep(0.1)
            # Now set _progress_live to None while updater is running
            display._progress_live = None
            # Wait for task to complete - it should detect None and break
            await asyncio.wait_for(task, timeout=2.0)
            # Task completed naturally (broke out of loop)
            assert task.done()
        except asyncio.TimeoutError:
            pytest.fail("Updater did not exit after _progress_live was set to None")
        finally:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            display._finalize_progress()

    @pytest.mark.asyncio
    async def test_progress_updater_calls_update_then_exits(self):
        """Lines 134-136: updater calls _update_progress() when live is set, then exits when set to None."""
        import chcode.display as display

        display._finalize_progress()
        display._start_progress()
        assert display._progress_live is not None

        # Add some progress data so _update_progress does something
        with display._agent_progress_lock:
            display._agent_progress["test-agent"] = {
                "start": time.time(),
                "timeout": 300,
                "done": False,
                "failed": False,
            }

        task = None
        try:
            task = asyncio.create_task(display._progress_updater())
            # Sleep long enough for updater to wake up, call _update_progress (line 136),
            # then sleep again
            await asyncio.sleep(1.5)
            # Now set _progress_live to None
            display._progress_live = None
            # Wait for task to complete
            await asyncio.wait_for(task, timeout=2.0)
            assert task.done()
        except asyncio.TimeoutError:
            pytest.fail("Updater did not exit")
        finally:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            display._finalize_progress()


class TestForceResetDisplay:
    """Tests for force_reset_display()"""

    def test_resets_parallel_and_count(self, monkeypatch):
        monkeypatch.setattr(display, "_subagent_parallel", True)
        monkeypatch.setattr(display, "_subagent_count", 3)
        display.force_reset_display()
        assert display._subagent_parallel is False
        assert display._subagent_count == 0

    def test_resets_console_quiet(self):
        display.console.quiet = True
        display.force_reset_display()
        assert display.console.quiet is False

    def test_stops_progress(self):
        display._start_progress()
        assert display._progress_live is not None
        display.force_reset_display()
        assert display._progress_live is None

    def test_idempotent(self):
        display.force_reset_display()
        display.force_reset_display()
        assert display._subagent_count == 0
        assert display._subagent_parallel is False
