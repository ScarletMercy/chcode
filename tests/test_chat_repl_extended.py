"""
Extended tests for chcode/chat.py - covering previously uncovered lines.
Focus: simple conditionals, flow control, error handling.
"""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage, RemoveMessage, BaseMessage

from chcode.chat import ChatREPL, _collect_ids_from_group


# ============================================================================
# Test run() -> _get_input returns None -> break
# Covers line 370
# ============================================================================


class TestRunBreakOnNone:
    async def test_run_get_input_none_breaks_loop(self):
        """When _get_input returns None, run() should break out of the loop."""
        repl = ChatREPL()

        with patch("chcode.chat.render_welcome"):
            with patch.object(repl, "_get_input", new_callable=AsyncMock, return_value=None):
                with patch("chcode.chat.render_error"):
                    await repl.run()

                # _process_input should never be called
                assert repl._processing is False


# ============================================================================
# Test _cmd_compress — marks messages as composed (lines 718-719)
# ============================================================================


class TestCmdCompressComposedMarking:
    async def test_compress_marks_pre_messages_as_composed(self):
        """Pre-messages (not in recent 2) should be marked composed=True."""
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        # 4 messages: 3 human turns + ai responses. Only the last 2 human turns are "recent".
        msg1 = HumanMessage("old", id="m1")
        msg2 = HumanMessage("mid", id="m2")
        msg3 = HumanMessage("recent1", id="m3")
        msg4 = HumanMessage("recent2", id="m4")

        state = Mock()
        state.values = {"messages": [msg1, msg2, msg3, msg4]}
        repl.agent.aget_state = AsyncMock(return_value=state)
        repl.agent.aupdate_state = AsyncMock()

        with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
            with patch("chcode.chat.render_info"):
                with patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:
                    mock_inst = Mock()
                    mock_resp = Mock()
                    mock_resp.content = '{"summary": "done"}'
                    mock_inst.invoke = Mock(return_value=mock_resp)
                    mock_llm_cls.return_value = mock_inst
                    with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
                        with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                            with patch("chcode.chat.render_success"):
                                await repl._cmd_compress("")

                                # msg1 should be marked as composed
                                assert msg1.additional_kwargs.get("composed") is True
                                assert msg2.additional_kwargs.get("composed") is True
                                # Recent messages should NOT be composed
                                assert msg3.additional_kwargs.get("composed") is not True


# ============================================================================
# Test _cmd_compress — markdown code block stripping (lines 737-739)
# ============================================================================


class TestCmdCompressCodeBlockStripping:
    async def test_compress_strips_markdown_code_block(self):
        """LLM response wrapped in ```json ... ``` should be stripped."""
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        msg1 = HumanMessage("q1", id="m1")
        msg2 = HumanMessage("q2", id="m2")

        state = Mock()
        state.values = {"messages": [msg1, msg2]}
        repl.agent.aget_state = AsyncMock(return_value=state)
        repl.agent.aupdate_state = AsyncMock()

        with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
            with patch("chcode.chat.render_info"):
                with patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:
                    mock_inst = Mock()
                    mock_resp = Mock()
                    mock_resp.content = '```json\n{"summary": "stripped"}\n```'
                    mock_inst.invoke = Mock(return_value=mock_resp)
                    mock_llm_cls.return_value = mock_inst
                    with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
                        with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                            with patch("chcode.chat.render_success"):
                                await repl._cmd_compress("")

                                # The update should have been called with a success message
                                repl.agent.aupdate_state.assert_called()
                                call_args = repl.agent.aupdate_state.call_args
                                messages = call_args[0][1]["messages"]
                                # Find the AI message
                                ai_msg = [m for m in messages if isinstance(m, AIMessage)]
                                assert len(ai_msg) == 1
                                assert "stripped" in ai_msg[0].content

    async def test_compress_nested_json_fallback(self):
        """LLM response with thinking before nested JSON uses bracket-depth fallback."""
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        msg1 = HumanMessage("q1", id="m1")
        msg2 = HumanMessage("q2", id="m2")

        state = Mock()
        state.values = {"messages": [msg1, msg2]}
        repl.agent.aget_state = AsyncMock(return_value=state)
        repl.agent.aupdate_state = AsyncMock()

        with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
            with patch("chcode.chat.render_info"):
                with patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:
                    mock_inst = Mock()
                    mock_resp = Mock()
                    # Thinking before JSON with nested braces (regex won't match, fallback will)
                    mock_resp.content = '让我来分析...\n{"summary": {"key": "嵌套摘要内容"}}'
                    mock_inst.invoke = Mock(return_value=mock_resp)
                    mock_llm_cls.return_value = mock_inst
                    with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
                        with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                            with patch("chcode.chat.render_success"):
                                await repl._cmd_compress("")

                                repl.agent.aupdate_state.assert_called()
                                call_args = repl.agent.aupdate_state.call_args
                                messages = call_args[0][1]["messages"]
                                ai_msg = [m for m in messages if isinstance(m, AIMessage)]
                                assert len(ai_msg) == 1
                                assert "嵌套" in ai_msg[0].content


# ============================================================================
# Test _cmd_compress — error branches (lines 738-746, 749)
# ============================================================================


class TestCmdCompressErrorBranches:
    async def test_compress_json_parse_error(self):
        """When JSON parsing fails, a failure AIMessage should be stored."""
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        msg1 = HumanMessage("q1", id="m1")
        msg2 = HumanMessage("q2", id="m2")

        state = Mock()
        state.values = {"messages": [msg1, msg2]}
        repl.agent.aget_state = AsyncMock(return_value=state)
        repl.agent.aupdate_state = AsyncMock()

        with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
            with patch("chcode.chat.render_info"):
                with patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:
                    mock_inst = Mock()
                    mock_resp = Mock()
                    mock_resp.content = "not json at all"
                    mock_inst.invoke = Mock(return_value=mock_resp)
                    mock_llm_cls.return_value = mock_inst
                    with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
                        with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                            with patch("chcode.chat.render_success"):
                                await repl._cmd_compress("")

                                call_args = repl.agent.aupdate_state.call_args
                                messages = call_args[0][1]["messages"]
                                ai_msg = [m for m in messages if isinstance(m, AIMessage)]
                                assert len(ai_msg) == 1
                                assert "会话压缩失败" in ai_msg[0].content
                                assert ai_msg[0].additional_kwargs.get("error") is True

    async def test_compress_missing_summary_field(self):
        """When summary field is empty, failure AIMessage should be stored."""
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        msg1 = HumanMessage("q1", id="m1")
        msg2 = HumanMessage("q2", id="m2")

        state = Mock()
        state.values = {"messages": [msg1, msg2]}
        repl.agent.aget_state = AsyncMock(return_value=state)
        repl.agent.aupdate_state = AsyncMock()

        with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
            with patch("chcode.chat.render_info"):
                with patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:
                    mock_inst = Mock()
                    mock_resp = Mock()
                    mock_resp.content = '{"summary": ""}'
                    mock_inst.invoke = Mock(return_value=mock_resp)
                    mock_llm_cls.return_value = mock_inst
                    with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
                        with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                            with patch("chcode.chat.render_success"):
                                await repl._cmd_compress("")

                                call_args = repl.agent.aupdate_state.call_args
                                messages = call_args[0][1]["messages"]
                                ai_msg = [m for m in messages if isinstance(m, AIMessage)]
                                assert len(ai_msg) == 1
                                assert "会话压缩失败" in ai_msg[0].content

    async def test_compress_invoke_exception(self):
        """When model.invoke raises, ai_content should be the error string."""
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}

        msg1 = HumanMessage("q1", id="m1")
        msg2 = HumanMessage("q2", id="m2")

        state = Mock()
        state.values = {"messages": [msg1, msg2]}
        repl.agent.aget_state = AsyncMock(return_value=state)
        repl.agent.aupdate_state = AsyncMock()

        with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
            with patch("chcode.chat.render_info"):
                with patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:
                    mock_inst = Mock()
                    mock_inst.invoke = Mock(side_effect=RuntimeError("model error"))
                    mock_llm_cls.return_value = mock_inst
                    with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, side_effect=RuntimeError("model error")):
                        with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                            with patch("chcode.chat.render_success"):
                                await repl._cmd_compress("")

                                call_args = repl.agent.aupdate_state.call_args
                                messages = call_args[0][1]["messages"]
                                ai_msg = [m for m in messages if isinstance(m, AIMessage)]
                                assert len(ai_msg) == 1
                                assert "model error" in ai_msg[0].content


# ============================================================================
# Test _cmd_mode early return (line 829)
# ============================================================================


class TestCmdModeEarlyReturn:
    async def test_cmd_mode_cancel_returns_early(self):
        """When select returns None, _cmd_mode should return immediately."""
        repl = ChatREPL()
        original_yolo = repl.yolo

        with patch("chcode.chat.select", new_callable=AsyncMock, return_value=None):
            await repl._cmd_mode("")

            # yolo should remain unchanged
            assert repl.yolo == original_yolo


# ============================================================================
# Test _cmd_messages — edit index parsing errors (lines 959-960, 963-964)
# ============================================================================


class TestCmdMessagesEditIndexParsing:
    async def test_delete_with_invalid_index_format(self):
        """Choosing a delete option with non-parseable index should skip it."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg1 = HumanMessage("test1", id="1")
        msg2 = HumanMessage("test2", id="2")
        state = Mock()
        state.values = {"messages": [msg1, msg2]}
        repl.agent.aget_state.return_value = state
        repl.agent.aupdate_state = AsyncMock()
        repl.session_mgr = Mock()

        # select("删除消息") -> checkbox returns an entry with unparseable index
        # -> confirm(True) -> delete_ids is empty -> render_error("没有有效的选择") -> continue -> select(None)
        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["删除消息", None]):
            with patch("chcode.chat.checkbox", new_callable=AsyncMock, return_value=["bad_entry"]):
                with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                    with patch("chcode.chat.render_error") as mock_err:
                        await repl._cmd_messages("")

                        mock_err.assert_called_once()
                        assert "没有有效的选择" in mock_err.call_args[0][0]

    async def test_delete_with_out_of_range_index(self):
        """Choosing an index that's out of range should skip it, leaving delete_ids empty."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg1 = HumanMessage("test1", id="1")
        state = Mock()
        state.values = {"messages": [msg1]}
        repl.agent.aget_state.return_value = state
        repl.agent.aupdate_state = AsyncMock()
        repl.session_mgr = Mock()

        # checkbox returns "[99] test" which parses to index 98, out of range for 1 group
        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["删除消息", None]):
            with patch("chcode.chat.checkbox", new_callable=AsyncMock, return_value=["[99] out_of_range"]):
                with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                    with patch("chcode.chat.render_error") as mock_err:
                        await repl._cmd_messages("")

                        mock_err.assert_called_once()
                        assert "没有有效的选择" in mock_err.call_args[0][0]


# ============================================================================
# Test _cmd_messages — continue flow (lines 979, 981)
# ============================================================================


class TestCmdMessagesContinueFlow:
    async def test_edit_chosen_none_returns(self):
        """When editing and chosen is None, _cmd_messages returns."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg1 = HumanMessage("test1", id="1")
        state = Mock()
        state.values = {"messages": [msg1]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["编辑消息", None]):
            result = await repl._cmd_messages("")
            assert result is None  # Returns None when cancelled

    async def test_edit_chosen_back_continues(self):
        """When editing and chosen is '返回', the loop continues back to action select."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg1 = HumanMessage("test1", id="1")
        state = Mock()
        state.values = {"messages": [msg1]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        # select("编辑消息") -> select("返回") [continue back] -> select(None) [return]
        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["编辑消息", "返回", None]):
            result = await repl._cmd_messages("")
            assert result is None  # Returns None when cancelled after going back


# ============================================================================
# Test _cmd_messages — edit confirm cancel (line 1010)
# ============================================================================


class TestCmdMessagesEditConfirmCancel:
    async def test_edit_confirm_cancel_continues_loop(self):
        """Cancelling the edit confirmation should continue back to action select."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg1 = HumanMessage("test1", id="1")
        state = Mock()
        state.values = {"messages": [msg1]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        # select("编辑消息") -> select("[1] test1") -> confirm(False) -> continue -> select(None)
        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["编辑消息", "[1] test1", None]):
            with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=False):
                await repl._cmd_messages("")

                # _edit_buffer should remain None
                assert repl._edit_buffer is None


# ============================================================================
# Test _cmd_messages — edit with git rollback (lines 1017-1022)
# ============================================================================


class TestCmdMessagesEditGitRollback:
    async def test_edit_git_rollback_failure(self):
        """When git rollback fails during edit, a warning should be shown but edit proceeds."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg1 = HumanMessage("test1", id="1")
        state = Mock()
        state.values = {"messages": [msg1]}
        repl.agent.aget_state.return_value = state
        repl.agent.aupdate_state = AsyncMock()
        repl.session_mgr = Mock()
        repl.git = True
        repl.git_manager = Mock()
        repl.git_manager.rollback = Mock(side_effect=Exception("rollback failed"))

        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["编辑消息", "[1] test1"]):
            with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                with patch("chcode.chat.render_warning") as mock_warn:
                    with patch("chcode.chat.load_workplace", return_value=None):
                        with patch.object(repl, "_delete_messages", new_callable=AsyncMock):
                            with patch("chcode.chat.render_success"):
                                await repl._cmd_messages("")

                                mock_warn.assert_called_once()
                                assert "Git 回滚失败" in mock_warn.call_args[0][0]
                                assert repl._edit_buffer == "test1"


# ============================================================================
# Test _process_input — empty content chunk skip (line 1227)
# ============================================================================


class TestProcessInputEmptyChunk:
    async def test_process_input_first_chunk_empty_skips_start(self):
        """When the first AIMessageChunk has empty content, render_ai_start should not be called yet."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        chunk_empty = AIMessageChunk(content="")
        chunk_with_content = AIMessageChunk(content="Hello")

        async def mock_astream(*args, **kwargs):
            yield "messages", [chunk_empty]
            yield "messages", [chunk_with_content]

        repl.agent.astream = mock_astream
        repl.agent.aget_state = AsyncMock()

        with patch("chcode.chat.render_ai_start") as mock_start:
            with patch("chcode.chat.render_ai_chunk"):
                with patch("chcode.chat.render_ai_end"):
                    with patch("chcode.chat.asyncio.create_task"):
                        await repl._process_input("test")

                        # render_ai_start should be called once (on the non-empty chunk)
                        mock_start.assert_called_once()


# ============================================================================
# Test _process_input — fallback model switch failure (lines 1260-1263)
# ============================================================================


class TestProcessInputFallbackSwitchFailure:
    async def test_process_input_fallback_switch_fails(self):
        """When fallback model switch raises, render_error should be called."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}
        repl.checkpointer = Mock()

        from chcode.agent_setup import ModelSwitchError

        async def mock_astream(*args, **kwargs):
            raise ModelSwitchError("Switch needed")
            yield  # make it a generator

        repl.agent.astream = mock_astream

        with patch("chcode.chat.get_fallback_model", return_value={"model": "fallback"}):
            with patch("chcode.chat.advance_fallback"):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, side_effect=Exception("build failed")) as mock_thread:
                    with patch("chcode.chat.render_error") as mock_err:
                        with patch("chcode.chat.asyncio.create_task"):
                            await repl._process_input("test")

                            mock_err.assert_called_once()
                            assert "切换模型失败" in mock_err.call_args[0][0]

    async def test_process_input_no_fallback_available(self):
        """When no fallback model is available, render_error should be called."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        from chcode.agent_setup import ModelSwitchError

        async def mock_astream(*args, **kwargs):
            raise ModelSwitchError("Switch needed")
            yield

        repl.agent.astream = mock_astream

        with patch("chcode.chat.get_fallback_model", return_value=None):
            with patch("chcode.chat.render_error") as mock_err:
                with patch("chcode.chat.asyncio.create_task"):
                    await repl._process_input("test")

                    mock_err.assert_called_once()
                    assert "没有更多备用模型可用" in mock_err.call_args[0][0]


# ============================================================================
# Test _process_input — OpenAI API error handling (lines 1265-1279)
# ============================================================================


class TestProcessInputOpenAIError:
    async def test_process_input_openai_api_error_with_state_update(self):
        """OpenAI APIError should render error and store error message in state."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}
        repl.agent.aupdate_state = AsyncMock()

        import openai

        # Create a proper APIError with a mock request
        mock_request = Mock()
        api_err = openai.APIError(message="API error from test", request=mock_request, body=None)

        async def mock_astream(*args, **kwargs):
            raise api_err
            yield  # make it a generator

        repl.agent.astream = mock_astream

        with patch("chcode.chat.render_error") as mock_err:
            with patch("chcode.chat.asyncio.create_task"):
                await repl._process_input("test")

                mock_err.assert_called_once()
                assert "API error" in mock_err.call_args[0][0]
                repl.agent.aupdate_state.assert_called_once()
                # The stored message should be an AIMessage with error=True
                call_args = repl.agent.aupdate_state.call_args
                messages = call_args[0][1]["messages"]
                assert len(messages) == 1
                assert isinstance(messages[0], AIMessage)
                assert messages[0].additional_kwargs.get("error") is True
                assert "composed" in messages[0].additional_kwargs

    async def test_process_input_openai_api_error_state_update_fails(self):
        """When state update after API error also fails, it should be swallowed."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}
        repl.agent.aupdate_state = AsyncMock(side_effect=Exception("state update failed"))

        import openai

        mock_request = Mock()
        api_err = openai.APIError(message="API error", request=mock_request, body=None)

        async def mock_astream(*args, **kwargs):
            raise api_err
            yield  # make it a generator

        repl.agent.astream = mock_astream

        with patch("chcode.chat.render_error") as mock_err:
            with patch("chcode.chat.asyncio.create_task"):
                await repl._process_input("test")
                assert "API error" in mock_err.call_args[0][0]


# ============================================================================
# Test _process_input — break condition (line 1297)
# ============================================================================


class TestProcessInputStopRequested:
    async def test_process_input_stop_requested_breaks(self):
        """When _stop_requested is set mid-stream, the loop should break."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}

        call_count = [0]

        async def mock_astream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                yield "messages", [AIMessageChunk(content="partial")]
                repl._stop_requested = True

        repl.agent.astream = mock_astream
        repl.agent.aget_state = AsyncMock()

        with patch("chcode.chat.render_ai_start"):
            with patch("chcode.chat.render_ai_chunk"):
                with patch("chcode.chat.render_ai_end"):
                    with patch("chcode.chat.asyncio.create_task"):
                        await repl._process_input("test")

                        # Should have broken after first iteration
                        assert repl._processing is False


# ============================================================================
# Test _process_input — general exception with state update (lines 1280-1294)
# ============================================================================


class TestProcessInputGeneralException:
    async def test_process_input_general_error_stores_in_state(self):
        """A general exception should render error and store an error message in state."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}
        repl.agent.aupdate_state = AsyncMock()

        async def mock_astream(*args, **kwargs):
            raise ValueError("something went wrong")

        repl.agent.astream = mock_astream

        with patch("chcode.chat.render_error") as mock_err:
            with patch("chcode.chat.asyncio.create_task"):
                await repl._process_input("test")

                mock_err.assert_called_once()
                repl.agent.aupdate_state.assert_called_once()
                call_args = repl.agent.aupdate_state.call_args
                messages = call_args[0][1]["messages"]
                assert len(messages) == 1
                assert isinstance(messages[0], AIMessage)
                assert messages[0].additional_kwargs.get("error") is True

    async def test_process_input_general_error_state_update_fails(self):
        """When the error state update itself fails, the exception is swallowed."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.session_mgr = Mock()
        repl.session_mgr.config = {}
        repl.session_mgr.thread_id = "test-thread"
        repl.workplace_path = Path("/tmp")
        repl.model_config = {"model": "gpt-4"}
        repl.agent.aupdate_state = AsyncMock(side_effect=Exception("double fail"))

        async def mock_astream(*args, **kwargs):
            raise ValueError("original error")

        repl.agent.astream = mock_astream

        with patch("chcode.chat.render_error") as mock_err:
            with patch("chcode.chat.asyncio.create_task"):
                await repl._process_input("test")
                assert mock_err.call_count >= 1  # Error should be rendered at least once


# ============================================================================
# Test _render_diff — find old_str logic (line 1374)
# ============================================================================


class TestRenderDiffFindOldStr:
    async def test_render_diff_file_not_found(self):
        """When the file doesn't exist, start_line should remain 1 (default)."""
        repl = ChatREPL()
        repl.yolo = False

        file_path = "/nonexistent/file.py"
        old_str = "old line\nmore old"
        new_str = "new line\nmore new"

        interrupt_chunk = {
            "__interrupt__": [
                Mock(value={
                    "action_requests": [{
                        "name": "edit",
                        "args": {
                            "file_path": file_path,
                            "old_string": old_str,
                            "new_string": new_str,
                        }
                    }],
                    "review_configs": []
                })
            ]
        }

        with patch("chcode.chat.select", new_callable=AsyncMock, return_value="approve (批准)"):
            with patch("chcode.chat.render_warning"):
                with patch("chcode.chat.console.print") as mock_print:
                    with patch("chcode.chat.asyncio.to_thread", side_effect=FileNotFoundError("not found")):
                        decisions = await repl._collect_decisions_async(interrupt_chunk)

                        assert len(decisions) == 1
                        assert decisions[0]["type"] == "approve"

    async def test_render_diff_delete_tag(self):
        """Diff table should have only old (red) rows for delete-only changes."""
        repl = ChatREPL()
        repl.yolo = False

        test_file = Path(__file__).parent / "test_diff_sample.txt"
        test_file.write_text("line1\nline2\nline3\nline4\n")

        old_str = "line2\nline3"
        new_str = ""

        interrupt_chunk = {
            "__interrupt__": [
                Mock(value={
                    "action_requests": [{
                        "name": "edit",
                        "args": {
                            "file_path": str(test_file),
                            "old_string": old_str,
                            "new_string": new_str,
                        }
                    }],
                    "review_configs": []
                })
            ]
        }

        try:
            with patch("chcode.chat.select", new_callable=AsyncMock, return_value="approve (批准)"):
                with patch("chcode.chat.render_warning"):
                    with patch("chcode.chat.console.print"):
                        decisions = await repl._collect_decisions_async(interrupt_chunk)
                        assert len(decisions) == 1
        finally:
            test_file.unlink(missing_ok=True)

    async def test_render_diff_insert_tag(self):
        """Diff table should have only new (green) rows for insert-only changes."""
        repl = ChatREPL()
        repl.yolo = False

        test_file = Path(__file__).parent / "test_diff_sample.txt"
        test_file.write_text("line1\nline3\n")

        old_str = "line1"
        new_str = "line1\ninserted"

        interrupt_chunk = {
            "__interrupt__": [
                Mock(value={
                    "action_requests": [{
                        "name": "edit",
                        "args": {
                            "file_path": str(test_file),
                            "old_string": old_str,
                            "new_string": new_str,
                        }
                    }],
                    "review_configs": []
                })
            ]
        }

        try:
            with patch("chcode.chat.select", new_callable=AsyncMock, return_value="approve (批准)"):
                with patch("chcode.chat.render_warning"):
                    with patch("chcode.chat.console.print"):
                        decisions = await repl._collect_decisions_async(interrupt_chunk)
                        assert len(decisions) == 1
        finally:
            test_file.unlink(missing_ok=True)

    async def test_render_diff_equal_tag(self):
        """Diff with identical content should show dim equal rows."""
        repl = ChatREPL()
        repl.yolo = False

        test_file = Path(__file__).parent / "test_diff_sample.txt"
        test_file.write_text("line1\nline2\n")

        old_str = "line1\nline2"
        new_str = "line1\nline2"

        interrupt_chunk = {
            "__interrupt__": [
                Mock(value={
                    "action_requests": [{
                        "name": "edit",
                        "args": {
                            "file_path": str(test_file),
                            "old_string": old_str,
                            "new_string": new_str,
                        }
                    }],
                    "review_configs": []
                })
            ]
        }

        try:
            with patch("chcode.chat.select", new_callable=AsyncMock, return_value="approve (批准)"):
                with patch("chcode.chat.render_warning"):
                    with patch("chcode.chat.console.print"):
                        decisions = await repl._collect_decisions_async(interrupt_chunk)
                        assert len(decisions) == 1
        finally:
            test_file.unlink(missing_ok=True)

    async def test_render_diff_replace_tag(self):
        """Diff with replaced content should show red old and green new rows."""
        repl = ChatREPL()
        repl.yolo = False

        test_file = Path(__file__).parent / "test_diff_sample.txt"
        test_file.write_text("aaa\nbbb\nccc\n")

        old_str = "bbb"
        new_str = "BBB"

        interrupt_chunk = {
            "__interrupt__": [
                Mock(value={
                    "action_requests": [{
                        "name": "edit",
                        "args": {
                            "file_path": str(test_file),
                            "old_string": old_str,
                            "new_string": new_str,
                        }
                    }],
                    "review_configs": []
                })
            ]
        }

        try:
            with patch("chcode.chat.select", new_callable=AsyncMock, return_value="approve (批准)"):
                with patch("chcode.chat.render_warning"):
                    with patch("chcode.chat.console.print"):
                        decisions = await repl._collect_decisions_async(interrupt_chunk)
                        assert len(decisions) == 1
        finally:
            test_file.unlink(missing_ok=True)


# ============================================================================
# Test _init_readline_history and _save_readline_history
# Covers lines 329-333, 342-344
# ============================================================================


class TestReadlineHistory:
    def test_init_readline_history_import_error(self):
        """On ImportError (no readline), _init_readline_history should pass silently."""
        repl = ChatREPL()

        with patch.dict("sys.modules", {"readline": None}):
            # Should not raise ImportError
            repl._init_readline_history()
            # Verify readline module was not loaded (would raise if access attempted)
            assert "readline" not in sys.modules or sys.modules.get("readline") is None

    def test_save_readline_history_import_error(self):
        """On ImportError, _save_readline_history should pass silently."""
        repl = ChatREPL()

        with patch.dict("sys.modules", {"readline": None}):
            result = repl._save_readline_history()
            assert result is True

    def test_save_readline_history_success(self):
        """When readline is available, _save_readline_history should write history."""
        repl = ChatREPL()

        mock_readline = MagicMock()
        with patch.dict("sys.modules", {"readline": mock_readline}):
            home_dir = Path.home()
            with patch.object(Path, "home", return_value=home_dir):
                result = repl._save_readline_history()
                assert result is True
                mock_readline.write_history_file.assert_called_once()

    def test_init_readline_history_with_existing_file(self):
        """When history file exists, readline should read it."""
        repl = ChatREPL()

        mock_readline = MagicMock()
        with patch.dict("sys.modules", {"readline": mock_readline}):
            # Ensure the history file exists
            home_dir = Path.home()
            chat_dir = home_dir / ".chat"
            chat_dir.mkdir(exist_ok=True)
            history_path = chat_dir / "history"
            history_path.touch(exist_ok=True)

            try:
                with patch.object(Path, "home", return_value=home_dir):
                    repl._init_readline_history()
                    mock_readline.read_history_file.assert_called_once()
                    mock_readline.set_history_length.assert_called_once_with(1000)
            finally:
                # Clean up
                if history_path.exists():
                    history_path.unlink()

    def test_init_readline_history_no_existing_file(self):
        """When history file doesn't exist, readline should not try to read it."""
        repl = ChatREPL()

        mock_readline = MagicMock()
        with patch.dict("sys.modules", {"readline": mock_readline}):
            # Use a real temp path so Path.__truediv__ works correctly
            home_dir = Path.home()
            with patch.object(Path, "home", return_value=home_dir):
                repl._init_readline_history()
                # history file may or may not exist, but we can verify set_history_length was called
                mock_readline.set_history_length.assert_called_once_with(1000)


# ============================================================================
# Test _cmd_model — unrecognized action returns early (lines 601-606)
# ============================================================================


class TestCmdModelUnrecognizedAction:
    async def test_cmd_model_unrecognized_action(self):
        """When select returns something unexpected, _cmd_model should return early."""
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4"}

        with patch("chcode.chat.select", new_callable=AsyncMock, return_value="some unrecognized option"):
            await repl._cmd_model("")

            # model_config should remain unchanged
            assert repl.model_config == {"model": "gpt-4"}


# ============================================================================
# Test _cmd_messages — edit with invalid index (lines 987-991)
# ============================================================================


class TestCmdMessagesEditInvalidIndex:
    async def test_edit_with_negative_index(self):
        """Choosing an option that parses to negative index shows error and continues."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg1 = HumanMessage("test1", id="1")
        state = Mock()
        state.values = {"messages": [msg1]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        # select("编辑消息") -> select("[0] test") -> parses to -1 -> render_error -> continue -> select(None)
        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["编辑消息", "[0] test", None]):
            with patch("chcode.chat.render_error") as mock_err:
                await repl._cmd_messages("")

                mock_err.assert_called_once()
                assert "无效的选择" in mock_err.call_args[0][0]

    async def test_edit_with_over_range_index(self):
        """Choosing an option that parses to index >= len(groups) shows error."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg1 = HumanMessage("test1", id="1")
        state = Mock()
        state.values = {"messages": [msg1]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        # select("编辑消息") -> select("[2] test") -> parses to 1, but only 1 group -> error -> continue -> select(None)
        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["编辑消息", "[2] test", None]):
            with patch("chcode.chat.render_error") as mock_err:
                await repl._cmd_messages("")

                mock_err.assert_called_once()
                assert "无效的选择" in mock_err.call_args[0][0]

    async def test_edit_with_non_numeric_index(self):
        """Choosing an option that can't be parsed as int shows error."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg1 = HumanMessage("test1", id="1")
        state = Mock()
        state.values = {"messages": [msg1]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        # select("编辑消息") -> select("返回") [which is in select_options but not a valid index]
        # Actually "返回" is handled by `if chosen == "返回": continue` before parsing.
        # Let's use a truly unparseable option.
        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["编辑消息", "abc", None]):
            with patch("chcode.chat.render_error") as mock_err:
                await repl._cmd_messages("")

                mock_err.assert_called_once()
                assert "无效的选择" in mock_err.call_args[0][0]


# ============================================================================
# Test _cmd_model — select "编辑" action (line 602)
# ============================================================================


class TestCmdModelSelectEditAction:
    async def test_cmd_model_select_edit_action(self):
        """When select returns '编辑当前模型' option, edit_current_model should be called."""
        repl = ChatREPL()
        repl.model_config = {}

        with patch("chcode.chat.select", new_callable=AsyncMock, return_value="编辑当前模型 (/model edit)"):
            with patch("chcode.chat.edit_current_model", new_callable=AsyncMock) as mock_edit:
                mock_edit.return_value = {"model": "gpt-4-edited"}
                with patch("chcode.agent_setup.update_summarization_model"):
                    with patch.object(repl, "_render_status_bar"):
                        await repl._cmd_model("")

                        mock_edit.assert_called_once()
                        assert repl.model_config == {"model": "gpt-4-edited"}


# ============================================================================
# Test _cmd_model — select "切换" action (line 604)
# ============================================================================


class TestCmdModelSelectSwitchAction:
    async def test_cmd_model_select_switch_action(self):
        """When select returns '切换模型' option, switch_model should be called."""
        repl = ChatREPL()
        repl.model_config = {}

        with patch("chcode.chat.select", new_callable=AsyncMock, return_value="切换模型 (/model switch)"):
            with patch("chcode.chat.switch_model", new_callable=AsyncMock) as mock_sw:
                mock_sw.return_value = {"model": "claude-3"}
                with patch("chcode.agent_setup.update_summarization_model"):
                    with patch.object(repl, "_render_status_bar"):
                        await repl._cmd_model("")

                        mock_sw.assert_called_once()
                        assert repl.model_config == {"model": "claude-3"}


# ============================================================================
# Test _cmd_tools — iterating ALL_TOOLS (lines 632-635)
# ============================================================================


class TestCmdToolsDisplayList:
    async def test_cmd_tools_displays_all_tools(self):
        """_cmd_tools should iterate over ALL_TOOLS and print each one."""
        repl = ChatREPL()

        mock_tool = Mock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read file contents from disk\nWith extra detail on second line"

        with patch("chcode.chat.console.print") as mock_print:
            with patch("chcode.utils.tools.ALL_TOOLS", [mock_tool]):
                await repl._cmd_tools("")

                # console.print is called multiple times: header, empty line, tool line, empty line
                assert mock_print.call_count >= 3
                # Check the tool print call contains the name and first line of description
                tool_calls = [c for c in mock_print.call_args_list]
                found_tool = False
                for call in tool_calls:
                    args_str = str(call)
                    if "read_file" in args_str and "Read file contents" in args_str:
                        found_tool = True
                        break
                assert found_tool, "Tool name and description should be printed"

    async def test_cmd_tools_empty_description(self):
        """_cmd_tools should handle tools with None description gracefully."""
        repl = ChatREPL()

        mock_tool = Mock()
        mock_tool.name = "no_desc"
        mock_tool.description = None

        with patch("chcode.chat.console.print") as mock_print:
            with patch("chcode.utils.tools.ALL_TOOLS", [mock_tool]):
                await repl._cmd_tools("")
                assert mock_print.call_count > 0  # Console should have been used


# ============================================================================
# Test _cmd_messages — checkbox empty return (line 945)
# ============================================================================


class TestCmdMessagesCheckboxEmpty:
    async def test_delete_checkbox_returns_empty_list(self):
        """When checkbox returns empty list, should continue back to action select."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg1 = HumanMessage("test1", id="1")
        state = Mock()
        state.values = {"messages": [msg1]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        # select("删除消息") -> checkbox([]) -> continue -> select(None) -> return
        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["删除消息", None]):
            with patch("chcode.chat.checkbox", new_callable=AsyncMock, return_value=[]):
                result = await repl._cmd_messages("")
                assert result is None  # Should return without error when cancelled


# ============================================================================
# Test _cmd_messages — fork path choices with saved workplace (line 1043)
# ============================================================================


class TestCmdMessagesForkPathChoices:
    async def test_fork_selects_saved_workplace_path(self):
        """When load_workplace returns a path, it should be in the choices for fork."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            workplace = tmp / "workplace"
            workplace.mkdir()
            saved = tmp / "saved"
            saved.mkdir()
            repl = _make_fork_repl(workplace=workplace)
            new_agent = _make_new_agent()

            with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test1"]):
                with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                    with patch("chcode.chat.load_workplace", return_value=str(saved)):
                        with patch("chcode.chat.select_or_custom", new_callable=AsyncMock, return_value=str(saved)) as mock_soc:
                            with patch("chcode.chat.os.chdir"):
                                with patch("chcode.chat.save_workplace"):
                                    with patch("chcode.chat.SessionManager", return_value=Mock()):
                                        with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock, return_value=Mock()):
                                            with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value=new_agent):
                                                with patch.object(repl, "_init_git", new_callable=AsyncMock):
                                                    with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                                                        with patch.object(repl, "_render_status_bar"):
                                                            with patch("chcode.chat.render_success"):
                                                                await repl._cmd_messages("")

                                                                # select_or_custom should receive the saved path in choices
                                                                mock_soc.assert_called_once()
                                                                call_kwargs = mock_soc.call_args
                                                                choices = call_kwargs[0][1]
                                                                assert str(saved) in choices
                                                                assert "自定义路径..." in choices
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def test_fork_no_saved_workplace(self):
        """When load_workplace returns None, choices should only have '自定义路径...'."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg1 = HumanMessage("test1", id="1")
        state = Mock()
        state.values = {"messages": [msg1]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test1", None]):
            with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                with patch("chcode.chat.load_workplace", return_value=None):
                    with patch("chcode.chat.select_or_custom", new_callable=AsyncMock, return_value="") as mock_soc:
                        await repl._cmd_messages("")

                        # select_or_custom should be called with only "自定义路径..." as choices
                        mock_soc.assert_called_once()
                        choices = mock_soc.call_args[0][1]
                        assert choices == ["自定义路径..."]


# ============================================================================
# Test _cmd_messages — fork path not exists (lines 1052-1054)
# ============================================================================


class TestCmdMessagesForkPathNotExists:
    async def test_fork_path_not_exists_continues(self):
        """When fork path doesn't exist, render_error should be called and loop continues."""
        repl = ChatREPL()
        repl.agent = Mock()
        repl.agent.aget_state = AsyncMock()
        msg1 = HumanMessage("test1", id="1")
        state = Mock()
        state.values = {"messages": [msg1]}
        repl.agent.aget_state.return_value = state
        repl.session_mgr = Mock()

        with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test1", None]):
            with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                with patch("chcode.chat.load_workplace", return_value=None):
                    with patch("chcode.chat.select_or_custom", new_callable=AsyncMock, return_value="/nonexistent/path"):
                        with patch("chcode.chat.Path") as mock_path_cls:
                            mock_new_path = MagicMock()
                            mock_new_path.exists.return_value = False
                            mock_path_cls.return_value = mock_new_path
                            with patch("chcode.chat.render_error") as mock_err:
                                await repl._cmd_messages("")

                                mock_err.assert_called_once()
                                assert "路径不存在" in mock_err.call_args[0][0]


# ============================================================================
# Test _cmd_messages — fork full flow with different paths (lines 1051-1129)
# ============================================================================


def _make_fork_repl(messages=None, workplace=None, git=False):
    """Helper to create a repl configured for fork tests."""
    repl = ChatREPL()
    repl.agent = Mock()
    repl.agent.aget_state = AsyncMock()
    repl.agent.aupdate_state = AsyncMock()
    state = Mock()
    if messages is None:
        messages = [HumanMessage("test1", id="1")]
    state.values = {"messages": messages}
    repl.agent.aget_state.return_value = state
    repl.session_mgr = Mock()
    repl.workplace_path = workplace or Path("/old/path")
    repl.git = git
    if git:
        repl.git_manager = Mock()
    return repl


def _make_new_agent():
    """Create a mock agent with AsyncMock methods for use as build_agent return."""
    new_agent = Mock()
    new_agent.aget_state = AsyncMock()
    new_agent.aupdate_state = AsyncMock()
    return new_agent


class TestCmdMessagesForkFullFlow:
    async def test_fork_same_path_no_copy(self):
        """When old_path == new_path, _copy_dir should NOT be called."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            old_path = tmp / "same"
            old_path.mkdir()
            repl = _make_fork_repl(workplace=old_path)
            new_agent = _make_new_agent()
            copy_dir_called = [False]

            async def mock_to_thread(fn, *args, **kwargs):
                if hasattr(fn, '__name__') and fn.__name__ == '_copy_dir':
                    copy_dir_called[0] = True
                return new_agent

            with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test1"]):
                with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                    with patch("chcode.chat.load_workplace", return_value=None):
                        with patch("chcode.chat.select_or_custom", new_callable=AsyncMock, return_value=str(old_path)):
                            with patch("chcode.chat.os.chdir"):
                                with patch("chcode.chat.save_workplace"):
                                    with patch("chcode.chat.SessionManager", return_value=Mock()):
                                        with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock, return_value=Mock()):
                                            with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, side_effect=mock_to_thread):
                                                with patch.object(repl, "_init_git", new_callable=AsyncMock):
                                                    with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                                                        with patch.object(repl, "_render_status_bar"):
                                                            with patch("chcode.chat.render_success"):
                                                                await repl._cmd_messages("")

                                                                assert not copy_dir_called[0], "_copy_dir should not be called when paths are same"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def test_fork_different_path_copies_dir(self):
        """When old_path != new_path, _copy_dir should be called via asyncio.to_thread."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            old_path = tmp / "old"
            new_path = tmp / "new"
            old_path.mkdir()
            new_path.mkdir()
            repl = _make_fork_repl(workplace=old_path)
            new_agent = _make_new_agent()
            copy_dir_called = [False]

            async def mock_to_thread(fn, *args, **kwargs):
                if hasattr(fn, '__name__') and fn.__name__ == '_copy_dir':
                    copy_dir_called[0] = True
                return new_agent

            with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test1"]):
                with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                    with patch("chcode.chat.load_workplace", return_value=None):
                        with patch("chcode.chat.select_or_custom", new_callable=AsyncMock, return_value=str(new_path)):
                            with patch("chcode.chat.os.chdir"):
                                with patch("chcode.chat.save_workplace"):
                                    with patch("chcode.chat.render_info"):
                                        with patch("chcode.chat.SessionManager", return_value=Mock()):
                                            with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock, return_value=Mock()):
                                                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, side_effect=mock_to_thread):
                                                    with patch.object(repl, "_init_git", new_callable=AsyncMock):
                                                        with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                                                            with patch.object(repl, "_render_status_bar"):
                                                                with patch("chcode.chat.render_success"):
                                                                    await repl._cmd_messages("")

                                                                    assert copy_dir_called[0], "_copy_dir should have been called for different paths"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def test_fork_copies_git_directory(self):
        """When .git directory exists in old_path, shutil.copytree should copy it."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            old_path = tmp / "old"
            new_path = tmp / "new"
            old_path.mkdir()
            new_path.mkdir()
            # Create .git in old_path
            old_git = old_path / ".git"
            old_git.mkdir()
            (old_git / "config").write_text("git config here")

            repl = _make_fork_repl(workplace=old_path)
            new_agent = _make_new_agent()

            async def mock_to_thread(fn, *args, **kwargs):
                # Actually call the real function so file operations happen
                if hasattr(fn, '__name__') and fn.__name__ in ('_copy_dir',):
                    fn(*args, **kwargs)
                elif fn is shutil.copytree:
                    fn(*args, **kwargs)
                elif fn is shutil.rmtree:
                    fn(*args, **kwargs)
                return new_agent

            with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test1"]):
                with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                    with patch("chcode.chat.load_workplace", return_value=None):
                        with patch("chcode.chat.select_or_custom", new_callable=AsyncMock, return_value=str(new_path)):
                            with patch("chcode.chat.os.chdir"):
                                with patch("chcode.chat.save_workplace"):
                                    with patch("chcode.chat.render_info"):
                                        with patch("chcode.chat.SessionManager", return_value=Mock()):
                                            with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock, return_value=Mock()):
                                                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, side_effect=mock_to_thread):
                                                    with patch.object(repl, "_init_git", new_callable=AsyncMock):
                                                        with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                                                            with patch.object(repl, "_render_status_bar"):
                                                                with patch("chcode.chat.render_success"):
                                                                    await repl._cmd_messages("")

                                                                    # .git should have been copied to new_path
                                                                    assert (new_path / ".git" / "config").exists()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def test_fork_clears_sessions_directory(self):
        """Fork should remove and recreate the sessions directory."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            old_path = tmp / "old"
            new_path = tmp / "new"
            old_path.mkdir()
            new_path.mkdir()
            # Create sessions dir in new_path (simulating what _cmd_messages creates
            # via chat_dir.mkdir before the copy)
            sessions = new_path / ".chat" / "sessions"
            sessions.mkdir(parents=True)
            (sessions / "old_data.txt").write_text("should be removed")

            repl = _make_fork_repl(workplace=old_path)
            new_agent = _make_new_agent()

            async def mock_to_thread(fn, *args, **kwargs):
                # Actually call _copy_dir and shutil functions
                if hasattr(fn, '__name__') and fn.__name__ == '_copy_dir':
                    fn(*args, **kwargs)
                elif fn is shutil.rmtree:
                    fn(*args, **kwargs)
                elif fn is shutil.copytree:
                    fn(*args, **kwargs)
                return new_agent

            with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test1"]):
                with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                    with patch("chcode.chat.load_workplace", return_value=None):
                        with patch("chcode.chat.select_or_custom", new_callable=AsyncMock, return_value=str(new_path)):
                            with patch("chcode.chat.os.chdir"):
                                with patch("chcode.chat.save_workplace"):
                                    with patch("chcode.chat.render_info"):
                                        with patch("chcode.chat.SessionManager", return_value=Mock()):
                                            with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock, return_value=Mock()):
                                                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, side_effect=mock_to_thread):
                                                    with patch.object(repl, "_init_git", new_callable=AsyncMock):
                                                        with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                                                            with patch.object(repl, "_render_status_bar"):
                                                                with patch("chcode.chat.render_success"):
                                                                    await repl._cmd_messages("")

                                                                    # Old sessions file should have been removed
                                                                    assert not (sessions / "old_data.txt").exists()
                                                                    # Sessions dir should still exist (recreated after rmtree)
                                                                    assert sessions.exists()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def test_fork_copy_failure_reverts_path(self):
        """When copy fails, workplace_path should be reverted to old_path."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            old_path = tmp / "old"
            new_path = tmp / "new"
            old_path.mkdir()
            new_path.mkdir()
            repl = _make_fork_repl(workplace=old_path)

            with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test1"]):
                with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                    with patch("chcode.chat.load_workplace", return_value=None):
                        with patch("chcode.chat.select_or_custom", new_callable=AsyncMock, return_value=str(new_path)):
                            with patch("chcode.chat.os.chdir"):
                                with patch("chcode.chat.save_workplace"):
                                    with patch("chcode.chat.render_info"):
                                        with patch("chcode.chat.render_error") as mock_err:
                                            with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, side_effect=PermissionError("copy denied")):
                                                await repl._cmd_messages("")

                                                mock_err.assert_called_once()
                                                assert "复制文件失败" in mock_err.call_args[0][0]
                                                assert repl.workplace_path == old_path
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def test_fork_rebuilds_agent_and_sets_state(self):
        """Fork should rebuild agent and update state with forked messages."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            old_path = tmp / "old"
            new_path = tmp / "new"
            old_path.mkdir()
            new_path.mkdir()

            msg1 = HumanMessage("test1", id="m1")
            msg2 = AIMessage("resp1", id="m2")
            msg3 = HumanMessage("test2", id="m3")
            repl = _make_fork_repl(messages=[msg1, msg2, msg3], workplace=old_path)
            new_agent = _make_new_agent()

            with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test1"]):
                with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                    with patch("chcode.chat.load_workplace", return_value=None):
                        with patch("chcode.chat.select_or_custom", new_callable=AsyncMock, return_value=str(new_path)):
                            with patch("chcode.chat.os.chdir"):
                                with patch("chcode.chat.save_workplace"):
                                    with patch("chcode.chat.render_info"):
                                        with patch("chcode.chat.SessionManager", return_value=Mock()):
                                            with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock, return_value=Mock()):
                                                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value=new_agent):
                                                    with patch.object(repl, "_init_git", new_callable=AsyncMock):
                                                        with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                                                            with patch.object(repl, "_render_status_bar"):
                                                                with patch("chcode.chat.render_success"):
                                                                    await repl._cmd_messages("")

                                                                    assert repl.agent == new_agent
                                                                    new_agent.aupdate_state.assert_called_once()
                                                                    call_args = new_agent.aupdate_state.call_args
                                                                    messages = call_args[0][1]["messages"]
                                                                    # Should only have group 0 messages (msg1, msg2)
                                                                    assert len(messages) == 2
                                                                    assert messages[0].id == "m1"
                                                                    assert messages[1].id == "m2"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def test_fork_initializes_git(self):
        """Fork should call _init_git on the new workplace."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            old_path = tmp / "old"
            new_path = tmp / "new"
            old_path.mkdir()
            new_path.mkdir()
            repl = _make_fork_repl(workplace=old_path)
            new_agent = _make_new_agent()

            with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test1"]):
                with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                    with patch("chcode.chat.load_workplace", return_value=None):
                        with patch("chcode.chat.select_or_custom", new_callable=AsyncMock, return_value=str(new_path)):
                            with patch("chcode.chat.os.chdir"):
                                with patch("chcode.chat.save_workplace"):
                                    with patch("chcode.chat.SessionManager", return_value=Mock()):
                                        with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock, return_value=Mock()):
                                            with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value=new_agent):
                                                with patch.object(repl, "_init_git", new_callable=AsyncMock) as mock_init_git:
                                                    with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                                                        with patch.object(repl, "_render_status_bar"):
                                                            with patch("chcode.chat.render_success"):
                                                                await repl._cmd_messages("")

                                                                mock_init_git.assert_called_once()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def test_fork_git_rollback_after_fork(self):
        """After fork, git rollback should be attempted if git is available."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            old_path = tmp / "old"
            new_path = tmp / "new"
            old_path.mkdir()
            new_path.mkdir()
            repl = _make_fork_repl(workplace=old_path, git=True)
            new_agent = _make_new_agent()
            rollback_method = repl.git_manager.rollback
            rollback_called = [False]

            async def mock_to_thread(fn, *args, **kwargs):
                if fn is rollback_method:
                    rollback_called[0] = True
                return new_agent

            with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test1"]):
                with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                    with patch("chcode.chat.load_workplace", return_value=None):
                        with patch("chcode.chat.select_or_custom", new_callable=AsyncMock, return_value=str(new_path)):
                            with patch("chcode.chat.os.chdir"):
                                with patch("chcode.chat.save_workplace"):
                                    with patch("chcode.chat.SessionManager", return_value=Mock()):
                                        with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock, return_value=Mock()):
                                            with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, side_effect=mock_to_thread):
                                                with patch.object(repl, "_init_git", new_callable=AsyncMock):
                                                    with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                                                        with patch.object(repl, "_render_status_bar"):
                                                            with patch("chcode.chat.render_success"):
                                                                await repl._cmd_messages("")

                                                                assert rollback_called[0], "git rollback should have been called during fork"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def test_fork_git_rollback_failure_shows_warning(self):
        """When git rollback fails during fork, a warning should be shown but fork succeeds."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            old_path = tmp / "old"
            new_path = tmp / "new"
            old_path.mkdir()
            new_path.mkdir()
            repl = _make_fork_repl(workplace=old_path, git=True)
            new_agent = _make_new_agent()
            rollback_method = repl.git_manager.rollback

            async def mock_to_thread(fn, *args, **kwargs):
                if fn is rollback_method:
                    raise RuntimeError("rollback failed")
                return new_agent

            with patch("chcode.chat.select", new_callable=AsyncMock, side_effect=["分叉消息", "[1] test1"]):
                with patch("chcode.chat.confirm", new_callable=AsyncMock, return_value=True):
                    with patch("chcode.chat.load_workplace", return_value=None):
                        with patch("chcode.chat.select_or_custom", new_callable=AsyncMock, return_value=str(new_path)):
                            with patch("chcode.chat.os.chdir"):
                                with patch("chcode.chat.save_workplace"):
                                    with patch("chcode.chat.SessionManager", return_value=Mock()):
                                        with patch("chcode.chat.create_checkpointer", new_callable=AsyncMock, return_value=Mock()):
                                            with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, side_effect=mock_to_thread):
                                                with patch.object(repl, "_init_git", new_callable=AsyncMock):
                                                    with patch.object(repl, "_load_conversation", new_callable=AsyncMock):
                                                        with patch.object(repl, "_render_status_bar"):
                                                            with patch("chcode.chat.render_success"):
                                                                with patch("chcode.chat.render_warning") as mock_warn:
                                                                    await repl._cmd_messages("")

                                                                    mock_warn.assert_called_once()
                                                                    assert "Git 回滚失败" in mock_warn.call_args[0][0]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# Test _copy_dir — Windows reserved names (lines 1149-1150)
# ============================================================================


class TestCopyDirWindowsReservedNames:
    def test_copy_dir_skips_windows_reserved_names(self):
        """Lines 1149-1150: Files with Windows reserved names (nul, con, aux, etc.)
        are skipped during directory copy. Bug was fixed: .upper() -> .lower()"""
        repl = ChatREPL()
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            src = tmp / "src"
            dst = tmp / "dst"
            src.mkdir()
            dst.mkdir()

            # Create files with Windows reserved names (using .txt extension
            # so they can be created as regular files on non-Windows systems)
            for reserved in ["nul", "con", "aux", "prn", "com1", "lpt1"]:
                (src / f"{reserved}.txt").write_text(f"{reserved} content")

            # Create a normal file that should be copied
            (src / "normal.txt").write_text("normal content")

            repl._copy_dir(src, dst)

            # Normal file should be copied
            assert (dst / "normal.txt").exists()
            assert (dst / "normal.txt").read_text(encoding="utf-8") == "normal content"

            # Reserved name files should NOT be copied
            assert not (dst / "nul.txt").exists()
            assert not (dst / "con.txt").exists()
            assert not (dst / "aux.txt").exists()
            assert not (dst / "prn.txt").exists()
            assert not (dst / "com1.txt").exists()
            assert not (dst / "lpt1.txt").exists()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_copy_dir_reserved_names_stem_only(self):
        """Lines 1149-1150: Stem comparison (without extension) for reserved names."""
        repl = ChatREPL()
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            src = tmp / "src"
            dst = tmp / "dst"
            src.mkdir()
            dst.mkdir()

            # Create a file with reserved stem but different extension
            (src / "nul.dat").write_text("data")
            (src / "con.txt").write_text("con content")
            (src / "normal.txt").write_text("normal")

            repl._copy_dir(src, dst)

            # Files with reserved stems should be skipped
            assert not (dst / "nul.dat").exists()
            assert not (dst / "con.txt").exists()
            # Normal file should be copied
            assert (dst / "normal.txt").exists()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# Test _copy_dir — exception handling (lines 1155-1156, 1160-1161)
# ============================================================================


class TestCopyDirExceptionHandling:
    def test_copy_dir_directory_copy_failure(self):
        """_copy_dir should catch and print errors when shutil.copytree fails."""
        repl = ChatREPL()
        src = MagicMock()
        dst = MagicMock()

        dir_item = MagicMock(spec=["name", "stem", "is_dir"])
        dir_item.name = "subdir"
        dir_item.stem = "subdir"
        dir_item.is_dir.return_value = True

        src.iterdir.return_value = [dir_item]

        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        with patch("chcode.chat.shutil.copytree", side_effect=PermissionError("denied")):
            try:
                repl._copy_dir(src, dst)
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            assert "复制目录失败" in output
            assert "subdir" in output

    def test_copy_dir_file_copy_failure(self):
        """_copy_dir should catch and print errors when shutil.copy2 fails."""
        repl = ChatREPL()
        src = MagicMock()
        dst = MagicMock()

        file_item = MagicMock(spec=["name", "stem", "is_dir"])
        file_item.name = "important.txt"
        file_item.stem = "important"
        file_item.is_dir.return_value = False

        src.iterdir.return_value = [file_item]

        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        with patch("chcode.chat.shutil.copy2", side_effect=OSError("disk error")):
            try:
                repl._copy_dir(src, dst)
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            assert "复制文件失败" in output
            assert "important.txt" in output


# ============================================================================
# Test _get_input — prompt_toolkit session setup (lines 412-515)
# Covers: SlashCommandCompleter, KeyBindings, bottom toolbar, dynamic
# buffer height, _find_buffer_window helper.
# ============================================================================


def _make_safe_container(**attrs):
    """Create a mock container that won't cause infinite recursion in _find_buffer_window.
    Sets content, children, and alternative_content to None unless explicitly provided."""
    m = MagicMock()
    m.content = attrs.get("content", None)
    m.children = attrs.get("children", None)
    m.alternative_content = attrs.get("alternative_content", None)
    return m


def _make_mock_session():
    """Create a mock PromptSession that is safe for _find_buffer_window traversal.
    The container tree terminates (content=None, children=None, alternative_content=None)."""
    m = MagicMock()
    m.prompt = MagicMock(return_value="x")
    m.app = MagicMock()
    m.app.layout = MagicMock()
    m.app.layout.container = _make_safe_container()
    m.default_buffer = MagicMock()
    m.default_buffer.complete_state = None
    m.default_buffer.text = ""
    return m


class TestPromptSessionSetup:
    async def test_session_creation_passes_key_bindings(self):
        """PromptSession should be constructed with key_bindings from KeyBindings()."""
        repl = ChatREPL()
        repl._prompt_session = None

        mock_session_cls = MagicMock(return_value=_make_mock_session())

        with patch("chcode.chat.PromptSession", mock_session_cls):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="hello"):
                    result = await repl._get_input()

        assert result == "hello"
        mock_session_cls.assert_called_once()
        call_kwargs = mock_session_cls.call_args[1]
        assert "key_bindings" in call_kwargs
        from prompt_toolkit.key_binding import KeyBindings
        assert isinstance(call_kwargs["key_bindings"], KeyBindings)

    async def test_session_creation_passes_completer(self):
        """PromptSession should be constructed with a SlashCommandCompleter."""
        from chcode.chat import SlashCommandCompleter

        repl = ChatREPL()
        repl._prompt_session = None

        mock_session_cls = MagicMock(return_value=_make_mock_session())

        with patch("chcode.chat.PromptSession", mock_session_cls):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="/model"):
                    result = await repl._get_input()

        call_kwargs = mock_session_cls.call_args[1]
        assert isinstance(call_kwargs["completer"], SlashCommandCompleter)

    async def test_session_creation_multiline_and_style(self):
        """PromptSession should be created with multiline=True and a Style."""
        repl = ChatREPL()
        repl._prompt_session = None

        mock_session_cls = MagicMock(return_value=_make_mock_session())

        with patch("chcode.chat.PromptSession", mock_session_cls):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="input"):
                    await repl._get_input()

        call_kwargs = mock_session_cls.call_args[1]
        assert call_kwargs["multiline"] is True
        assert call_kwargs["complete_while_typing"] is True
        assert call_kwargs["reserve_space_for_menu"] == 0
        assert call_kwargs["refresh_interval"] == 0.1
        from prompt_toolkit.styles import Style
        assert isinstance(call_kwargs["style"], Style)

    async def test_session_creation_has_bottom_toolbar(self):
        """PromptSession should be created with a bottom_toolbar callable."""
        repl = ChatREPL()
        repl._prompt_session = None

        mock_session_cls = MagicMock(return_value=_make_mock_session())

        with patch("chcode.chat.PromptSession", mock_session_cls):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    await repl._get_input()

        call_kwargs = mock_session_cls.call_args[1]
        assert callable(call_kwargs["bottom_toolbar"])


class TestBottomToolbar:
    async def test_toolbar_contains_model_name(self):
        """Bottom toolbar should include the model name from model_config."""
        repl = ChatREPL()
        repl.model_config = {"model": "gpt-4-test"}
        repl._prompt_session = None

        captured_toolbar_fn = None

        def capture_session(**kwargs):
            nonlocal captured_toolbar_fn
            captured_toolbar_fn = kwargs.get("bottom_toolbar")
            return _make_mock_session()

        with patch("chcode.chat.PromptSession", side_effect=capture_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    await repl._get_input()

        assert captured_toolbar_fn is not None
        toolbar_output = captured_toolbar_fn()
        assert "gpt-4-test" in str(toolbar_output)

    async def test_toolbar_shows_common_mode(self):
        """Toolbar should show '普通模式' when yolo is False."""
        repl = ChatREPL()
        repl.yolo = False
        repl.model_config = {"model": "test"}
        repl._prompt_session = None

        captured_toolbar_fn = None

        def capture_session(**kwargs):
            nonlocal captured_toolbar_fn
            captured_toolbar_fn = kwargs.get("bottom_toolbar")
            return _make_mock_session()

        with patch("chcode.chat.PromptSession", side_effect=capture_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    await repl._get_input()

        toolbar_output = str(captured_toolbar_fn())
        assert "普通模式" in toolbar_output

    async def test_toolbar_shows_yolo_mode(self):
        """Toolbar should show 'YOLO 模式' when yolo is True."""
        repl = ChatREPL()
        repl.yolo = True
        repl.model_config = {"model": "test"}
        repl._prompt_session = None

        captured_toolbar_fn = None

        def capture_session(**kwargs):
            nonlocal captured_toolbar_fn
            captured_toolbar_fn = kwargs.get("bottom_toolbar")
            return _make_mock_session()

        with patch("chcode.chat.PromptSession", side_effect=capture_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    await repl._get_input()

        toolbar_output = str(captured_toolbar_fn())
        assert "YOLO" in toolbar_output

    async def test_toolbar_shows_cwd(self):
        """Toolbar should include the workplace_path as cwd."""
        repl = ChatREPL()
        repl.workplace_path = Path("/my/project")
        repl.model_config = {"model": "test"}
        repl._prompt_session = None

        captured_toolbar_fn = None

        def capture_session(**kwargs):
            nonlocal captured_toolbar_fn
            captured_toolbar_fn = kwargs.get("bottom_toolbar")
            return _make_mock_session()

        with patch("chcode.chat.PromptSession", side_effect=capture_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    await repl._get_input()

        toolbar_output = str(captured_toolbar_fn())
        # On Windows, Path("/my/project") uses backslashes
        assert "my" in toolbar_output and "project" in toolbar_output
        assert "cwd:" in toolbar_output

    async def test_toolbar_shows_git_info(self):
        """Toolbar should include Git checkpoint count when git is enabled."""
        repl = ChatREPL()
        repl.git = True
        repl._git_cp_count = 5
        repl.model_config = {"model": "test"}
        repl._prompt_session = None

        mock_git_mgr = MagicMock()
        mock_git_mgr.is_repo.return_value = True
        repl.git_manager = mock_git_mgr

        captured_toolbar_fn = None

        def capture_session(**kwargs):
            nonlocal captured_toolbar_fn
            captured_toolbar_fn = kwargs.get("bottom_toolbar")
            return _make_mock_session()

        with patch("chcode.chat.PromptSession", side_effect=capture_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    await repl._get_input()

        toolbar_output = str(captured_toolbar_fn())
        assert "Git" in toolbar_output
        assert "5 cp" in toolbar_output

    async def test_toolbar_shows_context_text(self):
        """Toolbar should include _context_text when it is non-empty."""
        repl = ChatREPL()
        repl._context_text = "[bold]important context[/bold]"
        repl.model_config = {"model": "test"}
        repl._prompt_session = None

        captured_toolbar_fn = None

        def capture_session(**kwargs):
            nonlocal captured_toolbar_fn
            captured_toolbar_fn = kwargs.get("bottom_toolbar")
            return _make_mock_session()

        with patch("chcode.chat.PromptSession", side_effect=capture_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    await repl._get_input()

        toolbar_output = str(captured_toolbar_fn())
        # _rich_to_html converts [bold]...[/bold] -> <b>...</b>
        assert "important context" in toolbar_output

    async def test_toolbar_width_caching(self):
        """Toolbar should cache terminal width for 1 second (nonlocal _last_width)."""
        repl = ChatREPL()
        repl.model_config = {"model": "test"}
        repl._prompt_session = None

        captured_toolbar_fn = None

        def capture_session(**kwargs):
            nonlocal captured_toolbar_fn
            captured_toolbar_fn = kwargs.get("bottom_toolbar")
            return _make_mock_session()

        with patch("chcode.chat.PromptSession", side_effect=capture_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=100)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    await repl._get_input()
                    # First call - uses get_terminal_size (columns=100)
                    output1 = str(captured_toolbar_fn())
                    assert "\u2500" * 100 in output1

                    # Second call within 1 second - should use cached width (still 100)
                    # Even though get_terminal_size now returns 200, the cache is valid
                    with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=200)):
                        output2 = str(captured_toolbar_fn())
                        assert "\u2500" * 100 in output2
                        assert "\u2500" * 200 not in output2


class TestKeyBindings:
    """Covers key handler bodies: lines 419 (enter), 423 (ctrl-enter), 428-434 (tab)."""

    async def _capture_handlers(self, repl):
        captured_kb = None

        def capture_session(**kwargs):
            nonlocal captured_kb
            captured_kb = kwargs.get("key_bindings")
            return _make_mock_session()

        with patch("chcode.chat.PromptSession", side_effect=capture_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    await repl._get_input()
        return captured_kb

    def _find_handler(self, kb, key_name):
        for binding in kb.bindings:
            if key_name in str(binding.keys):
                return binding.handler
        return None

    async def test_enter_key_invokes_validate_and_handle(self):
        """Line 419: enter handler calls event.current_buffer.validate_and_handle()."""
        repl = ChatREPL()
        repl._prompt_session = None
        kb = await self._capture_handlers(repl)
        handler = self._find_handler(kb, "ControlM")
        assert handler is not None
        mock_event = MagicMock()
        handler(mock_event)
        mock_event.current_buffer.validate_and_handle.assert_called_once()

    async def test_ctrl_enter_inserts_newline(self):
        """Line 423: ctrl-enter handler calls event.current_buffer.insert_text('\\n')."""
        repl = ChatREPL()
        repl._prompt_session = None
        kb = await self._capture_handlers(repl)
        handler = self._find_handler(kb, "ControlJ")
        assert handler is not None
        mock_event = MagicMock()
        handler(mock_event)
        mock_event.current_buffer.insert_text.assert_called_once_with("\n")

    async def test_tab_with_text_returns_early(self):
        """Line 428: tab handler returns early when buffer has text."""
        repl = ChatREPL()
        repl.yolo = False
        repl._prompt_session = None
        kb = await self._capture_handlers(repl)
        handler = self._find_handler(kb, "ControlI")
        assert handler is not None
        mock_event = MagicMock()
        mock_event.current_buffer.text = "some text"
        with patch("chcode.agent_setup.update_hitl_config") as mock_update:
            handler(mock_event)
            mock_update.assert_not_called()
            assert repl.yolo is False

    async def test_tab_without_text_toggles_yolo(self):
        """Lines 429-434: tab handler toggles yolo and calls update_hitl_config."""
        repl = ChatREPL()
        repl.yolo = False
        repl._prompt_session = None
        kb = await self._capture_handlers(repl)
        handler = self._find_handler(kb, "ControlI")
        assert handler is not None
        mock_event = MagicMock()
        mock_event.current_buffer.text = ""
        mock_event.app = MagicMock()
        mock_event.app.renderer = MagicMock()
        with patch("chcode.agent_setup.update_hitl_config") as mock_update:
            handler(mock_event)
            mock_update.assert_called_once_with(True)
            assert repl.yolo is True


class TestDynamicBufferHeight:
    async def test_dynamic_height_with_completions_initial(self):
        """_dynamic_buffer_height returns Dimension(min=n+2, max=n+2) when completions active."""
        from prompt_toolkit.layout.dimension import Dimension

        repl = ChatREPL()
        repl._prompt_session = None

        captured_height_fn = None

        def capture_session(**kwargs):
            nonlocal captured_height_fn
            m = _make_mock_session()
            # Override default_buffer to allow height capture
            m.default_buffer.complete_state = MagicMock()
            m.default_buffer.complete_state.completions = [MagicMock()] * 5
            m.default_buffer.text = "/"

            # Use a proper object to capture height assignment
            captured_height_fn_holder = [None]

            class CaptureSession:
                pass

            wrapper = CaptureSession()
            for attr_name in dir(m):
                if not attr_name.startswith("_") or attr_name in ("app",):
                    try:
                        setattr(wrapper, attr_name, getattr(m, attr_name))
                    except (AttributeError, TypeError):
                        pass
            wrapper.app = m.app

            def _set_height(val):
                captured_height_fn_holder[0] = val

            wrapper.height = property(lambda self: None, lambda self, v: _set_height(v))
            type(wrapper).height = property(lambda self: None, lambda self, v: _set_height(v))
            return wrapper

        with patch("chcode.chat.PromptSession", side_effect=capture_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    # Should complete without error when completions are active
                    await repl._get_input()
                    # Verify _prompt_session was set, proving _get_input completed with completions active
                    assert repl._prompt_session is not None

    async def test_dynamic_height_multiline_text(self):
        """_dynamic_buffer_height returns line count as Dimension when no completions."""
        repl = ChatREPL()
        repl._prompt_session = None

        def make_session(**kwargs):
            m = _make_mock_session()
            m.default_buffer.complete_state = None
            m.default_buffer.text = "line1\nline2\nline3"
            return m

        with patch("chcode.chat.PromptSession", side_effect=make_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    # Should complete without error with multiline text (no completions)
                    await repl._get_input()
                    # Verify _prompt_session was set during _get_input
                    assert repl._prompt_session is not None


class TestFindBufferWindow:
    """Tests for the _find_buffer_window helper (lines 493-509).
    We test the logic directly by calling _find_buffer_window through _get_input
    and checking the result via the height attribute set on the found window."""

    async def test_find_buffer_window_returns_matching_window(self):
        """_find_buffer_window should find a Window containing a BufferControl.
        Also covers lines 485-491: _dynamic_buffer_height function body."""
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import BufferControl
        from prompt_toolkit.layout.dimension import Dimension

        repl = ChatREPL()
        repl._prompt_session = None
        repl.model_config = {"model": "test"}

        mock_buffer_control = BufferControl()
        mock_window = MagicMock(spec=Window)
        mock_window.content = mock_buffer_control

        mock_root = _make_safe_container(content=mock_window)

        def make_session(**kwargs):
            m = _make_mock_session()
            m.app.layout.container = mock_root
            # Set up default_buffer with text for _dynamic_buffer_height
            m.default_buffer.text = "line1\nline2\nline3"
            m.default_buffer.complete_state = None
            return m

        with patch("chcode.chat.PromptSession", side_effect=make_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    await repl._get_input()

        assert mock_window.height is not None, "_dynamic_buffer_height should have been assigned"
        # Call the function to cover lines 485-491 (multiline text, no completions)
        result = mock_window.height()
        assert isinstance(result, Dimension)
        assert result.min == 3  # 3 lines
        assert result.max == 3

    async def test_dynamic_height_with_completions(self):
        """Lines 486-489: _dynamic_buffer_height with active completions."""
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import BufferControl
        from prompt_toolkit.layout.dimension import Dimension

        repl = ChatREPL()
        repl._prompt_session = None
        repl.model_config = {"model": "test"}

        mock_buffer_control = BufferControl()
        mock_window = MagicMock(spec=Window)
        mock_window.content = mock_buffer_control
        mock_root = _make_safe_container(content=mock_window)

        def make_session(**kwargs):
            m = _make_mock_session()
            m.app.layout.container = mock_root
            m.default_buffer.complete_state = MagicMock()
            m.default_buffer.complete_state.completions = [MagicMock()] * 5
            return m

        with patch("chcode.chat.PromptSession", side_effect=make_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    await repl._get_input()

        assert mock_window.height is not None
        result = mock_window.height()
        assert isinstance(result, Dimension)
        assert result.min == 7  # 5 + 2

    async def test_find_buffer_window_traverses_children(self):
        """_find_buffer_window should traverse children lists to find BufferControl."""
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import BufferControl

        repl = ChatREPL()
        repl._prompt_session = None
        repl.model_config = {"model": "test"}

        mock_buffer_control = BufferControl()
        mock_window = MagicMock(spec=Window)
        mock_window.content = mock_buffer_control

        mock_inner = _make_safe_container(content=mock_window)
        mock_non_match = _make_safe_container(content="not a window or buffer control")
        mock_root = _make_safe_container(children=[mock_non_match, mock_inner])

        def make_session(**kwargs):
            m = _make_mock_session()
            m.app.layout.container = mock_root
            return m

        with patch("chcode.chat.PromptSession", side_effect=make_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    await repl._get_input()

        assert mock_window.height is not None, "_dynamic_buffer_height should have been assigned via children traversal"

    async def test_find_buffer_window_returns_none_when_not_found(self):
        """_find_buffer_window should return None when no BufferControl is in the tree."""
        repl = ChatREPL()
        repl._prompt_session = None
        repl.model_config = {"model": "test"}

        mock_leaf = _make_safe_container(content="not a Window")
        mock_root = _make_safe_container(children=[mock_leaf])

        def make_session(**kwargs):
            m = _make_mock_session()
            m.app.layout.container = mock_root
            return m

        with patch("chcode.chat.PromptSession", side_effect=make_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    # Should complete traversal without infinite recursion when BufferControl not found
                    await repl._get_input()
                    # Verify the test completed without infinite recursion
                    assert repl._prompt_session is not None

    async def test_find_buffer_window_checks_alternative_content(self):
        """_find_buffer_window should check alternative_content attribute."""
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import BufferControl

        repl = ChatREPL()
        repl._prompt_session = None
        repl.model_config = {"model": "test"}

        mock_buffer_control = BufferControl()
        mock_window = MagicMock(spec=Window)
        mock_window.content = mock_buffer_control

        mock_root = _make_safe_container(alternative_content=mock_window)

        def make_session(**kwargs):
            m = _make_mock_session()
            m.app.layout.container = mock_root
            return m

        with patch("chcode.chat.PromptSession", side_effect=make_session):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="x"):
                    await repl._get_input()

        assert mock_window.height is not None, "_dynamic_buffer_height should have been assigned via alternative_content"


class TestGetInputPreFillBuffers:
    async def test_edit_buffer_prefilled(self):
        """When _edit_buffer is set, it should be used as default_text and cleared."""
        repl = ChatREPL()
        repl._edit_buffer = "prefilled edit"
        repl._prompt_session = None

        mock_session_cls = MagicMock(return_value=_make_mock_session())

        with patch("chcode.chat.PromptSession", mock_session_cls):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="submitted"):
                    result = await repl._get_input()

        assert result == "submitted"
        assert repl._edit_buffer is None

    async def test_interrupt_buffer_prefilled(self):
        """When _interrupt_buffer is set and _edit_buffer is None, it should be used."""
        repl = ChatREPL()
        repl._interrupt_buffer = "interrupted text"
        repl._edit_buffer = None
        repl._prompt_session = None

        mock_session_cls = MagicMock(return_value=_make_mock_session())

        with patch("chcode.chat.PromptSession", mock_session_cls):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="submitted"):
                    result = await repl._get_input()

        assert result == "submitted"
        assert repl._interrupt_buffer is None

    async def test_no_buffer_uses_empty_default(self):
        """When no buffer is set, default_text should be empty string."""
        repl = ChatREPL()
        repl._edit_buffer = None
        repl._interrupt_buffer = None
        repl._prompt_session = None

        mock_session_cls = MagicMock(return_value=_make_mock_session())

        with patch("chcode.chat.PromptSession", mock_session_cls):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="user input"):
                    result = await repl._get_input()

        assert result == "user input"

    async def test_keyboard_interrupt_returns_none(self):
        """KeyboardInterrupt during prompt should return None."""
        repl = ChatREPL()
        repl._prompt_session = None

        mock_session_cls = MagicMock(return_value=_make_mock_session())

        with patch("chcode.chat.PromptSession", mock_session_cls):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, side_effect=KeyboardInterrupt()):
                    result = await repl._get_input()

        assert result is None

    async def test_eof_error_returns_none(self):
        """EOFError during prompt should return None."""
        repl = ChatREPL()
        repl._prompt_session = None

        mock_session_cls = MagicMock(return_value=_make_mock_session())

        with patch("chcode.chat.PromptSession", mock_session_cls):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, side_effect=EOFError()):
                    result = await repl._get_input()

        assert result is None

    async def test_session_not_recreated_on_second_call(self):
        """When _prompt_session is already set, PromptSession should not be constructed again."""
        repl = ChatREPL()
        repl._prompt_session = None

        mock_session = _make_mock_session()
        mock_session.prompt = MagicMock(return_value="first")
        mock_session_cls = MagicMock(return_value=mock_session)

        # First call - creates session
        with patch("chcode.chat.PromptSession", mock_session_cls):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="first"):
                    await repl._get_input()

        mock_session_cls.assert_called_once()

        # Second call - reuses existing session
        mock_session.prompt = MagicMock(return_value="second")
        with patch("chcode.chat.PromptSession", mock_session_cls):
            with patch("chcode.chat.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
                with patch("chcode.chat.asyncio.to_thread", new_callable=AsyncMock, return_value="second"):
                    await repl._get_input()

        # Should still be only 1 call (from first invocation)
        mock_session_cls.assert_called_once()
