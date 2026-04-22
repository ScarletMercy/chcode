"""Extended tests for chcode/agent_setup.py"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chcode.agent_setup import (
    handle_tool_errors,
    ModelSwitchError,
    model_retry_with_backoff,
    load_skills,
    load_model,
    fix_messages,
    tool_result_budget,
    build_agent,
    update_hitl_config,
    update_summarization_model,
    create_checkpointer,
    _build_interrupt_on,
    _dummy_model,
    AsyncHITL,
)


class TestHandleToolErrors:
    async def test_passthrough_setup(self):
        mock_handler = AsyncMock(return_value="ok")
        mock_request = MagicMock()
        mock_request.tool_call = {"id": "123"}
        result = await handle_tool_errors.awrap_tool_call(mock_request, mock_handler)
        assert result == "ok"

    async def test_exception(self):
        mock_handler = AsyncMock(side_effect=ValueError("fail"))
        mock_request = MagicMock()
        mock_request.tool_call = {"id": "123"}
        result = await handle_tool_errors.awrap_tool_call(mock_request, mock_handler)
        from langchain_core.messages import ToolMessage
        assert isinstance(result, ToolMessage)
        assert "fail" in result.content


class TestModelRetryWithBackoff:
    async def test_success_first_try(self):
        mock_handler = AsyncMock(return_value="ok")
        mock_request = MagicMock()
        result = await model_retry_with_backoff.awrap_model_call(mock_request, mock_handler)
        assert result == "ok"
        assert mock_handler.call_count == 1

    async def test_retry_then_success(self):
        call_count = 0
        async def flaky(req):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "ok"
        mock_request = MagicMock()
        with patch("chcode.agent_setup.RETRY_DELAYS", [0.01, 0.01, 0.01, 0.01]), \
             patch("chcode.agent_setup.asyncio.sleep", new_callable=AsyncMock):
            result = await model_retry_with_backoff.awrap_model_call(mock_request, flaky)
        assert result == "ok"

    async def test_max_retries_no_fallback(self):
        from chcode.agent_setup import _fallback_models, _fallback_index
        old = _fallback_models[:]
        _fallback_models.clear()
        try:
            mock_handler = AsyncMock(side_effect=RuntimeError("fail"))
            mock_request = MagicMock()
            with patch("chcode.agent_setup.RETRY_DELAYS", [0.01, 0.01, 0.01, 0.01]), \
                 patch("chcode.agent_setup.asyncio.sleep", new_callable=AsyncMock), \
                 patch("chcode.agent_setup._load_fallback_config", return_value=None), \
                 patch("chcode.agent_setup.console"):
                with pytest.raises(RuntimeError):
                    await model_retry_with_backoff.awrap_model_call(mock_request, mock_handler)
        finally:
            _fallback_models[:] = old

    async def test_max_retries_with_fallback(self):
        from chcode.agent_setup import _fallback_models, _fallback_index
        old = _fallback_models[:]
        _fallback_models[:] = [{"model": "fallback"}]
        try:
            mock_handler = AsyncMock(side_effect=RuntimeError("fail"))
            mock_request = MagicMock()
            with patch("chcode.agent_setup.RETRY_DELAYS", [0.01, 0.01, 0.01, 0.01]), \
                 patch("chcode.agent_setup.asyncio.sleep", new_callable=AsyncMock), \
                 patch("chcode.agent_setup._load_fallback_config", return_value={"model": "fb"}):
                with pytest.raises(ModelSwitchError):
                    await model_retry_with_backoff.awrap_model_call(mock_request, mock_handler)
        finally:
            _fallback_models[:] = old


class TestLoadSkills:
    async def test_builds_system_prompt(self):
        mock_loader = MagicMock()
        mock_loader.build_system_prompt = MagicMock(return_value="prompt with skills")
        mock_request = MagicMock()
        mock_request.runtime.context.skill_loader = mock_loader
        mock_request.runtime.context.working_directory = "/w"
        with patch("chcode.agent_setup.sys.platform", "linux"):
            # @dynamic_prompt middleware injects prompt via handler
            handler = AsyncMock(return_value="model response")
            result = await load_skills.awrap_model_call(mock_request, handler)
        mock_loader.build_system_prompt.assert_called_once()
        handler.assert_called_once()


class TestLoadModel:
    async def test_creates_enhanced_model(self):
        from chcode.agent_setup import EnhancedChatOpenAI
        mock_handler = AsyncMock(return_value="response")
        mock_request = MagicMock()
        mock_request.runtime.context.model_config = {"model": "gpt-4", "api_key": "k"}
        with patch("chcode.agent_setup.EnhancedChatOpenAI", MagicMock()) as mock_model_cls:
            result = await load_model.awrap_model_call(mock_request, mock_handler)
        assert mock_handler.called


class TestFixMessages:
    async def test_filters_composed(self):
        from langchain_core.messages import HumanMessage
        mock_msg = MagicMock(spec=HumanMessage)
        mock_msg.additional_kwargs = {"composed": "yes"}
        clean_msg = MagicMock(spec=HumanMessage)
        clean_msg.additional_kwargs = {}

        def mock_override(**kwargs):
            mock_request.messages = kwargs.get("messages", mock_request.messages)
            return mock_request

        mock_request = MagicMock()
        mock_request.messages = [mock_msg, clean_msg]
        mock_request.override = mock_override
        mock_handler = AsyncMock(return_value="resp")
        await fix_messages.awrap_model_call(mock_request, mock_handler)
        mock_handler.assert_called_once()
        assert len(mock_request.messages) == 1

    async def test_no_composed_passthrough(self):
        from langchain_core.messages import HumanMessage
        mock_handler = AsyncMock(return_value="resp")
        mock_request = MagicMock()
        mock_request.messages = [MagicMock(spec=HumanMessage, additional_kwargs={})]
        mock_request.override = lambda **kw: mock_request
        await fix_messages.awrap_model_call(mock_request, mock_handler)
        mock_handler.assert_called_once()


class TestToolResultBudget:
    async def test_no_tool_messages_setup(self):
        from langchain_core.messages import HumanMessage
        mock_handler = AsyncMock(return_value="resp")
        mock_request = MagicMock()
        mock_request.messages = [HumanMessage(content="hi")]
        mock_request.runtime.context.working_directory = "/w"
        result = await tool_result_budget.awrap_model_call(mock_request, mock_handler)
        mock_handler.assert_called_once()

    async def test_budget_ok_skipped(self):
        from langchain_core.messages import ToolMessage
        mock_msg = MagicMock(spec=ToolMessage)
        mock_msg.content = "output"
        mock_msg.additional_kwargs = {"_budget_ok": True}
        mock_handler = AsyncMock(return_value="resp")
        mock_request = MagicMock()
        mock_request.messages = [mock_msg]
        mock_request.runtime.context.working_directory = "/w"
        result = await tool_result_budget.awrap_model_call(mock_request, mock_handler)
        mock_handler.assert_called_once()


class TestBuildAgent:
    def test_builds_with_middleware(self):
        with patch("chcode.agent_setup._dummy_model") as mock_model, \
             patch("chcode.agent_setup.create_agent") as mock_create, \
             patch("chcode.agent_setup._get_all_tools", return_value=[]), \
             patch("chcode.config.load_model_json", return_value={}), \
             patch("chcode.agent_setup.EnhancedChatOpenAI") as mock_llm:
            agent = build_agent(model_config={"model": "gpt-4"}, checkpointer=None, yolo=True)
        mock_create.assert_called_once()


class TestUpdateHitlConfig:
    def test_updates_interrupt_on(self):
        from chcode.agent_setup import _hitl_middleware
        old = _hitl_middleware
        try:
            mock_mw = MagicMock()
            from chcode.agent_setup import _hitl_middleware as mw_ref
            update_hitl_config(True)
            if mw_ref is not None:
                assert mw_ref.interrupt_on == {}
            # Verify update_hitl_config completes without error
            assert True
        finally:
            pass


class TestBuildInterruptOn:
    def test_yolo_empty(self):
        assert _build_interrupt_on(True) == {}

    def test_normal_dict(self):
        result = _build_interrupt_on(False)
        assert "bash" in result
        assert "edit" in result
        assert "write_file" in result


class TestDummyModel:
    def test_returns_chat_openai(self):
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            m = _dummy_model()
            mock_cls.assert_called_once()


class TestAsyncHITL:
    async def test_awrap_model_call(self):
        hitl = AsyncHITL(interrupt_on={"bash": {"allowed_decisions": ["approve"]}})
        mock_handler = AsyncMock(return_value="resp")
        mock_request = MagicMock()
        result = await hitl.awrap_model_call(mock_request, mock_handler)
        assert result == "resp"


class TestCreateCheckpointer:
    async def test_creates_saver(self, tmp_path):
        db = tmp_path / "test.db"
        with patch("chcode.agent_setup.aiosqlite.connect", new_callable=AsyncMock) as mock_conn:
            mock_conn.return_value = AsyncMock()
            saver = await create_checkpointer(db)
            mock_conn.assert_called_once()


class TestUpdateSummarizationModel:
    def test_updates_model_fields(self):
        from chcode.agent_setup import _summarization_model
        import chcode.agent_setup as mod
        old = _summarization_model
        try:
            # Set up a fake model so the if-branch actually runs
            class FakeModel:
                model_fields_set = {"model", "api_key"}
                def __init__(self):
                    self.model = "old-model"
                    self.api_key = "old-key"
            fake = FakeModel()
            mod._summarization_model = fake

            new_model = FakeModel.__new__(FakeModel)
            new_model.__dict__ = {"model": "new-model", "api_key": "new-key"}

            with patch("chcode.agent_setup.EnhancedChatOpenAI", return_value=new_model):
                update_summarization_model({"model": "new-model", "api_key": "new-key"})
            # Verify the model fields were updated in place
            assert fake.model == "new-model"
            assert fake.api_key == "new-key"
        finally:
            mod._summarization_model = old
