from unittest.mock import MagicMock

from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI


class TestExtractReasoning:
    def test_auto_detect_field(self):
        model = EnhancedChatOpenAI(model="test", api_key="sk-test")
        data = {"reasoning_content": "thinking...", "content": "answer"}
        result = model._extract_reasoning(data)
        assert result == "thinking..."

    def test_specific_field(self):
        model = EnhancedChatOpenAI(
            model="test", api_key="sk-test", reasoning_field="thinking"
        )
        data = {"thinking": "deep thought", "content": "answer"}
        result = model._extract_reasoning(data)
        assert result == "deep thought"

    def test_no_reasoning(self):
        model = EnhancedChatOpenAI(model="test", api_key="sk-test")
        data = {"content": "just answer"}
        result = model._extract_reasoning(data)
        assert result is None


class TestProcessMessageWithReasoning:
    def test_adds_to_additional_kwargs(self):
        model = EnhancedChatOpenAI(model="test", api_key="sk-test")
        msg = {"reasoning_content": "think", "content": "ans"}
        result = model._process_message_with_reasoning(msg)
        assert result["additional_kwargs"]["reasoning"] == "think"

    def test_removes_reasoning_field(self):
        model = EnhancedChatOpenAI(model="test", api_key="sk-test")
        msg = {"reasoning_content": "think", "content": "ans"}
        result = model._process_message_with_reasoning(msg)
        assert "reasoning_content" not in result

    def test_no_reasoning_passthrough(self):
        model = EnhancedChatOpenAI(model="test", api_key="sk-test")
        msg = {"content": "ans"}
        result = model._process_message_with_reasoning(msg)
        assert result == {"content": "ans"}

    def test_include_in_content(self):
        model = EnhancedChatOpenAI(
            model="test",
            api_key="sk-test",
            include_reasoning_in_content=True,
        )
        msg = {"reasoning_content": "think", "content": "ans"}
        result = model._process_message_with_reasoning(msg)
        assert "think" in result["content"]
        assert "ans" in result["content"]


class TestExtractReasoningFromMessage:
    def test_none_message(self):
        model = EnhancedChatOpenAI(model="test", api_key="sk-test")
        assert model._extract_reasoning_from_message(None) is None

    def test_reasoning_content_attr(self):
        model = EnhancedChatOpenAI(model="test", api_key="sk-test")
        msg = MagicMock()
        msg.reasoning_content = "I thought about it"
        msg.content_blocks = None
        msg.model_dump = MagicMock(return_value={})
        result = model._extract_reasoning_from_message(msg)
        assert "thought" in result

    def test_content_blocks_thinking(self):
        model = EnhancedChatOpenAI(model="test", api_key="sk-test")
        msg = MagicMock()
        msg.reasoning_content = None
        block = {"type": "thinking", "thinking": "deep thought"}
        msg.content_blocks = [block]
        msg.model_dump = MagicMock(return_value={})
        result = model._extract_reasoning_from_message(msg)
        assert "deep thought" in result
