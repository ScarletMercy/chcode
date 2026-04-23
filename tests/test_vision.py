"""Tests for chcode/utils/tools.py - vision function"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.messages import AIMessage


@pytest.fixture
def mock_runtime():
    runtime = MagicMock()
    runtime.context = MagicMock()
    runtime.context.working_directory = Path("/tmp/workplace")
    return runtime


@pytest.fixture
def temp_image_file(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test.png"
    img.save(img_path)
    return img_path


def _mock_llm_ainvoke(return_content="OK"):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=return_content))
    return mock_llm


class TestVisionFileValidation:

    @pytest.mark.asyncio
    async def test_file_not_found(self, mock_runtime):
        from chcode.utils.tools import vision

        with patch("chcode.utils.tools.resolve_path") as mock_resolve:
            mock_resolve.return_value = Path("/nonexistent.png")
            result = await vision.coroutine("test.png", runtime=mock_runtime)
            assert "[FAILED]" in result
            assert "File not found" in result

    @pytest.mark.asyncio
    async def test_not_a_file(self, mock_runtime, tmp_path):
        from chcode.utils.tools import vision

        dir_path = tmp_path / "subdir"
        dir_path.mkdir()

        with patch("chcode.utils.tools.resolve_path") as mock_resolve:
            mock_resolve.return_value = dir_path
            result = await vision.coroutine("test.png", runtime=mock_runtime)
            assert "[FAILED]" in result
            assert "Not a file" in result

    @pytest.mark.asyncio
    async def test_unsupported_format(self, mock_runtime, tmp_path):
        from chcode.utils.tools import vision

        img_path = tmp_path / "test.xyz"
        img_path.write_text("not an image")

        with patch("chcode.utils.tools.resolve_path") as mock_resolve:
            mock_resolve.return_value = img_path
            result = await vision.coroutine("test.xyz", runtime=mock_runtime)
            assert "[FAILED]" in result
            assert "Unsupported image format" in result

    @pytest.mark.asyncio
    async def test_file_too_large(self, mock_runtime, tmp_path):
        from chcode.utils.tools import vision

        img_path = tmp_path / "large.png"
        img_path.write_bytes(b"x" * (21 * 1024 * 1024))

        with patch("chcode.utils.tools.resolve_path") as mock_resolve:
            mock_resolve.return_value = img_path
            result = await vision.coroutine("large.png", runtime=mock_runtime)
            assert "[FAILED]" in result
            assert "Image too large" in result


class TestVisionImageProcessing:

    @pytest.mark.asyncio
    async def test_successful_image_read_and_encode(self, mock_runtime, temp_image_file):
        from chcode.utils.tools import vision

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "test-model", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = []
            mock_llm_cls.return_value = _mock_llm_ainvoke("Image description")

            result = await vision.coroutine("test.png", runtime=mock_runtime)
            assert "[OK]" in result
            assert "Image description" in result
            assert "test-model" in result


class TestVisionNoApiKey:

    @pytest.mark.asyncio
    async def test_returns_error_when_no_api_key_and_no_auto_config(self, mock_runtime, temp_image_file):
        from chcode.utils.tools import vision

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.auto_configure_vision") as mock_auto, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = None
            mock_auto.return_value = None
            mock_get_fb.return_value = []

            result = await vision.coroutine("test.png", runtime=mock_runtime)
            assert "[FAILED]" in result
            assert "视觉模型未配置" in result

    @pytest.mark.asyncio
    async def test_auto_configure_succeeds(self, mock_runtime, temp_image_file):
        from chcode.utils.tools import vision

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.vision_config.auto_configure_vision") as mock_auto, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = None
            mock_auto.return_value = {"model": "auto-model", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = []
            mock_llm_cls.return_value = _mock_llm_ainvoke("Auto config works")

            result = await vision.coroutine("test.png", runtime=mock_runtime)
            assert "[OK]" in result


class TestVisionFallback:

    @pytest.mark.asyncio
    async def test_continues_to_fallback_on_invoke_error(self, mock_runtime, temp_image_file):
        from chcode.utils.tools import vision

        mock_llm_fail = MagicMock()
        mock_llm_fail.ainvoke = AsyncMock(side_effect=Exception("Model error"))

        mock_llm_ok = _mock_llm_ainvoke("Fallback worked")

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "default-model", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [
                {"model": "fallback-model", "api_key": "key", "base_url": "http://x.com"}
            ]
            mock_llm_cls.side_effect = [mock_llm_fail, mock_llm_ok]

            result = await vision.coroutine("test.png", runtime=mock_runtime)
            assert "[OK]" in result
            assert "Fallback worked" in result

    @pytest.mark.asyncio
    async def test_continues_on_empty_content(self, mock_runtime, temp_image_file):
        from chcode.utils.tools import vision

        mock_llm_empty = _mock_llm_ainvoke("")

        mock_llm_ok = _mock_llm_ainvoke("Success after empty content")

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model1", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [{"model": "model2", "api_key": "key", "base_url": "http://x.com"}]
            mock_llm_cls.side_effect = [mock_llm_empty, mock_llm_ok]

            result = await vision.coroutine("test.png", runtime=mock_runtime)
            assert "[OK]" in result

    @pytest.mark.asyncio
    async def test_timeout_continues_to_fallback(self, mock_runtime, temp_image_file):
        from chcode.utils.tools import vision

        mock_llm_timeout = MagicMock()
        mock_llm_timeout.ainvoke = AsyncMock(side_effect=TimeoutError("Request timed out"))

        mock_llm_ok = _mock_llm_ainvoke("Success after timeout")

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model1", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [{"model": "model2", "api_key": "key", "base_url": "http://x.com"}]
            mock_llm_cls.side_effect = [mock_llm_timeout, mock_llm_ok]

            result = await vision.coroutine("test.png", runtime=mock_runtime)
            assert "[OK]" in result

    @pytest.mark.asyncio
    async def test_other_exception_continues_to_fallback(self, mock_runtime, temp_image_file):
        from chcode.utils.tools import vision

        mock_llm_err = MagicMock()
        mock_llm_err.ainvoke = AsyncMock(side_effect=RuntimeError("Generic error"))

        mock_llm_ok = _mock_llm_ainvoke("Success after exception")

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model1", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [{"model": "model2", "api_key": "key", "base_url": "http://x.com"}]
            mock_llm_cls.side_effect = [mock_llm_err, mock_llm_ok]

            result = await vision.coroutine("test.png", runtime=mock_runtime)
            assert "[OK]" in result

    @pytest.mark.asyncio
    async def test_all_models_fail_returns_final_error(self, mock_runtime, temp_image_file):
        from chcode.utils.tools import vision

        mock_llm_fail = MagicMock()
        mock_llm_fail.ainvoke = AsyncMock(side_effect=Exception("Server error"))

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model1", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [{"model": "model2", "api_key": "key", "base_url": "http://x.com"}]
            mock_llm_cls.return_value = mock_llm_fail

            result = await vision.coroutine("test.png", runtime=mock_runtime)
            assert "[FAILED]" in result
            assert "所有视觉模型均调用失败" in result


class TestVisionDeduplication:

    @pytest.mark.asyncio
    async def test_deduplicates_model_list(self, mock_runtime, temp_image_file):
        from chcode.utils.tools import vision

        invoke_count = 0

        def make_llm(**kwargs):
            nonlocal invoke_count
            invoke_count += 1
            mock_llm = MagicMock()
            if invoke_count <= 2:
                mock_llm.ainvoke = AsyncMock(side_effect=Exception(f"model {invoke_count} failed"))
            else:
                mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Success"))
            return mock_llm

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model1", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [
                {"model": "model1", "api_key": "key", "base_url": "http://x.com"},
                {"model": "model2", "api_key": "key", "base_url": "http://x.com"},
                {"model": "model3", "api_key": "key", "base_url": "http://x.com"},
            ]
            mock_llm_cls.side_effect = make_llm

            await vision.coroutine("test.png", runtime=mock_runtime)
            assert invoke_count == 3


class TestVisionCustomPrompt:

    @pytest.mark.asyncio
    async def test_uses_custom_prompt(self, mock_runtime, temp_image_file):
        from chcode.utils.tools import vision

        custom_prompt = "What is in this image?"
        captured_messages = None

        async def capture_ainvoke(messages, config=None):
            nonlocal captured_messages
            captured_messages = messages
            return AIMessage(content="OK")

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = []

            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=capture_ainvoke)
            mock_llm_cls.return_value = mock_llm

            await vision.coroutine("test.png", prompt=custom_prompt, runtime=mock_runtime)

            msg_content = captured_messages[0].content
            text_parts = [p for p in msg_content if isinstance(p, dict) and p.get("type") == "text"]
            assert text_parts[0]["text"] == custom_prompt


class TestVisionResizing:

    @pytest.mark.asyncio
    async def test_resizes_large_image(self, mock_runtime, tmp_path):
        from PIL import Image
        from chcode.utils.tools import vision
        import io

        img = Image.new("RGB", (4000, 3000), color="blue")
        img_path = tmp_path / "large.png"
        img.save(img_path)

        saved_sizes = []

        class FakeBytesIO(io.BytesIO):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def write(self, data):
                saved_sizes.append(len(data))
                super().write(data)

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:

            mock_resolve.return_value = img_path
            mock_get_default.return_value = {"model": "model", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = []
            mock_llm_cls.return_value = _mock_llm_ainvoke("OK")

            with patch("io.BytesIO", FakeBytesIO):
                result = await vision.coroutine("large.png", runtime=mock_runtime)

            assert len(saved_sizes) >= 1
            assert "[OK]" in result


class TestVisionPilError:

    @pytest.mark.asyncio
    async def test_pil_read_error(self, mock_runtime, tmp_path):
        from PIL import Image
        from chcode.utils.tools import vision

        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.auto_configure_vision") as mock_auto:

            mock_resolve.return_value = img_path
            mock_get_default.return_value = None
            mock_auto.return_value = None

            with patch("PIL.Image.open", side_effect=IOError("corrupted image")):
                result = await vision.coroutine("test.png", runtime=mock_runtime)

            assert "[FAILED]" in result
            assert "Failed to read image" in result


class TestVisionOptionalHyperparameters:

    @pytest.mark.asyncio
    async def test_uses_optional_temperature(self, mock_runtime, temp_image_file):
        from chcode.utils.tools import vision

        captured_kwargs = {}

        def capture_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="OK"))
            return mock_llm

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {
                "model": "model", "api_key": "key", "base_url": "http://x.com",
                "temperature": 0.7,
            }
            mock_get_fb.return_value = []
            mock_llm_cls.side_effect = capture_llm

            await vision.coroutine("test.png", runtime=mock_runtime)
            assert captured_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_uses_optional_top_p(self, mock_runtime, temp_image_file):
        from chcode.utils.tools import vision

        captured_kwargs = {}

        def capture_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="OK"))
            return mock_llm

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {
                "model": "model", "api_key": "key", "base_url": "http://x.com",
                "top_p": 0.9,
            }
            mock_get_fb.return_value = []
            mock_llm_cls.side_effect = capture_llm

            await vision.coroutine("test.png", runtime=mock_runtime)
            assert captured_kwargs["top_p"] == 0.9


class TestVisionSkipEmptyApiKey:

    @pytest.mark.asyncio
    async def test_skips_model_with_no_api_key(self, mock_runtime, temp_image_file):
        from chcode.utils.tools import vision

        invoke_count = 0

        def make_llm(*args, **kwargs):
            nonlocal invoke_count
            invoke_count += 1
            return _mock_llm_ainvoke("OK")

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.enhanced_chat_openai.EnhancedChatOpenAI") as mock_llm_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model1", "api_key": "", "base_url": "http://x.com"}
            mock_get_fb.return_value = [
                {"model": "model2", "api_key": "key", "base_url": "http://x.com"},
            ]
            mock_llm_cls.side_effect = make_llm

            result = await vision.coroutine("test.png", runtime=mock_runtime)
            assert "[OK]" in result
            assert invoke_count == 1
