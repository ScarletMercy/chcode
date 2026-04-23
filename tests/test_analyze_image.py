"""Tests for chcode/utils/tools.py - analyze_image function"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


@pytest.fixture
def mock_runtime():
    runtime = MagicMock()
    runtime.context = MagicMock()
    runtime.context.working_directory = Path("/tmp/workplace")
    return runtime


@pytest.fixture
def temp_image_file(tmp_path):
    """Create a temporary valid image file."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test.png"
    img.save(img_path)
    return img_path


class TestAnalyzeImageFileValidation:
    """Tests for file validation branches (lines 1584-1609)."""

    @pytest.mark.asyncio
    async def test_file_not_found(self, mock_runtime):
        """Should return error when file doesn't exist."""
        from chcode.utils.tools import analyze_image

        with patch("chcode.utils.tools.resolve_path") as mock_resolve:
            mock_resolve.return_value = Path("/nonexistent.png")

            result = await analyze_image.coroutine("test.png", runtime=mock_runtime)

            assert "[FAILED]" in result
            assert "File not found" in result

    @pytest.mark.asyncio
    async def test_not_a_file(self, mock_runtime, tmp_path):
        """Should return error when path is a directory."""
        from chcode.utils.tools import analyze_image

        dir_path = tmp_path / "subdir"
        dir_path.mkdir()

        with patch("chcode.utils.tools.resolve_path") as mock_resolve:
            mock_resolve.return_value = dir_path

            result = await analyze_image.coroutine("test.png", runtime=mock_runtime)

            assert "[FAILED]" in result
            assert "Not a file" in result

    @pytest.mark.asyncio
    async def test_unsupported_format(self, mock_runtime, tmp_path):
        """Should return error for unsupported image format."""
        from chcode.utils.tools import analyze_image

        img_path = tmp_path / "test.xyz"
        img_path.write_text("not an image")

        with patch("chcode.utils.tools.resolve_path") as mock_resolve:
            mock_resolve.return_value = img_path

            result = await analyze_image.coroutine("test.xyz", runtime=mock_runtime)

            assert "[FAILED]" in result
            assert "Unsupported image format" in result

    @pytest.mark.asyncio
    async def test_file_too_large(self, mock_runtime, tmp_path):
        """Should return error when file exceeds size limit."""
        from chcode.utils.tools import analyze_image

        img_path = tmp_path / "large.png"
        img_path.write_bytes(b"x" * (21 * 1024 * 1024))

        with patch("chcode.utils.tools.resolve_path") as mock_resolve:
            mock_resolve.return_value = img_path

            result = await analyze_image.coroutine("large.png", runtime=mock_runtime)

            assert "[FAILED]" in result
            assert "Image too large" in result


class TestAnalyzeImageImageProcessing:
    """Tests for image processing branches (lines 1611-1644)."""

    @pytest.mark.asyncio
    async def test_successful_image_read_and_encode(self, mock_runtime, temp_image_file):
        """Should read and encode image successfully."""
        from chcode.utils.tools import analyze_image

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.tools.httpx.AsyncClient") as mock_client_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "test-model", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = []

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Image description"}}]
            }
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = await analyze_image.coroutine("test.png", runtime=mock_runtime)

            assert "[OK]" in result
            assert "Image description" in result
            assert "test-model" in result


class TestAnalyzeImageNoApiKey:
    """Tests for no API key branches (lines 1646-1662)."""

    @pytest.mark.asyncio
    async def test_returns_error_when_no_api_key_and_no_auto_config(self, mock_runtime, temp_image_file):
        """Should return error when no API key configured."""
        from chcode.utils.tools import analyze_image

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.auto_configure_vision") as mock_auto:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = None
            mock_auto.return_value = None

            result = await analyze_image.coroutine("test.png", runtime=mock_runtime)

            assert "[FAILED]" in result
            assert "视觉模型未配置" in result

    @pytest.mark.asyncio
    async def test_auto_configure_succeeds(self, mock_runtime, temp_image_file):
        """Should use auto-configured model."""
        from chcode.utils.tools import analyze_image

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.vision_config.auto_configure_vision") as mock_auto, \
             patch("chcode.utils.tools.httpx.AsyncClient") as mock_client_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = None
            mock_auto.return_value = {"model": "auto-model", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = []

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Auto config works"}}]
            }
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = await analyze_image.coroutine("test.png", runtime=mock_runtime)

            assert "[OK]" in result


class TestAnalyzeImageHttpErrors:
    """Tests for HTTP error handling branches (lines 1693-1759)."""

    @pytest.mark.asyncio
    async def test_continues_to_fallback_on_http_error(self, mock_runtime, temp_image_file):
        """Should try fallback model when default fails with HTTP error."""
        from chcode.utils.tools import analyze_image

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.tools.httpx.AsyncClient") as mock_client_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "default-model", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [
                {"model": "fallback-model", "api_key": "key", "base_url": "http://x.com"}
            ]

            mock_response_fail = MagicMock()
            mock_response_fail.status_code = 500
            mock_response_fail.text = "Server error"

            mock_response_ok = MagicMock()
            mock_response_ok.status_code = 200
            mock_response_ok.json.return_value = {
                "choices": [{"message": {"content": "Fallback worked"}}]
            }

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.side_effect = [mock_response_fail, mock_response_ok]
            mock_client_cls.return_value = mock_client

            result = await analyze_image.coroutine("test.png", runtime=mock_runtime)

            assert "[OK]" in result
            assert "Fallback worked" in result

    @pytest.mark.asyncio
    async def test_continues_on_json_decode_error(self, mock_runtime, temp_image_file):
        """Should continue to fallback on JSON decode error."""
        from chcode.utils.tools import analyze_image

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.tools.httpx.AsyncClient") as mock_client_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model1", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [{"model": "model2", "api_key": "key", "base_url": "http://x.com"}]

            mock_response_invalid = MagicMock()
            mock_response_invalid.status_code = 200
            mock_response_invalid.json.side_effect = json.JSONDecodeError("invalid", "", 0)

            mock_response_ok = MagicMock()
            mock_response_ok.status_code = 200
            mock_response_ok.json.return_value = {
                "choices": [{"message": {"content": "Success after invalid JSON"}}]
            }

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.side_effect = [mock_response_invalid, mock_response_ok]
            mock_client_cls.return_value = mock_client

            result = await analyze_image.coroutine("test.png", runtime=mock_runtime)

            assert "[OK]" in result
            assert "Success after invalid JSON" in result

    @pytest.mark.asyncio
    async def test_continues_on_empty_choices(self, mock_runtime, temp_image_file):
        """Should continue to fallback when response has no choices."""
        from chcode.utils.tools import analyze_image

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.tools.httpx.AsyncClient") as mock_client_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model1", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [{"model": "model2", "api_key": "key", "base_url": "http://x.com"}]

            mock_response_empty = MagicMock()
            mock_response_empty.status_code = 200
            mock_response_empty.json.return_value = {"choices": []}

            mock_response_ok = MagicMock()
            mock_response_ok.status_code = 200
            mock_response_ok.json.return_value = {
                "choices": [{"message": {"content": "Success after empty choices"}}]
            }

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.side_effect = [mock_response_empty, mock_response_ok]
            mock_client_cls.return_value = mock_client

            result = await analyze_image.coroutine("test.png", runtime=mock_runtime)

            assert "[OK]" in result

    @pytest.mark.asyncio
    async def test_continues_on_empty_content(self, mock_runtime, temp_image_file):
        """Should continue to fallback when content is empty."""
        from chcode.utils.tools import analyze_image

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.tools.httpx.AsyncClient") as mock_client_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model1", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [{"model": "model2", "api_key": "key", "base_url": "http://x.com"}]

            mock_response_empty = MagicMock()
            mock_response_empty.status_code = 200
            mock_response_empty.json.return_value = {"choices": [{"message": {"content": ""}}]}

            mock_response_ok = MagicMock()
            mock_response_ok.status_code = 200
            mock_response_ok.json.return_value = {
                "choices": [{"message": {"content": "Success after empty content"}}]
            }

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.side_effect = [mock_response_empty, mock_response_ok]
            mock_client_cls.return_value = mock_client

            result = await analyze_image.coroutine("test.png", runtime=mock_runtime)

            assert "[OK]" in result

    @pytest.mark.asyncio
    async def test_timeout_continues_to_fallback(self, mock_runtime, temp_image_file):
        """Should continue to fallback on timeout."""
        from chcode.utils.tools import analyze_image

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.tools.httpx.AsyncClient") as mock_client_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model1", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [{"model": "model2", "api_key": "key", "base_url": "http://x.com"}]

            mock_response_ok = MagicMock()
            mock_response_ok.status_code = 200
            mock_response_ok.json.return_value = {
                "choices": [{"message": {"content": "Success after timeout"}}]
            }

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.side_effect = [httpx.TimeoutException("timeout"), mock_response_ok]
            mock_client_cls.return_value = mock_client

            result = await analyze_image.coroutine("test.png", runtime=mock_runtime)

            assert "[OK]" in result

    @pytest.mark.asyncio
    async def test_other_exception_continues_to_fallback(self, mock_runtime, temp_image_file):
        """Should continue to fallback on other exceptions."""
        from chcode.utils.tools import analyze_image

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.tools.httpx.AsyncClient") as mock_client_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model1", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [{"model": "model2", "api_key": "key", "base_url": "http://x.com"}]

            mock_response_ok = MagicMock()
            mock_response_ok.status_code = 200
            mock_response_ok.json.return_value = {
                "choices": [{"message": {"content": "Success after exception"}}]
            }

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.side_effect = [Exception("Generic error"), mock_response_ok]
            mock_client_cls.return_value = mock_client

            result = await analyze_image.coroutine("test.png", runtime=mock_runtime)

            assert "[OK]" in result

    @pytest.mark.asyncio
    async def test_all_models_fail_returns_final_error(self, mock_runtime, temp_image_file):
        """Should return final error when all models fail."""
        from chcode.utils.tools import analyze_image

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.tools.httpx.AsyncClient") as mock_client_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model1", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [{"model": "model2", "api_key": "key", "base_url": "http://x.com"}]

            mock_response_fail = MagicMock()
            mock_response_fail.status_code = 500
            mock_response_fail.text = "Server error"

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.side_effect = [mock_response_fail, mock_response_fail]
            mock_client_cls.return_value = mock_client

            result = await analyze_image.coroutine("test.png", runtime=mock_runtime)

            assert "[FAILED]" in result
            assert "所有视觉模型均调用失败" in result


class TestAnalyzeImageDeduplication:
    """Tests for model deduplication (lines 1683-1692)."""

    @pytest.mark.asyncio
    async def test_deduplicates_model_list(self, mock_runtime, temp_image_file):
        """Should deduplicate models with same name."""
        from chcode.utils.tools import analyze_image

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.tools.httpx.AsyncClient") as mock_client_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model1", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = [
                {"model": "model1", "api_key": "key", "base_url": "http://x.com"},  # duplicate, filtered
                {"model": "model2", "api_key": "key", "base_url": "http://x.com"},
                {"model": "model3", "api_key": "key", "base_url": "http://x.com"},
            ]

            mock_response_200 = MagicMock()
            mock_response_200.status_code = 200
            mock_response_200.json.return_value = {
                "choices": [{"message": {"content": "Success"}}]
            }
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            # Model1 fails, model2 fails, model3 succeeds
            mock_client.post.side_effect = [
                Exception("model1 failed"),  # default fails
                Exception("model2 failed"),  # fallback[0] fails  
                mock_response_200,  # fallback[1] succeeds
            ]
            mock_client_cls.return_value = mock_client

            await analyze_image.coroutine("test.png", runtime=mock_runtime)

            assert mock_client.post.call_count == 3  # model1, model2, model3 all tried


class TestAnalyzeImageCustomPrompt:
    """Tests for custom prompt parameter."""

    @pytest.mark.asyncio
    async def test_uses_custom_prompt(self, mock_runtime, temp_image_file):
        """Should use custom prompt when provided."""
        from chcode.utils.tools import analyze_image

        custom_prompt = "What is in this image?"
        captured_payload = {}

        async def capture_post(url, json=None, **kw):
            captured_payload["json"] = json
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "OK"}}]
            }
            return mock_response

        with patch("chcode.utils.tools.resolve_path") as mock_resolve, \
             patch("chcode.vision_config.get_vision_default_model") as mock_get_default, \
             patch("chcode.vision_config.get_vision_fallback_models") as mock_get_fb, \
             patch("chcode.utils.tools.httpx.AsyncClient") as mock_client_cls:

            mock_resolve.return_value = temp_image_file
            mock_get_default.return_value = {"model": "model", "api_key": "key", "base_url": "http://x.com"}
            mock_get_fb.return_value = []

            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.side_effect = capture_post
            mock_client_cls.return_value = mock_client

            await analyze_image.coroutine("test.png", prompt=custom_prompt, runtime=mock_runtime)

            assert captured_payload["json"]["messages"][0]["content"][1]["text"] == custom_prompt