"""Tests for chcode/utils/multimodal.py — multimodal detection, media encoding, path extraction, message building."""

from pathlib import Path
from unittest.mock import patch

import pytest

from langchain_core.messages import HumanMessage


# ─── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def temp_image(tmp_path):
    """Create a small 100x100 red PNG image."""
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    path = tmp_path / "test.png"
    img.save(path)
    return path


@pytest.fixture
def temp_large_image(tmp_path):
    """Create a 4000x3000 image that exceeds the 2048px resize threshold."""
    from PIL import Image

    img = Image.new("RGB", (4000, 3000), color="blue")
    path = tmp_path / "large.png"
    img.save(path)
    return path


@pytest.fixture
def temp_jpeg(tmp_path):
    """Create a small JPEG image."""
    from PIL import Image

    img = Image.new("RGB", (50, 50), color="green")
    path = tmp_path / "photo.jpg"
    img.save(path, format="JPEG")
    return path


@pytest.fixture
def temp_video(tmp_path):
    """Create a tiny MP4 file (not a real video, just bytes)."""
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00" * 1024)
    return path


# ─── is_multimodal_model ──────────────────────────────────────


class TestIsMultimodalModel:
    def test_kimi_k25_full_name(self):
        from chcode.utils.multimodal import is_multimodal_model

        assert is_multimodal_model("moonshotai/Kimi-K2.5") is True

    def test_kimi_k25_short_name(self):
        from chcode.utils.multimodal import is_multimodal_model

        assert is_multimodal_model("Kimi-K2.5") is True

    def test_qwen3_vl(self):
        from chcode.utils.multimodal import is_multimodal_model

        assert is_multimodal_model("Qwen/Qwen3-VL-235B-A22B-Instruct") is True

    def test_qwen3_vl_short(self):
        from chcode.utils.multimodal import is_multimodal_model

        assert is_multimodal_model("Qwen3-VL-30B-A3B-Instruct") is True

    def test_qwen3_5_397b(self):
        from chcode.utils.multimodal import is_multimodal_model

        assert is_multimodal_model("Qwen/Qwen3.5-397B-A17B") is True

    def test_qwen3_5_122b(self):
        from chcode.utils.multimodal import is_multimodal_model

        assert is_multimodal_model("Qwen/Qwen3.5-122B-A10B") is True

    def test_qwen3_5_35b(self):
        from chcode.utils.multimodal import is_multimodal_model

        assert is_multimodal_model("Qwen/Qwen3.5-35B-A3B") is True

    def test_qwen3_5_27b(self):
        from chcode.utils.multimodal import is_multimodal_model

        assert is_multimodal_model("Qwen/Qwen3.5-27B") is True

    def test_case_insensitive(self):
        from chcode.utils.multimodal import is_multimodal_model

        assert is_multimodal_model("MOONSHOTAI/KIMI-K2.5") is True

    def test_non_multimodal_model(self):
        from chcode.utils.multimodal import is_multimodal_model

        assert is_multimodal_model("glm-5") is False

    def test_deepseek_not_multimodal(self):
        from chcode.utils.multimodal import is_multimodal_model

        assert is_multimodal_model("deepseek-chat") is False

    def test_empty_string(self):
        from chcode.utils.multimodal import is_multimodal_model

        assert is_multimodal_model("") is False

    def test_none_value(self):
        from chcode.utils.multimodal import is_multimodal_model

        assert is_multimodal_model(None) is False


# ─── encode_media_as_base64 ───────────────────────────────────


class TestEncodeMediaAsBase64:
    def test_encode_png(self, temp_image):
        from chcode.utils.multimodal import encode_media_as_base64

        b64, mime = encode_media_as_base64(temp_image)
        assert isinstance(b64, str)
        assert len(b64) > 0
        assert mime == "image/png"

    def test_encode_jpeg(self, temp_jpeg):
        from chcode.utils.multimodal import encode_media_as_base64

        b64, mime = encode_media_as_base64(temp_jpeg)
        assert isinstance(b64, str)
        assert len(b64) > 0
        assert mime == "image/jpeg"

    def test_encode_video(self, temp_video):
        from chcode.utils.multimodal import encode_media_as_base64

        b64, mime = encode_media_as_base64(temp_video)
        assert isinstance(b64, str)
        assert len(b64) > 0
        assert mime == "video/mp4"

    def test_resize_large_image(self, temp_large_image):
        from chcode.utils.multimodal import encode_media_as_base64

        b64, mime = encode_media_as_base64(temp_large_image, max_side=2048)
        # Should succeed — large image is resized
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_file_not_found(self, tmp_path):
        from chcode.utils.multimodal import encode_media_as_base64

        with pytest.raises(FileNotFoundError):
            encode_media_as_base64(tmp_path / "nonexistent.png")

    def test_not_a_file(self, tmp_path):
        from chcode.utils.multimodal import encode_media_as_base64

        with pytest.raises(ValueError, match="Not a file"):
            encode_media_as_base64(tmp_path)

    def test_unsupported_format(self, tmp_path):
        from chcode.utils.multimodal import encode_media_as_base64

        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("not media")

        with pytest.raises(ValueError, match="Unsupported"):
            encode_media_as_base64(bad_file)

    def test_file_too_large(self, tmp_path):
        from chcode.utils.multimodal import encode_media_as_base64

        big_file = tmp_path / "big.png"
        big_file.write_bytes(b"\x00" * (21 * 1024 * 1024))

        with pytest.raises(ValueError, match="too large"):
            encode_media_as_base64(big_file)

    def test_corrupted_image(self, tmp_path):
        from chcode.utils.multimodal import encode_media_as_base64

        bad_img = tmp_path / "corrupt.png"
        bad_img.write_bytes(b"not a real image")

        with pytest.raises(IOError):
            encode_media_as_base64(bad_img)


# ─── extract_media_paths ──────────────────────────────────────


class TestExtractMediaPaths:
    def test_single_image_path(self, tmp_path):
        from chcode.utils.multimodal import extract_media_paths

        img = tmp_path / "test.png"
        img.write_bytes(b"\x00")

        result = extract_media_paths(f"看看这张图 {img}", tmp_path)
        assert len(result) == 1
        assert result[0] == img

    def test_quoted_image_path(self, tmp_path):
        from chcode.utils.multimodal import extract_media_paths

        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\x00")

        result = extract_media_paths(f'分析 "{img}"', tmp_path)
        assert len(result) == 1
        assert result[0] == img

    def test_quoted_path_with_spaces(self, tmp_path):
        """测试带空格的文件名（需要用引号括起来）"""
        from chcode.utils.multimodal import extract_media_paths

        img = tmp_path / "屏幕截图 2026-04-24 154035.png"
        img.write_bytes(b"\x00")

        result = extract_media_paths(f'分析 "{img}"', tmp_path)
        assert len(result) == 1
        assert result[0] == img

    def test_video_path(self, tmp_path):
        from chcode.utils.multimodal import extract_media_paths

        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"\x00")

        result = extract_media_paths(f"看这个视频 {vid}", tmp_path)
        assert len(result) == 1
        assert result[0] == vid

    def test_multiple_media_paths(self, tmp_path):
        from chcode.utils.multimodal import extract_media_paths

        img1 = tmp_path / "a.png"
        img1.write_bytes(b"\x00")
        img2 = tmp_path / "b.jpg"
        img2.write_bytes(b"\x00")
        vid = tmp_path / "c.mp4"
        vid.write_bytes(b"\x00")

        result = extract_media_paths(f"对比 {img1} 和 {img2} 还有 {vid}", tmp_path)
        assert len(result) == 3

    def test_no_media_paths(self, tmp_path):
        from chcode.utils.multimodal import extract_media_paths

        result = extract_media_paths("帮我写个hello world", tmp_path)
        assert len(result) == 0

    def test_nonexistent_path_filtered(self, tmp_path):
        from chcode.utils.multimodal import extract_media_paths

        result = extract_media_paths("看看 /nonexistent/image.png", tmp_path)
        assert len(result) == 0

    def test_no_duplicates(self, tmp_path):
        from chcode.utils.multimodal import extract_media_paths

        img = tmp_path / "test.png"
        img.write_bytes(b"\x00")

        result = extract_media_paths(f"看看 {img} 还有 {img}", tmp_path)
        assert len(result) == 1

    def test_relative_path_resolved(self, tmp_path):
        from chcode.utils.multimodal import extract_media_paths

        img = tmp_path / "test.png"
        img.write_bytes(b"\x00")

        result = extract_media_paths("看看 ./test.png", tmp_path)
        assert len(result) == 1
        assert result[0] == img

    def test_bare_filename_not_matched(self, tmp_path):
        """Bare filenames without path prefix should not be matched (avoids false positives)."""
        from chcode.utils.multimodal import extract_media_paths

        img = tmp_path / "test.png"
        img.write_bytes(b"\x00")

        # "test.png" without path prefix should NOT match
        result = extract_media_paths("看看 test.png", tmp_path)
        assert len(result) == 0

    def test_url_not_matched(self, tmp_path):
        """URLs like https://example.com/image.png should not be matched."""
        from chcode.utils.multimodal import extract_media_paths

        result = extract_media_paths("访问 https://example.com/image.png 查看图片", tmp_path)
        assert len(result) == 0

    def test_code_reference_not_matched(self, tmp_path):
        """Code references like 'output.png successfully' should not be matched."""
        from chcode.utils.multimodal import extract_media_paths

        img = tmp_path / "output.png"
        img.write_bytes(b"\x00")

        # Log-like text mentioning a filename should not match
        result = extract_media_paths("INFO processed output.png successfully", tmp_path)
        assert len(result) == 0

    def test_windows_backslash_path(self, tmp_path):
        """Windows-style backslash paths should be matched."""
        from chcode.utils.multimodal import extract_media_paths

        img = tmp_path / "test.png"
        img.write_bytes(b"\x00")

        # Use forward slash since Python returns forward slashes on Windows too
        result = extract_media_paths(f"看看 {tmp_path}\\test.png", tmp_path)
        assert len(result) == 1


# ─── build_multimodal_message ─────────────────────────────────


class TestBuildMultimodalMessage:
    def test_single_image(self, temp_image):
        from chcode.utils.multimodal import build_multimodal_message

        msg = build_multimodal_message("看看这张图 /path/test.png", [temp_image])

        assert isinstance(msg, HumanMessage)
        assert isinstance(msg.content, list)

        # Should have text block + image_url block
        types = [block["type"] for block in msg.content]
        assert "text" in types
        assert "image_url" in types

        # Text should have path replaced with [image: ...]
        text_block = [b for b in msg.content if b["type"] == "text"][0]
        assert "[image: test.png]" in text_block["text"]

        # Image block should have data URL
        img_block = [b for b in msg.content if b["type"] == "image_url"][0]
        assert img_block["image_url"]["url"].startswith("data:image/png;base64,")

    def test_video(self, temp_video):
        from chcode.utils.multimodal import build_multimodal_message

        msg = build_multimodal_message("看这个视频 /path/clip.mp4", [temp_video])

        assert isinstance(msg, HumanMessage)
        types = [block["type"] for block in msg.content]
        assert "text" in types
        assert "video_url" in types

        text_block = [b for b in msg.content if b["type"] == "text"][0]
        assert "[video: clip.mp4]" in text_block["text"]

        video_block = [b for b in msg.content if b["type"] == "video_url"][0]
        assert video_block["video_url"]["url"].startswith("data:video/mp4;base64,")

    def test_multiple_media(self, temp_image, temp_video):
        from chcode.utils.multimodal import build_multimodal_message

        msg = build_multimodal_message("对比图片和视频", [temp_image, temp_video])

        types = [block["type"] for block in msg.content]
        assert types.count("text") == 1
        assert "image_url" in types
        assert "video_url" in types

    def test_text_without_media_paths(self, tmp_path):
        from chcode.utils.multimodal import build_multimodal_message

        msg = build_multimodal_message("just some text", [])

        assert isinstance(msg, HumanMessage)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        assert msg.content[0]["type"] == "text"
        assert msg.content[0]["text"] == "just some text"

    def test_same_name_files_different_dirs(self, temp_image):
        """Two files with same name in different dirs should both be embedded correctly."""
        from chcode.utils.multimodal import build_multimodal_message

        dir1 = temp_image.parent / "dir1"
        dir2 = temp_image.parent / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        img1 = dir1 / "test.png"
        img2 = dir2 / "test.png"
        import shutil
        shutil.copy(temp_image, img1)
        shutil.copy(temp_image, img2)

        text = f"对比 {img1} 和 {img2}"
        msg = build_multimodal_message(text, [img1, img2])

        types = [block["type"] for block in msg.content]
        assert types.count("image_url") == 2
