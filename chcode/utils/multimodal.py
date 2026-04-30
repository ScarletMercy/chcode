"""
Multimodal model detection and media encoding utilities.

Provides:
- is_multimodal_model(): Check if a model name indicates native vision capability
- encode_media_as_base64(): Read and base64-encode an image or video
- extract_media_paths(): Detect image/video file paths in user text
- build_multimodal_message(): Construct a HumanMessage with embedded media
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

from langchain_core.messages import HumanMessage

# ─── Multimodal model patterns ──────────────────────────────────

# Models whose short name (after /) or full name matches these patterns
# are considered multimodal (native vision capability).
MULTIMODAL_MODEL_PATTERNS: list[str] = [
    # Kimi K2.5 series
    "Kimi-K2",
    # Qwen3 VL (dedicated vision-language)
    "Qwen3-VL",
    # Qwen3.5 MoE models with vision
    "Qwen3.5-397B",
    "Qwen3.5-122B",
    "Qwen3.5-35B",
    "Qwen3.5-27B",
    # Intern-S1 series
    "Intern-S1",
]


def is_multimodal_model(model_name: str) -> bool:
    """Check if a model name indicates native multimodal (vision) capability.

    Handles both short names (e.g., "Kimi-K2.5") and full names
    (e.g., "moonshotai/Kimi-K2.5"). Case-insensitive.
    """
    if not model_name:
        return False
    short_name = model_name.split("/")[-1]
    lower_name = model_name.lower()
    lower_short = short_name.lower()
    for pattern in MULTIMODAL_MODEL_PATTERNS:
        lower_pattern = pattern.lower()
        if lower_pattern in lower_short or lower_pattern in lower_name:
            return True
    return False


# ─── Supported formats ─────────────────────────────────────────

_IMAGE_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif",
})

_VIDEO_EXTS = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
})

_ALL_MEDIA_EXTS = _IMAGE_EXTS | _VIDEO_EXTS

_VIDEO_EXT_NAMES = frozenset(e.lstrip(".") for e in _VIDEO_EXTS)

_MIME_MAP: dict[str, str] = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "webp": "image/webp",
    "tiff": "image/tiff",
    "tif": "image/tiff",
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
}

# ─── Media encoding ────────────────────────────────────────────


def encode_media_as_base64(
    path: Path,
    max_side: int = 2048,
) -> tuple[str, str]:
    """Read an image or video, return (base64_data, mime_type).

    For images larger than max_side pixels, the image is resized
    before encoding. Videos are encoded without modification.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid or too large.
        IOError: If the image cannot be read/decoded.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    ext = path.suffix.lower().lstrip(".")
    if ext not in _MIME_MAP:
        raise ValueError(f"Unsupported media format: {path.suffix}")

    mime_type = _MIME_MAP.get(ext, "application/octet-stream")

    is_video = ext in _VIDEO_EXT_NAMES

    if is_video:
        file_size = path.stat().st_size
        if file_size > 14.9 * 1024 * 1024:
            raise ValueError(
                f"Video too large: {file_size / 1024 / 1024:.1f}MB (max 14.9MB)"
            )
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    else:
        from PIL import Image

        with Image.open(path) as img:
            w, h = img.size
            resized = max(w, h) > max_side
            if resized:
                scale = max_side / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                mime_type = "image/png"

            # 分辨率缩放后，如果体积仍超过 5MB，逐步降低 JPEG quality 压缩到 5MB 以内
            MAX_BYTES = 5 * 1024 * 1024
            buf = io.BytesIO()
            save_format = "PNG" if resized else (img.format or "PNG")
            img.save(buf, format=save_format)

            if buf.tell() > MAX_BYTES:
                if img.mode in ("RGBA", "P", "LA"):
                    img = img.convert("RGB")
                for quality in range(85, 4, -15):
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=quality)
                    if buf.tell() <= MAX_BYTES:
                        break
                mime_type = "image/jpeg"

            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return b64, mime_type


# ─── Media path extraction ─────────────────────────────────────

_MEDIA_EXT_PATTERN = "|".join(
    re.escape(ext.lstrip("."))
    for ext in sorted(_ALL_MEDIA_EXTS)
)

_MEDIA_PATH_PATTERN = re.compile(
    r"(?:"
    # Quoted path: "path/to/image.png" or 'path\to\image.jpg'
    # Use [^"']+ to allow spaces inside quoted paths
    r'(["\'])([^"\']+\.(?:' + _MEDIA_EXT_PATTERN + r'))\1'
    r"|"
    # Bare path: must start with /, \, ~, ./, ../, or a drive letter (C:)
    # This avoids matching URLs or code references like output.png
    # Note: bare paths cannot contain spaces (use quotes for that)
    r'((?:[/~\\]|[.]{1,2}[/\\]|[A-Za-z]:[/\\])[^\s]*\.(?:' + _MEDIA_EXT_PATTERN + r'))'
    r")",
    re.IGNORECASE,
)


def extract_media_paths(text: str, working_directory: Path) -> list[Path]:
    """Extract valid media file paths from user input text.

    Returns paths that actually exist on disk, resolved relative
    to working_directory. Deduplicates results.
    """
    found: list[Path] = []
    seen: set[str] = set()

    for match in _MEDIA_PATH_PATTERN.finditer(text):
        # Group 1-2: quoted path; Group 3: bare path
        raw_path = match.group(2) or match.group(3)
        if not raw_path:
            continue

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = working_directory / path

        path_str = str(path.resolve())
        if path_str in seen:
            continue

        if (
            path.exists()
            and path.is_file()
            and path.suffix.lower() in _ALL_MEDIA_EXTS
        ):
            seen.add(path_str)
            found.append(path)

    return found


# ─── Multimodal message builder ─────────────────────────────────


def build_multimodal_message(
    text: str,
    media_paths: list[Path],
    max_side: int = 2048,
) -> HumanMessage:
    """Build a HumanMessage with text and embedded media.

    Image paths in text are replaced with [image: filename].
    Video paths are replaced with [video: filename].
    Images use image_url type, videos use video_url type.
    """
    content_blocks: list[dict] = []

    # Replace media paths in text with reference markers.
    # Use single-pass replacement to avoid collision when multiple files
    # share the same filename (e.g., dir1/test.png and dir2/test.png).
    # Build a mapping from original text span → marker, then replace
    # from longest matches first to prevent partial replacements.
    replacements: list[tuple[str, str]] = []  # (original_text, marker)
    for media_path in media_paths:
        is_vid = media_path.suffix.lower() in _VIDEO_EXTS
        marker = f"[video: {media_path.name}]" if is_vid else f"[image: {media_path.name}]"
        matched = False
        # Try quoted and full path representations
        for sep in [f'"{media_path}"', f"'{media_path}'", str(media_path)]:
            if sep in text:
                replacements.append((sep, marker))
                matched = True
                break
        if not matched and media_path.name in text:
            replacements.append((media_path.name, marker))

    # Sort by length descending so longer paths are replaced first
    replacements.sort(key=lambda r: len(r[0]), reverse=True)
    clean_text = text
    for original, marker in replacements:
        clean_text = clean_text.replace(original, marker, 1)

    content_blocks.append({"type": "text", "text": clean_text})

    for media_path in media_paths:
        b64, mime = encode_media_as_base64(media_path, max_side=max_side)
        data_url = f"data:{mime};base64,{b64}"

        if media_path.suffix.lower() in _VIDEO_EXTS:
            content_blocks.append({
                "type": "video_url",
                "video_url": {"url": data_url},
            })
        else:
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })

    return HumanMessage(content=content_blocks)
