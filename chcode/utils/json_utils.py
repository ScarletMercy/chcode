"""
JSON 文件读写工具 — 原子写入 + mtime 缓存
"""

from __future__ import annotations

import json
from pathlib import Path


def atomic_write_json(
    path: Path,
    data: dict,
    *,
    indent: int = 4,
    ensure_dir: bool = False,
) -> None:
    content = json.dumps(data, indent=indent, ensure_ascii=False)
    if ensure_dir:
        path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        path.write_text(content, encoding="utf-8")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


class CachedJsonFile:
    def __init__(
        self,
        path: Path,
        *,
        indent: int = 4,
        ensure_dir: bool = False,
        on_error: object = None,
    ):
        self.path = path
        self.indent = indent
        self.ensure_dir = ensure_dir
        self._on_error = on_error
        self._cache: tuple[float, dict] | None = None

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            mtime = self.path.stat().st_mtime
            if self._cache and self._cache[0] == mtime:
                return self._cache[1]
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._cache = (mtime, data)
            return data
        except Exception as e:
            if self._on_error:
                self._on_error(e, self.path)
            return {}

    def save(self, data: dict) -> None:
        atomic_write_json(
            self.path, data, indent=self.indent, ensure_dir=self.ensure_dir
        )
        self._cache = None

    def invalidate(self) -> None:
        self._cache = None


def build_default_fallback_config(
    presets: list[dict], api_key: str, *, default_index: int = 0
) -> dict:
    default_cfg = dict(presets[default_index])
    default_cfg["api_key"] = api_key
    fallback = {}
    for i, preset in enumerate(presets):
        if i == default_index:
            continue
        cfg = dict(preset)
        cfg["api_key"] = api_key
        fallback[cfg["model"]] = cfg
    return {"default": default_cfg, "fallback": fallback}
