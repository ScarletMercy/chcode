"""
国际化（i18n）核心 — 轻量字典方案

提供 t(key) 翻译入口与语言管理。文案存放在 chcode/strings/{zh,en}.py。
回退链：当前语言 → 中文 → key 本身（永不抛异常）。
"""

from __future__ import annotations

import locale as _locale

from chcode.strings import CATALOGS

SUPPORTED_LANGS = ("zh", "en")
DEFAULT_LANG = "zh"

_lang: str = DEFAULT_LANG


def set_language(lang: str | None) -> str:
    """设置当前语言。无效值回退到默认语言。返回设置后的语言。"""
    global _lang
    if lang and lang.lower() in SUPPORTED_LANGS:
        _lang = lang.lower()
    else:
        _lang = DEFAULT_LANG
    return _lang


def get_language() -> str:
    """获取当前语言。"""
    return _lang


def detect_locale_language() -> str:
    """根据系统 locale 推断语言。zh* → 中文，否则英文。"""
    try:
        loc = _locale.getlocale()[0] or ""
    except Exception:
        loc = ""
    loc = (loc or os_env_lang()).lower()
    # Windows getlocale() 返回语言全名如 "Chinese (Simplified)_China"，
    # Unix 返回 locale code 如 "zh_CN"；两者均需识别
    if loc.startswith("zh") or "chinese" in loc:
        return "zh"
    return "en"


def os_env_lang() -> str:
    """读取 LANG / LC_ALL 等环境变量（跨平台兜底）。"""
    import os
    for var in ("LANG", "LC_ALL", "LC_MESSAGES"):
        val = os.getenv(var, "")
        if val:
            return val
    return ""


def t(key: str, /, **fmt) -> str:
    """翻译。回退链：当前语言 → 中文 → key 本身。支持 {name} 占位符。

    key 为位置专属参数，避免与 fmt 占位符（如 {key}）冲突。
    """
    table = CATALOGS.get(_lang, CATALOGS[DEFAULT_LANG])
    s = table.get(key)
    if s is None:
        s = CATALOGS[DEFAULT_LANG].get(key, key)
    return s.format(**fmt) if fmt else s
