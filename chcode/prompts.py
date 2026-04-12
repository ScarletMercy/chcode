"""
统一交互层 — 所有用户交互都通过此模块

用 questionary 实现下拉列表、确认框、文本输入等。
在 async 上下文中用 asyncio.to_thread 包装同步的 questionary 调用。
"""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar
from pathlib import Path

import questionary
from rich.console import Console

console = Console()

T = TypeVar("T")


async def select(
    message: str,
    choices: list[str],
    default: str | None = None,
) -> str | None:
    """下拉单选"""
    def _ask():
        return questionary.select(
            message=message,
            choices=choices,
            default=default,
        ).ask()
    return await asyncio.to_thread(_ask)


async def confirm(message: str, default: bool = True) -> bool:
    """确认框"""
    def _ask():
        return questionary.confirm(
            message=message,
            default=default,
        ).ask()
    return await asyncio.to_thread(_ask)


async def checkbox(message: str, choices: list[str]) -> list[str]:
    """多选框"""
    def _ask():
        return questionary.checkbox(
            message=message,
            choices=choices,
        ).ask()
    return await asyncio.to_thread(_ask)


async def text(message: str, default: str = "") -> str:
    """文本输入"""
    def _ask():
        return questionary.text(
            message=message,
            default=default,
        ).ask()
    return await asyncio.to_thread(_ask)


async def password(message: str) -> str:
    """密码输入（隐藏回显）"""
    def _ask():
        return questionary.password(
            message=message,
        ).ask()
    return await asyncio.to_thread(_ask)


async def path_input(message: str, default: str = "") -> str:
    """路径输入"""
    def _ask():
        return questionary.path(
            message=message,
            default=default,
            only_directories=True,
        ).ask()
    return await asyncio.to_thread(_ask)


async def select_or_custom(
    message: str,
    preset_choices: list[str],
    custom_label: str = "自定义输入...",
    custom_prompt: str = "请输入: ",
) -> str:
    """下拉选择 + 自定义输入。末尾有「自定义输入...」选项。"""
    choices = list(preset_choices) + [custom_label]
    result = await select(message, choices)
    if result == custom_label:
        return await text(custom_prompt)
    return result


# ─── 模型配置表单专用 ──────────────────────────────────────────

MODEL_PRESETS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-4-20250514",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "deepseek-chat",
    "glm-4-plus",
]

BASE_URL_PRESETS = [
    "https://api.openai.com/v1",
    "https://api-inference.modelscope.cn/v1",
    "https://open.bigmodel.cn/api/paas/v4",
    "https://api.deepseek.com/v1",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
]

API_KEY_ENV_VARS = [
    ("BIGMODEL_API_KEY", "智谱 GLM"),
    ("ModelScopeToken", "ModelScope"),
    ("OPENAI_API_KEY", "OpenAI"),
    ("DEEPSEEK_API_KEY", "DeepSeek"),
    ("DASHSCOPE_API_KEY", "通义千问"),
    ("ANTHROPIC_API_KEY", "Anthropic Claude"),
]

TEMPERATURE_PRESETS = ["0", "0.3", "0.5", "0.7", "1.0", "1.5", "2.0"]
TOP_P_PRESETS = ["0.5", "0.7", "0.9", "1.0"]
MAX_TOKENS_PRESETS = ["1024", "2048", "4096", "8192", "16384", "32768", "65536"]
FREQ_PENALTY_PRESETS = ["0", "0.2", "0.5", "1.0", "1.5", "2.0"]
PRESENCE_PENALTY_PRESETS = ["0", "0.2", "0.5", "1.0", "1.5", "2.0"]


async def model_config_form(
    existing_config: dict | None = None,
) -> dict | None:
    """
    模型配置表单 — 全部用下拉列表 + 文本输入

    Args:
        existing_config: 现有配置（编辑模式）

    Returns:
        配置字典，用户取消返回 None
    """
    import os

    cfg = dict(existing_config) if existing_config else {}

    # ─── 必填字段 ───
    model_name = cfg.get("model", "")
    if model_name:
        result = await select_or_custom(
            "选择模型:",
            MODEL_PRESETS,
            custom_prompt="输入模型名称: ",
        )
    else:
        result = await select_or_custom(
            "选择模型:",
            MODEL_PRESETS,
            custom_prompt="输入模型名称: ",
        )
    if result is None:
        return None
    model_name = result

    base_url = cfg.get("base_url", "")
    result = await select_or_custom(
        "选择 API Base URL:",
        BASE_URL_PRESETS,
        custom_prompt="输入 Base URL: ",
    )
    if result is None:
        return None
    base_url = result

    # API Key — 先展示环境变量快捷选择
    env_choices = [f"{var} ({desc})" for var, desc in API_KEY_ENV_VARS if os.getenv(var)]
    env_choices.append("手动输入 API Key...")
    if env_choices:
        result = await select("选择 API Key 来源:", env_choices)
        if result is None:
            return None
        if result == "手动输入 API Key...":
            api_key = await password("输入 API Key: ")
        else:
            var_name = result.split(" (")[0]
            api_key = os.getenv(var_name, "")
    else:
        api_key = await password("输入 API Key: ")

    if not api_key:
        console.print("[red]API Key 不能为空[/red]")
        return None

    config: dict[str, Any] = {
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
        "stream_usage": True,
    }

    # ─── 超参（可选） ───
    want_hyperparams = await confirm("配置超参数？", default=False)
    if want_hyperparams:
        # temperature
        t = cfg.get("temperature", "1.0")
        t_str = str(t) if t else "1.0"
        result = await select_or_custom("Temperature:", TEMPERATURE_PRESETS, custom_prompt="输入 temperature: ")
        if result is not None:
            config["temperature"] = float(result)

        # top_p
        tp = cfg.get("top_p", "1.0")
        tp_str = str(tp) if tp else "1.0"
        result = await select_or_custom("Top P:", TOP_P_PRESETS, custom_prompt="输入 top_p: ")
        if result is not None:
            config["top_p"] = float(result)

        # max_tokens
        mt = cfg.get("max_tokens", "")
        mt_str = str(mt) if mt else ""
        result = await select_or_custom("Max Tokens:", MAX_TOKENS_PRESETS, custom_prompt="输入 max_tokens: ")
        if result is not None:
            config["max_tokens"] = int(result)

        # frequency_penalty
        fp = cfg.get("frequency_penalty", "0")
        fp_str = str(fp) if fp else "0"
        result = await select_or_custom("Frequency Penalty:", FREQ_PENALTY_PRESETS, custom_prompt="输入 frequency_penalty: ")
        if result is not None:
            config["frequency_penalty"] = float(result)

        # presence_penalty
        pp = cfg.get("presence_penalty", "0")
        pp_str = str(pp) if pp else "0"
        result = await select_or_custom("Presence Penalty:", PRESENCE_PENALTY_PRESETS, custom_prompt="输入 presence_penalty: ")
        if result is not None:
            config["presence_penalty"] = float(result)

    return config
