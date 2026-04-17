"""
统一交互层 — 所有用户交互都通过此模块

用 questionary 实现下拉列表、确认框、文本输入等。
在 async 上下文中用 asyncio.to_thread 包装同步的 questionary 调用。
"""

from __future__ import annotations

import asyncio
from typing import Any

import questionary
from rich.console import Console

console = Console()


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
        return questionary.checkbox(message=message, choices=choices).ask()

    return await asyncio.to_thread(_ask) or []


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


async def select_or_custom(
    message: str,
    preset_choices: list[str],
    custom_label: str = "自定义输入...",
    custom_prompt: str = "请输入: ",
    default: str | None = None,
) -> str:
    """下拉选择 + 自定义输入。末尾有「自定义输入...」选项。"""
    choices = list(preset_choices) + [custom_label]
    result = await select(message, choices, default=default)
    if result == custom_label:
        return await text(custom_prompt)
    return result


# ─── 模型配置表单专用 ──────────────────────────────────────────

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
TOP_P_PRESETS = ["0.5", "0.7", "0.9", "0.95", "1.0"]
TOP_K_PRESETS = ["1", "5", "10", "20", "50"]
MAX_TOKENS_PRESETS = ["32768", "65536", "122880", "204800"]
MAX_COMPLETION_TOKENS_PRESETS = ["122880", "204800", "256000", "1024000"]
MAX_RETRIES_PRESETS = ["0", "1", "2", "3", "5"]
FREQ_PENALTY_PRESETS = ["0", "0.2", "0.5", "1.0", "1.5", "2.0"]
PRESENCE_PENALTY_PRESETS = ["0", "0.2", "0.5", "1.0", "1.5", "2.0"]

SKIP_LABEL = "跳过 (不设置)"


class _SkipSentinel:
    """哨兵对象，区分「跳过此字段」和「用户取消整个表单」。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "SKIP"


_SKIP = _SkipSentinel()


async def _ask_hyperparam(
    message: str,
    preset_choices: list[str],
    existing_value: str | None = None,
    custom_prompt: str = "请输入: ",
) -> Any:
    """单个超参输入，支持「跳过」。返回值 / _SKIP / None(取消)。"""
    choices = [SKIP_LABEL] + list(preset_choices) + ["自定义输入..."]

    default = None
    if existing_value is not None and existing_value in preset_choices:
        default = existing_value

    result = await select(message, choices, default=default)
    if result is None:
        return None
    if result == SKIP_LABEL:
        return _SKIP
    if result == "自定义输入...":
        raw = await text(custom_prompt)
        if raw is None or raw.strip() == "":
            return _SKIP
        return raw.strip()
    return result


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
    is_editing = bool(cfg)
    KEEP_LABEL = "保持当前值"

    # ── 模型名称 ──
    model_name = cfg.get("model", "")
    if not model_name:
        model_name = await text("输入模型名称: ")
        if not model_name or not model_name.strip():
            return None
        model_name = model_name.strip()

    # ── Base URL ──
    base_url = cfg.get("base_url", "")
    if is_editing and base_url:
        _keep_url = f"{KEEP_LABEL} ({base_url})"
        _url_choices = [_keep_url] + list(BASE_URL_PRESETS) + ["自定义输入..."]
        result = await select("选择 API Base URL:", _url_choices, default=_keep_url)
        if result is None:
            return None
        base_url = (
            base_url
            if result == _keep_url
            else (
                await text("输入 Base URL: ") if result == "自定义输入..." else result
            )
        )
    else:
        result = await select_or_custom(
            "选择 API Base URL:",
            BASE_URL_PRESETS,
            custom_prompt="输入 Base URL: ",
        )
        if result is None:
            return None
        base_url = result

    # API Key — 先展示环境变量快捷选择
    existing_api_key = cfg.get("api_key", "")

    env_choices = [
        f"{var} ({desc})" for var, desc in API_KEY_ENV_VARS if os.getenv(var)
    ]

    if is_editing:
        _masked = (
            existing_api_key[:6] + "****" + existing_api_key[-4:]
            if len(existing_api_key) > 10
            else "****"
        )
        env_choices.insert(0, f"保持当前 Key ({_masked})")

    env_choices.append("手动输入 API Key...")
    if env_choices:
        result = await select("选择 API Key 来源:", env_choices)
        if result is None:
            return None
        if result.startswith("保持当前 Key"):
            api_key = existing_api_key
        elif result == "手动输入 API Key...":
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
        t_val = str(cfg["temperature"]) if "temperature" in cfg else None
        result = await _ask_hyperparam(
            "Temperature:",
            TEMPERATURE_PRESETS,
            existing_value=t_val,
            custom_prompt="输入 temperature: ",
        )
        if result is None:
            return None
        if result is not _SKIP:
            config["temperature"] = float(result)
        else:
            config.pop("temperature", None)

        # top_p
        tp_val = str(cfg["top_p"]) if "top_p" in cfg else None
        result = await _ask_hyperparam(
            "Top P:",
            TOP_P_PRESETS,
            existing_value=tp_val,
            custom_prompt="输入 top_p: ",
        )
        if result is None:
            return None
        if result is not _SKIP:
            config["top_p"] = float(result)
        else:
            config.pop("top_p", None)

        # top_k → extra_body
        existing_extra = cfg.get("extra_body", {})
        tk_val = (
            str(existing_extra["top_k"])
            if isinstance(existing_extra, dict) and "top_k" in existing_extra
            else None
        )
        result = await _ask_hyperparam(
            "Top K:",
            TOP_K_PRESETS,
            existing_value=tk_val,
            custom_prompt="输入 top_k: ",
        )
        if result is None:
            return None
        if result is not _SKIP:
            # 合并到已有的 extra_body（可能已有 max_completion_tokens）
            _eb = dict(existing_extra) if isinstance(existing_extra, dict) else {}
            _eb["top_k"] = int(result)
            config["extra_body"] = _eb
        else:
            # 跳过 top_k，但仍保留 extra_body 中的其他字段（如 max_completion_tokens）
            if isinstance(existing_extra, dict):
                _eb = {k: v for k, v in existing_extra.items() if k != "top_k"}
                if _eb:
                    config["extra_body"] = _eb

        # max_tokens
        mt_val = str(cfg["max_tokens"]) if "max_tokens" in cfg else None
        result = await _ask_hyperparam(
            "Max Tokens:",
            MAX_TOKENS_PRESETS,
            existing_value=mt_val,
            custom_prompt="输入 max_tokens: ",
        )
        if result is None:
            return None
        if result is not _SKIP:
            config["max_tokens"] = int(result)
        else:
            config.pop("max_tokens", None)

        # max_completion_tokens → extra_body
        _eb = config.get("extra_body", {})
        mct_val = (
            str(_eb["max_completion_tokens"])
            if isinstance(_eb, dict) and "max_completion_tokens" in _eb
            else None
        )
        result = await _ask_hyperparam(
            "Max Completion Tokens:",
            MAX_COMPLETION_TOKENS_PRESETS,
            existing_value=mct_val,
            custom_prompt="输入 max_completion_tokens: ",
        )
        if result is None:
            return None
        if result is not _SKIP:
            _eb = dict(_eb) if isinstance(_eb, dict) else {}
            _eb["max_completion_tokens"] = int(result)
            config["extra_body"] = _eb
        else:
            if isinstance(_eb, dict):
                _eb = {k: v for k, v in _eb.items() if k != "max_completion_tokens"}
                if _eb:
                    config["extra_body"] = _eb
                else:
                    config.pop("extra_body", None)

        # stop_sequences
        ss_val = None
        if "stop_sequences" in cfg:
            v = cfg["stop_sequences"]
            ss_val = ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)
        result = await _ask_hyperparam(
            "Stop Sequences:",
            [],  # 无预设，只有自定义
            existing_value=ss_val,
            custom_prompt="输入停止序列 (逗号分隔): ",
        )
        if result is None:
            return None
        if result is not _SKIP:
            config["stop_sequences"] = [
                s.strip() for s in str(result).split(",") if s.strip()
            ]
        else:
            config.pop("stop_sequences", None)

        # frequency_penalty
        fp_val = str(cfg["frequency_penalty"]) if "frequency_penalty" in cfg else None
        result = await _ask_hyperparam(
            "Frequency Penalty:",
            FREQ_PENALTY_PRESETS,
            existing_value=fp_val,
            custom_prompt="输入 frequency_penalty: ",
        )
        if result is None:
            return None
        if result is not _SKIP:
            config["frequency_penalty"] = float(result)
        else:
            config.pop("frequency_penalty", None)

        # presence_penalty
        pp_val = str(cfg["presence_penalty"]) if "presence_penalty" in cfg else None
        result = await _ask_hyperparam(
            "Presence Penalty:",
            PRESENCE_PENALTY_PRESETS,
            existing_value=pp_val,
            custom_prompt="输入 presence_penalty: ",
        )
        if result is None:
            return None
        if result is not _SKIP:
            config["presence_penalty"] = float(result)
        else:
            config.pop("presence_penalty", None)

        # max_retries
        mr_val = str(cfg["max_retries"]) if "max_retries" in cfg else None
        result = await _ask_hyperparam(
            "Max Retries:",
            MAX_RETRIES_PRESETS,
            existing_value=mr_val,
            custom_prompt="输入 max_retries: ",
        )
        if result is None:
            return None
        if result is not _SKIP:
            config["max_retries"] = int(result)
        else:
            config.pop("max_retries", None)

    return config
