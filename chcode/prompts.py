"""
统一交互层 — 所有用户交互都通过此模块

用 questionary 实现下拉列表、确认框、文本输入等。
在 async 上下文中用 asyncio.to_thread 包装同步的 questionary 调用。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import questionary

from chcode.display import console
from chcode.i18n import t
from chcode.utils.text_utils import mask_api_key
from chcode.utils.json_utils import build_default_fallback_config


async def select(
    message: str,
    choices: list[str],
    default: str | None = None,
    style=None,
) -> str | None:
    """下拉单选"""

    def _ask():
        return questionary.select(
            message=message,
            choices=choices,
            default=default,
            style=style,
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
    custom_label: str | None = None,
    custom_prompt: str | None = None,
    default: str | None = None,
) -> str | None:
    """下拉选择 + 自定义输入。末尾有「自定义输入...」选项。"""
    custom_label = custom_label or t("prompt.custom_input")
    custom_prompt = custom_prompt or t("prompt.please_input")
    choices = list(preset_choices) + [custom_label]
    result = await select(message, choices, default=default)
    if result == custom_label:
        return await text(custom_prompt)
    return result


# ─── 模型配置表单专用 ──────────────────────────────────────────

MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"

# 每个模型只需声明差异字段，base_url / stream_usage 由生成器统一填充。
# context_length 写进 metadata（ChatOpenAI 合法字段、不透传到 API），读取时与自定义模型对齐。
_MODELSCOPE_MODELS: list[dict] = [
    {"model": "Qwen/Qwen3.5-397B-A17B", "temperature": 0.6, "top_p": 0.95, "extra_body": {"top_k": 20, "repetition_penalty": 1.0}, "metadata": {"context_length": 256000}},
    {"model": "Qwen/Qwen3-235B-A22B-Thinking-2507", "temperature": 0.6, "top_p": 0.95, "extra_body": {"top_k": 20}, "metadata": {"context_length": 256000}},
    {"model": "Qwen/Qwen3-235B-A22B-Instruct-2507", "temperature": 0.7, "top_p": 0.8, "extra_body": {"top_k": 20}, "metadata": {"context_length": 256000}},
    {"model": "deepseek-ai/DeepSeek-V3.2", "temperature": 1.0, "top_p": 0.95, "metadata": {"context_length": 128000}},
    {"model": "MiniMax/MiniMax-M2.5", "temperature": 1.0, "top_p": 0.95, "extra_body": {"top_k": 40}, "metadata": {"context_length": 200000}},
    {"model": "moonshotai/Kimi-K2.5", "temperature": 1.0, "top_p": 0.95, "metadata": {"context_length": 262144}},
    {"model": "ZhipuAI/GLM-5.1", "temperature": 1.0, "top_p": 0.95, "metadata": {"context_length": 200000}},
    {"model": "Qwen/Qwen3-Next-80B-A3B-Thinking", "temperature": 0.6, "top_p": 0.95, "extra_body": {"top_k": 20}, "metadata": {"context_length": 256000}},
    {"model": "deepseek-ai/DeepSeek-V4-Flash", "temperature": 1.0, "top_p": 1.0, "metadata": {"context_length": 1048576}},
]

MODELSCOPE_PRESETS = [
    {"base_url": MODELSCOPE_BASE_URL, "stream_usage": True, **spec}
    for spec in _MODELSCOPE_MODELS
]

API_KEY_ENV_VARS = [
    ("BIGMODEL_API_KEY", "provider.bigmodel"),
    ("ModelScopeToken", "ModelScope"),
    ("OPENAI_API_KEY", "OpenAI"),
    ("DEEPSEEK_API_KEY", "DeepSeek"),
    ("DASHSCOPE_API_KEY", "provider.qwen"),
    ("ANTHROPIC_API_KEY", "Anthropic Claude"),
]

TEMPERATURE_PRESETS = ["0", "0.3", "0.5", "0.6", "0.7", "1.0", "1.5", "2.0"]
TOP_P_PRESETS = ["0.5", "0.7", "0.9", "0.95", "1.0"]
TOP_K_PRESETS = ["1", "5", "10", "20", "40", "50"]
MAX_COMPLETION_TOKENS_PRESETS = ["32768", "65536", "122880", "204800"]
FREQ_PENALTY_PRESETS = ["0", "0.2", "0.5", "1.0", "1.5", "2.0"]
PRESENCE_PENALTY_PRESETS = ["0", "0.2", "0.5", "1.0", "1.5", "2.0"]


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
    custom_prompt: str | None = None,
) -> Any:
    """单个超参输入，支持「跳过」。返回值 / _SKIP / None(取消)。"""
    custom_prompt = custom_prompt or t("prompt.please_input")
    skip_label = t("prompt.skip")
    custom_label = t("prompt.custom_input")
    choices = [skip_label] + list(preset_choices) + [custom_label]

    default = None
    if existing_value is not None and existing_value in preset_choices:
        default = existing_value

    result = await select(message, choices, default=default)
    if result is None:
        return None
    if result == skip_label:
        return _SKIP
    if result == custom_label:
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
    cfg = dict(existing_config) if existing_config else {}

    # ─── 必填字段 ───
    is_editing = bool(cfg)

    # ── 模型名称 ──
    model_name = cfg.get("model", "")
    if not model_name:
        model_name = await text(t("form.model_name"))
        if not model_name or not model_name.strip():
            return None
        model_name = model_name.strip()

    # ── Base URL ──  (直接输入，编辑模式预填当前值)
    base_url = cfg.get("base_url", "")
    default_url = base_url if (is_editing and base_url) else ""
    base_url = await text(t("form.base_url"), default=default_url)
    if not base_url or not base_url.strip():
        return None
    base_url = base_url.strip()

    # API Key — 非编辑直接输入；编辑模式可选保持当前或重新输入
    existing_api_key = cfg.get("api_key", "")
    if is_editing and existing_api_key:
        _masked = mask_api_key(existing_api_key, mask="****", short_mask="****")
        _keep_label = t("form.keep_current_key", masked=_masked)
        _reinput_label = t("form.reinput_key")
        result = await select(
            t("form.api_key_source"),
            [_keep_label, _reinput_label],
            default=_keep_label,
        )
        if result is None:
            return None
        api_key = existing_api_key if result == _keep_label else await password(t("form.input_key"))
    else:
        api_key = await password(t("form.input_key"))

    if not api_key:
        console.print(f"[red]{t('form.api_key_empty')}[/red]")
        return None

    config: dict[str, Any] = {
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
        "stream_usage": True,
    }

    # 编辑旧配置时清理已移除的 max_tokens 字段
    cfg.pop("max_tokens", None)

    # ─── 超参（可选） ───
    want_hyperparams = await confirm(t("form.configure_hyperparams"), default=False)
    if want_hyperparams:
        # temperature
        t_val = str(cfg["temperature"]) if "temperature" in cfg else None
        result = await _ask_hyperparam(
            "Temperature:",
            TEMPERATURE_PRESETS,
            existing_value=t_val,
            custom_prompt=t("form.input_temperature"),
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
            custom_prompt=t("form.input_top_p"),
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
            custom_prompt=t("form.input_top_k"),
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
            custom_prompt=t("form.input_max_tokens"),
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
            custom_prompt=t("form.input_stop"),
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
            custom_prompt=t("form.input_freq_penalty"),
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
            custom_prompt=t("form.input_presence_penalty"),
        )
        if result is None:
            return None
        if result is not _SKIP:
            config["presence_penalty"] = float(result)
        else:
            config.pop("presence_penalty", None)

        # max_retries - 固定为 4（失败 5 次后自动切换备用模型），不可配置
        config["max_retries"] = 4

    return config


async def configure_modelscope() -> dict | None:
    """魔搭快捷配置 — 只需 API Key，自动生成预定义模型配置。"""
    # 收集 API Key
    manual_label = t("modelscope.manual_input")
    env_choices = [
        f"{var} ({t(desc)})"
        for var, desc in API_KEY_ENV_VARS
        if var == "ModelScopeToken" and os.getenv(var)
    ]
    if env_choices:
        result = await select(
            t("modelscope.detected_token"), env_choices + [manual_label]
        )
        if result is None:
            return None
        if result == manual_label:
            api_key = await password(t("modelscope.input_key"))
        else:
            # 提取 env var 名
            var_name = result.split(" (")[0]
            api_key = os.getenv(var_name, "")
    else:
        api_key = await password(t("modelscope.input_key"))

    if not api_key or not api_key.strip():
        return None
    api_key = api_key.strip()

    return build_default_fallback_config(MODELSCOPE_PRESETS, api_key)
