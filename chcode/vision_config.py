"""
视觉模型配置管理 — 读取/保存 vision_model.json，配置视觉理解模型

视觉模型通过 ModelScope OpenAI 兼容 API 调用，
发送 base64 编码图片 + 文本 prompt，获取图像理解结果。
"""

from __future__ import annotations

import copy
import json
import os

from rich.console import Console

from chcode.config import CONFIG_DIR, ensure_config_dir
from chcode.prompts import select, confirm, password

console = Console()

VISION_JSON = CONFIG_DIR / "vision_model.json"

MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"

# 视觉模型预设 — 参数全部相同，只需列出模型名
_VISION_MODEL_NAMES = [
    "moonshotai/Kimi-K2.5",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-27B",
]

VISION_MODEL_PRESETS = [
    {"model": name, "base_url": MODELSCOPE_BASE_URL, "temperature": 1.0, "top_p": 0.95, "stream_usage": True}
    for name in _VISION_MODEL_NAMES
]


_vision_json_cache: tuple[float, dict] | None = None


def load_vision_json() -> dict:
    """加载 vision_model.json，带 mtime 缓存"""
    global _vision_json_cache
    if not VISION_JSON.exists():
        return {}
    try:
        mtime = VISION_JSON.stat().st_mtime
        if _vision_json_cache and _vision_json_cache[0] == mtime:
            return _vision_json_cache[1]
        data = json.loads(VISION_JSON.read_text(encoding="utf-8"))
        _vision_json_cache = (mtime, data)
        return data
    except Exception:
        return {}


def save_vision_json(data: dict) -> None:
    global _vision_json_cache
    content = json.dumps(data, indent=4, ensure_ascii=False)
    ensure_config_dir()
    tmp = VISION_JSON.with_suffix(".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(VISION_JSON)
    except Exception:
        VISION_JSON.write_text(content, encoding="utf-8")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    _vision_json_cache = None


def get_vision_default_model() -> dict | None:
    """获取当前默认视觉模型配置"""
    data = load_vision_json()
    default = data.get("default")
    if default and default.get("api_key"):
        return default
    return None


def get_vision_fallback_models() -> list[dict]:
    """获取备用视觉模型列表"""
    data = load_vision_json()
    fallback = data.get("fallback", {})
    return [v for k, v in fallback.items() if v.get("api_key")]


def _detect_modelscope_api_key() -> str | None:
    """检测 ModelScope API Key（环境变量 → model.json）"""
    # 优先从环境变量
    key = os.getenv("ModelScopeToken", "")
    if key:
        return key

    # 从已配置的 model.json 中找 ModelScope 的 key
    model_json_path = CONFIG_DIR / "model.json"
    if model_json_path.exists():
        try:
            data = json.loads(model_json_path.read_text(encoding="utf-8"))
            default = data.get("default", {})
            if default.get("base_url") == MODELSCOPE_BASE_URL and default.get("api_key"):
                return default["api_key"]
            # 检查 fallback
            for cfg in data.get("fallback", {}).values():
                if cfg.get("base_url") == MODELSCOPE_BASE_URL and cfg.get("api_key"):
                    return cfg["api_key"]
        except Exception:  # pragma: no cover
            pass  # pragma: no cover
    return None


def _build_vision_config(api_key: str) -> dict:
    """用预设模型 + api_key 构建完整视觉配置"""
    default_cfg = dict(VISION_MODEL_PRESETS[0])
    default_cfg["api_key"] = api_key

    fallback = {}
    for preset in VISION_MODEL_PRESETS[1:]:
        cfg = dict(preset)
        cfg["api_key"] = api_key
        fallback[cfg["model"]] = cfg

    return {"default": default_cfg, "fallback": fallback}


def auto_configure_vision() -> dict | None:
    """自动配置视觉模型（静默模式，不需要用户交互）。

    从环境变量或已配置的 ModelScope key 自动生成视觉配置。
    与已有的视觉模型配置合并，不覆盖已有的默认模型。
    返回默认模型配置，失败返回 None。
    """
    api_key = _detect_modelscope_api_key()
    if not api_key:
        return None

    data = copy.deepcopy(load_vision_json())
    existing_default = data.get("default", {})
    existing_fallback = data.get("fallback", {})

    # 已有相同 key 的 ModelScope 默认配置则跳过
    if (
        existing_default.get("base_url") == MODELSCOPE_BASE_URL
        and existing_default.get("api_key") == api_key
    ):
        return existing_default

    # 已有其他提供商的默认视觉模型 → 只把 ModelScope 模型加入 fallback
    if existing_default and existing_default.get("api_key"):
        # 将 ModelScope 预设模型加入 fallback（去重）
        for preset in VISION_MODEL_PRESETS:
            cfg = dict(preset)
            cfg["api_key"] = api_key
            if cfg["model"] not in existing_fallback:
                existing_fallback[cfg["model"]] = cfg
        data["fallback"] = existing_fallback
        save_vision_json(data)
        return existing_default

    # 没有默认视觉模型 → ModelScope 设为默认
    new_default = dict(VISION_MODEL_PRESETS[0])
    new_default["api_key"] = api_key
    new_fallback = {}
    for preset in VISION_MODEL_PRESETS[1:]:
        cfg = dict(preset)
        cfg["api_key"] = api_key
        new_fallback[cfg["model"]] = cfg

    data["default"] = new_default
    data["fallback"] = {**existing_fallback, **new_fallback}
    save_vision_json(data)
    return data["default"]


async def configure_vision_interactive() -> dict | None:
    """交互式配置视觉模型（/vision 命令调用）"""
    ensure_config_dir()

    current = load_vision_json()
    current_default = current.get("default", {})
    has_config = bool(current_default and current_default.get("api_key"))

    if has_config:
        action = await select(
            "视觉模型配置:",
            ["查看当前配置", "重新配置", "切换模型", "返回"],
        )
    else:
        action = await select(
            "视觉模型未配置，是否现在配置？",
            ["配置视觉模型", "返回"],
        )

    if action is None or action == "返回":
        return None

    if action == "查看当前配置":
        _display_vision_config(current)
        return None

    if action == "切换模型":
        return await _switch_vision_model()

    # 配置
    return await _configure_vision_wizard()


async def _configure_vision_wizard() -> dict | None:
    """配置向导"""
    # 选择 API Key 来源
    env_key = os.getenv("ModelScopeToken", "")
    choices = []
    if env_key:
        choices.append(f"使用环境变量 ModelScopeToken ({env_key[:6]}...{env_key[-4:]})")
    choices.append("手动输入 API Key")

    result = await select("选择 API Key 来源:", choices)
    if result is None:
        return None

    if result.startswith("使用环境变量"):
        api_key = env_key
    else:
        api_key = await password("输入 ModelScope API Key:")
        if not api_key or not api_key.strip():
            return None
        api_key = api_key.strip()

    # 选择默认模型
    preset_names = [p["model"] for p in VISION_MODEL_PRESETS]
    default_choice = await select("选择默认视觉模型:", preset_names, default=preset_names[0])
    if default_choice is None:
        return None

    # 构建：用户选的模型作为 default，其余作为 fallback
    all_presets = {p["model"]: p for p in VISION_MODEL_PRESETS}
    default_preset = all_presets[default_choice]
    default_cfg = dict(default_preset)
    default_cfg["api_key"] = api_key

    fallback = {}
    for model_name, preset in all_presets.items():
        if model_name == default_choice:
            continue
        cfg = dict(preset)
        cfg["api_key"] = api_key
        fallback[model_name] = cfg

    config = {"default": default_cfg, "fallback": fallback}
    save_vision_json(config)

    console.print(f"[green]视觉模型配置完成: {default_choice} (默认)[/green]")
    fallback_names = ", ".join(fallback.keys())
    console.print(f"[dim]备用模型 ({len(fallback)} 个): {fallback_names}[/dim]")

    return default_cfg


async def _switch_vision_model() -> dict | None:
    """切换视觉模型（从 fallback 列表选择）"""
    data = copy.deepcopy(load_vision_json())
    default = data.get("default", {})
    fallback = data.get("fallback", {})

    if not default:  # pragma: no cover
        console.print("[yellow]请先配置默认视觉模型[/yellow]")  # pragma: no cover
        return await _configure_vision_wizard()  # pragma: no cover

    if not fallback:  # pragma: no cover
        console.print("[yellow]没有备用视觉模型可切换[/yellow]")  # pragma: no cover
        return None  # pragma: no cover

    current_name = default.get("model", "")
    choices = []
    for name in fallback:
        tag = " (当前默认)" if name == current_name else ""
        choices.append(f"{name}{tag}")

    result = await select("选择要使用的视觉模型:", choices)
    if result is None:  # pragma: no cover
        return None  # pragma: no cover

    selected_name = result.replace(" (当前默认)", "")

    ok = await confirm(f"确定切换到 {selected_name}？当前默认将移至备用列表")
    if not ok:
        return None

    selected_config = fallback.pop(selected_name)
    if default:
        fallback[current_name] = default

    data["default"] = selected_config
    data["fallback"] = fallback
    save_vision_json(data)
    console.print(f"[green]已切换到: {selected_name}[/green]")
    return selected_config


def _display_vision_config(config: dict) -> None:
    """显示当前视觉模型配置"""
    from rich.table import Table

    default = config.get("default", {})
    fallback = config.get("fallback", {})

    if not default:
        console.print("[yellow]未配置视觉模型[/yellow]")
        return

    console.print(f"[bold]默认视觉模型:[/bold] {default.get('model', '未知')}")

    if fallback:
        table = Table(title="备用视觉模型")
        table.add_column("模型", style="cyan")
        table.add_column("状态", style="green")
        for name in fallback:
            table.add_row(name, "✓")
        console.print(table)
    else:
        console.print("[dim]无备用模型[/dim]")
