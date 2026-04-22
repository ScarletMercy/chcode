"""
模型配置管理 — 读取/保存 model.json，切换模型
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from chcode.prompts import select, confirm, model_config_form, text

console = Console()

CONFIG_DIR = Path.home() / ".chat"
MODEL_JSON = CONFIG_DIR / "model.json"
SETTING_JSON = CONFIG_DIR / "chagent.json"


ENV_TO_CONFIG: dict[str, dict[str, str | list[str]]] = {
    "BIGMODEL_API_KEY": {
        "name": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4.7", "glm-5","glm-5-turbo","glm-5.1"],
    },
    "OPENAI_API_KEY": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-5.4", "gpt-5.3"],
    },
    "DEEPSEEK_API_KEY": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat"],
    },
    "DASHSCOPE_API_KEY": {
        "name": "通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen3.5-plus", "qwen-turbo"],
    },
    "ModelScopeToken": {
        "name": "ModelScope",
        "base_url": "https://api-inference.modelscope.cn/v1",
        "models": ["Qwen/Qwen3-235B-A22B-Thinking-2507"],
    },
    "ANTHROPIC_API_KEY": {
        "name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-sonnet-4.6"],
    },
}

# 确保.chat配置目录存在
def ensure_config_dir() -> Path:
    CONFIG_DIR.mkdir(exist_ok=True)
    return CONFIG_DIR


_model_json_cache: tuple[float, dict] | None = None


def load_model_json() -> dict:
    """加载 model.json，带 mtime 缓存"""
    global _model_json_cache
    if not MODEL_JSON.exists():
        return {}
    try:
        mtime = MODEL_JSON.stat().st_mtime
        if _model_json_cache and _model_json_cache[0] == mtime:
            return _model_json_cache[1]
        data = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
        _model_json_cache = (mtime, data)
        return data
    except Exception:
        return {}


def save_model_json(data: dict) -> None:
    global _model_json_cache
    MODEL_JSON.write_text(
        json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    _model_json_cache = None


def get_default_model_config() -> dict | None:
    """获取当前默认模型配置"""
    data = load_model_json()
    return data.get("default") or None


def detect_env_api_keys() -> list[dict]:
    """检测环境变量中的 API Key，返回推荐配置列表"""
    results = []
    for var, cfg in ENV_TO_CONFIG.items():
        key = os.getenv(var, "")
        if key:
            results.append({"env_var": var, "api_key": key, **cfg})
    return results


async def first_run_configure() -> dict | None:
    """首次运行配置引导"""
    console.print()
    console.print(
        Panel(
            "[bold]ChCode[/bold] — 终端 AI 编程助手\n\n"
            "首次运行需要配置 AI 模型连接。\n"
            "设置环境变量后可自动检测（推荐），或手动填写配置。",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    detected = detect_env_api_keys()

    if detected:
        choices = [f"{d['name']} (检测到 {d['env_var']})" for d in detected]
        choices.append("魔搭快捷配置...")
        choices.append("手动配置...")
        choices.append("退出")

        result = await select("选择配置方式:", choices)
        if result is None or "退出" in result:
            console.print(
                "[dim]设置环境变量后重新运行，或执行 chcode config new 手动配置[/dim]"
            )
            return None

        if "手动" in result:
            return await configure_new_model()

        if "魔搭" in result:
            return await _configure_modelscope_with_test()

        idx = choices.index(result)
        chosen = detected[idx]

        model_list = chosen["models"]
        model = await select("选择模型:", model_list)
        if model is None:
            return None

        config: dict[str, Any] = {
            "model": model,
            "base_url": chosen["base_url"],
            "api_key": chosen["api_key"],
            "stream_usage": True,
        }

        console.print("[yellow]测试连接中...[/yellow]")
        try:
            from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI

            model_inst = EnhancedChatOpenAI(**config)
            await asyncio.to_thread(model_inst.invoke, "你好")
        except Exception as e:
            if "null value" not in str(e):
                console.print(f"[red]连接失败: {e}[/red]")
                return None

        data = {"default": config, "fallback": {}}
        save_model_json(data)
        console.print(f"[green]配置完成: {model}[/green]")

        await configure_tavily()
        return config
    else:
        console.print("[yellow]未检测到环境变量中的 API Key[/yellow]")
        choices = ["魔搭快捷配置...", "手动配置...", "退出"]
        result = await select("选择:", choices)
        if result is None or "退出" in result:
            console.print("[dim]提示: 在环境变量中设置 API Key 后重新运行，例如:[/dim]")
            console.print("[dim]  set BIGMODEL_API_KEY=your_key[/dim]")
            console.print("[dim]或执行 chcode config new 手动配置[/dim]")
            return None
        if "魔搭" in result:
            return await _configure_modelscope_with_test()
        return await configure_new_model()


async def configure_new_model() -> dict | None:
    """新建模型配置（交互式表单）"""
    ensure_config_dir()
    result = await select("配置方式:", ["魔搭快捷配置...", "手动配置..."])
    if result is None:
        return None
    if "魔搭" in result:
        return await _configure_modelscope_with_test()
    config = await model_config_form()
    if config is None:
        return None

    # 测试连接
    console.print("[yellow]测试连接中...[/yellow]")
    try:
        from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI

        model = EnhancedChatOpenAI(**config)
        await asyncio.to_thread(model.invoke, "你好")
    except Exception as e:
        import traceback

        err_msg = str(e)
        if "null value for 'choices'" not in err_msg:
            console.print(f"[red]连接测试失败: {err_msg}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return None
    data = load_model_json()
    old_default = data.get("default")
    fallback = data.get("fallback", {})

    if not old_default:
        # 第一次配置 — 直接设为默认
        data["default"] = config
        data["fallback"] = {}
    else:
        # 已有默认 — 新配置加入 fallback
        if config["model"] not in fallback:
            fallback[config["model"]] = config
            data["fallback"] = fallback

    save_model_json(data)
    console.print(f"[green]模型配置已保存: {config['model']}[/green]")

    await configure_tavily()
    return config


async def _configure_modelscope_with_test() -> dict | None:
    """魔搭快捷配置：收集 API Key → 测试连接 → 保存 12 个预定义模型。"""
    from chcode.prompts import configure_modelscope

    ms_config = await configure_modelscope()
    if ms_config is None:
        return None

    default = ms_config["default"]

    # 测试连接
    console.print("[yellow]测试连接中...[/yellow]")
    try:
        from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI

        model_inst = EnhancedChatOpenAI(**default)
        await asyncio.to_thread(model_inst.invoke, "你好")
    except Exception as e:
        import traceback

        err_msg = str(e)
        if "null value for 'choices'" not in err_msg:
            console.print(f"[red]连接测试失败: {err_msg}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return None

    # 合并到已有配置，保留非魔搭的已有模型
    data = load_model_json()
    old_default = data.get("default")
    existing_fallback = data.get("fallback", {})

    if not old_default:
        # 首次配置 — 魔搭直接作为完整配置
        save_model_json(ms_config)
    else:
        # 已有配置 — 旧的 default 移入 fallback，魔搭作为新 default，合并 fallback
        if old_default["model"] not in ms_config["fallback"]:
            existing_fallback[old_default["model"]] = old_default
        existing_fallback.update(ms_config["fallback"])
        data["default"] = ms_config["default"]
        data["fallback"] = existing_fallback
        save_model_json(data)
    fallback_names = ", ".join(ms_config["fallback"].keys())
    console.print(f"[green]配置完成: {default['model']} (默认)[/green]")
    console.print(f"[dim]备用模型 ({len(ms_config['fallback'])} 个): {fallback_names}[/dim]")

    await configure_tavily()
    return default


async def edit_current_model() -> dict | None:
    """编辑当前默认模型"""
    data = load_model_json()
    current = data.get("default", {})
    if not current:
        console.print("[yellow]没有当前模型配置，请新建[/yellow]")
        return await configure_new_model()

    config = await model_config_form(existing_config=current)
    if config is None:
        return None

    # 测试连接
    console.print("[yellow]测试连接中...[/yellow]")
    try:
        from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI

        model = EnhancedChatOpenAI(**config)
        await asyncio.to_thread(model.invoke, "你好")
    except Exception as e:
        import traceback

        err_msg = str(e)
        if "null value for 'choices'" not in err_msg:
            console.print(f"[red]连接测试失败: {err_msg}[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return None
    data["default"] = config
    save_model_json(data)
    console.print(f"[green]模型配置已更新: {config['model']}[/green]")
    return config


async def switch_model() -> dict | None:
    """切换模型（从 fallback 列表选择）"""
    data = load_model_json()
    default = data.get("default", {})
    fallback = data.get("fallback", {})

    if not default:
        console.print("[yellow]请先配置默认模型[/yellow]")
        return await configure_new_model()

    if not fallback:
        console.print("[yellow]没有备用模型可切换[/yellow]")
        return None

    # 构建选项列表
    current_name = default.get("model", "")
    choices = []
    for name in fallback:
        tag = " (当前默认)" if name == current_name else ""
        choices.append(f"{name}{tag}")

    result = await select("选择要使用的模型:", choices)
    if result is None:
        return None

    # 提取模型名（去掉 " (当前默认)" 后缀）
    selected_name = result.replace(" (当前默认)", "")

    ok = await confirm(f"确定切换到 {selected_name}？当前默认将移至备用列表")
    if not ok:
        return None

    selected_config = fallback.pop(selected_name)
    if default:
        fallback[current_name] = default

    data["default"] = selected_config
    data["fallback"] = fallback
    save_model_json(data)
    console.print(f"[green]已切换到: {selected_name}[/green]")
    return selected_config


def load_workplace() -> Path | None:
    """加载上次的工作目录"""
    if SETTING_JSON.exists():
        try:
            data = json.loads(SETTING_JSON.read_text(encoding="utf-8"))
            wp = data.get("workplace_path", "")
            if wp:
                return Path(wp)
        except Exception:
            pass
    return None


def save_workplace(path: Path) -> None:
    ensure_config_dir()
    data = {}
    if SETTING_JSON.exists():
        try:
            data = json.loads(SETTING_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    data["workplace_path"] = str(path)
    SETTING_JSON.write_text(
        json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8"
    )


def load_tavily_api_key() -> str:
    """加载 Tavily API Key"""
    if SETTING_JSON.exists():
        try:
            data = json.loads(SETTING_JSON.read_text(encoding="utf-8"))
            return data.get("tavily_api_key", "")
        except Exception:
            pass
    return os.getenv("TAVILY_API_KEY", "")


def save_tavily_api_key(api_key: str) -> None:
    """保存 Tavily API Key"""
    ensure_config_dir()
    data = {}
    if SETTING_JSON.exists():
        try:
            data = json.loads(SETTING_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    data["tavily_api_key"] = api_key
    SETTING_JSON.write_text(
        json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8"
    )


# ─── 上下文窗口大小 ──────────────────────────────────────────

CONTEXT_WINDOW_SIZES: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "claude-sonnet-4-20250514": 200000,
    "deepseek-chat": 65536,
    "deepseek-v3.2": 128000,
    "deepseek-r1-0528": 65536,
    "glm-5.1": 200000,
    "glm-5": 200000,
    "glm-4.7": 200000,
    "minimax-m2": 204800,
    "minimax-m2.5": 200000,
    "kimi-k2": 256000,
    "mimo-v2-flash": 256000,
    "qwen3.5-plus": 1000000,
    "qwen3.6-plus": 1000000,
    "qwen": 256000,
}

_DEFAULT_CONTEXT_WINDOW = 256000


def get_context_window_size(model_name: str) -> int:
    """根据模型名获取上下文窗口大小，无匹配时返回默认值"""
    if not model_name:
        return _DEFAULT_CONTEXT_WINDOW
    # 精确匹配
    if model_name in CONTEXT_WINDOW_SIZES:
        return CONTEXT_WINDOW_SIZES[model_name]
    # 前缀匹配（去掉 org/ 前缀后匹配）
    short = model_name.split("/")[-1].lower()
    if short in CONTEXT_WINDOW_SIZES:
        return CONTEXT_WINDOW_SIZES[short]
    for key, size in CONTEXT_WINDOW_SIZES.items():
        if key in model_name.lower():
            return size
    return _DEFAULT_CONTEXT_WINDOW


async def configure_tavily() -> None:
    """首次引导时配置 Tavily"""
    tavily_env = os.getenv("TAVILY_API_KEY")

    if tavily_env:
        save_tavily_api_key(tavily_env)
        from chcode.utils.tools import update_tavily_api_key

        update_tavily_api_key(tavily_env)
        console.print("[dim]检测到 TAVILY_API_KEY 环境变量，已自动配置 Tavily[/dim]")
        return

    if SETTING_JSON.exists():
        try:
            data = json.loads(SETTING_JSON.read_text(encoding="utf-8"))
            current = data.get("tavily_api_key", "")
            if current:
                from chcode.utils.tools import update_tavily_api_key

                update_tavily_api_key(current)
                console.print(
                    f"[dim]已配置 Tavily: {current[:6]}...{current[-4:]}[/dim]"
                )
                return
        except Exception:
            pass

    console.print()
    result = await select("是否配置 Tavily 搜索引擎?", ["是", "否"])
    if result is None or result == "否":
        console.print("[dim]已跳过，后续可通过 /search 命令配置[/dim]")
        return

    new_key = await text("请输入 Tavily API Key:")
    if new_key:
        save_tavily_api_key(new_key)
        from chcode.utils.tools import update_tavily_api_key

        update_tavily_api_key(new_key)
        console.print("[green]Tavily API Key 已保存并生效[/green]")
    else:
        console.print("[dim]已取消[/dim]")
