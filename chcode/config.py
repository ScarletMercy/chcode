"""
模型配置管理 — 读取/保存 model.json，切换模型
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from chcode.prompts import select, confirm, model_config_form

console = Console()

CONFIG_DIR = Path.home() / ".chat"
MODEL_JSON = CONFIG_DIR / "model.json"
SETTING_JSON = CONFIG_DIR / "chagent.json"


def ensure_config_dir() -> Path:
    CONFIG_DIR.mkdir(exist_ok=True)
    return CONFIG_DIR


def load_model_json() -> dict:
    """加载 model.json，不存在返回空 dict"""
    if not MODEL_JSON.exists():
        return {}
    try:
        return json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_model_json(data: dict) -> None:
    MODEL_JSON.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def get_default_model_config() -> dict | None:
    """获取当前默认模型配置"""
    data = load_model_json()
    return data.get("default") or None


def get_fallback_models() -> dict[str, dict]:
    """获取所有备用模型"""
    data = load_model_json()
    return data.get("fallback", {})


async def configure_new_model() -> dict | None:
    """新建模型配置（交互式表单）"""
    ensure_config_dir()
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
        err_msg = str(e)
        if "null value for 'choices'" not in err_msg:
            console.print(f"[red]连接测试失败: {err_msg}[/red]")
            return None

    # 保存
    data = load_model_json()
    old_default = data.get("default", {})
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
    return config


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
        err_msg = str(e)
        if "null value for 'choices'" not in err_msg:
            console.print(f"[red]连接测试失败: {err_msg}[/red]")
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
    SETTING_JSON.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def load_setting(key: str, default=None):
    if SETTING_JSON.exists():
        try:
            data = json.loads(SETTING_JSON.read_text(encoding="utf-8"))
            return data.get(key, default)
        except Exception:
            pass
    return default
