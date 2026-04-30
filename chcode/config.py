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

from chcode.prompts import select, confirm, model_config_form, text, configure_longcat

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
    except Exception as e:
        console.print(f"[red]Warning: 加载 {MODEL_JSON} 失败: {e}[/red]")
        return {}


def save_model_json(data: dict) -> None:
    global _model_json_cache
    content = json.dumps(data, indent=4, ensure_ascii=False)
    tmp = MODEL_JSON.with_suffix(".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(MODEL_JSON)
    except Exception:
        MODEL_JSON.write_text(content, encoding="utf-8")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
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
        choices.append("LongCat 快捷配置...")
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

        if "LongCat" in result:
            return await _configure_longcat_with_test()

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

        data = load_model_json()
        old_default = data.get("default")
        fallback = data.get("fallback", {})
        if old_default:
            old_name = old_default.get("model", "")
            if old_name and old_name not in fallback:
                fallback[old_name] = old_default
        data["default"] = config
        data["fallback"] = fallback
        save_model_json(data)
        console.print(f"[green]配置完成: {model}[/green]")

        await configure_tavily()
        return config
    else:
        console.print("[yellow]未检测到环境变量中的 API Key[/yellow]")
        choices = ["魔搭快捷配置...", "LongCat 快捷配置...", "手动配置...", "退出"]
        result = await select("选择:", choices)
        if result is None or "退出" in result:
            console.print("[dim]提示: 在环境变量中设置 API Key 后重新运行，例如:[/dim]")
            console.print("[dim]  set BIGMODEL_API_KEY=your_key[/dim]")
            console.print("[dim]或执行 chcode config new 手动配置[/dim]")
            return None
        if "魔搭" in result:
            return await _configure_modelscope_with_test()
        if "LongCat" in result:
            return await _configure_longcat_with_test()
        return await configure_new_model()


async def configure_new_model() -> dict | None:
    """新建模型配置（交互式表单）"""
    ensure_config_dir()
    result = await select("配置方式:", ["魔搭快捷配置...", "LongCat 快捷配置...", "手动配置..."])
    if result is None:
        return None
    if "魔搭" in result:
        return await _configure_modelscope_with_test()
    if "LongCat" in result:
        return await _configure_longcat_with_test()
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
        # 已有默认 — 新模型设为默认，旧默认移到 fallback
        old_name = old_default.get("model", "")
        if old_name and old_name not in fallback:
            fallback[old_name] = old_default
        data["default"] = config
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
        if old_default["model"] not in existing_fallback:
            existing_fallback[old_default["model"]] = old_default
        existing_fallback.update(ms_config["fallback"])
        data["default"] = ms_config["default"]
        data["fallback"] = existing_fallback
        save_model_json(data)
    fallback_names = ", ".join(ms_config["fallback"].keys())
    console.print(f"[green]配置完成: {default['model']} (默认)[/green]")
    console.print(f"[dim]备用模型 ({len(ms_config['fallback'])} 个): {fallback_names}[/dim]")

    # 魔搭配置完成后，自动同步视觉模型配置
    from chcode.vision_config import auto_configure_vision
    vision_default = auto_configure_vision()
    if vision_default:
        console.print(f"[dim]视觉模型已自动配置: {vision_default.get('model', '未知')}[/dim]")

    await configure_tavily()
    return default


async def _configure_longcat_with_test() -> dict | None:
    """LongCat 快捷配置：收集 API Key → 测试连接 → 保存 4 个预定义模型。"""
    lc_config = await configure_longcat()
    if lc_config is None:
        return None

    default = lc_config["default"]

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

    # 合并到已有配置，保留非 LongCat 的已有模型
    data = load_model_json()
    old_default = data.get("default")
    existing_fallback = data.get("fallback", {})

    if not old_default:
        save_model_json(lc_config)
    else:
        if old_default["model"] not in existing_fallback:
            existing_fallback[old_default["model"]] = old_default
        existing_fallback.update(lc_config["fallback"])
        data["default"] = lc_config["default"]
        data["fallback"] = existing_fallback
        save_model_json(data)
    fallback_names = ", ".join(lc_config["fallback"].keys())
    console.print(f"[green]配置完成: {default['model']} (默认)[/green]")
    console.print(f"[dim]备用模型 ({len(lc_config['fallback'])} 个): {fallback_names}[/dim]")

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
    if default and current_name not in fallback:
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
    "deepseek-v4-pro": 1000000,
    "deepseek-v4-flash": 1000000,
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
    "longcat-2.0-preview": 1000000,
    "longcat-flash-chat": 262144,
    "longcat-flash-thinking": 262144,
    "longcat-flash-lite": 500000,
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


# ─── LangSmith 配置 ──────────────────────────────────────

LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"


def load_langsmith_config() -> dict:
    """加载 LangSmith 配置（环境变量优先，其次 SETTING_JSON）"""
    tracing_env = os.getenv("LANGCHAIN_TRACING", "")
    tracing_explicit = tracing_env.lower() in ("true", "false")
    config = {
        "tracing": tracing_env.lower() == "true",
        "project": os.getenv("LANGCHAIN_PROJECT", ""),
        "api_key": os.getenv("LANGSMITH_API_KEY", ""),
    }
    # api_key 有值但 project 缺失时默认 chcode
    if config["api_key"] and not config["project"]:
        config["project"] = "chcode"
    # 如果环境变量不完整，尝试从配置文件补充
    if SETTING_JSON.exists():
        try:
            data = json.loads(SETTING_JSON.read_text(encoding="utf-8"))
            if "langsmith_tracing" in data and not data.get("langsmith_api_key"):
                return {"tracing": False, "project": "", "api_key": ""}
            if not tracing_explicit and data.get("langsmith_tracing"):
                config["tracing"] = bool(data["langsmith_tracing"])
            if not config["project"]:
                config["project"] = data.get("langsmith_project", "")
            if not config["api_key"]:
                config["api_key"] = data.get("langsmith_api_key", "")
        except Exception:
            pass
    return config


def save_langsmith_config(tracing: bool, project: str, api_key: str) -> None:
    """保存 LangSmith 配置到 SETTING_JSON"""
    ensure_config_dir()
    data = {}
    if SETTING_JSON.exists():
        try:
            data = json.loads(SETTING_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    data["langsmith_tracing"] = tracing
    data["langsmith_project"] = project
    data["langsmith_api_key"] = api_key
    SETTING_JSON.write_text(
        json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8"
    )


def _apply_langsmith_env(tracing: bool, project: str, api_key: str) -> None:
    """将 LangSmith 配置写入环境变量"""
    os.environ["LANGCHAIN_TRACING"] = "true" if tracing else "false"
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    os.environ["LANGCHAIN_PROJECT"] = project or ""
    os.environ["LANGSMITH_API_KEY"] = api_key or ""
    os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT


async def configure_langsmith() -> dict:
    """首次引导时配置 LangSmith，返回配置 dict"""
    # 1. 环境变量已有 API Key（project 缺失时默认 chcode）
    env_key = os.getenv("LANGSMITH_API_KEY", "")
    env_project = os.getenv("LANGCHAIN_PROJECT", "")
    if env_key:
        project = env_project or "chcode"
        tracing = os.getenv("LANGCHAIN_TRACING", "true").lower() == "true"
        _apply_langsmith_env(tracing, project, env_key)
        console.print("[dim]检测到 LANGSMITH_API_KEY 环境变量，已自动配置 LangSmith[/dim]")
        return {"tracing": tracing, "project": project, "api_key": env_key}

    # 2. 配置文件已有
    if SETTING_JSON.exists():
        try:
            data = json.loads(SETTING_JSON.read_text(encoding="utf-8"))
            saved_project = data.get("langsmith_project", "")
            saved_key = data.get("langsmith_api_key", "")
            if saved_key and saved_project:
                tracing = bool(data.get("langsmith_tracing", True))
                _apply_langsmith_env(tracing, saved_project, saved_key)
                masked = saved_key[:6] + "..." + saved_key[-4:] if len(saved_key) > 10 else "***"
                console.print(
                    f"[dim]已配置 LangSmith: 项目={saved_project}, Key={masked}[/dim]"
                )
                return {"tracing": tracing, "project": saved_project, "api_key": saved_key}
            elif "langsmith_tracing" in data and not saved_key:
                return {"tracing": False, "project": "", "api_key": ""}
        except Exception:
            pass

    # 3. 引导配置
    console.print()
    result = await select("是否配置 LangSmith 追踪?", ["是", "否"])
    if result is None or result == "否":
        save_langsmith_config(False, "", "")
        console.print("[dim]已跳过，后续可通过 /langsmith 命令配置[/dim]")
        return {"tracing": False, "project": "", "api_key": ""}

    project_name = await text("请输入 LangSmith 项目名称:", default="chcode")
    api_key = await text("请输入 LangSmith API Key:")

    if not api_key:
        console.print("[dim]已取消[/dim]")
        return {"tracing": False, "project": "", "api_key": ""}

    project_name = project_name.strip() or "chcode"
    _apply_langsmith_env(True, project_name, api_key)
    save_langsmith_config(True, project_name, api_key)
    console.print("[green]LangSmith 配置已保存并生效[/green]")
    return {"tracing": True, "project": project_name, "api_key": api_key}
