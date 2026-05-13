"""
模型配置管理 — 读取/保存 model.json，切换模型
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from rich.panel import Panel

from chcode.display import console
from chcode.prompts import select, confirm, model_config_form, text, configure_longcat
from chcode.utils.json_utils import CachedJsonFile
from chcode.utils.text_utils import mask_api_key

CONFIG_DIR = Path.home() / ".chat"
MODEL_JSON = CONFIG_DIR / "model.json"
SETTING_JSON = CONFIG_DIR / "chagent.json"
HOMEPAGE_URL = "https://github.com/ScarletMercy/chcode"

def _log_model_json_error(e: Exception, path: Path) -> None:
    console.print(f"[red]Warning: 加载 {path} 失败: {e}[/red]")


_model_json = CachedJsonFile(MODEL_JSON, ensure_dir=True, on_error=_log_model_json_error)


ENV_TO_CONFIG: dict[str, dict[str, str | list[str]]] = {
    # "ZAI_API_KEY": {
    #     "name": "智谱 GLM Coding Plan",
    #     "base_url": "https://open.bigmodel.cn/api/coding/paas/v4 ",
    #     "models": ["glm-4.7", "glm-5","glm-5-turbo","glm-5.1"],
    # },  # 智谱官方coding plan暂不支持本编程工具
    "OPENAI_API_KEY": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-5.4", "gpt-5.3"],
    },
    "DEEPSEEK_API_KEY": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-v4-flash","deepseek-v4-pro","deepseek-r1-0528","deepseek-v3.2"],
    },
    "MINIMAX_TOKEN_PLAN_KEY":{
        "name":"MiniMaxToken",
        "base_url":"https://api.minimaxi.com/v1",
        "models":["MiniMax-M2.7","MiniMax-M2.5","MiniMax-M2.1","MiniMax-M2","MiniMax-M2.7-highspeed","MiniMax-M2.5-highspeed","MiniMax-M2.1-highspeed"]
    },
    "KIMI_API_KEY":{
        'name':"KimiCode",
        "base_url":"https://api.kimi.com/coding/v1",
        "models":["kimi-for-coding"]
    }
    # "DASHSCOPE_API_KEY": {
    #     "name": "通义千问",
    #     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    #     "models": ["qwen3.5-plus", "qwen-turbo"],
    # },
}

# 确保.chat配置目录存在
def ensure_config_dir() -> Path:
    CONFIG_DIR.mkdir(exist_ok=True)
    return CONFIG_DIR


def load_model_json() -> dict:
    _model_json.path = MODEL_JSON
    return _model_json.load()


def save_model_json(data: dict) -> None:
    _model_json.path = MODEL_JSON
    ensure_config_dir()
    _model_json.save(data)


async def _test_connection(
    config: dict, *, quiet: bool = False, brief: bool = False, return_error: bool = False
) -> bool | str:
    """测试模型连接，成功返回 True。

    quiet=True 时不打印任何输出（用于重试循环）。
    brief=True 时只打印简短错误（不输出 traceback）。
    return_error=True 时，失败返回错误信息字符串而非 False。
    """
    if not quiet:
        console.print("[yellow]测试连接中...[/yellow]")
    try:
        from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI

        model = EnhancedChatOpenAI(**config)
        await asyncio.to_thread(model.invoke, "你好")
    except Exception as e:
        err_msg = str(e)
        if "null value" in err_msg and "choices" in err_msg:
            return True
        if not quiet:
            console.print(f"[red]连接测试失败: {err_msg}[/red]")
            if not brief:
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
        if return_error:
            return f"{err_msg}\n{traceback.format_exc()}"
        return False
    return True


def _merge_and_save_config(
    new_config: dict, fallback_updates: dict | None = None
) -> None:
    """将新配置合并到 model.json：old default → fallback，new_config → default。"""
    data = load_model_json()
    old_default = data.get("default")
    fallback = data.get("fallback", {})

    if old_default:
        old_name = old_default.get("model", "")
        if old_name and old_name not in fallback:
            fallback[old_name] = old_default
    if fallback_updates:
        fallback.update(fallback_updates)

    data["default"] = new_config
    data["fallback"] = fallback
    save_model_json(data)


def _load_setting() -> dict:
    """读取 SETTING_JSON，失败返回空 dict。"""
    if SETTING_JSON.exists():
        try:
            return json.loads(SETTING_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _update_setting(**kwargs) -> None:
    """更新 SETTING_JSON 中的指定字段。"""
    ensure_config_dir()
    data = _load_setting()
    data.update(kwargs)
    SETTING_JSON.write_text(
        json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8"
    )


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

        if not await _test_connection(config, brief=True):
            return None

        _merge_and_save_config(config)
        console.print(f"[green]配置完成: {model}[/green]")

        await configure_tavily()
        await configure_langsmith()
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

    if not await _test_connection(config):
        return None

    _merge_and_save_config(config)
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

    # 测试连接（依次尝试 default + 2 个备用模型，应对速率限制）
    console.print("[yellow]测试连接中...[/yellow]")
    test_configs = [default]
    for i, (_, cfg) in enumerate(ms_config["fallback"].items()):
        if i >= 2:
            break
        test_configs.append(cfg)

    connected = False
    last_err_detail = None
    for tc in test_configs:
        result = await _test_connection(tc, quiet=True, return_error=True)
        if result is True:
            connected = True
            break
        last_err_detail = result

    if not connected:
        console.print(f"[red]连接测试失败: {last_err_detail.split(chr(10))[0] if last_err_detail else ''}[/red]")
        if last_err_detail:
            _, *tb_lines = last_err_detail.split("\n")
            console.print(f"[dim]{chr(10).join(tb_lines)}[/dim]")
        return None

    _merge_and_save_config(default, fallback_updates=ms_config["fallback"])
    fallback_names = ", ".join(ms_config["fallback"].keys())
    console.print(f"[green]配置完成: {default['model']} (默认)[/green]")
    console.print(f"[dim]备用模型 ({len(ms_config['fallback'])} 个): {fallback_names}[/dim]")

    # 魔搭配置完成后，自动同步视觉模型配置
    from chcode.vision_config import auto_configure_vision
    vision_default = auto_configure_vision()
    if vision_default:
        console.print(f"[dim]视觉模型已自动配置: {vision_default.get('model', '未知')}[/dim]")

    await configure_tavily()
    await configure_langsmith()
    return default


async def _configure_longcat_with_test() -> dict | None:
    """LongCat 快捷配置：收集 API Key → 测试连接 → 保存 4 个预定义模型。"""
    lc_config = await configure_longcat()
    if lc_config is None:
        return None

    default = lc_config["default"]

    if not await _test_connection(default):
        return None

    _merge_and_save_config(default, fallback_updates=lc_config["fallback"])
    fallback_names = ", ".join(lc_config["fallback"].keys())
    console.print(f"[green]配置完成: {default['model']} (默认)[/green]")
    console.print(f"[dim]备用模型 ({len(lc_config['fallback'])} 个): {fallback_names}[/dim]")

    await configure_tavily()
    await configure_langsmith()
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

    if not await _test_connection(config):
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
    data = _load_setting()
    wp = data.get("workplace_path", "")
    return Path(wp) if wp else None


def save_workplace(path: Path) -> None:
    _update_setting(workplace_path=str(path))


def load_tavily_api_key() -> str:
    """加载 Tavily API Key"""
    data = _load_setting()
    if "tavily_api_key" in data:
        return data["tavily_api_key"]
    return os.getenv("TAVILY_API_KEY", "")


def save_tavily_api_key(api_key: str) -> None:
    """保存 Tavily API Key"""
    _update_setting(tavily_api_key=api_key)


# ─── 上下文窗口大小 ──────────────────────────────────────────

CONTEXT_WINDOW_SIZES: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "claude-sonnet-4-20250514": 200000,
    "deepseek-chat": 65536,
    "deepseek-v3.2": 128000,
    "deepseek-r1-0528": 65536,
    "deepseek-v4-pro": 1048576,
    "deepseek-v4-flash": 1048576,
    "glm-5.1": 200000,
    "glm-5": 200000,
    "glm-4.7": 200000,
    "minimax-m2": 204800,
    "minimax-m2.5": 200000,
    "kimi-k2": 262144,
    "mimo-v2-flash": 262144,
    "qwen3.5-plus": 1048576,
    "qwen3.6-plus": 1048576,
    "qwen": 256000,
    "longcat-2.0-preview": 1048576,
    "longcat-flash-chat": 262144,
    "longcat-flash-thinking": 262144,
    "longcat-flash-lite": 262144,
}

_DEFAULT_CONTEXT_WINDOW = 204800


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

    data = _load_setting()
    current = data.get("tavily_api_key", "")
    if current:
        from chcode.utils.tools import update_tavily_api_key

        update_tavily_api_key(current)
        console.print(
            f"[dim]已配置 Tavily: {mask_api_key(current)}[/dim]"
        )
        return

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


def _sync_langsmith_config() -> None:
    """启动时同步 LangSmith 配置到 chagent.json（Windows 额外从注册表刷新 os.environ）"""
    if sys.platform == "win32":
        import winreg
        for var_name in ("LANGSMITH_TRACING", "LANGSMITH_PROJECT", "LANGSMITH_API_KEY"):
            value = None
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                    value, _ = winreg.QueryValueEx(key, var_name)
            except (FileNotFoundError, OSError):
                pass
            try:
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                ) as key:
                    value, _ = winreg.QueryValueEx(key, var_name)
            except (FileNotFoundError, OSError):
                pass
            if value is not None:
                os.environ[var_name] = str(value)

    # 只同步非空值到 chagent.json，避免空值覆盖已有配置
    settings = _load_setting()

    tracing = os.environ.get("LANGSMITH_TRACING", "")
    project = os.environ.get("LANGSMITH_PROJECT", "")
    api_key = os.environ.get("LANGSMITH_API_KEY", "")

    if tracing:
        settings["langsmith_tracing"] = tracing.lower() == "true"
    if project:
        settings["langsmith_project"] = project
    if api_key:
        settings["langsmith_api_key"] = api_key

    _update_setting(**settings)


def load_langsmith_config() -> dict:
    """加载 LangSmith 配置（注册表 → chagent.json → 读取）"""
    _sync_langsmith_config()
    settings = _load_setting()

    api_key = settings.get("langsmith_api_key", "")
    project = settings.get("langsmith_project", "")

    return {
        "tracing": settings.get("langsmith_tracing", False),
        "project": project or ("chcode" if api_key else ""),
        "api_key": api_key,
    }


def _persist_env_linux(name: str, value: str) -> None:
    """写入环境变量到 Linux shell 配置文件"""
    home = Path.home()

    # 按优先级查找配置文件
    target_file = home / ".profile"
    for pf in [home / ".bashrc", home / ".zshrc", home / ".profile"]:
        if pf.exists():
            target_file = pf
            break

    # 转义特殊字符
    escaped_value = value.replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
    export_line = f'export {name}="{escaped_value}"'

    # 读取并更新内容
    content = target_file.read_text(encoding="utf-8") if target_file.exists() else ""

    import re
    pattern = rf'^export\s+{re.escape(name)}=.*$'
    if re.search(pattern, content, re.MULTILINE):
        content = re.sub(pattern, export_line, content, flags=re.MULTILINE)
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content += export_line + "\n"

    target_file.write_text(content, encoding="utf-8")


def _persist_env(name: str, value: str) -> None:
    """写入环境变量到系统（Windows 注册表 / Linux shell 配置文件）"""
    if sys.platform == "win32":
        r = subprocess.run(["setx", "/M", name, value], capture_output=True)
        if r.returncode != 0:
            subprocess.run(["setx", name, value], capture_output=True)
    else:
        _persist_env_linux(name, value)


def _apply_langsmith_env(tracing: bool, project: str, api_key: str) -> None:
    """将 LangSmith 配置写入环境变量（当前进程 + 注册表永久生效）+ chagent.json"""
    env_map = {
        "LANGSMITH_TRACING": "true" if tracing else "false",
        "LANGSMITH_PROJECT": project or "",
        "LANGSMITH_API_KEY": api_key or "",
    }
    os.environ.update(env_map)
    _update_setting(langsmith_tracing=tracing, langsmith_project=project, langsmith_api_key=api_key)
    # LANGSMITH_ENDPOINT 固定值，缺失时自动补上
    if not os.getenv("LANGSMITH_ENDPOINT"):
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        _persist_env("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    # 写入环境变量到系统（Windows 注册表 / Linux shell 配置文件）
    for name, value in env_map.items():
        _persist_env(name, value)


async def configure_langsmith() -> dict:
    """首次引导时配置 LangSmith，返回配置 dict"""
    # 1. LANGSMITH_TRACING 已设置 → 用户已做过选择，不再重复提示
    tracing_env = os.getenv("LANGSMITH_TRACING", "")
    if tracing_env:
        env_key = os.getenv("LANGSMITH_API_KEY", "")
        project = os.getenv("LANGSMITH_PROJECT", "") or "chcode"
        tracing = tracing_env.lower() == "true"
        return {"tracing": tracing, "project": project, "api_key": env_key}

    # 2. 环境变量已有 API Key（project 缺失时默认 chcode）
    env_key = os.getenv("LANGSMITH_API_KEY", "")
    if env_key:
        project = os.getenv("LANGSMITH_PROJECT", "") or "chcode"
        tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        _apply_langsmith_env(tracing, project, env_key)
        console.print("[dim]检测到 LANGSMITH_API_KEY 环境变量，已自动配置 LangSmith[/dim]")
        return {"tracing": tracing, "project": project, "api_key": env_key}

    # 3. 引导配置
    console.print()
    result = await select("是否配置 LangSmith 追踪?", ["是", "否"])
    if result is None or result == "否":
        os.environ["LANGSMITH_TRACING"] = "false"
        _persist_env("LANGSMITH_TRACING", "false")
        console.print("[dim]已跳过，后续可通过 /langsmith 命令配置[/dim]")
        return {"tracing": False, "project": "", "api_key": ""}

    project_name = await text("请输入 LangSmith 项目名称:", default="chcode")
    api_key = await text("请输入 LangSmith API Key:")

    if not api_key:
        console.print("[dim]已取消[/dim]")
        return {"tracing": False, "project": "", "api_key": ""}

    project_name = project_name.strip() or "chcode"
    _apply_langsmith_env(True, project_name, api_key)
    console.print("[green]LangSmith 配置已写入环境变量，重启后生效[/green]")
    return {"tracing": True, "project": project_name, "api_key": api_key}
