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
from chcode.i18n import set_language, t
from chcode.prompts import select, confirm, model_config_form, text
from chcode.utils.json_utils import CachedJsonFile
from chcode.utils.text_utils import mask_api_key

HOMEPAGE_URL = "https://github.com/ScarletMercy/chcode" # 项目github地址
CONFIG_DIR = Path.home() / ".chat"
MODEL_JSON = CONFIG_DIR / "model.json"
SETTING_JSON = CONFIG_DIR / "chagent.json"

def _log_model_json_error(e: Exception, path: Path) -> None:
    console.print(f"[red]{t('config.load_failed', path=path, e=e)}[/red]")


_model_json = CachedJsonFile(MODEL_JSON, ensure_dir=True, on_error=_log_model_json_error)


ENV_TO_CONFIG: dict[str, dict[str, str | list[str]]] = {
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
        console.print(f"[yellow]{t('connection.testing')}[/yellow]")
    try:
        from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI

        model = EnhancedChatOpenAI(**config)
        await asyncio.to_thread(model.invoke, "你好")
    except Exception as e:
        err_msg = str(e)
        if "null value" in err_msg and "choices" in err_msg:
            return True
        if not quiet:
            console.print(f"[red]{t('connection.failed', error=err_msg)}[/red]")
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


def load_language() -> str | None:
    """读取已保存的 UI 语言（'zh' | 'en'），未设置返回 None。"""
    data = _load_setting()
    lang = data.get("language")
    if lang in ("zh", "en"):
        return lang
    return None


def save_language(lang: str) -> None:
    """持久化 UI 语言到 SETTING_JSON。"""
    if lang not in ("zh", "en"):
        return
    _update_setting(language=lang)


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


async def _ask_language_first_run() -> str:
    """首次运行：选择语言（向导第一项）。默认中文。"""
    zh_label = "中文 (Chinese)"
    en_label = "English"
    from questionary import Style
    _no_bg = Style([("highlighted", "noinherit"), ("selected", "noinherit")])
    result = await select(t("cmd.lang"), [zh_label, en_label], default=zh_label, style=_no_bg)
    lang = "en" if result == en_label else "zh"
    set_language(lang)
    save_language(lang)
    console.print(
        f"[green]{t('lang.saved_zh') if lang == 'zh' else t('lang.saved_en')}[/green]"
    )
    return lang


async def first_run_configure() -> dict | None:
    """首次运行配置引导"""
    # 第一项：选择语言（决定后续整个向导的提示语言）
    await _ask_language_first_run()

    console.print()
    console.print(
        Panel(
            t("firstrun.panel"),
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    detected = detect_env_api_keys()

    if detected:
        choices: list[str] = []
        choice_ids: list = []  # ("env", dict) | "modelscope" | "manual" | "exit"
        for d in detected:
            choices.append(
                t("firstrun.env_detected", name=d["name"], env_var=d["env_var"])
            )
            choice_ids.append(("env", d))
        choices.append(t("firstrun.modelscope_quick"))
        choice_ids.append("modelscope")
        choices.append(t("firstrun.manual"))
        choice_ids.append("manual")
        choices.append(t("firstrun.exit"))
        choice_ids.append("exit")

        result = await select(t("firstrun.select_method"), choices)
        if result is None:
            console.print(f"[dim]{t('firstrun.exit_hint')}[/dim]")
            return None
        idx = choices.index(result)
        action = choice_ids[idx]
        if action == "exit":
            console.print(f"[dim]{t('firstrun.exit_hint')}[/dim]")
            return None
        if action == "manual":
            return await configure_new_model(skip_method_select=True)
        if action == "modelscope":
            return await _configure_modelscope_with_test()

        # env provider
        chosen = action[1]

        model_list = chosen["models"]
        model = await select(t("model.select_to_use"), model_list)
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
        console.print(f"[green]{t('firstrun.config_done', model=model)}[/green]")

        await configure_tavily()
        await configure_langsmith()
        return config
    else:
        console.print(f"[yellow]{t('firstrun.no_env_key')}[/yellow]")
        choices = [
            t("firstrun.modelscope_quick"),
            t("firstrun.manual"),
            t("firstrun.exit"),
        ]
        choice_ids = ["modelscope", "manual", "exit"]
        result = await select(t("firstrun.select"), choices)
        if result is None:
            console.print(f"[dim]{t('firstrun.env_hint')}[/dim]")
            console.print(f"[dim]{t('firstrun.env_example')}[/dim]")
            console.print(f"[dim]{t('firstrun.manual_cmd_hint')}[/dim]")
            return None
        idx = choices.index(result)
        action = choice_ids[idx]
        if action == "exit":
            console.print(f"[dim]{t('firstrun.env_hint')}[/dim]")
            console.print(f"[dim]{t('firstrun.env_example')}[/dim]")
            console.print(f"[dim]{t('firstrun.manual_cmd_hint')}[/dim]")
            return None
        if action == "modelscope":
            return await _configure_modelscope_with_test()
        return await configure_new_model(skip_method_select=True)


async def configure_new_model(*, skip_method_select: bool = False) -> dict | None:
    """新建模型配置（交互式表单）

    skip_method_select=True 时跳过"配置方式"选择、直接进手动表单
    （供 first_run_configure 等已选过方式的入口调用，避免重复询问）。
    """
    ensure_config_dir()
    if not skip_method_select:
        choices = [t("firstrun.modelscope_quick"), t("firstrun.manual")]
        result = await select(t("model.method_select"), choices)
        if result is None:
            return None
        if choices.index(result) == 0:
            return await _configure_modelscope_with_test()
    config = await model_config_form()
    if config is None:
        return None

    if not await _test_connection(config):
        return None

    # 上下文长度（默认 1M）—— 多模态询问之前；存入 metadata（ChatOpenAI 合法字段，
    # 不会透传到 API 请求），随 config 一起落盘到 model.json。
    raw_ctx = await text(t("model.context_length"), default="1048576") or "1048576"
    try:
        ctx_len = int(raw_ctx.strip().replace(",", ""))
    except ValueError:
        ctx_len = 1048576
    if ctx_len <= 0:
        ctx_len = 1048576
    config["metadata"] = {**(config.get("metadata") or {}), "context_length": ctx_len}

    _merge_and_save_config(config)
    console.print(f"[green]{t('model.saved', model=config['model'])}[/green]")

    # 多模态询问（仅手动配置入口触发；魔搭快捷配置已处理视觉）
    if await confirm(t("model.multimodal_ask"), default=False):
        from chcode.vision_config import add_vision_model

        try:
            role = add_vision_model(config)
            if role == "default":
                console.print(f"[green]{t('model.vision_added_default', model=config['model'])}[/green]")
            elif role == "fallback":
                console.print(f"[green]{t('model.vision_added_fallback', model=config['model'])}[/green]")
            else:
                console.print(f"[yellow]{t('model.vision_duplicate')}[/yellow]")
        except Exception as e:
            # 视觉配置失败不阻断主流程（模型本身已保存）
            console.print(f"[yellow]{t('model.vision_failed', error=e)}[/yellow]")

    await configure_tavily()
    await configure_langsmith()
    return config


async def _configure_modelscope_with_test() -> dict | None:
    """魔搭快捷配置：收集 API Key → 测试连接 → 保存预定义模型。"""
    from chcode.prompts import configure_modelscope

    ms_config = await configure_modelscope()
    if ms_config is None:
        return None

    default = ms_config["default"]

    # 测试连接（依次尝试 default + 2 个备用模型，应对速率限制）
    console.print(f"[yellow]{t('modelscope.testing')}[/yellow]")
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
        console.print(f"[red]{t('modelscope.connect_failed', detail=last_err_detail.split(chr(10))[0] if last_err_detail else '')}[/red]")
        if last_err_detail:
            _, *tb_lines = last_err_detail.split("\n")
            console.print(f"[dim]{chr(10).join(tb_lines)}[/dim]")
        return None

    _merge_and_save_config(default, fallback_updates=ms_config["fallback"])
    fallback_names = ", ".join(ms_config["fallback"].keys())
    console.print(f"[green]{t('modelscope.config_done', model=default['model'])}[/green]")
    console.print(f"[dim]{t('modelscope.fallback_count', count=len(ms_config['fallback']), names=fallback_names)}[/dim]")

    # 魔搭配置完成后，自动同步视觉模型配置
    from chcode.vision_config import auto_configure_vision
    vision_default = auto_configure_vision()
    if vision_default:
        console.print(f"[dim]{t('modelscope.vision_auto', model=vision_default.get('model', t('modelscope.vision_unknown')))}[/dim]")

    await configure_tavily()
    await configure_langsmith()
    return default


async def edit_current_model() -> dict | None:
    """编辑当前默认模型"""
    data = load_model_json()
    current = data.get("default", {})
    if not current:
        console.print(f"[yellow]{t('model.edit_none')}[/yellow]")
        return await configure_new_model()

    config = await model_config_form(existing_config=current)
    if config is None:
        return None

    # 保留非表单字段(如自定义 context_length),避免编辑后回退默认值
    if isinstance(current.get("metadata"), dict):
        config["metadata"] = {**current["metadata"], **(config.get("metadata") or {})}

    if not await _test_connection(config):
        return None

    data["default"] = config
    save_model_json(data)
    console.print(f"[green]{t('model.updated', model=config['model'])}[/green]")
    return config


async def switch_model() -> dict | None:
    """切换模型（从 fallback 列表选择）"""
    data = load_model_json()
    default = data.get("default", {})
    fallback = data.get("fallback", {})

    if not default:
        console.print(f"[yellow]{t('model.switch_no_default')}[/yellow]")
        return await configure_new_model()

    if not fallback:
        console.print(f"[yellow]{t('model.switch_no_fallback')}[/yellow]")
        return None

    # 构建选项列表（带“当前默认”标记）；choice_names 平行保存干净模型名，
    # 供语言无关地取回选中项（避免依赖翻译后的标记文本）
    current_name = default.get("model", "")
    choices = []
    choice_names = []
    for name in fallback:
        tag = t("model.current_default_tag") if name == current_name else ""
        choices.append(f"{name}{tag}")
        choice_names.append(name)

    result = await select(t("model.select_to_use"), choices)
    if result is None:
        return None

    selected_name = choice_names[choices.index(result)]

    ok = await confirm(t("model.switch_confirm", model=selected_name))
    if not ok:
        return None

    selected_config = fallback.pop(selected_name)
    if default and current_name not in fallback:
        fallback[current_name] = default

    data["default"] = selected_config
    data["fallback"] = fallback
    save_model_json(data)
    console.print(f"[green]{t('model.switched', model=selected_name)}[/green]")
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

# 上下文长度写在每个模型配置的 metadata.context_length 里(预定义预设已内置、
# 手动配置时会提示);此常量仅作缺失该字段时的兜底(环境变量检测路径 / 旧配置)。
_DEFAULT_CONTEXT_WINDOW = 204800


async def configure_tavily() -> None:
    """首次引导时配置 Tavily"""
    tavily_env = os.getenv("TAVILY_API_KEY")

    if tavily_env:
        save_tavily_api_key(tavily_env)
        from chcode.utils.tools import update_tavily_api_key

        update_tavily_api_key(tavily_env)
        console.print(f"[dim]{t('tavily.detected_env')}[/dim]")
        return

    data = _load_setting()
    current = data.get("tavily_api_key", "")
    if current:
        from chcode.utils.tools import update_tavily_api_key

        update_tavily_api_key(current)
        console.print(
            f"[dim]{t('tavily.configured', key=mask_api_key(current))}[/dim]"
        )
        return

    console.print()
    result = await select(t("tavily.ask_configure"), [t("common.yes"), t("common.no")])
    if result is None or result == t("common.no"):
        console.print(f"[dim]{t('tavily.skipped')}[/dim]")
        return

    new_key = await text(t("tavily.input_key"))
    if new_key:
        save_tavily_api_key(new_key)
        from chcode.utils.tools import update_tavily_api_key

        update_tavily_api_key(new_key)
        console.print(f"[green]{t('tavily.saved')}[/green]")
    else:
        console.print(f"[dim]{t('common.cancelled')}[/dim]")


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
        console.print(f"[dim]{t('langsmith.detected_env')}[/dim]")
        return {"tracing": tracing, "project": project, "api_key": env_key}

    # 3. 引导配置
    console.print()
    result = await select(t("langsmith.ask_configure"), [t("common.yes"), t("common.no")])
    if result is None or result == t("common.no"):
        os.environ["LANGSMITH_TRACING"] = "false"
        _persist_env("LANGSMITH_TRACING", "false")
        console.print(f"[dim]{t('langsmith.skipped')}[/dim]")
        return {"tracing": False, "project": "", "api_key": ""}

    project_name = await text(t("langsmith.input_project"), default="chcode") or "chcode"
    api_key = await text(t("langsmith.input_key"))

    if not api_key:
        console.print(f"[dim]{t('common.cancelled')}[/dim]")
        return {"tracing": False, "project": "", "api_key": ""}

    project_name = project_name.strip() or "chcode"
    _apply_langsmith_env(True, project_name, api_key)
    console.print(f"[green]{t('langsmith.saved')}[/green]")
    return {"tracing": True, "project": project_name, "api_key": api_key}
