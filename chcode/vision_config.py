"""
视觉模型配置管理 — 读取/保存 vision_model.json，配置视觉理解模型

视觉模型通过 ModelScope OpenAI 兼容 API 调用，
发送 base64 编码图片 + 文本 prompt，获取图像理解结果。
"""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import os

from chcode.config import CONFIG_DIR, ensure_config_dir
from chcode.display import console
from chcode.i18n import t
from chcode.prompts import select, confirm, password, text
from chcode.utils.json_utils import CachedJsonFile, build_default_fallback_config
from chcode.utils.text_utils import mask_api_key

VISION_JSON = CONFIG_DIR / "vision_model.json"
_vision_json = CachedJsonFile(VISION_JSON, ensure_dir=True)

MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
MODELSCOPE_INTL_BASE_URL = "https://api-inference.modelscope.ai/v1"

# 视觉模型预设 — 参数全部相同，只需列出模型名
_VISION_MODEL_NAMES = [
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

# 国际版：预置模型与参数完全复用，仅 base_url 不同
VISION_MODEL_INTL_PRESETS = [
    {**preset, "base_url": MODELSCOPE_INTL_BASE_URL}
    for preset in VISION_MODEL_PRESETS
]


def load_vision_json() -> dict:
    _vision_json.path = VISION_JSON
    return _vision_json.load()


def save_vision_json(data: dict) -> None:
    _vision_json.path = VISION_JSON
    ensure_config_dir()
    _vision_json.save(data)


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


def _detect_modelscope_api_key(base_url: str) -> str | None:
    """检测指定 base_url 家族的 ModelScope API Key（环境变量 → model.json）"""
    # 优先从环境变量（与历史行为一致：env 键归国内版家族，避免双家族同名
    # 模型互相覆盖；国际版家族仅通过 model.json 中的 .ai 配置检出）
    key = os.getenv("ModelScopeToken", "")
    if key and base_url == MODELSCOPE_BASE_URL:
        return key

    # 从已配置的 model.json 中找该家族 ModelScope 的 key
    model_json_path = CONFIG_DIR / "model.json"
    if model_json_path.exists():
        try:
            data = json.loads(model_json_path.read_text(encoding="utf-8"))
            default = data.get("default", {})
            if default.get("base_url") == base_url and default.get("api_key"):
                return default["api_key"]
            # 检查 fallback
            for cfg in data.get("fallback", {}).values():
                if cfg.get("base_url") == base_url and cfg.get("api_key"):
                    return cfg["api_key"]
        except Exception:  # pragma: no cover
            pass  # pragma: no cover
    return None


def _build_vision_config(api_key: str) -> dict:
    return build_default_fallback_config(VISION_MODEL_PRESETS, api_key)


def _detect_api_keys() -> list[tuple[str, list[dict]]]:
    """检测所有可用 API Key，返回 [(api_key, matching_presets), ...]。

    每个元素对应一个提供商家族（国内版 cn / 国际版 ai），
    api_key 只会匹配同 base_url 家族的预设。
    """
    results: list[tuple[str, list[dict]]] = []

    for presets in (VISION_MODEL_PRESETS, VISION_MODEL_INTL_PRESETS):
        key = _detect_modelscope_api_key(presets[0]["base_url"])
        if key:
            results.append((key, presets))

    return results


def auto_configure_vision() -> dict | None:
    """自动配置视觉模型（静默模式，不需要用户交互）。

    从环境变量或已配置的 API Key 自动生成视觉配置。
    与已有的视觉模型配置合并，不覆盖已有的默认模型。
    返回默认模型配置，失败返回 None。
    """
    key_groups = _detect_api_keys()
    if not key_groups:
        return None

    data = copy.deepcopy(load_vision_json())
    existing_default = data.get("default", {})
    existing_fallback: dict = dict(data.get("fallback", {}))
    changed = False

    for api_key, presets in key_groups:
        # 已有相同 key 的相同提供商默认配置则跳过
        if (
            existing_default.get("base_url") == presets[0]["base_url"]
            and existing_default.get("api_key") == api_key
        ):
            continue

        for preset in presets:
            cfg = dict(preset)
            cfg["api_key"] = api_key
            model = cfg["model"]
            # 已在 fallback 或是当前默认 → 跳过
            if model in existing_fallback:
                continue
            if existing_default.get("model") == model and existing_default.get("api_key") == api_key:
                continue
            existing_fallback[model] = cfg
            changed = True

        # 没有默认视觉模型 → 用当前提供商的第一个预设设为默认
        if not (existing_default and existing_default.get("api_key")):
            existing_default = dict(presets[0])
            existing_default["api_key"] = api_key
            changed = True

    if not changed:
        return existing_default

    data["default"] = existing_default
    data["fallback"] = existing_fallback
    save_vision_json(data)
    return data["default"]


# 视觉相关字段白名单（与消费侧 tools.py 视觉请求所读字段对齐）
_VISION_FIELDS = ("model", "base_url", "api_key", "temperature", "top_p", "stream_usage")


def _vision_equal(existing: dict, vision_cfg: dict) -> bool:
    """现有视觉条目与新配置在白名单字段上是否一致（用于幂等判断）。"""
    if not isinstance(existing, dict):
        return False
    return all(existing.get(k) == vision_cfg.get(k) for k in _VISION_FIELDS)


def add_vision_model(config: dict) -> str | None:
    """把一个模型配置加入视觉模型列表；同名模型就地更新。

    - 同名已在 default → 更新 default
    - 同名已在 fallback → 更新该 fallback 条目
    - 新模型：有有效 default（api_key 非空）→ 加入 fallback；否则设为 default
    - 内容完全一致时跳过写入（幂等）；缺 model/api_key 返回 None。

    Returns: "default" | "fallback" 表示实际落入/更新的角色；None 表示未写入。
    """
    model = (config or {}).get("model", "")
    api_key = (config or {}).get("api_key", "")
    if not model or not api_key:
        return None

    # 只保留视觉相关字段，丢弃 extra_body/stop_sequences 等文本侧字段
    vision_cfg = {k: config[k] for k in _VISION_FIELDS if k in config}

    data = copy.deepcopy(load_vision_json())
    existing_default = data.get("default", {})
    existing_fallback: dict = dict(data.get("fallback", {}))

    # 同名已在 default → 就地更新
    if existing_default.get("model") == model:
        if _vision_equal(existing_default, vision_cfg):
            return None
        data["default"] = vision_cfg
        data["fallback"] = existing_fallback
        save_vision_json(data)
        return "default"

    # 同名已在 fallback → 就地更新
    if model in existing_fallback:
        if _vision_equal(existing_fallback[model], vision_cfg):
            return None
        existing_fallback[model] = vision_cfg
        data["default"] = existing_default
        data["fallback"] = existing_fallback
        save_vision_json(data)
        return "fallback"

    # 新模型：有有效 default → 加入 fallback
    if existing_default and existing_default.get("api_key"):
        existing_fallback[model] = vision_cfg
        data["default"] = existing_default
        data["fallback"] = existing_fallback
        save_vision_json(data)
        return "fallback"

    # 无有效 default → 设为 default（保留已存在的 fallback）
    data["default"] = vision_cfg
    data["fallback"] = existing_fallback
    save_vision_json(data)
    return "default"


async def configure_vision_interactive() -> dict | None:
    """交互式配置视觉模型（/vision 命令调用）"""
    ensure_config_dir()

    current = load_vision_json()
    current_default = current.get("default", {})
    has_config = bool(current_default and current_default.get("api_key"))

    back_label = t("common.back")
    if has_config:
        action = await select(
            t("vision.menu"),
            [t("vision.view"), t("vision.new_model"), t("vision.modelscope_quick"), t("vision.modelscope_quick_intl"), t("vision.switch"), back_label],
        )
    else:
        action = await select(
            t("vision.unconfigured_ask"),
            [t("vision.configure"), t("vision.configure_intl"), t("vision.new_model"), back_label],
        )

    if action is None or action == back_label:
        return None

    if action == t("vision.view"):
        _display_vision_config(current)
        return None

    if action == t("vision.switch"):
        return await _switch_vision_model()

    if action == t("vision.new_model"):
        return await _configure_vision_custom()

    if action == t("vision.modelscope_quick"):
        return await _configure_vision_modelscope()

    if action == t("vision.modelscope_quick_intl"):
        return await _configure_vision_modelscope(intl=True)

    if action == t("vision.configure_intl"):
        return await _configure_vision_wizard(intl=True)

    # 预设快捷配置
    return await _configure_vision_wizard()


async def _configure_vision_wizard(*, intl: bool = False) -> dict | None:
    """配置向导（intl=True 时走国际版端点 .ai，模型/参数与国内版一致）"""
    presets = VISION_MODEL_INTL_PRESETS if intl else VISION_MODEL_PRESETS

    # 选择 API Key 来源
    env_key = os.getenv("ModelScopeToken", "")
    manual_label = t("vision.manual_key")
    choices = []
    if env_key:
        choices.append(t("vision.use_env_token", masked=mask_api_key(env_key)))
    choices.append(manual_label)

    result = await select(t("vision.select_key_source"), choices)
    if result is None:
        return None

    if result == manual_label:
        api_key = await password(t("vision.input_key"))
        if not api_key or not api_key.strip():
            return None
        api_key = api_key.strip()
    else:
        api_key = env_key

    # 选择默认模型
    preset_names = [p["model"] for p in presets]
    default_choice = await select(t("vision.select_default"), preset_names, default=preset_names[0])
    if default_choice is None:
        return None

    default_idx = preset_names.index(default_choice)
    config = build_default_fallback_config(presets, api_key, default_index=default_idx)

    existing_data = load_vision_json()
    existing_fallback = existing_data.get("fallback", {})
    merged_fallback = {**existing_fallback, **config["fallback"]}
    config["fallback"] = merged_fallback
    save_vision_json(config)

    fallback = config["fallback"]
    console.print(f"[green]{t('vision.config_done', model=default_choice)}[/green]")
    fallback_names = ", ".join(fallback.keys())
    console.print(f"[dim]{t('vision.fallback_count', count=len(fallback), names=fallback_names)}[/dim]")

    return config["default"]


async def _switch_vision_model() -> dict | None:
    """切换视觉模型（从 fallback 列表选择）"""
    data = copy.deepcopy(load_vision_json())
    default = data.get("default", {})
    fallback = data.get("fallback", {})

    if not default:  # pragma: no cover
        console.print(f"[yellow]{t('vision.no_default')}[/yellow]")  # pragma: no cover
        return await _configure_vision_wizard()  # pragma: no cover

    if not fallback:  # pragma: no cover
        console.print(f"[yellow]{t('vision.no_fallback')}[/yellow]")  # pragma: no cover
        return None  # pragma: no cover

    current_name = default.get("model", "")
    tag = t("model.current_default_tag")
    choices = []
    for name in fallback:
        suffix = tag if name == current_name else ""
        choices.append(f"{name}{suffix}")

    result = await select(t("vision.select_to_use"), choices)
    if result is None:  # pragma: no cover
        return None  # pragma: no cover

    # 提取模型名（去掉翻译后的“当前默认”标记）
    selected_name = result.replace(tag, "")

    ok = await confirm(t("vision.switch_confirm", model=selected_name))
    if not ok:
        return None

    selected_config = fallback.pop(selected_name)
    if default:
        fallback[current_name] = default

    data["default"] = selected_config
    data["fallback"] = fallback
    save_vision_json(data)
    console.print(f"[green]{t('vision.switched', model=selected_name)}[/green]")
    return selected_config


def _display_vision_config(config: dict) -> None:
    """显示当前视觉模型配置"""
    from rich.table import Table

    default = config.get("default", {})
    fallback = config.get("fallback", {})

    if not default:
        console.print(f"[yellow]{t('vision.not_configured')}[/yellow]")
        return

    console.print(f"[bold]{t('vision.default_label')}[/bold] {default.get('model', t('vision.unknown'))}")

    if fallback:
        table = Table(title=t("vision.fallback_table_title"))
        table.add_column(t("vision.col_model"), style="cyan")
        table.add_column(t("vision.col_status"), style="green")
        for name in fallback:
            table.add_row(name, "✓")
        console.print(table)
    else:
        console.print(f"[dim]{t('vision.no_fallback_dim')}[/dim]")


async def _test_vision_connection(
    config: dict, *, quiet: bool = False, return_error: bool = False
) -> bool | str:
    """用一张 Pillow 生成的纯色小图 + "这是什么颜色"测试视觉模型连接。

    不报错即视为通过（不强制答对颜色，不检查响应内容）。
    return_error=True 时失败返回错误摘要字符串。
    """
    if not quiet:
        console.print(f"[yellow]{t('vision.testing')}[/yellow]")
    try:
        from PIL import Image
        from langchain_core.messages import HumanMessage

        from chcode.utils.enhanced_chat_openai import EnhancedChatOpenAI

        # 生成 8x8 纯红色 PNG
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        media_url = f"data:image/png;base64,{b64}"

        messages = [
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": media_url}},
                    {"type": "text", "text": "这是什么颜色"},
                ]
            )
        ]

        llm = EnhancedChatOpenAI(
            model=config["model"],
            base_url=config["base_url"],
            api_key=config["api_key"],
            max_tokens=64,
            max_retries=0,
            timeout=60,
        )
        await asyncio.to_thread(llm.invoke, messages)
        return True
    except Exception as e:
        # 与文本侧 _test_connection 一致：某些供应商连接成功但返回 null choices，
        # SDK 解析抛异常--视为连接通过（与请求是文本还是视觉无关，发生在响应解析阶段）
        err_msg = str(e)
        if "null value" in err_msg and "choices" in err_msg:
            return True
        if return_error:
            return err_msg
        return False


async def _configure_vision_custom() -> dict | None:
    """新建自定义视觉模型：收集 model/base_url/api_key -> 图片连接测试 -> add_vision_model。"""
    from chcode.config import _ask_conn_retry_action, _ask_context_length, _add_to_model_fallback
    from chcode.prompts import _ask_hyperparam, _SKIP, TEMPERATURE_PRESETS, TOP_P_PRESETS

    async def _collect() -> dict | None:
        """收集表单（model/base_url/api_key/可选超参），取消返回 None。"""
        model_name = await text(t("vision.custom_form_model"))
        if not model_name or not model_name.strip():
            return None
        model_name = model_name.strip()

        base_url = await text(t("vision.custom_form_url"))
        if not base_url or not base_url.strip():
            return None
        base_url = base_url.strip()

        api_key = await password(t("vision.custom_form_key"))
        if not api_key or not api_key.strip():
            return None
        api_key = api_key.strip()

        # 可选超参：默认与预设视觉模型对齐（1.0 / 0.95）。用户选择配置时用
        # 文本侧同一套 _ask_hyperparam 交互（预设列表 + 自定义输入 + 跳过）。
        temperature = 1.0
        top_p = 0.95
        if await confirm(t("form.configure_hyperparams"), default=False):
            result = await _ask_hyperparam(
                "Temperature:",
                TEMPERATURE_PRESETS,
                custom_prompt=t("form.input_temperature"),
            )
            if result is None:
                return None
            if result is not _SKIP:
                temperature = float(result)

            result = await _ask_hyperparam(
                "Top P:",
                TOP_P_PRESETS,
                custom_prompt=t("form.input_top_p"),
            )
            if result is None:
                return None
            if result is not _SKIP:
                top_p = float(result)

        return {
            "model": model_name,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": temperature,
            "top_p": top_p,
            "stream_usage": True,
        }

    config = await _collect()
    if config is None:
        return None

    # 测试连接：失败弹"重试/重新输入配置/放弃"。
    # retry 只重测；reinput 重新收集；与文本侧 configure_new_model 结构对齐（#6）。
    while True:
        result = await _test_vision_connection(config, return_error=True)
        if result is True:
            break
        err_summary = str(result).split("\n", 1)[0]
        # 复用文本侧的重试菜单（_ask_conn_retry_action 内部打印红字摘要 + 弹三选项）
        action = await _ask_conn_retry_action(err_summary)
        if action == t("connection.retry"):
            continue
        if action == t("connection.reinput"):
            config = await _collect()
            if config is None:
                return None
            continue
        return None  # 放弃 或 用户取消菜单


    role = add_vision_model(config)
    model_name = config["model"]
    if role == "default":
        console.print(f"[green]{t('vision.custom_added_default', model=model_name)}[/green]")
    elif role == "fallback":
        console.print(f"[green]{t('vision.custom_added_fallback', model=model_name)}[/green]")
    else:
        console.print(f"[yellow]{t('model.vision_duplicate')}[/yellow]")

    # 多模态模型同步到主模型备用列表（补问上下文长度，不替换当前默认）
    await _ask_context_length(config)
    _add_to_model_fallback(config)
    console.print(f"[dim]{t('vision.synced_to_main', model=model_name)}[/dim]")
    return config


async def _configure_vision_modelscope(*, intl: bool = False) -> dict | None:
    """魔搭快捷配置（视觉）：只需 API Key，把预设视觉模型补进 fallback。

    intl=True 时走国际版端点（https://api-inference.modelscope.ai/v1），
    预置模型与参数与国内版完全一致，仅 base_url 不同。

    只追加、不改现有 default（跳过与 default 同名的预设，避免被预设值覆盖）；
    不同步 model.json。

    连接测试照搬文本侧 _configure_modelscope_with_test 的策略：测前 3 个预设
    （代表模型），任一通过即认为 Key 有效，全失败弹"重试/重新输入/放弃"菜单。
    测试通过后才批量落盘，避免写入无法使用的配置。

    前提：调用方应保证已有 default（菜单仅在 has_config=True 时提供本入口）。
    无 default 直接调用时，所有预设进 fallback、default 保持为空，函数返回空 dict；
    不会自行设置 default--是否设默认由调用方决定。
    """
    from chcode.config import _ask_conn_retry_action

    presets = VISION_MODEL_INTL_PRESETS if intl else VISION_MODEL_PRESETS

    async def _collect_key() -> str | None:
        """收集 API Key（环境变量 ModelScopeToken / 手填），取消返回 None。"""
        env_key = os.getenv("ModelScopeToken", "")
        manual_label = t("vision.manual_key")
        choices = []
        if env_key:
            choices.append(t("vision.use_env_token", masked=mask_api_key(env_key)))
        choices.append(manual_label)

        result = await select(t("vision.select_key_source"), choices)
        if result is None:
            return None

        if result == manual_label:
            api_key = await password(t("vision.input_key"))
            if not api_key or not api_key.strip():
                return None
            return api_key.strip()
        return env_key

    api_key = await _collect_key()
    if api_key is None:
        return None

    # 测试连接（依次尝试前 3 个预设，应对速率限制/单模型下线），与文本侧
    # _configure_modelscope_with_test 的"default + 2 备用"策略对齐。
    # 视觉侧无用户 default，故取预设列表前 3 个作代表。
    while True:
        console.print(f"[yellow]{t('vision.testing')}[/yellow]")
        test_presets = presets[:3]
        connected = False
        last_err_summary = ""
        for preset in test_presets:
            result = await _test_vision_connection(
                {**preset, "api_key": api_key}, quiet=True, return_error=True
            )
            if result is True:
                connected = True
                break
            last_err_summary = str(result).split("\n", 1)[0]

        if connected:
            break

        # 全部代表模型都失败 -> 弹菜单
        action = await _ask_conn_retry_action(last_err_summary)
        if action == t("connection.retry"):
            continue
        if action == t("connection.reinput"):
            api_key = await _collect_key()
            if api_key is None:
                return None
            continue
        return None  # 放弃 或 用户取消菜单

    # 测试通过 -> 批量把预设补进 fallback：一次性 load + 内存合并 + 单次 save
    # （避免逐个 add_vision_model 触发 N 次磁盘读写，且保证写入原子性）。
    # 只追加、不改现有 default；跳过与 default 同名的预设，避免被预设值覆盖。
    data = copy.deepcopy(load_vision_json())
    default = data.get("default") or {}
    fallback = dict(data.get("fallback") or {})
    default_model = default.get("model", "")
    for preset in presets:
        if preset["model"] == default_model:
            continue
        # 与 add_vision_model 一致：只保留视觉白名单字段 + 注入 api_key
        cfg = {k: preset[k] for k in _VISION_FIELDS if k in preset}
        cfg["api_key"] = api_key
        fallback[preset["model"]] = cfg

    data["default"] = default
    data["fallback"] = fallback
    save_vision_json(data)

    # 仅报告 fallback（config_done 是"设默认"语义，本流程不动 default，不适用）
    fallback_names = ", ".join(fallback.keys())
    console.print(f"[dim]{t('vision.fallback_count', count=len(fallback), names=fallback_names)}[/dim]")
    return default
