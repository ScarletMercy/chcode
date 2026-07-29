import pytest


@pytest.fixture(autouse=True)
def _isolate_config_files(tmp_path_factory, monkeypatch):
    """强制把所有配置文件路径重定向到独立临时目录，防止测试污染真实的 ~/.chat/。

    function 级：每个测试拿到独立的临时目录，配置状态不跨测试泄漏。
    测试内自己的 monkeypatch 会覆盖这里的指向。
    """
    isolated_root = tmp_path_factory.mktemp("isolated_config")
    # 重定向配置文件路径到独立临时目录，防止测试污染真实的 ~/.chat/。
    # 不吞异常：import/setattr/invalidate 任何一步失败都意味着隔离失效，
    # 应当显式报错而非静默--否则测试会写到真实 ~/.chat/ 还一声不吭。
    import chcode.config as config_mod
    monkeypatch.setattr(config_mod, "CONFIG_DIR", isolated_root)
    monkeypatch.setattr(config_mod, "MODEL_JSON", isolated_root / "model.json")
    monkeypatch.setattr(config_mod, "SETTING_JSON", isolated_root / "chagent.json")
    config_mod._model_json.invalidate()

    import chcode.vision_config as vision_mod
    monkeypatch.setattr(vision_mod, "CONFIG_DIR", isolated_root)
    monkeypatch.setattr(vision_mod, "VISION_JSON", isolated_root / "vision_model.json")
    vision_mod._vision_json.invalidate()
    yield


@pytest.fixture(autouse=True)
def reset_global_state():
    # 每个测试前确保 UI 语言为默认中文（避免上一个测试/ locale 检测污染全局 _lang）
    try:
        from chcode.i18n import set_language
        set_language("zh")
    except Exception:
        pass
    yield
    try:
        from chcode.i18n import set_language
        set_language("zh")

        from chcode.agent_setup import set_fallback_models
        set_fallback_models([])

        import chcode.config as config_mod
        config_mod._model_json.invalidate()

        import chcode.utils.tools as tools_mod
        tools_mod._tavily_api_key = ""
        tools_mod._tavily_key_loaded = False
        tools_mod._tavily_client = None
    except Exception:
        pass
