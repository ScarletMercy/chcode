import pytest


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
