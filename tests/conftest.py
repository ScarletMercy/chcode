import pytest


@pytest.fixture(autouse=True)
def reset_global_state():
    yield
    try:
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
