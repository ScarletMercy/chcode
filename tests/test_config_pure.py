from chcode.config import (
    CONTEXT_WINDOW_SIZES,
    ENV_TO_CONFIG,
    get_context_window_size,
    detect_env_api_keys,
    _DEFAULT_CONTEXT_WINDOW,
)


class TestGetContextWindowSize:
    def test_exact_match(self):
        assert get_context_window_size("gpt-4o") == 128000

    def test_prefix_match(self):
        assert get_context_window_size("org/gpt-4o") == 128000

    def test_substring_match(self):
        assert get_context_window_size("my-deepseek-chat-v2") == 65536

    def test_no_match(self):
        assert get_context_window_size("unknown-model") == _DEFAULT_CONTEXT_WINDOW

    def test_empty_config_pure(self):
        assert get_context_window_size("") == _DEFAULT_CONTEXT_WINDOW


class TestDetectEnvApiKeys:
    def test_no_keys(self, monkeypatch):
        for var in ENV_TO_CONFIG:
            monkeypatch.delenv(var, raising=False)
        result = detect_env_api_keys()
        assert result == []

    def test_with_key(self, monkeypatch):
        for var in ENV_TO_CONFIG:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = detect_env_api_keys()
        assert len(result) == 1
        assert result[0]["name"] == "OpenAI"
        assert result[0]["api_key"] == "sk-test"

    def test_multiple_keys(self, monkeypatch):
        for var in ENV_TO_CONFIG:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-2")
        result = detect_env_api_keys()
        assert len(result) == 2


class TestEnvToConfig:
    def test_has_known_providers(self):
        expected = {"OPENAI_API_KEY", "DEEPSEEK_API_KEY", "MINIMAX_TOKEN_PLAN_KEY", "KIMI_API_KEY"}
        assert expected.issubset(set(ENV_TO_CONFIG.keys()))

    def test_each_entry_has_required_fields(self):
        for var, cfg in ENV_TO_CONFIG.items():
            assert "name" in cfg
            assert "base_url" in cfg
            assert "models" in cfg

