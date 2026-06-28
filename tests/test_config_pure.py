from chcode.config import ENV_TO_CONFIG, detect_env_api_keys


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


class TestPredefinedContextLength:
    """删表后,预定义模型预设必须各自携带 metadata.context_length。"""

    def test_modelscope_presets_carry_context_length(self):
        from chcode.prompts import MODELSCOPE_PRESETS

        assert MODELSCOPE_PRESETS
        for p in MODELSCOPE_PRESETS:
            assert p["metadata"]["context_length"] > 0, p["model"]

    def test_inner_model_config_carries_context_length(self):
        from chcode.agent_setup import INNER_MODEL_CONFIG

        assert INNER_MODEL_CONFIG["metadata"]["context_length"] > 0

