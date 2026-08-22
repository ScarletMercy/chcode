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
        expected = {
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "MINIMAX_TOKEN_PLAN_KEY",
            "KIMI_API_KEY",
        }
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

    def test_modelscope_intl_presets_reuse_models_and_params(self):
        """国际版预设：参数与国内版完全一致，base_url 的 cn → ai；GLM 模型的 namespace 从
        ZhipuAI（国内版）换成 zai-org（国际版），其余模型 id 不变。"""
        from chcode.prompts import (
            MODELSCOPE_BASE_URL,
            MODELSCOPE_INTL_BASE_URL,
            MODELSCOPE_INTL_PRESETS,
            MODELSCOPE_PRESETS,
        )

        assert len(MODELSCOPE_INTL_PRESETS) == len(MODELSCOPE_PRESETS)
        for cn, intl in zip(MODELSCOPE_PRESETS, MODELSCOPE_INTL_PRESETS):
            assert intl["base_url"] == "https://api-inference.modelscope.ai/v1"
            # GLM 在两版有不同 namespace（ZhipuAI/zai-org），其余模型 id 必须一致
            expected_model = (
                cn["model"].replace("ZhipuAI/", "zai-org/")
                if cn["model"].startswith("ZhipuAI/")
                else cn["model"]
            )
            assert intl["model"] == expected_model
            assert intl["temperature"] == cn["temperature"]
            assert intl["top_p"] == cn["top_p"]
            # 国际版 metadata 是国内版的超集：继承 context_length 等，并打 region="intl"
            # 标记，用于在 fallback 列表中与同名国内版模型区分（显示 (国际版) 后缀）。
            assert intl["metadata"] == {**cn["metadata"], "region": "intl"}
        assert MODELSCOPE_INTL_BASE_URL != MODELSCOPE_BASE_URL


class TestRegionKey:
    """region_key 决定 cfg 在 fallback 字典中的 key，区分国内版/国际版同名模型。"""

    def test_cn_returns_plain_model_name(self):
        from chcode.utils.json_utils import region_key

        cn = {"model": "ZhipuAI/GLM-5.2", "metadata": {"context_length": 1048576}}
        assert region_key(cn) == "ZhipuAI/GLM-5.2"

    def test_intl_appends_suffix(self):
        from chcode.utils.json_utils import region_key

        intl = {
            "model": "zai-org/GLM-5.2",
            "metadata": {"context_length": 1048576, "region": "intl"},
        }
        assert region_key(intl) == "zai-org/GLM-5.2 (国际版)"

    def test_missing_metadata_returns_plain_name(self):
        from chcode.utils.json_utils import region_key

        assert region_key({"model": "X"}) == "X"

    def test_empty_cfg_returns_empty(self):
        from chcode.utils.json_utils import region_key

        assert region_key({}) == ""

    def test_non_intl_region_treated_as_cn(self):
        """region 非 'intl'（含缺失）一律按国内版处理，不加后缀。"""
        from chcode.utils.json_utils import region_key

        assert region_key({"model": "M", "metadata": {"region": "cn"}}) == "M"
        assert region_key({"model": "M", "metadata": {"region": ""}}) == "M"

    def test_build_default_fallback_separates_same_name_regions(self):
        """同名国内/国际预设共存时，fallback key 必须分离，互不覆盖。"""
        from chcode.utils.json_utils import build_default_fallback_config

        presets = [
            {"model": "M", "metadata": {"context_length": 100}},
            {"model": "M", "metadata": {"context_length": 200, "region": "intl"}},
        ]
        res = build_default_fallback_config(presets, "k", default_index=0)
        # cn 版 "M" 作为 default 完整保留，未被 intl 同名覆盖
        assert res["default"]["model"] == "M"
        assert res["default"]["metadata"] == {"context_length": 100}  # cn 版，无 region
        assert set(res["fallback"].keys()) == {"M (国际版)"}
        # 国际版条目完整保留
        intl_cfg = res["fallback"]["M (国际版)"]
        assert intl_cfg["model"] == "M"
        assert intl_cfg["metadata"]["region"] == "intl"

    def test_inner_model_config_carries_context_length(self):
        from chcode.agent_setup import INNER_MODEL_CONFIG

        assert INNER_MODEL_CONFIG["metadata"]["context_length"] > 0
