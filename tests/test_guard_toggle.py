from unittest.mock import patch

import pytest

from chcode.utils.shell.guard import (
    check_command,
    is_guard_enabled,
    set_guard_enabled,
)


@pytest.fixture(autouse=True)
def _mock_persistence():
    """拦截对 chagent.json 的真实读写，避免测试污染用户配置。"""
    with (
        patch("chcode.config._load_setting", return_value={}),
        patch("chcode.config._update_setting"),
    ):
        # 模块级 _load_persisted_flag 已在本 fixture 之前跑过（导入时），
        # 这里再强制复位到默认 True，保证用例起点一致。
        set_guard_enabled(True)
        # is_guard_enabled 读的是内存变量，但 set_guard_enabled 会触发 mock 的写盘——无害。
        yield
        set_guard_enabled(True)


class TestGuardToggle:
    """危险命令拦截总开关（/danger）"""

    def test_default_enabled(self):
        # 加载 mock 后默认值仍为 True（_load_setting 返回空 dict → 保持默认）
        assert is_guard_enabled() is True

    def test_toggle_off_then_blocked_command_passes(self):
        set_guard_enabled(False)
        assert is_guard_enabled() is False
        # 关闭后危险命令一律放行
        assert check_command("rm -rf /").blocked is False
        assert check_command("shutdown -h now").blocked is False
        assert check_command("kill -9 1234").blocked is False

    def test_toggle_on_then_command_blocked_again(self):
        set_guard_enabled(False)
        assert check_command("rm -rf /").blocked is False
        set_guard_enabled(True)
        assert is_guard_enabled() is True
        assert check_command("rm -rf /").blocked is True

    def test_toggle_off_safe_commands_unaffected(self):
        set_guard_enabled(False)
        # 安全命令在开关关闭时本来就不拦，行为不变
        assert check_command("ls -la").blocked is False
        assert check_command("echo hello").blocked is False

    def test_repeated_toggle(self):
        set_guard_enabled(False)
        set_guard_enabled(True)
        set_guard_enabled(False)
        assert is_guard_enabled() is False
        assert check_command("reboot").blocked is False


class TestGuardPersistence:
    """持久化行为（mock chagent.json）"""

    def test_set_guard_enabled_writes_to_settings(self):
        with patch("chcode.config._update_setting") as mock_update:
            set_guard_enabled(False)
            mock_update.assert_called_once_with(guard_enabled=False)
            set_guard_enabled(True)
            assert mock_update.call_args.kwargs == {"guard_enabled": True}

    def test_load_persisted_flag_applies_saved_value(self):
        # 模拟 chagent.json 中已保存 guard_enabled=False
        with patch("chcode.config._load_setting", return_value={"guard_enabled": False}):
            from chcode.utils.shell.guard import _load_persisted_flag

            _load_persisted_flag()
            assert is_guard_enabled() is False

    def test_load_persisted_flag_ignores_non_bool(self):
        # 非 bool 值应被忽略，保持默认 True
        with patch("chcode.config._load_setting", return_value={"guard_enabled": "yes"}):
            from chcode.utils.shell.guard import _load_persisted_flag

            _load_persisted_flag()
            assert is_guard_enabled() is True

    def test_load_persisted_flag_swallows_errors(self):
        # 读盘异常不应抛错，保持默认值
        with patch("chcode.config._load_setting", side_effect=Exception("disk error")):
            from chcode.utils.shell.guard import _load_persisted_flag

            _load_persisted_flag()
            assert is_guard_enabled() is True

    def test_ensure_guard_config_written_writes_default_when_missing(self):
        # chagent.json 中无 guard_enabled → 写入默认 True
        with (
            patch("chcode.config._load_setting", return_value={}),
            patch("chcode.config._update_setting") as mock_update,
        ):
            from chcode.utils.shell.guard import ensure_guard_config_written

            ensure_guard_config_written()
            mock_update.assert_called_once_with(guard_enabled=True)

    def test_ensure_guard_config_written_does_not_overwrite_existing(self):
        # chagent.json 中已有 guard_enabled=False → 不覆盖
        with (
            patch("chcode.config._load_setting", return_value={"guard_enabled": False}),
            patch("chcode.config._update_setting") as mock_update,
        ):
            from chcode.utils.shell.guard import ensure_guard_config_written

            ensure_guard_config_written()
            mock_update.assert_not_called()

    def test_ensure_guard_config_written_swallows_errors(self):
        with (
            patch("chcode.config._load_setting", side_effect=Exception("disk error")),
            patch("chcode.config._update_setting") as mock_update,
        ):
            from chcode.utils.shell.guard import ensure_guard_config_written

            ensure_guard_config_written()
            mock_update.assert_not_called()
