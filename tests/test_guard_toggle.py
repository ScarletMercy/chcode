from unittest.mock import patch

import pytest

from chcode.utils.shell.guard import (
    check_command,
    get_disabled_categories,
    is_category_enabled,
    is_guard_enabled,
    set_category_enabled,
    set_guard_enabled,
)


@pytest.fixture(autouse=True)
def _mock_persistence():
    """拦截对 chagent.json 的真实读写，避免测试污染用户配置。"""
    with (
        patch("chcode.config._load_setting", return_value={}),
        patch("chcode.config._update_setting"),
    ):
        set_guard_enabled(True)
        set_category_enabled("recursive_delete", True)
        set_category_enabled("force_kill", True)
        set_category_enabled("system_damage", True)
        set_category_enabled("shutdown", True)
        yield
        set_guard_enabled(True)
        set_category_enabled("recursive_delete", True)
        set_category_enabled("force_kill", True)
        set_category_enabled("system_damage", True)
        set_category_enabled("shutdown", True)


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
            mock_update.assert_called_once_with(guard_enabled=False, disabled_categories=[])
            set_guard_enabled(True)
            assert mock_update.call_args.kwargs["guard_enabled"] is True

    def test_load_persisted_state_applies_saved_value(self):
        # 模拟 chagent.json 中已保存 guard_enabled=False
        with patch("chcode.config._load_setting", return_value={"guard_enabled": False}):
            from chcode.utils.shell.guard import _load_persisted_state

            _load_persisted_state()
            assert is_guard_enabled() is False

    def test_load_persisted_state_ignores_non_bool(self):
        # 非 bool 值应被忽略，保持默认 True
        with patch("chcode.config._load_setting", return_value={"guard_enabled": "yes"}):
            from chcode.utils.shell.guard import _load_persisted_state

            _load_persisted_state()
            assert is_guard_enabled() is True

    def test_load_persisted_state_swallows_errors(self):
        # 读盘异常不应抛错，保持默认值
        with patch("chcode.config._load_setting", side_effect=Exception("disk error")):
            from chcode.utils.shell.guard import _load_persisted_state

            _load_persisted_state()
            assert is_guard_enabled() is True

    def test_ensure_guard_config_written_writes_default_when_missing(self):
        # chagent.json 中无字段 → 写入默认值（guard_enabled=True, disabled_categories=[]）
        with (
            patch("chcode.config._load_setting", return_value={}),
            patch("chcode.config._update_setting") as mock_update,
        ):
            from chcode.utils.shell.guard import ensure_guard_config_written

            ensure_guard_config_written()
            mock_update.assert_called_once_with(guard_enabled=True, disabled_categories=[])

    def test_ensure_guard_config_written_does_not_overwrite_existing(self):
        # chagent.json 中两个字段都已存在 → 不写入
        with (
            patch(
                "chcode.config._load_setting",
                return_value={"guard_enabled": False, "disabled_categories": ["force_kill"]},
            ),
            patch("chcode.config._update_setting") as mock_update,
        ):
            from chcode.utils.shell.guard import ensure_guard_config_written

            ensure_guard_config_written()
            mock_update.assert_not_called()

    def test_ensure_guard_config_written_fills_missing_disabled_categories(self):
        # guard_enabled 已存在但 disabled_categories 缺失 → 只补写后者
        with (
            patch(
                "chcode.config._load_setting",
                return_value={"guard_enabled": False},
            ),
            patch("chcode.config._update_setting") as mock_update,
        ):
            from chcode.utils.shell.guard import ensure_guard_config_written

            ensure_guard_config_written()
            mock_update.assert_called_once_with(disabled_categories=[])

    def test_ensure_guard_config_written_swallows_errors(self):
        with (
            patch("chcode.config._load_setting", side_effect=Exception("disk error")),
            patch("chcode.config._update_setting") as mock_update,
        ):
            from chcode.utils.shell.guard import ensure_guard_config_written

            ensure_guard_config_written()
            mock_update.assert_not_called()


class TestCategoryToggle:
    """按类别控制拦截"""

    def test_disable_recursive_delete_category(self):
        set_category_enabled("recursive_delete", False)
        assert is_category_enabled("recursive_delete") is False
        # rm -rf 不再被拦截
        assert check_command("rm -rf /").blocked is False
        # 其他类别仍拦截
        assert check_command("shutdown -h now").blocked is True
        assert check_command("kill -9 1234").blocked is True

    def test_disable_force_kill_category(self):
        set_category_enabled("force_kill", False)
        assert check_command("kill -9 1234").blocked is False
        assert check_command("taskkill /f /im app.exe").blocked is False
        # 其他类别不受影响
        assert check_command("rm -rf /").blocked is True

    def test_disable_system_damage_category(self):
        set_category_enabled("system_damage", False)
        assert check_command("mkfs.ext4 /dev/sda1").blocked is False
        assert check_command("dd if=/dev/zero of=/dev/sda").blocked is False
        # 其他类别仍拦截
        assert check_command("reboot").blocked is True

    def test_disable_shutdown_category(self):
        set_category_enabled("shutdown", False)
        assert check_command("shutdown -h now").blocked is False
        assert check_command("reboot").blocked is False
        assert check_command("init 0").blocked is False
        # 其他类别仍拦截
        assert check_command("rm -rf /").blocked is True

    def test_multiple_disabled_categories(self):
        set_category_enabled("recursive_delete", False)
        set_category_enabled("force_kill", False)
        assert check_command("rm -rf /").blocked is False
        assert check_command("kill -9 1234").blocked is False
        assert check_command("shutdown -h now").blocked is True

    def test_reenable_category(self):
        set_category_enabled("recursive_delete", False)
        assert check_command("rm -rf /").blocked is False
        set_category_enabled("recursive_delete", True)
        assert check_command("rm -rf /").blocked is True

    def test_get_disabled_categories(self):
        set_category_enabled("recursive_delete", False)
        set_category_enabled("system_damage", False)
        disabled = get_disabled_categories()
        assert disabled == {"recursive_delete", "system_damage"}

    def test_master_switch_overrides_categories(self):
        # 总开关关闭时，即使类别启用也不拦截
        set_guard_enabled(False)
        assert check_command("rm -rf /").blocked is False
        assert check_command("shutdown -h now").blocked is False

    def test_set_category_enabled_persists(self):
        with patch("chcode.config._update_setting") as mock_update:
            set_category_enabled("recursive_delete", False)
            mock_update.assert_called_once()
            kwargs = mock_update.call_args.kwargs
            assert kwargs["disabled_categories"] == ["recursive_delete"]

    def test_set_category_enabled_ignores_invalid_category(self):
        set_category_enabled("nonexistent", False)
        assert "nonexistent" not in get_disabled_categories()

    def test_load_persisted_disabled_categories(self):
        with patch(
            "chcode.config._load_setting",
            return_value={"disabled_categories": ["force_kill", "system_damage"]},
        ):
            from chcode.utils.shell.guard import _load_persisted_state

            _load_persisted_state()
            assert is_category_enabled("force_kill") is False
            assert is_category_enabled("system_damage") is False
            assert is_category_enabled("recursive_delete") is True
            assert is_category_enabled("shutdown") is True
