"""Extended tests for chcode/skill_manager.py"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chcode.utils.skill_manager import (
    manage_skills,
    _list_skills,
    _show_skill_detail,
    _delete_skill,
    _install_skill,
)


@pytest.fixture
def mock_session(tmp_path):
    s = MagicMock()
    s.workplace_path = tmp_path / "workplace"
    s.workplace_path.mkdir(parents=True, exist_ok=True)
    return s


class TestManageSkills:
    async def test_return_early(self, mock_session):
        with patch("chcode.utils.skill_manager.select", new_callable=AsyncMock, return_value="返回"):
            result = await manage_skills(mock_session)
            assert result is None  # manage_skills returns None when returning early

    async def test_view_skills_branch(self, mock_session):
        with patch("chcode.utils.skill_manager.select", new_callable=AsyncMock, side_effect=["查看已安装技能", "返回"]), \
             patch("chcode.utils.skill_manager._list_skills", new_callable=AsyncMock) as mock_list:
            await manage_skills(mock_session)
            assert mock_list.called  # _list_skills should be called

    async def test_install_skill_branch(self, mock_session):
        with patch("chcode.utils.skill_manager.select", new_callable=AsyncMock, side_effect=["安装新技能", "返回"]), \
             patch("chcode.utils.skill_manager._install_skill", new_callable=AsyncMock) as mock_install:
            await manage_skills(mock_session)
            assert mock_install.called  # _install_skill should be called

    async def test_none_returns(self, mock_session):
        with patch("chcode.utils.skill_manager.select", new_callable=AsyncMock, return_value=None):
            result = await manage_skills(mock_session)
            assert result is None  # manage_skills returns None when select returns None


class TestListSkills:
    async def test_empty_skills(self, mock_session):
        with patch("chcode.utils.skill_manager.scan_all_skills", return_value=[]):
            result = await _list_skills(mock_session)
            assert result is None  # _list_skills returns None when no skills

    async def test_skills_with_operations(self, mock_session):
        skills = [{"name": "s1", "type": "project", "description": "desc", "path": "/p"}]
        with patch("chcode.utils.skill_manager.scan_all_skills", return_value=skills), \
             patch("chcode.utils.skill_manager.select", new_callable=AsyncMock, return_value="返回") as mock_sel:
            await _list_skills(mock_session)
            assert mock_sel.called  # select should be called when skills exist

    async def test_view_detail(self, mock_session, tmp_path):
        skill_dir = tmp_path / "skill1"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill\nInstructions here.", encoding="utf-8")
        skills = [{"name": "s1", "type": "project", "description": "desc", "path": str(skill_dir)}]
        with patch("chcode.utils.skill_manager.scan_all_skills", return_value=skills), \
             patch("chcode.utils.skill_manager.select", new_callable=AsyncMock, side_effect=["s1 (project)", "查看详情"]) as mock_sel, \
             patch("chcode.utils.skill_manager._show_skill_detail", new_callable=AsyncMock) as mock_show:
            await _list_skills(mock_session)
            assert mock_show.called  # _show_skill_detail should be called


class TestShowSkillDetail:
    async def test_file_not_exists_manager(self, tmp_path):
        skill = {"name": "missing", "path": str(tmp_path / "nope")}
        result = await _show_skill_detail(skill)
        assert result is None  # Should return None when file doesn't exist

    async def test_file_exists(self, tmp_path):
        d = tmp_path / "skill"
        d.mkdir()
        (d / "SKILL.md").write_text("# Test\nBody", encoding="utf-8")
        skill = {"name": "test", "path": str(d)}
        # _show_skill_detail prints output but returns None
        # Just verify it runs without exception
        await _show_skill_detail(skill)
        assert (d / "SKILL.md").exists()  # File should still exist


class TestDeleteSkill:
    async def test_user_cancels(self, tmp_path):
        skill = {"name": "s1", "path": str(tmp_path)}
        with patch("chcode.utils.skill_manager.confirm", new_callable=AsyncMock, return_value=False) as mock_confirm:
            await _delete_skill(skill, MagicMock())
            mock_confirm.assert_called_once()

    async def test_success_manager(self, tmp_path):
        d = tmp_path / "skill"
        d.mkdir()
        skill = {"name": "s1", "path": str(d)}
        with patch("chcode.utils.skill_manager.confirm", new_callable=AsyncMock, return_value=True):
            await _delete_skill(skill, MagicMock())
        assert not d.exists()

    async def test_failure(self, tmp_path):
        skill = {"name": "s1", "path": str(tmp_path / "nope")}
        with patch("chcode.utils.skill_manager.confirm", new_callable=AsyncMock, return_value=True) as mock_confirm:
            await _delete_skill(skill, MagicMock())
            mock_confirm.assert_called_once()


class TestInstallSkill:
    async def test_empty_path(self, mock_session):
        with patch("chcode.utils.skill_manager.text", new_callable=AsyncMock, return_value=""):
            result = await _install_skill(mock_session)
            assert result is None  # Should return None when path is empty

    async def test_file_not_exists_for_install(self, mock_session):
        with patch("chcode.utils.skill_manager.text", new_callable=AsyncMock, return_value="/nonexistent.zip"):
            result = await _install_skill(mock_session)
            assert result is None  # Should return None when file doesn't exist

    async def test_invalid_package_manager(self, mock_session, tmp_path):
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_bytes(b"not a real zip")
        with patch("chcode.utils.skill_manager.text", new_callable=AsyncMock, return_value=str(bad_zip)), \
             patch("chcode.utils.skill_manager.validate_skill_package", return_value=None):
            result = await _install_skill(mock_session)
            assert result is None  # Should return None for invalid package

    async def test_valid_install(self, mock_session, tmp_path):
        import zipfile
        zip_path = tmp_path / "good.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: test\ndescription: d\n---\nInstructions")
        with patch("chcode.utils.skill_manager.text", new_callable=AsyncMock, return_value=str(zip_path)), \
             patch("chcode.utils.skill_manager.validate_skill_package", return_value={"name": "test"}), \
             patch("chcode.utils.skill_manager.select", new_callable=AsyncMock, return_value="项目级 (当前工作目录)"), \
             patch("chcode.utils.skill_manager.install_skill", return_value=True) as mock_install:
            await _install_skill(mock_session)
            assert mock_install.called  # install_skill should be called


class TestListSkillsOperationBranches:
    """Cover lines 79-88: operation select branches."""

    async def test_view_detail_branch(self, mock_session, tmp_path):
        """Cover lines 83-84: select returns '查看详情'."""
        skill_dir = tmp_path / "skill1"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill\nInstructions here.", encoding="utf-8")
        skills = [{"name": "s1", "type": "project", "description": "desc", "path": str(skill_dir)}]

        with patch("chcode.utils.skill_manager.scan_all_skills", return_value=skills), \
             patch("chcode.utils.skill_manager.select", new_callable=AsyncMock, side_effect=["s1 (project)", "查看详情"]) as mock_sel, \
             patch("chcode.utils.skill_manager._show_skill_detail", new_callable=AsyncMock) as mock_show:
            await _list_skills(mock_session)
            assert mock_show.called  # _show_skill_detail should be called

    async def test_delete_branch(self, mock_session, tmp_path):
        """Cover lines 85-86: select returns '删除技能'."""
        skill_dir = tmp_path / "skill1"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill\nInstructions here.", encoding="utf-8")
        skills = [{"name": "s1", "type": "project", "description": "desc", "path": str(skill_dir)}]

        with patch("chcode.utils.skill_manager.scan_all_skills", return_value=skills), \
             patch("chcode.utils.skill_manager.select", new_callable=AsyncMock, side_effect=["s1 (project)", "删除技能"]) as mock_sel, \
             patch("chcode.utils.skill_manager._delete_skill", new_callable=AsyncMock) as mock_delete:
            await _list_skills(mock_session)
            assert mock_delete.called  # _delete_skill should be called

    async def test_return_from_operations(self, mock_session, tmp_path):
        """Cover lines 87-88: select returns '返回' from operations."""
        skill_dir = tmp_path / "skill1"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill\nInstructions here.", encoding="utf-8")
        skills = [{"name": "s1", "type": "project", "description": "desc", "path": str(skill_dir)}]

        with patch("chcode.utils.skill_manager.scan_all_skills", return_value=skills), \
             patch("chcode.utils.skill_manager.select", new_callable=AsyncMock, side_effect=["s1 (project)", "返回"]) as mock_sel:
            result = await _list_skills(mock_session)
            assert result is None  # Should return None when returning from operations


class TestManageSkillsBranches:
    """Cover lines 30-41: main menu branches."""

    async def test_view_installed_skills(self, mock_session):
        """Cover lines 38-39: '查看已安装技能' branch."""
        with patch("chcode.utils.skill_manager.select", new_callable=AsyncMock, side_effect=["查看已安装技能", "返回"]), \
             patch("chcode.utils.skill_manager._list_skills", new_callable=AsyncMock) as mock_list:
            await manage_skills(mock_session)
            assert mock_list.called  # _list_skills should be called

    async def test_install_new_skill(self, mock_session):
        """Cover lines 40-41: '安装新技能' branch."""
        with patch("chcode.utils.skill_manager.select", new_callable=AsyncMock, side_effect=["安装新技能", "返回"]), \
             patch("chcode.utils.skill_manager._install_skill", new_callable=AsyncMock) as mock_install:
            await manage_skills(mock_session)
            assert mock_install.called  # _install_skill should be called


class TestShowSkillDetailFileNotExists:
    """Cover lines 94-96: SKILL.md file doesn't exist."""

    async def test_skill_md_not_exists(self, tmp_path):
        """Cover lines 94-96: SKILL.md file doesn't exist."""
        skill = {"name": "missing", "path": str(tmp_path / "nope")}
        result = await _show_skill_detail(skill)
        assert result is None  # Should return None when SKILL.md doesn't exist


class TestDeleteSkillException:
    """Cover lines 123-124: exception during deletion."""

    async def test_delete_raises_exception(self, tmp_path):
        """Cover lines 123-124: shutil.rmtree raises exception."""
        d = tmp_path / "skill"
        d.mkdir()
        skill = {"name": "s1", "path": str(d)}

        with patch("chcode.utils.skill_manager.confirm", new_callable=AsyncMock, return_value=True) as mock_confirm, \
             patch("shutil.rmtree", side_effect=PermissionError("Access denied")):
            # Should handle PermissionError gracefully
            await _delete_skill(skill, MagicMock())
            # Verify confirm was called and directory still exists (rmtree failed)
            mock_confirm.assert_called_once()
            assert d.exists(), "Directory should still exist because rmtree failed"


class TestInstallSkillGlobalLocation:
    """Cover line 156: global install location."""

    async def test_install_global_location(self, mock_session, tmp_path):
        """Cover line 156: select '全局级' install location."""
        import zipfile
        zip_path = tmp_path / "good.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: test\ndescription: d\n---\nInstructions")

        with patch("chcode.utils.skill_manager.text", new_callable=AsyncMock, return_value=str(zip_path)), \
             patch("chcode.utils.skill_manager.validate_skill_package", return_value={"name": "test"}), \
             patch("chcode.utils.skill_manager.select", new_callable=AsyncMock, return_value="全局级 (用户目录)"), \
             patch("chcode.utils.skill_manager.install_skill", return_value=True) as mock_install:
            await _install_skill(mock_session)
            assert mock_install.called  # install_skill should be called


class TestInstallSkillInvalidPackage:
    """Cover line 142-143: invalid skill package."""

    async def test_invalid_package_no_skill_md(self, mock_session, tmp_path):
        """Cover lines 142-143: validate_skill_package returns None."""
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_bytes(b"not a real zip")

        with patch("chcode.utils.skill_manager.text", new_callable=AsyncMock, return_value=str(bad_zip)), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("chcode.utils.skill_manager.validate_skill_package", return_value=None):
            result = await _install_skill(mock_session)
            assert result is None  # Should return None for invalid package
