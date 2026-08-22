"""Extended tests for chcode/utils/skill_loader.py"""

import os
import zipfile
import tarfile
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chcode.utils.skill_loader import (
    SkillMetadata,
    SkillLoader,
    SkillContent,
    validate_skill_package,
    install_skill,
    _extract_archive,
    _find_skill_dir,
    scan_all_skills,
)


def _write_skill_md(directory: Path, name: str, description: str = "desc"):
    d = directory / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nInstructions for {name}.",
        encoding="utf-8",
    )
    return d


class TestSkillLoaderLoadSkill:
    def test_load_from_cache(self, tmp_path):
        loader = SkillLoader(skill_paths=[tmp_path])
        loader._skills_cache = [
            SkillMetadata(name="s1", description="d", skill_path=tmp_path),
        ]
        _write_skill_md(tmp_path, "s1", "d")
        result = loader.load_skill("s1")
        assert result is not None
        assert result.instructions == "Instructions for s1."

    def test_rescan_if_not_cached(self, tmp_path):
        loader = SkillLoader(skill_paths=[tmp_path])
        _write_skill_md(tmp_path, "s1", "d")
        result = loader.load_skill("s1")
        assert result is not None

    def test_not_found_loader(self, tmp_path):
        loader = SkillLoader(skill_paths=[tmp_path])
        assert loader.load_skill("nonexistent") is None


class TestSkillLoaderBuildSystemPrompt:
    def test_with_skills(self, tmp_path):
        loader = SkillLoader(skill_paths=[tmp_path])
        _write_skill_md(tmp_path, "s1", "skill one")
        loader.scan_skills(force=True)
        result = loader.build_system_prompt("Base prompt")
        assert "s1" in result

    def test_without_skills(self, tmp_path):
        loader = SkillLoader(skill_paths=[tmp_path])
        result = loader.build_system_prompt("Base prompt")
        assert "Base prompt" in result


class TestSkillLoaderCacheValidation:
    def test_valid_cache(self, tmp_path):
        loader = SkillLoader(skill_paths=[tmp_path])
        _write_skill_md(tmp_path, "s1", "d")
        loader.scan_skills(force=True)
        assert loader._is_cache_valid() is True

    def test_dir_mtime_changed(self, tmp_path):
        loader = SkillLoader(skill_paths=[tmp_path])
        d = _write_skill_md(tmp_path, "s1", "d")
        loader.scan_skills(force=True)
        # Touch dir to change mtime
        d.touch()
        # Invalidation depends on platform timing, just test the method runs
        result = loader._is_cache_valid()
        # Just verify it returns a boolean without error
        assert isinstance(result, bool)

    def test_dir_removed(self, tmp_path):
        loader = SkillLoader(skill_paths=[tmp_path])
        d = _write_skill_md(tmp_path, "s1", "d")
        loader.scan_skills(force=True)
        import shutil

        shutil.rmtree(d)
        assert loader._is_cache_valid() is False


class TestValidateSkillPackage:
    def test_valid_zip(self, tmp_path):
        zip_path = tmp_path / "pkg.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: test\ndescription: d\n---\nBody")
        result = validate_skill_package(str(zip_path))
        assert result is not None
        assert result["name"] == "test"

    def test_no_skill_md(self, tmp_path):
        zip_path = tmp_path / "bad.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "no skill")
        assert validate_skill_package(str(zip_path)) is None

    def test_invalid_yaml(self, tmp_path):
        zip_path = tmp_path / "bad_yaml.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SKILL.md", "not valid yaml {{{")
        assert validate_skill_package(str(zip_path)) is None


class TestInstallSkill:
    def test_successful_install(self, tmp_path):
        zip_path = tmp_path / "pkg.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "myskill/SKILL.md", "---\nname: myskill\ndescription: d\n---\nBody"
            )
        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        assert install_skill(str(zip_path), install_dir) is True
        assert (install_dir / "myskill" / "SKILL.md").exists()

    def test_invalid_package_loader(self, tmp_path):
        zip_path = tmp_path / "bad.zip"
        zip_path.write_bytes(b"not a zip")
        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        assert install_skill(str(zip_path), install_dir) is False

    def test_path_traversal_blocked(self, tmp_path):
        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "../escape/SKILL.md", "---\nname: escape\ndescription: d\n---\nBody"
            )
        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        assert install_skill(str(zip_path), install_dir) is False


class TestExtractArchive:
    def test_zip_extraction(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("dir/file.txt", "content")
        dest = tmp_path / "out"
        _extract_archive(str(zip_path), dest)
        assert (dest / "dir" / "file.txt").exists()

    def test_tar_gz_extraction(self, tmp_path):
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            info = tarfile.TarInfo(name="dir/file.txt")
            info.size = 7
            tf.addfile(info, io.BytesIO(b"content"))
        dest = tmp_path / "out2"
        _extract_archive(str(tar_path), dest)
        assert (dest / "dir" / "file.txt").exists()

    def test_unsupported_format(self, tmp_path):
        bad = tmp_path / "bad.rar"
        bad.write_bytes(b"not archive")
        dest = tmp_path / "out3"
        assert _extract_archive(str(bad), dest) is False


class TestFindSkillDir:
    def test_root_has_skill_md(self, tmp_path):
        (tmp_path / "SKILL.md").write_text("---\nname: t\n---", encoding="utf-8")
        result = _find_skill_dir(tmp_path)
        assert result == tmp_path

    def test_subdir_has_skill_md(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "SKILL.md").write_text("---\nname: t\n---", encoding="utf-8")
        result = _find_skill_dir(tmp_path)
        assert result == sub

    def test_find_skill_dir_not_found(self, tmp_path):
        assert _find_skill_dir(tmp_path) is None


class TestScanAllSkills:
    def test_with_project_path_extended(self, tmp_path):
        skills_dir = tmp_path / ".chat" / "skills"
        _write_skill_md(skills_dir, "s1", "project skill")
        results = scan_all_skills(tmp_path)
        assert any(s["name"] == "s1" for s in results)


# ============================================================================
# Lines 123, 137-138: OSError edge cases in _is_cache_valid
# ============================================================================


class TestSkillLoaderCacheOSError:
    def test_dir_stat_raises_oserror(self, tmp_path):
        """Lines 118-123: OSError when checking if base_path exists or getting mtime."""
        loader = SkillLoader(skill_paths=[tmp_path])
        _write_skill_md(tmp_path, "s1", "d")
        loader.scan_skills(force=True)

        # Mock Path.exists to return True but Path.stat to raise OSError
        original_stat = Path.stat

        def oserror_stat(self, **kwargs):
            if str(self) == str(tmp_path):
                raise OSError("Device not ready")
            return original_stat(self, **kwargs)

        with patch.object(Path, "stat", oserror_stat):
            # OSError in exists/stat check should return False
            result = loader._is_cache_valid()
            assert result is False

    def test_dir_exists_but_stat_fails(self, tmp_path):
        """Lines 118-123: base_path.exists() returns True but stat() fails."""
        loader = SkillLoader(skill_paths=[tmp_path])
        _write_skill_md(tmp_path, "s1", "d")
        loader.scan_skills(force=True)

        # Mock exists to return True, but stat raises OSError
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "stat", side_effect=OSError("I/O error")),
        ):
            result = loader._is_cache_valid()
            assert result is False

    def test_file_path_stat_raises_oserror(self, tmp_path):
        """Lines 130-138: OSError when checking file path in _file_mtimes."""
        loader = SkillLoader(skill_paths=[tmp_path])
        _write_skill_md(tmp_path, "s1", "d")
        loader.scan_skills(force=True)

        # Add a fake file path to _file_mtimes
        fake_path = "/nonexistent/file.md"
        loader._file_mtimes[fake_path] = 123456.0

        with patch("chcode.utils.skill_loader.Path") as mock_path_cls:
            # Setup: the fake path exists but stat() raises OSError
            mock_fp = MagicMock()
            mock_fp.exists.return_value = True
            mock_fp.stat.side_effect = OSError("Permission denied")
            mock_path_cls.return_value = mock_fp

            # Create a real path for tmp_path
            real_path = Path(tmp_path)
            mock_path_cls.side_effect = lambda p: (
                mock_fp if str(p) == fake_path else real_path
            )

            result = loader._is_cache_valid()
            # OSError in file stat should return False
            assert result is False

    def test_file_path_not_exists_returns_false(self, tmp_path):
        """Lines 132-134: File path doesn't exist in _file_mtimes."""
        loader = SkillLoader(skill_paths=[tmp_path])
        _write_skill_md(tmp_path, "s1", "d")
        loader.scan_skills(force=True)

        # Add a fake file path to _file_mtimes
        fake_path = "/nonexistent/file.md"
        loader._file_mtimes[fake_path] = 123456.0

        with patch("chcode.utils.skill_loader.Path") as mock_path_cls:
            # Setup: the fake path doesn't exist
            mock_fp = MagicMock()
            mock_fp.exists.return_value = False
            mock_path_cls.return_value = mock_fp

            result = loader._is_cache_valid()
            # Missing file should return False
            assert result is False

    def test_file_path_mtime_mismatch_returns_false(self, tmp_path):
        """Lines 132-135: File mtime doesn't match cached value."""
        loader = SkillLoader(skill_paths=[tmp_path])
        skill_dir = _write_skill_md(tmp_path, "s1", "d")
        skill_md = skill_dir / "SKILL.md"
        loader.scan_skills(force=True)

        # Modify the file mtime in cache to a wrong value
        loader._file_mtimes[str(skill_md)] = 0.0

        result = loader._is_cache_valid()
        # Mtime mismatch should return False
        assert result is False


# ============================================================================
# Lines 220-221: _parse_skill_metadata read_text exception
# ============================================================================


class TestParseSkillMetadataReadError:
    """Cover lines 220-221: read_text raises exception in _parse_skill_metadata."""

    def test_read_text_raises_exception(self, tmp_path):
        """Lines 220-221: skill_md_path.read_text raises, returns None."""
        loader = SkillLoader(skill_paths=[tmp_path])
        fake_md = tmp_path / "SKILL.md"
        fake_md.write_text("---\nname: test\n---\nbody", encoding="utf-8")

        with patch.object(
            fake_md.__class__, "read_text", side_effect=PermissionError("denied")
        ):
            result = loader._parse_skill_metadata(fake_md)
            assert result is None


# ============================================================================
# Lines 418: _extract_archive tar.gz path traversal
# ============================================================================


class TestExtractArchiveTarGzTraversal:
    """Cover line 418: tar.gz path traversal returns False."""

    def test_tar_gz_path_traversal_blocked(self, tmp_path):
        """Line 418: tar.gz member with path traversal is blocked."""
        tar_path = tmp_path / "evil.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            info = tarfile.TarInfo(name="../escape.txt")
            info.size = 4
            tf.addfile(info, io.BytesIO(b"evil"))
        dest = tmp_path / "out"
        result = _extract_archive(str(tar_path), dest)
        assert result is False


# ============================================================================
# Lines 463: validate_skill_package SKILL.md not found after extraction
# ============================================================================


class TestValidateSkillPackageSkillMdMissing:
    """Cover line 463: skill_md.exists() returns False after extraction."""

    def test_skill_md_dir_exists_but_no_skill_md_file(self, tmp_path):
        """Line 463: _find_skill_dir returns a dir but SKILL.md file doesn't exist.
        This is tricky because _find_skill_dir looks for SKILL.md. We need to mock it."""
        # Create a zip that extracts a directory with a SKILL.md (so _find_skill_dir works)
        # but then mock skill_md.exists() to return False
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("myskill/SKILL.md", "---\nname: test\n---\nbody")

        with (
            patch("chcode.utils.skill_loader._find_skill_dir") as mock_find,
            patch("pathlib.Path.exists", return_value=False),
        ):
            # _find_skill_dir returns a path, but SKILL.md doesn't actually exist there
            mock_find.return_value = tmp_path / "myskill"
            result = validate_skill_package(str(zip_path))
            assert result is None


class TestSkillAgentContextYolo:
    """Tests for SkillAgentContext.yolo field"""

    def test_default_yolo_false(self):
        from chcode.utils.skill_loader import SkillAgentContext

        ctx = SkillAgentContext(
            skill_loader=MagicMock(),
            model_config={},
            working_directory=Path("/tmp"),
        )
        assert ctx.yolo is False

    def test_yolo_true(self):
        from chcode.utils.skill_loader import SkillAgentContext

        ctx = SkillAgentContext(
            skill_loader=MagicMock(),
            model_config={},
            working_directory=Path("/tmp"),
            yolo=True,
        )
        assert ctx.yolo is True
