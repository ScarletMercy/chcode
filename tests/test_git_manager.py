import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from chcode.utils.git_manager import GitManager


def _mock_run(returncode=0, stdout="", stderr=""):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


class TestGitManager:
    def test_is_repo_true(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        with patch.object(gm, "_run", return_value=_mock_run(0)):
            assert gm.is_repo() is True

    def test_is_repo_false(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        gm._is_repo = None
        with patch.object(gm, "_run", return_value=_mock_run(128)):
            assert gm.is_repo() is False

    def test_is_repo_cached(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        gm._is_repo = True
        assert gm.is_repo() is True

    def test_init_already_repo(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)
        with patch.object(gm, "is_repo", return_value=True), \
             patch.object(gm, "_run", return_value=_mock_run(0, stdout="abc123\n")):
            result = gm.init()
            assert result is False

    def test_init_new_repo(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)
        gm.gitignore_file = tmp_path / ".gitignore"
        with patch.object(gm, "is_repo", return_value=False), \
             patch.object(gm, "_run", return_value=_mock_run(0)), \
             patch.object(gm, "create_gitignore", return_value=True):
            result = gm.init()
            assert result is True

    def test_add_commit_success(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)
        call_count = 0

        def mock_run(args, **kwargs):
            nonlocal call_count
            call_count += 1
            if args[0] == "add":
                return _mock_run(0)
            elif args[0] == "commit":
                return _mock_run(0)
            elif args[0] == "rev-parse":
                return _mock_run(0, stdout="abc123\n")
            return _mock_run(0)

        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.add_commit("msg1")
            assert result == 1

    def test_add_commit_add_fails(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        with patch.object(gm, "_run", return_value=_mock_run(1)):
            result = gm.add_commit("msg1")
            assert result is False

    def test_count_checkpoints_file(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)
        gm.checkpoints_file.write_text(json.dumps({"a": "h1", "b": "h2"}))
        assert gm.count_checkpoints() == 2

    def test_count_checkpoints_no_file(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        assert gm.count_checkpoints() == 0

    def test_count_checkpoints_with_arg(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        assert gm.count_checkpoints(5) == 5

    def test_create_gitignore(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        gm.gitignore_file = tmp_path / ".gitignore"
        with patch.object(gm, "_run", return_value=_mock_run(0)):
            result = gm.create_gitignore()
            assert result is True
            assert gm.gitignore_file.exists()

    def test_create_gitignore_custom_content(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        gm.gitignore_file = tmp_path / ".gitignore"
        with patch.object(gm, "_run", return_value=_mock_run(0)):
            gm.create_gitignore("custom\n")
            assert gm.gitignore_file.read_text() == "custom\n"

    def test_run_timeout(self, tmp_path: Path):
        import subprocess

        gm = GitManager(str(tmp_path))
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 30)):
            with pytest.raises(RuntimeError, match="超时"):
                gm._run(["status"])
