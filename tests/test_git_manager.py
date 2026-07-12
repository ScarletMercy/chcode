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
    def test_init_shadow_fresh(self, tmp_path: Path):
        """cp-repo 不存在 -> 完整建仓 + 配置 + add+commit"""
        gm = GitManager(str(tmp_path))
        calls = []

        def mock_run(args, **kwargs):
            calls.append(args)
            return _mock_run(0, stdout="abc123\n" if args[0] == "rev-list" else "")

        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.init_shadow()
        assert result is True
        assert calls[0] == ["init"]
        assert ["config", "--local", "core.bare", "false"] in calls
        assert ["config", "--local", "user.name", "chcode"] in calls
        assert ["config", "--local", "core.autocrlf", "false"] in calls
        assert ["add", "."] in calls
        assert ["commit", "-m", "init", "--allow-empty"] in calls
        assert (gm.cp_repo_dir / "info" / "exclude").read_text() == gm.SHADOW_EXCLUDE

    def test_init_shadow_includes_existing_files(self, tmp_path: Path):
        """有文件时 init 含项目初始文件；rollback 到 init 保留原有文件、删除后续改动"""
        import subprocess
        (tmp_path / "原有.py").write_text("src", encoding="utf-8")
        gm = GitManager(str(tmp_path))
        assert gm.init_shadow() is True
        r = subprocess.run(
            ["git", f"--git-dir={gm.cp_repo_dir}", "show", "HEAD:原有.py"],
            cwd=str(tmp_path), capture_output=True, text=True,
        )
        assert r.returncode == 0 and r.stdout == "src"
        (tmp_path / "新.txt").write_text("new", encoding="utf-8")
        assert gm.add_commit("msg1") == 2
        assert gm.rollback(["msg1"], ["msg1"]) == 1
        assert (tmp_path / "原有.py").exists()
        assert not (tmp_path / "新.txt").exists()

    def test_init_shadow_idempotent(self, tmp_path: Path):
        """cp-repo 已存在 -> 跳过重建，仅 _ensure_init_checkpoint"""
        gm = GitManager(str(tmp_path))
        gm.cp_repo_dir.mkdir(parents=True, exist_ok=True)  # 模拟已存在
        calls = []

        def mock_run(args, **kwargs):
            calls.append(args)
            return _mock_run(0, stdout="abc123\n" if args[0] == "rev-list" else "")

        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.init_shadow()
        assert result is True
        assert ["init"] not in calls
        assert ["add", "-A"] not in calls
        assert not any(c[0] == "commit" for c in calls)
        assert not any(c[0] == "config" for c in calls)

    def test_init_shadow_init_fails(self, tmp_path: Path):
        """git init 返回非 0 -> 早退返回 False，不继续 config/commit"""
        gm = GitManager(str(tmp_path))
        calls = []

        def mock_run(args, **kwargs):
            calls.append(args)
            if args[0] == "init":
                return _mock_run(128, stderr="init failed")
            return _mock_run(0)

        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.init_shadow()
        assert result is False
        assert not any(c[0] == "config" for c in calls)
        assert not any(c[0] == "commit" for c in calls)

    def test_init_shadow_fresh_real(self, tmp_path: Path):
        """真实 git：init_shadow 后 add_commit / rollback 正常"""
        gm = GitManager(str(tmp_path))
        assert gm.init_shadow() is True
        (tmp_path / "a.txt").write_text("v1", encoding="utf-8")
        assert gm.add_commit("msg1") == 2
        (tmp_path / "a.txt").write_text("v2", encoding="utf-8")
        (tmp_path / "b.txt").write_text("new", encoding="utf-8")
        assert gm.add_commit("msg2") == 3
        # rollback 到 msg1 的父提交(=init)；本用例 init 时目录为空，故 a/b 均被移除
        assert gm.rollback(["msg1"], ["msg1", "msg2"]) == 1
        assert not (tmp_path / "a.txt").exists()
        assert not (tmp_path / "b.txt").exists()

    def test_init_shadow_fork_real(self, tmp_path: Path):
        """真实 git：fork(复制 .chat/cp-repo)后新位置追踪自己的文件，不串旧路径"""
        import shutil
        import subprocess
        src = tmp_path / "src"
        src.mkdir()
        gm = GitManager(str(src))
        assert gm.init_shadow() is True
        (src / "a.txt").write_text("src-v1", encoding="utf-8")
        assert gm.add_commit("msg1") == 2

        dst = tmp_path / "dst"
        shutil.copytree(src, dst)
        (dst / "a.txt").write_text("dst-v2", encoding="utf-8")
        (dst / "b.txt").write_text("dst-new", encoding="utf-8")

        gm2 = GitManager(str(dst))
        assert gm2.init_shadow() is True  # cp-repo 已存在，跳过 init
        assert gm2.add_commit("forkmsg") == 3

        def head_show(path):
            r = subprocess.run(
                ["git", f"--git-dir={gm2.cp_repo_dir}", "show", f"HEAD:{path}"],
                cwd=str(dst), capture_output=True, text=True,
            )
            return r.stdout

        assert head_show("a.txt") == "dst-v2"  # 不是 src 的 src-v1
        assert head_show("b.txt") == "dst-new"

    def test_init_shadow_corrupt_real(self, tmp_path: Path):
        """真实 git：cp-repo 存在但无 HEAD（如 fork 复制中断）-> 删除重建后可用"""
        gm = GitManager(str(tmp_path))
        gm.cp_repo_dir.mkdir(parents=True, exist_ok=True)
        (gm.cp_repo_dir / "partial").write_text("junk", encoding="utf-8")
        assert gm.init_shadow() is True
        # 半残内容被清理
        assert not (gm.cp_repo_dir / "partial").exists()
        # 仓库正常可用
        (tmp_path / "a.txt").write_text("v1", encoding="utf-8")
        assert gm.add_commit("m1") == 2

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

    def test_run_timeout(self, tmp_path: Path):
        import subprocess

        gm = GitManager(str(tmp_path))
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 30)):
            with pytest.raises(RuntimeError, match="超时"):
                gm._run(["status"])
