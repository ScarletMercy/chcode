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
        assert gm.add_commit("msg1") is True
        assert gm.rollback(["msg1"], ["msg1"]) is True
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

    def test_init_shadow_probe_exception(self, tmp_path: Path):
        """探针 _run 抛异常(超时/杀软锁/权限)-> 降级返回 False 并告警，不穿到启动"""
        gm = GitManager(str(tmp_path))
        gm.cp_repo_dir.mkdir(parents=True)  # 短路 and 需 exists=True 才会触发探针

        def mock_run(args, **kwargs):
            if args[0] == "rev-parse":
                raise RuntimeError("杀软锁定 cp-repo")
            return _mock_run(0)

        with patch("chcode.utils.git_manager.render_warning") as rw:
            with patch.object(gm, "_run", side_effect=mock_run):
                result = gm.init_shadow()
        assert result is False
        rw.assert_called_once()

    def test_init_shadow_fresh_real(self, tmp_path: Path):
        """真实 git：init_shadow 后 add_commit / rollback 正常"""
        gm = GitManager(str(tmp_path))
        assert gm.init_shadow() is True
        (tmp_path / "a.txt").write_text("v1", encoding="utf-8")
        assert gm.add_commit("msg1") is True
        (tmp_path / "a.txt").write_text("v2", encoding="utf-8")
        (tmp_path / "b.txt").write_text("new", encoding="utf-8")
        assert gm.add_commit("msg2") is True
        # rollback 到 msg1 的父提交(=init)；本用例 init 时目录为空，故 a/b 均被移除
        assert gm.rollback(["msg1"], ["msg1", "msg2"]) is True
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
        assert gm.add_commit("msg1") is True

        dst = tmp_path / "dst"
        shutil.copytree(src, dst)
        (dst / "a.txt").write_text("dst-v2", encoding="utf-8")
        (dst / "b.txt").write_text("dst-new", encoding="utf-8")

        gm2 = GitManager(str(dst))
        assert gm2.init_shadow() is True  # cp-repo 已存在，跳过 init
        # 仅复制层继承（源全部+新消息）；完整 fork 经 rollback 裁剪见 test_fork_rollback_trims_checkpoints_real
        assert gm2.add_commit("forkmsg") is True

        def head_show(path):
            r = subprocess.run(
                ["git", f"--git-dir={gm2.cp_repo_dir}", "show", f"HEAD:{path}"],
                cwd=str(dst), capture_output=True, text=True,
            )
            return r.stdout

        assert head_show("a.txt") == "dst-v2"  # 不是 src 的 src-v1
        assert head_show("b.txt") == "dst-new"

    def test_fork_rollback_trims_checkpoints_real(self, tmp_path: Path):
        """真实 git：完整 fork 序列经 rollback 裁剪 fork 点及之后的检查点

        test_init_shadow_fork_real 只测复制层继承（计数=源全部+新消息）；
        本条覆盖 rollback 裁剪，锁定 fork 后计数 = fork 点之前 + 新消息。
        """
        import shutil
        src = tmp_path / "src"
        src.mkdir()
        gm = GitManager(str(src))
        assert gm.init_shadow() is True
        (src / "a.txt").write_text("v1", encoding="utf-8")
        assert gm.add_commit("m1") is True
        (src / "a.txt").write_text("v2", encoding="utf-8")
        assert gm.add_commit("m2") is True  # fork 点
        (src / "a.txt").write_text("v3", encoding="utf-8")
        assert gm.add_commit("m3") is True

        dst = tmp_path / "dst"
        shutil.copytree(src, dst)

        gm2 = GitManager(str(dst))
        assert gm2.init_shadow() is True  # cp-repo 已存在，跳过 init
        assert len(json.loads(gm2.checkpoints_file.read_text(encoding="utf-8"))) == 4  # 复制后全量继承源会话

        # fork 点 = m2：rollback 裁剪 m2/m3，工作目录回到 m1
        assert gm2.rollback(["m2", "m3"], ["m1", "m2", "m3"]) is True
        assert (dst / "a.txt").read_text(encoding="utf-8") == "v1"  # 回到 m1，不是 v3

        data = json.loads(gm2.checkpoints_file.read_text(encoding="utf-8"))
        assert "init" in data and "m1" in data
        assert "m2" not in data and "m3" not in data

        # fork 后新消息会改文件；计数 = fork 点之前(init+m1) + 新消息
        (dst / "a.txt").write_text("fork-v", encoding="utf-8")
        assert gm2.add_commit("forkmsg") is True

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
        assert gm.add_commit("m1") is True

    def test_add_commit_success(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)
        call_count = 0

        def mock_run(args, **kwargs):
            nonlocal call_count
            call_count += 1
            if args[0] == "add":
                return _mock_run(0)
            elif args[0] == "diff":
                return _mock_run(1)
            elif args[0] == "commit":
                return _mock_run(0)
            elif args[0] == "rev-parse":
                return _mock_run(0, stdout="abc123\n")
            return _mock_run(0)

        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.add_commit("msg1")
            assert result is True

    def test_add_commit_add_fails(self, tmp_path: Path):
        gm = GitManager(str(tmp_path))
        with patch.object(gm, "_run", return_value=_mock_run(1)):
            result = gm.add_commit("msg1")
            assert result is False

    def test_add_commit_no_changes(self, tmp_path: Path):
        """真实 git:无改动时 add_commit 返回 True,不新增 checkpoint"""
        gm = GitManager(str(tmp_path))
        assert gm.init_shadow() is True
        assert gm.add_commit("m1") is True
        data = json.loads(gm.checkpoints_file.read_text(encoding="utf-8"))
        assert "m1" not in data

    def test_add_commit_exclude_lost_keeps_cp_repo_untracked(self, tmp_path: Path):
        """exclude 丢失后 add_commit 补写,不把 cp-repo 自身暂存(防自毁链)"""
        gm = GitManager(str(tmp_path))
        assert gm.init_shadow() is True
        (gm.cp_repo_dir / "info" / "exclude").unlink()  # 模拟杀软/误删
        (tmp_path / "a.txt").write_text("v1", encoding="utf-8")
        assert gm.add_commit("m1") is True
        tree = gm._run(["ls-tree", "-r", "--name-only", "HEAD"]).stdout
        assert ".chat/cp-repo" not in tree
        assert "a.txt" in tree

    def test_add_commit_json_corrupt(self, tmp_path: Path):
        """existing json 损坏:特有告警 + 返回 False,不崩"""
        gm = GitManager(str(tmp_path))
        assert gm.init_shadow() is True
        gm.checkpoints_file.write_text("{ broken json", encoding="utf-8")
        (tmp_path / "a.txt").write_text("v1", encoding="utf-8")
        with patch("chcode.utils.git_manager.render_warning"):
            assert gm.add_commit("m1") is False
        log = gm._run(["log", "--format=%s"]).stdout
        assert "m1" not in log  # 边角 2:损坏 fail-fast,commit 未发生,不留孤儿

    def test_rollback_json_corrupt(self, tmp_path: Path):
        """checkpoints.json 损坏:特有告警 + 返回 False,不崩,文件不动"""
        gm = GitManager(str(tmp_path))
        assert gm.init_shadow() is True
        gm.checkpoints_file.write_text("{ broken json", encoding="utf-8")
        with patch("chcode.utils.git_manager.render_warning"):
            assert gm.rollback(["m1"], ["m1"]) is False
        # fail-fast:读失败不写,文件保持损坏原样(对齐 add_commit)
        assert gm.checkpoints_file.read_text(encoding="utf-8") == "{ broken json"

    def test_init_shadow_json_corrupt(self, tmp_path: Path):
        """checkpoints.json 损坏:init 告警 + return,不崩,文件不动,不被覆写成 {init}"""
        gm = GitManager(str(tmp_path))
        assert gm.init_shadow() is True
        gm.checkpoints_file.write_text("{ broken json", encoding="utf-8")
        with patch("chcode.utils.git_manager.render_warning"):
            assert gm.init_shadow() is True
        # fail-fast:读失败不写,文件保持损坏原样(对齐 add_commit/rollback)
        assert gm.checkpoints_file.read_text(encoding="utf-8") == "{ broken json"

    def test_add_commit_write_fails_rolls_back_commit(self, tmp_path: Path):
        """write-json 失败:撤 commit,工作区改动保留"""
        gm = GitManager(str(tmp_path))
        assert gm.init_shadow() is True
        (tmp_path / "a.txt").write_text("v1", encoding="utf-8")
        with patch.object(gm, "_write_checkpoints", side_effect=OSError("disk full")):
            assert gm.add_commit("m1") is False
        log = gm._run(["log", "--format=%s"]).stdout
        assert "m1" not in log
        # 工作区改动保留(reset --mixed 不删工作区文件)
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "v1"
        data = json.loads(gm.checkpoints_file.read_text(encoding="utf-8"))
        assert "m1" not in data

    def test_undo_commit_mixed_nonzero_falls_back_to_soft(self, tmp_path: Path):
        """mixed rc≠0(如盘满)时落 soft：锁定 A 修复"""
        gm = GitManager(str(tmp_path))
        calls = []

        def mock_run(args, **kwargs):
            calls.append(args)
            if args[0] == "reset" and args[1] == "--mixed":
                return _mock_run(1)  # mixed 失败(盘满写不了 index)
            return _mock_run(0)  # soft 成功

        with patch.object(gm, "_run", side_effect=mock_run):
            gm._undo_commit("deadbeef")
        # soft 被调 = A 已修(修前 mixed rc≠0 不 raise，soft 不触发)
        assert any(a[0] == "reset" and a[1] == "--soft" for a in calls)

    def test_undo_commit_both_fail_warns(self, tmp_path: Path):
        """mixed 和 soft 都败时告警，不静默留孤儿：锁定 E 修复"""
        gm = GitManager(str(tmp_path))

        with patch.object(gm, "_run", return_value=_mock_run(1)):
            with patch("chcode.utils.git_manager.render_warning") as rw:
                gm._undo_commit("deadbeef")
        rw.assert_called_once()

    def test_run_timeout(self, tmp_path: Path):
        import subprocess

        gm = GitManager(str(tmp_path))
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 30)):
            with pytest.raises(RuntimeError, match="超时"):
                gm._run(["status"])
