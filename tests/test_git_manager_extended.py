"""Extended tests for chcode/utils/git_manager.py - coverage improvement"""

import json
import subprocess
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


def _cp_keys(gm):
    """读取 checkpoints.json 的键集合"""
    return set(json.loads(gm.checkpoints_file.read_text()).keys())


class TestRollback:
    """Tests for rollback with fuzzy matching"""

    def test_rollback_exact_match(self, tmp_path: Path):
        """Exact match in checkpoints - direct rollback"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup checkpoints file
        checkpoints = {
            "init": "abc123",
            "msg1&msg2": "def456",
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        with patch.object(gm, "_run", return_value=_mock_run(0)):
            result = gm.rollback(["msg1", "msg2"], ["msg1", "msg2"])
            assert result is True
            assert _cp_keys(gm) == {"init"}  # Only "init" remains

    def test_rollback_reset_fail_preserves_json(self, tmp_path: Path):
        """reset --hard rc≠0 时不写 json--json 与仓库一致性守恒"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)
        original = {"init": "abc123", "msg1&msg2": "def456"}
        gm.checkpoints_file.write_text(json.dumps(original))

        with patch.object(gm, "_run", return_value=_mock_run(1, stderr="reset fail")):
            result = gm.rollback(["msg1", "msg2"], ["msg1", "msg2"])
        assert result is False
        assert json.loads(gm.checkpoints_file.read_text()) == original

    def test_rollback_fuzzy_has_before_has_after(self, tmp_path: Path):
        """Fuzzy match: has_before + has_after -> rollback to before"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup: before checkpoint, then looking for msg3, after checkpoint
        checkpoints = {
            "init": "abc000",
            "msg1": "def111",
            "msg4": "ghi444",  # This is after msg3
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        # Looking for msg3, which doesn't exist
        # msg1 is before, msg4 is after
        with patch.object(gm, "_run", return_value=_mock_run(0)):
            result = gm.rollback(["msg3"], ["msg1", "msg3", "msg4"])
            assert result is True
            # Should rollback to msg1, remove msg4
            assert _cp_keys(gm) == {"init", "msg1"}

    def test_rollback_fuzzy_no_before_has_after(self, tmp_path: Path):
        """Fuzzy match: no_before + has_after -> rollback to init"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup: no before checkpoint, only after
        checkpoints = {
            "init": "abc000",
            "msg3": "def333",  # This is after msg1
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        # Looking for msg1, which doesn't exist
        # No before, msg3 is after
        with patch.object(gm, "_run", return_value=_mock_run(0)):
            result = gm.rollback(["msg1"], ["msg1", "msg3"])
            assert result is True
            # Should rollback to init, remove msg3
            assert _cp_keys(gm) == {"init"}

    def test_rollback_fuzzy_has_before_no_after(self, tmp_path: Path):
        """Fuzzy match: has_before + no_after -> no rollback"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup: before checkpoint only
        checkpoints = {
            "init": "abc000",
            "msg1": "def111",
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        # Looking for msg3, which doesn't exist
        # msg1 is before, no after -> no rollback
        with patch.object(gm, "_run", return_value=_mock_run(0)):
            result = gm.rollback(["msg3"], ["msg1", "msg3"])
            assert result is True
            # Should NOT rollback, file unchanged
            assert _cp_keys(gm) == {"init", "msg1"}

    def test_rollback_no_checkpoints_file(self, tmp_path: Path):
        """No checkpoints file returns False"""
        gm = GitManager(str(tmp_path))
        result = gm.rollback(["msg1"], ["msg1"])
        assert result is False

    def test_rollback_reset_fails(self, tmp_path: Path):
        """Reset failure returns False"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        checkpoints = {
            "init": "abc000",
            "msg1": "def111",
            "msg2": "ghi222",  # This is after the target
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        # msg1 is before, msg2 is after, so will try to reset
        with patch.object(gm, "_run", return_value=_mock_run(1)):
            result = gm.rollback(["msg1.5"], ["msg1", "msg1.5", "msg2"])
            # Reset fails, returns False
            assert result is False

    def test_rollback_exception_handling(self, tmp_path: Path):
        """Exception in reset returns False"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        checkpoints = {
            "init": "abc000",
            "msg1": "def111",
            "msg2": "ghi222",  # This is after the target
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        def raise_exception(*args, **kwargs):
            raise RuntimeError("Test error")

        # msg1 is before, msg2 is after, so will try to reset
        with patch("chcode.utils.git_manager.render_warning") as rw:
            with patch.object(gm, "_run", side_effect=raise_exception):
                result = gm.rollback(["msg1.5"], ["msg1", "msg1.5", "msg2"])
                # Exception in reset, returns False
                assert result is False
        rw.assert_called_once()


class TestRunErrorCases:
    """Tests for _run error cases"""

    def test_run_non_silent_error(self, tmp_path: Path, capsys):
        """Non-silent mode prints error info"""
        gm = GitManager(str(tmp_path))

        with patch.object(gm, "_run", wraps=gm._run):
            with patch("subprocess.run", return_value=_mock_run(1, stderr="Error occurred")):
                result = gm._run(["status"], silent=False)
                assert result.returncode == 1

                captured = capsys.readouterr()
                # Should print error info in non-silent mode
                assert "1" in captured.out

    def test_run_timeout_exception(self, tmp_path: Path):
        """Timeout exception raises RuntimeError"""
        gm = GitManager(str(tmp_path))

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 30)):
            with pytest.raises(RuntimeError) as exc_info:
                gm._run(["status"])
            assert "超时" in str(exc_info.value)

    def test_run_general_exception(self, tmp_path: Path):
        """General exception raises RuntimeError"""
        gm = GitManager(str(tmp_path))

        with patch("subprocess.run", side_effect=OSError("Permission denied")):
            with pytest.raises(RuntimeError) as exc_info:
                gm._run(["status"])
            assert "执行失败" in str(exc_info.value)


class TestAddCommit:
    """Tests for add_commit with edge cases"""

    def test_add_commit_revparse_failure(self, tmp_path: Path):
        """Commit success but rev-parse failure"""
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
                return _mock_run(1, stderr="Not found")
            return _mock_run(0)

        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.add_commit("msg1")
            assert result is False

    def test_add_commit_with_specific_files(self, tmp_path: Path):
        """Commit specific files"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        calls = []

        def mock_run(args, **kwargs):
            calls.append(args)
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
            result = gm.add_commit("msg1", files=["file1.py", "file2.py"])
            assert result is True
            # Check that add was called with specific files
            assert calls[0] == ["add", "file1.py", "file2.py"]

    def test_add_commit_commit_fails(self, tmp_path: Path):
        """Commit operation fails"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        def mock_run(args, **kwargs):
            if args[0] == "add":
                return _mock_run(0)
            elif args[0] == "diff":
                return _mock_run(1)
            elif args[0] == "commit":
                return _mock_run(1)
            return _mock_run(0)

        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.add_commit("msg1")
            assert result is False

    def test_add_commit_commit_fail_skips_undo(self, tmp_path: Path):
        """commit 失败(committed=False)不触发 undo--committed 标志守恒"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        def mock_run(args, **kwargs):
            if args[0] == "add":
                return _mock_run(0)
            elif args[0] == "diff":
                return _mock_run(1)
            elif args[0] == "commit":
                return _mock_run(1)
            return _mock_run(0)

        with patch.object(gm, "_undo_commit") as undo:
            with patch.object(gm, "_run", side_effect=mock_run):
                result = gm.add_commit("msg1")
        assert result is False
        undo.assert_not_called()

    def test_add_commit_run_raises_warns(self, tmp_path: Path):
        """_run 抛 RuntimeError(超时/杀软锁/权限)-> 降级返回 False 并告警"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        def mock_run(args, **kwargs):
            if args[0] == "add":
                return _mock_run(0)
            elif args[0] == "diff":
                return _mock_run(1)
            elif args[0] == "commit":
                raise RuntimeError("杀软锁定 index")
            return _mock_run(0)

        with patch("chcode.utils.git_manager.render_warning") as rw:
            with patch.object(gm, "_run", side_effect=mock_run):
                result = gm.add_commit("msg1")
        assert result is False
        rw.assert_called_once()


class TestRollbackClassifyCheckpoints:
    """Tests for _classify_checkpoint_keys helper in rollback"""

    def test_classify_with_init_key(self, tmp_path: Path):
        """Init key is excluded from classification"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        checkpoints = {
            "init": "abc000",
            "msg1": "def111",
            "msg2": "ghi222",
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        # Rolling back msg3, msg1 is before, msg2 is after
        with patch.object(gm, "_run", return_value=_mock_run(0)):
            result = gm.rollback(["msg3"], ["msg1", "msg3", "msg2"])
            assert result is True
            # Should only keep init and msg1
            assert _cp_keys(gm) == {"init", "msg1"}

    def test_classify_msg_not_in_all_ids(self, tmp_path: Path):
        """Message IDs not in all_ids are excluded from classification but remain"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        checkpoints = {
            "init": "abc000",
            "msg1": "def111",
            "unknown_msg": "xxx000",  # Not in all_ids - excluded from classification
            "msg3": "ghi333",  # This is after msg2
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        with patch.object(gm, "_run", return_value=_mock_run(0)):
            result = gm.rollback(["msg2"], ["msg1", "msg2", "msg3"])
            assert result is True
            # After rollback: init + msg1 + unknown_msg remain (msg3 removed)
            assert _cp_keys(gm) == {"init", "msg1", "unknown_msg"}

    def test_orphan_not_flip_case2_to_case1(self, tmp_path: Path):
        """orphan 不得作为唯一 before 把 Case 2(回溯 init)翻成 Case 1(回溯到 orphan)"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoints = {"init": "abc000", "msg1": "def111", "msg2": "ghi222"}
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        reset_targets = []

        def mock_run(args, **kwargs):
            if args[:2] == ["reset", "--hard"]:
                reset_targets.append(args[2])
            return _mock_run(0)

        # 编辑第一组(msg2 为 all_ids 首位 => 无合法 before),msg1 为 orphan
        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.rollback(["msg2", "msg3"], ["msg2", "msg3"])
            assert result is True
        # 应回溯 init(abc000),而非 orphan msg1 的提交(def111)
        assert reset_targets == ["abc000"]

    def test_multiple_orphans_all_skipped_to_init(self, tmp_path: Path):
        """多个 orphan 全部跳过,不挑其中任意一个当目标,统一回溯 init"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoints = {
            "init": "abc000",
            "del1": "def111",  # orphan:已删轮次 1
            "del2": "ghi222",  # orphan:已删轮次 2
            "cur": "jkl333",
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        reset_targets = []

        def mock_run(args, **kwargs):
            if args[:2] == ["reset", "--hard"]:
                reset_targets.append(args[2])
            return _mock_run(0)

        # 编辑第一组(cur 为 all_ids 首位 => 无合法 before),del1/del2 均 orphan
        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.rollback(["cur", "cur2"], ["cur", "cur2"])
            assert result is True
        # 回溯 init,而非 del1/del2 任一;两个 orphan 都保留
        assert reset_targets == ["abc000"]
        assert _cp_keys(gm) == {"init", "del1", "del2"}

    def test_orphan_not_target_when_legit_before_exists(self, tmp_path: Path):
        """有合法 before 时,目标取合法前驱,orphan 既不当目标也不被删"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoints = {
            "init": "abc000",
            "old": "bef111",  # 合法 before
            "del": "def222",  # orphan
            "cur": "ghi333",
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        reset_targets = []

        def mock_run(args, **kwargs):
            if args[:2] == ["reset", "--hard"]:
                reset_targets.append(args[2])
            return _mock_run(0)

        # 编辑第二组:old(idx0)<fork_index(1) => 合法 before;del 是 orphan
        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.rollback(["cur", "x"], ["old", "cur"])
            assert result is True
        # 目标是合法前驱 old(bef111),不是 orphan del(def222);del 保留
        assert reset_targets == ["bef111"]
        assert _cp_keys(gm) == {"init", "old", "del"}

    def test_exact_match_path_preserves_orphan(self, tmp_path: Path):
        """精确匹配路径不受 orphan 影响:目标取 key 父提交,orphan 保留"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoints = {
            "init": "abc000",
            "del": "def111",  # orphan
            "cur": "ghi333",
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        reset_targets = []

        def mock_run(args, **kwargs):
            if args[:2] == ["reset", "--hard"]:
                reset_targets.append(args[2])
            return _mock_run(0)

        # message_ids_str="cur" 命中键 => 精确匹配路径
        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.rollback(["cur"], ["cur"])
            assert result is True
        # 目标是 cur 的父提交(ghi333~1),orphan del 保留
        assert reset_targets == ["ghi333~1"]
        assert _cp_keys(gm) == {"init", "del"}

    def test_classify_compound_key(self, tmp_path: Path):
        """Compound keys (msg1&msg2) use first part for classification"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        checkpoints = {
            "init": "abc000",
            "msg1&msg2": "def111",
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        # Looking for msg3, msg1 is before
        with patch.object(gm, "_run", return_value=_mock_run(0)):
            result = gm.rollback(["msg3"], ["msg1", "msg3"])
            assert result is True
            # Should keep init and compound key
            assert _cp_keys(gm) == {"init", "msg1&msg2"}


class TestRollbackForkIndex:
    """Tests for fork index calculation in rollback"""

    def test_fork_not_in_all_ids(self, tmp_path: Path):
        """Fork ID not in all_ids defaults to -1"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        checkpoints = {
            "init": "abc000",
            "msg1": "def111",
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        # Fork msg99 not in all_ids
        with patch.object(gm, "_run", return_value=_mock_run(0)):
            result = gm.rollback(["msg99"], ["msg1", "msg2"])
            assert result is True
            # With fork_index=-1, everything is "at_or_after"
            # So no before, has after -> rollback to init
            assert _cp_keys(gm) == {"init"}


class TestRollbackMultipleKeys:
    """Tests for handling multiple keys in checkpoints"""

    def test_rollback_removes_multiple_at_or_after(self, tmp_path: Path):
        """Multiple at_or_after keys all get removed"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        checkpoints = {
            "init": "abc000",
            "msg1": "def111",
            "msg4": "ghi444",
            "msg5": "jkl555",
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        # Looking for msg3, msg1 is before, msg4+msg5 are after
        with patch.object(gm, "_run", return_value=_mock_run(0)):
            result = gm.rollback(["msg3"], ["msg1", "msg3", "msg4", "msg5"])
            assert result is True
            # Should keep init and msg1 only
            assert _cp_keys(gm) == {"init", "msg1"}


# ============================================================================
# Line 47: _run non-silent mode with stdout
# ============================================================================


class TestRunNonSilentWithStdout:
    """Cover line 47: non-silent mode prints STDOUT when result.stdout is truthy."""

    def test_run_non_silent_stdout_printed(self, tmp_path: Path, capsys):
        """Line 47: result.stdout is truthy, gets printed in non-silent mode."""
        gm = GitManager(str(tmp_path))

        with patch("subprocess.run", return_value=_mock_run(1, stdout="some output", stderr="")):
            result = gm._run(["status"], silent=False)
            assert result.returncode == 1
            captured = capsys.readouterr()
            assert "STDOUT" in captured.out
            assert "some output" in captured.out


# ============================================================================
# Line 103: add_commit with existing checkpoints file
# ============================================================================


class TestAddCommitExistingCheckpoints:
    """Cover line 103: add_commit reads and updates existing checkpoints file."""

    def test_add_commit_merges_with_existing_checkpoints(self, tmp_path: Path):
        """Line 103: checkpoints_file exists, reads JSON and merges."""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        # Pre-populate checkpoints file
        existing = {"existing_msg": "abc000"}
        gm.checkpoints_file.write_text(json.dumps(existing))

        def mock_run(args, **kwargs):
            if args[0] == "add":
                return _mock_run(0)
            elif args[0] == "diff":
                return _mock_run(1)
            elif args[0] == "commit":
                return _mock_run(0)
            elif args[0] == "rev-parse":
                return _mock_run(0, stdout="def111\n")
            return _mock_run(0)

        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.add_commit("msg1")
            assert result is True
            # Verify the file was updated
            data = json.loads(gm.checkpoints_file.read_text())
            assert "existing_msg" in data
            assert "msg1" in data


# ============================================================================
# Lines 174-176: rollback exact match with exception
# ============================================================================


class TestRollbackExactMatchException:
    """Cover lines 174-176: exact match path, _run raises exception."""

    def test_rollback_exact_match_run_raises(self, tmp_path: Path):
        """Lines 174-176: exact match found but reset raises exception."""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        checkpoints = {
            "init": "abc000",
            "msg1&msg2": "def456",
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        def raise_error(*args, **kwargs):
            raise RuntimeError("git reset failed")

        with patch("chcode.utils.git_manager.render_warning") as rw:
            with patch.object(gm, "_run", side_effect=raise_error):
                result = gm.rollback(["msg1", "msg2"], ["msg1", "msg2"])
                assert result is False
        rw.assert_called_once()


# ============================================================================
# Lines 204-205: rollback else branch (no before, no after)
# ============================================================================


class TestRollbackElseBranch:
    """Cover lines 204-205: rollback else branch, no before and no after."""

    def test_rollback_no_before_no_after(self, tmp_path: Path):
        """Lines 204-205: neither before nor after checkpoints, returns True (no-op)."""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        # Only init key, all message IDs are "at_or_after" init
        checkpoints = {
            "init": "abc000",
        }
        gm.checkpoints_file.write_text(json.dumps(checkpoints))

        # all_ids contains nothing that matches any checkpoint key (except init
        # which is excluded from classification), so both before and after are empty
        with patch.object(gm, "_run", return_value=_mock_run(0)):
            result = gm.rollback(["msg1"], ["msg1"])
            assert result is True
            # else branch: no reset, file unchanged (only "init")
            assert _cp_keys(gm) == {"init"}


# ============================================================================
# Cross-session rollback tests (mocked _run)
# ============================================================================


