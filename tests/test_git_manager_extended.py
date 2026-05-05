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
            assert result == 1  # Only "init" remains

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
            # Should rollback to msg1, remove msg4
            assert result == 2  # init + msg1 remain

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
            # Should rollback to init, remove msg3
            assert result == 1  # Only init remains

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
            # Should NOT rollback, return current count
            assert result == 2  # init + msg1

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
        with patch.object(gm, "_run", side_effect=raise_exception):
            result = gm.rollback(["msg1.5"], ["msg1", "msg1.5", "msg2"])
            # Exception in reset, returns False
            assert result is False


class TestCreateGitignore:
    """Tests for create_gitignore with default content"""

    def test_create_gitignore_default_content(self, tmp_path: Path):
        """Creates .gitignore with default MINIMAL_GITIGNORE"""
        gm = GitManager(str(tmp_path))
        gm.gitignore_file = tmp_path / ".gitignore"

        with patch.object(gm, "_run", return_value=_mock_run(0)):
            result = gm.create_gitignore()
            assert result is True
            content = gm.gitignore_file.read_text()
            assert ".git" in content
            assert ".chat" in content
            assert ".venv" in content

    def test_create_gitignore_exception_handling(self, tmp_path: Path):
        """Exception handling returns False"""
        gm = GitManager(str(tmp_path))

        # Make the path invalid to trigger exception
        gm.gitignore_file = tmp_path / "nonexistent" / ".gitignore"

        result = gm.create_gitignore()
        assert result is False


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


class TestInit:
    """Tests for init with existing repo"""

    def test_init_existing_repo_creates_checkpoints(self, tmp_path: Path):
        """Init on existing repo creates checkpoints file"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        # Simulate existing repo
        with patch.object(gm, "is_repo", return_value=True), \
             patch.object(gm, "_run", return_value=_mock_run(0, stdout="abc123\n")):
            result = gm.init()
            assert result is False
            assert gm.checkpoints_file.exists()
            content = json.loads(gm.checkpoints_file.read_text())
            assert "init" in content

    def test_init_existing_repo_preserves_checkpoints(self, tmp_path: Path):
        """Init preserves existing checkpoints file"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        # Create existing checkpoints
        existing = {"msg1": "abc123"}
        gm.checkpoints_file.write_text(json.dumps(existing))

        with patch.object(gm, "is_repo", return_value=True), \
             patch.object(gm, "_run", return_value=_mock_run(0, stdout="def456\n")):
            result = gm.init()
            assert result is False
            content = json.loads(gm.checkpoints_file.read_text())
            assert content["msg1"] == "abc123"
            assert "init" in content

    def test_init_new_repo_creates_checkpoints(self, tmp_path: Path):
        """Init on new repo creates checkpoints file"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)
        gm.gitignore_file = tmp_path / ".gitignore"

        calls = []

        def mock_run(args, **kwargs):
            calls.append(args)
            return _mock_run(0)

        with patch.object(gm, "is_repo", return_value=False), \
             patch.object(gm, "_run", side_effect=mock_run), \
             patch.object(gm, "create_gitignore", return_value=True):
            result = gm.init()
            assert result is True
            assert gm.checkpoints_file.exists()

    def test_init_new_repo_creates_gitignore(self, tmp_path: Path):
        """Init on new repo creates gitignore if missing"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        with patch.object(gm, "is_repo", return_value=False), \
             patch.object(gm, "_run", return_value=_mock_run(0)):
            result = gm.init()
            assert result is True
            assert gm.gitignore_file.exists()


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
            elif args[0] == "commit":
                return _mock_run(0)
            elif args[0] == "rev-parse":
                return _mock_run(0, stdout="abc123\n")
            return _mock_run(0)

        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.add_commit("msg1", files=["file1.py", "file2.py"])
            assert result == 1
            # Check that add was called with specific files
            assert calls[0] == ["add", "file1.py", "file2.py"]

    def test_add_commit_commit_fails(self, tmp_path: Path):
        """Commit operation fails"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        def mock_run(args, **kwargs):
            if args[0] == "add":
                return _mock_run(0)
            elif args[0] == "commit":
                return _mock_run(1)
            return _mock_run(0)

        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.add_commit("msg1")
            assert result is False


class TestIsRepo:
    """Additional tests for is_repo"""

    def test_is_repo_exception(self, tmp_path: Path):
        """Exception during is_repo check"""
        gm = GitManager(str(tmp_path))

        with patch.object(gm, "_run", side_effect=OSError("Error")):
            result = gm.is_repo()
            assert result is False


class TestCountCheckpoints:
    """Additional tests for count_checkpoints"""

    def test_count_checkpoints_with_file(self, tmp_path: Path):
        """Count from existing file"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        content = {"msg1": "hash1", "msg2": "hash2", "init": "hash0"}
        gm.checkpoints_file.write_text(json.dumps(content))

        result = gm.count_checkpoints()
        assert result == 3

    def test_count_checkpoints_file_invalid_json(self, tmp_path: Path):
        """Invalid JSON raises JSONDecodeError"""
        gm = GitManager(str(tmp_path))
        gm.checkpoints_file.parent.mkdir(parents=True, exist_ok=True)

        gm.checkpoints_file.write_text("invalid json")

        with pytest.raises(json.JSONDecodeError):
            gm.count_checkpoints()
        assert True  # pytest.raises verifies the exception


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
            # Should only keep init and msg1
            assert result == 2

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
            # msg1 is before, msg3 is after
            # unknown_msg is excluded from classification (not in all_ids)
            # After rollback: init + msg1 + unknown_msg remain (msg3 removed)
            assert result == 3

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
            # Should keep init and compound key
            assert result == 2


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
            # With fork_index=-1, everything is "at_or_after"
            # So no before, has after -> rollback to init
            assert result == 1


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
            # Should keep init and msg1 only
            assert result == 2


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
            elif args[0] == "commit":
                return _mock_run(0)
            elif args[0] == "rev-parse":
                return _mock_run(0, stdout="def111\n")
            return _mock_run(0)

        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.add_commit("msg1")
            # Should have 2 entries: existing_msg + msg1
            assert result == 2
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

        with patch.object(gm, "_run", side_effect=raise_error):
            result = gm.rollback(["msg1", "msg2"], ["msg1", "msg2"])
            assert result is False


# ============================================================================
# Lines 204-205: rollback else branch (no before, no after)
# ============================================================================


class TestRollbackElseBranch:
    """Cover lines 204-205: rollback else branch, no before and no after."""

    def test_rollback_no_before_no_after(self, tmp_path: Path):
        """Lines 204-205: neither before nor after checkpoints, returns count."""
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
            # else branch: returns len(checkpointer_dict) = 1 (only "init")
            assert result == 1


# ============================================================================
# Cross-session rollback tests (mocked _run)
# ============================================================================


