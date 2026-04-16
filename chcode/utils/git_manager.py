#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path
from typing import Optional
import json


class GitManager:
    """增强版Git检查点管理器，支持.gitignore管理"""

    MINIMAL_GITIGNORE = ".git\n.chat\n.venv\n.gitignore\n__pycache__\n*.pyc\n.pytest_cache\n.coverage\n.pytest_cache/\n"

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.git_cmd = "git"
        self.checkpoints_file = self.repo_path / ".git" / "checkpoints.json"
        self.gitignore_file = self.repo_path / ".gitignore"
        self.current_id = 0
        self._is_repo: bool | None = None

    def _run(
        self, args: list, timeout: int = 30, silent: bool = True
    ) -> subprocess.CompletedProcess:
        """执行Git命令

        Args:
            args: Git 命令参数
            timeout: 超时时间（秒）
            silent: 是否静默输出（默认 True，不打印调试信息）
        """
        try:
            result = subprocess.run(
                [self.git_cmd] + args,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
            )

            if result.returncode != 0 and not silent:
                print(f"Git命令返回码: {result.returncode}")
                if result.stderr:
                    print(f" STDERR: {result.stderr.strip()}")
                if result.stdout:
                    print(f" STDOUT: {result.stdout.strip()}")

            return result
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Git命令超时（{timeout}秒）: git {' '.join(args)}")
        except Exception as e:
            raise RuntimeError(f"Git命令执行失败: {e}")

    def is_repo(self) -> bool:
        """检查是否为Git仓库"""
        if self._is_repo is not None:
            return self._is_repo
        try:
            self._is_repo = self._run(["rev-parse", "--git-dir"]).returncode == 0
            return self._is_repo
        except Exception:
            return False

    def init(self) -> bool:
        """初始化Git仓库"""
        if self.is_repo():
            if not self.checkpoints_file.exists():
                self.checkpoints_file.write_text(
                    json.dumps({}, indent=4), encoding="utf-8"
                )
            return False
        if not self.gitignore_file.exists():
            self.create_gitignore()
        result = self._run(["init"])
        if result.returncode == 0:
            # 初始空提交，确保后续 commit 不会因空仓库失败
            self._run(["commit", "-m", "init", "--allow-empty"])
        if not self.checkpoints_file.exists():
            self.checkpoints_file.write_text(json.dumps({}, indent=4), encoding="utf-8")
        return result.returncode == 0

    def add_commit(self, message_ids: str, files: list | None = None) -> bool | int:
        """添加文件并提交"""
        if files is None:
            files = ["."]
        if self._run(["add"] + files).returncode != 0:
            return False

        # 提交
        commit_msg = f"{message_ids} (CP#{self.current_id + 1})"
        commit_result = self._run(["commit", "-m", commit_msg])

        if commit_result.returncode == 0:
            # 获取提交ID
            hash_result = self._run(["rev-parse", "HEAD"])
            if hash_result.returncode == 0:
                commit_id = hash_result.stdout.strip()

                checkpoint_dict = {}
                checkpoint_dict[message_ids] = commit_id
                if self.checkpoints_file.exists():
                    checkpoint_dict.update(
                        json.loads(self.checkpoints_file.read_text(encoding="utf-8"))
                    )
                count = len(checkpoint_dict)
                self.checkpoints_file.write_text(
                    json.dumps(checkpoint_dict, indent=4), encoding="utf-8"
                )

                self.current_id += 1
                return count
        return False

    def rollback(self, message_ids: list[str], all_ids: list[str]) -> bool | int:
        """回滚到指定检查点
        第一步：检查是否存在精确匹配（存在于JSON中有对应提交的ID），如果有则直接回溯到其上一次提交
        第二步：如果没有精确匹配，才进入模糊匹配逻辑，按以下三种情况进行处理：
        前有提交后有提交：直接回溯到前面最近的提交
        前无提交：回溯到初始提交
        前有提交后无提交：不回溯，返回当前计数
        """
        if not self.checkpoints_file.exists():
            return False

        json_data = self.checkpoints_file.read_text(encoding="utf-8")
        checkpointer_dict: dict = json.loads(json_data)

        message_ids_str = "&".join(message_ids)

        # -- 辅助：根据 all_ids 位置将非 init 的 checkpoint 分为 before / at_or_after --
        def _classify_checkpoint_keys():
            before = []
            at_or_after = []
            fork_id = message_ids[0]
            fork_index = all_ids.index(fork_id) if fork_id in all_ids else -1

            for k in list(checkpointer_dict.keys()):
                if k == "init":
                    continue
                first_msg_id = k.split("&")[0]
                if first_msg_id not in all_ids:
                    continue
                idx = all_ids.index(first_msg_id)
                if idx < fork_index:
                    before.append((idx, k))
                else:
                    at_or_after.append(k)

            before.sort(key=lambda x: x[0])
            return before, at_or_after

        # -- 第一步：精确匹配 --
        if message_ids_str in checkpointer_dict:
            aim_id = checkpointer_dict[message_ids_str] + "~1"

            _, keys_to_remove = _classify_checkpoint_keys()
            keys_to_remove_set = set(keys_to_remove)
            keys_to_remove_set.add(message_ids_str)

            for k in keys_to_remove_set:
                checkpointer_dict.pop(k, None)

            count = len(checkpointer_dict)

            try:
                reset_result = self._run(["reset", "--hard", aim_id])
                if reset_result.returncode == 0:
                    self.checkpoints_file.write_text(
                        json.dumps(checkpointer_dict, indent=4), encoding="utf-8"
                    )
                    return count
                else:
                    return False
            except Exception:
                return False

        # -- 第二步：模糊匹配 --
        before_keys, at_or_after_keys = _classify_checkpoint_keys()

        has_before = len(before_keys) > 0
        has_after = len(at_or_after_keys) > 0

        if has_before and has_after:
            # Case 1：前有提交后有提交 -> 回溯到前面最近的提交（保留该提交本身的状态）
            latest_before_key = before_keys[-1][1]
            aim_id = checkpointer_dict[latest_before_key]

            for k in at_or_after_keys:
                checkpointer_dict.pop(k)

        elif not has_before and has_after:
            # Case 2：前无提交后有提交 -> 回溯到初始提交
            aim_id = checkpointer_dict["init"]

            for k in at_or_after_keys:
                checkpointer_dict.pop(k)

        elif has_before and not has_after:
            # Case 3：前有提交后无提交 -> 不回溯
            count = len(checkpointer_dict)
            return count
        else:
            count = len(checkpointer_dict)
            return count

        count = len(checkpointer_dict)

        try:
            reset_result = self._run(["reset", "--hard", aim_id])
            if reset_result.returncode == 0:
                self.checkpoints_file.write_text(
                    json.dumps(checkpointer_dict, indent=4), encoding="utf-8"
                )
                return count
            else:
                return False
        except Exception:
            return False

    def count_checkpoints(self, count: int | None = None) -> int:
        """统计检查点数量"""
        if count is None:
            if not self.checkpoints_file.exists():
                return 0
            json_data = self.checkpoints_file.read_text(encoding="utf-8")
            checkpointer_dict = json.loads(json_data)
            return len(checkpointer_dict)
        else:
            return count

    def create_gitignore(self, content: Optional[str] = None) -> bool:
        """创建.gitignore文件，屏蔽.git和.venv等"""
        try:
            if content is None:
                content = self.MINIMAL_GITIGNORE

            with open(self.gitignore_file, "w", encoding="utf-8") as f:
                f.write(content)

            self._run(["add", ".gitignore"])
            return True
        except Exception as e:
            print(f"创建.gitignore失败: {e}")
            return False
