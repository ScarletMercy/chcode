#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path
import json
import shutil


class GitManager:
    """影子 git 检查点管理器：仓库在 .chat/cp-repo，不碰用户 .git"""

    SHADOW_EXCLUDE = (
        ".chat/\n.git/\n.venv/\n__pycache__/\n*.pyc\nnode_modules/\n"
        "dist/\nbuild/\n.pytest_cache/\n.coverage\n*.egg-info/\n"
    )

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.cp_repo_dir = self.repo_path / ".chat" / "cp-repo"
        self.checkpoints_file = self.cp_repo_dir / "checkpoints.json"

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
                ["git", f"--git-dir={self.cp_repo_dir}"] + args,
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

    def init_shadow(self) -> bool:
        """初始化影子 git 仓库，幂等：已存在且有效则跳过，半残则删除重建。

        git-dir 在非 .git 路径下 init 默认是 bare；翻 core.bare=false 后以 cwd 为
        工作树（_run 的 cwd 即 repo_path），不写 core.worktree，fork 后天然正确。
        """
        valid = (
            self.cp_repo_dir.exists()
            and self._run(["rev-parse", "--verify", "HEAD"]).returncode == 0
        )
        if not valid:
            # 目录在但 HEAD 无效（如 fork 复制中断）视为半残，删除重建
            if self.cp_repo_dir.exists():
                shutil.rmtree(self.cp_repo_dir, ignore_errors=True)
            self.cp_repo_dir.parent.mkdir(parents=True, exist_ok=True)
            if self._run(["init"]).returncode != 0:
                return False
            # 非 bare 才能用 cwd 当工作树；不继承用户 git 配置
            self._run(["config", "--local", "core.bare", "false"])
            self._run(["config", "--local", "user.name", "chcode"])
            self._run(["config", "--local", "user.email", "chcode@local"])
            self._run(["config", "--local", "core.autocrlf", "false"])
            exclude_file = self.cp_repo_dir / "info" / "exclude"
            exclude_file.parent.mkdir(parents=True, exist_ok=True)
            exclude_file.write_text(self.SHADOW_EXCLUDE, encoding="utf-8")
            # add 后 commit：有文件则 init 含项目初始文件，空目录则 --allow-empty 兜底空提交
            self._run(["add", "."])
            self._run(["commit", "-m", "init", "--allow-empty"])
        self._ensure_init_checkpoint()
        return self._run(["rev-parse", "--verify", "HEAD"]).returncode == 0

    def _ensure_init_checkpoint(self) -> None:
        """确保 checkpoints.json 中存在 "init" 条目，供 rollback 使用"""
        if not self.checkpoints_file.exists():
            self.checkpoints_file.write_text(
                json.dumps({}, indent=4), encoding="utf-8"
            )
        data = json.loads(self.checkpoints_file.read_text(encoding="utf-8"))
        if "init" in data:
            return
        hash_result = self._run(["rev-list", "--max-parents=0", "HEAD"])
        if hash_result.returncode == 0 and hash_result.stdout.strip():
            data["init"] = hash_result.stdout.strip().split("\n")[-1]
            self.checkpoints_file.write_text(
                json.dumps(data, indent=4), encoding="utf-8"
            )

    def add_commit(self, message_ids: str, files: list | None = None) -> bool | int:
        """添加文件并提交"""
        if files is None:
            files = ["."]
        if self._run(["add"] + files).returncode != 0:
            return False

        existing: dict = {}
        if self.checkpoints_file.exists():
            existing = json.loads(self.checkpoints_file.read_text(encoding="utf-8"))

        # CP# 取写入前的检查点数，回滚后不跳号
        commit_msg = f"{message_ids} (CP#{len(existing)})"
        commit_result = self._run(["commit", "-m", commit_msg])

        if commit_result.returncode == 0:
            # 获取提交ID
            hash_result = self._run(["rev-parse", "HEAD"])
            if hash_result.returncode == 0:
                commit_id = hash_result.stdout.strip()

                checkpoint_dict = {message_ids: commit_id, **existing}
                count = len(checkpoint_dict)
                self.checkpoints_file.write_text(
                    json.dumps(checkpoint_dict, indent=4), encoding="utf-8"
                )

                return count
        return False

    def rollback(self, message_ids: list[str], all_ids: list[str]) -> bool | int:
        """回滚到指定检查点
        第一步：检查是否存在精确匹配（存在于JSON中有对应提交的ID），如果有则直接回溯到其上一次提交
        第二步：如果没有精确匹配，才进入模糊匹配逻辑，按以下三种情况进行处理：
        前有提交后有提交：直接回溯到前面最近的提交
        前无提交：回溯到初始提交
        前有提交后无提交：不回溯
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

            unknown_idx = -1
            for k in list(checkpointer_dict.keys()):
                if k == "init":
                    continue
                first_msg_id = k.split("&")[0]
                if first_msg_id not in all_ids:
                    before.append((unknown_idx, k))
                    unknown_idx -= 1
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
