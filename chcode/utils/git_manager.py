#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path
import json
import shutil
import os

from chcode.display import render_warning
from chcode.i18n import t


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
        try:
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
                self._ensure_exclude()
                # add 后 commit：有文件则 init 含项目初始文件，空目录则 --allow-empty 兜底空提交
                self._run(["add", "."])
                self._run(["commit", "-m", "init", "--allow-empty"])
            self._ensure_init_checkpoint()
            return self._run(["rev-parse", "--verify", "HEAD"]).returncode == 0
        except RuntimeError as e:
            render_warning(t("chat.git.shadow_init_exception", error=e))
            return False
        except OSError as e:
            render_warning(t("chat.git.shadow_init_exception", error=e))
            return False

    def migrate_legacy_git(self) -> None:
        """【旧版检查点迁移】旧版把检查点提交进用户真实 .git（污染用户仓库），新版
        改用独立影子仓库 .chat/cp-repo。检测到 .git/checkpoints.json 且无 cp-repo 时，
        复制 .git -> cp-repo 并规范化 config/exclude/hooks。仅在 init_shadow 前调用一次；
        失败则清理半残 cp-repo，由随后的 init_shadow 全新建仓兜底。"""
        if self.cp_repo_dir.exists():
            return
        legacy_git = self.repo_path / ".git"
        if not legacy_git.is_dir() or not (legacy_git / "checkpoints.json").exists():
            return
        try:
            self.cp_repo_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(legacy_git, self.cp_repo_dir)
            # 副本沿用用户原 config：无 chcode 身份（全局无 user 时 commit 失败），
            # 且可能含 core.worktree 导致 --git-dir 指错工作树，故统一覆写/清理
            self._run(["config", "--local", "core.bare", "false"])
            self._run(["config", "--local", "user.name", "chcode"])
            self._run(["config", "--local", "user.email", "chcode@local"])
            self._run(["config", "--local", "core.autocrlf", "false"])
            self._run(["config", "--local", "--unset", "core.worktree"])  # 键不存在 rc=5，silent 吞掉
            self._ensure_exclude()
            self._clean_hooks()
        except (OSError, RuntimeError) as e:
            # 半残 cp-repo 会让后续 init_shadow 误判有效而跳过补全，清理后由下方全新建仓兜底
            shutil.rmtree(self.cp_repo_dir, ignore_errors=True)
            render_warning(t("chat.git.migrate_copy_failed", error=e))

    def _clean_hooks(self) -> None:
        """【迁移】删复制带过来的用户活动 hook（.sample 除外），防 chcode commit 误触发。"""
        hooks_dir = self.cp_repo_dir / "hooks"
        if not hooks_dir.is_dir():
            return
        for hook in hooks_dir.iterdir():
            if hook.is_file() and not hook.name.endswith(".sample"):
                try:
                    hook.unlink()
                except OSError:
                    pass

    def _write_checkpoints(self, data: dict) -> None:
        tmp = self.checkpoints_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=4), encoding="utf-8")
            os.replace(tmp, self.checkpoints_file)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def _ensure_init_checkpoint(self) -> None:
        """确保 checkpoints.json 中存在 "init" 条目，供 rollback 使用"""
        if not self.checkpoints_file.exists():
            self._write_checkpoints({})
        try:
            data = json.loads(self.checkpoints_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            render_warning(t("chat.git.checkpoint_json_corrupt"))
            return
        except OSError as e:
            render_warning(t("chat.git.checkpoint_read_failed", error=e))
            return
        if "init" in data:
            return
        hash_result = self._run(["rev-list", "--max-parents=0", "HEAD"])
        if hash_result.returncode == 0 and hash_result.stdout.strip():
            data["init"] = hash_result.stdout.strip().split("\n")[-1]
            self._write_checkpoints(data)

    def _ensure_exclude(self) -> None:
        """add 前补写 info/exclude：丢失则 git 会把 cp-repo 自身暂存进历史，rollback 时 reset --hard 删 git-dir。失败抛 OSError 由调用方 except 兜底（不在此静默，否则 add 裸奔）"""
        exclude_file = self.cp_repo_dir / "info" / "exclude"
        if not exclude_file.exists() or exclude_file.read_text(encoding="utf-8") != self.SHADOW_EXCLUDE:
            exclude_file.parent.mkdir(parents=True, exist_ok=True)
            exclude_file.write_text(self.SHADOW_EXCLUDE, encoding="utf-8")

    def _undo_commit(self, pre_head: str) -> None:
        """撤回刚生成的 commit，尽力回退 HEAD 到 pre_head，不抛异常。"""
        # mixed 失败(rc≠0 或异常)退 --soft：soft 只移 HEAD、不写 index，比 mixed 快得多，
        # mixed 超时(写整个 index)时 soft 仍可能成
        if self._reset_ok("--mixed", pre_head) or self._reset_ok("--soft", pre_head):
            return
        # 两者都败：commit 未撤成，留孤儿
        render_warning(t("chat.git.undo_failed"))

    def _reset_ok(self, mode: str, pre_head: str) -> bool:
        try:
            return self._run(["reset", mode, pre_head]).returncode == 0
        except Exception:
            return False

    def add_commit(self, message_ids: str, files: list | None = None) -> bool:
        """True=已提交或无改动；False=失败（调用方告警）。"""
        if files is None:
            files = ["."]
        pre_head = ""
        committed = False
        try:
            self._ensure_exclude()
            add_result = self._run(["add"] + files)
            if add_result.returncode != 0:
                render_warning(t("chat.git.checkpoint_failed", error=add_result.stderr.strip()))
                return False
            # diff --cached --quiet: 0=无差异, 1=有差异, 其他=git 出错
            diff_result = self._run(["diff", "--cached", "--quiet"])
            if diff_result.returncode == 0:
                return True
            if diff_result.returncode != 1:
                render_warning(t("chat.git.checkpoint_failed", error=diff_result.stderr.strip()))
                return False
            # HEAD 必存在（init_shadow 已建立），故不校验 returncode
            pre_head = self._run(["rev-parse", "HEAD"]).stdout.strip()
            existing: dict = {}
            if self.checkpoints_file.exists():
                try:
                    existing = json.loads(self.checkpoints_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    render_warning(t("chat.git.checkpoint_json_corrupt"))
                    return False
                except OSError as e:
                    render_warning(t("chat.git.checkpoint_read_failed", error=e))
                    return False
            commit_result = self._run(["commit", "-m", message_ids])
            if commit_result.returncode != 0:
                render_warning(t("chat.git.checkpoint_failed", error=commit_result.stderr.strip()))
                return False
            committed = True
            hash_result = self._run(["rev-parse", "HEAD"])
            if hash_result.returncode != 0:
                render_warning(t("chat.git.checkpoint_failed", error=hash_result.stderr.strip()))
                self._undo_commit(pre_head)
                return False
            commit_id = hash_result.stdout.strip()
        except (RuntimeError, OSError) as e:
            if committed:
                self._undo_commit(pre_head)
            render_warning(t("chat.git.checkpoint_failed", error=e))
            return False
        try:
            # existing = 此前各轮写入的检查点。message_ids 由本轮 human 消息 ID 拼成，每轮唯一、
            # 不会已在 existing 中，故 {message_ids: ..., **existing} 不存在旧值覆盖新值的问题。
            self._write_checkpoints({message_ids: commit_id, **existing})
        except OSError as e:
            # 撤 commit 以保 json 与 commit 一致；只捕 OSError（data 均为 str，json.dumps 不抛其它）
            render_warning(t("chat.git.checkpoint_failed", error=e))
            self._undo_commit(pre_head)
            return False
        return True

    def rollback(self, message_ids: list[str], all_ids: list[str]) -> bool:
        """回滚到指定检查点
        第一步：检查是否存在精确匹配（存在于JSON中有对应提交的ID），如果有则直接回溯到其上一次提交
        第二步：如果没有精确匹配，才进入模糊匹配逻辑，按以下三种情况进行处理：
        前有提交后有提交：直接回溯到前面最近的提交
        前无提交：回溯到初始提交
        前有提交后无提交：不回溯
        """
        if not self.checkpoints_file.exists():
            render_warning(t("chat.git.rollback_no_checkpoints"))
            return False

        try:
            json_data = self.checkpoints_file.read_text(encoding="utf-8")
            checkpointer_dict: dict = json.loads(json_data)
        except json.JSONDecodeError:
            render_warning(t("chat.git.checkpoint_json_corrupt"))
            return False
        except OSError as e:
            render_warning(t("chat.git.checkpoint_read_failed", error=e))
            return False

        message_ids_str = "&".join(message_ids)

        # -- 辅助：根据 all_ids 位置将非 init 的 checkpoint 分为 before / at_or_after --
        def _classify_checkpoint_keys():
            before = []
            at_or_after = []
            # message_ids 非空由调用方保证：edit/fork 已校验选中组含 human，分组结构每组亦非空
            fork_id = message_ids[0]
            fork_index = all_ids.index(fork_id) if fork_id in all_ids else -1

            for k in list(checkpointer_dict.keys()):
                if k == "init":
                    continue
                first_msg_id = k.split("&")[0]
                if first_msg_id not in all_ids:
                    # orphan(跨会话键或已删轮次)：保留不删，但不参与分类，
                    # 否则成唯一 before 键会把 Case 2(回溯 init)翻成 Case 1(回溯到它)
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

            try:
                reset_result = self._run(["reset", "--hard", aim_id])
                if reset_result.returncode == 0:
                    self._write_checkpoints(checkpointer_dict)
                    return True
                else:
                    render_warning(
                        t("chat.git.rollback_failed", error=reset_result.stderr.strip())
                    )
                    return False
            except RuntimeError as e:
                render_warning(t("chat.git.rollback_failed", error=e))
                return False
            except OSError as e:
                render_warning(t("chat.git.rollback_failed", error=e))
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
            # Case 2：前无提交后有提交 -> 回溯到初始提交（orphan 跳过后无合法 before 也落此分支）
            aim_id = checkpointer_dict.get("init")
            if aim_id is None:
                render_warning(t("chat.git.rollback_no_init"))
                return False  # init 缺失（启动时 rev-list 失败未回填）-> 无法回溯，保持原状
            for k in at_or_after_keys:
                checkpointer_dict.pop(k)

        elif has_before and not has_after:
            # Case 3：前有提交后无提交 -> 不回溯
            return True
        else:
            return True

        try:
            reset_result = self._run(["reset", "--hard", aim_id])
            if reset_result.returncode == 0:
                self._write_checkpoints(checkpointer_dict)
                return True
            else:
                render_warning(
                    t("chat.git.rollback_failed", error=reset_result.stderr.strip())
                )
                return False
        except RuntimeError as e:
            render_warning(t("chat.git.rollback_failed", error=e))
            return False
        except OSError as e:
            render_warning(t("chat.git.rollback_failed", error=e))
            return False
