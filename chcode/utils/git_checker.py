#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git可用性检查工具
用于判断系统中Git是否可用
"""
import subprocess

from chcode.i18n import t


def check_git_availability() -> tuple[bool, str, str|None]:
    """
    检查Git是否可用

    Returns:
        tuple[bool, str, str|None]: (是否可用, 状态描述, Git版本信息)
    """
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0: #  0 表示命令执行成功
            version_info = result.stdout.strip()
            return True, t("git.available"), version_info
        else: # 非0值表示执行失败
            error_msg = result.stderr.strip() if result.stderr else t("git.unknown_error")
            return False, t("git.cmd_failed", error=error_msg), None

    except FileNotFoundError:
        return False, t("git.not_found"), None
    except subprocess.TimeoutExpired:
        return False, t("git.timeout"), None
    except Exception as e:
        return False, t("git.exception", error=str(e)), None
