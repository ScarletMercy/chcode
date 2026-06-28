#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git可用性检查工具
用于判断系统中Git是否可用
"""
import subprocess


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
            return True, "Git可用", version_info
        else: # 非0值表示执行失败
            error_msg = result.stderr.strip() if result.stderr else "未知错误"
            return False, f"Git命令执行失败: {error_msg}", None

    except FileNotFoundError:
        return False, "未找到Git命令，请确保Git已安装并添加到PATH环境变量中", None
    except subprocess.TimeoutExpired:
        return False, "Git命令执行超时", None
    except Exception as e:
        return False, f"检查Git时发生异常: {str(e)}", None
