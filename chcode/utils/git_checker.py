#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git可用性检查工具
用于判断系统中Git是否可用
"""
import subprocess
from typing import Tuple, Optional


def check_git_availability() -> Tuple[bool, str, Optional[str]]:
    """
    检查Git是否可用

    Returns:
        Tuple[bool, str, Optional[str]]: (是否可用, 状态描述, Git版本信息)
    """
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            version_info = result.stdout.strip()
            return True, "Git可用", version_info
        else:
            error_msg = result.stderr.strip() if result.stderr else "未知错误"
            return False, f"Git命令执行失败: {error_msg}", None

    except FileNotFoundError:
        return False, "未找到Git命令，请确保Git已安装并添加到PATH环境变量中", None
    except subprocess.TimeoutExpired:
        return False, "Git命令执行超时", None
    except Exception as e:
        return False, f"检查Git时发生异常: {str(e)}", None
