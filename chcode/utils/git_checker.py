#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git可用性检查工具
用于判断系统中Git是否可用
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional


def check_git_availability() -> Tuple[bool, str, Optional[str]]:
    """
    检查Git是否可用

    Returns:
        Tuple[bool, str, Optional[str]]: (是否可用, 状态描述, Git版本信息)
    """
    try:
        # 尝试执行git --version命令
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=10  # 10秒超时
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


def check_git_in_path() -> bool:
    """
    检查Git是否在系统PATH中

    Returns:
        bool: Git是否在PATH中
    """
    try:
        # 在Windows上检查git.exe，在Unix/Linux上检查git
        git_executable = "git.exe" if sys.platform == "win32" else "git"

        # 遍历PATH环境变量中的所有目录
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)

        for path_dir in path_dirs:
            git_path = Path(path_dir) / git_executable
            if git_path.exists() and os.access(git_path, os.X_OK):
                return True

        return False
    except Exception:
        return False


def get_git_executable_path() -> Optional[str]:
    """
    获取Git可执行文件的完整路径

    Returns:
        Optional[str]: Git可执行文件路径，如果找不到则返回None
    """
    try:
        result = subprocess.run(
            ["where", "git"] if sys.platform == "win32" else ["which", "git"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            git_path = result.stdout.strip().split('\n')[0]  # 取第一个结果
            return git_path
        return None
    except Exception:
        return None


def check_git_repository(path: str = ".") -> Tuple[bool, str]:
    """
    检查指定路径是否为Git仓库

    Args:
        path (str): 要检查的路径，默认为当前目录

    Returns:
        Tuple[bool, str]: (是否为Git仓库, 状态描述)
    """
    try:
        repo_path = Path(path).resolve()

        # 执行git rev-parse --git-dir命令检查是否为Git仓库
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            git_dir = result.stdout.strip()
            return True, f"Git仓库路径: {git_dir}"
        else:
            return False, "不是Git仓库或Git不可用"

    except Exception as e:
        return False, f"检查Git仓库时发生异常: {str(e)}"


def main():
    """主函数 - 命令行测试"""
    import os

    print("Git可用性检查")
    print("=" * 50)

    # 1. 检查Git基本可用性
    is_available, status, version = check_git_availability()
    print(f"Git可用性: {'OK' if is_available else 'FAIL'} {status}")
    if version:
        print(f"Git版本: {version}")

    print()

    # 2. 检查Git是否在PATH中
    in_path = check_git_in_path()
    print(f"Git在PATH中: {'OK' if in_path else 'FAIL'}")

    # 3. 获取Git可执行文件路径
    git_path = get_git_executable_path()
    if git_path:
        print(f"Git路径: {git_path}")
    else:
        print("无法找到Git可执行文件路径")

    print()

    # 4. 检查当前目录是否为Git仓库
    is_repo, repo_status = check_git_repository()
    print(f"当前目录Git仓库状态: {'OK' if is_repo else 'FAIL'} {repo_status}")

    print()
    print("=" * 50)

    if is_available:
        print("Git环境配置正常！")
        return 0
    else:
        print("Git环境存在问题，请检查安装和配置")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
