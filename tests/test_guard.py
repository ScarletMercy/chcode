import pytest

from chcode.utils.shell.guard import check_command


# (命令, 期望类别) — 应被拦截的危险命令
BLOCKED_CASES = [
    # 递归删除 — bash
    ("rm -rf /", "recursive_delete"),
    ("rm -fr /tmp", "recursive_delete"),
    ("rm -r --force dir", "recursive_delete"),
    ("rm --recursive dir", "recursive_delete"),
    ("sudo rm -rf /home", "recursive_delete"),
    # 递归删除 — powershell
    ("Remove-Item -Recurse -Force ./dir", "recursive_delete"),
    ("ri -Recurse foo", "recursive_delete"),
    # 递归删除 — cmd
    ("rmdir /s /q dir", "recursive_delete"),
    ("rd /s /q dir", "recursive_delete"),
    ("del /s /q *.tmp", "recursive_delete"),
    ("erase /s *.bak", "recursive_delete"),
    ("cmd /c rmdir /s /q dir", "recursive_delete"),
    # 关机 / 重启 — 命令位置
    ("shutdown /s /t 0", "shutdown"),
    ("shutdown -h now", "shutdown"),
    ("sudo shutdown -h now", "shutdown"),
    ("doas reboot", "shutdown"),
    ("Stop-Computer", "shutdown"),
    ("Restart-Computer -Force", "shutdown"),
    ("halt", "shutdown"),
    ("poweroff", "shutdown"),
    ("reboot", "shutdown"),
    ("init 0", "shutdown"),
    ("init 6", "shutdown"),
    # 关机 — systemctl 子命令形式（危险词作为参数）
    ("systemctl poweroff", "shutdown"),
    ("systemctl reboot", "shutdown"),
    ("systemctl halt", "shutdown"),
    # 链式命令：危险段在第二个仍应拦截
    ("echo hi && shutdown now", "shutdown"),
    ("echo hi; reboot", "shutdown"),
    # 强制结束进程 — cmd
    ("taskkill /F /IM notepad.exe", "force_kill"),
    ("taskkill /f /pid 1234", "force_kill"),
    ("taskkill /T /PID 1234", "force_kill"),
    # 强制结束进程 — bash
    ("kill -9 1234", "force_kill"),
    ("kill -KILL 1234", "force_kill"),
    ("killall -9 python", "force_kill"),
    ("pkill -9 -f chrome", "force_kill"),
    # 强制结束进程 — powershell
    ("Stop-Process -Name chrome -Force", "force_kill"),
    # 破坏性系统操作
    ("mkfs.ext4 /dev/sda1", "system_damage"),
    ("dd if=/dev/zero of=/dev/sda", "system_damage"),
    ("dd if=ubuntu.iso of=/dev/sdb", "system_damage"),
    ("echo x > /dev/sda", "system_damage"),
    (":(){ :|:& };:", "system_damage"),
    ("format C:", "system_damage"),
]

# 应放行的普通命令
SAFE_CASES = [
    "",
    "   ",
    "echo hello",
    "ls -la",
    "rm file.txt",            # 非递归删除
    "rm -f file.txt",         # 仅强制、无递归
    "rm --verbose file.txt",  # 长标志含 r 但非递归
    "kill 1234",              # 普通终止信号
    "kill -15 1234",          # SIGTERM，默认优雅终止
    "kill -19 1234",          # SIGSTOP，含 9 但非 SIGKILL
    "git status",
    "npm run build",
    "python -m pytest",
    "tasklist",               # 仅列出进程
    "Get-Process",            # 仅列出进程
    "cd /tmp && pwd",
    "grep -r foo .",          # grep 的 -r 不是 rm
    "grep --recursive foo .", # --recursive 锚定到 rm，grep 不误伤
    "cp --recursive src dst", # 同上
    "format-string",          # 子串不应误伤
    "skill list",             # 含 kill 子串但非 kill 命令
    "prmsg",                  # 含 rm 子串
    # dd 写入普通文件是安全操作，不应误伤
    "dd if=/dev/zero of=/tmp/file bs=1M",
    "dd if=backup.img of=/home/user/disk.img",
    # 相对路径调用同名工具 —— 视为用户项目内的文件/脚本，豁免
    "./reboot",
    "./shutdown -h now",
    "./reboot.sh",
    "./mkfs.ext4",
    "subdir/reboot",
    "../shutdown",
    ".\\reboot.bat",              # Windows 相对路径
    "scripts\\shutdown.ps1",      # Windows 反斜杠相对路径
    # init 切换非关机运行级别
    "init 3",
    # systemctl 非关机子命令
    "systemctl status",
    "systemctl restart nginx",
    # 危险词出现在文件名/参数里 —— 必须放行（修复裸命令词误报）
    "cat shutdown.log",
    "grep reboot /var/log/syslog",
    "ls reboot backup.txt",
    "echo shutdown.log",
    "type shutdown.log",                 # cmd 的 type（即 cat）
    "Select-String reboot syslog.txt",   # powershell 的 grep
    "cat mkfs.txt",
    "ls halt.notes",
    "grep poweroff /var/log/messages",
    # 链式命令：两段都是安全的
    "echo hi && ls -la",
    "cd /tmp; pwd",
]


@pytest.mark.parametrize("command,expected_category", BLOCKED_CASES)
def test_blocked_commands(command, expected_category):
    result = check_command(command)
    assert result.blocked is True
    assert result.category == expected_category


@pytest.mark.parametrize("command", SAFE_CASES)
def test_safe_commands(command):
    result = check_command(command)
    assert result.blocked is False
