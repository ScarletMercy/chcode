"""
技能管理 — 扫描/列表/查看详情/安装/删除，全部用下拉列表交互
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from chcode.prompts import select, confirm, text
from chcode.utils.skill_loader import (
    scan_all_skills,
    validate_skill_package,
    install_skill,
)

if TYPE_CHECKING:
    from chcode.session import SessionManager

console = Console()


async def manage_skills(session: SessionManager) -> None:
    """技能管理主菜单"""
    while True:
        action = await select(
            "技能管理:",
            ["查看已安装技能", "安装新技能", "返回"],
        )
        if action is None or action == "返回":
            return

        if action == "查看已安装技能":
            await _list_skills(session)
        elif action == "安装新技能":
            await _install_skill(session)


async def _list_skills(session: SessionManager) -> None:
    """列出所有已安装技能，支持下拉选择操作"""
    skills = scan_all_skills(session.workplace_path)
    if not skills:
        console.print("[yellow]没有发现已安装的技能[/yellow]")
        return

    # 构建表格
    table = Table(title="已安装技能")
    table.add_column("名称", style="cyan")
    table.add_column("类型", style="green")
    table.add_column("描述", style="white")
    table.add_column("路径", style="dim")
    for s in skills:
        desc = s["description"]
        if len(desc) > 60:
            desc = desc[:57] + "..."
        table.add_row(s["name"], s["type"], desc, str(s["path"]))
    console.print(table)

    # 选择操作
    names = [f"{s['name']} ({s['type']})" for s in skills]
    action = await select(
        "选择技能进行操作:",
        names + ["返回"],
    )
    if action is None or action == "返回":
        return

    # 找到选中的技能
    selected_name = action.split(" (")[0]
    skill = next((s for s in skills if s["name"] == selected_name), None)
    if not skill:
        return

    op = await select(
        f"对技能 '{skill['name']}' 的操作:",
        ["查看详情", "删除技能", "返回"],
    )
    if op == "查看详情":
        await _show_skill_detail(skill)
    elif op == "删除技能":
        await _delete_skill(skill, session)
    elif op == "返回":
        return


async def _show_skill_detail(skill: dict) -> None:
    """查看技能详情"""
    skill_md = Path(skill["path"]) / "SKILL.md"
    if not skill_md.exists():
        console.print("[red]技能文件不存在[/red]")
        return

    content = skill_md.read_text(encoding="utf-8")
    console.print(
        Panel(
            Markdown(content),
            title=f"技能: {skill['name']}",
            border_style="cyan",
            padding=(1, 2),
        )
    )


async def _delete_skill(skill: dict, session: SessionManager) -> None:
    """删除技能"""
    ok = await confirm(
        f"确定删除技能 '{skill['name']}'？此操作不可撤销！", default=False
    )
    if not ok:
        return

    import shutil

    skill_path = Path(skill["path"])
    try:
        shutil.rmtree(skill_path)
        console.print(f"[green]技能 '{skill['name']}' 已删除[/green]")
    except Exception as e:
        console.print(f"[red]删除失败: {e}[/red]")


async def _install_skill(session: SessionManager) -> None:
    """安装技能"""
    file_path = await text("输入技能压缩包路径 (.zip/.tar.gz/.tgz):")
    if not file_path:
        return

    path = Path(file_path)
    if not path.exists():
        console.print("[red]文件不存在[/red]")
        return

    # 验证
    console.print("[yellow]验证技能包...[/yellow]")
    skill_info = validate_skill_package(str(path))
    if not skill_info:
        console.print("[red]无效的技能包，必须包含 SKILL.md[/red]")
        return

    # 选择安装位置
    location = await select(
        "选择安装位置:",
        ["项目级 (当前工作目录)", "全局级 (用户目录)"],
    )
    if location is None:
        return

    if "项目级" in location:
        install_path = session.workplace_path / ".chat" / "skills"
    else:
        install_path = Path.home() / ".chat" / "skills"

    install_path.mkdir(parents=True, exist_ok=True)

    console.print("[yellow]安装中...[/yellow]")
    if install_skill(str(path), install_path):
        name = skill_info["name"]
        console.print(f"[green]技能 '{name}' 安装成功！[/green]")
    else:
        console.print("[red]安装失败[/red]")
