"""
技能管理 — 扫描/列表/查看详情/安装/删除，全部用下拉列表交互
"""

from __future__ import annotations

from pathlib import Path

from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from chcode.display import console
from chcode.i18n import t
from chcode.prompts import select, confirm, text
from chcode.utils.skill_loader import (
    scan_all_skills,
    validate_skill_package,
    install_skill,
)


async def manage_skills(workplace_path: Path) -> None:
    """技能管理主菜单"""
    view_label = t("skill.view_installed")
    install_label = t("skill.install_new")
    back_label = t("common.back")
    while True:
        action = await select(
            t("skill.menu"),
            [view_label, install_label, back_label],
        )
        if action is None or action == back_label:
            return

        if action == view_label:
            await _list_skills(workplace_path)
        elif action == install_label:
            await _install_skill(workplace_path)


def _skill_type_label(type_value: str) -> str:
    """技能类型 → 当前语言显示文案。未知类型原样返回。"""
    translated = t(f"skill.type.{type_value}")
    return translated if translated != f"skill.type.{type_value}" else type_value


async def _list_skills(workplace_path: Path) -> None:
    """列出所有已安装技能，支持下拉选择操作"""
    skills = scan_all_skills(workplace_path)
    if not skills:
        console.print(f"[yellow]{t('skill.none_installed')}[/yellow]")
        return

    # 构建表格
    table = Table(title=t("skill.installed_table_title"))
    table.add_column(t("skill.col_name"), style="cyan")
    table.add_column(t("skill.col_type"), style="green")
    table.add_column(t("skill.col_desc"), style="white")
    table.add_column(t("skill.col_path"), style="dim")
    for s in skills:
        desc = s["description"]
        if len(desc) > 60:
            desc = desc[:57] + "..."
        table.add_row(s["name"], _skill_type_label(s["type"]), desc, str(s["path"]))
    console.print(table)

    # 选择操作（选项中类型用翻译文案；回选用 split(" (")[0] 取回名称）
    back_label = t("common.back")
    names = [f"{s['name']} ({_skill_type_label(s['type'])})" for s in skills]
    action = await select(
        t("skill.select_to_operate"),
        names + [back_label],
    )
    if action is None or action == back_label:
        return

    # 找到选中的技能
    selected_name = action.split(" (")[0]
    skill = next((s for s in skills if s["name"] == selected_name), None)
    if not skill:
        return

    view_detail_label = t("skill.view_detail")
    delete_label = t("skill.delete")
    op = await select(
        t("skill.action_on", name=skill["name"]),
        [view_detail_label, delete_label, back_label],
    )
    if op == view_detail_label:
        await _show_skill_detail(skill)
    elif op == delete_label:
        await _delete_skill(skill)
    elif op == back_label:
        return


async def _show_skill_detail(skill: dict) -> None:
    """查看技能详情"""
    skill_md = Path(skill["path"]) / "SKILL.md"
    if not skill_md.exists():
        console.print(f"[red]{t('skill.skill_file_not_exist')}[/red]")
        return

    content = skill_md.read_text(encoding="utf-8")
    console.print(
        Panel(
            Markdown(content),
            title=t("skill.detail_title", name=skill["name"]),
            border_style="cyan",
            padding=(1, 2),
        )
    )


async def _delete_skill(skill: dict) -> None:
    """删除技能"""
    ok = await confirm(
        t("skill.delete_confirm", name=skill["name"]), default=False
    )
    if not ok:
        return

    import shutil

    skill_path = Path(skill["path"])
    try:
        shutil.rmtree(skill_path)
        console.print(f"[green]{t('skill.deleted', name=skill['name'])}[/green]")
    except Exception as e:
        console.print(f"[red]{t('skill.delete_failed', error=e)}[/red]")


async def _install_skill(workplace_path: Path) -> None:
    """安装技能"""
    file_path = await text(t("skill.input_archive_path"))
    if not file_path:
        return

    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]{t('skill.file_not_exist')}[/red]")
        return

    # 验证
    console.print(f"[yellow]{t('skill.validating')}[/yellow]")
    skill_info = validate_skill_package(str(path))
    if not skill_info:
        console.print(f"[red]{t('skill.invalid_package')}[/red]")
        return

    # 选择安装位置（语言无关：按索引判定项目级/全局级）
    project_label = t("skill.install_project")
    choices = [project_label, t("skill.install_global")]
    location = await select(
        t("skill.select_install_location"),
        choices,
    )
    if location is None:
        return

    if choices.index(location) == 0:
        install_path = workplace_path / ".chat" / "skills"
    else:
        install_path = Path.home() / ".chat" / "skills"

    install_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[yellow]{t('skill.installing')}[/yellow]")
    if install_skill(str(path), install_path):
        name = skill_info["name"]
        console.print(f"[green]{t('skill.install_success', name=name)}[/green]")
    else:
        console.print(f"[red]{t('skill.install_failed')}[/red]")
