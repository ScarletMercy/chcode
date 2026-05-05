"""
Skills 发现和加载器（ps: 自 https://github.com/NanmiCoder/skills-agent-proto 二次开发 ）

演示 Skills 三层加载机制的核心实现：
- Level 1: scan_skills() - 扫描并加载所有 Skills 元数据到 system prompt
- Level 2: load_skill(skill_name: str) - 根据skill name加载指定 Skill 的详细指令（只返回 instructions - skill.md文档)）
- Level 3: 由 bash tool 执行脚本（见 tools.py），大模型从指令中自己发现脚本

核心设计理念：
    让大模型成为真正的"智能体"，自己阅读指令、发现脚本、决定执行。
    代码层面不需要特殊处理脚本发现/执行逻辑。

Skills 目录结构：
    my-skill/
    ├── SKILL.md          # 必需：指令和元数据
    ├── scripts/          # 可选：可执行脚本
    ├── references/       # 可选：参考文档
    └── assets/           # 可选：模板和资源

SKILL.md 格式：
    ---
    name: skill-name
    description: 何时使用此 skill 的描述
    ---
    # Skill Title
    详细指令内容...
"""

import re
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import yaml


# 默认 Skills 搜索路径（项目级优先，用户级兜底）
DEFAULT_SKILL_PATHS = [
    Path.cwd() / ".chat" / "skills",  # 项目级 Skills (.chat/skills/) - 优先
    Path.home() / ".chat" / "skills",  # 用户级 Skills (~/.chat/skills/) - 兜底
]


@dataclass
class SkillMetadata:
    """
    Skill 元数据（Level 1）

    启动时从 YAML frontmatter 解析，用于注入 system prompt。
    每个 skill 约 100 tokens。
    """

    name: str  # skill 唯一名称
    description: str  # 何时使用此 skill 的描述
    skill_path: Path  # skill 目录路径

    def to_prompt_line(self) -> str:
        """生成 system prompt 中的单行描述"""
        return f"- **{self.name}**: {self.description}"


@dataclass
class SkillContent:
    """
    Skill 完整内容（Level 2）

    用户请求匹配时加载，包含 SKILL.md 的完整指令。
    约 5k tokens。

    注意：不收集 scripts 和 additional_docs，让大模型从指令中自己发现。
    这是 Anthropic Skills 的核心设计理念。
    """

    metadata: SkillMetadata
    instructions: str  # SKILL.md body 内容


class SkillLoader:
    """
    Skills 加载器

    核心职责：
    1. scan_skills(): 发现文件系统中的 Skills，解析元数据
    2. load_skill(): 按需加载 Skill 详细内容
    3. build_system_prompt(): 生成包含 Skills 列表的 system prompt

    使用示例：
        loader = SkillLoader()

        # Level 1: 获取 system prompt
        system_prompt = loader.build_system_prompt()

        # Level 2: 加载具体 skill
        skill = loader.load_skill("news-extractor")
        print(skill.instructions)
    """

    def __init__(self, skill_paths: list[Path] | None = None):
        """
        初始化加载器

        Args:
            skill_paths: 自定义 Skills 搜索路径，默认为:
                - .claude/skills/ (项目级，优先)
                - ~/.claude/skills/ (用户级，兜底)
        """
        self.skill_paths = skill_paths or DEFAULT_SKILL_PATHS
        self._metadata_cache: dict[str, SkillMetadata] = {}
        self._scan_cache: list[SkillMetadata] | None = None
        self._dir_mtimes: dict[str, float] = {}
        self._file_mtimes: dict[str, float] = {}

    def _is_cache_valid(self) -> bool:
        for base_path in self.skill_paths:
            key = str(base_path)
            try:
                if base_path.exists():
                    dir_mtime = base_path.stat().st_mtime
                    if key not in self._dir_mtimes:
                        return False
                    if dir_mtime != self._dir_mtimes[key]:
                        return False
                else:
                    if key in self._dir_mtimes:
                        return False
            except OSError:
                return False

        for fpath, cached_mtime in self._file_mtimes.items():
            try:
                if (
                    not Path(fpath).exists()
                    or Path(fpath).stat().st_mtime != cached_mtime
                ):
                    return False
            except OSError:
                return False

        return True

    def _save_mtimes(self) -> None:
        self._dir_mtimes.clear()
        self._file_mtimes.clear()
        for base_path in self.skill_paths:
            try:
                if base_path.exists():
                    self._dir_mtimes[str(base_path)] = base_path.stat().st_mtime
                    for skill_dir in base_path.iterdir():
                        if skill_dir.is_dir():
                            skill_md = skill_dir / "SKILL.md"
                            if skill_md.exists():
                                self._file_mtimes[str(skill_md)] = (
                                    skill_md.stat().st_mtime
                                )
            except OSError:
                pass

    def scan_skills(self, *, force: bool = False) -> list[SkillMetadata]:
        """
        Level 1: 扫描所有 Skills 元数据

        遍历 skill_paths，查找包含 SKILL.md 的目录，
        解析 YAML frontmatter 提取 name 和 description。

        Args:
            force: 强制忽略缓存，重新扫描磁盘

        Returns:
            所有发现的 Skills 元数据列表
        """
        if not force and self._scan_cache is not None and self._is_cache_valid():
            return self._scan_cache

        skills = []
        seen_names = set()

        for base_path in self.skill_paths:
            if not base_path.exists():
                continue

            for skill_dir in base_path.iterdir():
                if not skill_dir.is_dir():
                    continue

                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue

                metadata = self._parse_skill_metadata(skill_md)
                if metadata and metadata.name not in seen_names:
                    skills.append(metadata)
                    seen_names.add(metadata.name)
                    self._metadata_cache[metadata.name] = metadata

        self._scan_cache = skills
        self._save_mtimes()
        return skills

    # 解析skill元数据
    def _parse_skill_metadata(self, skill_md_path: Path) -> Optional[SkillMetadata]:
        try:
            content = skill_md_path.read_text(encoding="utf-8")
        except Exception:
            return None

        from chcode.utils.frontmatter import parse_frontmatter

        result = parse_frontmatter(content)
        if not result:
            return None

        try:
            name = result.frontmatter.get("name", "")
            description = result.frontmatter.get("description", "")
            if not name:
                return None
            return SkillMetadata(
                name=name,
                description=description,
                skill_path=skill_md_path.parent,
            )
        except Exception:
            return None

    # 从skill字典读取skill完整数据
    def load_skill(self, skill_name: str) -> Optional[SkillContent]:
        """
        Level 2: 加载 Skill 完整内容

        读取 SKILL.md 的完整指令，以及其他 .md 文件和脚本列表。
        这是 load_skill tool 的核心实现。

        Args:
            skill_name: Skill 名称（如 "news-extractor"）

        Returns:
            Skill 完整内容，未找到返回 None
        """
        # 先检查缓存
        metadata = self._metadata_cache.get(skill_name)

        # 原始冗余代码
        if not metadata:
            # 尝试重新扫描
            self.scan_skills()
            metadata = self._metadata_cache.get(skill_name)

        if not metadata:
            return None

        # 读取 SKILL.md 完整内容
        skill_md = metadata.skill_path / "SKILL.md"
        try:
            content = skill_md.read_text(encoding="utf-8")
        except Exception:
            return None

        from chcode.utils.frontmatter import parse_frontmatter

        fm_result = parse_frontmatter(content)
        instructions = fm_result.body if fm_result else content

        # 只返回 instructions，让大模型从指令中自己发现脚本和文档
        return SkillContent(
            metadata=metadata,
            instructions=instructions,
        )

    def build_system_prompt(self, base_prompt: str = "") -> str:
        """
        构建包含 Skills 列表的 system prompt

        这是 Level 1 的核心输出：将所有 Skills 的元数据
        注入到 system prompt 中。

        Args:
            base_prompt: 基础 system prompt（可选）

        Returns:
            完整的 system prompt
        """
        skills = self.scan_skills()

        # 构建 Skills 部分
        if skills:
            skills_section = "## Available Skills\n\n"
            skills_section += "You have access to the following specialized skills:\n\n"
            for skill in skills:
                skills_section += skill.to_prompt_line() + "\n"
            skills_section += "\n"
            skills_section += "### How to Use Skills\n\n"
            skills_section += "1. **Discover**: Review the skills list above\n"
            skills_section += (
                "2. **Load**: When a user request matches a skill's description, "
            )
            skills_section += (
                "use `load_skill(skill_name)` to get detailed instructions\n"
            )
            skills_section += (
                "3. **Execute**: Follow the skill's instructions, which may include "
            )
            skills_section += "running scripts via `bash`\n\n"
            skills_section += "**Important**: Only load a skill when it's relevant to the user's request. "
            skills_section += (
                "Script code never enters the context - only their output does.\n"
            )
        else:
            skills_section = "## Skills\n\nNo skills currently available.\n"

        # 组合完整 prompt
        if base_prompt:
            return f"{base_prompt}\n\n{skills_section}"
        else:
            return f"You are a helpful coding assistant.\n\n{skills_section}"


def scan_all_skills(project_path: Path | None = None) -> list[dict]:
    """扫描所有技能（项目级和全局级）

    Args:
        project_path: 项目路径，如果提供则同时扫描项目级技能

    Returns:
        技能信息列表，每个技能包含 name, type, description, path
    """
    skills = []
    loader = SkillLoader()

    # 扫描项目级技能
    if project_path:
        project_skills_path = project_path / ".chat" / "skills"
        if project_skills_path.exists():
            project_skills = _scan_skills_in_path(project_skills_path, "项目", loader)
            skills.extend(project_skills)

    # 扫描全局技能
    global_skills_path = Path.home() / ".chat" / "skills"
    if global_skills_path.exists():
        global_skills = _scan_skills_in_path(global_skills_path, "全局", loader)
        skills.extend(global_skills)

    return skills


def _scan_skills_in_path(
    skills_path: Path, skill_type: str, loader: SkillLoader
) -> list[dict]:
    """扫描指定路径下的技能"""
    skills = []

    if not skills_path.exists():
        return skills

    for skill_dir in skills_path.iterdir():
        if not skill_dir.is_dir():
            continue

        # 检查是否存在 SKILL.md
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue

        # 解析技能元数据
        metadata = loader._parse_skill_metadata(skill_md)
        if metadata:
            skills.append(
                {
                    "name": metadata.name,
                    "type": skill_type,
                    "description": metadata.description,
                    "path": str(skill_dir),
                }
            )

    return skills


def _extract_archive(archive_path, target_path):
    """安全解压压缩包，过滤路径穿越攻击"""
    resolved_target = target_path.resolve()
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                if not (target_path / member).resolve().is_relative_to(resolved_target):
                    return False
            zip_ref.extractall(target_path)
    elif archive_path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            for member in tar_ref.getmembers():
                if (
                    not (target_path / member.name)
                    .resolve()
                    .is_relative_to(resolved_target)
                ):
                    return False
            tar_ref.extractall(target_path)
    elif archive_path.endswith(".tar.bz2"):
        with tarfile.open(archive_path, "r:bz2") as tar_ref:
            for member in tar_ref.getmembers():
                if (
                    not (target_path / member.name)
                    .resolve()
                    .is_relative_to(resolved_target)
                ):
                    return False
            tar_ref.extractall(target_path)
    else:
        return False
    return True


def validate_skill_package(archive_path: str) -> dict | None:
    """验证技能包是否有效

    Args:
        archive_path: 压缩包路径

    Returns:
        技能信息字典或None（如果无效）
    """
    try:
        import tempfile
        from pathlib import Path

        loader = SkillLoader()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            if not _extract_archive(archive_path, temp_path):
                return None

            # 查找包含SKILL.md的目录
            skill_dir = _find_skill_dir(temp_path)
            if not skill_dir:
                return None

            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                return None

            # 解析元数据
            metadata = loader._parse_skill_metadata(skill_md)
            if not metadata:
                return None

            return {
                "name": metadata.name,
                "description": metadata.description,
                "source_path": str(skill_dir),
            }

    except Exception as e:
        print(f"验证技能包失败: {e}")
        return None


def _find_skill_dir(search_path: Path) -> Path | None:
    """在搜索路径中查找包含SKILL.md的目录"""
    # 首先检查根目录
    if (search_path / "SKILL.md").exists():
        return search_path

    # 检查子目录
    for item in search_path.iterdir():
        if item.is_dir():
            if (item / "SKILL.md").exists():
                return item
            # 递归检查更深层的目录
            result = _find_skill_dir(item)
            if result:
                return result

    return None


def install_skill(archive_path: str, install_path: Path) -> bool:
    """安装技能到指定位置

    Args:
        archive_path: 压缩包路径
        install_path: 安装目录路径

    Returns:
        是否安装成功
    """
    try:
        import tempfile
        import shutil
        from pathlib import Path

        # 创建安装目录
        install_path.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            if not _extract_archive(archive_path, temp_path):
                return False

            # 查找技能目录
            skill_source_dir = _find_skill_dir(temp_path)
            if not skill_source_dir:
                return False

            # 获取技能名称
            skill_md = skill_source_dir / "SKILL.md"
            loader = SkillLoader()
            metadata = loader._parse_skill_metadata(skill_md)
            if not metadata:
                return False

            skill_name = metadata.name
            target_dir = install_path / skill_name

            # 如果已存在，先删除
            if target_dir.exists():
                shutil.rmtree(target_dir)

            # 复制到目标位置
            shutil.copytree(skill_source_dir, target_dir)

            return True

    except Exception as e:
        print(f"安装技能失败: {e}")
        return False


@dataclass
class SkillAgentContext:
    """
    Agent 运行时上下文

    通过 ToolRuntime[SkillAgentContext] 在 tool 中访问
    """

    skill_loader: SkillLoader
    model_config: dict
    working_directory: Path
    thread_id: str = ""
    extra: dict = field(default_factory=dict)
