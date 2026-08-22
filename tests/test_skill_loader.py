from pathlib import Path

from chcode.utils.skill_loader import (
    SkillLoader,
    SkillMetadata,
    SkillContent,
    scan_all_skills,
)


def _create_skill(
    directory: Path, name: str, description: str = "desc", body: str = "# Instructions"
):
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}",
        encoding="utf-8",
    )
    return skill_dir


class TestSkillMetadata:
    def test_to_prompt_line(self):
        m = SkillMetadata(
            name="test-skill", description="A test", skill_path=Path("/tmp")
        )
        line = m.to_prompt_line()
        assert "**test-skill**" in line
        assert "A test" in line


class TestSkillLoader:
    def test_scan_empty_dir(self, tmp_path: Path):
        loader = SkillLoader(skill_paths=[tmp_path / "skills"])
        assert loader.scan_skills() == []

    def test_scan_finds_skill(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "my-skill", "Does something")
        loader = SkillLoader(skill_paths=[skills_dir])
        skills = loader.scan_skills()
        assert len(skills) == 1
        assert skills[0].name == "my-skill"

    def test_scan_caches(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "cached")
        loader = SkillLoader(skill_paths=[skills_dir])
        s1 = loader.scan_skills()
        s2 = loader.scan_skills()
        assert s1 is s2

    def test_scan_force_refresh(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "force-test")
        loader = SkillLoader(skill_paths=[skills_dir])
        s1 = loader.scan_skills()
        s2 = loader.scan_skills(force=True)
        assert s1 is not s2

    def test_load_skill(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "loadable", body="# Do stuff\nMore details")
        loader = SkillLoader(skill_paths=[skills_dir])
        loader.scan_skills()
        content = loader.load_skill("loadable")
        assert content is not None
        assert "Do stuff" in content.instructions

    def test_load_skill_not_found(self, tmp_path: Path):
        loader = SkillLoader(skill_paths=[tmp_path / "skills"])
        assert loader.load_skill("nonexistent") is None

    def test_build_system_prompt_with_skills(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "prompt-skill", "Does things")
        loader = SkillLoader(skill_paths=[skills_dir])
        prompt = loader.build_system_prompt("Base prompt")
        assert "Base prompt" in prompt
        assert "prompt-skill" in prompt

    def test_build_system_prompt_no_skills(self, tmp_path: Path):
        loader = SkillLoader(skill_paths=[tmp_path / "empty"])
        prompt = loader.build_system_prompt()
        assert "No skills" in prompt

    def test_duplicate_name_first_wins(self, tmp_path: Path):
        dir1 = tmp_path / "s1"
        dir2 = tmp_path / "s2"
        _create_skill(dir1, "dup", "First")
        _create_skill(dir2, "dup", "Second")
        loader = SkillLoader(skill_paths=[dir1, dir2])
        skills = loader.scan_skills()
        assert len(skills) == 1
        assert skills[0].description == "First"

    def test_skips_non_dirs_loader(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "readme.txt").write_text("not a skill")
        loader = SkillLoader(skill_paths=[skills_dir])
        assert loader.scan_skills() == []

    def test_skips_no_skill_md_loader(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_sub = skills_dir / "no-md"
        skill_sub.mkdir(parents=True)
        loader = SkillLoader(skill_paths=[skills_dir])
        assert loader.scan_skills() == []

    def test_parse_invalid_yaml(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "bad-yaml"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\n: invalid\n---\nbody", encoding="utf-8")
        loader = SkillLoader(skill_paths=[skills_dir])
        skills = loader.scan_skills()
        assert len(skills) == 0

    def test_parse_no_name(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "no-name"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\ndescription: test\n---\nbody", encoding="utf-8")
        loader = SkillLoader(skill_paths=[skills_dir])
        assert loader.scan_skills() == []


class TestScanAllSkills:
    def test_with_project_path_loader(self, tmp_path: Path, monkeypatch):
        skills_dir = tmp_path / ".chat" / "skills"
        _create_skill(skills_dir, "proj-skill", "Project skill")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = scan_all_skills(tmp_path)
        assert any(s["name"] == "proj-skill" for s in result)

    def test_empty_loader(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "nonexistent")
        result = scan_all_skills(tmp_path)
        assert result == []
