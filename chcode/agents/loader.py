from __future__ import annotations

from pathlib import Path

from chcode.agents.definitions import AgentDefinition
from chcode.utils.frontmatter import parse_frontmatter

DEFAULT_AGENT_PATHS = [
    Path.cwd() / ".chat" / "agents",
    Path.home() / ".chat" / "agents",
]


def _parse_agent_md(md_path: Path) -> AgentDefinition | None:
    try:
        content = md_path.read_text(encoding="utf-8")
    except Exception:
        return None

    fm_result = parse_frontmatter(content)
    if not fm_result:
        return None

    fm = fm_result.frontmatter
    system_prompt = fm_result.body
    if not system_prompt:
        return None

    agent_type = fm.get("name", "")
    description = fm.get("description", "")

    if not agent_type or not description:
        return None

    tools_raw = fm.get("tools")
    tools = (
        [t.strip() for t in tools_raw.split(",") if t.strip()]
        if isinstance(tools_raw, str)
        else None
    )

    disallowed_raw = fm.get("disallowed_tools")
    disallowed_tools = (
        [t.strip() for t in disallowed_raw.split(",") if t.strip()]
        if isinstance(disallowed_raw, str)
        else []
    )

    model = fm.get("model") or None
    read_only = bool(fm.get("read_only", False))

    return AgentDefinition(
        agent_type=agent_type,
        when_to_use=description.replace("\\n", "\n"),
        system_prompt=system_prompt,
        tools=tools,
        disallowed_tools=disallowed_tools,
        model=model,
        read_only=read_only,
        source="custom",
    )


_agents_cache: dict[str, AgentDefinition] | None = None


def load_agents(extra_paths: list[Path] | None = None) -> dict[str, AgentDefinition]:
    global _agents_cache

    if _agents_cache is not None and not extra_paths:
        return dict(_agents_cache)

    from chcode.agents.definitions import BUILT_IN_AGENTS

    result: dict[str, AgentDefinition] = dict(BUILT_IN_AGENTS)

    paths = list(DEFAULT_AGENT_PATHS)
    if extra_paths:
        paths = extra_paths + paths

    for base_path in paths:
        if not base_path.exists():
            continue
        for item in base_path.iterdir():
            if not item.is_file() or not item.suffix == ".md":
                continue
            agent = _parse_agent_md(item)
            if agent and agent.agent_type not in result:
                result[agent.agent_type] = agent

    if not extra_paths:
        _agents_cache = result

    return result
