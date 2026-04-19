from __future__ import annotations

import re
from pathlib import Path

import yaml

from chcode.agents.definitions import AgentDefinition

DEFAULT_AGENT_PATHS = [
    Path.cwd() / ".chat" / "agents",
    Path.home() / ".chat" / "agents",
]


def _parse_agent_md(md_path: Path) -> AgentDefinition | None:
    try:
        content = md_path.read_text(encoding="utf-8")
    except Exception:
        return None

    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not fm_match:
        return None

    try:
        frontmatter = yaml.safe_load(fm_match.group(1))
    except yaml.YAMLError:
        return None

    if not isinstance(frontmatter, dict):
        return None

    agent_type = frontmatter.get("name", "")
    description = frontmatter.get("description", "")

    if not agent_type or not description:
        return None

    body = content[fm_match.end() :]
    system_prompt = body.strip()
    if not system_prompt:
        return None

    tools_raw = frontmatter.get("tools")
    tools = (
        [t.strip() for t in tools_raw.split(",") if t.strip()]
        if isinstance(tools_raw, str)
        else None
    )

    disallowed_raw = frontmatter.get("disallowed_tools")
    disallowed_tools = (
        [t.strip() for t in disallowed_raw.split(",") if t.strip()]
        if isinstance(disallowed_raw, str)
        else []
    )

    model = frontmatter.get("model") or None
    read_only = bool(frontmatter.get("read_only", False))

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
