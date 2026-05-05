"""
YAML Frontmatter 解析工具
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

_FM_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class FrontMatterResult:
    frontmatter: dict
    body: str


def parse_frontmatter(content: str) -> FrontMatterResult | None:
    m = _FM_PATTERN.match(content)
    if not m:
        return None
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(fm, dict):
        return None
    body = content[m.end():].strip()
    return FrontMatterResult(frontmatter=fm, body=body)
