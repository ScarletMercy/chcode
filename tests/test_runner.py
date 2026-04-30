from unittest.mock import MagicMock

from chcode.agents.definitions import AgentDefinition
from chcode.agents.runner import _resolve_tools


def _tool(name):
    t = MagicMock()
    t.name = name
    return t


class TestResolveTools:
    def test_excludes_agent_tool(self):
        tools = [_tool("bash"), _tool("agent"), _tool("read")]
        ad = AgentDefinition(agent_type="t", when_to_use="w", system_prompt="p")
        result = _resolve_tools(ad, tools)
        names = [t.name for t in result]
        assert "agent" not in names
        assert "bash" in names

    def test_excludes_disallowed(self):
        tools = [_tool("bash"), _tool("write_file"), _tool("edit")]
        ad = AgentDefinition(
            agent_type="t",
            when_to_use="w",
            system_prompt="p",
            disallowed_tools=["write_file", "edit"],
        )
        result = _resolve_tools(ad, tools)
        names = [t.name for t in result]
        assert "write_file" not in names
        assert "edit" not in names

    def test_whitelist_tools(self):
        tools = [_tool("bash"), _tool("read"), _tool("edit")]
        ad = AgentDefinition(
            agent_type="t",
            when_to_use="w",
            system_prompt="p",
            tools=["bash", "read"],
        )
        result = _resolve_tools(ad, tools)
        names = [t.name for t in result]
        assert names == ["bash", "read"]

    def test_no_restrictions(self):
        tools = [_tool("bash"), _tool("read")]
        ad = AgentDefinition(agent_type="t", when_to_use="w", system_prompt="p")
        result = _resolve_tools(ad, tools)
        assert len(result) == 2

    def test_fallback_to_func_name_when_name_missing(self):
        t = MagicMock()
        t.name = None
        t.func = MagicMock(__name__="custom_tool")
        tools = [t]
        ad = AgentDefinition(agent_type="t", when_to_use="w", system_prompt="p")
        result = _resolve_tools(ad, tools)
        assert len(result) == 1
