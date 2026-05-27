from __future__ import annotations

from langchain_core.tools import BaseTool

from src.models.state import AgentRole
from src.tools.web_search import web_search_tool, file_read_tool, file_write_tool

# Per-agent tool scoping: the supervisor decides which tools each agent can invoke.
_AGENT_TOOL_SCOPE: dict[AgentRole, list[str]] = {
    AgentRole.SUPERVISOR: [],
    AgentRole.RESEARCHER: ["web_search_tool", "file_read_tool"],
    AgentRole.ANALYST: ["file_read_tool"],
    AgentRole.WRITER: ["file_read_tool", "file_write_tool"],
    AgentRole.REVIEWER: ["file_read_tool"],
}

_ALL_TOOLS: dict[str, BaseTool] = {
    "web_search_tool": web_search_tool,
    "file_read_tool": file_read_tool,
    "file_write_tool": file_write_tool,
}

# Tools that require human-in-the-loop approval before execution
APPROVAL_REQUIRED_TOOLS: set[str] = {"file_write_tool"}


class ToolRegistry:
    """Scoped tool registry — each agent only sees tools it's authorized to use."""

    def __init__(self) -> None:
        self._scopes = dict(_AGENT_TOOL_SCOPE)
        self._tools = dict(_ALL_TOOLS)

    def get_tools_for_agent(self, role: AgentRole) -> list[BaseTool]:
        allowed = self._scopes.get(role, [])
        return [self._tools[name] for name in allowed if name in self._tools]

    def get_tool_names_for_agent(self, role: AgentRole) -> list[str]:
        return list(self._scopes.get(role, []))

    def requires_approval(self, tool_name: str) -> bool:
        return tool_name in APPROVAL_REQUIRED_TOOLS

    def register_tool(self, name: str, tool: BaseTool, agents: list[AgentRole]) -> None:
        self._tools[name] = tool
        for agent in agents:
            if agent not in self._scopes:
                self._scopes[agent] = []
            if name not in self._scopes[agent]:
                self._scopes[agent].append(name)
