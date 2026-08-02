"""Map agent tool references to Claude Agent SDK MCP server configuration.

Each agent declares the MCP tools it may use (e.g. ``mcp__PubMed__search_articles``).
This module groups those into the server blocks the Claude Agent SDK expects and
provides a default configuration for the scientific servers this framework targets.

The concrete transport (stdio command, URL, headers) is environment-specific, so
this returns a template you fill in with real endpoints/keys before passing to
``ClaudeAgentClient(mcp_servers=...)``.
"""

from __future__ import annotations

from .agents.registry import AGENT_CLASSES

# The scientific MCP servers the agents are designed around.
SCIENTIFIC_SERVERS = [
    "PubMed",
    "Consensus",
    "Scholar_Gateway",
    "Open_Targets",
    "Clinical_Trials",
    "alphaXiv",
]


def tools_by_server() -> dict[str, set[str]]:
    """Which MCP tools each server must expose, derived from agent declarations."""
    grouped: dict[str, set[str]] = {}
    for cls in AGENT_CLASSES:
        for tool in cls.tools:
            # tool ids look like mcp__<Server>__<tool>
            parts = tool.split("__")
            if len(parts) >= 3:
                server = parts[1]
                grouped.setdefault(server, set()).add(tool)
    return grouped


def all_allowed_tools() -> list[str]:
    """Flat list of every MCP tool any agent may call (for allowed_tools)."""
    tools: set[str] = set()
    for cls in AGENT_CLASSES:
        tools.update(cls.tools)
    return sorted(tools)


def default_server_template() -> dict[str, dict]:
    """A template mcp_servers config. Fill in transport details per server.

    Example (stdio):
        cfg = default_server_template()
        cfg["PubMed"] = {"type": "stdio", "command": "pubmed-mcp", "args": []}
    """
    template: dict[str, dict] = {}
    for server in SCIENTIFIC_SERVERS:
        template[server] = {
            "type": "http",           # or "stdio"
            "url": f"https://<your-{server.lower()}-mcp-endpoint>",
            "headers": {},
        }
    return template
