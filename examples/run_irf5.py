"""Worked example: the causal role of IRF5 in ABC-driven lupus nephritis.

Runs entirely offline (deterministic stubs) so it works with no API key. To run
with live literature/genetics/trial tools, construct a ClaudeAgentClient wired to
the scientific MCP servers and pass it to LupusCoScientist (see the block at the
bottom, commented out).
"""

from lupus_coscientist import LupusCoScientist, RunConfig
from lupus_coscientist.report import render_markdown


def main() -> None:
    engine = LupusCoScientist(config=RunConfig(verbose=True))
    report = engine.investigate(
        question="What is the causal role of IRF5 in ABC-driven lupus nephritis, "
                 "and is the effect B-cell-intrinsic?",
        focus="IRF5",
    )

    print("\n" + "=" * 70)
    print(render_markdown(report))

    fr = report.final_report
    print("Lead mechanism:", fr["headline_mechanism"])
    print("Critical experiment(s):", fr.get("critical_experiments"))


# --- Live mode (requires claude-agent-sdk + configured MCP servers) -----------
# from lupus_coscientist import ClaudeAgentClient
# from lupus_coscientist.mcp_config import default_server_template
#
# def main_live() -> None:
#     servers = default_server_template()       # fill in real endpoints/keys
#     client = ClaudeAgentClient(mcp_servers=servers)
#     engine = LupusCoScientist(client=client)
#     report = engine.investigate("...", focus="IRF5")
#     print(render_markdown(report))


if __name__ == "__main__":
    main()
