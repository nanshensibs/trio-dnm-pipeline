"""Command-line interface for the Lupus Co-Scientist.

    python -m lupus_coscientist investigate "Question..." --focus IRF5
    python -m lupus_coscientist list
"""

from __future__ import annotations

import argparse
import sys

from .orchestrator import LupusCoScientist, RunConfig
from .report import render_markdown


def _cmd_investigate(args: argparse.Namespace) -> int:
    client = None
    if args.live:
        try:
            from .llm import ClaudeAgentClient
            from .mcp_config import default_server_template
            client = ClaudeAgentClient(mcp_servers=default_server_template())
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"Could not start live client ({exc}); falling back to offline mode.",
                  file=sys.stderr)
            client = None

    config = RunConfig(
        min_survivors=args.min_survivors,
        experiment_budget_months=args.budget,
        max_experiments=args.max_experiments,
        verbose=not args.quiet,
    )
    engine = LupusCoScientist(client=client, config=config)
    report = engine.investigate(args.question, focus=args.focus)

    md = render_markdown(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(md)
        print(f"\nWrote report to {args.out}")
    else:
        print("\n" + md)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from .agents.registry import catalogue
    layers = {1: "Knowledge Acquisition", 2: "Mechanism Discovery", 3: "Hypothesis Generation",
              4: "Validation", 5: "Translation", 6: "Meta Reasoning", 0: "Cross-cutting (novel)"}
    current = None
    for row in catalogue():
        if row["layer"] != current:
            current = row["layer"]
            print(f"\n== Layer {current}: {layers.get(current,'')} ==")
        tools = f"  tools: {', '.join(row['tools'])}" if row["tools"] else ""
        print(f"  {row['n']:>2}. {row['name']} ({row['id']}) — {row['purpose']}{tools}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lupus_coscientist",
                                     description="Mechanism-discovery co-scientist for SLE.")
    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("investigate", help="Run the full pipeline on a question.")
    inv.add_argument("question", help="The research question.")
    inv.add_argument("--focus", default="", help="Optional focus, e.g. IRF5.")
    inv.add_argument("--live", action="store_true",
                     help="Use live Claude Agent SDK + MCP tools (default: offline demo).")
    inv.add_argument("--min-survivors", type=int, default=3, dest="min_survivors")
    inv.add_argument("--budget", type=float, default=18.0, help="Experiment budget (lab-months).")
    inv.add_argument("--max-experiments", type=int, default=3, dest="max_experiments")
    inv.add_argument("--out", default="", help="Write markdown report to this path.")
    inv.add_argument("--quiet", action="store_true", help="Suppress per-agent progress.")
    inv.set_defaults(func=_cmd_investigate)

    lst = sub.add_parser("list", help="List the agent roster.")
    lst.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
