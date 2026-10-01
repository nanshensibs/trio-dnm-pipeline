"""Render a RunReport as a Markdown briefing."""

from __future__ import annotations

from .orchestrator import RunReport


def render_markdown(report: RunReport) -> str:
    ctx = report.context
    fr = report.final_report
    lines: list[str] = []
    lines.append(f"# Lupus Co-Scientist report\n")
    lines.append(f"**Question:** {ctx.question}\n")
    if ctx.focus:
        lines.append(f"**Focus:** {ctx.focus}\n")

    lines.append("\n## Headline mechanism\n")
    lines.append(fr.get("headline_mechanism", "(none)") + "\n")

    lines.append("\n## Surviving hypotheses (ranked by belief)\n")
    lines.append("| id | belief | conf | evidence | weakest link | indep. support | novelty | causal chain |")
    lines.append("|----|--------|------|----------|--------------|----------------|---------|--------------|")
    for h in fr.get("hypotheses", []):
        lines.append(
            f"| {h['id']} | {h['belief']:.2f} | {h['confidence']} | {h['evidence_score']:.2f} "
            f"| {h['weakest_link']:.2f} | {h['independent_support']} | {h['novelty']:.2f} "
            f"| {h['causal_chain']} |"
        )

    elim = fr.get("eliminated", [])
    if elim:
        lines.append("\n## Eliminated hypotheses\n")
        for e in elim:
            lines.append(f"- **{e['id']}** — {e['statement']}\n    - {e['why']}")

    plan = report.experiment_plan
    if plan is not None:
        lines.append("\n## Decision-theoretic experiment plan\n")
        lines.append(f"Uncertainty over live hypotheses: "
                     f"**{plan.prior_entropy_bits:.2f} bits -> {plan.residual_entropy_bits:.2f} bits** "
                     f"after the chosen experiments.\n")
        for row in plan.describe():
            lines.append(row)

    audit = ctx.get("causal_audit")
    if audit:
        lines.append("\n## Causal evidence graph audit\n")
        lines.append(f"- Nodes/edges coverage: {audit.get('coverage')}")
        lines.append(f"- Contradictions: {audit.get('contradictions')}")
        lines.append(f"- Weak edges: {audit.get('n_weak_edges')}; contested edges: {audit.get('n_contested_edges')}")

    consensus = ctx.get("consensus")
    if consensus:
        lines.append("\n## Consensus & open questions\n")
        lines.append(f"**Consensus model:** {consensus.get('consensus_model','')}")
        for d in consensus.get("open_disagreements", []):
            lines.append(f"- Open: {d}")

    skeptic = ctx.get("skeptic")
    if skeptic:
        lines.append("\n## Falsification signatures\n")
        for t in skeptic.get("falsification_tests", []):
            lines.append(f"- {t.get('conclusion','')}\n    - breaks if: {t.get('breaking_observation','')}")

    lines.append("\n## Pipeline stages\n")
    lines.append("| stage | active hyp | uncertainty (bits) | detail |")
    lines.append("|-------|-----------|--------------------|--------|")
    for s in report.stages:
        lines.append(f"| {s.stage} | {s.active_hypotheses} | {s.entropy_bits} | {s.detail} |")

    return "\n".join(lines) + "\n"
