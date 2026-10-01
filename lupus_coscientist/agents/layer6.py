"""Layer 6 — Meta Reasoning agents (Debate, Consensus, Skeptic, Meta Reviewer)."""

from __future__ import annotations

from typing import Any

from .. import parsing, prompts
from ..agent_base import AgentResult, ScientificAgent
from ..context import ResearchContext
from ..hypothesis import HypothesisStatus


class DebateAgent(ScientificAgent):
    agent_id = "debate"
    name = "Debate Agent"
    layer = 6
    purpose = "Structured debate between top hypotheses; emit belief log-odds deltas."
    system_prompt = prompts.DEBATE

    def build_prompt(self, ctx: ResearchContext) -> str:
        top = "\n".join(f"- {h.id}: {h.statement} (belief {h.belief:.2f})" for h in ctx.pool.ranked()[:6])
        return super().build_prompt(ctx) + "\n\nDebate these top hypotheses:\n" + top

    def act(self, ctx: ResearchContext) -> AgentResult:
        payload = self._offline(ctx) if self.client is None else self.client.complete_json(self.request(ctx))
        deltas = {e["hypothesis_id"]: parsing.as_float(e.get("log_odds_delta"), 0.0)
                  for e in payload.get("exchanges", [])}
        ctx.pool.refresh_beliefs(debate_deltas=deltas)
        ctx.store("debate", payload)
        summary = f"Debate applied belief deltas to {len(deltas)} hypotheses."
        ctx.record(self.agent_id, self.layer, summary)
        return AgentResult(agent=self.agent_id, summary=summary, payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"exchanges": [
            {"hypothesis_id": "H01", "for": "B-cell-intrinsic genetics + independent validation + switch mechanism",
             "against": "IRF5 undruggable directly", "log_odds_delta": 0.6},
            {"hypothesis_id": "H02", "for": "myeloid IFN is real", "against": "cannot explain B-cell-intrinsic eQTL",
             "log_odds_delta": -0.5},
            {"hypothesis_id": "H04", "for": "dissociation artifacts exist", "against": "refuted by snRNA + spatial",
             "log_odds_delta": -1.2},
        ]}


class ConsensusAgent(ScientificAgent):
    agent_id = "consensus"
    name = "Consensus Agent"
    layer = 6
    purpose = "Find agreement; produce the consensus causal model; list open disagreements."
    system_prompt = prompts.CONSENSUS

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        ctx.store("consensus", payload)
        return AgentResult(agent=self.agent_id,
                           summary=f"Consensus model set; {len(payload.get('open_disagreements', []))} open disagreements.",
                           payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        top = ctx.pool.ranked()
        lead = top[0].statement if top else ""
        return {"consensus_model": lead,
                "agreements": ["ABC state is real (not artifact)", "IRF5 has a B-cell-intrinsic effect"],
                "open_disagreements": ["Relative contribution of NCF1/ROS (H03) vs IRF5 (H01)"]}


class SkepticAgent(ScientificAgent):
    agent_id = "skeptic"
    name = "Skeptic Agent"
    layer = 6
    purpose = "Attempt to falsify every surviving conclusion; demand a breaking observation."
    system_prompt = prompts.SKEPTIC

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        ctx.store("skeptic", payload)
        unfalsifiable = [t for t in payload.get("falsification_tests", []) if not parsing.as_bool(t.get("falsifiable"), True)]
        return AgentResult(agent=self.agent_id,
                           summary=f"Falsification tests defined; {len(unfalsifiable)} conclusions not yet falsifiable.",
                           payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"falsification_tests": [
            {"conclusion": "IRF5 dosage flips the ZEB2-MEF2B switch to drive ABCs",
             "breaking_observation": "B-cell-conditional Irf5 KO leaves ABC fraction unchanged",
             "already_exists": False, "falsifiable": True},
        ]}


class MetaReviewerAgent(ScientificAgent):
    agent_id = "meta_reviewer"
    name = "Meta Reviewer"
    layer = 6
    purpose = "Final report: evidence score, confidence, uncertainty, critical experiment, publication."
    system_prompt = prompts.META_REVIEWER

    def act(self, ctx: ResearchContext) -> AgentResult:
        # Build the report deterministically from the pool + planner so it always
        # reflects the actual belief state, augmenting any LLM narrative.
        ranked = ctx.pool.ranked()
        plan = ctx.get("experiment_plan")
        report = {
            "headline_mechanism": ranked[0].statement if ranked else "(no surviving hypothesis)",
            "hypotheses": [],
            "eliminated": [
                {"id": h.id, "statement": h.statement, "why": h.history[-1] if h.history else ""}
                for h in ctx.pool.all if h.status is HypothesisStatus.ELIMINATED
            ],
        }
        for h in ranked:
            chain = h.best_chain(ctx.graph)
            report["hypotheses"].append({
                "id": h.id, "statement": h.statement,
                "belief": round(h.belief, 3),
                "evidence_score": round(h.causal_chain_score(ctx.graph), 3),
                "confidence": _confidence(h.belief),
                "causal_chain": chain.describe() if chain else "(no traceable chain)",
                "weakest_link": round(chain.weakest_link, 3) if chain else 0.0,
                "independent_support": h.n_independent_support(),
                "novelty": round(h.novelty, 2),
                "remaining_uncertainty": h.history[-1] if h.history else "",
            })
        if plan is not None:
            report["critical_experiments"] = [e.id + ": " + e.title for e in plan.chosen]
            report["uncertainty_bits_before"] = round(plan.prior_entropy_bits, 3)
            report["uncertainty_bits_after"] = round(plan.residual_entropy_bits, 3)

        # Optional LLM narrative layer.
        if self.client is not None:
            try:
                narrative = self.client.complete_json(self.request(ctx))
                report["narrative"] = narrative
            except Exception:  # pragma: no cover - narrative is best-effort
                pass

        ctx.store("final_report", report)
        summary = (f"Final report: lead mechanism {ranked[0].id if ranked else 'none'}, "
                   f"{len(ranked)} survivors, {len(report['eliminated'])} eliminated.")
        ctx.record(self.agent_id, self.layer, summary)
        return AgentResult(agent=self.agent_id, summary=summary, payload=report)


def _confidence(belief: float) -> str:
    if belief >= 0.66:
        return "high"
    if belief >= 0.4:
        return "moderate"
    return "low"
