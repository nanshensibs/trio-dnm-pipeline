"""The three capabilities that take this beyond a generic hypothesis generator:

* Causal Evidence Graph Agent — audits that every hypothesis is backed by a
  traceable causal chain and surfaces contradictions / weak links.
* Negative Evidence Agent — actively hunts contradictory datasets, failed
  replications, null perturbations, and boundary conditions.
* Decision-Theoretic Experiment Planner — ranks experiments by expected
  information gain and returns the smallest discriminating set.

They are cross-cutting: they run at specific points in the pipeline rather than
belonging to a single layer.
"""

from __future__ import annotations

from typing import Any

from .. import parsing, prompts
from ..agent_base import AgentResult, ScientificAgent
from ..context import ResearchContext
from ..evidence import Evidence, EvidenceTier, Polarity
from ..experiment import DecisionTheoreticPlanner
from ..hypothesis import HypothesisStatus

NEGATIVE_EVIDENCE_PROMPT = prompts._p("""ROLE: Negative Evidence Agent.
Scientific literature is biased toward positive findings. Actively search for evidence
that CONTRADICTS each surviving hypothesis: failed replications, null perturbation
results (e.g. a knockout with no phenotype), datasets where the association disappears,
and boundary conditions where the mechanism does not hold. Report each with its tier
and the hypothesis it undercuts. Absence of a search hit is not evidence of absence —
say so.
JSON: {
  "negatives": [ {"hypothesis_id": "...", "finding": "...", "kind": "failed_replication|null_perturbation|contradictory_dataset|boundary_condition",
                  "tier": "human_genetics|perturbation|animal|association|review", "strength": 0.0, "source": "..."} ]
}""")


class CausalEvidenceGraphAgent(ScientificAgent):
    """Enforces the 'every hypothesis needs a traceable causal chain' rule."""

    agent_id = "causal_graph"
    name = "Causal Evidence Graph Agent"
    layer = 0  # cross-cutting
    purpose = "Audit causal chains; surface contradictions and weakest links."

    def act(self, ctx: ResearchContext) -> AgentResult:
        graph = ctx.graph
        orphaned = 0
        for hyp in ctx.pool.active:
            score = hyp.causal_chain_score(graph)
            chain = hyp.best_chain(graph)
            if score <= 0 or chain is None:
                hyp.note("causal-graph: no traceable chain to a clinical endpoint")
                orphaned += 1
            else:
                hyp.note(f"causal-graph: chain score {score:.2f}, weakest link {chain.weakest_link:.2f}")

        contradictions = graph.contradictions()
        weak_edges = graph.unsupported_edges(threshold=0.2)
        contested = graph.contested_edges()

        audit = {
            "orphaned_hypotheses": orphaned,
            "contradictions": contradictions,
            "n_weak_edges": len(weak_edges),
            "n_contested_edges": len(contested),
            "coverage": graph.coverage(),
            "weak_edges": [f"{e.source}->{e.target} ({e.support:.2f})" for e in weak_edges[:20]],
        }
        ctx.store("causal_audit", audit)
        summary = (f"Causal graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges; "
                   f"{orphaned} orphaned hypotheses, {len(contradictions)} contradictions, "
                   f"{len(weak_edges)} weak edges, {len(contested)} contested edges.")
        ctx.record(self.agent_id, self.layer, summary)
        return AgentResult(agent=self.agent_id, summary=summary, payload=audit)


class NegativeEvidenceAgent(ScientificAgent):
    """Deliberately searches for evidence against the surviving hypotheses."""

    agent_id = "negative_evidence"
    name = "Negative Evidence Agent"
    layer = 0  # cross-cutting
    purpose = "Hunt contradictory data, failed replications, null perturbations, boundaries."
    tools = ("mcp__PubMed__search_articles", "mcp__Consensus__search",
             "mcp__Open_Targets__query_open_targets_graphql")
    system_prompt = NEGATIVE_EVIDENCE_PROMPT

    def build_prompt(self, ctx: ResearchContext) -> str:
        live = "\n".join(f"- {h.id}: {h.statement}" for h in ctx.pool.ranked())
        return super().build_prompt(ctx) + "\n\nFind negative evidence against these:\n" + live

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        for neg in payload.get("negatives", []):
            hyp = ctx.pool.get(neg.get("hypothesis_id", ""))
            if not hyp:
                continue
            ev = Evidence(
                id=f"neg:{neg['hypothesis_id']}:{n}",
                tier=parsing.tier(neg.get("tier"), EvidenceTier.ASSOCIATION),
                polarity=Polarity.CONTRADICTS,
                summary=f"[{neg.get('kind','')}] {neg.get('finding','')}",
                source=neg.get("source", ""), agent=self.agent_id,
                strength=parsing.as_float(neg.get("strength"), 0.5),
            )
            ctx.register_evidence(ev)
            hyp.attach(ev)
            hyp.note(f"negative evidence: {neg.get('kind','')}")
            n += 1
        return AgentResult(agent=self.agent_id, produced_evidence=n,
                           summary=f"Surfaced {n} pieces of negative evidence.", payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"negatives": [
            {"hypothesis_id": "H03", "finding": "NCF1 low-ROS mouse shows ABC expansion only with a second hit; "
             "single-variant effect null in one cohort", "kind": "boundary_condition",
             "tier": "animal", "strength": 0.5, "source": "illustrative"},
            {"hypothesis_id": "H01", "finding": "one myeloid-specific IRF5 KO study reports partial ABC reduction, "
             "arguing IRF5 is not purely B-cell-intrinsic", "kind": "contradictory_dataset",
             "tier": "animal", "strength": 0.35, "source": "illustrative"},
        ]}


class DecisionTheoreticExperimentPlannerAgent(ScientificAgent):
    """Ranks experiments by expected information gain and picks a discriminating set."""

    agent_id = "experiment_planner"
    name = "Decision-Theoretic Experiment Planner"
    layer = 0  # cross-cutting (runs after Experimental Design)
    purpose = "Rank experiments by expected information gain / cost; pick the smallest set."

    def __init__(self, client=None, *, budget_months: float = 18.0, max_experiments: int = 3) -> None:
        super().__init__(client)
        self.budget_months = budget_months
        self.max_experiments = max_experiments

    def act(self, ctx: ResearchContext) -> AgentResult:
        prior = ctx.pool.belief_distribution()
        planner = DecisionTheoreticPlanner()
        plan = planner.plan(
            ctx.experiments, prior,
            budget_months=self.budget_months, max_experiments=self.max_experiments,
        )
        ctx.store("experiment_plan", plan)
        summary = (f"Experiment plan: {len(plan.chosen)} experiments chosen, "
                   f"uncertainty {plan.prior_entropy_bits:.2f} -> {plan.residual_entropy_bits:.2f} bits.")
        ctx.record(self.agent_id, self.layer, summary)
        return AgentResult(agent=self.agent_id, summary=summary,
                           payload={"chosen": [e.id for e in plan.chosen],
                                    "prior_bits": plan.prior_entropy_bits,
                                    "residual_bits": plan.residual_entropy_bits})
