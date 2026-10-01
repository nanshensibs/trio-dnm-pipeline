"""The Lupus Co-Scientist orchestrator.

Implements the six-layer pipeline as a sequence of stages with belief refresh and
*progressive elimination* interleaved, so the system literally does what it claims:
eliminates weak hypotheses while accumulating causal evidence.

Pipeline (matching the architecture's evidence flow):

    Knowledge Acquisition
        -> Mechanism Discovery (populate causal graph)
        -> [Causal Graph audit]
        -> Hypothesis Generation (Mechanism/Alternative/Counterfactual)
        -> Reviewer + Novelty challenge
        -> [Negative Evidence hunt]  -> Debate  -> PRUNE round 1
        -> Validation (independent data, Statistics, Experimental Design)
        -> [Negative Evidence hunt]  -> PRUNE round 2
        -> [Decision-Theoretic Experiment Planner]
        -> Clinical Translation
        -> Meta Reasoning (Debate / Consensus / Skeptic / Meta Review)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .agents.registry import build_agents
from .context import ResearchContext
from .llm import LLMClient


@dataclass
class RunConfig:
    belief_floor: float = 0.12
    min_survivors: int = 3
    max_negative: int = 3
    experiment_budget_months: float = 18.0
    max_experiments: int = 3
    verbose: bool = True


@dataclass
class StageOutcome:
    stage: str
    detail: str
    active_hypotheses: int
    entropy_bits: float


@dataclass
class RunReport:
    context: ResearchContext
    stages: list[StageOutcome] = field(default_factory=list)

    @property
    def final_report(self) -> dict:
        return self.context.get("final_report", {})

    @property
    def experiment_plan(self):
        return self.context.get("experiment_plan")


class LupusCoScientist:
    """Top-level driver. Construct with an LLM client (or None for offline mode)."""

    def __init__(self, client: Optional[LLMClient] = None, config: Optional[RunConfig] = None) -> None:
        self.client = client
        self.config = config or RunConfig()
        self.agents = build_agents(client)

    # -- helpers ------------------------------------------------------------
    def _run(self, ctx: ResearchContext, *agent_ids: str) -> None:
        for aid in agent_ids:
            agent = self.agents[aid]
            result = agent.act(ctx)
            if self.config.verbose:
                print(f"  [{agent.name}] {result.summary}")

    def _stage(self, report: RunReport, name: str, detail: str) -> None:
        ctx = report.context
        outcome = StageOutcome(
            stage=name, detail=detail,
            active_hypotheses=len(ctx.pool.active),
            entropy_bits=round(ctx.pool.entropy(), 3),
        )
        report.stages.append(outcome)
        if self.config.verbose:
            print(f"== {name}: {detail} | active={outcome.active_hypotheses} "
                  f"| uncertainty={outcome.entropy_bits} bits\n")

    def _prune(self, ctx: ResearchContext, label: str, min_survivors: int) -> None:
        eliminated = ctx.pool.prune(
            belief_floor=self.config.belief_floor,
            require_chain=True,
            min_survivors=min_survivors,
            max_negative=self.config.max_negative,
        )
        if self.config.verbose:
            for h in eliminated:
                print(f"    - ELIMINATED {h.id}: {h.history[-1] if h.history else ''}")
            print(f"  [{label}] eliminated {len(eliminated)}; "
                  f"{len(ctx.pool.active)} active, {len(ctx.pool.survivors)} survivors")

    # -- main pipeline ------------------------------------------------------
    def investigate(self, question: str, focus: str = "") -> RunReport:
        ctx = ResearchContext(question=question, focus=focus)
        report = RunReport(context=ctx)

        # Layer 1 — Knowledge Acquisition
        self._run(ctx, "literature", "evidence_ranking", "public_database", "clinical_guideline")
        self._stage(report, "Layer 1 Knowledge Acquisition", "evidence collected & tiered")

        # Layer 2 — Mechanism Discovery
        self._run(ctx, "cell_state", "trajectory", "regulatory_network", "ligand_receptor",
                  "genetic_causality", "evolutionary", "tissue_ecology", "metabolism")
        self._stage(report, "Layer 2 Mechanism Discovery", "causal graph populated")

        # Novel: causal-graph audit of the mechanism scaffold
        self._run(ctx, "causal_graph")

        # Layer 3 — Hypothesis Generation
        self._run(ctx, "mechanism_generator", "alternative", "counterfactual", "reviewer", "novelty")
        self._stage(report, "Layer 3 Hypothesis Generation", "candidate mechanisms proposed")

        # Novel: causal-graph audit now that hypotheses exist; negative-evidence hunt
        self._run(ctx, "causal_graph", "negative_evidence")

        # Debate then first elimination round — gentle: keep the field open early
        # (min_survivors from config), only trimming clearly dominated ideas.
        self._run(ctx, "debate")
        self._prune(ctx, "Prune round 1", min_survivors=self.config.min_survivors)
        self._stage(report, "Challenge & Prune #1", "debate + negative evidence")

        # Layer 4 — Validation
        self._run(ctx, "public_validation", "statistics", "experimental_design")
        self._stage(report, "Layer 4 Validation", "independent validation + stats audit")

        # Second negative-evidence hunt + elimination round after validation —
        # decisive: independent data is in, so allow the field to narrow to the
        # single best-supported mechanism if that is where the evidence points.
        self._run(ctx, "negative_evidence")
        self._prune(ctx, "Prune round 2", min_survivors=1)
        self._stage(report, "Challenge & Prune #2", "post-validation elimination")

        # Novel: decision-theoretic experiment planning over survivors
        planner = self.agents["experiment_planner"]
        if isinstance(planner, object):
            planner.budget_months = self.config.experiment_budget_months
            planner.max_experiments = self.config.max_experiments
        self._run(ctx, "experiment_planner")
        self._stage(report, "Experiment Planning", "EIG-ranked discriminating experiments")

        # Layer 5 — Translation (survivors only)
        self._run(ctx, "drug_discovery", "biomarker", "precision_medicine", "trial_design")
        self._stage(report, "Layer 5 Translation", "targets / biomarkers / endotypes / trial")

        # Layer 6 — Meta Reasoning
        self._run(ctx, "debate", "consensus", "skeptic", "meta_reviewer")
        self._stage(report, "Layer 6 Meta Review", "final report assembled")

        return report
