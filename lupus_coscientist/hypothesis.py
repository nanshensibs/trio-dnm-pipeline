"""Hypotheses and the progressive-elimination pool.

The system does not generate one hypothesis and defend it. It generates many
(20-50), then *progressively eliminates* the weak ones while accumulating causal
evidence for the survivors. This module holds the belief state that makes that
possible: each hypothesis carries a running belief (posterior-ish) that is
updated by supporting evidence, negative evidence, debate outcomes, and
validation on independent data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .evidence import CausalChain, CausalEvidenceGraph, Evidence, Polarity


class HypothesisStatus(Enum):
    ACTIVE = "active"            # still in contention
    ELIMINATED = "eliminated"    # pruned by negative evidence / debate / no causal chain
    VALIDATED = "validated"      # survived, confirmed on independent data
    PARKED = "parked"            # plausible but under-powered; awaiting an experiment


@dataclass
class Hypothesis:
    """A candidate causal mechanism.

    Attributes:
        id: stable id, e.g. "H07".
        statement: one-sentence causal claim.
        mechanism: the molecular/cellular mechanism proposed.
        chain_endpoints: (source_node, terminal_node) the hypothesis must connect
            in the causal evidence graph. A hypothesis with no traceable chain
            cannot accrue belief — this is the "causal, not descriptive" gate.
        belief: current belief in [0, 1]. Starts at a prior and is updated.
        novelty: 0 (already published) .. 1 (major conceptual advance).
        supporting / contradicting: attached evidence.
        touches_switch/ecology/fork: program-alignment flags (bistable switch,
            Clone Ecology, fork model) — a program-relevant hypothesis is favored
            only as a tie-breaker, never over evidence.
    """

    id: str
    statement: str
    mechanism: str = ""
    #: the full ordered node path this hypothesis declares (preferred).
    chain_nodes: Optional[tuple[str, ...]] = None
    #: fallback when only the two endpoints are known (best path is searched).
    chain_endpoints: Optional[tuple[str, str]] = None
    belief: float = 0.2
    prior: float = 0.2
    novelty: float = 0.5
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    supporting: list[Evidence] = field(default_factory=list)
    contradicting: list[Evidence] = field(default_factory=list)
    touches_switch: bool = False
    touches_ecology: bool = False
    touches_fork: bool = False
    tags: tuple[str, ...] = ()
    history: list[str] = field(default_factory=list)

    # -- evidence ingestion -------------------------------------------------
    def attach(self, ev: Evidence) -> None:
        if ev.polarity is Polarity.CONTRADICTS:
            self.contradicting.append(ev)
        elif ev.polarity is Polarity.SUPPORTS:
            self.supporting.append(ev)

    @property
    def program_relevance(self) -> float:
        return 0.34 * sum((self.touches_switch, self.touches_ecology, self.touches_fork))

    def best_chain(self, graph: CausalEvidenceGraph) -> Optional[CausalChain]:
        # Prefer the hypothesis's own declared path so chains for hypotheses that
        # share endpoints don't cross-contaminate.
        if self.chain_nodes:
            chain = graph.chain_from_nodes(list(self.chain_nodes))
            if chain is not None:
                return chain
        if self.chain_endpoints:
            src, tgt = self.chain_endpoints
            return graph.best_chain(src, tgt)
        return None

    def causal_chain_score(self, graph: CausalEvidenceGraph) -> float:
        chain = self.best_chain(graph)
        return chain.score() if chain else 0.0

    # -- belief update ------------------------------------------------------
    def evidence_log_odds(self) -> float:
        """Net signed, tier-weighted evidence as a log-odds nudge."""
        return sum(e.signed_weight for e in self.supporting) + sum(
            e.signed_weight for e in self.contradicting
        )

    def update_belief(self, graph: CausalEvidenceGraph, debate_delta: float = 0.0) -> None:
        """Recompute belief from prior + causal chain + hypothesis-level evidence + debate.

        Logistic update in log-odds space so evidence composes smoothly and belief
        stays in (0, 1). The causal chain is the dominant *positive* signal — a
        hypothesis is believable to the extent it is backed by a traceable,
        evidence-bearing chain to a clinical endpoint. Hypothesis-level evidence
        (independent validation up, reviewer/negative/statistics down) then adjusts
        it. A hypothesis with no traceable chain is strongly penalized regardless of
        how many correlative papers mention it.
        """
        prior_lo = _logit(self.prior)
        ev_lo = self.evidence_log_odds()
        chain = self.causal_chain_score(graph)
        if chain <= 0:
            chain_lo = -2.0          # no traceable causal chain: not yet a mechanism
        else:
            chain_lo = 4.0 * chain - 0.5   # chain 0.5 -> +1.5; weak proposed-only chain -> ~ -0.3
        total = prior_lo + chain_lo + 1.0 * ev_lo + debate_delta
        self.belief = _sigmoid(total)

    # -- lifecycle ----------------------------------------------------------
    def n_independent_support(self) -> int:
        return sum(1 for e in self.supporting if e.is_independent and e.strength >= 0.5)

    @property
    def is_active(self) -> bool:
        return self.status is HypothesisStatus.ACTIVE

    def note(self, msg: str) -> None:
        self.history.append(msg)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


class HypothesisPool:
    """Manages the population of hypotheses through elimination rounds."""

    def __init__(self, graph: CausalEvidenceGraph) -> None:
        self.graph = graph
        self._hyps: dict[str, Hypothesis] = {}
        self.round = 0

    def add(self, hyp: Hypothesis) -> Hypothesis:
        self._hyps[hyp.id] = hyp
        return hyp

    def get(self, hid: str) -> Optional[Hypothesis]:
        return self._hyps.get(hid)

    @property
    def all(self) -> list[Hypothesis]:
        return list(self._hyps.values())

    @property
    def active(self) -> list[Hypothesis]:
        return [h for h in self._hyps.values() if h.status is HypothesisStatus.ACTIVE]

    @property
    def survivors(self) -> list[Hypothesis]:
        return [h for h in self._hyps.values()
                if h.status in (HypothesisStatus.ACTIVE, HypothesisStatus.VALIDATED)]

    def refresh_beliefs(self, debate_deltas: Optional[dict[str, float]] = None) -> None:
        debate_deltas = debate_deltas or {}
        for h in self.active:
            h.update_belief(self.graph, debate_deltas.get(h.id, 0.0))

    def ranked(self) -> list[Hypothesis]:
        """Active hypotheses, strongest first. Program relevance is only a tie-break."""
        return sorted(
            self.active,
            key=lambda h: (round(h.belief, 4), round(h.causal_chain_score(self.graph), 4),
                           h.program_relevance, h.novelty),
            reverse=True,
        )

    def belief_distribution(self) -> dict[str, float]:
        """Normalized belief over the currently active hypotheses (a probability
        simplex used by the decision-theoretic experiment planner)."""
        active = self.active
        total = sum(h.belief for h in active)
        if total <= 0:
            n = len(active) or 1
            return {h.id: 1 / n for h in active}
        return {h.id: h.belief / total for h in active}

    def entropy(self) -> float:
        """Shannon entropy (bits) of the belief distribution — our uncertainty.

        The goal of experimentation is to drive this down: a peaked distribution
        means one mechanism is winning; a flat one means the field is unresolved.
        """
        dist = self.belief_distribution()
        return -sum(p * math.log2(p) for p in dist.values() if p > 0)

    def prune(
        self,
        *,
        belief_floor: float = 0.12,
        require_chain: bool = True,
        min_survivors: int = 3,
        max_negative: int = 3,
    ) -> list[Hypothesis]:
        """Eliminate weak hypotheses. Returns the list eliminated this round.

        A hypothesis is eliminated when it is dominated on the evidence: belief
        below the floor, OR no traceable causal chain, OR buried under negative
        evidence with no high-tier support. We always keep at least `min_survivors`
        so the debate never collapses to a single favored idea prematurely.
        """
        self.round += 1
        self.refresh_beliefs()
        eliminated: list[Hypothesis] = []
        ranked = self.ranked()
        protected = {h.id for h in ranked[:min_survivors]}

        for h in ranked:
            if h.id in protected:
                continue
            reasons = []
            if h.belief < belief_floor:
                reasons.append(f"belief {h.belief:.2f} < floor {belief_floor:.2f}")
            if require_chain and h.causal_chain_score(self.graph) <= 0:
                reasons.append("no traceable causal chain to a clinical endpoint")
            n_neg = len(h.contradicting)
            top_tier = max((e.tier for e in h.supporting), default=None)
            if n_neg >= max_negative and (top_tier is None or int(top_tier) < 4):
                reasons.append(f"{n_neg} contradicting lines, no perturbation/genetic support")
            if reasons:
                h.status = HypothesisStatus.ELIMINATED
                h.note("ELIMINATED round %d: %s" % (self.round, "; ".join(reasons)))
                eliminated.append(h)
        return eliminated

    def park_underpowered(self, chain_floor: float = 0.25) -> list[Hypothesis]:
        """Move plausible-but-unproven hypotheses to PARKED so the experiment
        planner is pointed at exactly the uncertainty worth resolving."""
        parked = []
        for h in self.active:
            if h.belief >= 0.12 and h.causal_chain_score(self.graph) < chain_floor:
                if h.n_independent_support() == 0:
                    h.status = HypothesisStatus.PARKED
                    h.note("PARKED round %d: plausible but under-powered" % self.round)
                    parked.append(h)
        return parked

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for h in self._hyps.values():
            counts[h.status.value] = counts.get(h.status.value, 0) + 1
        return counts
