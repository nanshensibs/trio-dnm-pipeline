"""Decision-theoretic experiment planning.

Instead of proposing a long wish-list of experiments, this module estimates the
*expected information gain* (EIG) of each candidate experiment — how much it is
expected to reduce our uncertainty over the competing hypotheses — and ranks by
EIG per unit cost. The output is the smallest set of experiments that most
sharply discriminates among surviving mechanisms.

The math is standard Bayesian experimental design:

    EIG(e) = H(prior) - E_{outcome o}[ H(posterior | o) ]

where H is Shannon entropy over the hypothesis belief distribution, and the
per-hypothesis likelihoods of each outcome are supplied by the (LLM-backed)
Experimental Design Agent as a small predicted-outcome table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ExperimentType(Enum):
    LOSS_OF_FUNCTION = "loss_of_function"
    GAIN_OF_FUNCTION = "gain_of_function"
    CRISPR_SCREEN = "crispr_screen"
    PERTURB_SEQ = "perturb_seq"
    BASE_EDIT = "base_edit"
    MOUSE_IN_VIVO = "mouse_in_vivo"
    HUMAN_TISSUE = "human_tissue"
    SPATIAL = "spatial"
    TIME_COURSE = "time_course"
    CLINICAL_ARM = "clinical_arm"


# Rough relative cost/time priors (in "lab-months") used when an agent doesn't
# supply its own. These are deliberately conservative and easy to override.
_DEFAULT_COST = {
    ExperimentType.LOSS_OF_FUNCTION: 3.0,
    ExperimentType.GAIN_OF_FUNCTION: 3.0,
    ExperimentType.CRISPR_SCREEN: 6.0,
    ExperimentType.PERTURB_SEQ: 9.0,
    ExperimentType.BASE_EDIT: 5.0,
    ExperimentType.MOUSE_IN_VIVO: 12.0,
    ExperimentType.HUMAN_TISSUE: 6.0,
    ExperimentType.SPATIAL: 5.0,
    ExperimentType.TIME_COURSE: 7.0,
    ExperimentType.CLINICAL_ARM: 36.0,
}


@dataclass
class PredictedOutcome:
    """One possible readout of an experiment and how each hypothesis predicts it.

    likelihoods maps hypothesis_id -> P(this outcome | that hypothesis is true),
    i.e. a column of the likelihood table. The planner does not need these to be
    perfectly calibrated; even coarse 0.1/0.5/0.9 predictions separate
    experiments that discriminate from experiments that don't.
    """

    label: str
    likelihoods: dict[str, float]


@dataclass
class Experiment:
    """A candidate experiment scored for expected information gain."""

    id: str
    title: str
    etype: ExperimentType
    hypotheses_tested: tuple[str, ...]
    outcomes: list[PredictedOutcome] = field(default_factory=list)
    cost_months: Optional[float] = None
    feasibility: float = 0.8         # 0..1, probability the assay actually works
    reversibility_note: str = ""
    rationale: str = ""

    @property
    def cost(self) -> float:
        return self.cost_months if self.cost_months is not None else _DEFAULT_COST[self.etype]

    def expected_information_gain(self, prior: dict[str, float]) -> float:
        """EIG in bits over the hypotheses this experiment addresses.

        prior: normalized belief over ALL active hypotheses. We restrict to the
        subset this experiment predicts outcomes for, renormalize, and compute the
        entropy reduction. Feasibility scales the realized gain.
        """
        ids = [h for h in self.hypotheses_tested if h in prior]
        if not ids or not self.outcomes:
            return 0.0
        mass = sum(prior[h] for h in ids)
        if mass <= 0:
            return 0.0
        p = {h: prior[h] / mass for h in ids}
        h_prior = _entropy(p.values())

        # Marginal probability of each outcome: P(o) = sum_h P(o|h) P(h)
        expected_posterior_entropy = 0.0
        for outcome in self.outcomes:
            p_o = sum(outcome.likelihoods.get(h, 0.0) * p[h] for h in ids)
            if p_o <= 0:
                continue
            # Posterior over hypotheses given this outcome (Bayes).
            post = {
                h: outcome.likelihoods.get(h, 0.0) * p[h] / p_o for h in ids
            }
            expected_posterior_entropy += p_o * _entropy(post.values())

        gain = h_prior - expected_posterior_entropy
        return max(0.0, gain) * self.feasibility

    def value_score(self, prior: dict[str, float]) -> float:
        """EIG per lab-month — the quantity to maximize when time is the constraint."""
        eig = self.expected_information_gain(prior)
        return eig / max(self.cost, 0.5)


def _entropy(probs) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


@dataclass
class ExperimentPlan:
    """A ranked, budgeted plan produced by the planner."""

    ranked: list[tuple[Experiment, float, float]]  # (exp, eig, eig_per_month)
    chosen: list[Experiment]
    residual_entropy_bits: float
    prior_entropy_bits: float

    def describe(self) -> list[str]:
        lines = [
            f"Prior uncertainty over live hypotheses: {self.prior_entropy_bits:.2f} bits",
            f"Expected residual after chosen experiments: {self.residual_entropy_bits:.2f} bits",
            "Ranked experiments (EIG bits | EIG/month):",
        ]
        for exp, eig, per in self.ranked:
            star = "  * CHOSEN" if exp in self.chosen else ""
            lines.append(f"  - {exp.id} {exp.title} [{exp.etype.value}] "
                         f"| {eig:.3f} | {per:.3f} | {exp.cost:.0f}mo{star}")
        return lines


class DecisionTheoreticPlanner:
    """Greedy budgeted selection of maximally-discriminating experiments.

    The greedy step is myopic-optimal for submodular information gain: pick the
    highest-EIG experiment, assume its most-likely outcome updates the belief,
    then re-score the rest. This naturally avoids buying two experiments that
    discriminate the *same* pair of hypotheses.
    """

    def plan(
        self,
        experiments: list[Experiment],
        prior: dict[str, float],
        *,
        budget_months: float = 18.0,
        max_experiments: int = 4,
    ) -> ExperimentPlan:
        prior = _normalize(prior)
        prior_entropy = _entropy(prior.values())

        ranked = sorted(
            experiments,
            key=lambda e: e.value_score(prior),
            reverse=True,
        )
        ranked_view = [
            (e, e.expected_information_gain(prior), e.value_score(prior)) for e in ranked
        ]

        chosen: list[Experiment] = []
        belief = dict(prior)
        spent = 0.0
        remaining = list(experiments)
        while remaining and len(chosen) < max_experiments:
            remaining.sort(key=lambda e: e.value_score(belief), reverse=True)
            best = remaining[0]
            if best.expected_information_gain(belief) <= 1e-6:
                break
            if spent + best.cost > budget_months:
                # Skip experiments we can't afford; try the next best.
                remaining = [e for e in remaining[1:] if spent + e.cost <= budget_months]
                continue
            chosen.append(best)
            spent += best.cost
            belief = _simulate_update(best, belief)
            remaining = [e for e in remaining if e.id != best.id]

        residual = _entropy(belief.values())
        return ExperimentPlan(
            ranked=ranked_view,
            chosen=chosen,
            residual_entropy_bits=residual,
            prior_entropy_bits=prior_entropy,
        )


def _normalize(dist: dict[str, float]) -> dict[str, float]:
    total = sum(v for v in dist.values() if v > 0)
    if total <= 0:
        n = len(dist) or 1
        return {k: 1 / n for k in dist}
    return {k: max(0.0, v) / total for k, v in dist.items()}


def _simulate_update(exp: Experiment, belief: dict[str, float]) -> dict[str, float]:
    """Assume the most-probable outcome occurs and Bayes-update the belief.

    Used only to make the greedy selection diversity-aware; it is not a claim
    about what the experiment will actually show.
    """
    ids = [h for h in exp.hypotheses_tested if h in belief]
    if not ids or not exp.outcomes:
        return belief
    sub = _normalize({h: belief[h] for h in ids})
    # Most likely outcome under current sub-belief.
    def p_outcome(o: PredictedOutcome) -> float:
        return sum(o.likelihoods.get(h, 0.0) * sub[h] for h in ids)

    outcome = max(exp.outcomes, key=p_outcome)
    p_o = p_outcome(outcome) or 1.0
    updated = dict(belief)
    for h in ids:
        post = outcome.likelihoods.get(h, 0.0) * sub[h] / p_o
        updated[h] = post * sum(belief[i] for i in ids)  # keep total mass on subset
    return _normalize(updated)
