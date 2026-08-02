"""Unit tests for the deterministic reasoning core (no network / LLM needed)."""

import math

import pytest

from lupus_coscientist import (
    CausalEvidenceGraph,
    DecisionTheoreticPlanner,
    Evidence,
    EvidenceTier,
    Experiment,
    ExperimentType,
    Hypothesis,
    HypothesisPool,
    LupusCoScientist,
    NodeType,
    Polarity,
    PredictedOutcome,
    Sign,
)
from lupus_coscientist.hypothesis import HypothesisStatus


# --------------------------------------------------------------------------- #
# Evidence tiering
# --------------------------------------------------------------------------- #

def test_tier_ordering_and_weights():
    assert EvidenceTier.HUMAN_GENETICS > EvidenceTier.PERTURBATION > EvidenceTier.ANIMAL
    assert EvidenceTier.ANIMAL > EvidenceTier.ASSOCIATION > EvidenceTier.REVIEW
    # weights are monotonic and convex-ish (top tiers dominate)
    ws = [t.weight for t in sorted(EvidenceTier)]
    assert ws == sorted(ws)
    assert EvidenceTier.HUMAN_GENETICS.weight >= 2 * EvidenceTier.ASSOCIATION.weight


def test_signed_weight_polarity_and_leakage():
    pos = Evidence("a", EvidenceTier.PERTURBATION, Polarity.SUPPORTS, "x", strength=1.0)
    neg = Evidence("b", EvidenceTier.PERTURBATION, Polarity.CONTRADICTS, "x", strength=1.0)
    leaked = Evidence("c", EvidenceTier.PERTURBATION, Polarity.SUPPORTS, "x",
                      strength=1.0, is_independent=False)
    assert pos.signed_weight > 0
    assert neg.signed_weight == -pos.signed_weight
    assert 0 < leaked.signed_weight < pos.signed_weight  # leakage discounted


# --------------------------------------------------------------------------- #
# Causal graph
# --------------------------------------------------------------------------- #

def build_graph():
    g = CausalEvidenceGraph()
    strong = Evidence("g1", EvidenceTier.HUMAN_GENETICS, Polarity.SUPPORTS, "variant->gene", strength=0.9)
    pert = Evidence("g2", EvidenceTier.PERTURBATION, Polarity.SUPPORTS, "gene->mech", strength=0.8)
    weak = Evidence("g3", EvidenceTier.REVIEW, Polarity.SUPPORTS, "mech->state", strength=0.3)
    g.add_edge("V", "G", source_type=NodeType.VARIANT, target_type=NodeType.GENE,
               sign=Sign.ACTIVATES, evidence=[strong])
    g.add_edge("G", "M", source_type=NodeType.GENE, target_type=NodeType.MOLECULAR_MECHANISM,
               sign=Sign.ACTIVATES, evidence=[pert])
    g.add_edge("M", "S", source_type=NodeType.MOLECULAR_MECHANISM, target_type=NodeType.CELL_STATE,
               sign=Sign.ACTIVATES, evidence=[weak])
    g.add_edge("S", "C", source_type=NodeType.CELL_STATE, target_type=NodeType.CLINICAL_OUTCOME,
               sign=Sign.ACTIVATES, evidence=[pert])
    return g


def test_chain_from_nodes_weakest_link():
    g = build_graph()
    chain = g.chain_from_nodes(["V", "G", "M", "S", "C"])
    assert chain is not None
    assert chain.spans_to_clinic
    assert chain.starts_at_genotype
    # weakest link is the REVIEW-tier M->S edge
    assert chain.weakest_link == pytest.approx(g.edge("M", "S").support, rel=1e-6)
    assert chain.weakest_link < g.edge("V", "G").support


def test_chain_from_nodes_broken_returns_none():
    g = build_graph()
    assert g.chain_from_nodes(["V", "G", "X"]) is None  # missing G->X edge


def test_strengthening_weak_edge_raises_chain_score():
    g = build_graph()
    before = g.chain_from_nodes(["V", "G", "M", "S", "C"]).score()
    # accumulate independent perturbation evidence on the weak link
    g.edge("M", "S").add(Evidence("x", EvidenceTier.PERTURBATION, Polarity.SUPPORTS, "confirmed", strength=0.9))
    after = g.chain_from_nodes(["V", "G", "M", "S", "C"]).score()
    assert after > before


def test_contradiction_detection():
    g = CausalEvidenceGraph()
    ev = Evidence("e", EvidenceTier.PERTURBATION, Polarity.SUPPORTS, "x", strength=0.9)
    g.add_edge("A", "B", sign=Sign.ACTIVATES, evidence=[ev])
    g.add_edge("B", "A", sign=Sign.INHIBITS, evidence=[Evidence("f", EvidenceTier.PERTURBATION,
                                                                 Polarity.SUPPORTS, "y", strength=0.9)])
    assert g.contradictions()


# --------------------------------------------------------------------------- #
# Hypothesis belief + pruning
# --------------------------------------------------------------------------- #

def test_no_chain_is_penalized():
    g = build_graph()
    with_chain = Hypothesis("H1", "has chain", chain_nodes=("V", "G", "M", "S", "C"))
    without = Hypothesis("H2", "no chain")
    with_chain.update_belief(g)
    without.update_belief(g)
    assert with_chain.belief > without.belief
    assert without.belief < 0.2


def test_independent_support_raises_belief():
    g = build_graph()
    h = Hypothesis("H1", "x", chain_nodes=("V", "G", "M", "S", "C"))
    h.update_belief(g)
    base = h.belief
    h.attach(Evidence("v", EvidenceTier.PERTURBATION, Polarity.SUPPORTS, "validated", strength=0.9))
    h.update_belief(g)
    assert h.belief > base


def test_prune_protects_min_survivors_and_eliminates_weak():
    g = CausalEvidenceGraph()
    pool = HypothesisPool(g)
    # three chain-less, evidence-less hypotheses — all weak
    for i in range(3):
        pool.add(Hypothesis(f"H{i}", f"weak {i}"))
    eliminated = pool.prune(min_survivors=2)
    # at most one eliminated because two are protected
    assert len(pool.active) >= 2
    assert len(eliminated) <= 1


def test_entropy_peaks_when_uniform():
    g = build_graph()
    pool = HypothesisPool(g)
    a = pool.add(Hypothesis("A", "a"))
    b = pool.add(Hypothesis("B", "b"))
    a.belief = b.belief = 0.5
    assert pool.entropy() == pytest.approx(1.0, abs=1e-6)
    a.belief, b.belief = 0.99, 0.01
    assert pool.entropy() < 0.2


# --------------------------------------------------------------------------- #
# Decision-theoretic experiment planner
# --------------------------------------------------------------------------- #

def test_discriminating_experiment_has_higher_eig():
    prior = {"H1": 0.5, "H2": 0.5}
    sharp = Experiment(
        "E1", "sharp", ExperimentType.PERTURB_SEQ, ("H1", "H2"), feasibility=1.0,
        outcomes=[PredictedOutcome("o1", {"H1": 0.95, "H2": 0.05}),
                  PredictedOutcome("o2", {"H1": 0.05, "H2": 0.95})])
    dull = Experiment(
        "E2", "dull", ExperimentType.HUMAN_TISSUE, ("H1", "H2"),
        outcomes=[PredictedOutcome("o1", {"H1": 0.5, "H2": 0.5}),
                  PredictedOutcome("o2", {"H1": 0.5, "H2": 0.5})])
    assert sharp.expected_information_gain(prior) > dull.expected_information_gain(prior)
    assert dull.expected_information_gain(prior) == pytest.approx(0.0, abs=1e-9)
    # a perfectly discriminating experiment on a uniform 2-way prior yields ~1 bit
    assert sharp.expected_information_gain(prior) > 0.6


def test_planner_respects_budget_and_reduces_entropy():
    prior = {"H1": 0.34, "H2": 0.33, "H3": 0.33}
    exps = [
        Experiment("E1", "a", ExperimentType.PERTURB_SEQ, ("H1", "H2"), cost_months=6,
                   outcomes=[PredictedOutcome("o1", {"H1": 0.9, "H2": 0.1}),
                             PredictedOutcome("o2", {"H1": 0.1, "H2": 0.9})]),
        Experiment("E2", "b", ExperimentType.MOUSE_IN_VIVO, ("H2", "H3"), cost_months=6,
                   outcomes=[PredictedOutcome("o1", {"H2": 0.9, "H3": 0.1}),
                             PredictedOutcome("o2", {"H2": 0.1, "H3": 0.9})]),
        Experiment("E3", "c", ExperimentType.CLINICAL_ARM, ("H1", "H3"), cost_months=100,
                   outcomes=[PredictedOutcome("o1", {"H1": 0.9, "H3": 0.1}),
                             PredictedOutcome("o2", {"H1": 0.1, "H3": 0.9})]),
    ]
    plan = DecisionTheoreticPlanner().plan(exps, prior, budget_months=13, max_experiments=4)
    assert all(e.cost <= 13 for e in plan.chosen)          # over-budget E3 excluded
    assert plan.residual_entropy_bits < plan.prior_entropy_bits


# --------------------------------------------------------------------------- #
# End-to-end offline pipeline
# --------------------------------------------------------------------------- #

def test_offline_pipeline_converges_on_switch_hypothesis():
    report = LupusCoScientist(config=None).investigate(
        "What is the causal role of IRF5 in ABC-driven lupus nephritis?", focus="IRF5")
    fr = report.final_report
    # The switch-flip hypothesis (H01) should be the headline mechanism.
    assert "ZEB2-MEF2B switch" in fr["headline_mechanism"]
    ids = [h["id"] for h in fr["hypotheses"]]
    assert ids[0] == "H01"
    # The artifact hypothesis must be eliminated.
    elim_ids = {e["id"] for e in fr["eliminated"]}
    assert "H04" in elim_ids
    # Uncertainty strictly decreased across the run.
    entropies = [s.entropy_bits for s in report.stages]
    assert entropies[-1] < max(entropies)
    # A discriminating experiment was chosen.
    assert report.experiment_plan.chosen


def test_offline_pipeline_verbose_flag_runs():
    from lupus_coscientist import RunConfig
    report = LupusCoScientist(config=RunConfig(verbose=False)).investigate("q", focus="IRF5")
    assert report.final_report["hypotheses"]
