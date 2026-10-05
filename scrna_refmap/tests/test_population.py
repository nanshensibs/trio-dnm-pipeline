"""Population-scale mapping (Figure 2): sample embeddings, similarity, sample
classification and multi-instance attribution.

Ground truth (``simulate_cohort``): subtype B samples expand Memory B (x2.5) and
switch on the ``subtype_B`` programme in CD14 Mono only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from refmap.mapping import map_query
from refmap.population import (classify_samples, mil_attribution, sample_representation,
                               sample_similarity)
from refmap.reference import build_reference
from refmap.simulate import simulate_cohort


@pytest.fixture(scope="module")
def cohort():
    ref_a, q, world = simulate_cohort(n_genes=500, n_samples=16, cells_per_sample=200,
                                      n_query_samples=8, seed=0)
    ref = build_reference(ref_a, ["cell_type_l1", "cell_type_l2"], batch_key="batch",
                          sample_key="sample", n_hvg=500)
    res = map_query(ref, q)
    # reference samples: integrated reference embedding + curated labels;
    # query samples: joint-space embedding + transferred labels
    rep_ref = sample_representation(ref.Z_corr, ref.labels("cell_type_l2"),
                                    ref.obs["sample"].to_numpy())
    rep_q = sample_representation(res.Zq_corr, res.labels["cell_type_l2_pred"].to_numpy(),
                                  q.obs["sample"].to_numpy(), impute_from=rep_ref)
    y_ref = ref_a.obs.groupby("sample")["phenotype"].first()
    y_q = q.obs.groupby("sample")["phenotype"].first()
    return dict(ref=ref, res=res, q=q, rep_ref=rep_ref, rep_q=rep_q, y_ref=y_ref, y_q=y_q)


def test_sample_representation_layout(cohort):
    rr, rq = cohort["rep_ref"], cohort["rep_q"]
    d = cohort["ref"].n_dims
    n_t = len(rr.cell_types)
    assert rr.table.shape == (16, n_t * (1 + d))
    assert list(rq.table.columns) == list(rr.table.columns)
    assert rq.table.shape[0] == 8
    comp = rr.block("composition")
    assert np.allclose(comp.sum(axis=1), 1.0)
    assert set(rr.features["block"]) == {"composition", "latent"}
    assert np.isfinite(rq.table.to_numpy()).all()
    # Memory B is expanded in subtype B
    mb = comp["prop|Memory B"]
    y = cohort["y_ref"].reindex(comp.index)
    assert mb[y == "subtype_B"].mean() > 1.5 * mb[y == "subtype_A"].mean()


def test_missing_cell_types_are_imputed():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 3))
    labels = np.array(["A"] * 30 + ["B"] * 30)
    samples = np.array(["s1"] * 15 + ["s2"] * 15 + ["s1"] * 28 + ["s2"] * 2)
    rep = sample_representation(Z, labels, samples, min_cells=5)
    assert rep.imputed.loc["s2", "B"] and not rep.imputed.loc["s1", "B"]
    b = [f"B|z{i}" for i in range(3)]
    assert np.allclose(rep.table.loc["s2", b], rep.table.loc["s1", b])


def test_sample_similarity(cohort):
    sim = sample_similarity(cohort["rep_q"], cohort["rep_ref"], metric="correlation")
    S = sim.similarity
    assert S.shape == (8, 16)
    assert sorted(sim.query_order) == sorted(S.index)
    assert sorted(sim.ref_order) == sorted(S.columns)
    y_q, y_ref = cohort["y_q"], cohort["y_ref"]
    same = np.array([[y_q[a] == y_ref[b] for b in S.columns] for a in S.index])
    assert S.to_numpy()[same].mean() > S.to_numpy()[~same].mean()
    nearest = sim.nearest["nearest_reference"].map(y_ref)
    assert (nearest == y_q.reindex(nearest.index)).mean() >= 0.75


def test_classify_query_samples(cohort):
    out = classify_samples(cohort["rep_ref"], cohort["y_ref"], cohort["rep_q"],
                           model="logistic", cv="loo")
    assert out.cv_accuracy >= 0.75 and out.cv_auc >= 0.8
    qp = out.query_predictions
    acc = (qp["predicted"] == cohort["y_q"].reindex(qp.index)).mean()
    assert acc >= 0.75
    assert np.allclose(qp[[f"P({c})" for c in out.classes]].sum(axis=1), 1.0)
    # the most informative features point at the responsible populations
    top = out.feature_weights.index[:10]
    assert any(t.startswith(("CD14 Mono|", "prop|Memory B")) for t in top)


def test_mil_attribution_localises_responsible_populations(cohort):
    ref = cohort["ref"]
    out = mil_attribution(ref.Z_corr, ref.obs["sample"].to_numpy(), cohort["y_ref"],
                          ref.labels("cell_type_l2"), positive="subtype_B", seed=0)
    assert out.cell_scores.shape == (ref.n_cells,) and np.isfinite(out.cell_scores).all()
    assert out.bag_auc >= 0.9
    att = out.attribution.set_index("cell_type")
    top2 = list(out.attribution["cell_type"][:2])
    assert top2[0] in ("CD14 Mono", "Memory B")
    assert set(top2) == {"CD14 Mono", "Memory B"}
    # CD14 Mono: state signal (scores shift within the type); Memory B: abundance
    assert att["score_diff"].idxmax() == "CD14 Mono"
    assert att.loc["CD14 Mono", "p"] < 0.01
    assert att["delta_proportion"].idxmax() == "Memory B"
    # top-quantile aggregation also separates the bags
    out_q = mil_attribution(ref.Z_corr, ref.obs["sample"].to_numpy(), cohort["y_ref"],
                            ref.labels("cell_type_l2"), positive="subtype_B",
                            aggregate="quantile", seed=0)
    assert out_q.bag_auc >= 0.9


def test_mil_on_joint_reference_and_query(cohort):
    ref, res, q = cohort["ref"], cohort["res"], cohort["q"]
    Z = np.vstack([ref.Z_corr, res.Zq_corr])
    samples = np.r_[ref.obs["sample"].to_numpy(), q.obs["sample"].to_numpy()]
    labels = np.r_[ref.labels("cell_type_l2"), res.labels["cell_type_l2_pred"].to_numpy()]
    y = pd.concat([cohort["y_ref"], cohort["y_q"]])
    out = mil_attribution(Z, samples, y, labels, positive="subtype_B", seed=1)
    assert out.bag_auc >= 0.9
    assert out.attribution["cell_type"].iloc[0] in ("CD14 Mono", "Memory B")
