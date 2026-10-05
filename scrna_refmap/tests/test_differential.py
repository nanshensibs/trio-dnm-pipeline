"""Differential composition / abundance / state / prioritisation on mapped queries.

Ground truth (``simulate_reference_query``): disease donors carry an unseen
"Disease-associated Mono" state, CD14 Mono expansion (x1.8), Naive B depletion (x0.4)
and an interferon programme in CD8 T and CD14 Mono only.
"""
from __future__ import annotations

import numpy as np
import pytest

from refmap.differential import (augur, composition_test, milo, pseudobulk_de,
                                 tmm_factors)
from refmap.mapping import final_labels, map_query
from refmap.reference import build_reference
from refmap.simulate import simulate_reference_query

CASE, CONTROL = "disease", "control"


@pytest.fixture(scope="module")
def mapped():
    ref_a, q, world = simulate_reference_query(
        n_genes=500, cells_per_ref_sample=300, cells_per_query_sample=300,
        n_query_ctrl=5, n_query_dis=5, seed=0)
    ref = build_reference(ref_a, ["cell_type_l1", "cell_type_l2"], batch_key="batch",
                          sample_key="sample", n_hvg=500)
    res = map_query(ref, q, condition_key="condition")
    # transferred labels; cells in novel query clusters are reported as "Novel"
    lab = final_labels(res, "cell_type_l2")
    return dict(q=q, world=world, res=res, labels=np.asarray(lab).astype(str),
                samples=q.obs["sample"].to_numpy(), condition=q.obs["condition"].to_numpy())


def test_composition_detects_expansion_and_depletion(mapped):
    lab, s, c = mapped["labels"], mapped["samples"], mapped["condition"]
    # test the composition of reference cell types (the novel state is reported
    # separately; its 15% share would otherwise deflate every other proportion)
    m = lab != "Novel"
    pooled = composition_test(lab[m], s[m], c[m], CASE, CONTROL,
                              dispersion="pooled").set_index("cell_type")
    assert {"prop_disease", "prop_control", "log2FC", "coef", "p", "fdr"} <= set(pooled.columns)
    assert pooled.loc["CD14 Mono", "log2FC"] > 0 and pooled.loc["CD14 Mono", "fdr"] < 0.1
    assert pooled.loc["Naive B", "log2FC"] < 0 and pooled.loc["Naive B", "fdr"] < 0.1
    unaffected = pooled.drop(["CD14 Mono", "Naive B"])
    assert (unaffected["fdr"] < 0.1).sum() <= 1
    # per-cell-type quasi-binomial (R quasibinomial): same direction, CD14 Mono significant
    per = composition_test(lab[m], s[m], c[m], CASE, CONTROL).set_index("cell_type")
    assert per.loc["CD14 Mono", "fdr"] < 0.1 and per.loc["CD14 Mono", "coef"] > 0
    assert per.loc["Naive B", "coef"] < 0 and per.loc["Naive B", "p"] < 0.2
    # one X2 scale per type on 6 residual df is noisier than the pooled estimate
    assert pooled.loc["Naive B", "p"] < per.loc["Naive B", "p"]
    assert (per["dispersion"] >= 1).all()
    # with the novel cells included, the novel state is the strongest expansion
    full = composition_test(lab, s, c, CASE, CONTROL, dispersion="pooled").set_index("cell_type")
    assert full["log2FC"].idxmax() == "Novel" and full.loc["Novel", "log2FC"] > 2
    assert full.loc["Novel", "fdr"] < 0.1


def test_milo_finds_novel_disease_neighbourhoods(mapped):
    q, res = mapped["q"], mapped["res"]
    out = milo(res.Zq_corr, mapped["samples"], mapped["condition"], CASE, CONTROL,
               k=30, prop=0.1, seed=0)
    nh = out.nhoods
    assert out.membership.shape == (q.n_obs, len(nh))
    assert (nh["size"] == 31).all()
    assert np.isfinite(nh["p"]).all()
    novel_frac = out.fraction(q.obs["is_novel"].to_numpy())
    novel = novel_frac > 0.5
    sig_up = (nh["spatial_fdr"] < 0.1) & (nh["logFC"] > 0)
    assert novel.sum() >= 3
    assert sig_up[novel].mean() >= 0.8             # novel nhoods: enriched in disease
    assert sig_up.sum() > sig_up[novel].sum()      # CD14 expansion is found too
    # novel-dominated nhoods have the largest effects
    assert nh.loc[novel, "logFC"].median() >= np.quantile(nh.loc[~novel, "logFC"], 0.9)
    ann = out.annotate(mapped["labels"])
    assert (ann.loc[novel, "label"] == "Novel").mean() >= 0.8
    assert ann["label_fraction"].between(0, 1).all()
    da = out.cell_da_score(alpha=0.1)
    is_novel = q.obs["is_novel"].to_numpy()
    cd4 = q.obs["cell_type_l2"].to_numpy() == "CD4 T"
    assert da[is_novel].mean() > 1.0
    assert da[is_novel].mean() > da[cd4].mean() + 1.0


def test_pseudobulk_de_recovers_interferon_program(mapped):
    q, world = mapped["q"], mapped["world"]
    de = pseudobulk_de(q.X, q.var_names, mapped["labels"], mapped["samples"],
                       mapped["condition"], CASE, CONTROL, min_cells=10)
    assert {"cell_type", "gene", "logFC", "t", "p", "fdr"} <= set(de.columns)
    ifn = set(world.genes[world.programs["interferon"]])
    for ct in ("CD8 T", "CD14 Mono"):
        d = de[de["cell_type"] == ct]
        hits = d[d["fdr"] < 0.05]
        tp = hits["gene"].isin(ifn)
        assert tp.sum() / len(ifn) >= 0.7, ct                     # recall
        assert tp.mean() >= 0.7, ct                               # precision
        assert (hits.loc[tp, "logFC"] > 0).all()
    for ct in ("CD4 T", "NK"):
        d = de[de["cell_type"] == ct]
        assert d["gene"].isin(ifn).any()                          # genes were tested
        assert (d["fdr"] < 0.05).sum() <= 3, ct
        assert not (d.loc[d["gene"].isin(ifn), "fdr"] < 0.05).any(), ct


def test_tmm_factors_are_neutral_without_composition_shift():
    rng = np.random.default_rng(0)
    mu = rng.gamma(1.0, 50.0, 400)
    Y = rng.poisson(np.outer(mu, [1.0, 2.0, 0.5, 1.5]))
    f = tmm_factors(Y)
    assert np.allclose(f, 1.0, atol=0.05)
    # a strong programme in one sample inflates its library; TMM effective library
    # sizes recover the scale of the untouched genes
    Y2 = Y.copy()
    Y2[:40, 0] *= 8
    eff = Y2.sum(axis=0) * tmm_factors(Y2)
    truth = Y[40:].sum(axis=0)
    assert np.allclose(eff / eff[1], truth / truth[1], rtol=0.05)
    raw = Y2.sum(axis=0)
    assert abs(raw[0] / raw[1] - truth[0] / truth[1]) > 0.2


def test_augur_prioritises_responding_cell_types(mapped):
    lab, c = mapped["labels"], mapped["condition"]
    keep = lab != "Novel"
    au = augur(mapped["q"].X[keep], lab[keep], c[keep], n_subsamples=5,
               subsample_size=20, folds=3, n_estimators=50, seed=0)
    au = au.set_index("cell_type")
    assert au.index[0] in ("CD14 Mono", "CD8 T")
    unaffected = au.drop(["CD14 Mono", "CD8 T"], errors="ignore")
    assert len(unaffected) >= 4
    assert au.loc["CD14 Mono", "auc"] > unaffected["auc"].max() + 0.2
    assert au.loc["CD8 T", "auc"] > unaffected["auc"].max() + 0.2
    assert au.loc[["CD14 Mono", "CD8 T"], "auc"].min() > 0.85
