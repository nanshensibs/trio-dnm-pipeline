"""Cross-modality mapping (Figure 3A): gene activity + CCA anchors vs bridge integration."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from refmap.crossmodal import (bridge_integration, cca_anchor_transfer, gene_activity,
                               gene_activity_transfer, lsi_fit)
from refmap.preprocess import normalize_total_log1p, to_dense
from refmap.simulate_modalities import simulate_multiome


@pytest.fixture(scope="module", autouse=True)
def _single_thread_blas():
    # small matrices: multithreaded BLAS only adds contention on loaded machines
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:          # pragma: no cover
        yield
        return
    with threadpool_limits(1):
        yield


@pytest.fixture(scope="module")
def data():
    return simulate_multiome(n_genes=600, cells_per_ref_batch=500, n_query=600, n_bridge=300,
                             seed=0)


@pytest.fixture(scope="module")
def ga_result(data):
    return gene_activity_transfer(data.ref_rna, data.ref_rna.obs["cell_type_l2"].to_numpy(),
                                  data.query_atac, data.gene_coords, ref_batch_key="batch")


def test_gene_activity_window_is_strand_aware():
    genes = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [10_000, 50_000],
                          "end": [20_000, 60_000], "strand": ["+", "-"]},
                         index=["gplus", "gminus"])
    peaks = pd.DataFrame({"chrom": "chr1",
                          "start": [8_500, 7_000, 15_000, 20_500, 49_000, 61_000, 63_000],
                          "end": [9_000, 7_500, 15_500, 21_000, 49_500, 61_500, 63_500]})
    # peaks: +upstream, too far upstream, +body, past +TES, past -TES, -upstream, too far
    X = sp.csr_matrix(np.array([[1, 2, 4, 8, 16, 32, 64]], float))
    atac = ad.AnnData(X=X, var=peaks.set_axis([f"p{i}" for i in range(7)]))
    ga = gene_activity(atac, genes, upstream=2000)
    assert list(ga.var_names) == ["gplus", "gminus"]
    np.testing.assert_allclose(to_dense(ga.X), [[1 + 4, 32]])
    assert ga.var["n_peaks"].tolist() == [2, 1]


def test_gene_activity_tracks_rna(data):
    ga = gene_activity(data.bridge_atac, data.gene_coords)
    R = to_dense(normalize_total_log1p(data.bridge_rna.X))
    G = to_dense(normalize_total_log1p(ga.X))
    markers = np.unique(np.concatenate([v for k, v in data.world.programs.items()
                                        if k.startswith(("coarse:", "fine:"))]))
    t = data.bridge_rna.obs["cell_type_l2"].to_numpy()
    Rm = pd.DataFrame(R[:, markers]).groupby(t).mean().to_numpy()
    Gm = pd.DataFrame(G[:, markers]).groupby(t).mean().to_numpy()
    pb = np.array([np.corrcoef(Rm[:, j], Gm[:, j])[0, 1] for j in range(len(markers))])
    cell = np.array([np.corrcoef(R[:, j], G[:, j])[0, 1] for j in markers])
    assert np.median(pb) > 0.4                       # cell-type level: clearly coupled
    assert np.nanmean(cell) > 0.02 and (cell > 0).mean() > 0.6   # single cells: weak but +
    assert np.median(pb) < 0.99                      # ...and imperfect (coupling < 1)


def test_lsi_drops_depth_component_and_projects(data):
    lsi = lsi_fit(data.bridge_atac.X, data.bridge_atac.var_names, 20)
    assert not lsi.keep[0] and abs(lsi.depth_cor[0]) > 0.5
    proj = lsi.transform(data.bridge_atac.X, data.bridge_atac.var_names)
    np.testing.assert_allclose(proj, lsi.embedding, atol=1e-6)


def test_cca_anchors_same_modality(data):
    Xr = normalize_total_log1p(data.ref_rna.X)
    Xq = normalize_total_log1p(data.bridge_rna.X)
    genes = np.asarray(data.ref_rna.var_names)
    res = cca_anchor_transfer(Xr, data.ref_rna.obs["cell_type_l2"], Xq, genes)
    assert (res.pred == data.bridge_rna.obs["cell_type_l2"].to_numpy()).mean() > 0.95
    s = res.anchors["score"].to_numpy()
    assert s.min() >= 0 and s.max() <= 1 and np.isclose(s.max(), 1.0)
    np.testing.assert_allclose(np.asarray(res.weights.sum(axis=1)).ravel(), 1.0)
    np.testing.assert_allclose(res.probs.sum(axis=1), 1.0)


def test_gene_activity_cca_transfer_to_atac(data, ga_result):
    res, ga = ga_result
    q1 = data.query_atac.obs["cell_type_l1"].to_numpy()
    q2 = data.query_atac.obs["cell_type_l2"].to_numpy()
    p1, s1, _ = res.transfer_labels(data.ref_rna.obs["cell_type_l1"])
    assert (p1 == q1).mean() >= 0.75                 # measured 0.85
    assert (res.pred == q2).mean() >= 0.55           # measured 0.72
    assert len(res.anchors) < res.n_anchors_unfiltered       # filtering removed anchors
    ok = res.pred == q2
    assert res.score[ok].mean() > res.score[~ok].mean()       # score reflects reliability
    # continuous transfer: impute reference RNA into ATAC cells
    Xr = to_dense(normalize_total_log1p(data.ref_rna.X))
    imp = res.impute(Xr)
    assert imp.shape == (data.query_atac.n_obs, data.ref_rna.n_vars)
    cent = pd.DataFrame(Xr).groupby(data.ref_rna.obs["cell_type_l1"].to_numpy()).mean()
    C = np.corrcoef(np.vstack([imp, cent.to_numpy()]))[:imp.shape[0], imp.shape[0]:]
    assert (cent.index.to_numpy()[C.argmax(axis=1)] == q1).mean() >= 0.75


def test_bridge_integration_beats_feature_conversion(data, ga_result):
    res, _ = ga_result
    q1 = data.query_atac.obs["cell_type_l1"].to_numpy()
    q2 = data.query_atac.obs["cell_type_l2"].to_numpy()
    br = bridge_integration(data.ref_rna, data.ref_rna.obs[["cell_type_l1", "cell_type_l2"]],
                            data.bridge_rna, data.bridge_atac, data.query_atac)
    lab = br.labels
    acc1 = (lab["cell_type_l1_pred"].to_numpy() == q1).mean()
    acc2 = (lab["cell_type_l2_pred"].to_numpy() == q2).mean()
    p1, _, _ = res.transfer_labels(data.ref_rna.obs["cell_type_l1"])
    assert acc1 >= 0.85                              # measured 0.92
    assert acc1 >= (p1 == q1).mean() - 0.02          # >= feature conversion (0.85)
    assert acc2 >= 0.65
    assert br.lsi_dropped_first
    assert br.Z_ref.shape[1] == br.Z_query.shape[1] == br.Z_bridge.shape[1]
    ok = lab["cell_type_l1_pred"].to_numpy() == q1
    unc = lab["cell_type_l1_uncertainty"].to_numpy()
    assert unc[~ok].mean() > unc[ok].mean() + 0.1    # uncertainty flags errors
