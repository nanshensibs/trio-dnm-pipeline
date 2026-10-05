"""Data compression: metacells, geometric sketching, out-of-core reference building."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from refmap.compression import geometric_sketch, incremental_reference, metacells
from refmap.mapping import map_query
from refmap.reference import build_reference
from refmap.simulate import (DEFAULT_PROPS, _sample_blocks, draw_dataset, make_world,
                             simulate_reference_query)

KEYS = ["cell_type_l1", "cell_type_l2"]


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    """Tiny matrices: BLAS/OpenMP threading only adds overhead (and oversubscribes
    shared CI machines)."""
    with threadpool_limits(1):
        yield


@pytest.fixture(scope="module")
def rare():
    """Three batches with dendritic cells at ~1% of cells."""
    world = make_world(n_genes=600, seed=0)
    rng = np.random.default_rng(1)
    props = dict(DEFAULT_PROPS, DC=0.015)
    design = []
    for b in range(3):
        design += _sample_blocks(rng, f"s{b}", f"b{b}", 1500, props)
    a = draw_dataset(world, design, seed=2)
    ref = build_reference(a, KEYS, batch_key="batch", n_hvg=300, n_pcs=20, calibrate=False)
    return a, ref


def test_geometric_sketch_keeps_rare_type(rare):
    a, ref = rare
    dc = (a.obs["cell_type_l2"] == "DC").to_numpy()
    assert 0.005 < dc.mean() < 0.02
    n = 300
    gs, un = [], []
    for s in range(5):
        idx, info = geometric_sketch(ref.Z_corr, n, seed=s, return_info=True)
        assert len(idx) == n and len(np.unique(idx)) == n
        assert n <= info["n_boxes"] <= 1.1 * n
        gs.append(dc[idx].mean())
        un.append(dc[np.random.default_rng(s).choice(len(dc), n, replace=False)].mean())
    assert np.mean(gs) > 2 * np.mean(un)
    assert np.mean(gs) > 1.5 * dc.mean()
    # fewer boxes than requested -> top up (both strategies) to exactly n cells
    for mode in ("uniform", "boxes"):
        idx = geometric_sketch(ref.Z_corr[:, :2], 2000, seed=0, max_iter=3, top_up=mode)
        assert len(np.unique(idx)) == 2000


def test_metacells_are_pure_and_batch_specific(rare):
    a, ref = rare
    mc = metacells(ref.Z_corr, cells_per_metacell=20, by="batch", adata=a, label_keys=KEYS)
    assert abs(mc.n_metacells - a.n_obs / 20) <= 3
    assert mc.purity("cell_type_l2") > 0.97 and mc.purity("cell_type_l1") > 0.99
    # metacells never mix batches
    b = a.obs["batch"].to_numpy()
    for m in np.unique(mc.membership):
        assert len(np.unique(b[mc.membership == m])) == 1
    assert (mc.obs.groupby("batch")["n_cells"].sum() == a.obs["batch"].value_counts()).all()
    # summed counts are conserved
    assert np.isclose(mc.adata.X.sum(), a.X.sum())
    assert mc.adata.obsm["X_latent"].shape == (mc.n_metacells, ref.n_dims)
    # n_metacells allocation is proportional across groups
    mc2 = metacells(ref.Z_corr, 60, by=b)
    assert mc2.n_metacells == 60 and (mc2.obs.groupby("group").size() == 20).all()


def test_incremental_reference_matches_in_memory():
    ref_ad, q, _ = simulate_reference_query(
        n_genes=600, n_ref_batches=3, ref_samples_per_batch=1, cells_per_ref_sample=600,
        n_query_ctrl=2, n_query_dis=0, cells_per_query_sample=400, seed=3)
    full = build_reference(ref_ad, KEYS, batch_key="batch", n_hvg=300, n_pcs=20)
    inc = incremental_reference(ref_ad, KEYS, chunk_size=250, batch_key="batch",
                                n_hvg=300, n_pcs=20)
    assert inc.params["incremental"]["n_chunks"] == 8
    # streamed batch-aware HVG selection and scaling reproduce the in-memory ones
    assert np.array_equal(inc.genes, full.genes)
    assert np.allclose(inc.mean, full.mean) and np.allclose(inc.sd, full.sd)
    # leading principal subspace agrees
    sv = np.linalg.svd(full.loadings[:, :5].T @ inc.loadings[:, :5], compute_uv=False)
    assert sv.min() > 0.9
    assert inc.obs.shape[0] == full.obs.shape[0] and inc.calibration
    truth = q.obs["cell_type_l2"].to_numpy()
    acc = {}
    for nm, r in (("full", full), ("inc", inc)):
        m = map_query(r, q, batch_key="batch")
        acc[nm] = (m.labels["cell_type_l2_pred"].to_numpy() == truth).mean()
    assert acc["full"] > 0.9 and abs(acc["full"] - acc["inc"]) < 0.03
    # list-of-chunks input, fixed HVGs, Harmony on a geometric sketch
    chunks = [ref_ad[i:i + 333].copy() for i in range(0, ref_ad.n_obs, 333)]
    sk = incremental_reference(chunks, KEYS, batch_key="batch", hvg_genes=full.genes,
                               n_pcs=20, harmony_sketch=600, calibrate=False)
    m = map_query(sk, q, batch_key="batch")
    assert abs((m.labels["cell_type_l2_pred"].to_numpy() == truth).mean() - acc["full"]) < 0.03
    with pytest.raises(TypeError):
        incremental_reference((c for c in chunks), KEYS)
