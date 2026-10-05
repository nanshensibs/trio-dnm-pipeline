"""Core reference building, Symphony mapping, label transfer and novelty detection."""
import numpy as np
import pandas as pd
import pytest

from refmap.harmony import run_harmony, symphony_map
from refmap.mapping import final_labels, map_query
from refmap.metrics import evaluate_mapping, mixing_entropy
from refmap.reference import Reference, build_reference
from refmap.simulate import simulate_reference_query
from refmap.transfer import knn_weights, transfer_continuous, transfer_hierarchical

LEVELS = ["cell_type_l1", "cell_type_l2"]


@pytest.fixture(scope="module")
def data():
    ref_ad, q_ad, world = simulate_reference_query(
        n_genes=600, n_ref_batches=3, ref_samples_per_batch=2, cells_per_ref_sample=350,
        n_query_ctrl=2, n_query_dis=2, cells_per_query_sample=400, seed=3)
    # a continuous reference annotation to transfer (e.g. a spatial coordinate)
    rng = np.random.default_rng(0)
    centre = {ct: rng.normal(0, 5, 2) for ct in ref_ad.obs.cell_type_l2.unique()}
    ref_ad.obsm["spatial"] = np.vstack([centre[c] for c in ref_ad.obs.cell_type_l2]) + \
        rng.normal(0, 0.3, (ref_ad.n_obs, 2))
    ref = build_reference(ref_ad, LEVELS, batch_key="batch", sample_key="sample",
                          n_hvg=600, n_pcs=20, continuous_obsm=("spatial",))
    res = map_query(ref, q_ad, condition_key="condition")
    return ref_ad, q_ad, ref, res, centre


def test_harmony_removes_batch_but_keeps_biology(data):
    ref_ad, _, ref, _, _ = data
    b = ref_ad.obs["batch"].to_numpy()
    lab = ref_ad.obs["cell_type_l2"].to_numpy()
    assert mixing_entropy(ref.Z_corr, b) > mixing_entropy(ref.Z, b) + 0.05
    assert mixing_entropy(ref.Z_corr, b) > 0.75
    assert mixing_entropy(ref.Z_corr, lab) < 0.2      # cell types stay separated


def test_symphony_reference_anchored_correction():
    from refmap.harmony import compress_reference
    rng = np.random.default_rng(1)
    centres = np.array([[0, 0, 0, 0, 0], [3, 3, 0, 0, 0]], float)
    shift = np.array([1.0, 0, 0, 0, 0])
    Z, batch, ctype = [], [], []
    for b, off in (("a", 0.0), ("b", 1.0)):
        for c in (0, 1):
            Z.append(rng.normal(centres[c] + off * shift, 0.3, (150, 5)))
            batch += [b] * 150
            ctype += [c] * 150
    Z, ctype = np.vstack(Z), np.array(ctype)
    h = run_harmony(Z, np.array(batch), n_clusters=6)
    comp, _ = compress_reference(h.Z_corr, h.Y, 0.1, 1.0)
    target = h.Z_corr[ctype == 0].mean(0)
    Zq = rng.normal(centres[0] + 2.5 * shift, 0.3, (100, 5))      # unseen lab, big shift
    Zc, Rq = symphony_map(Zq, comp, None)
    assert np.linalg.norm(Zc.mean(0) - target) < 0.5 * np.linalg.norm(Zq.mean(0) - target)
    assert np.allclose(Rq.sum(axis=0), 1)


def test_reference_calibration(data):
    ref = data[2]
    cal = ref.calibration
    assert cal["scheme"] == "leave-one-batch-out"
    assert cal["self_mapping_accuracy"]["cell_type_l1"] > 0.95
    assert cal["self_mapping_accuracy"]["cell_type_l2"] > 0.9
    assert len(cal["null"]["cluster_mahalanobis"]) > 5
    assert ref.hierarchy()["cell_type_l2"]["CD8 T"] == "T cell"


def test_label_transfer_and_hierarchy(data):
    _, q, ref, res, _ = data
    known = ~q.obs["is_novel"].to_numpy()
    for lv in LEVELS:
        acc = (res.labels[f"{lv}_pred"].to_numpy()[known] == q.obs[lv].to_numpy()[known]).mean()
        assert acc > 0.9, (lv, acc)
    # predictions respect the hierarchy
    h = ref.hierarchy()["cell_type_l2"]
    l1 = res.labels["cell_type_l1_pred"].to_numpy()
    l2 = res.labels["cell_type_l2_pred"].to_numpy()
    ok = [h.get(b, b) == a or b == a for a, b in zip(l1, l2)]
    assert all(ok)


def test_hierarchical_rejection_to_parent():
    obs = pd.DataFrame({"l1": ["T"] * 4 + ["B"] * 2, "l2": ["CD4", "CD4", "CD8", "CD8", "B", "B"]})
    idx = np.array([[0, 2, 1, 3]])         # neighbours split evenly between CD4 and CD8
    w = np.full((1, 4), 0.25)
    out = transfer_hierarchical(obs, ["l1", "l2"], idx, w, unknown_threshold=0.4)
    assert out.loc[0, "l1_pred"] == "T" and out.loc[0, "l2_pred"] == "T"
    assert not out.loc[0, "l2_resolved"]


def test_scarches_kernel():
    d = np.array([[1.0, 1.0, 5.0]])
    w = knn_weights(d)
    assert np.isclose(w.sum(), 1) and w[0, 0] > w[0, 2]


def test_continuous_transfer(data):
    _, q, ref, res, centre = data
    known = ~q.obs["is_novel"].to_numpy()
    pred = res.continuous["spatial"].to_numpy()[known]
    truth = np.vstack([centre[c] for c in q.obs["cell_type_l2"][known]])
    assert np.median(np.linalg.norm(pred - truth, axis=1)) < 1.0
    mu, sd = transfer_continuous(np.array([[0.0], [2.0]]), np.array([[0, 1]]),
                                 np.array([[0.5, 0.5]]))
    assert np.isclose(mu[0, 0], 1.0) and np.isclose(sd[0, 0], 1.0)


def test_novel_disease_state_flagged(data):
    _, q, ref, res, _ = data
    s = res.cluster_summary.set_index("cluster")
    novel_cells = q.obs["is_novel"].to_numpy()
    cl_novel_frac = pd.Series(novel_cells).groupby(res.clusters).mean()
    true_novel_clusters = set(cl_novel_frac[cl_novel_frac > 0.5].index)
    flagged = set(s.index[s["novel"]])
    assert true_novel_clusters and true_novel_clusters <= flagged
    assert flagged <= true_novel_clusters        # no known cluster flagged
    lab = final_labels(res, "cell_type_l2")
    assert (lab[novel_cells] == "Novel").mean() > 0.8
    m = evaluate_mapping(ref, res, q.obs, novel_key="is_novel")
    assert m["criterion3_novel_states"]["novel_recall"] > 0.8
    assert m["criterion3_novel_states"]["auroc_knn_distance"] > 0.6
    assert m["criterion2_integration"]["ref_query_mixing_mean"] > 0.3
    assert m["criterion2_integration"]["centroid_offset_mean"] < 0.3


def test_save_load_roundtrip(data, tmp_path):
    _, q, ref, res, _ = data
    ref.save(str(tmp_path / "ref"))
    ref2 = Reference.load(str(tmp_path / "ref"))
    assert ref2.content_hash() == ref.content_hash()
    assert ref2.label_keys == LEVELS and ref2.version == "1.0.0"
    res2 = map_query(ref2, q)
    assert (res2.labels["cell_type_l2_pred"] == res.labels["cell_type_l2_pred"]).all()


def test_missing_query_genes_are_zero_filled(data):
    _, q, ref, _, _ = data
    sub = q[:, q.var_names[50:]].copy()
    res = map_query(ref, sub)
    assert 0 < res.frac_missing_genes < 0.2
    known = ~sub.obs["is_novel"].to_numpy()
    acc = (res.labels["cell_type_l1_pred"].to_numpy()[known] ==
           sub.obs["cell_type_l1"].to_numpy()[known]).mean()
    assert acc > 0.9


def test_supervised_pca_reference(data):
    ref_ad, q, _, _, _ = data
    ref = build_reference(ref_ad, LEVELS, batch_key="batch", n_hvg=600, n_pcs=20,
                          transform="spca", calibrate=False)
    res = map_query(ref, q)
    known = ~q.obs["is_novel"].to_numpy()
    acc = (res.labels["cell_type_l2_pred"].to_numpy()[known] ==
           q.obs["cell_type_l2"].to_numpy()[known]).mean()
    assert acc > 0.9


def test_inconsistent_hierarchy_rejected(data):
    ref_ad = data[0].copy()
    ref_ad.obs["bad_l1"] = ref_ad.obs["cell_type_l1"].astype(str)
    ref_ad.obs.iloc[0, ref_ad.obs.columns.get_loc("bad_l1")] = "B cell"
    ref_ad.obs.iloc[1, ref_ad.obs.columns.get_loc("bad_l1")] = "T cell"
    with pytest.raises(ValueError):
        build_reference(ref_ad, ["bad_l1", "cell_type_l2"], calibrate=False)
