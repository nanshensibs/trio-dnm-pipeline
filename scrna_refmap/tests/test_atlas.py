"""Open-source atlasing: registry, reviewed updates, cross-reference harmonisation."""
import json
import os

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from refmap.atlas import (ReferenceRegistry, accept_update, bump_version, harmonize_references,
                          propose_update)
from refmap.reference import build_reference
from refmap.simulate import simulate_reference_query

KEYS = ["cell_type_l1", "cell_type_l2"]
NEW = "Disease-associated Mono"


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    """Tiny matrices: BLAS/OpenMP threading only adds overhead (and oversubscribes
    shared CI machines)."""
    with threadpool_limits(1):
        yield


@pytest.fixture(scope="module")
def data():
    ref_ad, q, _ = simulate_reference_query(
        n_genes=600, n_ref_batches=4, ref_samples_per_batch=1, cells_per_ref_sample=350,
        n_query_ctrl=1, n_query_dis=1, cells_per_query_sample=400, seed=0)
    return ref_ad, q


@pytest.fixture(scope="module")
def base(data):
    ref_ad, _ = data
    sub = ref_ad[ref_ad.obs["batch"].isin(["ref_batch0", "ref_batch1", "ref_batch2"])].copy()
    ref = build_reference(sub, KEYS, batch_key="batch", sample_key="sample", n_hvg=200,
                          n_pcs=20, name="pbmc")
    return sub, ref


def test_semver():
    assert bump_version("1.2.3", "patch") == "1.2.4"
    assert bump_version("1.2.3", "minor") == "1.3.0"
    assert bump_version("1.2.3", "major") == "2.0.0"
    with pytest.raises(ValueError):
        bump_version("1.2", "minor")


def test_registry_roundtrip(base, tmp_path):
    _, ref = base
    reg = ReferenceRegistry(str(tmp_path))
    path = reg.publish(ref)
    assert path == os.path.join(str(tmp_path), "pbmc", "1.0.0")
    assert reg.names() == ["pbmc"] and reg.versions("pbmc") == ["1.0.0"]
    assert reg.latest("pbmc") == "1.0.0"
    loaded = reg.load("pbmc")
    assert loaded.content_hash() == ref.content_hash() == reg.entry("pbmc")["content_sha256"]
    assert np.allclose(loaded.Z_corr, ref.Z_corr) and loaded.label_keys == KEYS
    idx = json.load(open(tmp_path / "index.json"))
    assert idx["references"]["pbmc"]["1.0.0"]["content_sha256"] == ref.content_hash()
    assert reg.publish(ref) == path                  # identical content: idempotent
    loaded.obs.iloc[0, 1] = "CD8 T"                   # same version, different content
    with pytest.raises(ValueError):
        reg.publish(loaded)
    idx["references"]["pbmc"]["1.0.0"]["content_sha256"] = "0" * 64
    json.dump(idx, open(tmp_path / "index.json", "w"))
    with pytest.raises(ValueError, match="hash"):
        reg.load("pbmc")
    with pytest.raises(KeyError):
        reg.latest("missing")


def test_propose_update_with_new_label(base, data, tmp_path):
    _, ref = base
    _, q = data
    reg = ReferenceRegistry(str(tmp_path))
    reg.publish(ref)
    prop = propose_update(ref, q, KEYS, batch_key="batch", sample_key="sample",
                          dataset="lab2_disease")
    rep = prop.report
    assert prop.bump == "minor" and prop.candidate.version == "1.1.0"
    assert rep["new_labels"]["cell_type_l2"] == [NEW] and rep["new_labels"]["cell_type_l1"] == []
    sub = rep["submission"]["levels"]["cell_type_l2"]
    assert NEW in sub["new_labels"] and sub["agreement"] > 0.9
    assert sub["new_labels"][NEW]["frac_in_novel_clusters"] > 0.8
    # a novel query cluster is dominated by the submitted new label
    assert any(max(c["submitted"], key=c["submitted"].get) == NEW
               for c in rep["submission"]["novel_clusters"])
    assert rep["cells_added_per_label"]["cell_type_l2"][NEW] == (q.obs["cell_type_l2"] == NEW).sum()
    stab = rep["old_cell_label_agreement"]["cell_type_l2"]
    assert stab["old_version"] > 0.95 and stab["new_version"] > 0.95
    assert set(rep["self_mapping_accuracy"]) == set(KEYS) and rep["passed"]
    cand = prop.candidate
    assert cand.n_cells == ref.n_cells + q.n_obs and np.array_equal(cand.genes, ref.genes)
    assert cand.provenance["parent_version"] == "1.0.0"
    assert "lab2_disease" in cand.provenance["changelog"][-1]
    assert "query_lab" in cand.provenance["datasets"]
    assert (cand.obs["added_in"] == "1.1.0").sum() == q.n_obs
    json.loads(prop.to_json())
    path = accept_update(reg, prop)
    assert reg.versions("pbmc") == ["1.0.0", "1.1.0"] and reg.latest("pbmc") == "1.1.0"
    assert os.path.exists(os.path.join(path, "review.json"))
    assert reg.entry("pbmc")["parent_version"] == "1.0.0"
    assert reg.load("pbmc").content_hash() == cand.content_hash()

    # re-annotation only -> patch
    rl = ref.obs[["cell_type_l2"]].iloc[:5].copy()
    rl["cell_type_l2"] = "CD8 T"
    rl = rl[ref.obs["cell_type_l2"].iloc[:5] != "CD8 T"]
    patch = propose_update(ref, relabel=rl)
    assert patch.bump == "patch" and patch.candidate.version == "1.0.1"
    assert patch.report["relabelled_cells"]["cell_type_l2"] == len(rl)


def test_transformation_change_is_major(base, data):
    sub, ref = base
    _, q = data
    prop = propose_update(ref, q, batch_key="batch", ref_adata=sub, n_pcs=15)
    assert prop.bump == "major" and prop.candidate.version == "2.0.0"
    assert prop.candidate.n_dims == 15 and prop.report["transform_changed"]


def test_harmonize_detects_split(data):
    ref_ad, _ = data
    a = ref_ad[ref_ad.obs["batch"].isin(["ref_batch0", "ref_batch1"])].copy()
    b = ref_ad[ref_ad.obs["batch"].isin(["ref_batch2", "ref_batch3"])].copy()
    # atlas A annotates T cells coarsely at its finest level
    l2 = a.obs["cell_type_l2"].astype(str)
    a.obs["cell_type_l2"] = l2.where(~l2.isin(["CD4 T", "CD8 T"]), "T cell")
    refA = build_reference(a, KEYS, batch_key="batch", n_hvg=200, n_pcs=20,
                           calibrate=False, name="A")
    refB = build_reference(b, KEYS, batch_key="batch", n_hvg=200, n_pcs=20,
                           calibrate=False, name="B")
    h = harmonize_references(refA, refB, a, b, "cell_type_l2")
    t = h.table.set_index("harmonized_label")
    assert t.loc["T cell", "relation"] == "split"
    assert t.loc["T cell", "labels_B"] == ["CD4 T", "CD8 T"]
    others = t.drop(index="T cell")
    assert (others["relation"] == "equivalent").all() and len(others) == 6
    assert h.tree["root"]["T cell"]["T cell"] == {"CD4 T": {}, "CD8 T": {}}
    assert h.tree["root"]["Myeloid"]["DC"] == {}
    assert h.contingency_ab.loc["T cell", ["CD4 T", "CD8 T"]].sum() > \
        0.9 * h.contingency_ab.loc["T cell"].sum()
