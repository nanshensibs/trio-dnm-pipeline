"""Perturbation atlases: scGen-style held-out prediction and signature matching."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from refmap.perturbation import (PerturbationModel, evaluate_holdout, latent_to_expression,
                                 match_signatures, query_response)
from refmap.reference import build_reference
from refmap.simulate import draw_dataset, simulate_perturbation_atlas

KEYS = ["cell_type_l1", "cell_type_l2"]


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    """Tiny matrices: BLAS/OpenMP threading only adds overhead (and oversubscribes
    shared CI machines)."""
    with threadpool_limits(1):
        yield


@pytest.fixture(scope="module")
def atlas():
    return simulate_perturbation_atlas(n_genes=600, cells_per_block=100, seed=0)


@pytest.fixture(scope="module")
def atlas_ref(atlas):
    a, _ = atlas
    return build_reference(a, KEYS, n_hvg=600, n_pcs=20, calibrate=False, name="screen")


def _query(world, drug, seed=99):
    design = [dict(sample=f"q_{ct}_{c}", batch="qlab", batch_sd=0.5, cell_type=ct, n=120,
                   condition=c, extra={f"pert:{drug}": 1.5} if c == "treated" else {})
              for ct in ("CD4 T", "CD14 Mono", "NK") for c in ("control", "treated")]
    return draw_dataset(world, design, seed=seed)


def test_model_arithmetic(atlas_ref):
    obs = atlas_ref.obs
    m = PerturbationModel.fit(atlas_ref.Z, obs["cell_type_l2"], obs["perturbation"])
    assert m.perturbations == ["KO1", "drugA", "drugB", "drugC"]
    others = [m.type_delta("drugB", c) for c in m.cell_types if c != "NK"]
    assert np.allclose(m.delta("drugB", exclude_cell_type="NK"), np.mean(others, axis=0))
    Zc = atlas_ref.Z[(obs["cell_type_l2"] == "NK").to_numpy()
                     & (obs["perturbation"] == "control").to_numpy()]
    Zp = m.predict(Zc, "drugB", cell_type="NK")
    assert Zp.shape == Zc.shape
    assert m.predict_expression(atlas_ref, Zc, "drugB", "NK").shape == (len(Zc), 600)
    assert m.signatures().shape == (4, atlas_ref.n_dims)
    assert set(m.signatures(by_cell_type=True)["drugA"].index) == set(m.cell_types)


@pytest.mark.parametrize("pert,ct", [("drugB", "CD14 Mono"), ("drugA", "NK")])
def test_heldout_celltype_prediction_beats_no_change(atlas, pert, ct):
    a, world = atlas
    held = ((a.obs["perturbation"] == pert) & (a.obs["cell_type_l2"] == ct)).to_numpy()
    # strict OOD: the held-out block is absent from the reference transformation too
    ref = build_reference(a[~held].copy(), KEYS, n_hvg=600, n_pcs=20, calibrate=False)
    genes = world.genes[world.programs[f"pert:{pert}"]]
    r = evaluate_holdout(ref, a, pert, ct, response_genes=genes)
    assert ct not in r["trained_cell_types"] and len(r["trained_cell_types"]) == 3
    resp, allg = r["response"], r["all"]
    # responses have cell-type-specific gains, so the balanced delta is not exact
    assert resp["mse_pred"] < 0.5 * resp["mse_baseline"]
    assert resp["r2_pred"] > resp["r2_baseline"]
    assert allg["r2_pred"] > allg["r2_baseline"]
    assert r["delta_cosine"] > 0.9
    # data-driven top-DEG evaluation (scGen) agrees
    r2 = evaluate_holdout(ref, a, pert, ct, n_top_de=50, decode="residual")
    assert r2["response"]["r2_pred"] > r2["response"]["r2_baseline"] + 0.3


@pytest.mark.parametrize("drug", ["drugB", "drugA"])
def test_signature_matching_ranks_true_perturbation_first(atlas, atlas_ref, drug):
    _, world = atlas
    q = _query(world, drug)
    obs = atlas_ref.obs
    model = PerturbationModel.fit(atlas_ref.Z, obs["cell_type_l2"], obs["perturbation"])
    qr = query_response(atlas_ref, q, condition_key="condition", treated="treated",
                        control="control", batch_key="batch")
    assert set(qr.per_type.index) == {"CD4 T", "CD14 Mono", "NK"}   # transferred labels
    per_type = match_signatures(qr.per_type, model.signatures(by_cell_type=True))
    assert per_type.iloc[0]["perturbation"] == drug
    assert per_type.iloc[0]["n_shared_cell_types"] == 3
    assert per_type.iloc[0]["cosine"] > 0.8 > per_type.iloc[1]["cosine"]
    glob = match_signatures(qr.global_delta, model.signatures())
    assert glob.iloc[0]["perturbation"] == drug
    # expression-space signatures give the same answer
    sig = model.signatures()
    expr = {p: latent_to_expression(atlas_ref, sig.loc[p].to_numpy()) for p in sig.index}
    top = match_signatures(latent_to_expression(atlas_ref, qr.global_delta), expr, top=1)
    assert top.iloc[0]["perturbation"] == drug
    # corrected (Symphony) space also works, reusing the mapping
    qc = query_response(atlas_ref, q, condition_key="condition", treated="treated",
                        control="control", space="corrected", mapping=qr.mapping)
    assert match_signatures(qc.per_type, model.signatures(True)).iloc[0]["perturbation"] == drug
