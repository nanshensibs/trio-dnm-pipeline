"""Cross-species mapping: ortholog conversion + CCA anchors, species-specific types."""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from refmap.crossspecies import (convert_orthologs, map_across_species,
                                 project_across_species)
from refmap.reference import build_reference
from refmap.simulate_modalities import simulate_species_pair


@pytest.fixture(scope="module", autouse=True)
def _single_thread_blas():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:          # pragma: no cover
        yield
        return
    with threadpool_limits(1):
        yield


@pytest.fixture(scope="module")
def pair():
    return simulate_species_pair(n_genes=600, cells_per_ref_batch=400, n_query=600, seed=0)


@pytest.fixture(scope="module")
def result(pair):
    return map_across_species(pair.ref, ["cell_type_l1", "cell_type_l2"], pair.query,
                              pair.orthologs, "mouse", "human", ref_batch_key="batch")


def test_convert_orthologs_modes():
    orth = pd.DataFrame({"human": ["A", "B", "B", "C", "D"],
                         "mouse": ["a", "b1", "b2", "c", "c"]})
    X = sp.csr_matrix(np.array([[1, 2, 3, 4, 5]], float))
    q = ad.AnnData(X=X, var=pd.DataFrame(index=["a", "b1", "b2", "c", "x"]))
    one, rep = convert_orthologs(q, orth, "mouse", "human")
    assert list(one.var_names) == ["A"]              # c maps to two human genes: dropped
    assert rep.n_unmatched == 1 and rep.n_genes_mapped == 1
    assert np.isclose(rep.frac_counts_mapped, 1 / 15)
    summed, rep2 = convert_orthologs(q, orth, "mouse", "human", mode="sum")
    assert dict(zip(summed.var_names, summed.X.toarray().ravel())) == {"A": 1, "B": 5}
    assert rep2.frac_counts_mapped > rep.frac_counts_mapped


def test_conversion_coverage(pair):
    one, rep = convert_orthologs(pair.query, pair.orthologs, "mouse", "human")
    summed, rep2 = convert_orthologs(pair.query, pair.orthologs, "mouse", "human", mode="sum")
    assert set(one.var_names) <= set(pair.ref.var_names)
    assert rep.n_genes_out == pair.info["n_one2one"]
    assert rep2.n_genes_out == pair.info["n_one2one"] + pair.info["n_one2many"]
    assert 0.8 < rep.frac_counts_mapped < rep2.frac_counts_mapped <= 1.0


def test_shared_types_recovered(pair, result):
    lab = result.labels
    shared = ~pair.query.obs["species_specific"].to_numpy()
    acc2 = (lab["pred"].to_numpy()[shared]
            == pair.query.obs["cell_type_l2"].to_numpy()[shared]).mean()
    acc1 = (lab["cell_type_l1_pred"].to_numpy()[shared]
            == pair.query.obs["cell_type_l1"].to_numpy()[shared]).mean()
    assert acc2 >= 0.9 and acc1 >= 0.95              # measured 1.0 / 1.0
    assert result.ortholog_report.mode == "one2one"
    assert result.params["n_shared"] == pair.info["n_one2one"]


def test_species_specific_type_has_low_scores(pair, result):
    lab = result.labels
    spec = pair.query.obs["species_specific"].to_numpy()
    s = lab["score"].to_numpy()
    assert np.median(s[spec]) < np.median(s[~spec]) - 0.3     # measured 0.32 vs 0.99
    assert s[spec].mean() < 0.6
    # the least confident query cluster is the species-specific population
    cs = result.cluster_summary
    worst = cs.iloc[0]["cluster"]
    in_worst = lab["query_cluster"].to_numpy() == worst
    assert spec[in_worst].mean() > 0.8
    assert bool(cs.iloc[0]["low_confidence"])
    flagged = lab["cluster_low_confidence"].to_numpy()
    assert flagged[spec].mean() > 0.8 and flagged[~spec].mean() < 0.1


def test_projection_route(pair):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = build_reference(pair.ref, ["cell_type_l1", "cell_type_l2"], batch_key="batch",
                              n_hvg=600, calibrate=False)
        res, rep = project_across_species(ref, pair.query, pair.orthologs, "mouse", "human")
    shared = ~pair.query.obs["species_specific"].to_numpy()
    pred = res.labels["cell_type_l1_pred"].to_numpy()
    assert (pred[shared] == pair.query.obs["cell_type_l1"].to_numpy()[shared]).mean() >= 0.9
    assert 0 < res.frac_missing_genes < 0.3          # genes without one-to-one ortholog
    assert rep.n_genes_out < pair.query.n_vars
