"""Count normalisation, feature selection and scaling.

The reference stores every parameter learned here (gene list, per-gene mean and
standard deviation) so that the *same* data transformation can later be applied to
a query dataset -- the first step of the common reference-mapping strategy described
in the perspective ("the same data transformation that is learned when assembling the
reference dataset is applied to the query").
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

SCALE_CLIP = 10.0  # Seurat ScaleData default


def to_csr(X) -> sp.csr_matrix:
    if sp.issparse(X):
        return X.tocsr().astype(np.float64)
    return sp.csr_matrix(np.asarray(X, dtype=np.float64))


def to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def normalize_total_log1p(X, target_sum: float = 1e4) -> sp.csr_matrix:
    """Library-size normalisation to ``target_sum`` counts followed by log1p.

    Library sizes are computed on *all* genes of the dataset, before any subsetting,
    so a query normalised on its own measured genes is on the reference's scale.
    """
    X = to_csr(X)
    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    X = sp.diags(target_sum / lib) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def _mean_var(X: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(X.mean(axis=0)).ravel()
    sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    n = X.shape[0]
    var = (sq - mean**2) * n / max(n - 1, 1)
    return mean, np.clip(var, 0, None)


def _binned_dispersion_rank(X: sp.csr_matrix, n_bins: int = 20) -> np.ndarray:
    """Seurat ``mean.var.plot`` style normalised dispersion (z-score within mean bins)."""
    mean, var = _mean_var(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.log(np.where(mean > 0, var / mean, np.nan))
    bins = pd.cut(mean, bins=n_bins)
    df = pd.DataFrame({"disp": disp, "bin": bins})
    grp = df.groupby("bin", observed=False)["disp"]
    mu = grp.transform("mean")
    sd = grp.transform("std").fillna(1.0).replace(0, 1.0)
    z = ((df["disp"] - mu) / sd).to_numpy()
    z[~np.isfinite(z)] = -np.inf
    z[mean <= 0.0125] = -np.inf  # drop near-silent genes
    return z


def select_hvgs(X_log, genes, n_top: int = 2000, batch=None) -> np.ndarray:
    """Batch-aware highly variable gene selection.

    Without batches, the top ``n_top`` genes by normalised dispersion are returned.
    With batches, dispersion is ranked within each batch and genes are ordered by the
    number of batches in which they are highly variable, then by median rank (the
    strategy used by Seurat ``SelectIntegrationFeatures`` and scanpy ``batch_key``),
    which keeps batch-specific technical genes out of the reference transformation.
    """
    genes = np.asarray(genes)
    X_log = to_csr(X_log)
    n_top = min(n_top, len(genes))
    if batch is None or len(pd.unique(np.asarray(batch))) < 2:
        z = _binned_dispersion_rank(X_log)
        order = np.argsort(-z, kind="stable")
        return genes[np.sort(order[:n_top])]
    batch = np.asarray(batch)
    ranks, n_hv = [], np.zeros(len(genes))
    for b in pd.unique(batch):
        m = batch == b
        if m.sum() < 20:
            continue
        z = _binned_dispersion_rank(X_log[m])
        r = np.empty(len(genes))
        r[np.argsort(-z, kind="stable")] = np.arange(len(genes))
        ranks.append(r)
        n_hv += r < n_top
    if not ranks:
        return select_hvgs(X_log, genes, n_top)
    med = np.median(np.vstack(ranks), axis=0)
    order = np.lexsort((med, -n_hv))
    return genes[np.sort(order[:n_top])]


def subset_genes(X, genes_have, genes_want) -> tuple[sp.csr_matrix, float]:
    """Reorder ``X`` columns to ``genes_want``; missing genes are filled with zeros.

    Returns the matrix and the fraction of wanted genes that were missing. Zero-filling
    missing features is what Symphony and Seurat do when a query lacks reference genes.
    """
    X = to_csr(X)
    idx = pd.Index(np.asarray(genes_have))
    if not idx.is_unique:
        raise ValueError("query gene names are not unique")
    pos = idx.get_indexer(np.asarray(genes_want))
    present = pos >= 0
    frac_missing = 1.0 - present.mean()
    out = sp.csr_matrix((X.shape[0], len(genes_want)))
    if present.any():
        sub = X[:, pos[present]]
        cols = np.flatnonzero(present)
        out = sp.csr_matrix((sub.data, cols[sub.indices], sub.indptr), shape=out.shape)
    if frac_missing > 0.2:
        warnings.warn(
            f"{frac_missing:.0%} of reference features are absent from the query; "
            "mapping quality will degrade", stacklevel=2)
    return out, float(frac_missing)


def scale_fit(X_log) -> tuple[np.ndarray, np.ndarray]:
    mean, var = _mean_var(to_csr(X_log))
    sd = np.sqrt(var)
    sd[sd == 0] = 1.0
    return mean, sd


def scale_apply(X_log, mean: np.ndarray, sd: np.ndarray, clip: float = SCALE_CLIP) -> np.ndarray:
    Z = (to_dense(X_log) - mean) / sd
    return np.clip(Z, -clip, clip)
