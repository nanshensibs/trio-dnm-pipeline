"""Cross-species mapping (perspective section "Cross-species mapping").

Mapping a dataset from one species onto a reference from another needs (i) a shared
feature space -- genes are converted through an ortholog table, keeping one-to-one
orthologs (optionally summing one-to-many paralogs) and reporting how much of the
data survives -- and (ii) an alignment that tolerates species-specific expression
divergence. The latter uses CCA (Butler et al. 2018, ref. 93), here in its Seurat v3
anchor form (Stuart et al. 2019, ref. 21; ``refmap.crossmodal.cca_anchor_transfer``):
canonical correlation finds the expression programmes *shared* by the two species,
so diverged genes contribute little.

Cell types that exist in only one species have no mutual-nearest-neighbour anchors
of their own; their anchor weights come from distant, inconsistent anchors, so they
surface as low prediction scores. Scores are summarised per query cluster
(clustered in the query's own space) to flag candidate species-specific populations,
in the spirit of the per-cluster mapping uncertainty of Figure 1C.

For comparison, ``project_across_species`` maps the ortholog-converted query through
an existing ``Reference`` exactly like a same-species query (``Reference.project`` ->
Symphony -> kNN transfer; ``refmap.mapping.map_query``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .crossmodal import AnchorTransferResult, _pca_loadings, cca_anchor_transfer
from .graph import cluster as louvain_cluster
from .preprocess import (normalize_total_log1p, scale_apply, scale_fit, select_hvgs,
                         subset_genes, to_csr)


@dataclass
class OrthologReport:
    mode: str
    n_genes_in: int                 # species-2 genes in the input
    n_genes_mapped: int             # species-2 genes that contributed to the output
    n_genes_out: int                # species-1 genes in the output
    frac_genes_mapped: float
    frac_counts_mapped: float       # fraction of total counts retained
    n_dropped_one2many: int         # species-2 genes dropped as non one-to-one
    n_unmatched: int                # species-2 genes without any ortholog

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def convert_orthologs(adata, ortholog_table: pd.DataFrame, from_col: str, to_col: str,
                      mode: str = "one2one", layer: str | None = None
                      ) -> tuple[ad.AnnData, OrthologReport]:
    """Rename ``adata`` genes from the ``from_col`` to the ``to_col`` namespace.

    ``mode="one2one"`` keeps only pairs that are unique in both directions of the
    table (computed from the table itself, not from an annotation column).
    ``mode="sum"`` additionally collapses one-to-many orthologs by summing the counts of
    all ``from`` genes that map to the same ``to`` gene; ``from`` genes with several
    targets are dropped (their counts cannot be attributed). Works on raw counts.
    """
    if mode not in ("one2one", "sum"):
        raise ValueError("mode must be 'one2one' or 'sum'")
    X = to_csr(adata.layers[layer] if layer else adata.X)
    genes = pd.Index(np.asarray(adata.var_names).astype(str))
    tab = ortholog_table[[from_col, to_col]].dropna().astype(str).drop_duplicates()
    tab = tab[tab[from_col].isin(genes)]
    n_from = tab.groupby(from_col)[to_col].transform("nunique")
    n_to = tab.groupby(to_col)[from_col].transform("nunique")
    if mode == "one2one":
        keep = tab[(n_from == 1) & (n_to == 1)]
    else:
        keep = tab[n_from == 1]
    targets = np.asarray(pd.unique(keep[to_col]))
    src = genes.get_indexer(keep[from_col])
    dst = pd.Index(targets).get_indexer(keep[to_col])
    M = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(len(genes), len(targets)))
    Y = (X @ M).tocsr()
    tot = X.sum()
    has_orth = genes.isin(tab[from_col])
    rep = OrthologReport(mode=mode, n_genes_in=len(genes), n_genes_mapped=int(len(src)),
                         n_genes_out=int(len(targets)),
                         frac_genes_mapped=float(len(src) / max(len(genes), 1)),
                         frac_counts_mapped=float(Y.sum() / tot) if tot > 0 else 0.0,
                         n_dropped_one2many=int(has_orth.sum() - len(src)),
                         n_unmatched=int((~has_orth).sum()))
    src_names = keep.groupby(to_col, sort=False)[from_col].agg(",".join).reindex(targets)
    out = ad.AnnData(X=Y.astype(np.float32), obs=adata.obs.copy(),
                     var=pd.DataFrame({from_col: src_names.to_numpy()}, index=targets))
    out.uns["ortholog_conversion"] = rep.as_dict()
    return out, rep


@dataclass
class CrossSpeciesResult:
    labels: pd.DataFrame              # pred, score, uncertainty, query_cluster[, level preds]
    probs: pd.DataFrame
    cluster_summary: pd.DataFrame     # per query cluster: size, majority label, scores, flag
    transfer: AnchorTransferResult
    ortholog_report: OrthologReport
    features: np.ndarray
    params: dict = field(default_factory=dict)


def _query_clusters(Xq_log: sp.csr_matrix, genes, n_pcs: int, k: int, resolution: float,
                    seed: int) -> np.ndarray:
    hv = select_hvgs(Xq_log, genes, n_top=min(2000, len(genes)))
    Xs, _ = subset_genes(Xq_log, genes, hv)
    S = scale_apply(Xs, *scale_fit(Xs))
    Z = S @ _pca_loadings(S, n_pcs, seed)
    return louvain_cluster(Z, k=k, resolution=resolution, seed=seed)


def summarize_score_clusters(clusters, pred, score, low_score: float = 0.5,
                             truth=None, z_threshold: float = -3.0) -> pd.DataFrame:
    """Per query cluster: size, majority predicted label, mean/median prediction score,
    fraction of low-score cells. A cluster is flagged ``low_confidence`` (candidate
    species-specific population) when its median score is below ``low_score`` or is a
    robust outlier among clusters (``score_z`` = (median - median of cluster medians) /
    (1.4826 MAD) below ``z_threshold``)."""
    df = pd.DataFrame({"cluster": clusters, "pred": pred, "score": score})
    if truth is not None:
        df["truth"] = np.asarray(truth).astype(str)
    g = df.groupby("cluster")
    out = pd.DataFrame({
        "n_cells": g.size(),
        "majority_pred": g["pred"].agg(lambda s: s.value_counts().index[0]),
        "frac_majority": g["pred"].agg(lambda s: s.value_counts(normalize=True).iloc[0]),
        "mean_score": g["score"].mean(),
        "median_score": g["score"].median(),
        "frac_low_score": g["score"].agg(lambda s: float((s < low_score).mean())),
    })
    if truth is not None:
        out["majority_truth"] = g["truth"].agg(lambda s: s.value_counts().index[0])
    med = out["median_score"].to_numpy()
    mad = 1.4826 * np.median(np.abs(med - np.median(med)))
    out["score_z"] = (med - np.median(med)) / max(mad, 0.02)
    out["low_confidence"] = (out["median_score"] < low_score) | (out["score_z"] < z_threshold)
    return out.reset_index().sort_values("median_score").reset_index(drop=True)


def map_across_species(ref_adata, ref_labels_key, query_adata, ortholog_table: pd.DataFrame,
                       from_col: str, to_col: str, *, mode: str = "one2one",
                       ref_batch_key: str | None = None, n_features: int = 2000,
                       query_cluster_key: str | None = None, cluster_resolution: float = 1.0,
                       cluster_k: int = 15, low_score: float = 0.5, n_pcs: int = 30,
                       seed: int = 0, **cca_kwargs) -> CrossSpeciesResult:
    """Map a species-2 query onto a labelled species-1 reference.

    1. Convert query genes to species-1 orthologs (:func:`convert_orthologs`).
    2. Restrict both datasets to the shared orthologs and log-normalise them on that
       common gene set (library sizes are then comparable across species).
    3. Features: reference HVGs (batch-aware with ``ref_batch_key``) among shared genes.
    4. Seurat v3 CCA anchors + TransferData (``cca_anchor_transfer``; extra keyword
       arguments are passed through).
    5. Per-query-cluster uncertainty: clusters from ``query_cluster_key`` or Louvain on
       the query's own PCA; clusters with median prediction score < ``low_score`` are
       flagged as candidate species-specific types.

    ``ref_labels_key`` may be a list of obs keys (coarse -> fine); the last one drives
    anchor transfer and every level is transferred with the same anchor weights.
    """
    keys = [ref_labels_key] if isinstance(ref_labels_key, str) else list(ref_labels_key)
    conv, rep = convert_orthologs(query_adata, ortholog_table, from_col, to_col, mode=mode)
    ref_genes = np.asarray(ref_adata.var_names).astype(str)
    shared = np.intersect1d(ref_genes, np.asarray(conv.var_names).astype(str))
    if len(shared) < 50:
        raise ValueError(f"only {len(shared)} shared orthologs between reference and query")
    Xr, _ = subset_genes(ref_adata.X, ref_genes, shared)
    Xq, _ = subset_genes(conv.X, conv.var_names, shared)
    Xr, Xq = normalize_total_log1p(Xr), normalize_total_log1p(Xq)
    batch = ref_adata.obs[ref_batch_key].astype(str).to_numpy() if ref_batch_key else None
    feats = select_hvgs(Xr, shared, n_top=n_features, batch=batch)
    labels_fine = ref_adata.obs[keys[-1]].astype(str).to_numpy()
    res = cca_anchor_transfer(Xr, labels_fine, Xq, feats, ref_genes=shared, query_genes=shared,
                              n_pcs=n_pcs, seed=seed, **cca_kwargs)
    tab = pd.DataFrame({"pred": res.pred, "score": res.score, "uncertainty": res.uncertainty},
                       index=np.asarray(query_adata.obs_names))
    for key in keys[:-1]:
        p, s, _ = res.transfer_labels(ref_adata.obs[key].astype(str).to_numpy())
        tab[f"{key}_pred"], tab[f"{key}_score"] = p, s
    if query_cluster_key:
        clusters = query_adata.obs[query_cluster_key].astype(str).to_numpy()
    else:
        clusters = _query_clusters(Xq, shared, min(n_pcs, 20), cluster_k, cluster_resolution,
                                   seed)
    tab["query_cluster"] = clusters
    summary = summarize_score_clusters(clusters, res.pred, res.score, low_score)
    tab["cluster_low_confidence"] = tab["query_cluster"].map(
        summary.set_index("cluster")["low_confidence"]).to_numpy()
    probs = res.probs.copy()
    probs.index = tab.index
    return CrossSpeciesResult(labels=tab, probs=probs, cluster_summary=summary, transfer=res,
                              ortholog_report=rep, features=feats,
                              params=dict(mode=mode, n_shared=int(len(shared)),
                                          n_features=int(len(feats)), low_score=low_score,
                                          **res.params))


def project_across_species(reference, query_adata, ortholog_table: pd.DataFrame,
                           from_col: str, to_col: str, *, mode: str = "one2one",
                           **map_kwargs):
    """Projection route: ortholog conversion, then the standard query mapping through a
    pre-built ``refmap.reference.Reference`` (``Reference.project`` + Symphony + kNN
    transfer + novelty scoring, ``refmap.mapping.map_query``). Reference genes without
    an ortholog are zero-filled by ``Reference.project``. Returns
    ``(MappingResult, OrthologReport)``."""
    from .mapping import map_query
    conv, rep = convert_orthologs(query_adata, ortholog_table, from_col, to_col, mode=mode)
    return map_query(reference, conv, **map_kwargs), rep


__all__ = ["convert_orthologs", "OrthologReport", "map_across_species", "CrossSpeciesResult",
           "project_across_species", "summarize_score_clusters"]
