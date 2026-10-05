"""Detection of query cell states absent from the reference (out-of-distribution).

The perspective's section "Identification of disease states by contextualizing disease
within a healthy reference" asks for an *uncertainty metric to discriminate new cell
states from existing references* and lists three concrete statistics, all implemented
here and calibrated against the reference's own self-mapping null distribution:

1. kNN label-transfer uncertainty (scArches/HLCA, refs. 3, 29);
2. kNN distance to the reference (distance-based OOD, refs. 3, 32);
3. Symphony's Mahalanobis distance of each query cell -- and of each query cluster --
   to reference centroids (ref. 5).

Query cells are clustered in the joint embedding and each cluster is summarised by
its mapping uncertainty (Figure 1C, "Mapping uncertainty" panel), so clusters made of
unseen states (e.g. disease-specific populations) stand out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .graph import cluster as louvain_cluster
from .harmony import mahalanobis_to_centroids
from .reference import cluster_mahalanobis


def empirical_pvalue(null_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    """P(null >= x) with add-one smoothing; ``null_sorted`` must be ascending."""
    n = len(null_sorted)
    ge = n - np.searchsorted(null_sorted, x, side="left")
    return (ge + 1.0) / (n + 1.0)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def score_cells(ref, Zq_corr: np.ndarray, Rq: np.ndarray, knn_dist: np.ndarray,
                uncertainty: np.ndarray, alpha: float = 0.01,
                unknown_threshold: float = 0.5) -> pd.DataFrame:
    null = ref.calibration.get("null", {})
    kd = knn_dist.mean(axis=1)
    maha = mahalanobis_to_centroids(Zq_corr, Rq, ref.comp)
    df = pd.DataFrame({"knn_distance": kd, "mahalanobis": maha, "uncertainty": uncertainty})
    if null:
        df["p_knn"] = empirical_pvalue(null["knn_distance"], kd)
        df["p_mahalanobis"] = empirical_pvalue(null["mahalanobis"], maha)
        # a cell is out-of-distribution when it is both far from its reference
        # neighbours and outside the covariance envelope of its reference centroid
        df["ood"] = (df["p_knn"] < alpha) & (df["p_mahalanobis"] < alpha)
    else:
        df["ood"] = False
    df["ood"] |= df["uncertainty"] > unknown_threshold
    return df


def cluster_query(Zq_corr: np.ndarray, k: int = 15, resolution: float = 1.0,
                  seed: int = 0) -> np.ndarray:
    return louvain_cluster(Zq_corr, k=k, resolution=resolution, seed=seed)


def summarize_clusters(clusters: np.ndarray, cell_scores: pd.DataFrame, Zq_corr: np.ndarray,
                       ref, predicted: np.ndarray, condition: np.ndarray | None = None,
                       min_ood_fraction: float = 0.5,
                       cluster_margin: float = 1.0) -> pd.DataFrame:
    """Per-query-cluster mapping report (Figure 1C).

    ``cluster_mahalanobis`` is Symphony's per-cluster metric: the Mahalanobis distance
    from the query cluster's mean to the closest reference centroid (closest under the
    centroid's own covariance)."""
    rows = []
    for c in np.unique(clusters):
        m = clusters == c
        cm = cluster_mahalanobis(Zq_corr[m].mean(axis=0), ref.comp)
        top = pd.Series(predicted[m]).value_counts()
        row = dict(cluster=int(c), n_cells=int(m.sum()),
                   top_label=top.index[0], top_label_fraction=float(top.iloc[0] / m.sum()),
                   median_uncertainty=float(cell_scores["uncertainty"][m].median()),
                   median_knn_distance=float(cell_scores["knn_distance"][m].median()),
                   ood_fraction=float(cell_scores["ood"][m].mean()),
                   cluster_mahalanobis=cm)
        if condition is not None:
            for lv, n in pd.Series(condition[m]).value_counts().items():
                row[f"frac_{lv}"] = float(n / m.sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    null = ref.calibration.get("null", {}).get("cluster_mahalanobis")
    if null is not None and len(null):
        df["p_cluster"] = empirical_pvalue(null, df["cluster_mahalanobis"].to_numpy())
        # beyond every in-distribution held-out group, by a margin
        df["cluster_outlier"] = df["cluster_mahalanobis"] > cluster_margin * null.max()
    else:
        df["p_cluster"], df["cluster_outlier"] = np.nan, False
    df["novel"] = (df["ood_fraction"] >= min_ood_fraction) | df["cluster_outlier"]
    return df.sort_values("cluster").reset_index(drop=True)
