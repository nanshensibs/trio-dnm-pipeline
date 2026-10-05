"""Evaluation of a mapping against the perspective's three success criteria.

"Successful mapping of disease queries should meet the following criteria:
(1) conservation of the heterogeneity of healthy cell states in the reference,
(2) integration of identical cell types in reference and query, and
(3) preservation of previously uncharacterized cell types and states emerging in
disease datasets that are not present in the reference."

Each criterion gets quantitative metrics (computed when ground truth is available, e.g.
on simulated data or on a query with author annotations):

1. label-transfer accuracy / macro-F1 per hierarchy level, and label silhouette of the
   query in the joint space;
2. reference/query mixing among shared cell types (kNN-based, normalised so 1 = as
   mixed as expected from the dataset sizes), centroid offset of shared types
   relative to the reference's within-type spread, and batch-mixing entropy;
3. novel-state detection AUROC of each per-cell statistic, and recall/precision of the
   cluster-level novel flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, silhouette_score

from .graph import knn


def label_metrics(true, pred, ignore=("Unknown", "Novel")) -> dict:
    true, pred = np.asarray(true).astype(str), np.asarray(pred).astype(str)
    keep = ~np.isin(pred, ignore)
    out = dict(accuracy=float((true == pred).mean()),
               accuracy_confident=float((true[keep] == pred[keep]).mean()) if keep.any() else np.nan,
               fraction_rejected=float(1 - keep.mean()),
               macro_f1=float(f1_score(true, pred, average="macro", zero_division=0)))
    out["recall_per_label"] = (pd.DataFrame({"t": true, "ok": true == pred})
                               .groupby("t")["ok"].mean().round(4).to_dict())
    return out


def mixing_entropy(Z: np.ndarray, groups, k: int = 30, max_cells: int = 5000,
                   seed: int = 0) -> float:
    """Mean normalised Shannon entropy of group labels among each cell's kNN
    (1 = perfectly mixed relative to the global group frequencies' maximum)."""
    groups = np.asarray(groups).astype(str)
    lv = np.unique(groups)
    if len(lv) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(groups), min(len(groups), max_cells), replace=False)
    _, idx = knn(Z, Z[sel], k=k)
    code = pd.Index(lv).get_indexer(groups)
    ent = []
    for row in code[idx]:
        p = np.bincount(row, minlength=len(lv)) / len(row)
        p = p[p > 0]
        ent.append(-(p * np.log(p)).sum() / np.log(len(lv)))
    return float(np.mean(ent))


def ref_query_mixing(Z_ref, Z_query, labels_ref, labels_query, k: int = 30) -> dict:
    """Criterion 2: for each cell type shared by reference and query, the fraction of
    a query cell's neighbours (in the pooled joint space) that are reference cells,
    divided by the expected fraction under perfect mixing of that type."""
    labels_ref = np.asarray(labels_ref).astype(str)
    labels_query = np.asarray(labels_query).astype(str)
    out = {}
    for lab in np.intersect1d(labels_ref, labels_query):
        r, q = Z_ref[labels_ref == lab], Z_query[labels_query == lab]
        if len(r) < k or len(q) < 5:
            continue
        pooled = np.vstack([r, q])
        is_ref = np.r_[np.ones(len(r), bool), np.zeros(len(q), bool)]
        _, idx = knn(pooled, q, k=k)
        out[lab] = float(is_ref[idx].mean() / (len(r) / len(pooled)))
    return out


def ref_query_alignment(Z_ref, Z_query, labels_ref, labels_query) -> dict:
    """Criterion 2, location-based: distance between the reference and query centroids
    of each shared cell type, divided by the reference's mean within-type spread
    (0 = same centre). Complements :func:`ref_query_mixing`, which also penalises a
    query that is *narrower* than the reference -- the normal outcome of projecting
    cells that lack the reference's own batch-specific noise directions."""
    labels_ref = np.asarray(labels_ref).astype(str)
    labels_query = np.asarray(labels_query).astype(str)
    out = {}
    for lab in np.intersect1d(labels_ref, labels_query):
        r, q = Z_ref[labels_ref == lab], Z_query[labels_query == lab]
        if len(r) < 5 or len(q) < 5:
            continue
        spread = np.linalg.norm(r - r.mean(0), axis=1).mean()
        out[lab] = float(np.linalg.norm(r.mean(0) - q.mean(0)) / max(spread, 1e-12))
    return out


def novelty_metrics(scores: pd.DataFrame, is_novel, cluster_novel=None) -> dict:
    is_novel = np.asarray(is_novel, bool)
    out = {}
    if 0 < is_novel.sum() < len(is_novel):
        for c in ("knn_distance", "mahalanobis", "uncertainty"):
            if c in scores:
                out[f"auroc_{c}"] = float(roc_auc_score(is_novel, scores[c]))
        if cluster_novel is not None:
            cn = np.asarray(cluster_novel, bool)
            out["novel_recall"] = float(cn[is_novel].mean())
            out["novel_precision"] = float(is_novel[cn].mean()) if cn.any() else np.nan
    return out


def evaluate_mapping(ref, result, query_obs: pd.DataFrame, truth_keys: dict | None = None,
                     novel_key: str | None = None, batch_key: str | None = None) -> dict:
    """Score a MappingResult; ``truth_keys`` maps reference label levels to query obs
    columns holding ground truth (e.g. ``{"cell_type_l2": "author_label"}``)."""
    truth_keys = truth_keys or {k: k for k in ref.label_keys if k in query_obs}
    novel = query_obs[novel_key].to_numpy(bool) if novel_key else np.zeros(len(query_obs), bool)
    out = {"criterion1_heterogeneity": {}, "criterion2_integration": {},
           "criterion3_novel_states": {}}
    fine = ref.label_keys[-1]
    for level, col in truth_keys.items():
        out["criterion1_heterogeneity"][level] = label_metrics(
            query_obs[col].to_numpy()[~novel], result.labels[f"{level}_pred"].to_numpy()[~novel])
    if fine in truth_keys:
        t = query_obs[truth_keys[fine]].astype(str).to_numpy()
        if len(np.unique(t[~novel])) > 1:
            rng = np.random.default_rng(0)
            sel = rng.choice(np.flatnonzero(~novel), min(int((~novel).sum()), 4000), replace=False)
            out["criterion1_heterogeneity"]["query_label_silhouette"] = float(
                silhouette_score(result.Zq_corr[sel], t[sel]))
        out["criterion2_integration"]["ref_query_mixing_per_type"] = ref_query_mixing(
            ref.Z_corr, result.Zq_corr, ref.labels(fine), t)
        vals = list(out["criterion2_integration"]["ref_query_mixing_per_type"].values())
        out["criterion2_integration"]["ref_query_mixing_mean"] = float(np.mean(vals)) if vals else np.nan
        al = ref_query_alignment(ref.Z_corr, result.Zq_corr[~novel], ref.labels(fine), t[~novel])
        out["criterion2_integration"]["centroid_offset_per_type"] = al
        out["criterion2_integration"]["centroid_offset_mean"] = float(np.mean(list(al.values()))) if al else np.nan
    if batch_key and batch_key in query_obs:
        out["criterion2_integration"]["query_batch_mixing_entropy"] = mixing_entropy(
            result.Zq_corr, query_obs[batch_key].to_numpy())
    cell_cluster_novel = pd.Series(result.clusters).map(
        result.cluster_summary.set_index("cluster")["novel"]).to_numpy(bool)
    out["criterion3_novel_states"] = novelty_metrics(result.scores, novel, cell_cluster_novel)
    return out
