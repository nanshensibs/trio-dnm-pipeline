"""Transfer of discrete labels and continuous values from reference to query.

Uses the weighted k-nearest-neighbour scheme of scArches/HLCA (Lotfollahi et al. 2022,
ref. 3; Sikkema et al. 2023, ref. 7): neighbour distances are turned into Gaussian
weights scaled by the spread of each cell's neighbour distances; the probability of a
label is the summed weight of neighbours carrying it, and the *uncertainty* is one
minus the winning probability. Cells whose uncertainty exceeds a threshold are
labelled ``Unknown`` -- the "uncertainty of transfer labels" the perspective proposes
for discriminating new cell states from existing references.

Hierarchical annotations (coarse -> fine) are transferred level by level and kept
consistent with the hierarchy; when a fine label cannot be resolved confidently, the
cell keeps its parent label (progressive/hierarchical classification with rejection,
Michielsen et al., refs. 30-31).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNKNOWN = "Unknown"


def knn_weights(dist: np.ndarray) -> np.ndarray:
    """scArches ``weighted_knn_transfer`` kernel."""
    std = np.std(dist, axis=1, keepdims=True)
    std[std == 0] = 1e-6
    scale = (2.0 / std) ** 2
    w = np.exp(-dist / scale)
    w /= w.sum(axis=1, keepdims=True)
    return w


def label_probabilities(labels: np.ndarray, idx: np.ndarray, w: np.ndarray,
                        classes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(str)
    classes = np.unique(labels) if classes is None else np.asarray(classes)
    code = pd.Index(classes).get_indexer(labels)
    P = np.zeros((idx.shape[0], len(classes)))
    nb = code[idx]
    for j in range(idx.shape[1]):
        np.add.at(P, (np.arange(idx.shape[0]), nb[:, j]), w[:, j])
    return P, classes


def transfer_labels(labels, idx, w, unknown_threshold: float = 0.5) -> pd.DataFrame:
    P, classes = label_probabilities(labels, idx, w)
    best = P.argmax(axis=1)
    prob = P[np.arange(len(best)), best]
    pred = classes[best].astype(object)
    unc = 1.0 - prob
    pred[unc > unknown_threshold] = UNKNOWN
    return pd.DataFrame({"pred": pred, "uncertainty": unc})


def transfer_hierarchical(ref_obs: pd.DataFrame, label_keys: list[str], idx, w,
                          unknown_threshold: float = 0.5) -> pd.DataFrame:
    """Level-wise transfer with hierarchy consistency and rejection to the parent."""
    out = {}
    parent_pred = None
    for li, key in enumerate(label_keys):
        labels = ref_obs[key].astype(str).to_numpy()
        P, classes = label_probabilities(labels, idx, w)
        if parent_pred is not None:
            parent_of = (ref_obs[[label_keys[li - 1], key]].astype(str)
                         .drop_duplicates().set_index(key)[label_keys[li - 1]])
            cls_parent = parent_of.reindex(classes).to_numpy()
            allowed = cls_parent[None, :] == parent_pred[:, None]
            # P(child | parent) = P(child) restricted to the predicted parent
            Pc = np.where(allowed, P, 0.0)
            mass = Pc.sum(axis=1, keepdims=True)
            cond = np.divide(Pc, mass, out=np.zeros_like(Pc), where=mass > 0)
            best = cond.argmax(axis=1)
            unc = 1.0 - P[np.arange(len(best)), best]   # unconditional uncertainty
            pred = classes[best].astype(object)
            resolved = (unc <= unknown_threshold) & (mass.ravel() > 0)
            pred[~resolved] = parent_pred[~resolved]
            pred[parent_pred == UNKNOWN] = UNKNOWN
            resolved &= parent_pred != UNKNOWN
        else:
            best = P.argmax(axis=1)
            unc = 1.0 - P[np.arange(len(best)), best]
            pred = classes[best].astype(object)
            resolved = unc <= unknown_threshold
            pred[~resolved] = UNKNOWN
        out[f"{key}_pred"] = pred
        out[f"{key}_uncertainty"] = unc
        out[f"{key}_resolved"] = resolved
        parent_pred = pred
    return pd.DataFrame(out)


def transfer_continuous(values: np.ndarray, idx, w) -> tuple[np.ndarray, np.ndarray]:
    """Weighted-kNN imputation of continuous reference values (e.g. spatial
    coordinates, protein abundance, chromatin accessibility; Figure 1C third row).
    Returns the weighted mean and weighted standard deviation (an uncertainty)."""
    V = np.asarray(values, float)
    nb = V[idx]                                      # (n, k, m)
    mean = np.einsum("nk,nkm->nm", w, nb)
    var = np.einsum("nk,nkm->nm", w, (nb - mean[:, None, :]) ** 2)
    return mean, np.sqrt(var)
