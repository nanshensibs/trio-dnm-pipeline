"""Harmony integration of the reference and Symphony mapping of query cells.

Implements the two statistical reference-mapping building blocks the perspective
describes for Symphony (Kang et al. 2021, ref. 5): a low-dimensional (PCA)
transformation in which cells are *softly assigned to clusters representing
different cell states*, with batch effects removed by Harmony's mixture-of-experts
(MoE) ridge regression (Korsunsky et al. 2019). The reference is compressed to the
cluster centroids ``Y``, per-cluster soft counts ``Nr`` and per-cluster sums
``C = R Z_corr`` -- the quantities Symphony needs to correct query batches *without*
the reference cells themselves.

Layout convention: embeddings are ``(n_cells, n_dims)``; soft assignments ``R`` are
``(K, n_cells)``; design matrices ``Phi`` are ``(n_levels, n_cells)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def l2_normalize(Z: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Z / n


def design_matrix(batch: pd.DataFrame | np.ndarray | None, n: int) -> tuple[np.ndarray, list]:
    """One-hot design ``Phi`` (levels x cells) for one or more batch covariates."""
    if batch is None:
        return np.ones((1, n)), ["all"]
    if not isinstance(batch, pd.DataFrame):
        batch = pd.DataFrame({"batch": np.asarray(batch)})
    blocks, levels = [], []
    for col in batch.columns:
        vals = batch[col].astype(str).to_numpy()
        lv = list(pd.unique(vals))
        blocks.append((vals[None, :] == np.asarray(lv)[:, None]).astype(float))
        levels += [f"{col}={v}" for v in lv]
    return np.vstack(blocks), levels


def soft_assign(Z_cos: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Soft cluster assignment ``R`` (K x n) from cosine distance to centroids."""
    dist = 2.0 * (1.0 - Y @ Z_cos.T)
    s = -dist / sigma
    s -= s.max(axis=0, keepdims=True)
    R = np.exp(s)
    return R / R.sum(axis=0, keepdims=True)


@dataclass
class HarmonyResult:
    Z_corr: np.ndarray       # corrected embedding (n, d)
    R: np.ndarray            # soft assignments (K, n)
    Y: np.ndarray            # L2-normalised centroids (K, d)
    objective: list
    converged: bool


def _moe_correct(Z: np.ndarray, R: np.ndarray, Phi: np.ndarray, lamb: float) -> np.ndarray:
    Phi_moe = np.vstack([np.ones((1, Z.shape[0])), Phi])
    lam = np.diag([0.0] + [lamb] * Phi.shape[0])
    Z_corr = Z.copy()
    for k in range(R.shape[0]):
        Phi_Rk = Phi_moe * R[k][None, :]
        W = np.linalg.solve(Phi_Rk @ Phi_moe.T + lam + 1e-8 * np.eye(len(lam)), Phi_Rk @ Z)
        W[0, :] = 0.0  # never remove the intercept (biology)
        Z_corr -= (W.T @ Phi_Rk).T
    return Z_corr


def run_harmony(Z: np.ndarray, batch, *, theta: float = 2.0, sigma: float = 0.1,
                lamb: float = 1.0, n_clusters: int | None = None, max_iter: int = 10,
                max_iter_cluster: int = 20, eps_cluster: float = 1e-5,
                eps_harmony: float = 1e-4, block_size: float = 0.05,
                seed: int = 0) -> HarmonyResult:
    """Harmony (Korsunsky et al. 2019): alternate diversity-penalised soft k-means
    and MoE linear correction until the objective converges.

    With a single batch, ``theta`` is irrelevant and no correction is applied; the
    soft clustering is still computed because Symphony needs the centroids.
    """
    rng = np.random.default_rng(seed)
    n, _ = Z.shape
    Phi, _ = design_matrix(batch, n)
    n_batches = Phi.shape[0]
    K = n_clusters or int(min(100, max(2, round(n / 30))))
    Pr_b = Phi.sum(axis=1) / n
    theta_v = np.full(n_batches, theta if n_batches > 1 else 0.0)

    Z_corr = Z.copy()
    Z_cos = l2_normalize(Z_corr)
    km = KMeans(n_clusters=K, n_init=4, random_state=seed).fit(Z_cos)
    Y = l2_normalize(km.cluster_centers_)
    R = soft_assign(Z_cos, Y, sigma)
    E = np.outer(R.sum(axis=1), Pr_b)
    O = R @ Phi.T

    def objective() -> float:
        dist = 2.0 * (1.0 - Y @ Z_cos.T)
        kmeans_err = np.sum(R * dist)
        entropy = sigma * np.sum(R * np.log(R + 1e-12))
        div = sigma * np.sum((theta_v[None, :] * np.log((O + 1) / (E + 1))) * (R @ Phi.T))
        return float(kmeans_err + entropy + div)

    objs, converged = [objective()], False
    n_iter = max_iter if n_batches > 1 else 1
    for _ in range(n_iter):
        # --- clustering -------------------------------------------------------
        cl_obj = [objs[-1]]
        for _ in range(max_iter_cluster):
            Y = l2_normalize(R @ Z_cos)
            dist = 2.0 * (1.0 - Y @ Z_cos.T)
            order = rng.permutation(n)
            nb = max(1, int(np.ceil(n * block_size)))
            for start in range(0, n, nb):
                b = order[start:start + nb]
                E -= np.outer(R[:, b].sum(axis=1), Pr_b)
                O -= R[:, b] @ Phi[:, b].T
                s = -dist[:, b] / sigma
                s -= s.max(axis=0, keepdims=True)
                Rb = np.exp(s)
                if n_batches > 1:
                    pen = ((E + 1) / (O + 1)) ** theta_v[None, :]
                    Rb = Rb * (pen @ Phi[:, b])
                Rb /= Rb.sum(axis=0, keepdims=True)
                R[:, b] = Rb
                E += np.outer(Rb.sum(axis=1), Pr_b)
                O += Rb @ Phi[:, b].T
            cl_obj.append(objective())
            if abs(cl_obj[-2] - cl_obj[-1]) < eps_cluster * abs(cl_obj[-2]):
                break
        objs.append(cl_obj[-1])
        if n_batches == 1:
            converged = True
            break
        # --- correction -------------------------------------------------------
        Z_corr = _moe_correct(Z, R, Phi, lamb)
        Z_cos = l2_normalize(Z_corr)
        if len(objs) > 2 and abs(objs[-2] - objs[-1]) < eps_harmony * abs(objs[-2]):
            converged = True
            break
    Y = l2_normalize(R @ Z_cos)
    return HarmonyResult(Z_corr=Z_corr, R=R, Y=Y, objective=objs, converged=converged)


@dataclass
class SymphonyCompression:
    """Everything Symphony needs from the reference to map queries."""
    Y: np.ndarray        # (K, d) normalised centroids
    Nr: np.ndarray       # (K,) soft cell counts per centroid
    C: np.ndarray        # (K, d) R @ Z_corr
    mu: np.ndarray       # (K, d) weighted centroid means in Z_corr space
    cov_inv: np.ndarray  # (K, d, d) inverse weighted covariances (Mahalanobis)
    sigma: float
    lamb: float


def compress_reference(Z_corr: np.ndarray, Y: np.ndarray, sigma: float, lamb: float,
                       shrinkage: float = 0.05) -> tuple[SymphonyCompression, np.ndarray]:
    Z_cos = l2_normalize(Z_corr)
    R = soft_assign(Z_cos, Y, sigma)
    Nr = R.sum(axis=1)
    C = R @ Z_corr
    mu = C / np.maximum(Nr, 1e-12)[:, None]
    d = Z_corr.shape[1]
    cov_inv = np.empty((len(Nr), d, d))
    glob = np.cov(Z_corr, rowvar=False)
    for k in range(len(Nr)):
        w = R[k] / max(R[k].sum(), 1e-12)
        D = Z_corr - mu[k]
        cov = (D * w[:, None]).T @ D
        # Ledoit-Wolf style shrinkage towards the global covariance for small clusters
        a = shrinkage + (1 - shrinkage) * min(1.0, d / max(Nr[k], 1.0))
        cov = (1 - a) * cov + a * glob
        cov_inv[k] = np.linalg.pinv(cov)
    return SymphonyCompression(Y=Y, Nr=Nr, C=C, mu=mu, cov_inv=cov_inv,
                               sigma=sigma, lamb=lamb), R


def symphony_map(Zq: np.ndarray, comp: SymphonyCompression, query_batch=None,
                 correct: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Symphony ``mapQuery``: soft-assign query cells to reference centroids and
    remove query batch effects with the reference-anchored MoE regression.

    The reference enters only through ``Nr`` (intercept denominator) and ``C``
    (intercept numerator), so the intercept is pinned to the reference and only the
    query-batch deviations are regressed out. When ``query_batch`` is None, the whole
    query is treated as one batch (removing the query-vs-reference dataset effect).
    """
    Zq_cos = l2_normalize(Zq)
    Rq = soft_assign(Zq_cos, comp.Y, comp.sigma)
    if not correct:
        return Zq.copy(), Rq
    Phi, _ = design_matrix(query_batch, Zq.shape[0])
    Phi_moe = np.vstack([np.ones((1, Zq.shape[0])), Phi])
    lam = np.diag([0.0] + [comp.lamb] * Phi.shape[0])
    Zc = Zq.copy()
    for k in range(Rq.shape[0]):
        Phi_Rk = Phi_moe * Rq[k][None, :]
        x = Phi_Rk @ Phi_moe.T + lam
        x[0, 0] += comp.Nr[k]
        comb = Phi_Rk @ Zq
        comb[0, :] += comp.C[k]
        W = np.linalg.solve(x + 1e-8 * np.eye(len(x)), comb)
        W[0, :] = 0.0
        Zc -= (W.T @ Phi_Rk).T
    return Zc, Rq


def mahalanobis_to_centroids(Z: np.ndarray, R: np.ndarray, comp: SymphonyCompression) -> np.ndarray:
    """Symphony per-cell mapping metric: Mahalanobis distance of each cell to the
    reference centroid it is most strongly assigned to."""
    k = R.argmax(axis=0)
    out = np.empty(Z.shape[0])
    for kk in np.unique(k):           # group by centroid: no (n, d, d) temporaries
        m = k == kk
        D = Z[m] - comp.mu[kk]
        out[m] = np.einsum("ni,ij,nj->n", D, comp.cov_inv[kk], D)
    return np.sqrt(np.maximum(out, 0.0))
