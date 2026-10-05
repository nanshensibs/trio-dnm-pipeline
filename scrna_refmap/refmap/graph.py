"""k-nearest-neighbour graphs and community detection."""
from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def knn(Z_index: np.ndarray, Z_query: np.ndarray | None = None, k: int = 30,
        exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean kNN. Returns ``(dist, idx)`` of shape (n_query, k)."""
    k_eff = min(k + int(exclude_self), Z_index.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff).fit(Z_index)
    dist, idx = nn.kneighbors(Z_index if Z_query is None else Z_query)
    if exclude_self:
        dist, idx = dist[:, 1:], idx[:, 1:]
    return dist, idx


def knn_graph(Z: np.ndarray, k: int = 15) -> sp.csr_matrix:
    """Symmetrised binary kNN adjacency (no self loops)."""
    _, idx = knn(Z, k=k, exclude_self=True)
    n = Z.shape[0]
    rows = np.repeat(np.arange(n), idx.shape[1])
    A = sp.csr_matrix((np.ones(rows.size), (rows, idx.ravel())), shape=(n, n))
    return ((A + A.T) > 0).astype(float).tocsr()


def louvain(A: sp.csr_matrix, resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    """Louvain communities on an adjacency matrix; labels ordered by size."""
    G = nx.from_scipy_sparse_array(A)
    comms = nx.community.louvain_communities(G, resolution=resolution, seed=seed)
    comms = sorted(comms, key=len, reverse=True)
    lab = np.empty(A.shape[0], dtype=int)
    for i, c in enumerate(comms):
        lab[list(c)] = i
    return lab


def cluster(Z: np.ndarray, k: int = 15, resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    return louvain(knn_graph(Z, k), resolution, seed)
