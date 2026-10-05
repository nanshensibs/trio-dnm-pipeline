"""Mapping across molecular modalities (perspective section "Single-cell data mapping
across molecular modalities", Figure 3A).

Two strategies are implemented, mirroring the two rows of Figure 3A:

* **Feature conversion** (Figure 3A, top). Chromatin accessibility is converted into
  RNA-like *gene activity* scores (Cicero/Signac style: the summed counts of peaks
  overlapping the gene body plus 2 kb upstream, strand aware) and the RNA reference is
  mapped onto them with the **Seurat v3 anchor** procedure (Stuart et al. 2019, ref. 21):
  diagonal CCA of the two scaled datasets, L2-normalised canonical vectors, mutual
  nearest neighbours as anchors, anchor filtering in the original feature space,
  shared-neighbour anchor scores and Gaussian-kernel anchor weights per query cell
  (``TransferData``). The same routine serves cross-species mapping
  (``refmap.crossspecies``). The perspective's caveat applies: this relies on
  RNA/chromatin correlation, which is imperfect.

* **Bridge integration** (Figure 3A, bottom; Seurat v5, Hao et al. 2024, ref. 71). A
  multiome *bridge* (paired RNA + ATAC) is used as a dictionary: every reference cell is
  expressed as a combination of bridge cells using RNA, every query cell as a
  combination of bridge cells using ATAC, each modality analysed in its own space
  (RNA: log-normalisation, HVGs, scaled PCA; ATAC: TF-IDF + truncated SVD = LSI). Both
  then live in a common bridge-cell space, reduced by Laplacian eigenmaps of the bridge
  graph, in which labels are transferred by weighted kNN with uncertainty
  (``refmap.transfer``). No feature conversion is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, svds
from sklearn.utils.extmath import randomized_svd

from .graph import knn
from .preprocess import (normalize_total_log1p, scale_apply, scale_fit, select_hvgs,
                         subset_genes, to_csr, to_dense)
from .transfer import knn_weights, transfer_hierarchical, transfer_labels

# ======================================================================== features


def gene_activity(atac_adata, gene_coords: pd.DataFrame, upstream: int = 2000,
                  downstream: int = 0, peak_cols=("chrom", "start", "end")) -> ad.AnnData:
    """Gene activity scores (Signac ``GeneActivity`` / Cicero style).

    For each gene, sums the counts of all peaks overlapping the gene body extended
    ``upstream`` bp 5' of the TSS (and ``downstream`` bp past the TES); strand aware.
    ``gene_coords`` is indexed by gene with columns ``chrom, start, end, strand``; peak
    coordinates come from ``atac_adata.var[peak_cols]`` or are parsed from names like
    ``chr1-1000-1500``. Returns an AnnData (cells x genes, raw-count-like) with
    ``var['n_peaks']`` and the peak-to-gene matrix in ``uns['peak_gene']`` (peaks x genes).
    """
    var = atac_adata.var
    if all(c in var for c in peak_cols):
        pchr = var[peak_cols[0]].astype(str).to_numpy()
        pst = var[peak_cols[1]].to_numpy(np.int64)
        pen = var[peak_cols[2]].to_numpy(np.int64)
    else:
        parts = pd.Series(np.asarray(atac_adata.var_names)).str.split(r"[-:_]", n=2, expand=True)
        pchr, pst, pen = parts[0].to_numpy(), parts[1].astype(np.int64).to_numpy(), \
            parts[2].astype(np.int64).to_numpy()
    gc = gene_coords
    plus = gc["strand"].astype(str).to_numpy() != "-"
    gs = gc["start"].to_numpy(np.int64)
    ge = gc["end"].to_numpy(np.int64)
    ws = np.where(plus, gs - upstream, gs - downstream)
    we = np.where(plus, ge + downstream, ge + upstream)
    gchr = gc["chrom"].astype(str).to_numpy()
    rows, cols = [], []
    max_len = int((pen - pst).max()) if len(pst) else 0
    for c in np.unique(gchr):
        pi = np.flatnonzero(pchr == c)
        if not len(pi):
            continue
        order = pi[np.argsort(pst[pi], kind="stable")]
        starts = pst[order]
        for gi in np.flatnonzero(gchr == c):
            lo = np.searchsorted(starts, ws[gi] - max_len, side="left")
            hi = np.searchsorted(starts, we[gi], side="right")
            cand = order[lo:hi]
            cand = cand[(pen[cand] >= ws[gi]) & (pst[cand] <= we[gi])]
            rows.append(cand)
            cols.append(np.full(len(cand), gi))
    rows = np.concatenate(rows) if rows else np.zeros(0, int)
    cols = np.concatenate(cols) if cols else np.zeros(0, int)
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(pst), len(gc)))
    A = to_csr(atac_adata.X) @ M
    out = ad.AnnData(X=sp.csr_matrix(A, dtype=np.float32), obs=atac_adata.obs.copy(),
                     var=pd.DataFrame({"n_peaks": np.asarray(M.sum(axis=0)).ravel().astype(int)},
                                      index=gc.index.astype(str)))
    out.uns["peak_gene"] = M
    return out


def select_transfer_features(ref_X_lognorm, ref_genes, query_genes, n_top: int = 2000,
                             batch=None, query_X=None) -> np.ndarray:
    """Reference HVGs (batch-aware) that are also measured in the query
    (``FindTransferAnchors(features = VariableFeatures(reference))``). If ``query_X`` is
    given, genes that are all-zero in the query (e.g. genes without peaks) are dropped."""
    ref_genes = np.asarray(ref_genes).astype(str)
    shared = np.intersect1d(ref_genes, np.asarray(query_genes).astype(str))
    if query_X is not None:
        qg = pd.Index(np.asarray(query_genes).astype(str))
        tot = np.asarray(to_csr(query_X).sum(axis=0)).ravel()
        shared = shared[tot[qg.get_indexer(shared)] > 0]
    Xs, _ = subset_genes(ref_X_lognorm, ref_genes, shared)
    return select_hvgs(Xs, shared, n_top=n_top, batch=batch)


# ============================================================ Seurat v3 anchors


@dataclass
class AnchorTransferResult:
    """Output of :func:`cca_anchor_transfer`.

    ``weights`` is the (n_query x n_anchors) TransferData weight matrix; any reference
    quantity (labels, expression, coordinates) is transferred as ``weights @ value``
    evaluated at each anchor's reference cell (``anchors.ref_idx``)."""
    pred: np.ndarray
    score: np.ndarray                 # prediction.score.max
    probs: pd.DataFrame               # prediction scores per class (rows sum to 1)
    anchors: pd.DataFrame             # ref_idx, query_idx, score_raw, score
    weights: sp.csr_matrix
    cca_ref: np.ndarray               # L2-normalised canonical vectors
    cca_query: np.ndarray
    features: np.ndarray
    filter_features: np.ndarray
    n_anchors_unfiltered: int
    imputed: np.ndarray | None = None
    params: dict = field(default_factory=dict)

    @property
    def uncertainty(self) -> np.ndarray:
        return 1.0 - self.score

    def transfer_labels(self, labels) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Transfer another set of reference labels with the same anchor weights."""
        lab = np.asarray(labels).astype(str)[self.anchors["ref_idx"].to_numpy()]
        classes = np.unique(np.asarray(labels).astype(str))
        onehot = sp.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)),
                                pd.Index(classes).get_indexer(lab))),
                               shape=(len(lab), len(classes)))
        P = to_dense(self.weights @ onehot)
        best = P.argmax(axis=1)
        return classes[best], P[np.arange(len(best)), best], pd.DataFrame(P, columns=classes)

    def impute(self, ref_values) -> np.ndarray:
        """Weighted-average transfer of continuous reference values (n_ref x m)."""
        V = ref_values[self.anchors["ref_idx"].to_numpy()]
        return to_dense(self.weights @ (V if sp.issparse(V) else np.asarray(V, float)))

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"pred": self.pred, "score": self.score,
                             "uncertainty": self.uncertainty})


def _standardize_cells(S: np.ndarray) -> np.ndarray:
    """Seurat ``Standardize``: centre and scale each cell across features."""
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (S - mu) / sd


def _l2(Z: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Z / n


def run_cca(S_ref: np.ndarray, S_query: np.ndarray, n_cc: int = 30, seed: int = 0
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seurat ``RunCCA``: SVD of the cross-product of the cell-standardised scaled
    matrices (diagonal CCA). The (n_ref x n_query) cross-product is never formed."""
    A, B = _standardize_cells(S_ref), _standardize_cells(S_query)
    n_cc = int(min(n_cc, min(A.shape[0], B.shape[0]) - 1))
    op = LinearOperator((A.shape[0], B.shape[0]), dtype=float,
                        matvec=lambda v: A @ (B.T @ np.ravel(v)),
                        rmatvec=lambda v: B @ (A.T @ np.ravel(v)),
                        matmat=lambda V: A @ (B.T @ V), rmatmat=lambda V: B @ (A.T @ V))
    v0 = np.random.default_rng(seed).uniform(-1, 1, min(op.shape))
    U, d, Vt = svds(op, k=n_cc, v0=v0)
    order = np.argsort(-d)
    U, d, V = U[:, order], d[order], Vt[order].T
    # Seurat sign convention: first entry (first reference cell) positive
    sgn = np.sign(U[0])
    sgn[sgn == 0] = 1.0
    return U * sgn, V * sgn, d


def _indicator(idx: np.ndarray, n_cols: int, offset: int = 0) -> sp.csr_matrix:
    n, k = idx.shape
    return sp.csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), idx.ravel() + offset)),
                         shape=(n, n_cols))


def _top_dim_features(loadings: np.ndarray, max_features: int) -> np.ndarray:
    """Seurat ``TopDimFeatures``: grow per-dimension top lists (both signs) until
    ``max_features`` features are collected."""
    order_pos = np.argsort(-loadings, axis=0)
    order_neg = np.argsort(loadings, axis=0)
    chosen: list = []
    seen = set()
    for r in range(loadings.shape[0]):
        for d in range(loadings.shape[1]):
            for f in (order_pos[r, d], order_neg[r, d]):
                if f not in seen:
                    seen.add(f)
                    chosen.append(f)
        if len(chosen) >= max_features:
            break
    return np.sort(np.asarray(chosen[:max_features], int))


def _pca_loadings(S: np.ndarray, n: int, seed: int) -> np.ndarray:
    n = int(min(n, min(S.shape) - 1))
    _, _, Vt = randomized_svd(S - S.mean(axis=0), n_components=n, random_state=seed)
    return Vt.T


def cca_anchor_transfer(ref_X_lognorm, ref_labels, query_X_lognorm, genes_shared,
                        n_cc: int = 30, k_anchor: int = 5, k_filter: int | None = 200,
                        k_score: int = 30, k_weight: int = 50, sd_weight: float = 1.0, *,
                        ref_genes=None, query_genes=None, max_filter_features: int = 200,
                        weight_reduction="pcaproject", n_pcs: int = 30, ref_values=None,
                        seed: int = 0) -> AnchorTransferResult:
    """Seurat v3 ``FindTransferAnchors(reduction="cca")`` + ``TransferData``.

    ``ref_X_lognorm`` / ``query_X_lognorm`` are log-normalised (cells x genes) matrices
    whose columns are ``genes_shared`` -- or, if ``ref_genes`` / ``query_genes`` are given,
    are subset to ``genes_shared`` by name. Steps:

    1. ScaleData each dataset separately; ``RunCCA`` (SVD of the cell-standardised
       cross-product); L2-normalise the canonical vectors of every cell.
    2. Anchors = mutual nearest neighbours (``k_anchor``) between reference and query.
    3. ``FilterAnchors``: keep an anchor only if its query cell is among the reference
       cell's ``k_filter`` nearest query cells in the original (cosine-normalised,
       log-normalised) space restricted to the top ``max_filter_features`` CCA-loading
       features; skipped when ``k_filter`` is None.
    4. ``ScoreAnchors``: shared-neighbour overlap of the two anchor cells' neighbourhoods
       (``k_score`` within + across), rescaled to [0, 1] by the 0.01/0.9 quantiles.
    5. ``FindWeights``: for each query cell, its ``k_weight`` nearest anchor query cells
       in ``weight_reduction`` space ("pcaproject": reference PCA projected onto the
       query; "cca": the L2 CCA vectors; or an (n_query x d) array, e.g. ATAC LSI);
       ``d~ = 1 - d / d_k``, ``w = 1 - exp(-d~ * anchor_score / (2/sd_weight)^2)``,
       normalised per query cell.
    6. Labels: prediction scores = ``W @ onehot(labels of anchor reference cells)``;
       predicted label = argmax, prediction score = max. Continuous ``ref_values``
       (n_ref x m) are transferred as weighted averages (``imputed``).
    """
    genes_shared = np.asarray(genes_shared).astype(str)
    Xr = subset_genes(ref_X_lognorm, ref_genes, genes_shared)[0] if ref_genes is not None \
        else to_csr(ref_X_lognorm)
    Xq = subset_genes(query_X_lognorm, query_genes, genes_shared)[0] if query_genes is not None \
        else to_csr(query_X_lognorm)
    if Xr.shape[1] != len(genes_shared) or Xq.shape[1] != len(genes_shared):
        raise ValueError("matrix columns must match genes_shared (or pass ref_genes/query_genes)")
    ref_labels = np.asarray(ref_labels).astype(str)
    n_r, n_q = Xr.shape[0], Xq.shape[0]
    Sr = scale_apply(Xr, *scale_fit(Xr))
    Sq = scale_apply(Xq, *scale_fit(Xq))

    # 1. CCA
    U, V, d = run_cca(Sr, Sq, n_cc, seed)
    Cr, Cq = _l2(U), _l2(V)

    # 2. mutual nearest neighbours
    k_nb = max(k_anchor, k_score)
    _, nn_rr = knn(Cr, k=k_nb + 1)                    # self included first
    _, nn_qq = knn(Cq, k=k_nb + 1)
    _, nn_rq = knn(Cq, Cr, k=k_nb)                    # ref cell -> query neighbours
    _, nn_qr = knn(Cr, Cq, k=k_nb)                    # query cell -> ref neighbours
    mutual = _indicator(nn_rq[:, :k_anchor], n_q).multiply(
        _indicator(nn_qr[:, :k_anchor], n_r).T).tocoo()
    a_r, a_q = mutual.row.astype(int), mutual.col.astype(int)
    n_unfiltered = len(a_r)
    if n_unfiltered == 0:
        raise RuntimeError("no anchors found between reference and query")

    # 3. filter anchors in the original feature space
    load = np.vstack([Sr, Sq]).T @ np.vstack([U, V])            # feature loadings
    ffeat = _top_dim_features(load, min(max_filter_features, len(genes_shared)))
    if k_filter:
        Fr = _l2(to_dense(Xr[:, ffeat]))
        Fq = _l2(to_dense(Xq[:, ffeat]))
        ur = np.unique(a_r)
        _, nn_f = knn(Fq, Fr[ur], k=min(k_filter, n_q))
        F = _indicator(nn_f, n_q)
        pos = pd.Index(ur).get_indexer(a_r)
        keep = np.asarray(F[pos, a_q]).ravel() > 0
        if keep.sum() == 0:
            keep[:] = True                            # nothing survives: skip filtering
        a_r, a_q = a_r[keep], a_q[keep]

    # 4. score anchors by shared neighbours
    NA = _indicator(nn_rr[:, :k_score], n_r + n_q) + _indicator(nn_rq[:, :k_score], n_r + n_q, n_r)
    NB = _indicator(nn_qr[:, :k_score], n_r + n_q) + _indicator(nn_qq[:, :k_score], n_r + n_q, n_r)
    raw = np.asarray(NA[a_r].multiply(NB[a_q]).sum(axis=1)).ravel()
    lo, hi = np.quantile(raw, 0.01), np.quantile(raw, 0.9)
    score = np.clip((raw - lo) / (hi - lo) if hi > lo else np.ones_like(raw, float), 0, 1)
    anchors = pd.DataFrame({"ref_idx": a_r, "query_idx": a_q, "score_raw": raw, "score": score})

    # 5. anchor weights per query cell
    if isinstance(weight_reduction, str) and weight_reduction == "pcaproject":
        Lp = _pca_loadings(Sr, n_pcs, seed)
        Eq = Sq @ Lp
    elif isinstance(weight_reduction, str) and weight_reduction == "cca":
        Eq = Cq
    else:
        Eq = np.asarray(weight_reduction, float)
        if Eq.shape[0] != n_q:
            raise ValueError("weight_reduction array must have one row per query cell")
    W = anchor_weights(Eq, anchors, k_weight=k_weight, sd_weight=sd_weight)

    # 6. transfer
    res = AnchorTransferResult(pred=np.array([]), score=np.array([]), probs=pd.DataFrame(),
                               anchors=anchors, weights=W, cca_ref=Cr, cca_query=Cq,
                               features=genes_shared, filter_features=genes_shared[ffeat],
                               n_anchors_unfiltered=n_unfiltered,
                               params=dict(n_cc=U.shape[1], k_anchor=k_anchor, k_filter=k_filter,
                                           k_score=k_score, k_weight=k_weight,
                                           sd_weight=sd_weight,
                                           weight_reduction=weight_reduction
                                           if isinstance(weight_reduction, str) else "custom",
                                           cca_sdev=d.tolist()))
    res.pred, res.score, res.probs = res.transfer_labels(ref_labels)
    if ref_values is not None:
        res.imputed = res.impute(ref_values)
    return res


def anchor_weights(Eq: np.ndarray, anchors: pd.DataFrame, k_weight: int = 50,
                   sd_weight: float = 1.0) -> sp.csr_matrix:
    """Seurat ``FindWeights`` (TransferData): (n_query x n_anchors) weight matrix."""
    a_q = anchors["query_idx"].to_numpy()
    sc = anchors["score"].to_numpy(float)
    uq = np.unique(a_q)
    k = int(min(k_weight, len(uq)))
    dist, idx = knn(Eq[uq], Eq, k=k)
    dmax = dist[:, -1:].copy()
    dmax[dmax == 0] = 1.0
    dt = 1.0 - dist / dmax
    n_q = Eq.shape[0]
    Dt = sp.csr_matrix((dt.ravel(), (np.repeat(np.arange(n_q), k), idx.ravel())),
                       shape=(n_q, len(uq)))
    Dt.eliminate_zeros()                              # d~ > min_dist (= 0) only
    Ua = sp.csr_matrix((np.ones(len(a_q)), (pd.Index(uq).get_indexer(a_q), np.arange(len(a_q)))),
                       shape=(len(uq), len(a_q)))
    base = (Dt @ Ua).tocsr()                          # d~ of each query cell to each anchor
    W = base.multiply(sc[None, :]).tocsr()
    W.data = 1.0 - np.exp(-W.data / (2.0 / sd_weight) ** 2)
    rs = np.asarray(W.sum(axis=1)).ravel()
    empty = rs <= 0
    if empty.any():                                   # all neighbouring anchors scored 0
        B = base.copy()
        B.data = 1.0 - np.exp(-B.data / (2.0 / sd_weight) ** 2)
        W = sp.vstack([W[i] if not empty[i] else B[i] for i in range(n_q)]).tocsr()
        rs = np.asarray(W.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return (sp.diags(1.0 / rs) @ W).tocsr()


def gene_activity_transfer(ref_rna, ref_labels, query_atac, gene_coords, *,
                           ref_batch_key: str | None = None, n_features: int = 2000,
                           upstream: int = 2000, weight_reduction="lsi", n_lsi: int = 30,
                           **kwargs) -> tuple[AnchorTransferResult, ad.AnnData]:
    """Feature-conversion route of Figure 3A: peaks -> gene activity -> CCA anchors.

    ``weight_reduction="lsi"`` uses the query's own LSI (on peaks) for the anchor
    weights, as in the Seurat v3 scATAC vignette; other values go to
    :func:`cca_anchor_transfer`. Returns the transfer result and the gene-activity AnnData.
    """
    ga = gene_activity(query_atac, gene_coords, upstream=upstream)
    Xr = normalize_total_log1p(ref_rna.X)
    Xq = normalize_total_log1p(ga.X)
    batch = ref_rna.obs[ref_batch_key].astype(str).to_numpy() if ref_batch_key else None
    feats = select_transfer_features(Xr, ref_rna.var_names, ga.var_names, n_features,
                                     batch=batch, query_X=ga.X)
    if isinstance(weight_reduction, str) and weight_reduction == "lsi":
        weight_reduction = lsi_fit(query_atac.X, query_atac.var_names, n_lsi).embedding
    res = cca_anchor_transfer(Xr, ref_labels, Xq, feats, ref_genes=ref_rna.var_names,
                              query_genes=ga.var_names, weight_reduction=weight_reduction,
                              **kwargs)
    return res, ga


# ======================================================================== LSI


def tfidf(X, idf: np.ndarray | None = None, scale_factor: float = 1e4
          ) -> tuple[sp.csr_matrix, np.ndarray]:
    """Signac ``RunTFIDF`` method 1: ``log1p(tf * idf * scale_factor)`` with
    ``tf = counts / cell total`` and ``idf = n_cells / peak total``."""
    X = to_csr(X)
    tot = np.asarray(X.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    if idf is None:
        ps = np.asarray(X.sum(axis=0)).ravel()
        idf = np.where(ps > 0, X.shape[0] / np.maximum(ps, 1e-12), 0.0)
    T = (sp.diags(1.0 / tot) @ X) @ sp.diags(idf * scale_factor)
    T = T.tocsr()
    T.data = np.log1p(T.data)
    return T, idf


@dataclass
class LSIModel:
    """TF-IDF + truncated SVD fitted on one dataset and re-applicable to another."""
    peaks: np.ndarray
    idf: np.ndarray
    V: np.ndarray            # (n_peaks, k) feature loadings
    d: np.ndarray            # singular values
    mean: np.ndarray         # embedding standardisation (Signac scale.embeddings)
    sd: np.ndarray
    keep: np.ndarray         # component mask (component 1 dropped if depth-correlated)
    depth_cor: np.ndarray
    embedding: np.ndarray    # standardised embedding of the fitting cells (kept comps)

    def transform(self, X, peaks) -> np.ndarray:
        """Project new cells (Seurat ``ProjectSVD``) using the fitted IDF and loadings."""
        Xs, _ = subset_genes(X, peaks, self.peaks)
        T, _ = tfidf(Xs, idf=self.idf)
        U = (T @ self.V) / self.d
        return ((U - self.mean) / self.sd)[:, self.keep]


def lsi_fit(X, peaks, n_components: int = 30, depth_cor_threshold: float = 0.5,
            seed: int = 0) -> LSIModel:
    """Latent semantic indexing of peak counts; component 1 is dropped when its
    absolute correlation with log sequencing depth exceeds ``depth_cor_threshold``
    (Signac ``DepthCor``; the first LSI component usually captures depth)."""
    T, idf = tfidf(X)
    k = int(min(n_components, min(T.shape) - 1))
    U, d, Vt = randomized_svd(T, n_components=k, random_state=seed)
    sgn = np.sign(Vt[np.arange(k), np.abs(Vt).argmax(axis=1)])
    U, Vt = U * sgn, Vt * sgn[:, None]
    mean, sd = U.mean(axis=0), U.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    depth = np.log1p(np.asarray(to_csr(X).sum(axis=1)).ravel())
    dc = np.array([np.corrcoef(U[:, i], depth)[0, 1] if U[:, i].std() > 0 else 0.0
                   for i in range(k)])
    keep = np.ones(k, bool)
    if abs(dc[0]) > depth_cor_threshold:
        keep[0] = False
    emb = ((U - mean) / sd)[:, keep]
    return LSIModel(peaks=np.asarray(peaks).astype(str), idf=idf, V=Vt.T, d=d, mean=mean,
                    sd=sd, keep=keep, depth_cor=dc, embedding=emb)


# ============================================================ bridge integration


@dataclass
class BridgeResult:
    labels: pd.DataFrame              # pred / uncertainty (per level if hierarchical)
    Z_ref: np.ndarray                 # reference cells in the reduced bridge space
    Z_query: np.ndarray               # query cells in the reduced bridge space
    Z_bridge: np.ndarray              # bridge cells' Laplacian eigenmap coordinates
    knn_idx: np.ndarray
    knn_dist: np.ndarray
    eigenvalues: np.ndarray
    dict_ref: sp.csr_matrix | np.ndarray   # reference cells x bridge cells
    dict_query: sp.csr_matrix | np.ndarray # query cells x bridge cells
    lsi_dropped_first: bool
    params: dict = field(default_factory=dict)


def _dictionary(X: np.ndarray, B: np.ndarray, method: str, k: int):
    """Represent cells ``X`` (n x d) as combinations of bridge atoms ``B`` (n_b x d).

    ``knn``: non-negative weights over the cell's ``k`` nearest bridge cells (scArches
    Gaussian kernel, rows sum to 1) -- local, robust to non-linear structure and to
    modality-specific offsets far from the cell. ``pinv``: Seurat v5's minimum-norm
    least-squares weights ``X @ pinv(B)`` over all bridge cells (``X ~ W @ B``).
    """
    if method == "pinv":
        return X @ np.linalg.pinv(B)
    if method != "knn":
        raise ValueError(f"unknown dictionary method {method!r}")
    dist, idx = knn(B, X, k=k)
    w = knn_weights(dist)
    n = X.shape[0]
    return sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), idx.shape[1]), idx.ravel())),
                         shape=(n, B.shape[0]))


def laplacian_eigenmaps(A: sp.spmatrix, n_components: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Bottom non-trivial eigenvectors of the symmetric normalised graph Laplacian
    (``RunGraphLaplacian``), returned as random-walk coordinates ``D^-1/2 u``."""
    A = sp.csr_matrix(A, dtype=float)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    Dm = sp.diags(1.0 / np.sqrt(deg))
    S = Dm @ A @ Dm
    m = int(min(n_components + 1, A.shape[0] - 2))
    if A.shape[0] <= 3000:
        ev, U = np.linalg.eigh(S.toarray())
        ev, U = ev[::-1][:m], U[:, ::-1][:, :m]
    else:
        ev, U = eigsh(S, k=m, which="LA", v0=np.ones(A.shape[0]))
        o = np.argsort(-ev)
        ev, U = ev[o], U[:, o]
    E = (Dm @ U)[:, 1:]
    E /= np.linalg.norm(E, axis=0, keepdims=True)
    return E * np.sqrt(A.shape[0]), 1.0 - ev[1:]


def bridge_integration(ref_rna, ref_labels, bridge_rna, bridge_atac, query_atac, *,
                       ref_batch_key: str | None = None, reference=None, n_hvg: int = 2000,
                       n_pcs: int = 30, n_lsi: int = 30, n_lap: int = 30, k_graph: int = 20,
                       dictionary: str = "pinv", k_dict: int = 20, k: int = 30,
                       unknown_threshold: float = 0.5, seed: int = 0) -> BridgeResult:
    """Seurat v5 bridge integration (dictionary learning) of an RNA reference and an
    ATAC query through a multiome bridge.

    1. RNA space: reference log-normalisation -> (batch-aware) HVGs -> scaling -> PCA;
       bridge RNA is projected with the reference's scaling and loadings. If a
       ``refmap.reference.Reference`` (or ``ref_batch_key``) is supplied, the
       Harmony-integrated reference space is used and bridge cells are mapped into it
       with Symphony.
    2. ATAC space: LSI fitted on the bridge ATAC; query projected with the bridge IDF and
       loadings (peaks matched by name); depth-correlated component 1 dropped.
    3. Dictionary representation: reference cells as combinations of bridge cells in RNA
       space, query cells as combinations of bridge cells in ATAC space (see
       :func:`_dictionary`; default ``knn``).
    4. Reduction: Laplacian eigenmaps of the bridge graph (union of the RNA and ATAC kNN
       graphs of the bridge cells, a simple stand-in for WNN); every cell's dictionary
       weights are multiplied by the eigenvectors and L2-normalised.
    5. Weighted-kNN label transfer (``refmap.transfer``) with uncertainty.

    ``ref_labels`` is an array or a DataFrame of hierarchical levels (coarse -> fine).
    """
    if not np.array_equal(np.asarray(bridge_rna.obs_names), np.asarray(bridge_atac.obs_names)):
        raise ValueError("bridge RNA and ATAC must have identical obs_names (paired cells)")
    # -- RNA space
    if reference is None and ref_batch_key is not None:
        from .reference import build_reference
        tmp = ad.AnnData(X=ref_rna.X, obs=pd.DataFrame(
            {"_label": np.asarray(ref_labels.iloc[:, -1] if isinstance(ref_labels, pd.DataFrame)
                                  else ref_labels).astype(str),
             "_batch": ref_rna.obs[ref_batch_key].astype(str).to_numpy()},
            index=ref_rna.obs_names), var=pd.DataFrame(index=ref_rna.var_names))
        reference = build_reference(tmp, ["_label"], batch_key="_batch", n_hvg=n_hvg,
                                    n_pcs=n_pcs, calibrate=False, seed=seed)
    if reference is not None:
        from .harmony import symphony_map
        Zr = reference.Z_corr
        Zb0, _ = reference.project(bridge_rna.X, bridge_rna.var_names)
        Zb, _ = symphony_map(Zb0, reference.comp, None)
    else:
        Xr = normalize_total_log1p(ref_rna.X)
        genes = select_hvgs(Xr, ref_rna.var_names, n_top=n_hvg)
        Xr_s, _ = subset_genes(Xr, ref_rna.var_names, genes)
        mean, sd = scale_fit(Xr_s)
        Sr = scale_apply(Xr_s, mean, sd)
        Lp = _pca_loadings(Sr, n_pcs, seed)
        Zr = Sr @ Lp
        Xb_s, _ = subset_genes(normalize_total_log1p(bridge_rna.X), bridge_rna.var_names, genes)
        Zb = scale_apply(Xb_s, mean, sd) @ Lp
    # -- ATAC space
    lsi = lsi_fit(bridge_atac.X, bridge_atac.var_names, n_lsi, seed=seed)
    Lb = lsi.embedding
    Lq = lsi.transform(query_atac.X, query_atac.var_names)
    # -- bridge graph and Laplacian eigenmaps
    from .graph import knn_graph
    A = knn_graph(Zb, k_graph) + knn_graph(Lb, k_graph)
    E, evals = laplacian_eigenmaps(A, n_lap)
    # -- dictionary representations -> common space
    Wr = _dictionary(Zr, Zb, dictionary, k_dict)
    Wq = _dictionary(Lq, Lb, dictionary, k_dict)
    Pr = _l2(np.asarray(Wr @ E))
    Pq = _l2(np.asarray(Wq @ E))
    # -- transfer
    dist, idx = knn(Pr, Pq, k=k)
    w = knn_weights(dist)
    if isinstance(ref_labels, pd.DataFrame):
        obs = ref_labels.astype(str).reset_index(drop=True)
        labels = transfer_hierarchical(obs, list(obs.columns), idx, w, unknown_threshold)
    else:
        labels = transfer_labels(np.asarray(ref_labels).astype(str), idx, w, unknown_threshold)
    labels.index = np.asarray(query_atac.obs_names)
    return BridgeResult(labels=labels, Z_ref=Pr, Z_query=Pq, Z_bridge=E, knn_idx=idx,
                        knn_dist=dist, eigenvalues=evals, dict_ref=Wr, dict_query=Wq,
                        lsi_dropped_first=bool(not lsi.keep[0]),
                        params=dict(dictionary=dictionary, k_dict=k_dict, n_lap=E.shape[1],
                                    k_graph=k_graph, k=k, n_pcs=Zr.shape[1],
                                    n_lsi=int(lsi.keep.sum()),
                                    rna_space="harmony" if reference is not None else "pca",
                                    lsi_depth_cor=lsi.depth_cor[:3].round(3).tolist()))


__all__ = ["gene_activity", "select_transfer_features", "cca_anchor_transfer",
           "AnchorTransferResult", "anchor_weights", "run_cca", "gene_activity_transfer",
           "tfidf", "lsi_fit", "LSIModel", "bridge_integration", "BridgeResult",
           "laplacian_eigenmaps"]
