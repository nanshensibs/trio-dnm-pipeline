"""Data compression for building references from millions of cells.

The perspective's closing section lists *data compression* strategies -- aggregating
homogeneous cells into metacells (MetaCell, Baran et al. 2019, ref. 111; SEACells,
Persad et al. 2023, ref. 98) and sketching (geometric sketching, Hie et al. 2019,
ref. 110) -- together with out-of-core processing as the way to keep reference
construction tractable as atlases grow. This module provides:

* ``metacells`` -- k-means metacells in the reference latent space, fitted within
  each group of ``by`` (batch/sample) so a metacell never mixes batches; counts are
  summed per metacell and labels summarised by majority vote and purity. (SEACells
  uses kernel archetypal analysis and MetaCell a graph partition; k-means on the
  integrated latent is the simplest homogeneous-aggregation stand-in.)
* ``geometric_sketch`` -- plaid covering with equal-side hypercubes, binary search
  over the side so that the number of non-empty boxes matches the sketch size, one
  cell per box, so rare states are sampled in proportion to the *volume* they occupy
  rather than their abundance.
* ``incremental_reference`` -- an out-of-core version of ``build_reference``: chunk
  passes for per-gene moments (HVG selection, scaling) and sklearn
  ``IncrementalPCA``, then Harmony/Symphony compression and calibration on the
  (small) latent matrix, optionally Harmony on a geometric sketch with Symphony
  mapping of the remaining cells.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA

from .harmony import compress_reference, run_harmony, symphony_map
from .preprocess import normalize_total_log1p, scale_apply, subset_genes, to_csr
from .reference import Reference, _jsonable, calibrate_reference

# ------------------------------------------------------------------ metacells


@dataclass
class MetacellResult:
    membership: np.ndarray          # (n_cells,) metacell index
    obs: pd.DataFrame               # per metacell: group, n_cells, majority labels, purity
    Z: np.ndarray                   # (n_metacells, d) latent centroids
    adata: ad.AnnData | None = None  # summed raw counts (metacells x genes)

    @property
    def n_metacells(self) -> int:
        return len(self.obs)

    def purity(self, key: str, weighted: bool = True) -> float:
        """Mean label purity (cell-weighted by default)."""
        p = self.obs[f"{key}_purity"].to_numpy()
        return float(np.average(p, weights=self.obs["n_cells"]) if weighted else p.mean())


def metacells(Z: np.ndarray, n_metacells: int | None = None, *,
              cells_per_metacell: int | None = None, by=None, adata=None,
              label_keys=(), layer: str | None = None, seed: int = 0) -> MetacellResult:
    """Aggregate homogeneous cells into metacells.

    Give either ``n_metacells`` (total, allocated to groups proportionally to their
    size) or ``cells_per_metacell`` (default 20). ``by`` is an array or an
    ``adata.obs`` column (e.g. batch or sample); clustering is done within each group.
    With ``adata``, raw counts (``X`` or ``layer``) are summed per metacell and
    ``label_keys`` columns are summarised by majority label and purity.
    """
    Z = np.asarray(Z, float)
    n = Z.shape[0]
    if isinstance(by, str):
        if adata is None:
            raise ValueError("by given as a column name requires adata")
        by_name, by = by, adata.obs[by].astype(str).to_numpy()
    else:
        by_name = "group"
        by = np.full(n, "all") if by is None else np.asarray(by).astype(str)
    if n_metacells is None and cells_per_metacell is None:
        cells_per_metacell = 20
    membership = np.empty(n, dtype=np.int64)
    groups, offset = [], 0
    for g in pd.unique(by):
        rows = np.flatnonzero(by == g)
        if n_metacells is not None:
            k = int(round(n_metacells * len(rows) / n))
        else:
            k = int(round(len(rows) / cells_per_metacell))
        k = int(np.clip(k, 1, len(rows)))
        if k == 1:
            lab = np.zeros(len(rows), int)
        else:
            km = (KMeans(n_clusters=k, n_init=3, random_state=seed) if len(rows) <= 50000
                  else MiniBatchKMeans(n_clusters=k, n_init=3, random_state=seed,
                                       batch_size=4096))
            lab = km.fit_predict(Z[rows])
        _, lab = np.unique(lab, return_inverse=True)   # drop empty clusters
        membership[rows] = lab + offset
        groups += [g] * (lab.max() + 1)
        offset += lab.max() + 1
    M = sp.csr_matrix((np.ones(n), (membership, np.arange(n))), shape=(offset, n))
    sizes = np.asarray(M.sum(axis=1)).ravel()
    obs = pd.DataFrame({by_name: groups, "n_cells": sizes.astype(int)},
                       index=[f"mc{i}" for i in range(offset)])
    for key in label_keys:
        lab = pd.DataFrame({"mc": membership, "l": adata.obs[key].astype(str).to_numpy()})
        tab = pd.crosstab(lab["mc"], lab["l"])
        obs[key] = tab.idxmax(axis=1).to_numpy()
        obs[f"{key}_purity"] = (tab.max(axis=1) / tab.sum(axis=1)).to_numpy()
    Zm = (M @ Z) / sizes[:, None]
    out = None
    if adata is not None:
        X = adata.layers[layer] if layer else adata.X
        Xm = M @ to_csr(X)
        out = ad.AnnData(X=sp.csr_matrix(Xm), obs=obs.copy(),
                         var=pd.DataFrame(index=adata.var_names.copy()))
        out.obsm["X_latent"] = Zm
    return MetacellResult(membership=membership, obs=obs, Z=Zm, adata=out)


# ----------------------------------------------------------- geometric sketch


def _box_keys(X: np.ndarray, side: float, mult: np.ndarray) -> np.ndarray:
    """Hash each cell's hypercube coordinates to one int64 (wrapping arithmetic)."""
    coords = np.floor(X / side).astype(np.int64)
    with np.errstate(over="ignore"):
        return (coords * mult[None, :]).sum(axis=1)


def geometric_sketch(Z: np.ndarray, n: int, seed: int = 0, *, n_dims: int | None = None,
                     tol: float = 0.1, max_iter: int = 60, top_up: str = "uniform",
                     return_info: bool = False):
    """Geometric sketching (Hie et al. 2019, ref. 110).

    Cells are min-shifted and divided by the global range (geometry preserved), the
    space is covered by a plaid grid of hypercubes of side ``s``, and ``s`` is found by
    binary search so the number of non-empty boxes falls in ``[n, n * (1 + tol)]``.
    ``n`` boxes are drawn uniformly and one cell is drawn from each. If fewer than
    ``n`` boxes are reachable, every box contributes one cell and the remainder is
    drawn uniformly (``top_up='uniform'``) or round-robin over boxes as in the
    ``geosketch`` package (``top_up='boxes'``). ``n_dims`` keeps the leading latent
    dimensions (Hie et al. sketch on top PCs). Returns sorted cell indices (and an
    info dict with ``return_info``).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(Z, float)
    if n_dims:
        X = X[:, :n_dims]
    N = X.shape[0]
    if n >= N:
        idx = np.arange(N)
        return (idx, dict(side=0.0, n_boxes=N, iterations=0)) if return_info else idx
    X = X - X.min(axis=0)
    X = X / max(X.max(), 1e-12)
    mult = rng.integers(1, 2**62, X.shape[1], dtype=np.int64) | 1
    lo, hi = 1e-6, 1.0 + 1e-9            # small side -> many boxes; large -> few
    best = None
    for it in range(max_iter):
        side = np.sqrt(lo * hi)          # geometric bisection over scales
        keys = _box_keys(X, side, mult)
        nb = len(np.unique(keys))
        if nb >= n and (best is None or nb < best[1]):
            best = (side, nb, keys)
        if n <= nb <= n * (1 + tol):
            break
        if nb < n:
            hi = side
        else:
            lo = side
    if best is None:                     # even the finest grid has < n boxes
        side = lo
        keys = _box_keys(X, side, mult)
        best = (side, len(np.unique(keys)), keys)
    side, nb, keys = best
    _, box = np.unique(keys, return_inverse=True)
    order = rng.permutation(N)
    # one random cell per box: first occurrence of each box in a random permutation
    _, first = np.unique(box[order], return_index=True)
    reps = order[first]
    if len(reps) >= n:
        chosen = rng.choice(reps, n, replace=False)
    elif top_up == "uniform":
        rest = np.setdiff1d(np.arange(N), reps)
        chosen = np.concatenate([reps, rng.choice(rest, n - len(reps), replace=False)])
    elif top_up == "boxes":
        taken = np.zeros(N, bool)
        taken[reps] = True
        chosen = list(reps)
        while len(chosen) < n:
            o = rng.permutation(np.flatnonzero(~taken))
            _, f = np.unique(box[o], return_index=True)
            add = rng.permutation(o[f])[: n - len(chosen)]
            taken[add] = True
            chosen += add.tolist()
        chosen = np.asarray(chosen)
    else:
        raise ValueError(f"unknown top_up {top_up!r}")
    idx = np.sort(chosen)
    return (idx, dict(side=float(side), n_boxes=int(nb), iterations=it + 1)) if return_info \
        else idx


# ------------------------------------------------------ incremental reference


def _chunk_source(data, chunk_size: int):
    """Return a zero-argument callable yielding AnnData chunks (re-iterable)."""
    if isinstance(data, ad.AnnData):
        return lambda: (data[i:i + chunk_size] for i in range(0, data.n_obs, chunk_size))
    if isinstance(data, (list, tuple)):
        return lambda: iter(data)
    if callable(data):
        return data
    raise TypeError("pass an AnnData, a list of AnnData chunks or a callable returning a "
                    "fresh iterator of chunks (several passes are needed)")


def _dispersion_z(mean: np.ndarray, var: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """``preprocess._binned_dispersion_rank`` computed from streamed moments."""
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.log(np.where(mean > 0, var / mean, np.nan))
    df = pd.DataFrame({"disp": disp, "bin": pd.cut(mean, bins=n_bins)})
    grp = df.groupby("bin", observed=False)["disp"]
    sd = grp.transform("std").fillna(1.0).replace(0, 1.0)
    z = ((df["disp"] - grp.transform("mean")) / sd).to_numpy()
    z[~np.isfinite(z)] = -np.inf
    z[mean <= 0.0125] = -np.inf
    return z


def _moments(n, s, ss):
    mean = s / n
    var = np.clip((ss / n - mean ** 2) * n / max(n - 1, 1), 0, None)
    return mean, var


def _select_hvgs_streamed(stats: dict, genes: np.ndarray, n_top: int) -> np.ndarray:
    """Same ranking as ``preprocess.select_hvgs`` from per-batch moments."""
    n_top = min(n_top, len(genes))
    tot_n = sum(v[0] for v in stats.values())
    tot = _moments(tot_n, sum(v[1] for v in stats.values()), sum(v[2] for v in stats.values()))
    usable = {b: v for b, v in stats.items() if v[0] >= 20}
    if len(stats) < 2 or not usable:
        order = np.argsort(-_dispersion_z(*tot), kind="stable")
        return genes[np.sort(order[:n_top])]
    ranks, n_hv = [], np.zeros(len(genes))
    for b, (nb, s, ss) in usable.items():
        z = _dispersion_z(*_moments(nb, s, ss))
        r = np.empty(len(genes))
        r[np.argsort(-z, kind="stable")] = np.arange(len(genes))
        ranks.append(r)
        n_hv += r < n_top
    order = np.lexsort((np.median(np.vstack(ranks), axis=0), -n_hv))
    return genes[np.sort(order[:n_top])]


def incremental_reference(data, label_keys, *, chunk_size: int = 10000,
                          batch_key: str | None = None, sample_key: str | None = None,
                          name: str = "reference", version: str = "1.0.0",
                          n_hvg: int = 2000, n_pcs: int = 30, hvg_genes=None,
                          layer: str | None = None, harmony_kwargs: dict | None = None,
                          sigma: float = 0.1, k_calibration: int = 30, calibrate: bool = True,
                          harmony_sketch: int | None = None, seed: int = 0,
                          provenance: dict | None = None) -> Reference:
    """Out-of-core ``build_reference`` (``transform='pca'``).

    ``data`` is an AnnData (sliced into ``chunk_size`` rows; works with backed
    AnnData), a list of AnnData chunks, or a callable returning a fresh chunk
    iterator. Passes over the chunks:

    1. log-normalised per-gene sums and sums of squares, per batch -> batch-aware HVG
       ranking (same rule as ``select_hvgs``) and the reference ``mean``/``sd``;
       ``obs`` is collected;
    2. ``IncrementalPCA.partial_fit`` on scaled HVG chunks -> loadings;
    3. projection of every chunk -> latent ``Z`` (n x n_pcs, held in memory).

    Harmony, Symphony compression and self-mapping calibration then run on ``Z``.
    With ``harmony_sketch``, Harmony is fitted on a geometric sketch of that many
    cells and the remaining cells are corrected by Symphony mapping (per batch).
    Unlike ``_pca`` in ``build_reference``, ``IncrementalPCA`` centres the scaled data
    (whose mean is ~0 up to clipping), so loadings agree up to numerical error.
    """
    if isinstance(label_keys, str):
        label_keys = [label_keys]
    label_keys = list(label_keys)
    chunks = _chunk_source(data, chunk_size)
    target_sum = 1e4

    def X_of(c):
        return c.layers[layer] if layer else c.X

    # ---- pass 1: moments + obs ---------------------------------------------
    genes_all, stats, obs_parts, n_chunks = None, {}, [], 0
    for c in chunks():
        if genes_all is None:
            genes_all = np.asarray(c.var_names)
        elif not np.array_equal(np.asarray(c.var_names), genes_all):
            raise ValueError("all chunks must share the same var_names")
        X_log = normalize_total_log1p(X_of(c), target_sum)
        b = c.obs[batch_key].astype(str).to_numpy() if batch_key else np.full(c.n_obs, "all")
        for lv in pd.unique(b):
            m = b == lv
            Xb = X_log[np.flatnonzero(m)] if not m.all() else X_log
            s = np.asarray(Xb.sum(axis=0)).ravel()
            ss = np.asarray(Xb.multiply(Xb).sum(axis=0)).ravel()
            if lv in stats:
                st = stats[lv]
                stats[lv] = (st[0] + m.sum(), st[1] + s, st[2] + ss)
            else:
                stats[lv] = (int(m.sum()), s, ss)
        obs_parts.append(c.obs.copy())
        n_chunks += 1
    obs_all = pd.concat(obs_parts)
    for parent, child in zip(label_keys[:-1], label_keys[1:]):
        nparent = obs_all.groupby(child, observed=True)[parent].nunique()
        if (nparent > 1).any():
            raise ValueError(f"labels in '{child}' have more than one '{parent}' parent")
    genes = (np.asarray(hvg_genes) if hvg_genes is not None
             else _select_hvgs_streamed(stats, genes_all, n_hvg))
    gidx = pd.Index(genes_all).get_indexer(genes)
    if (gidx < 0).any():
        raise ValueError("hvg_genes not found in the data")
    n_tot = sum(v[0] for v in stats.values())
    mean, var = _moments(n_tot, sum(v[1] for v in stats.values())[gidx],
                         sum(v[2] for v in stats.values())[gidx])
    sd = np.sqrt(var)
    sd[sd == 0] = 1.0

    def scaled(c):
        X_sub, _ = subset_genes(normalize_total_log1p(X_of(c), target_sum), genes_all, genes)
        return scale_apply(X_sub, mean, sd)

    # ---- pass 2: incremental PCA --------------------------------------------
    n_pcs = int(min(n_pcs, len(genes) - 1, n_tot - 1))
    ipca = IncrementalPCA(n_components=n_pcs)
    pending = None
    for c in chunks():
        S = scaled(c)
        if pending is None:
            pending = S
        elif pending.shape[0] >= n_pcs and S.shape[0] >= n_pcs:
            ipca.partial_fit(pending)
            pending = S
        else:
            pending = np.vstack([pending, S])
    if pending.shape[0] < n_pcs:
        raise ValueError("too few cells for the requested number of PCs")
    ipca.partial_fit(pending)
    V = ipca.components_.T
    loadings = V * np.sign(V[np.abs(V).argmax(axis=0), np.arange(V.shape[1])])

    # ---- pass 3: projection -------------------------------------------------
    Z = np.vstack([scaled(c) @ loadings for c in chunks()])

    # ---- integration, compression, assembly ---------------------------------
    batch = obs_all[batch_key].astype(str).to_numpy() if batch_key else None
    hk = dict(sigma=sigma, seed=seed)
    hk.update(harmony_kwargs or {})
    if harmony_sketch and harmony_sketch < len(Z):
        sk = geometric_sketch(Z, harmony_sketch, seed=seed)
        h = run_harmony(Z[sk], None if batch is None else batch[sk], **hk)
        comp0, _ = compress_reference(h.Z_corr, h.Y, hk["sigma"], hk.get("lamb", 1.0))
        Z_corr = np.empty_like(Z)
        Z_corr[sk] = h.Z_corr
        rest = np.setdiff1d(np.arange(len(Z)), sk)
        Z_corr[rest], _ = symphony_map(Z[rest], comp0, None if batch is None else batch[rest])
        Y, converged = h.Y, h.converged
    else:
        h = run_harmony(Z, batch, **hk)
        Z_corr, Y, converged = h.Z_corr, h.Y, h.converged
    comp, R = compress_reference(Z_corr, Y, hk["sigma"], hk.get("lamb", 1.0))

    obs_cols = list(dict.fromkeys(label_keys + [c for c in (batch_key, sample_key) if c]))
    obs = obs_all[obs_cols + [c for c in obs_all.columns if c not in obs_cols]].copy()
    for c in obs_cols:
        obs[c] = obs[c].astype(str)
    prov = dict(created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                parent_version=None, changelog=[f"{version}: initial release (incremental)"],
                datasets=sorted(obs[batch_key].unique().tolist()) if batch_key else [])
    prov.update(provenance or {})
    ref = Reference(
        name=name, version=version, genes=genes, mean=mean, sd=sd, loadings=loadings,
        Z=Z, Z_corr=Z_corr, obs=obs, label_keys=label_keys, batch_key=batch_key,
        sample_key=sample_key, comp=comp,
        params=dict(transform="pca", n_hvg=int(len(genes)), n_pcs=n_pcs,
                    target_sum=target_sum, clip=10.0, harmony=_jsonable(hk),
                    harmony_converged=bool(converged), n_centroids=int(len(comp.Nr)),
                    k_calibration=k_calibration,
                    incremental=dict(chunk_size=chunk_size, n_chunks=n_chunks,
                                     harmony_sketch=harmony_sketch)),
        provenance=prov)
    if calibrate:
        ref.calibration = calibrate_reference(ref, R, k=k_calibration, seed=seed)
    return ref


__all__ = ["MetacellResult", "metacells", "geometric_sketch", "incremental_reference"]
