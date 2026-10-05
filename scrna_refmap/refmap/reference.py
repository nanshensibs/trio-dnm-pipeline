"""Reference atlas: the learned transformation plus curated annotations.

The perspective defines a single-cell reference as two components: (1) a *data
transformation* projecting measurements into a low-dimensional space in which batch
effects are removed and similar biological states co-locate, and (2) *annotations*,
optionally following an ontology and exhibiting a hierarchical structure. This module
builds both, stores everything needed to re-apply the transformation to queries, and
attaches version/provenance metadata so references can be released and updated like
software (Git-style semantic versions; "Path toward machine-learning-based
open-source atlasing").

Transformations:

* ``pca``  -- PCA on scaled highly variable genes, then Harmony (Symphony-style
  statistical reference, ref. 5).
* ``spca`` -- supervised PCA (Seurat/Azimuth, refs. 6, 21): principal axes maximise
  the covariance of expression along a label-supervised kNN graph, so the
  annotated structure of the reference is emphasised in the latent space.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd

from . import __version__
from .graph import knn
from .harmony import (SymphonyCompression, compress_reference, l2_normalize,
                      mahalanobis_to_centroids, run_harmony, soft_assign, symphony_map)
from .preprocess import (normalize_total_log1p, scale_apply, scale_fit, select_hvgs,
                         subset_genes)
from .transfer import knn_weights, transfer_hierarchical


@dataclass
class Reference:
    name: str
    version: str
    genes: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    loadings: np.ndarray            # (n_genes, d)
    Z: np.ndarray                   # uncorrected reference embedding (n, d)
    Z_corr: np.ndarray              # integrated reference embedding (n, d)
    obs: pd.DataFrame
    label_keys: list
    batch_key: str | None
    sample_key: str | None
    comp: SymphonyCompression
    continuous: dict = field(default_factory=dict)        # name -> (n, m) values
    continuous_names: dict = field(default_factory=dict)  # name -> column names
    calibration: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    @property
    def n_cells(self) -> int:
        return self.Z_corr.shape[0]

    @property
    def n_dims(self) -> int:
        return self.Z_corr.shape[1]

    def hierarchy(self) -> dict:
        """``{fine_key: {child_label: parent_label}}`` for each non-root level."""
        out = {}
        for parent, child in zip(self.label_keys[:-1], self.label_keys[1:]):
            m = self.obs[[parent, child]].astype(str).drop_duplicates()
            out[child] = dict(zip(m[child], m[parent]))
        return out

    def labels(self, level: str | None = None) -> np.ndarray:
        level = level or self.label_keys[-1]
        return self.obs[level].astype(str).to_numpy()

    # -------------------------------------------------------------- transform
    def project(self, X_counts, genes) -> tuple[np.ndarray, float]:
        """Apply the reference transformation (normalise -> reference genes ->
        reference scaling -> reference loadings) to raw query counts."""
        X_log = normalize_total_log1p(X_counts, self.params.get("target_sum", 1e4))
        X_sub, frac_missing = subset_genes(X_log, genes, self.genes)
        S = scale_apply(X_sub, self.mean, self.sd, self.params.get("clip", 10.0))
        return S @ self.loadings, frac_missing

    def reconstruct(self, Z: np.ndarray) -> np.ndarray:
        """Map latent coordinates back to (approximate) log-normalised expression of
        the reference genes (used by latent-arithmetic perturbation prediction)."""
        S = Z @ self.loadings.T
        return S * self.sd + self.mean

    # ------------------------------------------------------------- persistence
    def content_hash(self) -> str:
        h = hashlib.sha256()
        for a in (self.genes.astype(str), self.loadings, self.mean, self.sd, self.Z_corr):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(self.obs[self.label_keys].astype(str).to_csv().encode())
        return h.hexdigest()

    def save(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        arrays = dict(genes=self.genes.astype(str), mean=self.mean, sd=self.sd,
                      loadings=self.loadings, Z=self.Z, Z_corr=self.Z_corr,
                      Y=self.comp.Y, Nr=self.comp.Nr, C=self.comp.C, mu=self.comp.mu,
                      cov_inv=self.comp.cov_inv)
        for k, v in self.continuous.items():
            arrays[f"cont__{k}"] = v
        for k, v in self.calibration.get("null", {}).items():
            arrays[f"null__{k}"] = v
        np.savez_compressed(os.path.join(path, "arrays.npz"), **arrays)
        self.obs.to_csv(os.path.join(path, "obs.csv.gz"))
        cal = {k: v for k, v in self.calibration.items() if k != "null"}
        manifest = dict(
            format="refmap-reference", format_version=1, refmap_version=__version__,
            name=self.name, version=self.version, label_keys=self.label_keys,
            batch_key=self.batch_key, sample_key=self.sample_key,
            n_cells=self.n_cells, n_genes=int(len(self.genes)), n_dims=self.n_dims,
            sigma=self.comp.sigma, lamb=self.comp.lamb,
            continuous_names=self.continuous_names, calibration=cal,
            params=self.params, provenance=self.provenance,
            content_sha256=self.content_hash(),
            hierarchy=self.hierarchy(),
            label_counts={k: self.obs[k].astype(str).value_counts().to_dict()
                          for k in self.label_keys},
        )
        with open(os.path.join(path, "manifest.json"), "w") as fh:
            json.dump(_jsonable(manifest), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "Reference":
        with open(os.path.join(path, "manifest.json")) as fh:
            m = json.load(fh)
        if m.get("format") != "refmap-reference":
            raise ValueError(f"{path} is not a refmap reference")
        A = np.load(os.path.join(path, "arrays.npz"), allow_pickle=False)
        obs = pd.read_csv(os.path.join(path, "obs.csv.gz"), index_col=0)
        for k in m["label_keys"] + [m.get("batch_key"), m.get("sample_key")]:
            if k and k in obs:
                obs[k] = obs[k].astype(str)
        comp = SymphonyCompression(Y=A["Y"], Nr=A["Nr"], C=A["C"], mu=A["mu"],
                                   cov_inv=A["cov_inv"], sigma=m["sigma"], lamb=m["lamb"])
        cont = {k[6:]: A[k] for k in A.files if k.startswith("cont__")}
        cal = dict(m.get("calibration", {}))
        cal["null"] = {k[6:]: A[k] for k in A.files if k.startswith("null__")}
        ref = cls(name=m["name"], version=m["version"], genes=A["genes"], mean=A["mean"],
                  sd=A["sd"], loadings=A["loadings"], Z=A["Z"], Z_corr=A["Z_corr"], obs=obs,
                  label_keys=m["label_keys"], batch_key=m.get("batch_key"),
                  sample_key=m.get("sample_key"), comp=comp, continuous=cont,
                  continuous_names=m.get("continuous_names", {}), calibration=cal,
                  params=m.get("params", {}), provenance=m.get("provenance", {}))
        if ref.content_hash() != m.get("content_sha256"):
            warnings.warn(f"content hash mismatch for reference at {path}", stacklevel=2)
        return ref


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ---------------------------------------------------------------- building


def _pca(S: np.ndarray, n_pcs: int, seed: int) -> np.ndarray:
    _, _, Vt = randomized_svd(S, n_components=n_pcs, random_state=seed)
    V = Vt.T
    # deterministic sign: largest-magnitude loading positive
    sgn = np.sign(V[np.abs(V).argmax(axis=0), np.arange(V.shape[1])])
    return V * sgn


def _spca(S: np.ndarray, labels: np.ndarray, n_pcs: int, k: int, seed: int) -> np.ndarray:
    """Supervised PCA: top eigenvectors of S^T L S, where L is a kNN graph (built in
    unsupervised PCA space) keeping only edges between cells with the same label."""
    V0 = _pca(S, min(50, min(S.shape) - 1), seed)
    _, idx = knn(S @ V0, k=k, exclude_self=True)
    n = S.shape[0]
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    keep = labels[rows] == labels[cols]
    L = sp.csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(n, n))
    L = (L + L.T) * 0.5 + sp.identity(n)
    Sc = S - S.mean(axis=0)
    M = Sc.T @ (L @ Sc)
    evals, evecs = np.linalg.eigh((M + M.T) / 2)
    V = evecs[:, ::-1][:, :n_pcs]
    sgn = np.sign(V[np.abs(V).argmax(axis=0), np.arange(V.shape[1])])
    return V * sgn


def build_reference(adata, label_keys, *, batch_key: str | None = None,
                    sample_key: str | None = None, name: str = "reference",
                    version: str = "1.0.0", n_hvg: int = 2000, n_pcs: int = 30,
                    transform: str = "pca", layer: str | None = None,
                    continuous_obsm: tuple = (), harmony_kwargs: dict | None = None,
                    sigma: float = 0.1, k_calibration: int = 30, calibrate: bool = True,
                    hvg_genes=None, seed: int = 0, provenance: dict | None = None
                    ) -> Reference:
    """Assemble a reference from annotated raw counts.

    ``label_keys`` lists annotation columns from coarse to fine (a hierarchy); every
    fine label must have a single parent. ``continuous_obsm`` names ``obsm`` entries
    (e.g. spatial coordinates, protein, chromatin features) to transfer to queries.
    """
    if isinstance(label_keys, str):
        label_keys = [label_keys]
    label_keys = list(label_keys)
    for parent, child in zip(label_keys[:-1], label_keys[1:]):
        nparent = adata.obs.groupby(child, observed=True)[parent].nunique()
        if (nparent > 1).any():
            raise ValueError(f"labels in '{child}' have more than one '{parent}' parent: "
                             f"{list(nparent[nparent > 1].index)}")
    X = adata.layers[layer] if layer else adata.X
    genes_all = np.asarray(adata.var_names)
    X_log = normalize_total_log1p(X)
    batch = adata.obs[batch_key].astype(str).to_numpy() if batch_key else None
    genes = np.asarray(hvg_genes) if hvg_genes is not None else select_hvgs(
        X_log, genes_all, n_top=n_hvg, batch=batch)
    X_sub, _ = subset_genes(X_log, genes_all, genes)
    mean, sd = scale_fit(X_sub)
    S = scale_apply(X_sub, mean, sd)
    n_pcs = min(n_pcs, min(S.shape) - 1)
    if transform == "pca":
        loadings = _pca(S, n_pcs, seed)
    elif transform == "spca":
        loadings = _spca(S, adata.obs[label_keys[-1]].astype(str).to_numpy(), n_pcs,
                         k=20, seed=seed)
    else:
        raise ValueError(f"unknown transform {transform!r}")
    Z = S @ loadings
    hk = dict(sigma=sigma, seed=seed)
    hk.update(harmony_kwargs or {})
    hres = run_harmony(Z, batch, **hk)
    comp, R = compress_reference(hres.Z_corr, hres.Y, hk["sigma"], hk.get("lamb", 1.0))

    obs_cols = list(dict.fromkeys(label_keys + [c for c in (batch_key, sample_key) if c]))
    extra_cols = [c for c in adata.obs.columns if c not in obs_cols]
    obs = adata.obs[obs_cols + extra_cols].copy()
    for c in obs_cols:
        obs[c] = obs[c].astype(str)
    continuous, cnames = {}, {}
    for key in continuous_obsm:
        v = adata.obsm[key]
        if isinstance(v, pd.DataFrame):
            cnames[key] = [str(c) for c in v.columns]
            v = v.to_numpy()
        else:
            v = np.asarray(v.toarray() if sp.issparse(v) else v)
            cnames[key] = [f"{key}_{i}" for i in range(v.shape[1])]
        continuous[key] = v.astype(float)

    prov = dict(created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                parent_version=None, changelog=[f"{version}: initial release"],
                datasets=sorted(obs[batch_key].unique().tolist()) if batch_key else [])
    prov.update(provenance or {})
    ref = Reference(
        name=name, version=version, genes=genes, mean=mean, sd=sd, loadings=loadings,
        Z=Z, Z_corr=hres.Z_corr, obs=obs, label_keys=label_keys, batch_key=batch_key,
        sample_key=sample_key, comp=comp, continuous=continuous, continuous_names=cnames,
        params=dict(transform=transform, n_hvg=int(len(genes)), n_pcs=int(n_pcs),
                    target_sum=1e4, clip=10.0, harmony=_jsonable(hk),
                    harmony_converged=bool(hres.converged), n_centroids=int(len(comp.Nr)),
                    k_calibration=k_calibration),
        provenance=prov)
    if calibrate:
        ref.calibration = calibrate_reference(ref, R, k=k_calibration, seed=seed)
    return ref


def calibrate_reference(ref: Reference, R: np.ndarray | None = None, k: int = 30,
                        n_folds: int = 5, seed: int = 0, max_null: int = 20000,
                        max_heldout_batches: int = 6, min_group: int = 20) -> dict:
    """Self-mapping cross-validation of the reference.

    When the reference has >= 2 batches, each batch in turn is held out, the reference
    is re-integrated (Harmony) without it, and the held-out batch is mapped back
    *exactly as a query would be* (projection -> Symphony correction -> kNN). This
    leave-one-batch-out scheme mimics a dataset from a new lab whose cell states are
    all represented in the reference. With a single batch, random folds are used.

    Returns (i) label-transfer accuracy per annotation level -- how well the reference
    conserves its own heterogeneity -- and (ii) null distributions of the novelty
    statistics (per-cell kNN distance, Symphony Mahalanobis distance and label
    uncertainty; per-group Symphony cluster Mahalanobis) for states that *are*
    represented. Query cells and clusters are scored against these nulls.
    """
    rng = np.random.default_rng(seed)
    n = ref.n_cells
    batches = ref.obs[ref.batch_key].to_numpy() if ref.batch_key else None
    ub = list(pd.unique(batches)) if batches is not None else []
    if len(ub) >= 2:
        scheme = "leave-one-batch-out"
        if len(ub) > max_heldout_batches:
            ub = list(rng.choice(ub, max_heldout_batches, replace=False))
        folds = [batches == b for b in ub]
    else:
        scheme = "random-folds"
        fold = rng.integers(0, n_folds, n)
        folds = [fold == f for f in range(n_folds)]
    hk = dict(ref.params.get("harmony", {}))
    lab = ref.labels()
    tested = np.zeros(n, bool)
    knn_d, unc, maha = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    acc = {key: np.zeros(n, bool) for key in ref.label_keys}
    group_maha = []
    for te in folds:
        tr = ~te
        if scheme == "leave-one-batch-out":
            h = run_harmony(ref.Z[tr], batches[tr], **hk)
            comp, _ = compress_reference(h.Z_corr, h.Y, ref.comp.sigma, ref.comp.lamb)
            Zte, Rte = symphony_map(ref.Z[te], comp, None)
            Ztr = h.Z_corr
        else:
            comp, _ = compress_reference(ref.Z_corr[tr], ref.comp.Y, ref.comp.sigma,
                                         ref.comp.lamb)
            Zte = ref.Z_corr[te]
            Rte = soft_assign(l2_normalize(Zte), comp.Y, comp.sigma)
            Ztr = ref.Z_corr[tr]
        dist, idx = knn(Ztr, Zte, k=k)
        w = knn_weights(dist)
        res = transfer_hierarchical(ref.obs.iloc[np.flatnonzero(tr)].reset_index(drop=True),
                                    ref.label_keys, idx, w, unknown_threshold=1.0)
        tested |= te
        knn_d[te] = dist.mean(axis=1)
        unc[te] = res[f"{ref.label_keys[-1]}_uncertainty"].to_numpy()
        maha[te] = mahalanobis_to_centroids(Zte, Rte, comp)
        for key in ref.label_keys:
            acc[key][te] = res[f"{key}_pred"].to_numpy() == ref.obs[key].to_numpy()[te]
        for l in np.unique(lab[te]):
            m = lab[te] == l
            if m.sum() >= min_group:
                group_maha.append(cluster_mahalanobis(Zte[m].mean(axis=0), comp))
    knn_d, unc, maha = knn_d[tested], unc[tested], maha[tested]
    sub = rng.choice(len(knn_d), min(len(knn_d), max_null), replace=False)
    per_label = (pd.DataFrame({"label": lab[tested], "ok": acc[ref.label_keys[-1]][tested]})
                 .groupby("label")["ok"].mean().round(4).to_dict())
    group_maha = np.sort(np.asarray(group_maha, float))
    return dict(
        k=k, scheme=scheme, n_heldout_groups=len(folds), n_cells_tested=int(tested.sum()),
        self_mapping_accuracy={key: float(v[tested].mean()) for key, v in acc.items()},
        self_mapping_accuracy_per_label=per_label,
        quantiles={name: {str(q): float(np.quantile(v, q)) for q in (0.5, 0.9, 0.95, 0.99)}
                   for name, v in (("knn_distance", knn_d), ("mahalanobis", maha),
                                   ("uncertainty", unc), ("cluster_mahalanobis", group_maha))
                   if len(v)},
        null=dict(knn_distance=np.sort(knn_d[sub]), mahalanobis=np.sort(maha[sub]),
                  uncertainty=np.sort(unc[sub]), cluster_mahalanobis=group_maha),
    )


def cluster_mahalanobis(mu: np.ndarray, comp: SymphonyCompression) -> float:
    """Symphony per-cluster metric: Mahalanobis distance from a group mean to the
    closest reference centroid (under that centroid's covariance)."""
    D = mu[None, :] - comp.mu
    return float(np.sqrt(np.maximum(np.einsum("ki,kij,kj->k", D, comp.cov_inv, D), 0)).min())
