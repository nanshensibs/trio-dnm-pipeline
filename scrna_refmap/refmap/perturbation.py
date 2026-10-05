"""Perturbation atlases: latent response arithmetic and query signature matching.

Implements the perspective's section "Construction of cellular perturbation atlases":
a reference built from a perturbation screen (cell types x perturbations) is used to

1. *predict* responses of cell types that were never perturbed, with scGen-style
   latent-space vector arithmetic (Lotfollahi et al. 2019, ref. 52): a perturbation is
   a displacement ``delta`` in the reference latent space, estimated from the cell types
   in which it was observed and added to control cells of an unseen cell type; the
   shifted latent is decoded back to expression;
2. *interpret* new perturbation experiments by mapping them onto the atlas and ranking
   atlas perturbations by the similarity of their response signatures -- the
   CMap/LINCS idea of reading a query's transcriptional signature against a library
   of reference signatures.

Deviations from scGen: the latent space is the reference's *linear* transformation
(scaled HVGs -> PCA loadings) instead of a VAE, so decoding is the linear
``Reference.reconstruct`` and the arithmetic is exact in the projected space (no
nonlinear decoder can bend the response). scGen balances cell types by resampling
before averaging; here ``delta`` is the unweighted mean of per-cell-type deltas, which
is the same balancing without randomness. When predicting a cell type, that cell
type's own observed delta is always excluded (out-of-distribution prediction).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .mapping import MappingResult, map_query
from .preprocess import normalize_total_log1p, subset_genes, to_dense
from .reference import Reference
from .transfer import UNKNOWN


@dataclass
class PerturbationModel:
    """Per-(perturbation, cell type) latent centroids and the derived deltas."""
    control: str
    perturbations: list
    cell_types: list
    centroids: dict = field(default_factory=dict)   # (pert, ct) -> (d,) mean latent
    n_cells: dict = field(default_factory=dict)     # (pert, ct) -> int

    @classmethod
    def fit(cls, Z: np.ndarray, cell_types, perturbations, control: str = "control",
            min_cells: int = 5) -> "PerturbationModel":
        """Estimate latent centroids of every (perturbation, cell type) block with at
        least ``min_cells`` cells. ``Z`` is a reference latent (e.g. ``ref.Z`` or a
        projection with ``ref.project``)."""
        Z = np.asarray(Z, float)
        ct = np.asarray(cell_types).astype(str)
        pt = np.asarray(perturbations).astype(str)
        if control not in set(pt):
            raise ValueError(f"control condition {control!r} not found in perturbations")
        cent, n = {}, {}
        for (p, c), rows in pd.DataFrame({"p": pt, "c": ct}).groupby(["p", "c"]).groups.items():
            if len(rows) >= min_cells:
                cent[(p, c)] = Z[np.asarray(rows)].mean(axis=0)
                n[(p, c)] = len(rows)
        perts = sorted({p for p, _ in cent if p != control})
        return cls(control=control, perturbations=perts, cell_types=sorted({c for _, c in cent}),
                   centroids=cent, n_cells=n)

    # ---------------------------------------------------------------- deltas
    def type_delta(self, perturbation: str, cell_type: str) -> np.ndarray | None:
        a = self.centroids.get((perturbation, cell_type))
        b = self.centroids.get((self.control, cell_type))
        return None if a is None or b is None else a - b

    def delta(self, perturbation: str, exclude_cell_type: str | None = None) -> np.ndarray:
        """Balanced global response: mean of per-cell-type deltas, leaving out
        ``exclude_cell_type`` (the cell type being predicted)."""
        ds = [d for c in self.cell_types if c != exclude_cell_type
              for d in [self.type_delta(perturbation, c)] if d is not None]
        if not ds:
            raise ValueError(f"no cell type with both {perturbation!r} and control "
                             f"(excluding {exclude_cell_type!r})")
        return np.mean(ds, axis=0)

    def signatures(self, by_cell_type: bool = False):
        """Latent response signatures: DataFrame (perturbation x dims), or with
        ``by_cell_type`` a dict perturbation -> DataFrame (cell type x dims)."""
        if not by_cell_type:
            return pd.DataFrame({p: self.delta(p) for p in self.perturbations}).T
        out = {}
        for p in self.perturbations:
            rows = {c: d for c in self.cell_types
                    for d in [self.type_delta(p, c)] if d is not None}
            out[p] = pd.DataFrame(rows).T
        return out

    # ------------------------------------------------------------ prediction
    def predict(self, Z_control: np.ndarray, perturbation: str,
                cell_type: str | None = None) -> np.ndarray:
        """scGen arithmetic: ``z_pred = z_control + delta``; if ``cell_type`` is given
        its own observed response is excluded from ``delta``."""
        return np.asarray(Z_control, float) + self.delta(perturbation, cell_type)[None, :]

    def predict_expression(self, ref: Reference, Z_control: np.ndarray, perturbation: str,
                           cell_type: str | None = None, X_control_log=None,
                           decode: str = "reconstruct") -> np.ndarray:
        """Predicted log-normalised expression (cells x ``ref.genes``).

        ``decode='reconstruct'`` decodes the shifted latent (scGen); ``'residual'`` adds
        the decoded shift to the observed control expression ``X_control_log``
        (already subset to ``ref.genes``), keeping what the PCA does not capture."""
        Zp = self.predict(Z_control, perturbation, cell_type)
        if decode == "reconstruct":
            return ref.reconstruct(Zp)
        if decode == "residual":
            if X_control_log is None:
                raise ValueError("decode='residual' needs X_control_log")
            shift = latent_to_expression(ref, self.delta(perturbation, cell_type))
            return to_dense(X_control_log) + shift[None, :]
        raise ValueError(f"unknown decode {decode!r}")


def latent_to_expression(ref: Reference, delta: np.ndarray) -> np.ndarray:
    """Decode a latent *displacement* into a log-expression change of ``ref.genes``
    (the linear decoder has no intercept for differences)."""
    return (np.asarray(delta, float) @ ref.loadings.T) * ref.sd


# ---------------------------------------------------------------- evaluation


def _r2(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Pearson correlation of mean expression (scGen's reported metric)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def evaluate_holdout(ref: Reference, adata, perturbation: str, cell_type: str, *,
                     perturbation_key: str = "perturbation", cell_type_key: str = "cell_type_l2",
                     control: str = "control", response_genes=None, n_top_de: int = 50,
                     decode: str = "reconstruct", layer: str | None = None) -> dict:
    """Out-of-distribution test: hide the (``perturbation``, ``cell_type``) block,
    predict it from that cell type's control cells and the other cell types'
    responses, and compare predicted vs. observed mean expression.

    ``response_genes`` (names) defaults to the ``n_top_de`` genes with the largest
    observed |perturbed - control| mean difference (scGen's top-DEG evaluation).
    The baseline is "no change": the observed control mean. Returns squared Pearson
    R^2 and MSE for all reference genes and for the response genes, plus the cosine
    between predicted and observed expression changes on the response genes.
    For a strict test, ``ref`` itself should be built without the held-out block.
    """
    X = adata.layers[layer] if layer else adata.X
    Z, _ = ref.project(X, adata.var_names)
    X_sub, _ = subset_genes(normalize_total_log1p(X, ref.params.get("target_sum", 1e4)),
                            adata.var_names, ref.genes)
    pt = adata.obs[perturbation_key].astype(str).to_numpy()
    ct = adata.obs[cell_type_key].astype(str).to_numpy()
    held = (pt == perturbation) & (ct == cell_type)
    ctrl = (pt == control) & (ct == cell_type)
    if not held.any() or not ctrl.any():
        raise ValueError("held-out block or its control cells are missing")
    model = PerturbationModel.fit(Z[~held], ct[~held], pt[~held], control=control)
    X_ctrl = to_dense(X_sub[np.flatnonzero(ctrl)])
    pred = model.predict_expression(ref, Z[ctrl], perturbation, cell_type,
                                    X_control_log=X_ctrl, decode=decode).mean(axis=0)
    truth = to_dense(X_sub[np.flatnonzero(held)]).mean(axis=0)
    base = X_ctrl.mean(axis=0)
    if response_genes is None:
        resp = np.argsort(-np.abs(truth - base))[:n_top_de]
    else:
        resp = np.flatnonzero(np.isin(ref.genes.astype(str), np.asarray(response_genes, str)))
        if not len(resp):
            raise ValueError("none of the response genes are reference genes")

    def block(ix):
        return dict(r2_pred=_r2(pred[ix], truth[ix]), r2_baseline=_r2(base[ix], truth[ix]),
                    mse_pred=float(np.mean((pred[ix] - truth[ix]) ** 2)),
                    mse_baseline=float(np.mean((base[ix] - truth[ix]) ** 2)))

    allix = np.arange(len(ref.genes))
    return dict(perturbation=perturbation, cell_type=cell_type, decode=decode,
                n_heldout=int(held.sum()), n_control=int(ctrl.sum()),
                n_response_genes=int(len(resp)), all=block(allix), response=block(resp),
                delta_cosine=_cos(pred[resp] - base[resp], truth[resp] - base[resp]),
                trained_cell_types=[c for c in model.cell_types if c != cell_type
                                    and model.type_delta(perturbation, c) is not None])


# ------------------------------------------------------- signature matching


@dataclass
class QueryResponse:
    per_type: pd.DataFrame          # cell type x dims: treated - control latent means
    global_delta: np.ndarray        # balanced mean over cell types
    n_cells: pd.DataFrame           # cell type x {treated, control}
    mapping: MappingResult | None = None


def query_response(ref: Reference, query, *, condition_key: str, treated: str,
                   control: str, cell_type_key: str | None = None, level: str | None = None,
                   space: str = "projected", min_cells: int = 10,
                   mapping: MappingResult | None = None, **map_kwargs) -> QueryResponse:
    """Map a perturbation experiment onto the atlas reference (``map_query``) and
    compute its latent response signature per cell type (treated minus control).

    Cell types come from ``cell_type_key`` or, by default, from the labels transferred
    at ``level`` (``Unknown`` cells are dropped). ``space='projected'`` uses the
    reference projection before Symphony correction: treated and control cells share
    the query batch, so the batch shift cancels in the difference, whereas the
    cluster-specific MoE correction could absorb part of a response that moves cells
    between reference clusters. ``space='corrected'`` uses ``Zq_corr``.
    """
    mapping = mapping or map_query(ref, query, **map_kwargs)
    Zq = mapping.Zq if space == "projected" else mapping.Zq_corr
    if cell_type_key is not None:
        ct = query.obs[cell_type_key].astype(str).to_numpy()
    else:
        ct = mapping.labels[f"{level or ref.label_keys[-1]}_pred"].astype(str).to_numpy()
    cond = query.obs[condition_key].astype(str).to_numpy()
    rows, counts = {}, {}
    for c in pd.unique(ct):
        if c == UNKNOWN:
            continue
        t, k = (ct == c) & (cond == treated), (ct == c) & (cond == control)
        counts[c] = dict(treated=int(t.sum()), control=int(k.sum()))
        if t.sum() >= min_cells and k.sum() >= min_cells:
            rows[c] = Zq[t].mean(axis=0) - Zq[k].mean(axis=0)
    if not rows:
        raise ValueError("no cell type has enough treated and control cells")
    per_type = pd.DataFrame(rows).T.sort_index()
    return QueryResponse(per_type=per_type, global_delta=per_type.to_numpy().mean(axis=0),
                         n_cells=pd.DataFrame(counts).T.sort_index(), mapping=mapping)


def match_signatures(query_delta, atlas_deltas, top: int | None = None) -> pd.DataFrame:
    """Rank atlas perturbations by cosine similarity to a query response signature.

    ``query_delta``: a vector (global signature) or a DataFrame (cell type x dims,
    e.g. ``QueryResponse.per_type``). ``atlas_deltas``: a DataFrame (perturbation x
    dims, e.g. ``PerturbationModel.signatures()``), a dict perturbation -> vector, or a
    dict perturbation -> DataFrame (cell type x dims, ``signatures(by_cell_type=True)``).
    With per-cell-type signatures on both sides the score is the mean cosine over
    shared cell types (cell-type-matched comparison); otherwise global signatures are
    compared. Works equally for expression-space signatures (see
    ``latent_to_expression``).
    """
    if isinstance(atlas_deltas, pd.DataFrame):
        atlas = {str(i): atlas_deltas.loc[i].to_numpy(float) for i in atlas_deltas.index}
    else:
        atlas = {str(k): (v if isinstance(v, pd.DataFrame) else np.asarray(v, float))
                 for k, v in atlas_deltas.items()}
    q_types = query_delta if isinstance(query_delta, pd.DataFrame) else None
    q_glob = (q_types.to_numpy(float).mean(axis=0) if q_types is not None
              else np.asarray(query_delta, float).ravel())
    rows = []
    for p, sig in atlas.items():
        shared = []
        if q_types is not None and isinstance(sig, pd.DataFrame):
            shared = [c for c in q_types.index if c in sig.index]
        if shared:
            cos = np.mean([_cos(q_types.loc[c].to_numpy(float), sig.loc[c].to_numpy(float))
                           for c in shared])
        else:
            a_glob = sig.to_numpy(float).mean(axis=0) if isinstance(sig, pd.DataFrame) else sig
            cos = _cos(q_glob, a_glob)
        a_vec = sig.to_numpy(float).mean(axis=0) if isinstance(sig, pd.DataFrame) else sig
        rows.append(dict(perturbation=p, cosine=float(cos), n_shared_cell_types=len(shared),
                         magnitude_ratio=float(np.linalg.norm(q_glob) /
                                               max(np.linalg.norm(a_vec), 1e-12))))
    df = pd.DataFrame(rows).sort_values("cosine", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df.head(top) if top else df


__all__ = ["PerturbationModel", "latent_to_expression", "evaluate_holdout",
           "QueryResponse", "query_response", "match_signatures"]
