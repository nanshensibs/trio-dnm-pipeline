"""Query-to-reference mapping (Figure 1B): project, correct, transfer, contextualise.

1. *Project* the query with the reference's own transformation (``Reference.project``).
2. *Correct* query batch effects against the frozen reference (Symphony MoE).
3. *Find neighbours* among reference cells in the integrated space.
4. *Transfer* hierarchical labels (with uncertainty) and continuous reference
   information (e.g. spatial location, protein, chromatin accessibility).
5. *Score* every cell for out-of-distribution states and summarise query clusters.

No re-clustering or manual re-annotation of the query is needed: the query is
interpreted directly in the reference's coordinate system.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .graph import knn
from .harmony import symphony_map
from .novelty import cluster_query, score_cells, summarize_clusters
from .reference import Reference
from .transfer import UNKNOWN, knn_weights, transfer_continuous, transfer_hierarchical


@dataclass
class MappingResult:
    reference: str
    reference_version: str
    Zq: np.ndarray                   # projected, uncorrected
    Zq_corr: np.ndarray              # projected and batch-corrected (joint space)
    Rq: np.ndarray                   # soft assignment to reference centroids
    knn_idx: np.ndarray
    knn_dist: np.ndarray
    labels: pd.DataFrame             # per level: pred, uncertainty, resolved
    scores: pd.DataFrame             # novelty statistics per cell
    clusters: np.ndarray             # query clusters in joint space
    cluster_summary: pd.DataFrame
    continuous: dict = field(default_factory=dict)      # name -> DataFrame(mean)
    continuous_sd: dict = field(default_factory=dict)   # name -> DataFrame(sd)
    frac_missing_genes: float = 0.0
    obs_names: np.ndarray | None = None

    def obs_table(self) -> pd.DataFrame:
        df = pd.concat([self.labels, self.scores.add_prefix("map_")], axis=1)
        df["query_cluster"] = self.clusters
        df["query_cluster_novel"] = df["query_cluster"].map(
            self.cluster_summary.set_index("cluster")["novel"])
        if self.obs_names is not None:
            df.index = self.obs_names
        return df

    def annotate(self, adata, prefix: str = "refmap") -> None:
        """Write results into an AnnData (obs columns and ``obsm`` embeddings)."""
        t = self.obs_table()
        for c in t.columns:
            adata.obs[f"{prefix}_{c}"] = t[c].to_numpy()
        adata.obsm[f"X_{prefix}"] = self.Zq_corr
        for k, v in self.continuous.items():
            adata.obsm[f"{prefix}_{k}"] = v.to_numpy()
        adata.uns[prefix] = dict(reference=self.reference, version=self.reference_version,
                                 frac_missing_genes=self.frac_missing_genes)


def map_query(ref: Reference, query, *, batch_key: str | None = None, layer: str | None = None,
              correct: bool = True, k: int | None = None, unknown_threshold: float = 0.5,
              alpha: float = 0.01, condition_key: str | None = None,
              cluster_resolution: float = 1.0, cluster_k: int = 15, seed: int = 0
              ) -> MappingResult:
    k = k or ref.calibration.get("k", 30)
    X = query.layers[layer] if layer else query.X
    Zq, frac_missing = ref.project(X, query.var_names)
    qbatch = query.obs[batch_key].astype(str).to_numpy() if batch_key else None
    Zq_corr, Rq = symphony_map(Zq, ref.comp, qbatch, correct=correct)
    dist, idx = knn(ref.Z_corr, Zq_corr, k=k)
    w = knn_weights(dist)
    labels = transfer_hierarchical(ref.obs, ref.label_keys, idx, w, unknown_threshold)
    cont, cont_sd = {}, {}
    for name, vals in ref.continuous.items():
        mu, sd = transfer_continuous(vals, idx, w)
        cols = ref.continuous_names.get(name) or [f"{name}_{i}" for i in range(mu.shape[1])]
        cont[name] = pd.DataFrame(mu, columns=cols, index=query.obs_names)
        cont_sd[name] = pd.DataFrame(sd, columns=cols, index=query.obs_names)
    fine = ref.label_keys[-1]
    scores = score_cells(ref, Zq_corr, Rq, dist, labels[f"{fine}_uncertainty"].to_numpy(),
                         alpha=alpha, unknown_threshold=unknown_threshold)
    clusters = cluster_query(Zq_corr, k=cluster_k, resolution=cluster_resolution, seed=seed)
    cond = query.obs[condition_key].astype(str).to_numpy() if condition_key else None
    summary = summarize_clusters(clusters, scores, Zq_corr, ref,
                                 labels[f"{fine}_pred"].to_numpy(), cond)
    return MappingResult(reference=ref.name, reference_version=ref.version, Zq=Zq,
                         Zq_corr=Zq_corr, Rq=Rq, knn_idx=idx, knn_dist=dist, labels=labels,
                         scores=scores, clusters=clusters, cluster_summary=summary,
                         continuous=cont, continuous_sd=cont_sd,
                         frac_missing_genes=frac_missing,
                         obs_names=np.asarray(query.obs_names))


def final_labels(result: MappingResult, level: str, novel_label: str = "Novel") -> np.ndarray:
    """Transferred labels with cells in novel query clusters relabelled ``novel_label``
    (they are reported as candidate new states rather than forced onto the reference)."""
    lab = result.labels[f"{level}_pred"].to_numpy().astype(object).copy()
    novel_clusters = set(result.cluster_summary.loc[result.cluster_summary["novel"], "cluster"])
    lab[np.isin(result.clusters, list(novel_clusters))] = novel_label
    return lab


__all__ = ["map_query", "MappingResult", "final_labels", "UNKNOWN"]
