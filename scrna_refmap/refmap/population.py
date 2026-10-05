"""Population-scale reference mapping: sample-level representations (Figure 2).

Implements the perspective's section "Population-scale reference mapping" and
Figure 2: once many samples (donors, patients, tumours) are mapped into a shared
reference space, each *sample* -- not each cell -- becomes the unit of analysis:

* ``sample_representation`` -- one vector per sample from its mapped cells
  (Fig. 2B). Stand-in for sample-aware integration such as MrVI: a
  *pseudobulk-in-latent-space* embedding = cell-type composition concatenated with
  the per-cell-type mean latent position.
* ``sample_similarity`` -- query x reference sample similarity with hierarchical
  clustering order (Fig. 2E, "which reference patients does this patient resemble?").
* ``classify_samples`` -- supervised sample-level phenotype prediction trained on
  annotated reference samples and applied to query samples (Fig. 2D).
* ``mil_attribution`` -- a multi-instance learning baseline (Box 1): sample labels
  supervise cell-level scores that are aggregated per sample and summarised per cell
  population, pointing to the "cell populations responsible for disease".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (LeaveOneOut, StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------- sample embedding


@dataclass
class SampleRepresentation:
    table: pd.DataFrame          # samples x features
    features: pd.DataFrame       # per feature: block ("composition"/"latent"), cell_type, dim
    n_cells: pd.DataFrame        # samples x cell types (cell counts)
    cell_types: list
    type_means: pd.DataFrame     # cell type x dim: across-sample mean (imputation values)
    imputed: pd.DataFrame        # samples x cell types: True where the mean was imputed

    def block(self, name: str) -> pd.DataFrame:
        return self.table.loc[:, (self.features["block"] == name).to_numpy()]


def sample_representation(Z: np.ndarray, labels, samples, min_cells: int = 5,
                          cell_types=None, impute_from: SampleRepresentation | None = None
                          ) -> SampleRepresentation:
    """Pseudobulk-in-latent-space sample embedding (stand-in for MrVI, Fig. 2B).

    Per sample: (i) the proportion of its cells in each cell type, and (ii) for each
    cell type, the mean latent position (``Z``, e.g. ``MappingResult.Zq_corr`` or
    ``Reference.Z_corr``) of its cells of that type. Mean positions estimated from
    fewer than ``min_cells`` cells are imputed by the across-sample mean for that
    type (taken from ``impute_from`` -- typically the reference cohort -- when given,
    so query samples are imputed on the reference scale). ``cell_types`` fixes the
    feature order (defaults to ``impute_from.cell_types`` or all labels); cells with
    other labels (e.g. ``Unknown``) still count towards each sample's total.
    Unlike MrVI, this does not learn sample effects jointly with the cell embedding.
    """
    Z = np.asarray(Z, float)
    labels = np.asarray(labels).astype(str)
    samples = np.asarray(samples).astype(str)
    if cell_types is None:
        cell_types = impute_from.cell_types if impute_from is not None else np.unique(labels)
    cell_types = [str(c) for c in cell_types]
    sams = np.unique(samples)
    d = Z.shape[1]
    counts = pd.crosstab(samples, labels).reindex(index=sams, columns=cell_types).fillna(0)
    total = pd.Series(samples).value_counts().reindex(sams).to_numpy(float)
    comp = counts.div(total, axis=0)
    # per (sample, type) latent sums -> means
    df = pd.DataFrame(Z, columns=[f"z{i}" for i in range(d)])
    df["s"], df["t"] = samples, labels
    df = df[df["t"].isin(cell_types)]
    means = df.groupby(["s", "t"]).mean()
    ok = counts.stack()
    ok = ok[ok >= min_cells].index
    means = means.loc[means.index.intersection(ok)]
    if impute_from is not None:
        type_means = impute_from.type_means.reindex(cell_types)
    else:
        type_means = means.groupby(level="t").mean().reindex(cell_types)
    type_means = type_means.fillna(pd.Series(Z.mean(axis=0), index=type_means.columns))
    blocks, imputed = [], pd.DataFrame(False, index=sams, columns=cell_types)
    for ct in cell_types:
        m = means.xs(ct, level="t") if ct in means.index.get_level_values("t") else None
        M = pd.DataFrame(index=sams, columns=type_means.columns, dtype=float)
        if m is not None:
            M.loc[m.index] = m.to_numpy()
        miss = M.isna().any(axis=1)
        imputed[ct] = miss.to_numpy()
        M.loc[miss] = type_means.loc[ct].to_numpy()
        M.columns = [f"{ct}|{c}" for c in M.columns]
        blocks.append(M)
    comp.columns = [f"prop|{c}" for c in cell_types]
    table = pd.concat([comp] + blocks, axis=1).astype(float)
    table.index.name = "sample"
    feats = pd.DataFrame({
        "block": ["composition"] * len(cell_types) + ["latent"] * (len(cell_types) * d),
        "cell_type": cell_types + [ct for ct in cell_types for _ in range(d)],
        "dim": [-1] * len(cell_types) + list(range(d)) * len(cell_types)},
        index=table.columns)
    return SampleRepresentation(table=table, features=feats, n_cells=counts.astype(int),
                                cell_types=cell_types, type_means=type_means,
                                imputed=imputed)


def _as_table(rep) -> pd.DataFrame:
    return rep.table if isinstance(rep, SampleRepresentation) else pd.DataFrame(rep)


def _features(rep, columns) -> pd.DataFrame | None:
    if isinstance(rep, SampleRepresentation):
        return rep.features.reindex(columns)
    return None


def _align(rep_query, rep_ref) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query features reindexed to the reference's; missing features take the
    reference mean (i.e. contribute nothing after standardisation)."""
    R, Q = _as_table(rep_ref), _as_table(rep_query)
    Q = Q.reindex(columns=R.columns)
    return Q.fillna(R.mean(axis=0)), R


# --------------------------------------------------------------- similarity


@dataclass
class SimilarityResult:
    similarity: pd.DataFrame      # query samples x reference samples
    query_order: list             # hierarchical-clustering leaf order (rows)
    ref_order: list               # hierarchical-clustering leaf order (columns)
    nearest: pd.DataFrame         # per query sample: most similar reference sample

    def ordered(self) -> pd.DataFrame:
        return self.similarity.loc[self.query_order, self.ref_order]


def sample_similarity(rep_query, rep_ref, metric: str = "correlation",
                      standardize: bool = True, equal_blocks: bool = True,
                      method: str = "average") -> SimilarityResult:
    """Query x reference sample similarity (Fig. 2E).

    Features are z-scored with the *reference* samples' mean/sd (so the per-type
    identity common to all samples cancels and only between-sample variation
    remains); with ``equal_blocks`` each block (composition, latent) is scaled by
    1/sqrt(n_features) so the many latent features do not swamp composition.
    ``metric``: ``correlation`` / ``cosine`` (similarity = 1 - distance) or
    ``euclidean`` (similarity = -distance). Rows and columns are ordered by
    ``scipy`` hierarchical clustering (``method`` linkage) of the similarity profiles.
    """
    Q, R = _align(rep_query, rep_ref)
    if standardize:
        mu, sd = R.mean(axis=0), R.std(axis=0, ddof=1).replace(0, 1.0).fillna(1.0)
        Q, R = (Q - mu) / sd, (R - mu) / sd
    feats = _features(rep_ref, R.columns)
    if equal_blocks and feats is not None:
        w = 1.0 / np.sqrt(feats.groupby("block")["block"].transform("size").to_numpy(float))
        Q, R = Q * w, R * w
    if metric not in ("correlation", "cosine", "euclidean"):
        raise ValueError(f"unknown metric {metric!r}")
    D = cdist(Q.to_numpy(), R.to_numpy(), metric=metric)
    S = -D if metric == "euclidean" else 1.0 - D
    sim = pd.DataFrame(S, index=Q.index, columns=R.index)

    def order(M: np.ndarray, names) -> list:
        if M.shape[0] < 3:
            return list(names)
        return list(np.asarray(names)[leaves_list(linkage(M, method=method))])

    nearest = pd.DataFrame({"nearest_reference": sim.idxmax(axis=1),
                            "similarity": sim.max(axis=1)})
    return SimilarityResult(similarity=sim, query_order=order(S, sim.index),
                            ref_order=order(S.T, sim.columns), nearest=nearest)


# ----------------------------------------------------------- classification


@dataclass
class ClassificationResult:
    cv_accuracy: float
    cv_auc: float                  # binary: ROC AUC; multiclass: one-vs-rest macro AUC
    cv_predictions: pd.DataFrame   # reference samples: true, predicted, P(class)
    query_predictions: pd.DataFrame  # query samples: predicted, P(class)
    classes: list
    model: object                  # fitted on all reference samples
    feature_weights: pd.Series | None = None
    params: dict = field(default_factory=dict)


def _make_model(model: str, C: float, seed: int):
    if model == "logistic":
        clf = LogisticRegression(C=C, max_iter=5000)
    elif model in ("rf", "random_forest"):
        clf = RandomForestClassifier(n_estimators=300, random_state=seed)
    else:
        raise ValueError(f"unknown model {model!r}")
    return make_pipeline(StandardScaler(), clf)


def classify_samples(rep_ref, y_ref, rep_query=None, model: str = "logistic",
                     cv: str = "loo", n_splits: int = 5, C: float = 0.1,
                     seed: int = 0) -> ClassificationResult:
    """Supervised sample-level phenotype classification (Fig. 2D).

    A standardised ``logistic`` (L2, strength ``C``) or random-forest (``rf``)
    classifier is trained on reference sample representations with labels ``y_ref``
    (Series indexed by sample, or array in row order). Generalisation is estimated on
    the reference by leave-one-sample-out (``cv="loo"``) or stratified k-fold CV; the
    model refitted on all reference samples predicts class probabilities for the
    query samples (features aligned to the reference's).
    """
    R = _as_table(rep_ref)
    y = (pd.Series(y_ref).astype(str).reindex(R.index) if isinstance(y_ref, pd.Series)
         else pd.Series(np.asarray(y_ref).astype(str), index=R.index))
    if y.isna().any():
        raise ValueError("y_ref is missing labels for some reference samples")
    classes = sorted(y.unique())
    if len(classes) < 2:
        raise ValueError("need at least two phenotype classes")
    pipe = _make_model(model, C, seed)
    if cv == "loo":
        splitter = LeaveOneOut()
    else:
        k = min(n_splits, y.value_counts().min())
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    P = cross_val_predict(pipe, R.to_numpy(), y.to_numpy(), cv=splitter,
                          method="predict_proba")
    pred = np.asarray(classes)[P.argmax(axis=1)]
    acc = accuracy_score(y, pred)
    if len(classes) == 2:
        auc = roc_auc_score((y == classes[1]).astype(int), P[:, 1])
    else:
        auc = roc_auc_score(y, P, multi_class="ovr", labels=classes)
    cvp = pd.DataFrame(P, index=R.index, columns=[f"P({c})" for c in classes])
    cvp.insert(0, "predicted", pred)
    cvp.insert(0, "true", y.to_numpy())
    pipe.fit(R.to_numpy(), y.to_numpy())
    qp = pd.DataFrame(columns=["predicted"] + list(cvp.columns[2:]))
    if rep_query is not None:
        Q, _ = _align(rep_query, rep_ref)
        PQ = pipe.predict_proba(Q.to_numpy())
        qp = pd.DataFrame(PQ, index=Q.index, columns=[f"P({c})" for c in classes])
        qp.insert(0, "predicted", np.asarray(classes)[PQ.argmax(axis=1)])
    fw = None
    if model == "logistic":
        coef = pipe[-1].coef_
        fw = pd.Series(coef[0] if coef.shape[0] == 1 else np.abs(coef).max(axis=0),
                       index=R.columns).sort_values(key=np.abs, ascending=False)
    return ClassificationResult(cv_accuracy=float(acc), cv_auc=float(auc),
                                cv_predictions=cvp, query_predictions=qp, classes=classes,
                                model=pipe, feature_weights=fw,
                                params=dict(model=model, cv=cv, C=C, seed=seed))


# ------------------------------------------------------- multi-instance learning


@dataclass
class MILResult:
    cell_scores: np.ndarray        # out-of-sample instance score P(positive bag | cell)
    bag_scores: pd.DataFrame       # per sample: label, score, aggregated by mean/quantile
    bag_auc: float
    bag_accuracy: float
    attribution: pd.DataFrame      # per cell type, sorted by contribution
    positive: str
    params: dict = field(default_factory=dict)


def mil_attribution(Z: np.ndarray, samples, sample_labels, labels, positive: str,
                    seed: int = 0, n_splits: int = 5, aggregate: str = "mean",
                    quantile: float = 0.9, C: float = 0.1,
                    max_cells_per_sample: int | None = 2000) -> MILResult:
    """Multi-instance learning baseline (Box 1): which cell populations drive a
    sample-level phenotype?

    Each sample is a *bag* of cells (instances) carrying the sample's label
    (``sample_labels``: mapping/Series sample -> label, or a per-cell array). An
    instance-level standardised logistic classifier on the joint embedding ``Z`` is
    trained with the bag label propagated to every cell, using sample-grouped
    cross-fitting (stratified ``GroupKFold`` over samples): each cell is scored by a
    model that never saw its sample, so scores reflect what generalises across
    donors. Instance scores are aggregated per bag (``mean`` or the mean of the top
    ``1 - quantile`` fraction) for sample classification (AUC; accuracy at 0.5 for
    ``mean``, at the median bag score otherwise).

    Per cell type (``labels``) the summary reports, over bags:
    ``score_diff`` -- mean instance score in positive minus negative bags (a
    cell-state signal); ``delta_proportion`` -- abundance change; and
    ``contribution`` -- the type's share of the difference in mean bag score,
    E_pos[p_t * c_t] - E_neg[p_t * c_t] with c_t the type's mean centred score
    (summing over types gives the bag-score difference, so both state and abundance
    effects count). ``p`` is a Welch t-test of per-bag mean scores. This is a simple
    instance-classifier MIL baseline, *not* attention-based MIL (e.g. ABMIL /
    ProtoCell4P): attribution comes from out-of-sample instance scores, not learned
    attention weights.
    """
    Z = np.asarray(Z, float)
    samples = np.asarray(samples).astype(str)
    labels = np.asarray(labels).astype(str)
    if isinstance(sample_labels, (pd.Series, dict)):
        bag_label = pd.Series(sample_labels).astype(str)
    else:
        sl = np.asarray(sample_labels).astype(str)
        bag_label = pd.Series(sl, index=samples).groupby(level=0).first()
    bags = np.unique(samples)
    bag_label = bag_label.reindex(bags)
    if bag_label.isna().any():
        raise ValueError("sample_labels is missing some samples")
    y_bag = (bag_label == positive).astype(int)
    if y_bag.nunique() < 2:
        raise ValueError(f"need positive ({positive!r}) and negative bags")
    rng = np.random.default_rng(seed)
    # optional per-bag subsampling for training (all cells are still scored)
    train_mask = np.ones(len(samples), bool)
    if max_cells_per_sample:
        for b in bags:
            idx = np.flatnonzero(samples == b)
            if len(idx) > max_cells_per_sample:
                drop = rng.choice(idx, len(idx) - max_cells_per_sample, replace=False)
                train_mask[drop] = False
    y_cell = y_bag.reindex(samples).to_numpy()
    k = int(min(n_splits, y_bag.value_counts().min()))
    if k < 2:
        raise ValueError("need >= 2 bags per class for grouped cross-fitting")
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = np.full(len(samples), np.nan)
    for tr, te in splitter.split(Z, y_cell, groups=samples):
        tr = tr[train_mask[tr]]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        clf.fit(Z[tr], y_cell[tr])
        scores[te] = clf.predict_proba(Z[te])[:, 1]

    df = pd.DataFrame({"s": samples, "t": labels, "score": scores})
    if aggregate == "mean":
        agg = df.groupby("s")["score"].mean()
        thr = 0.5
    elif aggregate == "quantile":
        agg = df.groupby("s")["score"].apply(
            lambda v: v[v >= np.quantile(v, quantile)].mean())
        thr = float(agg.median())
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}")
    agg = agg.reindex(bags)
    bag_scores = pd.DataFrame({"label": bag_label, "positive": y_bag.astype(bool),
                               "score": agg})
    bag_auc = roc_auc_score(y_bag, agg)
    bag_acc = accuracy_score(y_bag, (agg > thr).astype(int))

    # per-type attribution over bags
    centre = float(np.mean(scores))
    n_bt = pd.crosstab(df["s"], df["t"]).reindex(bags).fillna(0)
    prop = n_bt.div(n_bt.sum(axis=1), axis=0)
    mean_bt = df.groupby(["s", "t"])["score"].mean().unstack().reindex(index=bags,
                                                                      columns=n_bt.columns)
    pos = y_bag.to_numpy() == 1
    contrib_bt = prop * (mean_bt - centre).fillna(0.0)
    rows = []
    for t in n_bt.columns:
        mp, mn = mean_bt.loc[pos, t].dropna(), mean_bt.loc[~pos, t].dropna()
        p = (stats.ttest_ind(mp, mn, equal_var=False).pvalue
             if len(mp) >= 2 and len(mn) >= 2 else np.nan)
        rows.append(dict(
            cell_type=t, n_cells=int(n_bt[t].sum()),
            mean_score_pos=mp.mean(), mean_score_neg=mn.mean(),
            score_diff=mp.mean() - mn.mean(),
            delta_proportion=prop.loc[pos, t].mean() - prop.loc[~pos, t].mean(),
            contribution=contrib_bt.loc[pos, t].mean() - contrib_bt.loc[~pos, t].mean(),
            p=p))
    attribution = (pd.DataFrame(rows).sort_values("contribution", ascending=False)
                   .reset_index(drop=True))
    return MILResult(cell_scores=scores, bag_scores=bag_scores, bag_auc=float(bag_auc),
                     bag_accuracy=float(bag_acc), attribution=attribution,
                     positive=positive,
                     params=dict(seed=seed, n_splits=k, aggregate=aggregate,
                                 quantile=quantile, C=C,
                                 max_cells_per_sample=max_cells_per_sample))


__all__ = ["sample_representation", "SampleRepresentation", "sample_similarity",
           "SimilarityResult", "classify_samples", "ClassificationResult",
           "mil_attribution", "MILResult"]
