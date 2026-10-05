"""Differential analysis of mapped samples: composition, abundance, state, prioritisation.

Implements the perspective's paragraph in "Identification of disease states by
contextualizing disease within a healthy reference": *once multiple samples have been
mapped into a shared space, the corresponding cell states can be compared across
conditions ... to identify changes in cell-type abundance (refs. 33-35), cell-state
changes (refs. 36-37) and to prioritize cell types according to the magnitude of
responses to perturbations (ref. 38)*. Every test works on reference-mapped output
(transferred labels and/or the joint embedding ``MappingResult.Zq_corr``) and uses
*samples* (donors), not cells, as the unit of replication wherever the original method
does.

* ``composition_test`` -- discrete cell-type composition (MASC / scCODA family,
  refs. 33-34) as a sample-level quasi-binomial GLM per cell type.
* ``milo`` -- Milo differential abundance on kNN neighbourhoods of the joint
  embedding (Dann et al. 2022, ref. 35) with Milo's graph-weighted spatial FDR.
* ``pseudobulk_de`` -- muscat-style pseudobulk state-change DE per cell type
  (Crowell et al. 2020, ref. 37) with a limma-like moderated t.
* ``augur`` -- Augur cell-type prioritisation by condition separability
  (Skinnider et al. 2021, ref. 38).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from scipy import stats
from scipy.special import digamma, polygamma
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from .graph import knn
from .novelty import bh_fdr
from .preprocess import normalize_total_log1p, to_csr


# ------------------------------------------------------------------ helpers


def _sample_design(samples, condition, case: str, control: str) -> pd.Series:
    """Per-sample condition (``case``/``control`` samples only); every sample must
    carry exactly one condition."""
    df = pd.DataFrame({"s": np.asarray(samples).astype(str),
                       "c": np.asarray(condition).astype(str)})
    n_cond = df.groupby("s")["c"].nunique()
    if (n_cond > 1).any():
        raise ValueError(f"samples with more than one condition: {list(n_cond[n_cond > 1].index)}")
    cond = df.drop_duplicates("s").set_index("s")["c"]
    cond = cond[cond.isin([case, control])].sort_index()
    for g in (case, control):
        if (cond == g).sum() < 2:
            raise ValueError(f"need >= 2 samples in group {g!r}, got {(cond == g).sum()}")
    return cond


def _covariate_matrix(covariates, samples, index: pd.Index) -> pd.DataFrame:
    """Sample-level covariates (DataFrame indexed by sample, or per-cell DataFrame
    collapsed to the first value per sample) -> numeric design columns."""
    if covariates is None:
        return pd.DataFrame(index=index)
    cov = pd.DataFrame(covariates).copy()
    if len(cov) == len(samples) and not set(index).issubset(cov.index.astype(str)):
        cov.index = np.asarray(samples).astype(str)
        cov = cov[~cov.index.duplicated()]
    cov.index = cov.index.astype(str)
    cov = cov.reindex(index)
    return pd.get_dummies(cov, drop_first=True, dtype=float)


def _onehot(groups: np.ndarray, levels) -> sp.csr_matrix:
    code = pd.Index(levels).get_indexer(groups)
    keep = code >= 0
    return sp.csr_matrix((np.ones(keep.sum()), (np.flatnonzero(keep), code[keep])),
                         shape=(len(groups), len(levels)))


# ------------------------------------------------------- composition (MASC/scCODA)


def composition_test(labels, samples, condition, case: str, control: str,
                     covariates=None, pseudocount: float = 0.5,
                     use_t: bool = True) -> pd.DataFrame:
    """Per-cell-type change in proportion between ``case`` and ``control`` samples.

    For each cell type a binomial GLM is fitted to the per-sample counts
    (successes = cells of that type, trials = all cells of the sample) with condition
    (plus optional sample-level ``covariates``) as predictor. Over-dispersion between
    donors is absorbed by a quasi-likelihood scale estimated from the Pearson chi^2
    (``scale='X2'``, floored at 1), i.e. a *quasi-binomial GLM*. This is a frequentist,
    sample-level stand-in for the MASC (mixed-effects logistic) / scCODA (Bayesian
    Dirichlet-multinomial) family: it does not model the compositional constraint
    jointly across cell types and has no reference-cell-type or spike-and-slab prior.

    As in R's ``quasibinomial``, Wald tests use a t distribution on the residual
    degrees of freedom (``use_t``). Proportions are relative: a large expansion of one
    population (e.g. a novel disease state) lowers every other proportion, a caveat
    scCODA addresses with a reference cell type.

    Returns one row per cell type: mean proportion per group, log2 fold change of mean
    proportions, condition coefficient (log odds ratio), its SE, quasi-likelihood
    dispersion, p and BH FDR, sorted by p.
    """
    labels = np.asarray(labels).astype(str)
    samples = np.asarray(samples).astype(str)
    cond = _sample_design(samples, condition, case, control)
    keep = np.isin(samples, cond.index)
    counts = pd.crosstab(samples[keep], labels[keep]).reindex(cond.index).fillna(0)
    total = counts.sum(axis=1).to_numpy(float)
    X = pd.DataFrame({"const": 1.0, "case": (cond == case).astype(float)}, index=cond.index)
    X = pd.concat([X, _covariate_matrix(covariates, samples, cond.index)], axis=1)
    is_case = (cond == case).to_numpy()
    props = counts.div(total, axis=0)
    rows = []
    for ct in counts.columns:
        y = counts[ct].to_numpy(float)
        # continuity correction when a group has no cells of this type (separation)
        yy, nn = y, total
        if y[is_case].sum() == 0 or y[~is_case].sum() == 0:
            yy, nn = y + pseudocount, total + 2 * pseudocount
        coef = se = p = scale = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # proportions with binomial weights (statsmodels' two-column endog
                # mis-scales the Pearson X2 dispersion)
                model = sm.GLM(yy / nn, X.to_numpy(), family=sm.families.Binomial(),
                               var_weights=nn)
                fit = model.fit(scale="X2", use_t=use_t)
                if fit.scale < 1.0:   # never claim under-dispersion
                    fit = model.fit(scale=1.0, use_t=use_t)
            coef, se, p, scale = fit.params[1], fit.bse[1], fit.pvalues[1], fit.scale
        except Exception:     # pragma: no cover - degenerate design
            pass
        pc, pk = props.loc[is_case, ct].mean(), props.loc[~is_case, ct].mean()
        eps = 0.5 / total.mean()
        rows.append(dict(cell_type=ct, n_cells=int(y.sum()),
                         **{f"prop_{case}": pc, f"prop_{control}": pk},
                         log2FC=np.log2((pc + eps) / (pk + eps)), coef=coef, se=se,
                         dispersion=scale, p=p))
    df = pd.DataFrame(rows)
    pv = df["p"].fillna(1.0).to_numpy()
    df["fdr"] = bh_fdr(pv)
    return df.sort_values("p").reset_index(drop=True)


# ------------------------------------------------------------------------ Milo


@dataclass
class MiloResult:
    nhoods: pd.DataFrame            # one row per neighbourhood
    membership: sp.csr_matrix       # (n_cells, n_nhoods) binary
    counts: pd.DataFrame            # (n_nhoods, n_samples) cell counts
    design: pd.DataFrame            # per sample: condition, n_cells
    case: str
    control: str
    params: dict = field(default_factory=dict)

    def annotate(self, labels, name: str = "label") -> pd.DataFrame:
        """``nhoods`` plus the majority ``labels`` value of each neighbourhood and
        the fraction of its cells carrying it."""
        labels = np.asarray(labels).astype(str)
        levels = np.unique(labels)
        M = (self.membership.T @ _onehot(labels, levels)).toarray()   # nhood x level
        frac = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
        df = self.nhoods.copy()
        df[name] = levels[frac.argmax(axis=1)]
        df[f"{name}_fraction"] = frac.max(axis=1)
        return df

    def fraction(self, mask) -> np.ndarray:
        """Fraction of each neighbourhood's cells for which boolean ``mask`` holds."""
        m = np.asarray(mask, float)
        return (self.membership.T @ m) / np.asarray(self.membership.sum(axis=0)).ravel()

    def cell_da_score(self, alpha: float = 0.1) -> np.ndarray:
        """Per-cell DA score: mean logFC of the significant (spatial FDR < ``alpha``)
        neighbourhoods containing the cell; 0 for cells in none."""
        sig = (self.nhoods["spatial_fdr"] < alpha).to_numpy()
        lfc = np.where(sig, self.nhoods["logFC"].to_numpy(), 0.0)
        M = self.membership
        n_sig = M @ sig.astype(float)
        tot = M @ lfc
        return np.divide(tot, n_sig, out=np.zeros_like(tot), where=n_sig > 0)


def _refined_sampling(Z: np.ndarray, idx: np.ndarray, prop: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Milo refined sampling: random cells are replaced by the cell nearest to the
    median position of their kNN neighbourhood, then deduplicated."""
    n = Z.shape[0]
    m = max(1, int(round(prop * n)))
    init = rng.choice(n, m, replace=False)
    nb = np.concatenate([init[:, None], idx[init]], axis=1)
    med = np.median(Z[nb], axis=1)
    _, nearest = NearestNeighbors(n_neighbors=1).fit(Z).kneighbors(med)
    return np.unique(nearest.ravel())


def _nb_lrt(y: np.ndarray, X: np.ndarray, offset: np.ndarray, alpha: float
            ) -> tuple[float, float]:
    """NB GLM with fixed dispersion: condition coefficient (column 1) and its
    likelihood-ratio p. LRT stays valid under separation (all-zero groups), where a
    Wald test breaks down (Hauck-Donner)."""
    fam = sm.families.NegativeBinomial(alpha=alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = sm.GLM(y, X, family=fam, offset=offset).fit()
        red = sm.GLM(y, np.delete(X, 1, axis=1), family=fam, offset=offset).fit()
    lr = max(2.0 * (full.llf - red.llf), 0.0)
    return float(full.params[1]), float(stats.chi2.sf(lr, 1))


def _spatial_fdr(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Milo ``graphSpatialFDR``: weighted BH with weights = 1 / k-distance."""
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p) & np.isfinite(w)
    pv, wv = p[ok], w[ok]
    o = np.argsort(pv)
    adj = np.empty(len(pv))
    adj[o] = np.minimum.accumulate((wv.sum() * pv[o] / np.cumsum(wv[o]))[::-1])[::-1]
    out[ok] = np.minimum(adj, 1.0)
    return out


def milo(Z: np.ndarray, samples, condition, case: str, control: str, k: int = 30,
         prop: float = 0.1, seed: int = 0, prior_df: float = 10.0) -> MiloResult:
    """Milo differential abundance testing on the joint embedding.

    1. kNN graph on ``Z`` (e.g. ``MappingResult.Zq_corr``).
    2. Refined sampling of index cells; neighbourhood = index cell + its k neighbours.
    3. Count cells per (neighbourhood, sample).
    4. Per neighbourhood, NB GLM ``count ~ condition`` with offset log(total cells of
       the sample). Dispersion: per-neighbourhood moment estimate from a Poisson fit,
       squeezed towards the across-neighbourhood (10% trimmed) mean with ``prior_df`` prior
       degrees of freedom (an edgeR-like robust, shared estimate). Significance by a
       likelihood-ratio test (edgeR's QL F-test in the original); on convergence
       failure the neighbourhood falls back to a quasi-Poisson Wald test, else p=NaN.
    5. Spatial FDR: weighted BH with weight 1 / (distance from the index cell to its
       k-th nearest neighbour), correcting for overlap between neighbourhoods.

    ``logFC`` is log2(case / control).
    """
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, float)
    samples = np.asarray(samples).astype(str)
    cond = _sample_design(samples, condition, case, control)
    dist, idx = knn(Z, k=k, exclude_self=True)
    index_cells = _refined_sampling(Z, idx, prop, rng)
    n, m = Z.shape[0], len(index_cells)
    members = np.concatenate([index_cells[:, None], idx[index_cells]], axis=1)
    cols = np.repeat(np.arange(m), members.shape[1])
    membership = sp.csr_matrix((np.ones(cols.size), (members.ravel(), cols)), shape=(n, m))
    membership.data[:] = 1.0
    S = _onehot(samples, cond.index)
    counts = (membership.T @ S).toarray()                          # nhood x sample
    n_cells = np.asarray(S.sum(axis=0)).ravel()
    offset = np.log(n_cells)
    X = np.column_stack([np.ones(len(cond)), (cond == case).to_numpy(float)])
    df_res = len(cond) - X.shape[1]

    # dispersion: Poisson-based moments, squeezed to the common (median) value
    raw = np.full(m, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(m):
            y = counts[j]
            if y.sum() == 0:
                continue
            try:
                mu = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit().mu
                # unbiased (untruncated) moment estimate; may be negative
                raw[j] = (np.sum(((y - mu) ** 2 - mu) / np.maximum(mu, 1e-8) ** 2)
                          / max(df_res, 1))
            except Exception:
                pass
    fin = raw[np.isfinite(raw)]
    common = max(float(stats.trim_mean(fin, 0.1)), 1e-3) if len(fin) else 0.1
    disp = np.where(np.isfinite(raw), (prior_df * common + df_res * raw) / (prior_df + df_res),
                    common)
    disp = np.clip(disp, 1e-3, 10.0)

    coef, p = np.full(m, np.nan), np.full(m, np.nan)
    for j in range(m):
        y = counts[j]
        if y.sum() == 0:
            continue
        try:
            coef[j], p[j] = _nb_lrt(y, X, offset, disp[j])
        except Exception:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit(scale="X2")
                coef[j], p[j] = f.params[1], f.pvalues[1]
            except Exception:
                pass
    # descriptive logFC (0.5 pseudo-count) where the GLM failed or is separated
    # (a group with no cells -> the MLE diverges)
    rate = (counts + 0.5) / n_cells
    is_case = (cond == case).to_numpy()
    naive = np.log(rate[:, is_case].mean(axis=1) / rate[:, ~is_case].mean(axis=1))
    separated = (counts[:, is_case].sum(axis=1) == 0) | (counts[:, ~is_case].sum(axis=1) == 0)
    coef = np.where(np.isfinite(coef) & ~separated, coef, naive)
    kdist = dist[index_cells, -1]
    w = 1.0 / np.maximum(kdist, 1e-12)
    nh = pd.DataFrame(dict(nhood=np.arange(m), index_cell=index_cells,
                           size=np.asarray(membership.sum(axis=0)).ravel().astype(int),
                           kth_distance=kdist, logFC=coef / np.log(2), dispersion=disp,
                           p=p, spatial_fdr=_spatial_fdr(p, w)))
    design = pd.DataFrame({"condition": cond, "n_cells": n_cells.astype(int)})
    return MiloResult(nhoods=nh, membership=membership,
                      counts=pd.DataFrame(counts, columns=cond.index), design=design,
                      case=case, control=control,
                      params=dict(k=k, prop=prop, seed=seed, prior_df=prior_df,
                                  common_dispersion=common))


# ---------------------------------------------------- pseudobulk DE (muscat-style)


def _trigamma_inverse(x: float) -> float:
    """Solve trigamma(y) = x (limma ``trigammaInverse``, Newton iterations)."""
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = polygamma(1, y)
        dif = tri * (1.0 - tri / x) / polygamma(2, y)
        y += dif
        if -dif / y < 1e-8:
            break
    return float(y)


def _squeeze_var(s2: np.ndarray, d: float) -> tuple[float, float]:
    """limma ``fitFDist``: prior df ``d0`` and prior variance ``s0^2`` from the
    distribution of residual variances (method of moments on log s^2)."""
    ok = s2 > 0
    z = np.log(s2[ok])
    e = z - digamma(d / 2) + np.log(d / 2)
    emean = e.mean()
    evar = e.var(ddof=1) - polygamma(1, d / 2)
    if evar > 0:
        d0 = 2.0 * _trigamma_inverse(evar)
        s02 = np.exp(emean + digamma(d0 / 2) - np.log(d0 / 2))
    else:
        d0, s02 = np.inf, np.exp(emean)
    return d0, s02


def _moderated_t(Y: np.ndarray, X: np.ndarray, j: int = 1) -> dict:
    """Gene-wise OLS of ``Y`` (genes x samples) on design ``X`` with limma eBayes
    variance shrinkage; returns coefficient j, moderated t, p and the priors."""
    n, p = X.shape
    d = n - p
    XtX_inv = np.linalg.pinv(X.T @ X)
    B = Y @ X @ XtX_inv                         # genes x p
    resid = Y - B @ X.T
    s2 = (resid ** 2).sum(axis=1) / d
    d0, s02 = _squeeze_var(s2, d)
    if np.isfinite(d0):
        s2_post = (d0 * s02 + d * s2) / (d0 + d)
        df_total = d0 + d
    else:
        s2_post, df_total = np.full_like(s2, s02), np.inf
    se = np.sqrt(s2_post * XtX_inv[j, j])
    t = B[:, j] / se
    pval = 2 * (stats.norm.sf(np.abs(t)) if np.isinf(df_total)
                else stats.t.sf(np.abs(t), df_total))
    return dict(coef=B[:, j], t=t, p=pval, s2=s2, s2_post=s2_post, d0=d0, s02=s02,
                df=df_total)


def tmm_factors(Y: np.ndarray, trim_m: float = 0.3, trim_a: float = 0.05) -> np.ndarray:
    """edgeR TMM normalisation factors for a genes x samples count matrix
    (Robinson & Oshlack 2010), scaled to have geometric mean 1."""
    Y = np.asarray(Y, float)
    lib = Y.sum(axis=0)
    uq = np.quantile(Y / lib, 0.75, axis=0)
    r = int(np.argmin(np.abs(uq - uq.mean())))
    f = np.ones(Y.shape[1])
    for j in range(Y.shape[1]):
        if j == r:
            continue
        o, ref = Y[:, j], Y[:, r]
        ok = (o > 0) & (ref > 0)
        lo, lr = np.log2(o[ok] / lib[j]), np.log2(ref[ok] / lib[r])
        M, A = lo - lr, (lo + lr) / 2
        v = (lib[j] - o[ok]) / lib[j] / o[ok] + (lib[r] - ref[ok]) / lib[r] / ref[ok]
        n = len(M)
        if n < 10:
            continue
        rm, ra = stats.rankdata(M), stats.rankdata(A)
        lo_m, lo_a = np.floor(n * trim_m) + 1, np.floor(n * trim_a) + 1
        keep = (rm >= lo_m) & (rm <= n + 1 - lo_m) & (ra >= lo_a) & (ra <= n + 1 - lo_a)
        if keep.any():
            f[j] = 2 ** (np.sum(M[keep] / v[keep]) / np.sum(1 / v[keep]))
    return f / np.exp(np.mean(np.log(f)))


def pseudobulk_de(counts, genes, labels, samples, condition, case: str, control: str,
                  min_cells: int = 10, prior_count: float = 2.0, min_samples: int = 2,
                  cell_types=None, covariates=None,
                  normalization: str = "TMM") -> pd.DataFrame:
    """Pseudobulk differential state analysis per cell type (muscat, ref. 37).

    Raw counts are summed per (cell type, sample); samples with fewer than
    ``min_cells`` cells of the type are dropped and the type is skipped unless each
    group keeps ``min_samples`` samples. Library sizes are TMM-normalised (edgeR, as in
    muscat; ``normalization="none"`` for raw library sizes) so a strong programme
    switched on in one condition does not shift every other gene; pseudobulks are
    converted to log2-CPM with an edgeR-style library-scaled ``prior_count``; genes
    with zero counts in more than
    half the samples are not tested. Each gene is tested with a gene-wise linear
    model ``logCPM ~ condition (+ covariates)`` and a limma-like moderated t
    (empirical-Bayes shrinkage of residual variances towards a common prior; no
    mean-variance trend and no precision weights, i.e. limma without voom/trend).
    BH FDR is computed within each cell type.

    Returns a long DataFrame: cell_type, gene, logFC (log2 case/control), ave_logCPM,
    t, p, fdr, n_case, n_control, d0 (prior df).
    """
    X = to_csr(counts)
    genes = np.asarray(genes).astype(str)
    labels = np.asarray(labels).astype(str)
    samples = np.asarray(samples).astype(str)
    cond = _sample_design(samples, condition, case, control)
    cov = _covariate_matrix(covariates, samples, cond.index)
    types = np.unique(labels) if cell_types is None else np.asarray(cell_types).astype(str)
    out = []
    for ct in types:
        m = (labels == ct) & np.isin(samples, cond.index)
        if not m.any():
            continue
        ncell = pd.Series(samples[m]).value_counts().reindex(cond.index).fillna(0)
        keep_s = ncell.index[ncell >= min_cells]
        c = cond[keep_s]
        if (c == case).sum() < min_samples or (c == control).sum() < min_samples:
            continue
        mm = m & np.isin(samples, keep_s)
        Y = (_onehot(samples[mm], keep_s).T @ X[mm]).toarray().T      # genes x samples
        lib = Y.sum(axis=0) * (tmm_factors(Y) if normalization == "TMM" else 1.0)
        pc = prior_count * lib / lib.mean()
        logcpm = np.log2((Y + pc) / (lib + 2 * pc) * 1e6)
        tested = (Y > 0).sum(axis=1) >= np.ceil(len(keep_s) / 2)
        if tested.sum() < 3:
            continue
        D = np.column_stack([np.ones(len(keep_s)), (c == case).to_numpy(float),
                             cov.loc[keep_s].to_numpy(float) if cov.shape[1] else
                             np.empty((len(keep_s), 0))])
        if np.linalg.matrix_rank(D) < D.shape[1] or len(keep_s) <= D.shape[1]:
            continue
        r = _moderated_t(logcpm[tested], D)
        out.append(pd.DataFrame(dict(
            cell_type=ct, gene=genes[tested], logFC=r["coef"],
            ave_logCPM=logcpm[tested].mean(axis=1), t=r["t"], p=r["p"],
            fdr=bh_fdr(r["p"]), n_case=int((c == case).sum()),
            n_control=int((c == control).sum()), d0=r["d0"])))
    cols = ["cell_type", "gene", "logFC", "ave_logCPM", "t", "p", "fdr", "n_case",
            "n_control", "d0"]
    if not out:
        return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True)[cols]


# ----------------------------------------------------------------------- Augur


def _is_count_matrix(X) -> bool:
    v = X.data if sp.issparse(X) else np.asarray(X).ravel()
    v = v[:100000]
    return bool(len(v) and np.all(v >= 0) and np.allclose(v, np.round(v)) and v.max() > 1)


def _select_variance(Xt: np.ndarray, var_quantile: float) -> np.ndarray:
    """Augur ``select_variance``: keep genes whose coefficient of variation lies above
    the ``var_quantile`` quantile of residuals from a lowess fit of CV on mean."""
    mean, sd = Xt.mean(axis=0), Xt.std(axis=0, ddof=1)
    ok = np.flatnonzero((sd > 0) & (mean > 0))
    if len(ok) < 10:
        return ok
    cv = sd[ok] / mean[ok]
    fitted = sm.nonparametric.lowess(cv, mean[ok], frac=0.5, return_sorted=False)
    resid = cv - fitted
    return ok[resid >= np.quantile(resid, var_quantile)]


def augur(X, labels, condition, n_subsamples: int = 20, subsample_size: int = 20,
          folds: int = 3, var_quantile: float = 0.5, seed: int = 0, *,
          case: str | None = None, control: str | None = None, n_estimators: int = 100,
          feature_perc: float = 0.5, normalize: bool | None = None) -> pd.DataFrame:
    """Augur cell-type prioritisation (ref. 38): the cell types whose molecular state
    best separates the two conditions are those most responsive to the perturbation.

    ``X`` is log-normalised expression or raw counts (auto-detected unless
    ``normalize`` is given; counts are library-normalised + log1p). For each cell
    type, genes are filtered by Augur's variance selection (``var_quantile``); then
    ``n_subsamples`` times, ``subsample_size`` cells per condition and a random
    ``feature_perc`` of the selected genes are drawn and a random forest is evaluated
    by stratified ``folds``-fold cross-validated ROC AUC. Cell types with fewer than
    ``subsample_size`` cells in either condition are skipped. As in Augur, cells (not
    samples) are the instances, so the AUC reflects separability, not replication.

    Returns one row per cell type sorted by mean AUC (descending).
    """
    labels = np.asarray(labels).astype(str)
    condition = np.asarray(condition).astype(str)
    levels = [case, control] if case and control else sorted(pd.unique(condition))
    if len(levels) != 2:
        raise ValueError(f"augur needs exactly two conditions, got {levels}; set case/control")
    if normalize is None:
        normalize = _is_count_matrix(X)
    X = normalize_total_log1p(X) if normalize else to_csr(X)
    rng = np.random.default_rng(seed)
    rows = []
    for ct in np.unique(labels):
        m = (labels == ct) & np.isin(condition, levels)
        y_all = (condition[m] == levels[0]).astype(int)
        n_pos, n_neg = int(y_all.sum()), int((1 - y_all).sum())
        if min(n_pos, n_neg) < subsample_size:
            continue
        Xt = X[m].toarray()
        feats = _select_variance(Xt, var_quantile)
        if len(feats) < 2:
            continue
        pos, neg = np.flatnonzero(y_all == 1), np.flatnonzero(y_all == 0)
        aucs = []
        for _ in range(n_subsamples):
            cells = np.concatenate([rng.choice(pos, subsample_size, replace=False),
                                    rng.choice(neg, subsample_size, replace=False)])
            f = rng.choice(feats, max(2, int(round(feature_perc * len(feats)))),
                           replace=False)
            Xs, ys = Xt[np.ix_(cells, f)], y_all[cells]
            cv = StratifiedKFold(n_splits=folds, shuffle=True,
                                 random_state=int(rng.integers(1 << 31)))
            for tr, te in cv.split(Xs, ys):
                clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=1,
                                             random_state=int(rng.integers(1 << 31)))
                clf.fit(Xs[tr], ys[tr])
                aucs.append(roc_auc_score(ys[te], clf.predict_proba(Xs[te])[:, 1]))
        rows.append(dict(cell_type=ct, auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                         n_genes=int(len(feats)), **{f"n_{levels[0]}": n_pos,
                                                     f"n_{levels[1]}": n_neg}))
    df = pd.DataFrame(rows, columns=["cell_type", "auc", "auc_sd", "n_genes",
                                     f"n_{levels[0]}", f"n_{levels[1]}"])
    return df.sort_values("auc", ascending=False).reset_index(drop=True)


__all__ = ["composition_test", "milo", "MiloResult", "pseudobulk_de", "tmm_factors",
           "augur"]
