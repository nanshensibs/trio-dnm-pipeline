"""Figures and a self-contained HTML report for a reference-mapping run.

Figures mirror the perspective: the integrated reference + query embedding coloured by
transferred annotation (Figure 1C, first row), mapping uncertainty per query cluster
(Figure 1C, second row), differential abundance / composition / prioritisation of
responding populations, and sample-similarity maps (Figure 2E).
"""
from __future__ import annotations

import base64
import html
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2a6fdb", "#e8833a", "#3aa36b", "#c74652", "#8661c5", "#8a5a44", "#d16ba5",
           "#6b7a8f", "#b5a33a", "#2fa3b5", "#1b3a6b", "#a8432b", "#4f7f2a", "#6b2a5e"]
NOVEL_COLOUR = "#c0392b"


def _colors(levels) -> dict:
    return {lv: PALETTE[i % len(PALETTE)] for i, lv in enumerate(levels)}


def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ embedding
class ReferenceUMAP:
    """2-D view of the reference that queries are *projected into* (as Azimuth and
    scArches do) rather than re-embedded, so every query lands in the same map."""

    def __init__(self, ref, max_cells: int = 20000, seed: int = 0, use_umap: bool = True):
        rng = np.random.default_rng(seed)
        self.sel = np.sort(rng.choice(ref.n_cells, min(ref.n_cells, max_cells), replace=False))
        try:
            if not use_umap:
                raise ImportError
            import warnings

            import umap
            warnings.filterwarnings("ignore", message="n_jobs value", module="umap")
            self.model = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=seed)
            self.ref_xy = self.model.fit_transform(ref.Z_corr[self.sel])
            self.method = "UMAP"
        except Exception:  # umap-learn missing: fall back to the first two latent axes
            self.model = None
            self.ref_xy = ref.Z_corr[self.sel][:, :2]
            self.method = "latent dims 1-2"

    def transform(self, Z: np.ndarray) -> np.ndarray:
        return self.model.transform(Z) if self.model is not None else Z[:, :2]


def plot_mapping(ref, result, umap: ReferenceUMAP | None = None, level: str | None = None,
                 novel_labels: np.ndarray | None = None):
    level = level or ref.label_keys[-1]
    umap = umap or ReferenceUMAP(ref)
    qxy = umap.transform(result.Zq_corr)
    ref_lab = ref.labels(level)[umap.sel]
    q_lab = (novel_labels if novel_labels is not None
             else result.labels[f"{level}_pred"].to_numpy()).astype(str)
    cols = _colors(sorted(set(ref_lab)))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    for lab in sorted(set(ref_lab)):
        m = ref_lab == lab
        ax[0].scatter(*umap.ref_xy[m].T, s=2, c=cols[lab], label=lab, rasterized=True)
    ax[0].set_title(f"Reference atlas ({ref.name} v{ref.version})")
    ax[0].legend(markerscale=5, fontsize=7, frameon=False, loc="best")
    ax[1].scatter(*umap.ref_xy.T, s=2, c="#dfe3ea", rasterized=True)
    for lab in sorted(set(q_lab)):
        m = q_lab == lab
        c = NOVEL_COLOUR if lab in ("Novel", "Unknown") else cols.get(lab, "#444")
        ax[1].scatter(*qxy[m].T, s=3, c=c, label=lab, rasterized=True)
    ax[1].set_title("Query projected and annotated")
    ax[1].legend(markerscale=5, fontsize=7, frameon=False, loc="best")
    ax[2].scatter(*umap.ref_xy.T, s=2, c="#dfe3ea", rasterized=True)
    sc = ax[2].scatter(*qxy.T, s=3, c=result.scores["knn_distance"], cmap="magma_r",
                       rasterized=True)
    fig.colorbar(sc, ax=ax[2], shrink=0.7, label="mean kNN distance to reference")
    ax[2].set_title("Mapping distance (higher = less represented)")
    for a in ax:
        a.set_xticks([]), a.set_yticks([])
        a.set_xlabel(f"{umap.method} 1"), a.set_ylabel(f"{umap.method} 2")
    return fig


def plot_cluster_uncertainty(ref, result):
    """Figure 1C 'Mapping uncertainty': per-query-cluster distributions."""
    s = result.cluster_summary
    order = list(s["cluster"])
    novel = dict(zip(s["cluster"], s["novel"]))
    stats = [("knn_distance", "mean kNN distance", "knn_distance"),
             ("mahalanobis", "Symphony Mahalanobis distance", "mahalanobis"),
             ("uncertainty", "label-transfer uncertainty", "uncertainty")]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    q = ref.calibration.get("quantiles", {})
    for a, (col, title, qkey) in zip(ax, stats):
        data = [result.scores[col].to_numpy()[result.clusters == c] for c in order]
        bp = a.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, c in zip(bp["boxes"], order):
            patch.set_facecolor(NOVEL_COLOUR if novel[c] else "#2a6fdb")
            patch.set_alpha(0.75)
        if qkey in q and "0.99" in q[qkey]:
            a.axhline(q[qkey]["0.99"], ls="--", c="#555", lw=1)
            a.text(0.99, q[qkey]["0.99"], " reference 99%", ha="right", va="bottom",
                   fontsize=7, transform=a.get_yaxis_transform())
        a.set_xticks(range(1, len(order) + 1))
        a.set_xticklabels([f"C{c}" for c in order], fontsize=8)
        a.set_title(title)
        a.set_xlabel("query cluster (red = flagged novel)")
    return fig


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, highlight=None, xlabel=None):
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.32 * len(df) + 1)))
    colours = ["#c74652" if (highlight is not None and h) else "#2a6fdb"
               for h in (df[highlight] if highlight else [False] * len(df))]
    ax.barh(df[x].astype(str), df[y], color=colours)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel or y)
    ax.axvline(0, c="#333", lw=0.8)
    return fig


def plot_heatmap(mat: pd.DataFrame, title: str, cmap: str = "viridis"):
    fig, ax = plt.subplots(figsize=(min(16, 1 + 0.35 * mat.shape[1]),
                                    min(14, 1 + 0.35 * mat.shape[0])))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap=cmap)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(title)
    return fig


# ---------------------------------------------------------------------- HTML
CSS = """
:root{--fg:#1d2433;--muted:#5b6476;--bg:#fff;--card:#f6f8fb;--line:#dde3ec;--acc:#2a6fdb;--bad:#c0392b}
@media (prefers-color-scheme: dark){:root{--fg:#e6e9ef;--muted:#a3abba;--bg:#14171d;--card:#1c2029;--line:#2c3240;--acc:#6fa0f0;--bad:#ef6f5e}}
body{font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--fg);background:var(--bg);max-width:1180px;margin:0 auto;padding:24px 16px}
h1{font-size:24px;margin:0 0 4px} h2{font-size:18px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.sub{color:var(--muted)} .card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:12px 0}
img{max-width:100%;height:auto;background:#fff;border-radius:6px}
table{border-collapse:collapse;font-size:12.5px;width:100%;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap} th{background:var(--card)}
.flag{color:var(--bad);font-weight:600} code{font-size:12.5px}
"""


def df_html(df: pd.DataFrame, max_rows: int = 40, float_fmt: str = "{:.3g}") -> str:
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    more = f"<p class='sub'>showing {max_rows} of {len(df)} rows</p>" if len(df) > max_rows else ""
    return d.to_html(index=False, escape=True, border=0) + more


class Report:
    def __init__(self, title: str, subtitle: str = ""):
        self.title, self.subtitle, self.parts = title, subtitle, []

    def section(self, title: str):
        self.parts.append(f"<h2>{html.escape(title)}</h2>")

    def text(self, t: str):
        self.parts.append(f"<p>{t}</p>")

    def note(self, t: str):
        self.parts.append(f"<div class='card'>{t}</div>")

    def figure(self, fig, caption: str = ""):
        self.parts.append(f"<figure><img src='{fig_to_data_uri(fig)}'/>"
                          f"<figcaption class='sub'>{html.escape(caption)}</figcaption></figure>")

    def table(self, df: pd.DataFrame, max_rows: int = 40):
        self.parts.append(df_html(df, max_rows))

    def json(self, obj, title: str = ""):
        self.parts.append(f"<details><summary>{html.escape(title or 'details')}</summary><pre>"
                          f"{html.escape(json.dumps(obj, indent=2, default=str))}</pre></details>")

    def write(self, path: str) -> str:
        doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
               f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
               f"<title>{html.escape(self.title)}</title><style>{CSS}</style></head><body>"
               f"<h1>{html.escape(self.title)}</h1><p class='sub'>{html.escape(self.subtitle)}</p>"
               + "\n".join(self.parts) + "</body></html>")
        with open(path, "w") as fh:
            fh.write(doc)
        return path
