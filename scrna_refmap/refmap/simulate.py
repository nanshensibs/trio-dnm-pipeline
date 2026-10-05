"""Synthetic single-cell data with known ground truth for demos and tests.

A latent-program generative model: every cell type has a log-mean expression profile
built from a shared baseline plus coarse-lineage and fine-type marker programs; counts
are negative-binomial with per-cell size factors; each batch (lab/protocol) multiplies
genes by a batch-specific log-normal factor. Scenarios mirror the use cases of the
perspective (Figure 1C, Figure 2):

* a healthy multi-batch reference (PBMC-like hierarchy, Figure 1C first row),
* a query from a new lab with control and disease donors, where disease adds an
  *unseen* cell state, shifts composition and perturbs a cell state (Figure 1C,
  second row),
* a population cohort with a sample-level phenotype (Figure 2),
* a perturbation atlas (cell types x perturbations).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

HIERARCHY = {
    "T cell": ["CD4 T", "CD8 T"],
    "NK": ["NK"],
    "B cell": ["Naive B", "Memory B"],
    "Myeloid": ["CD14 Mono", "CD16 Mono", "DC"],
}
NOVEL_TYPE = ("Myeloid", "Disease-associated Mono")


@dataclass
class World:
    """Fixed biology shared by every dataset drawn from it."""
    genes: np.ndarray
    log_mu: dict            # fine type -> log-mean vector
    parent: dict            # fine -> coarse
    programs: dict = field(default_factory=dict)   # named gene-index programs
    seed: int = 0


def make_world(n_genes: int = 1000, seed: int = 0, n_coarse_markers: int = 40,
               n_fine_markers: int = 20, marker_strength: tuple = (1.0, 2.5)) -> World:
    rng = np.random.default_rng(seed)
    genes = np.array([f"GENE{i:05d}" for i in range(n_genes)])
    base = rng.normal(-0.5, 1.0, n_genes)
    log_mu, parent = {}, {}
    used = set()

    def pick(k):
        free = np.setdiff1d(np.arange(n_genes), list(used))
        idx = rng.choice(free, k, replace=False)
        used.update(idx.tolist())
        return idx

    hierarchy = {c: list(f) for c, f in HIERARCHY.items()}
    hierarchy[NOVEL_TYPE[0]] = hierarchy[NOVEL_TYPE[0]] + [NOVEL_TYPE[1]]
    programs = {}
    for coarse, fines in hierarchy.items():
        cidx = pick(n_coarse_markers)
        ceff = rng.uniform(*marker_strength, n_coarse_markers)
        programs[f"coarse:{coarse}"] = cidx
        for fine in fines:
            v = base.copy()
            v[cidx] += ceff
            fidx = pick(n_fine_markers)
            v[fidx] += rng.uniform(*marker_strength, n_fine_markers)
            programs[f"fine:{fine}"] = fidx
            log_mu[fine], parent[fine] = v, coarse
    # disease programme: interferon-like response used for state changes
    programs["interferon"] = pick(30)
    # sample-phenotype programme (e.g. tumour subtype) used in population scenario
    programs["subtype_B"] = pick(25)
    # perturbation programmes
    for p in ("drugA", "drugB", "drugC", "KO1"):
        programs[f"pert:{p}"] = pick(25)
    return World(genes=genes, log_mu=log_mu, parent=parent, programs=programs, seed=seed)


def _draw_counts(rng, log_mu: np.ndarray, n: int, batch_factor: np.ndarray,
                 depth: float, dispersion: float = 0.2) -> np.ndarray:
    size = rng.lognormal(np.log(depth), 0.35, n)
    mu = np.exp(log_mu + batch_factor)
    mu = mu / mu.sum() * size[:, None]
    shape = 1.0 / dispersion
    lam = rng.gamma(shape, mu / shape)
    return rng.poisson(lam)


def draw_dataset(world: World, design: list[dict], seed: int = 1,
                 depth: float = 2500.0) -> ad.AnnData:
    """Draw counts from ``world``.

    ``design`` is a list of dicts, one per (sample, cell type) block, with keys
    ``sample, batch, cell_type, n`` and optional ``extra`` (dict program -> effect),
    and any additional obs columns (e.g. ``condition``) that are copied through.
    """
    rng = np.random.default_rng(seed)
    batch_factors: dict = {}
    mats, obs = [], []
    for block in design:
        if block["n"] <= 0:
            continue
        b = block["batch"]
        if b not in batch_factors:
            sd = block.get("batch_sd", 0.4)
            batch_factors[b] = rng.normal(0.0, sd, len(world.genes))
        lm = world.log_mu[block["cell_type"]].copy()
        for prog, eff in block.get("extra", {}).items():
            lm[world.programs[prog]] += eff
        mats.append(_draw_counts(rng, lm, block["n"], batch_factors[b], depth))
        meta = {k: v for k, v in block.items() if k not in ("n", "extra", "batch_sd")}
        meta["cell_type_l1"] = world.parent[block["cell_type"]]
        meta["cell_type_l2"] = block["cell_type"]
        obs += [meta] * block["n"]
    X = sp.csr_matrix(np.vstack(mats).astype(np.float32))
    obs = pd.DataFrame(obs)
    obs = obs.drop(columns=["cell_type"])
    obs.index = [f"cell{i}" for i in range(len(obs))]
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=world.genes))
    return a


DEFAULT_PROPS = {"CD4 T": 0.25, "CD8 T": 0.15, "NK": 0.1, "Naive B": 0.1,
                 "Memory B": 0.06, "CD14 Mono": 0.2, "CD16 Mono": 0.08, "DC": 0.06}


def _sample_blocks(rng, sample, batch, n_cells, props, **meta):
    p = np.array(list(props.values()), float)
    p = p / p.sum()
    counts = rng.multinomial(n_cells, rng.dirichlet(p * 200))
    out = []
    for (ct, _), n in zip(props.items(), counts):
        out.append(dict(sample=sample, batch=batch, cell_type=ct, n=int(n), **meta))
    return out


def simulate_reference_query(n_genes: int = 1000, n_ref_batches: int = 4,
                             ref_samples_per_batch: int = 2, cells_per_ref_sample: int = 600,
                             n_query_ctrl: int = 3, n_query_dis: int = 3,
                             cells_per_query_sample: int = 500, seed: int = 0,
                             query_batch_sd: float = 0.5):
    """Healthy reference + disease/control query (Figure 1C).

    Disease donors carry (i) an unseen ``Disease-associated Mono`` state (15% of
    cells), (ii) expanded CD14 monocytes and depleted naive B cells (composition), and
    (iii) an interferon programme in CD8 T and CD14 Mono (cell-state change).
    """
    world = make_world(n_genes=n_genes, seed=seed)
    rng = np.random.default_rng(seed + 1)
    ref_design = []
    for b in range(n_ref_batches):
        for s in range(ref_samples_per_batch):
            ref_design += _sample_blocks(rng, f"ref_b{b}_s{s}", f"ref_batch{b}",
                                         cells_per_ref_sample, DEFAULT_PROPS,
                                         condition="healthy")
    ref = draw_dataset(world, ref_design, seed=seed + 2)
    ref.obs["is_novel"] = False

    q_design = []
    for i in range(n_query_ctrl):
        q_design += _sample_blocks(rng, f"q_ctrl{i}", "query_lab", cells_per_query_sample,
                                   DEFAULT_PROPS, condition="control", batch_sd=query_batch_sd)
    dis_props = dict(DEFAULT_PROPS)
    dis_props["CD14 Mono"] *= 1.8
    dis_props["Naive B"] *= 0.4
    for i in range(n_query_dis):
        blocks = _sample_blocks(rng, f"q_dis{i}", "query_lab",
                                int(cells_per_query_sample * 0.85), dis_props,
                                condition="disease", batch_sd=query_batch_sd)
        for blk in blocks:
            if blk["cell_type"] in ("CD8 T", "CD14 Mono"):
                blk["extra"] = {"interferon": 1.5}
        blocks.append(dict(sample=f"q_dis{i}", batch="query_lab", cell_type=NOVEL_TYPE[1],
                           n=int(cells_per_query_sample * 0.15), condition="disease",
                           batch_sd=query_batch_sd))
        q_design += blocks
    query = draw_dataset(world, q_design, seed=seed + 3)
    query.obs["is_novel"] = query.obs["cell_type_l2"] == NOVEL_TYPE[1]
    query.obs_names = [f"q{n}" for n in query.obs_names]
    return ref, query, world


def simulate_cohort(n_genes: int = 1000, n_samples: int = 24, cells_per_sample: int = 300,
                    n_query_samples: int = 8, seed: int = 0):
    """Population-scale cohort (Figure 2): samples from several labs labelled with a
    sample-level phenotype (subtype A/B). Subtype B expands memory B cells and switches
    on a ``subtype_B`` programme in CD14 monocytes only -- a cell-population-specific
    signal a multi-instance learner should localise."""
    world = make_world(n_genes=n_genes, seed=seed)
    rng = np.random.default_rng(seed + 11)

    def one(sample, batch, subtype):
        props = dict(DEFAULT_PROPS)
        if subtype == "B":
            props["Memory B"] *= 2.5
        blocks = _sample_blocks(rng, sample, batch, cells_per_sample, props,
                                phenotype=f"subtype_{subtype}")
        if subtype == "B":
            for blk in blocks:
                if blk["cell_type"] == "CD14 Mono":
                    blk["extra"] = {"subtype_B": 1.5}
        return blocks

    ref_design, q_design = [], []
    for i in range(n_samples):
        ref_design += one(f"ref_s{i}", f"lab{i % 3}", "AB"[i % 2])
    for i in range(n_query_samples):
        q_design += one(f"query_s{i}", "lab_query", "AB"[i % 2])
    ref = draw_dataset(world, ref_design, seed=seed + 12)
    query = draw_dataset(world, q_design, seed=seed + 13)
    query.obs_names = [f"q{n}" for n in query.obs_names]
    return ref, query, world


def simulate_perturbation_atlas(n_genes: int = 1000, cells_per_block: int = 150,
                                seed: int = 0,
                                perturbations=("control", "drugA", "drugB", "drugC", "KO1"),
                                cell_types=("CD4 T", "CD14 Mono", "Naive B", "NK")):
    """Perturbation atlas: each perturbation switches on a shared programme with a
    cell-type-specific gain (so responses are partly shared, partly unique)."""
    world = make_world(n_genes=n_genes, seed=seed)
    rng = np.random.default_rng(seed + 21)
    gain = {(p, ct): rng.uniform(0.8, 1.6) for p in perturbations for ct in cell_types}
    design = []
    for ct in cell_types:
        for p in perturbations:
            extra = {} if p == "control" else {f"pert:{p}": 1.5 * gain[(p, ct)]}
            design.append(dict(sample=f"{ct}_{p}", batch="screen", cell_type=ct,
                               n=cells_per_block, perturbation=p, extra=extra))
    atlas = draw_dataset(world, design, seed=seed + 22)
    return atlas, world
