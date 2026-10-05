"""Synthetic multimodal and cross-species data with known ground truth.

Extends ``refmap.simulate`` to the perspective's sections "Single-cell data mapping
across molecular modalities" (Figure 3) and "Cross-species mapping".

*Genomic layout.* Every gene of a :class:`~refmap.simulate.World` gets a coordinate
(chromosome, start, end, strand). ATAC peaks are of four kinds:

* ``promoter`` / ``genebody`` peaks inside the gene body + 2 kb upstream; their
  accessibility follows the gene's expression programme, but only partially
  (``coupling`` < 1 plus per-cell-type noise) -- the imperfect RNA/chromatin
  correspondence the perspective warns about when converting features;
* ``distal`` enhancer peaks (>= 3 kb from any gene window) that follow the programme
  of a nearby gene but are invisible to gene-activity scores;
* ``background`` peaks with no cell-type specificity.

Accessibility is a per-(cell type, peak) open probability; counts are Poisson with a
per-cell depth factor, so matrices are sparse and nearly binary as in scATAC-seq.

``simulate_multiome`` draws an RNA reference (multi-batch, labelled), an scATAC query
(labels kept in ``obs`` for evaluation only) and a 10x-multiome-like *bridge* of paired
RNA + ATAC cells (Seurat v5 bridge integration, Hao et al. 2024, ref. 71).

``simulate_species_pair`` draws a species-1 reference and a species-2 query with renamed
genes, an ortholog table (one-to-one, one-to-many, unmatched), species-specific
expression divergence of a subset of orthologs, and optionally a species-2-specific
cell type absent from species 1 (Butler et al. 2018, ref. 93).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .simulate import DEFAULT_PROPS, World, _sample_blocks, draw_dataset, make_world

UPSTREAM = 2000          # promoter window used by gene-activity scores (Signac default)
PEAK_WIDTH = 500
SPECIFIC_TYPE = ("Species-specific", "Species-specific cell")


# ---------------------------------------------------------------- genomic layout


@dataclass
class GenomeLayout:
    """Gene coordinates, peaks and the per-cell-type open probability of each peak."""
    gene_coords: pd.DataFrame     # index gene; chrom, start, end, strand
    peak_coords: pd.DataFrame     # index peak; chrom, start, end, peak_type, linked_gene
    open_prob: dict               # fine cell type -> (n_peaks,) open probability
    coupling: float


def _layout_genes(rng, genes, n_chrom):
    rows, gaps = [], []
    chrom_of = np.sort(rng.integers(0, n_chrom, len(genes)))
    cursor = {}
    for g, c in zip(genes, chrom_of):
        pos = cursor.get(c, 1_000_000)
        gap = int(rng.uniform(25_000, 90_000))
        length = int(np.clip(rng.lognormal(np.log(15_000), 0.6), 3_000, 120_000))
        start = pos + gap
        end = start + length
        gaps.append((f"chr{c + 1}", pos + 3_000, start - 3_000 - PEAK_WIDTH))
        rows.append((g, f"chr{c + 1}", start, end, "+" if rng.random() < 0.5 else "-"))
        cursor[c] = end + 3_000
    gc = pd.DataFrame(rows, columns=["gene", "chrom", "start", "end", "strand"]).set_index("gene")
    return gc, gaps


def make_layout(world: World, seed: int = 0, n_chrom: int = 10, coupling: float = 0.7,
                noise_sd: float = 0.6, mean_linked: float = 1.5, mean_distal: float = 1.0,
                mean_background: float = 1.0) -> GenomeLayout:
    """Place genes on chromosomes and generate peaks with cell-type accessibility.

    For a gene-linked peak ``j`` of gene ``g`` in cell type ``t`` the open probability is
    ``sigmoid(b_j + gamma_j * coupling * d_tg + noise_sd * eps_tj)`` with ``d_tg`` the
    gene's log-expression deviation from its cross-type mean, ``gamma_j ~ U(0.8, 1.5)``
    and ``eps_tj ~ N(0, 1)`` peak- and type-specific noise (imperfect coupling).
    """
    rng = np.random.default_rng(seed + 101)
    gene_coords, gaps = _layout_genes(rng, world.genes, n_chrom)
    types = list(world.log_mu)
    L = np.vstack([world.log_mu[t] for t in types])
    D = L - L.mean(axis=0)                                   # (types, genes)

    peaks = []                                               # chrom, start, kind, gene idx
    for gi, (g, r) in enumerate(gene_coords.iterrows()):
        n_link = 1 + min(rng.poisson(mean_linked - 1 + 1e-9), 3)
        tss = r.start if r.strand == "+" else r.end
        # promoter: centred within 1 kb upstream of the TSS (inside the 2 kb window)
        off = int(rng.uniform(200, 1_200))
        p0 = tss - off if r.strand == "+" else tss + off - PEAK_WIDTH
        peaks.append((r.chrom, p0, "promoter", gi))
        for _ in range(n_link - 1):
            peaks.append((r.chrom, int(rng.uniform(r.start, r.end - PEAK_WIDTH)), "genebody", gi))
        chrom, lo, hi = gaps[gi]
        if hi - lo > PEAK_WIDTH:
            for _ in range(rng.poisson(mean_distal)):
                peaks.append((chrom, int(rng.uniform(lo, hi)), "distal", gi))
            for _ in range(rng.poisson(mean_background)):
                peaks.append((chrom, int(rng.uniform(lo, hi)), "background", -1))
    # non-overlapping, sorted by position
    df = pd.DataFrame(peaks, columns=["chrom", "start", "peak_type", "gene_idx"])
    df["chrom_n"] = df["chrom"].str[3:].astype(int)
    df = df.sort_values(["chrom_n", "start"]).drop_duplicates(["chrom", "start"])
    df["end"] = df["start"] + PEAK_WIDTH
    df.index = [f"{c}-{s}-{e}" for c, s, e in zip(df["chrom"], df["start"], df["end"])]
    df["linked_gene"] = np.where(df["gene_idx"] >= 0,
                                 np.asarray(world.genes)[df["gene_idx"].clip(lower=0)], "")
    kind = df["peak_type"].to_numpy()
    gidx = df["gene_idx"].to_numpy()
    n_p = len(df)
    base = np.select([kind == "promoter", kind == "genebody", kind == "distal"],
                     [rng.normal(-1.0, 0.7, n_p), rng.normal(-2.3, 0.7, n_p),
                      rng.normal(-2.3, 0.8, n_p)], rng.normal(-2.5, 1.0, n_p))
    gamma = rng.uniform(0.8, 1.5, n_p)
    linked = gidx >= 0
    open_prob = {}
    for ti, t in enumerate(types):
        logit = base.copy()
        logit[linked] += gamma[linked] * coupling * D[ti, gidx[linked]]
        logit += noise_sd * rng.normal(0, 1, n_p) * linked
        open_prob[t] = 1.0 / (1.0 + np.exp(-logit))
    peak_coords = df[["chrom", "start", "end", "peak_type", "linked_gene"]].copy()
    return GenomeLayout(gene_coords=gene_coords, peak_coords=peak_coords,
                        open_prob=open_prob, coupling=coupling)


def draw_atac(layout: GenomeLayout, obs: pd.DataFrame, seed: int = 0, depth: float = 1.0,
              depth_sd: float = 0.5, batch_sd: float = 0.0) -> ad.AnnData:
    """Draw peak counts for cells whose fine type is ``obs['cell_type_l2']``.

    ``batch_sd`` adds a per-peak log-normal technical factor (protocol/lab effect)."""
    rng = np.random.default_rng(seed)
    n_p = len(layout.peak_coords)
    bf = np.exp(rng.normal(0, batch_sd, n_p)) if batch_sd > 0 else np.ones(n_p)
    types = obs["cell_type_l2"].astype(str).to_numpy()
    P = np.vstack([layout.open_prob[t] for t in types]) * bf
    s = rng.lognormal(np.log(depth), depth_sd, len(types))
    X = rng.poisson(np.clip(P * s[:, None], 0, None))
    var = layout.peak_coords.copy()
    return ad.AnnData(X=sp.csr_matrix(X.astype(np.float32)), obs=obs.copy(), var=var)


# ---------------------------------------------------------------- multiome


@dataclass
class MultiomeData:
    ref_rna: ad.AnnData           # labelled multi-batch RNA reference (raw counts)
    query_atac: ad.AnnData        # scATAC query, peaks; labels in obs for evaluation only
    bridge_rna: ad.AnnData        # paired multiome cells, RNA
    bridge_atac: ad.AnnData       # same cells (same obs_names), ATAC
    gene_coords: pd.DataFrame
    peak_coords: pd.DataFrame
    world: World
    layout: GenomeLayout


def simulate_multiome(n_genes: int = 800, n_ref_batches: int = 2, cells_per_ref_batch: int = 800,
                      n_query: int = 1200, n_bridge: int = 600, coupling: float = 0.7,
                      noise_sd: float = 0.6, atac_depth: float = 0.4,
                      query_atac_batch_sd: float = 0.3, seed: int = 0,
                      props: dict | None = None) -> MultiomeData:
    """RNA reference + scATAC query + multiome bridge (Figure 3A)."""
    props = dict(props or DEFAULT_PROPS)
    world = make_world(n_genes=n_genes, seed=seed)
    layout = make_layout(world, seed=seed, coupling=coupling, noise_sd=noise_sd)
    rng = np.random.default_rng(seed + 31)
    ref_design = []
    for b in range(n_ref_batches):
        ref_design += _sample_blocks(rng, f"ref_b{b}", f"rna_batch{b}", cells_per_ref_batch, props)
    ref_rna = draw_dataset(world, ref_design, seed=seed + 32)
    ref_rna.obs_names = [f"rna_{n}" for n in ref_rna.obs_names]

    br_design = _sample_blocks(rng, "multiome", "multiome_lab", n_bridge, props, batch_sd=0.3)
    bridge_rna = draw_dataset(world, br_design, seed=seed + 33)
    bridge_rna.obs_names = [f"bridge_{n}" for n in bridge_rna.obs_names]
    bridge_atac = draw_atac(layout, bridge_rna.obs, seed=seed + 34, depth=atac_depth)

    q = _sample_blocks(rng, "atac_query", "atac_lab", n_query, props)
    qobs = pd.DataFrame([{k: v for k, v in blk.items() if k not in ("n", "cell_type")}
                         | {"cell_type_l1": world.parent[blk["cell_type"]],
                            "cell_type_l2": blk["cell_type"]}
                         for blk in q for _ in range(blk["n"])])
    qobs.index = [f"atac_cell{i}" for i in range(len(qobs))]
    query_atac = draw_atac(layout, qobs, seed=seed + 35, depth=atac_depth,
                           batch_sd=query_atac_batch_sd)
    return MultiomeData(ref_rna=ref_rna, query_atac=query_atac, bridge_rna=bridge_rna,
                        bridge_atac=bridge_atac, gene_coords=layout.gene_coords,
                        peak_coords=layout.peak_coords, world=world, layout=layout)


# ---------------------------------------------------------------- species pair


@dataclass
class SpeciesPair:
    ref: ad.AnnData               # species 1, labelled, multi-batch
    query: ad.AnnData             # species 2 (renamed genes); labels for evaluation only
    orthologs: pd.DataFrame       # columns <species1>, <species2>, homology_type
    world1: World
    world2: World
    diverged_genes: np.ndarray    # species-1 names of orthologs with diverged expression
    specific_type: str | None     # fine label of the species-2-specific type
    species: tuple = ("human", "mouse")
    info: dict = field(default_factory=dict)


def simulate_species_pair(n_genes: int = 800, species: tuple = ("human", "mouse"),
                          n_ref_batches: int = 2, cells_per_ref_batch: int = 800,
                          n_query: int = 1000, frac_one2one: float = 0.85,
                          frac_one2many: float = 0.05, frac_diverged: float = 0.15,
                          divergence_sd: float = 1.0, species_shift_sd: float = 0.3,
                          frac_species2_genes: float = 0.05, specific_type: bool = True,
                          frac_specific: float = 0.1, query_batch_sd: float = 0.4,
                          seed: int = 0) -> SpeciesPair:
    """Species-1 reference and species-2 query linked by an ortholog table.

    Species-1 genes are ``GENE00012``; species-2 genes are ``Gene#####`` with a random
    numbering, so only the ortholog table links them. Of the species-1 genes,
    ``frac_one2one`` have a single ortholog, ``frac_one2many`` two species-2 paralogs
    that split the expression (60/40) and the rest none. Every ortholog's expression is
    shifted by a gene-wise species factor; ``frac_diverged`` of the one-to-one orthologs
    additionally get cell-type-specific divergence ``N(0, divergence_sd)``.
    """
    sp1, sp2 = species
    world1 = make_world(n_genes=n_genes, seed=seed)
    rng = np.random.default_rng(seed + 41)
    types = [t for t in DEFAULT_PROPS]
    G = len(world1.genes)
    kind = rng.choice(["one2one", "one2many", "none"], G,
                      p=[frac_one2one, frac_one2many, 1 - frac_one2one - frac_one2many])
    shift = rng.normal(0, species_shift_sd, G)
    diverged = (kind == "one2one") & (rng.random(G) < frac_diverged)
    L1 = {t: world1.log_mu[t] for t in types}
    base = np.median(np.vstack(list(L1.values())), axis=0)
    cols, src = [], []                  # species-2 gene columns: (log-mean per type, origin)
    for g in range(G):
        if kind[g] == "one2one":
            dv = {t: (rng.normal(0, divergence_sd) if diverged[g] else 0.0) for t in types}
            cols.append({t: L1[t][g] + shift[g] + dv[t] for t in types})
            src.append(g)
        elif kind[g] == "one2many":
            for share in (0.6, 0.4):
                cols.append({t: L1[t][g] + shift[g] + np.log(share) for t in types})
                src.append(g)
    n_extra = int(round(frac_species2_genes * G))
    for _ in range(n_extra):
        v = rng.normal(-0.5, 1.0)
        cols.append({t: v for t in types})
        src.append(-1)
    n2 = len(cols)
    names2 = np.array([f"Gene{i:05d}" for i in rng.permutation(n2)])
    log_mu2 = {t: np.array([c[t] for c in cols]) for t in types}
    parent2 = {t: world1.parent[t] for t in types}
    src = np.asarray(src)
    spec_name = None
    if specific_type:
        coarse, spec_name = SPECIFIC_TYPE
        base2 = np.where(src >= 0, base[np.clip(src, 0, None)] + shift[np.clip(src, 0, None)],
                         np.array([c[types[0]] for c in cols]))
        v = base2.copy()
        mk = rng.choice(n2, 50, replace=False)
        v[mk] += rng.uniform(1.0, 2.5, 50)
        log_mu2[spec_name], parent2[spec_name] = v, coarse
    world2 = World(genes=names2, log_mu=log_mu2, parent=parent2, programs={}, seed=seed)

    ref_design = []
    for b in range(n_ref_batches):
        ref_design += _sample_blocks(rng, f"{sp1}_b{b}", f"{sp1}_batch{b}",
                                     cells_per_ref_batch, DEFAULT_PROPS)
    ref = draw_dataset(world1, ref_design, seed=seed + 42)
    ref.obs_names = [f"{sp1}_{n}" for n in ref.obs_names]
    props2 = dict(DEFAULT_PROPS)
    if spec_name:
        props2 = {k: v * (1 - frac_specific) for k, v in props2.items()}
        props2[spec_name] = frac_specific
    q_design = _sample_blocks(rng, f"{sp2}_s0", f"{sp2}_lab", n_query, props2,
                              batch_sd=query_batch_sd)
    query = draw_dataset(world2, q_design, seed=seed + 43)
    query.obs_names = [f"{sp2}_{n}" for n in query.obs_names]
    query.obs["species_specific"] = query.obs["cell_type_l2"] == (spec_name or "")

    rows = [(world1.genes[s], n, "ortholog_one2one" if kind[s] == "one2one"
             else "ortholog_one2many") for s, n in zip(src, names2) if s >= 0]
    orth = pd.DataFrame(rows, columns=[sp1, sp2, "homology_type"])
    info = dict(n_one2one=int((kind == "one2one").sum()),
                n_one2many=int((kind == "one2many").sum()),
                n_unmatched_species1=int((kind == "none").sum()),
                n_species2_only=n_extra, n_diverged=int(diverged.sum()))
    return SpeciesPair(ref=ref, query=query, orthologs=orth, world1=world1, world2=world2,
                       diverged_genes=np.asarray(world1.genes)[diverged],
                       specific_type=spec_name, species=(sp1, sp2), info=info)
