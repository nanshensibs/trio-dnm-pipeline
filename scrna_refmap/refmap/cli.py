"""Command-line interface: ``refmap <command> ...``.

  build          build a reference from annotated raw counts (.h5ad)
  info           print a reference manifest
  map            map a query: annotation, imputation, novel states, condition effects
  population     sample-level classification / MIL / similarity (Figure 2)
  perturbation   match a perturbed query to a perturbation atlas
  cross-modality scATAC query onto an scRNA reference (gene activity and bridge)
  cross-species  map a query from another species through orthologs
  registry       publish / list / propose-update / harmonize versioned references
  sketch         geometric sketch or metacells of a reference
  demo           run every workflow on simulated data
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd


def _keys(s):
    return [k.strip() for k in s.split(",") if k.strip()] if s else []


def cmd_build(a):
    from .io import read_h5ad
    from .reference import build_reference
    ad = read_h5ad(a.h5ad)
    ref = build_reference(ad, _keys(a.labels), batch_key=a.batch_key, sample_key=a.sample_key,
                          name=a.name, version=a.version, n_hvg=a.n_hvg, n_pcs=a.n_pcs,
                          transform=a.transform, layer=a.layer,
                          continuous_obsm=tuple(_keys(a.continuous)), seed=a.seed)
    ref.save(a.out)
    print(json.dumps({"reference": a.out, "name": ref.name, "version": ref.version,
                      "n_cells": ref.n_cells,
                      "self_mapping_accuracy": ref.calibration.get("self_mapping_accuracy")},
                     indent=2))


def cmd_info(a):
    with open(os.path.join(a.reference, "manifest.json")) as fh:
        m = json.load(fh)
    for k in ("label_counts",):
        if not a.full:
            m.pop(k, None)
    print(json.dumps(m, indent=2))


def cmd_map(a):
    from .io import read_h5ad
    from .pipeline import contextualize
    from .reference import Reference
    from .report import Report
    ref = Reference.load(a.reference)
    q = read_h5ad(a.query)
    rep = Report(f"refmap: {os.path.basename(a.query)} → {ref.name} v{ref.version}")
    truth = dict(kv.split("=") for kv in _keys(a.truth)) if a.truth else None
    res = contextualize(ref, q, a.out, rep, batch_key=a.batch_key, sample_key=a.sample_key,
                        condition_key=a.condition_key, case=a.case, control=a.control,
                        truth_keys=truth, novel_key=a.novel_key, use_umap=not a.no_umap,
                        run_de=not a.no_de, run_augur=not a.no_augur, seed=a.seed)
    rep.write(os.path.join(a.out, "report.html"))
    print(json.dumps({"outputs": res["files"], "report": os.path.join(a.out, "report.html"),
                      "novel_clusters": int(res["mapping"].cluster_summary["novel"].sum())},
                     indent=2))


def cmd_population(a):
    from .io import read_h5ad
    from .pipeline import population
    from .reference import Reference
    from .report import Report
    ref = Reference.load(a.reference)
    q = read_h5ad(a.query)
    rep = Report(f"refmap population: {os.path.basename(a.query)}")
    r = population(ref, q, a.out, rep, sample_key=a.sample_key, phenotype_key=a.phenotype_key,
                   positive=a.positive, query_truth_key=a.truth_key, batch_key=a.batch_key)
    rep.write(os.path.join(a.out, "report.html"))
    print(r["query_predictions"].to_string())


def cmd_perturbation(a):
    from .io import read_h5ad
    from .pipeline import perturbation
    from .report import Report
    rep = Report("refmap perturbation matching")
    r = perturbation(read_h5ad(a.atlas), read_h5ad(a.query), a.out, rep,
                     label_keys=_keys(a.labels), perturbation_key=a.perturbation_key,
                     control=a.atlas_control, condition_key=a.condition_key, treated=a.treated,
                     query_control=a.query_control, batch_key=a.batch_key)
    rep.write(os.path.join(a.out, "report.html"))
    print(r["ranking"].to_string(index=False))


def cmd_cross_modality(a):
    from types import SimpleNamespace

    from .io import read_h5ad
    from .pipeline import cross_modality
    from .report import Report
    data = SimpleNamespace(ref_rna=read_h5ad(a.ref_rna), query_atac=read_h5ad(a.query_atac),
                           bridge_rna=read_h5ad(a.bridge_rna),
                           bridge_atac=read_h5ad(a.bridge_atac),
                           gene_coords=pd.read_csv(a.gene_coords, sep=None, engine="python"))
    rep = Report("refmap cross-modality mapping")
    r = cross_modality(data, a.out, rep, label_keys=_keys(a.labels))
    rep.write(os.path.join(a.out, "report.html"))
    print(json.dumps(r["files"], indent=2))


def cmd_cross_species(a):
    from types import SimpleNamespace

    from .io import read_h5ad
    from .pipeline import cross_species
    from .report import Report
    pair = SimpleNamespace(ref=read_h5ad(a.ref), query=read_h5ad(a.query),
                           orthologs=pd.read_csv(a.orthologs, sep=None, engine="python"))
    rep = Report("refmap cross-species mapping")
    r = cross_species(pair, a.out, rep, label_keys=_keys(a.labels), from_col=a.from_col,
                      to_col=a.to_col, ref_batch_key=a.batch_key)
    rep.write(os.path.join(a.out, "report.html"))
    print(r["result"].cluster_summary.to_string(index=False))


def cmd_registry(a):
    from .atlas import ReferenceRegistry, accept_update, harmonize_references, propose_update
    from .io import read_h5ad
    from .reference import Reference
    reg = ReferenceRegistry(a.root)
    if a.action == "publish":
        print(reg.publish(Reference.load(a.reference)))
    elif a.action == "list":
        for n in reg.names():
            print(reg.history(n).to_string(index=False))
    elif a.action == "propose-update":
        ref = reg.load(a.name, a.version)
        prop = propose_update(ref, read_h5ad(a.h5ad), _keys(a.labels) or None,
                              batch_key=a.batch_key, sample_key=a.sample_key,
                              dataset=a.dataset or os.path.basename(a.h5ad))
        path = prop.to_json(a.review or "update_review.json")
        print(f"candidate v{prop.candidate.version} ({prop.bump}); review: {path}; "
              f"passed={prop.report.get('passed')}")
        if a.accept:
            print("published:", accept_update(reg, prop, force=a.force))
    elif a.action == "harmonize":
        A, B = reg.load(a.name), reg.load(a.other)
        h = harmonize_references(A, B, read_h5ad(a.h5ad), read_h5ad(a.other_h5ad), a.level)
        print(h.table.to_string(index=False))
        print(json.dumps(h.tree, indent=2))


def cmd_sketch(a):
    import numpy as np

    from .compression import geometric_sketch, metacells
    from .reference import Reference
    ref = Reference.load(a.reference)
    if a.method == "geometric":
        idx = geometric_sketch(ref.Z_corr, a.n, seed=a.seed)
        pd.Series(np.asarray(ref.obs.index)[idx]).to_csv(a.out, index=False, header=["cell"])
    else:
        mc = metacells(ref.Z_corr, cells_per_metacell=a.n,
                       by=ref.obs[ref.batch_key].to_numpy() if ref.batch_key else None)
        pd.DataFrame({"cell": ref.obs.index, "metacell": mc.membership}).to_csv(
            a.out, sep="\t", index=False)
    print(a.out)


def cmd_demo(a):
    from .demo import run_demo
    s = run_demo(a.out, seed=a.seed, use_umap=not a.no_umap, fast=a.fast)
    print(json.dumps(s, indent=2, default=str))


def build_parser():
    p = argparse.ArgumentParser(prog="refmap", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a reference")
    b.add_argument("--h5ad", required=True)
    b.add_argument("--labels", required=True, help="comma-separated, coarse to fine")
    b.add_argument("--batch-key")
    b.add_argument("--sample-key")
    b.add_argument("--name", default="reference")
    b.add_argument("--version", default="1.0.0")
    b.add_argument("--n-hvg", type=int, default=2000)
    b.add_argument("--n-pcs", type=int, default=30)
    b.add_argument("--transform", choices=["pca", "spca"], default="pca")
    b.add_argument("--layer")
    b.add_argument("--continuous", help="comma-separated obsm keys to transfer")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("info", help="show a reference manifest")
    i.add_argument("reference")
    i.add_argument("--full", action="store_true")
    i.set_defaults(func=cmd_info)

    m = sub.add_parser("map", help="map a query onto a reference")
    m.add_argument("--reference", required=True)
    m.add_argument("--query", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--batch-key")
    m.add_argument("--sample-key")
    m.add_argument("--condition-key")
    m.add_argument("--case")
    m.add_argument("--control")
    m.add_argument("--truth", help="level=obs_column,... ground truth for evaluation")
    m.add_argument("--novel-key", help="boolean obs column of known novel cells (evaluation)")
    m.add_argument("--no-umap", action="store_true")
    m.add_argument("--no-de", action="store_true")
    m.add_argument("--no-augur", action="store_true")
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=cmd_map)

    po = sub.add_parser("population", help="population-scale sample analysis")
    po.add_argument("--reference", required=True)
    po.add_argument("--query", required=True)
    po.add_argument("--out", required=True)
    po.add_argument("--sample-key", required=True)
    po.add_argument("--phenotype-key", required=True, help="sample phenotype in reference obs")
    po.add_argument("--positive")
    po.add_argument("--truth-key")
    po.add_argument("--batch-key")
    po.set_defaults(func=cmd_population)

    pe = sub.add_parser("perturbation", help="match a query to a perturbation atlas")
    pe.add_argument("--atlas", required=True)
    pe.add_argument("--query", required=True)
    pe.add_argument("--out", required=True)
    pe.add_argument("--labels", required=True)
    pe.add_argument("--perturbation-key", required=True)
    pe.add_argument("--atlas-control", default="control")
    pe.add_argument("--condition-key", required=True)
    pe.add_argument("--treated", required=True)
    pe.add_argument("--query-control", required=True)
    pe.add_argument("--batch-key")
    pe.set_defaults(func=cmd_perturbation)

    cm = sub.add_parser("cross-modality", help="scATAC query onto scRNA reference")
    for f in ("ref-rna", "query-atac", "bridge-rna", "bridge-atac", "gene-coords", "out"):
        cm.add_argument(f"--{f}", required=True)
    cm.add_argument("--labels", required=True)
    cm.set_defaults(func=cmd_cross_modality)

    cs = sub.add_parser("cross-species", help="map across species via orthologs")
    for f in ("ref", "query", "orthologs", "out", "from-col", "to-col", "labels"):
        cs.add_argument(f"--{f}", required=True)
    cs.add_argument("--batch-key")
    cs.set_defaults(func=cmd_cross_species)

    rg = sub.add_parser("registry", help="versioned reference registry")
    rg.add_argument("action", choices=["publish", "list", "propose-update", "harmonize"])
    rg.add_argument("--root", required=True)
    rg.add_argument("--reference")
    rg.add_argument("--name")
    rg.add_argument("--version")
    rg.add_argument("--h5ad")
    rg.add_argument("--labels")
    rg.add_argument("--batch-key")
    rg.add_argument("--sample-key")
    rg.add_argument("--dataset")
    rg.add_argument("--review")
    rg.add_argument("--accept", action="store_true")
    rg.add_argument("--force", action="store_true")
    rg.add_argument("--other")
    rg.add_argument("--other-h5ad")
    rg.add_argument("--level")
    rg.set_defaults(func=cmd_registry)

    sk = sub.add_parser("sketch", help="geometric sketch or metacells")
    sk.add_argument("--reference", required=True)
    sk.add_argument("--method", choices=["geometric", "metacells"], default="geometric")
    sk.add_argument("--n", type=int, required=True,
                    help="sketch size, or cells per metacell")
    sk.add_argument("--out", required=True)
    sk.add_argument("--seed", type=int, default=0)
    sk.set_defaults(func=cmd_sketch)

    d = sub.add_parser("demo", help="run every workflow on simulated data")
    d.add_argument("--out", default="demo")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--no-umap", action="store_true")
    d.add_argument("--fast", action="store_true", help="half-size data")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
