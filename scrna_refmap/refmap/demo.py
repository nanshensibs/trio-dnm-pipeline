"""Run every workflow of the perspective end to end on simulated data with known truth.

``refmap demo --out demo/`` writes ``demo/report.html`` plus each workflow's tables.
All data are SIMULATED (``refmap.simulate``, ``refmap.simulate_modalities``); the
demo shows what the pipeline produces and checks the outputs against ground truth.
"""
from __future__ import annotations

import os
import time

from . import __version__
from .io import write_json
from .pipeline import (atlasing, compression, contextualize, cross_modality, cross_species,
                       perturbation, population)
from .reference import build_reference
from .report import Report
from .simulate import (draw_dataset, simulate_cohort, simulate_perturbation_atlas,
                       simulate_reference_query)
from .simulate_modalities import simulate_multiome, simulate_species_pair

KEYS = ["cell_type_l1", "cell_type_l2"]


def run_demo(out: str = "demo", seed: int = 0, use_umap: bool = True, fast: bool = False,
             log=print) -> dict:
    os.makedirs(out, exist_ok=True)
    scale = 0.5 if fast else 1.0
    rep = Report("refmap: automated single-cell analysis by reference mapping",
                 f"refmap {__version__} · demo on SIMULATED data with known ground truth · "
                 "workflows from Lotfollahi, Hao, Theis & Satija, Cell 187:2343 (2024)")
    rep.note("<b>All data in this report are simulated.</b> Each section runs one "
             "workflow from the perspective and compares the outputs with the "
             "simulation's ground truth.")
    summary, t0 = {}, time.time()

    def step(name):
        log(f"[{time.time() - t0:6.1f}s] {name}")

    # 1. reference + disease query --------------------------------------------------
    step("1/7 healthy reference + disease query (Figure 1)")
    ref_ad, q, _ = simulate_reference_query(seed=seed,
                                            cells_per_ref_sample=int(600 * scale),
                                            cells_per_query_sample=int(500 * scale))
    ref = build_reference(ref_ad, KEYS, batch_key="batch", sample_key="sample",
                          name="simulated-pbmc", n_hvg=1000, seed=seed)
    ref.save(os.path.join(out, "reference", "simulated-pbmc"))
    r1 = contextualize(ref, q, os.path.join(out, "1_disease_mapping"), rep, sample_key="sample",
                       condition_key="condition", case="disease", control="control",
                       novel_key="is_novel", use_umap=use_umap, seed=seed)
    summary["disease_mapping"] = r1.get("metrics")

    # 2. population ------------------------------------------------------------------
    step("2/7 population-scale cohort (Figure 2)")
    c_ref, c_q, _ = simulate_cohort(seed=seed, cells_per_sample=int(300 * scale))
    cref = build_reference(c_ref, KEYS, batch_key="batch", sample_key="sample",
                           name="simulated-cohort", n_hvg=1000, seed=seed)
    r2 = population(cref, c_q, os.path.join(out, "2_population"), rep, sample_key="sample",
                    phenotype_key="phenotype", positive="subtype_B",
                    query_truth_key="phenotype", seed=seed)
    qp = r2["query_predictions"]
    pred_col = "predicted" if "predicted" in qp else qp.columns[0]
    summary["population"] = {"query_sample_accuracy": float((qp[pred_col] == qp["truth"]).mean()),
                             "reference_cv_accuracy": r2["classification"].cv_accuracy,
                             "mil_top_population": str(r2["mil"].attribution.iloc[0, 0])}

    # 3. perturbation ----------------------------------------------------------------
    step("3/7 perturbation atlas")
    atlas, world = simulate_perturbation_atlas(seed=seed, cells_per_block=int(150 * scale))
    design = [dict(sample=f"q_{ct}_{c}", batch="new_lab", batch_sd=0.5, cell_type=ct,
                   n=int(150 * scale), condition=c,
                   extra={"pert:drugB": 1.5} if c == "treated" else {})
              for ct in ("CD4 T", "CD14 Mono", "NK") for c in ("control", "treated")]
    pq = draw_dataset(world, design, seed=seed + 5)
    r3 = perturbation(atlas, pq, os.path.join(out, "3_perturbation"), rep, label_keys=KEYS,
                      perturbation_key="perturbation", control="control",
                      condition_key="condition", treated="treated", query_control="control",
                      batch_key="batch", holdout=("drugB", "CD14 Mono"),
                      response_genes=world.genes[world.programs["pert:drugB"]], seed=seed)
    summary["perturbation"] = {"true": "drugB",
                               "top_match": str(r3["ranking"].iloc[0]["perturbation"]),
                               "holdout": r3.get("holdout", {}).get("response")}

    # 4. cross-modality --------------------------------------------------------------
    step("4/7 cross-modality (scATAC onto scRNA, Figure 3)")
    mo = simulate_multiome(seed=seed, cells_per_ref_batch=int(800 * scale),
                           n_query=int(1200 * scale), n_bridge=int(600 * scale))
    r4 = cross_modality(mo, os.path.join(out, "4_cross_modality"), rep, label_keys=KEYS)
    summary["cross_modality"] = r4["accuracy"].to_dict("records") if r4["accuracy"] is not None else None

    # 5. cross-species ---------------------------------------------------------------
    step("5/7 cross-species")
    pair = simulate_species_pair(seed=seed, cells_per_ref_batch=int(800 * scale),
                                 n_query=int(1000 * scale))
    r5 = cross_species(pair, os.path.join(out, "5_cross_species"), rep, label_keys=KEYS,
                       from_col=pair.species[1], to_col=pair.species[0],
                       truth_key="species_specific", ref_batch_key="batch")
    summary["cross_species"] = {"shared_type_accuracy": r5["shared_accuracy"]}

    # 6. atlasing --------------------------------------------------------------------
    step("6/7 versioned atlasing: pull request + harmonisation")
    a = ref_ad[ref_ad.obs["batch"].isin(["ref_batch0", "ref_batch1"])].copy()
    l2 = a.obs["cell_type_l2"].astype(str)
    a.obs["cell_type_l2"] = l2.where(~l2.isin(["CD4 T", "CD8 T"]), "T cell")
    other = build_reference(a, KEYS, batch_key="batch", name="coarse-lab-atlas",
                            n_hvg=1000, calibrate=False, seed=seed)
    r6 = atlasing(ref, q, os.path.join(out, "registry"), os.path.join(out, "6_atlasing"), rep,
                  label_keys=KEYS, batch_key="batch", sample_key="sample",
                  dataset="disease-cohort", other_ref=other, ref_adata=ref_ad, other_adata=a,
                  level="cell_type_l2")
    summary["atlasing"] = {"proposed_version": r6["proposal"].candidate.version,
                           "bump": r6["proposal"].bump,
                           "new_labels": r6["proposal"].report.get("new_labels"),
                           "review_passed": r6["proposal"].report.get("passed")}

    # 7. compression -----------------------------------------------------------------
    step("7/7 data compression")
    r7 = compression(ref, ref_ad, os.path.join(out, "7_compression"), rep, n_sketch=500,
                     seed=seed)
    summary["compression"] = {"DC_fraction": r7["composition"].loc["DC"].to_dict(),
                              "metacell_purity": r7["metacells"].purity(KEYS[-1])}

    step("writing report")
    rep.section("Run summary")
    rep.json(summary, "summary.json")
    rep.write(os.path.join(out, "report.html"))
    write_json(summary, os.path.join(out, "summary.json"))
    step(f"done -> {os.path.join(out, 'report.html')}")
    return summary
