"""End-to-end workflows: each function runs one application of reference mapping from
the perspective, writes its tables to ``out`` and adds a section to a ``Report``.

=====================  ==========================================================
workflow               perspective section / figure
=====================  ==========================================================
``contextualize``      Reference-mapping workflows (Fig. 1B), automated annotation
                       and identification of disease states (Fig. 1C)
``population``         Population-scale reference mapping (Fig. 2)
``perturbation``       Construction of cellular perturbation atlases
``cross_modality``     Single-cell data mapping across molecular modalities (Fig. 3)
``cross_species``      Cross-species mapping
``atlasing``           Path toward machine-learning-based open-source atlasing
``compression``        Data compression for million-cell references
=====================  ==========================================================
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .io import sanitize_obs, write_json, write_table
from .mapping import final_labels, map_query
from .metrics import evaluate_mapping
from .report import (Report, ReferenceUMAP, plot_bar, plot_cluster_uncertainty, plot_heatmap,
                     plot_mapping)


def _path(out, name):
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, name)


# ---------------------------------------------------------------- 1. map + disease
def contextualize(ref, query, out: str, report: Report | None = None, *,
                  batch_key: str | None = None, sample_key: str | None = None,
                  condition_key: str | None = None, case: str | None = None,
                  control: str | None = None, truth_keys: dict | None = None,
                  novel_key: str | None = None, use_umap: bool = True,
                  run_de: bool = True, run_augur: bool = True, write_h5ad: bool = True,
                  seed: int = 0) -> dict:
    """Map a query, annotate it, flag novel states and quantify condition effects."""
    from .differential import augur, composition_test, milo, pseudobulk_de

    fine = ref.label_keys[-1]
    res = map_query(ref, query, batch_key=batch_key, condition_key=condition_key, seed=seed)
    labels = final_labels(res, fine)
    query = query.copy()
    res.annotate(query)
    query.obs["refmap_final_label"] = labels
    out_files = {"mapping": write_table(res.obs_table(), _path(out, "mapping.tsv"), index=True),
                 "clusters": write_table(res.cluster_summary, _path(out, "query_clusters.tsv"))}
    for name, df in res.continuous.items():
        out_files[f"imputed_{name}"] = write_table(df, _path(out, f"imputed_{name}.tsv"), True)
    result = {"mapping": res, "final_labels": labels, "files": out_files}

    if truth_keys or novel_key:
        m = evaluate_mapping(ref, res, query.obs, truth_keys, novel_key, batch_key)
        result["metrics"] = m
        out_files["metrics"] = write_json(m, _path(out, "metrics.json"))

    if report is not None:
        report.section("Automated annotation by reference mapping (Figure 1B-C)")
        cal = ref.calibration.get("self_mapping_accuracy", {})
        report.text(f"Reference <b>{ref.name} v{ref.version}</b>: {ref.n_cells:,} cells, "
                    f"{len(ref.genes):,} features, {ref.n_dims} dims, "
                    f"{ref.params.get('n_centroids')} Symphony centroids. Self-mapping "
                    f"accuracy ({ref.calibration.get('scheme', 'n/a')}): "
                    + ", ".join(f"{k} {v:.3f}" for k, v in cal.items()) + ". "
                    f"Query: {query.n_obs:,} cells; {res.frac_missing_genes:.1%} of "
                    "reference features missing (zero-filled).")
        umap = ReferenceUMAP(ref, use_umap=use_umap, seed=seed)
        report.figure(plot_mapping(ref, res, umap, fine, labels),
                      "Left: reference. Middle: query projected into the reference map and "
                      "annotated by weighted-kNN transfer; red = novel/unknown. Right: "
                      "distance to the reference.")
        counts = pd.Series(labels).value_counts().rename_axis("label").reset_index(name="cells")
        report.table(counts)
        report.section("Contextualising the query: mapping uncertainty per cluster")
        report.figure(plot_cluster_uncertainty(ref, res),
                      "Per query cluster: kNN distance, Symphony Mahalanobis distance and "
                      "label uncertainty (red = flagged novel).")
        report.table(res.cluster_summary)
        if "metrics" in result:
            report.section("The perspective's three mapping criteria")
            report.table(_criteria_table(result["metrics"]))
            report.json(result["metrics"], "all metrics")

    if not (condition_key and sample_key and case and control):
        if write_h5ad:
            sanitize_obs(query)
            out_files["h5ad"] = _path(out, "query_mapped.h5ad")
            query.write_h5ad(out_files["h5ad"])
        return result

    samples = query.obs[sample_key].astype(str).to_numpy()
    cond = query.obs[condition_key].astype(str).to_numpy()
    comp = composition_test(labels, samples, cond, case, control)
    out_files["composition"] = write_table(comp, _path(out, "composition.tsv"))
    da = milo(res.Zq_corr, samples, cond, case, control, seed=seed)
    nh = da.annotate(labels, "label")
    query.obs["milo_da_score"] = da.cell_da_score()
    out_files["milo"] = write_table(nh, _path(out, "milo_nhoods.tsv"))
    result.update(composition=comp, milo=da, milo_nhoods=nh)
    de = aug = None
    if run_de:
        de = pseudobulk_de(query.X, np.asarray(query.var_names), labels, samples, cond,
                           case, control)
        out_files["pseudobulk_de"] = write_table(de, _path(out, "pseudobulk_de.tsv"))
        result["pseudobulk_de"] = de
    if run_augur:
        aug = augur(query.X, labels, cond, case=case, control=control, seed=seed)
        out_files["augur"] = write_table(aug, _path(out, "augur.tsv"))
        result["augur"] = aug
    if write_h5ad:
        sanitize_obs(query)
        out_files["h5ad"] = _path(out, "query_mapped.h5ad")
        query.write_h5ad(out_files["h5ad"])

    if report is not None:
        report.section(f"Responses to the condition ({case} vs {control})")
        report.text("Compositional change per cell type: sample-level quasi-binomial GLM "
                    "(MASC/scCODA family).")
        c = comp.copy()
        c["significant"] = c[_col(c, ("fdr", "FDR", "padj"))] < 0.1
        report.figure(plot_bar(c, _col(c, ("cell_type", "label")),
                               _col(c, ("log2FC", "log2fc", "logFC")),
                               "Composition change", "significant", "log2 fold change"))
        report.table(comp)
        report.text("Cluster-free differential abundance (Milo): neighbourhoods on the "
                    "joint embedding.")
        report.figure(_plot_milo(nh, da), "Each point is a neighbourhood; red = spatial "
                      "FDR < 0.1.")
        report.table(_milo_summary(nh))
        if de is not None:
            report.text("Within-cell-type state changes (muscat-style pseudobulk DE): "
                        "significant genes per cell type.")
            report.table(_de_summary(de))
        if aug is not None:
            report.text("Cell-type prioritisation (Augur): cross-validated AUC of "
                        "separating the conditions.")
            a = aug.reset_index() if "cell_type" not in aug.columns else aug
            report.figure(plot_bar(a, _col(a, ("cell_type", "index")), _col(a, ("auc", "AUC")),
                                   "Augur AUC", None, "AUC"))
            report.table(aug)
    return result


def _col(df, options):
    for o in options:
        if o in df.columns:
            return o
    raise KeyError(f"none of {options} in {list(df.columns)}")


def _criteria_table(m: dict) -> pd.DataFrame:
    rows = []
    for level, v in m.get("criterion1_heterogeneity", {}).items():
        if isinstance(v, dict):
            rows.append(("1 heterogeneity conserved", f"{level} accuracy", v["accuracy"]))
            rows.append(("1 heterogeneity conserved", f"{level} macro-F1", v["macro_f1"]))
        else:
            rows.append(("1 heterogeneity conserved", level, v))
    c2 = m.get("criterion2_integration", {})
    for k in ("ref_query_mixing_mean", "centroid_offset_mean", "query_batch_mixing_entropy"):
        if k in c2:
            rows.append(("2 identical types integrated", k, c2[k]))
    for k, v in m.get("criterion3_novel_states", {}).items():
        rows.append(("3 novel states preserved", k, v))
    return pd.DataFrame(rows, columns=["criterion", "metric", "value"])


def _plot_milo(nh: pd.DataFrame, da):
    import matplotlib.pyplot as plt
    fc = _col(nh, ("logFC", "log2FC", "logfc"))
    fdr = _col(nh, ("SpatialFDR", "spatial_fdr", "FDR"))
    labs = sorted(nh["label"].unique())
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(labs) + 1)))
    rng = np.random.default_rng(0)
    for i, lab in enumerate(labs):
        d = nh[nh["label"] == lab]
        y = i + rng.uniform(-0.3, 0.3, len(d))
        sig = d[fdr] < 0.1
        ax.scatter(d[fc][~sig], y[~sig], s=10, c="#b8c2d3")
        ax.scatter(d[fc][sig], y[sig], s=14, c="#c74652")
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=8)
    ax.axvline(0, c="#333", lw=0.8)
    ax.set_xlabel(f"log fold change ({da.case} vs {da.control})")
    ax.set_title("Milo neighbourhoods by majority label")
    return fig


def _milo_summary(nh: pd.DataFrame) -> pd.DataFrame:
    fc = _col(nh, ("logFC", "log2FC", "logfc"))
    fdr = _col(nh, ("SpatialFDR", "spatial_fdr", "FDR"))
    d = nh.assign(sig_up=(nh[fdr] < 0.1) & (nh[fc] > 0), sig_down=(nh[fdr] < 0.1) & (nh[fc] < 0))
    g = d.groupby("label")
    return pd.DataFrame({"nhoods": g.size(), "sig_up": g["sig_up"].sum(),
                         "sig_down": g["sig_down"].sum(),
                         "median_logFC": g[fc].median()}).reset_index()


def _de_summary(de: pd.DataFrame) -> pd.DataFrame:
    fdr = _col(de, ("fdr", "FDR", "padj"))
    ct = _col(de, ("cell_type", "label"))
    fc = _col(de, ("logFC", "log2FC", "logfc"))
    sig = de[de[fdr] < 0.05].sort_values(fdr)
    rows = [{ct: c, "n_sig": len(d), "n_up": int((d[fc] > 0).sum()),
             "top_genes": ", ".join(d["gene"].head(6))} for c, d in sig.groupby(ct)]
    return pd.DataFrame(rows, columns=[ct, "n_sig", "n_up", "top_genes"])


# ---------------------------------------------------------------- 2. population
def population(ref, query, out: str, report: Report | None = None, *, sample_key: str,
               phenotype_key: str, positive: str | None = None,
               query_truth_key: str | None = None, batch_key: str | None = None,
               seed: int = 0) -> dict:
    """Sample-level embedding, phenotype classification, MIL attribution and
    query-vs-reference sample similarity (Figure 2)."""
    from .population import (classify_samples, mil_attribution, sample_representation,
                             sample_similarity)

    fine = ref.label_keys[-1]
    ref_samples = ref.obs[sample_key].astype(str).to_numpy()
    y = ref.obs.groupby(sample_key, observed=True)[phenotype_key].agg(
        lambda s: s.astype(str).mode().iloc[0])
    rep_ref = sample_representation(ref.Z_corr, ref.labels(fine), ref_samples)
    res = map_query(ref, query, batch_key=batch_key, seed=seed)
    qlab = res.labels[f"{fine}_pred"].to_numpy()
    rep_q = sample_representation(res.Zq_corr, qlab, query.obs[sample_key].astype(str).to_numpy(),
                                  cell_types=rep_ref.cell_types, impute_from=rep_ref)
    clf = classify_samples(rep_ref, y.reindex(rep_ref.table.index), rep_q)
    sim = sample_similarity(rep_q, rep_ref)
    positive = positive or sorted(y.unique())[-1]
    mil = mil_attribution(ref.Z_corr, ref_samples, y, ref.labels(fine), positive, seed=seed)
    qp = clf.query_predictions.copy()
    if query_truth_key:
        truth = query.obs.groupby(sample_key, observed=True)[query_truth_key].agg(
            lambda s: s.astype(str).mode().iloc[0])
        qp["truth"] = truth.reindex(qp.index).to_numpy()
    files = {"sample_predictions": write_table(qp, _path(out, "sample_predictions.tsv"), True),
             "sample_similarity": write_table(sim.similarity, _path(out, "sample_similarity.tsv"), True),
             "mil_attribution": write_table(mil.attribution, _path(out, "mil_attribution.tsv"))}
    if report is not None:
        report.section("Population-scale mapping (Figure 2)")
        report.text(f"{len(rep_ref.table)} reference samples, {len(rep_q.table)} query samples. "
                    f"Sample-level classifier ({phenotype_key}): cross-validated accuracy "
                    f"{clf.cv_accuracy:.2f}, AUC {clf.cv_auc:.2f} on the reference.")
        report.table(qp.reset_index())
        report.figure(plot_heatmap(sim.ordered(), "Query x reference sample similarity"),
                      "Figure 2E: rows query samples, columns reference samples, ordered by "
                      "hierarchical clustering.")
        report.text(f"Multi-instance learning: which cell populations drive '{positive}' "
                    f"(bag AUC {mil.bag_auc:.2f}).")
        report.table(mil.attribution)
    return {"mapping": res, "classification": clf, "similarity": sim, "mil": mil,
            "query_predictions": qp, "files": files}


# ---------------------------------------------------------------- 3. perturbation
def perturbation(atlas, query, out: str, report: Report | None = None, *, label_keys,
                 perturbation_key: str, control: str, condition_key: str, treated: str,
                 query_control: str, batch_key: str | None = None,
                 holdout: tuple | None = None, response_genes=None, seed: int = 0) -> dict:
    """Map a perturbed query onto a perturbation atlas, rank atlas perturbations by
    signature similarity and (optionally) test prediction of a held-out cell type."""
    from .perturbation import (PerturbationModel, evaluate_holdout, match_signatures,
                               query_response)
    from .reference import build_reference

    fine = label_keys[-1]
    aref = build_reference(atlas, label_keys, name="perturbation-atlas", calibrate=False,
                           seed=seed)
    model = PerturbationModel.fit(aref.Z, aref.obs[fine], aref.obs[perturbation_key],
                                  control=control)
    qr = query_response(aref, query, condition_key=condition_key, treated=treated,
                        control=query_control, batch_key=batch_key)
    ranking = match_signatures(qr.per_type, model.signatures(by_cell_type=True))
    files = {"signature_ranking": write_table(ranking, _path(out, "perturbation_ranking.tsv"))}
    result = {"model": model, "ranking": ranking, "files": files}
    if holdout:
        pert, ct = holdout
        held = ((atlas.obs[perturbation_key] == pert) & (atlas.obs[fine] == ct)).to_numpy()
        href = build_reference(atlas[~held].copy(), label_keys, calibrate=False, seed=seed)
        ev = evaluate_holdout(href, atlas, pert, ct, perturbation_key=perturbation_key,
                              cell_type_key=fine, control=control,
                              response_genes=response_genes)
        result["holdout"] = ev
        files["holdout"] = write_json(ev, _path(out, "perturbation_holdout.json"))
    if report is not None:
        report.section("Perturbation atlas: interpreting and predicting responses")
        report.text(f"Atlas: {atlas.n_obs:,} cells, {len(model.perturbations)} perturbations x "
                    f"{len(model.cell_types)} cell types. The query's treated-vs-control "
                    "response (per transferred cell type, in the atlas latent space) is "
                    "matched against every atlas perturbation signature.")
        report.table(ranking)
        if holdout:
            ev = result["holdout"]
            rows = [(k, ev[k]["r2_pred"], ev[k]["r2_baseline"], ev[k]["mse_pred"],
                     ev[k]["mse_baseline"]) for k in ("response", "all")]
            report.text(f"scGen-style latent arithmetic: predict <b>{holdout[0]}</b> in "
                        f"<b>{holdout[1]}</b>, never seen in training (baseline = no change).")
            report.table(pd.DataFrame(rows, columns=["genes", "R2 predicted", "R2 baseline",
                                                     "MSE predicted", "MSE baseline"]))
    return result


# ---------------------------------------------------------------- 4. modalities
def cross_modality(data, out: str, report: Report | None = None, *, label_keys,
                   truth_available: bool = True) -> dict:
    """scATAC query onto an scRNA reference: feature conversion vs multi-omic bridge."""
    from .crossmodal import bridge_integration, gene_activity_transfer

    fine = label_keys[-1]
    fc, _ = gene_activity_transfer(data.ref_rna, data.ref_rna.obs[fine].to_numpy(),
                                   data.query_atac, data.gene_coords)
    br = bridge_integration(data.ref_rna, data.ref_rna.obs[label_keys], data.bridge_rna,
                            data.bridge_atac, data.query_atac)
    tab = pd.DataFrame({"feature_conversion_pred": fc.pred, "feature_conversion_score": fc.score,
                        "bridge_pred": br.labels[f"{fine}_pred"].to_numpy(),
                        "bridge_uncertainty": br.labels[f"{fine}_uncertainty"].to_numpy()},
                       index=data.query_atac.obs_names)
    files = {"predictions": write_table(tab, _path(out, "atac_predictions.tsv"), True)}
    acc = None
    if truth_available and fine in data.query_atac.obs:
        t = data.query_atac.obs[fine].astype(str).to_numpy()
        acc = pd.DataFrame({"method": ["feature conversion (gene activity + CCA anchors)",
                                       "multi-omic bridge (dictionary representation)"],
                            "fine-label accuracy": [(fc.pred == t).mean(),
                                                    (tab["bridge_pred"].to_numpy() == t).mean()]})
    if report is not None:
        report.section("Cross-modality mapping: scATAC query onto an scRNA reference (Figure 3)")
        report.text(f"Reference RNA {data.ref_rna.n_obs:,} cells; ATAC query "
                    f"{data.query_atac.n_obs:,} cells x {data.query_atac.n_vars:,} peaks; "
                    f"bridge {data.bridge_rna.n_obs:,} multiome cells.")
        if acc is not None:
            report.table(acc)
    return {"feature_conversion": fc, "bridge": br, "accuracy": acc, "files": files}


def cross_species(pair, out: str, report: Report | None = None, *, label_keys,
                  from_col: str, to_col: str, truth_key: str | None = None,
                  ref_batch_key: str | None = None) -> dict:
    from .crossspecies import map_across_species

    res = map_across_species(pair.ref, label_keys, pair.query, pair.orthologs, from_col, to_col,
                             ref_batch_key=ref_batch_key)
    files = {"labels": write_table(res.labels, _path(out, "species_labels.tsv")),
             "clusters": write_table(res.cluster_summary, _path(out, "species_clusters.tsv"))}
    acc = None
    fine = label_keys[-1]
    if truth_key and truth_key in pair.query.obs:
        spec = pair.query.obs[truth_key].to_numpy(bool)
        pred = res.labels["pred"].to_numpy()          # finest level
        acc = float((pred[~spec] == pair.query.obs[fine].to_numpy()[~spec]).mean())
    if report is not None:
        report.section("Cross-species mapping")
        report.text(f"Orthologs: {res.ortholog_report.as_dict()}. "
                    + (f"Accuracy on shared cell types: {acc:.3f}. " if acc is not None else "")
                    + "Query clusters with low anchor scores are candidate species-specific "
                    "populations.")
        report.table(res.cluster_summary)
    return {"result": res, "shared_accuracy": acc, "files": files}


# ---------------------------------------------------------------- 5. atlasing
def atlasing(ref, new_adata, registry_root: str, out: str, report: Report | None = None, *,
             label_keys, batch_key: str | None = None, sample_key: str | None = None,
             dataset: str = "submission", accept: bool = True,
             other_ref=None, ref_adata=None, other_adata=None, level: str | None = None) -> dict:
    """Publish a reference, review a 'pull request' that adds data, and harmonise it
    with a competing reference."""
    from .atlas import ReferenceRegistry, accept_update, harmonize_references, propose_update

    reg = ReferenceRegistry(registry_root)
    reg.publish(ref)
    prop = propose_update(ref, new_adata, label_keys, batch_key=batch_key,
                          sample_key=sample_key, dataset=dataset)
    files = {"review": prop.to_json(_path(out, "update_review.json"))}
    published = accept_update(reg, prop) if accept and prop.report.get("passed") else None
    result = {"registry": reg, "proposal": prop, "published": published, "files": files}
    if other_ref is not None:
        h = harmonize_references(ref, other_ref, ref_adata, other_adata, level)
        files["harmonization"] = write_table(h.table, _path(out, "harmonization.tsv"))
        result["harmonization"] = h
    if report is not None:
        r = prop.report
        report.section("Open-source atlasing: versioned references and 'pull requests'")
        report.text(f"Proposed update of <b>{ref.name}</b> v{ref.version} → "
                    f"<b>v{prop.candidate.version}</b> ({prop.bump} bump). Review "
                    f"{'passed' if r.get('passed') else 'FAILED'}; "
                    f"{'published' if published else 'not published'}.")
        report.json(r, "review report")
        report.table(reg.history(ref.name).reset_index(drop=True))
        if "harmonization" in result:
            report.text("Harmonisation of two independently annotated references "
                        "(reference cell tree).")
            report.table(result["harmonization"].table)
            report.json(result["harmonization"].tree, "harmonised label tree")
    return result


# ---------------------------------------------------------------- 6. compression
def compression(ref, adata, out: str, report: Report | None = None, *, n_sketch: int = 1000,
                cells_per_metacell: int = 20, rare_label: str | None = None,
                seed: int = 0) -> dict:
    from .compression import geometric_sketch, metacells

    fine = ref.label_keys[-1]
    lab = ref.labels(fine)
    sk = geometric_sketch(ref.Z_corr, n_sketch, seed=seed)
    rng = np.random.default_rng(seed)
    uni = rng.choice(ref.n_cells, len(sk), replace=False)
    comp = pd.DataFrame({"full": pd.Series(lab).value_counts(normalize=True),
                         "geometric_sketch": pd.Series(lab[sk]).value_counts(normalize=True),
                         "uniform": pd.Series(lab[uni]).value_counts(normalize=True)}).fillna(0)
    mc = metacells(ref.Z_corr, cells_per_metacell=cells_per_metacell,
                   by=ref.obs[ref.batch_key].to_numpy() if ref.batch_key else None,
                   adata=adata, label_keys=ref.label_keys, seed=seed)
    files = {"sketch_composition": write_table(comp, _path(out, "sketch_composition.tsv"), True)}
    if report is not None:
        report.section("Data compression for very large references")
        report.text(f"Geometric sketch of {len(sk):,} cells vs uniform sampling (cell-type "
                    f"fractions); {mc.n_metacells:,} metacells of ~{cells_per_metacell} cells, "
                    f"fine-label purity {mc.purity(fine):.3f}.")
        report.table(comp.sort_values("full").reset_index(names="label"))
    return {"sketch": sk, "composition": comp, "metacells": mc, "files": files}
