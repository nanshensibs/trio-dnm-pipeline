# refmap: automated single-cell analysis by reference mapping

A Python pipeline that implements the reference-mapping workflows proposed in:

> Lotfollahi M, Hao Y, Theis FJ, Satija R. **The future of rapid and automated single-cell
> data analysis using reference mapping.** *Cell* 187:2343–2358 (2024).
> doi:10.1016/j.cell.2024.03.009

The paper is a Perspective with no single algorithm of its own. `refmap` therefore
implements each workflow it describes with the methods it cites. These are listed in
[`docs/PAPER_TO_CODE.md`](docs/PAPER_TO_CODE.md), and the paper itself is analysed in
[`docs/PAPER_ANALYSIS.md`](docs/PAPER_ANALYSIS.md).

```
build reference ──► map query ──► annotate (hierarchical labels + uncertainty)
 (HVG, PCA/sPCA,     (same          ├─► impute continuous values (spatial, protein, ATAC)
  Harmony,           transform,     ├─► flag novel / disease states (Figure 1C)
  Symphony           Symphony       ├─► composition · Milo DA · pseudobulk DE · Augur
  compression,       correction,    ├─► sample embeddings · classification · MIL (Figure 2)
  calibration)       kNN)           └─► report.html + tables + annotated .h5ad
 + versioned registry, "pull-request" updates, atlas harmonisation, sketching/metacells
 + perturbation atlases · scATAC→scRNA (gene activity, bridge) · cross-species
```

## Install and run

```bash
cd scrna_refmap
pip install -e .              # numpy scipy pandas scikit-learn anndata statsmodels networkx matplotlib
pip install umap-learn        # optional: UMAP figures (otherwise latent dims are plotted)

refmap demo --out demo        # all 7 workflows on SIMULATED data -> demo/report.html
pytest -q                     # 48 tests, about 40 s
```

### Your own data

Inputs are `.h5ad` files with raw counts in `.X` (or `--layer`).

```bash
# 1. build a reference (labels coarse -> fine; every fine label must have one parent)
refmap build --h5ad atlas.h5ad --labels cell_type_l1,cell_type_l2 \
    --batch-key dataset --sample-key donor --name lung-atlas --version 1.0.0 --out ref/

# 2. map a query (disease vs control) -> annotation, novel states, DA, DE, Augur, report
refmap map --reference ref/ --query study.h5ad --out results/ \
    --sample-key donor --condition-key status --case disease --control healthy

# 3. population-scale: classify query donors and find the populations that drive it
refmap population --reference ref/ --query cohort.h5ad --out pop/ \
    --sample-key donor --phenotype-key subtype

# 4. versioned atlasing
refmap registry publish --root registry/ --reference ref/
refmap registry propose-update --root registry/ --name lung-atlas --h5ad new_lab.h5ad \
    --labels cell_type_l1,cell_type_l2 --batch-key dataset --review review.json --accept

# also available: perturbation, cross-modality, cross-species, sketch, info
refmap <command> -h
```

From Python:

```python
from refmap.reference import build_reference
from refmap.mapping import map_query, final_labels
ref = build_reference(atlas, ["cell_type_l1", "cell_type_l2"], batch_key="dataset")
res = map_query(ref, query, condition_key="status")
res.annotate(query)                       # obs columns + obsm["X_refmap"]
labels = final_labels(res, "cell_type_l2")  # novel clusters relabelled "Novel"
```

## Outputs of `refmap map`

| File | Content |
|---|---|
| `mapping.tsv` | per cell: label per level, uncertainty, resolved flag, kNN distance, Mahalanobis distance, empirical p-values, OOD flag, query cluster |
| `query_clusters.tsv` | per query cluster: top label, uncertainty, cluster Mahalanobis, robust z, condition fractions, `novel` |
| `imputed_<name>.tsv` | continuous reference values transferred to query cells |
| `composition.tsv`, `milo_nhoods.tsv`, `pseudobulk_de.tsv`, `augur.tsv` | condition effects |
| `metrics.json` | the paper's three criteria (when ground truth is supplied) |
| `query_mapped.h5ad` | query with all annotations and the joint embedding |
| `report.html` | self-contained report with figures |

## Validation on simulated data (`refmap demo`, seed 0)

| Workflow | Result |
|---|---|
| Label transfer onto a 4-lab reference | 100 % fine and coarse accuracy on known cell types |
| Novel disease state (absent from the reference) | its cluster is flagged; 97 % recall, 100 % precision |
| Novel-cluster rule across 20 simulated settings | novel cluster flagged in 20/20; one extra (known, rare) cluster flagged in 1/20 |
| Composition / Milo / pseudobulk / Augur | CD14 Mono ↑, naive B ↓, novel neighbourhoods ↑; interferon genes found in CD8 T and CD14 Mono; CD8 T and CD14 Mono top in Augur (AUC ≈ 0.99) |
| Population (8 query donors) | 8/8 subtypes correct; MIL ranks CD14 Mono first |
| Perturbation | true drug ranked first (cosine 0.98); held-out cell type response-gene R² 0.996 vs 0.93 for no change |
| scATAC → scRNA | fine-label accuracy 0.74 (gene activity + CCA anchors) vs 0.94 (multi-omic bridge) |
| Cross-species | 100 % on shared types; the species-specific type is the lowest-scoring cluster |
| Atlas "pull request" | 1.0.0 → 1.1.0 (minor), new label reported, review passed |
| Geometric sketch | rare DC fraction 9.6 % vs 4.8 % with uniform sampling |

## Known limitations

These are documented in [`docs/PAPER_ANALYSIS.md`](docs/PAPER_ANALYSIS.md):

* **Novel cells are hard to detect one at a time.** Disease programmes made of genes
  that never vary in a healthy reference are weakly visible in its latent space, which
  is the paper's own caveat (ref 28). Per-cell AUROC is about 0.8, so novelty is decided
  per query cluster, and matched controls plus Milo give the strongest signal.
* **Projected queries look less mixed than they are.** They are narrower than the
  reference, so the kNN mixing score is about 0.4 even when centroids align (offset
  about 0.12 of the within-type spread).
* **Not implemented:** scArches/scVI and the other deep-learning methods (no torch here),
  CPA/GEARS, LIGER/GLUE/MultiVI, SATURN.
* **Validation is on simulated data only.** Thresholds are calibrated from each
  reference, but real atlases should be checked against held-out author annotations
  with `--truth`.
