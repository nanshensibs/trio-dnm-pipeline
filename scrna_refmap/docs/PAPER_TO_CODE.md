# Paper → code map

Each row lists something the perspective (Lotfollahi, Hao, Theis & Satija, *Cell*
187:2343, 2024) proposes or cites, the code that implements it, and how this
implementation differs from the cited tool. Reference numbers are the paper's.

## Reference-mapping workflows (Figure 1A–B)

| Paper | Code | Notes / deviations |
|---|---|---|
| Reference = data transformation + annotations | `reference.Reference`, `reference.build_reference` | Stores genes, scaling, loadings, integrated embedding, Symphony compression, annotations, calibration and provenance. |
| Statistical transformation: PCA (Seurat, ref 21) | `build_reference(transform="pca")` | Batch-aware HVG selection, log-normalisation, scaling clipped at 10, randomized SVD. |
| Supervised PCA (Seurat/Azimuth, ref 6) | `build_reference(transform="spca")` | Supervision is a label-restricted kNN graph, not the WNN graph used in Azimuth. |
| Symphony: PCA + soft clusters + batch correction (ref 5) | `harmony.run_harmony`, `harmony.compress_reference`, `harmony.symphony_map` | Harmony objective, diversity penalty, MoE ridge correction; Symphony's reference-anchored query correction from `Nr` and `C`. |
| Deep generative mapping (scArches, ref 3) | — | Not implemented: torch is unavailable here and the paper names linear methods as equally valid. The kNN transfer and uncertainty follow scArches. |
| Apply the *same* transformation to the query | `Reference.project`, `mapping.map_query` | Missing reference genes are zero-filled; a warning is raised above 20 % missing. |
| Transfer discrete labels via neighbours | `transfer.knn_weights`, `transfer.transfer_labels` | Same Gaussian kernel as scArches `weighted_knn_transfer`. |
| Hierarchical annotations; hierarchical classifiers (refs 30–31) | `transfer.transfer_hierarchical` | Coarse → fine, consistent with the hierarchy; an unresolved fine label falls back to its parent. |
| Transfer continuous information (Figure 1C, third row) | `transfer.transfer_continuous`, `build_reference(continuous_obsm=...)` | Weighted kNN mean ± SD (spatial coordinates, protein, chromatin features…). |
| No re-clustering or manual annotation | `pipeline.contextualize`, `refmap map` | — |

## Identifying disease states (Figure 1C)

| Paper | Code | Notes / deviations |
|---|---|---|
| Three success criteria | `metrics.evaluate_mapping` | (1) accuracy / F1 / silhouette; (2) kNN ref–query mixing **and** centroid offset; (3) novelty AUROC and cluster-level recall/precision. |
| Uncertainty of label transfer (refs 3, 29) | `novelty.score_cells` (`uncertainty`) | — |
| Symphony per-cell and per-cluster Mahalanobis distance | `harmony.mahalanobis_to_centroids`, `reference.cluster_mahalanobis` | Per-centroid covariance shrunk towards the global covariance. |
| Out-of-distribution detection (ref 32) | `novelty.score_cells`, `novelty.summarize_clusters` | Calibrated against leave-one-batch-out re-mapping of the reference (`reference.calibrate_reference`). A cluster is novel if (a) it exceeds the median of held-out reference clusters **and** (b) it is a robust outlier among the query's own clusters. |
| Compositional change by cluster (MASC, scCODA; refs 33–34) | `differential.composition_test` | Quasi-binomial GLM per cell type (not Bayesian scCODA). |
| Cluster-free differential abundance (Milo; ref 35) | `differential.milo` | Refined sampling, NB GLM with a likelihood-ratio test (not edgeR QL), spatial FDR. |
| State change within cell types (muscat; ref 37) | `differential.pseudobulk_de` | Pseudobulk, TMM, limma-style moderated t. |
| Prioritising responsive populations (Augur; ref 38) | `differential.augur` | Random forest with subsampling and cross-validated AUC. |

## Population-scale mapping (Figure 2)

| Paper | Code | Notes / deviations |
|---|---|---|
| Cell and sample embeddings (Fig 2B–C) | `population.sample_representation` | Composition plus per-cell-type latent means; a fixed pseudobulk, not learned like MrVI. |
| Sample classification (Fig 2D) | `population.classify_samples` | Logistic regression or random forest; leave-one-out or stratified CV. |
| Multi-instance learning (Box 1) | `population.mil_attribution` | Instance classifier, sample-grouped cross-fitting, per-cell-type attribution. A baseline, not attention-MIL. |
| Sample-similarity map (Fig 2E) | `population.sample_similarity` | Correlation plus hierarchical ordering. |

## Perturbation atlases

| Paper | Code | Notes / deviations |
|---|---|---|
| Predict unseen responses with latent arithmetic (scGen; ref 52) | `perturbation.PerturbationModel`, `perturbation.evaluate_holdout` | The latent space is the reference PCA, not a VAE; decoding via `Reference.reconstruct`. |
| Interpret query signatures against an atlas (CMap/LINCS idea) | `perturbation.query_response`, `perturbation.match_signatures` | Cosine similarity per transferred cell type. |
| CPA, GEARS (refs 54, 56) | — | Not implemented: they require deep learning. |

## Across modalities (Figure 3)

| Paper | Code | Notes / deviations |
|---|---|---|
| Feature conversion: gene activity, gene body + 2 kb (Cicero; ref 65) | `crossmodal.gene_activity` | Strand-aware. |
| CCA anchors (Seurat v3; ref 21) | `crossmodal.cca_anchor_transfer`, `crossmodal.gene_activity_transfer` | Anchor filtering and scoring approximate Seurat's. |
| Multi-omic bridge, dictionary learning (Seurat v5; ref 71) | `crossmodal.bridge_integration` | Least-squares dictionary over bridge cells; Laplacian eigenmaps of a union RNA/ATAC bridge graph (simplifies WNN). |
| LSI for ATAC | `crossmodal.lsi_fit`, `crossmodal.tfidf` | Drops the first component when it correlates with depth. |
| LIGER, MultiMAP, GLUE, MultiVI, BABEL, StabMap | — | Not implemented. |

## Across species

| Paper | Code | Notes / deviations |
|---|---|---|
| Ortholog-based alignment with CCA (ref 93) | `crossspecies.convert_orthologs`, `crossspecies.map_across_species` | Species-specific types surface as low-score clusters. |
| Projection through an existing reference | `crossspecies.project_across_species` | Labels species-specific cells confidently; prefer anchor scores for detecting new types. |
| SATURN, protein language models (refs 98–99) | — | Not implemented. |

## Open-source atlasing

| Paper | Code | Notes / deviations |
|---|---|---|
| Semantic versioning of references (v1.0.0 → v1.1.0) | `atlas.ReferenceRegistry`, `atlas.bump_version` | Index with content hashes, parent version and changelog. |
| "Pull request" for a reference update | `atlas.propose_update`, `atlas.accept_update` | Maps the submission first, then rebuilds, reviews label stability and accuracy, and bumps patch, minor or major. |
| Reference cell tree / harmonising competing atlases (ref 113) | `atlas.harmonize_references` | Reciprocal mapping; equivalent, split, merge and unmatched classes; nested tree. |
| Data compression: metacells (refs 98, 111) | `compression.metacells` | k-means within batch, not SEACells archetypes. |
| Data compression: sketching (ref 110) | `compression.geometric_sketch` | Plaid covering with binary search over the box side. |
| Data compression: chunking | `compression.incremental_reference` | Streaming moments plus IncrementalPCA. |
