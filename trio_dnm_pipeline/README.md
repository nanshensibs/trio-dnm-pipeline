# Trio DNM Pipeline v2.0

Reference implementation of the *Trio DNM Pipeline Protocol v2.0* (`docs/Trio_DNM_Pipeline_Protocol_v2.0.docx`):
germline de novo mutation (DNM) detection from trio WGS/WES, post-zygotic and parental mosaic
tracks, functional annotation with calibrated ACMG/AMP evidence, biological sanity checks,
mutational signatures, and somatic analysis (tumour/affected tissue vs blood, clonal
haematopoiesis screen).

```
trio_dnm_pipeline/
├── main.nf, nextflow.config     Nextflow DSL2 workflow wiring the standard tools
├── modules/                     qc · germline · dnm · mosaic · somatic
├── trio_dnm/                    pure-Python (3.10+, stdlib only) post-calling engine
├── bin/                         trio-dnm, dng2tsv.py, vcf2mfbed.py (on PATH in Nextflow tasks)
├── conf/                        defaults.json, vcfanno.toml, example samplesheet / gene table
└── tests/                       pytest suite on a synthetic trio
```

## What the Python engine does

| Command | Protocol stage | Outputs |
|---|---|---|
| `trio-dnm qc-gate` | 1 — stop-gates (FREEMIX, somalier relatedness/IBS0, inferred sex, depth) | `qc_gate.json`, exit 1 on failure |
| `trio-dnm call` | 4–7 — Mendelian-violation candidates, 5-layer filter cascade, Bayesian trio posterior, sex-chromosome ploidy, mosaic + parental-mosaic tracks, multi-caller consensus tiers, DNM clustering, raw Mendelian error rate | `candidates.tsv` (every candidate + every failure reason), `dnm.vcf`, `call_summary.json` (waterfall) |
| `trio-dnm annotate` | 6, 8 — VEP CSQ parsing (MANE-first), LOFTEE/NMD, REVEL/AlphaMissense/CADD/BayesDel/PrimateAI/ESM1b/EVE/SpliceAI/phyloP, gene constraint, ClinGen-calibrated PP3/BP4, provisional ACMG with Tavtigian points, priority tiers + ranking score, sanity checks, SBS-96 + signature refit | `annotated.tsv`, `shortlist.tsv`, `qc.json`, `sbs96.tsv`, `report.html` |
| `trio-dnm somatic` | 10 — trio-aware germline subtraction, multi-caller consensus, FFPE flag, CN/LOH, CCF + clonality, TMB, drivers/hotspots, germline + somatic second hits, signatures | `somatic.tsv`, `somatic_summary.json`, `somatic.sbs96.tsv` |
| `trio-dnm ch-screen` | 10.4 — clonal haematopoiesis genes at 2–35 % VAF | `ch_screen.tsv`, `ch_summary.json` |
| `trio-dnm signatures` | any VCF → SBS-96 + NNLS refit | `sbs96.tsv` |
| `trio-dnm config` | print effective thresholds | JSON |

All thresholds live in `trio_dnm/config.py` (written out in `conf/defaults.json`); pass
`--config my.json` to override any subset, and `--data-type wes|panel` for the capture presets.

## Quick start

```bash
cd trio_dnm_pipeline
python -m pytest -q                                  # 18 tests, no dependencies beyond pytest

# post-calling only (you already have a joint, normalised trio VCF):
bin/trio-dnm call --vcf trio.norm.vcf.gz --ped trio.ped --fasta GRCh38.fa \
    --exclude-bed blacklist.bed --exclude-bed GRCh38_alllowmapandsegdup.bed \
    --caller deeptrio=deeptrio.dnm.vcf.gz --caller triodenovo=triodenovo.vcf --out FAM001
vep ... -i FAM001.dnm.vcf -o FAM001.vep.vcf         # see modules/dnm.nf for the full plugin set
bin/trio-dnm annotate --vcf FAM001.vep.vcf --call-summary FAM001.call_summary.json \
    --gene-table genes.tsv --fasta GRCh38.fa --signatures COSMIC_v3.4_SBS_GRCh38.txt \
    --paternal-age 34 --maternal-age 31 --out FAM001

# full workflow:
nextflow run main.nf -profile docker -params-file params.yaml
```

## Inputs you must supply

* **Gene table** (`--gene-table`): columns `gene pLI LOEUF mis_z shet hi_score disease inheritance mechanism cancer_role hotspots`.
  Build it from gnomAD v4.1 constraint, GeneBayes s_het, ClinGen dosage, OMIM/GenCC/G2P, and COSMIC CGC/cancerhotspots.
  `conf/gene_table.example.tsv` shows the format only — its values are illustrative, not curated.
* **Signature matrix** (`--signatures`): COSMIC SBS v3.4 GRCh38 TSV. Germline DNMs are refit to SBS1/SBS5/SBS40 by default.
* **Resource bundle** for Nextflow (`nextflow.config` → `params`): VEP cache + plugin data, gnomAD v4.1 joint sites,
  ClinVar, exclusion BEDs, Mutect2 germline resource/PoN, somalier sites, VerifyBamID2 SVD, MosaicForecast model.

## Status and caveats

* The Python engine is tested end-to-end on a synthetic trio (true DNM, parental leakage, parental mosaic,
  proband mosaic, common variant, masked region, homopolymer, low MQ, inherited, male chrX hemizygous DNM),
  on annotation/ACMG scenarios, and on somatic/CH scenarios.
* The Nextflow workflow has **not been executed** (Nextflow was not available when it was written). Every
  process has a `stub:` block, so `nextflow run main.nf -stub -profile test` is the first thing to run to check
  channel wiring; then validate on GIAB trio HG002/HG003/HG004 before production use.
* Container tags in `nextflow.config` are indicative. Verify each exists and pin by digest.
* The DeNovoGear output parser (`bin/dng2tsv.py`) accepts several field-name variants; confirm against your build.
* ACMG classes are **provisional**: only annotation-derivable criteria are automated. Phenotype specificity
  (ClinGen de novo points), segregation, functional evidence and the full PVS1 decision tree need expert curation.
* AlphaMissense ClinGen thresholds (Bergquist et al. 2025) are included as configurable values; verify them against
  the publication before clinical use. The default single PP3/BP4 tool is REVEL (Pejaver et al. 2022).
