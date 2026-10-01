# Trio DNM Pipeline v2.0

Reference implementation of the *Trio DNM Pipeline Protocol v2.0* (`docs/Trio_DNM_Pipeline_Protocol_v2.0.docx`):
germline de novo mutation (DNM) detection from trio WGS/WES, post-zygotic and parental mosaic
tracks, functional annotation with calibrated ACMG/AMP evidence, biological sanity checks,
mutational signatures, and somatic analysis (tumour/affected tissue vs blood, clonal
haematopoiesis screen).

```
trio_dnm_pipeline/
├── main.nf, nextflow.config     Nextflow DSL2 workflow wiring the standard tools
├── modules/                     align · qc · germline · dnm · mosaic · somatic
├── trio_dnm/                    pure-Python (3.10+, stdlib only) post-calling engine
├── bin/                         trio-dnm, dng2tsv.py, vcf2mfbed.py (on PATH in Nextflow tasks)
├── conf/                        defaults.json, vcfanno.toml, example samplesheet / gene table
└── tests/                       pytest suite on a synthetic trio
```

## What the Python engine does

Stage and section numbers are those of the protocol document.

| Command | Protocol stage (§) | Outputs |
|---|---|---|
| `trio-dnm qc-gate` | 1 (§3) — stop-gates: FREEMIX, somalier relatedness/IBS0, inferred sex of all three members (unknown = WARN), depth, LCL DNA flag (`--lcl`) | `qc_gate.json`, exit 1 on failure |
| `trio-dnm call` | 4, 5, 7 (§6, §7, §9) — Mendelian-violation candidates (plus DeepTrio-only ones via `--extra-vcf`, flagged `SECOND_ENGINE_ONLY`), layered filter cascade (FILTER, strand counts from `--strand-vcf`, masks, gnomAD/PoN), Bayesian trio posterior, sex-chromosome ploidy, mosaic + parental-mosaic tracks, multi-caller consensus tiers, DNM clustering, raw Mendelian error rate with stop-gate | `candidates.tsv` (every candidate + every failure reason), `dnm.vcf`, `call_summary.json` (waterfall); exit 3 when the Mendelian-error gate fails (`--no-halt` to override) |
| `trio-dnm annotate` | 6, 8 (§8, §10) — VEP CSQ parsing (MANE-first), LOFTEE/NMD, REVEL/AlphaMissense/CADD/BayesDel/PrimateAI/ESM1b/EVE/SpliceAI/phyloP, gene constraint, ClinGen-calibrated PP3/BP4, provisional ACMG with Tavtigian points, priority tiers + ranking score, sanity checks, SBS-96 + signature refit | `annotated.tsv`, `shortlist.tsv`, `qc.json`, `sbs96.tsv`, `report.html` |
| `trio-dnm somatic` | 9 (§11) — trio-aware germline subtraction, multi-caller consensus, FFPE flag, CN/LOH, CCF + clonality, TMB, drivers/hotspots, germline + somatic two-hit detection, signatures | `somatic.tsv`, `two_hit.tsv`, `somatic_summary.json`, `somatic.sbs96.tsv` |
| `trio-dnm ch-screen` | 9 (§11.7) — clonal haematopoiesis screen: CH genes at 2–35 % VAF | `ch_screen.tsv`, `ch_summary.json` |
| `trio-dnm signatures` | 6, 8, 9 (§8.3, §10.10, §11.4) — any VCF → SBS-96 + NNLS refit | `sbs96.tsv` |
| `trio-dnm demo` | run every stage on a small simulated trio | `demo/inputs/`, `demo/results/`, `README.txt` |
| `trio-dnm config` | print effective thresholds | JSON |

The filter, gate and classification thresholds live in `trio_dnm/config.py` (written out in
`conf/defaults.json`), including the Stage-1 QC gates and the Mendelian-error gate (section `qc`).
`call`, `annotate`, `somatic`, `ch-screen`, `qc-gate` and `config` accept `--config my.json`. Put only
the keys you change in that file: values equal to the defaults are ignored, so they do not override the
`--data-type wes|panel` presets. A few rule constants stay in code: the Tavtigian point-to-class
boundaries, the BP7 phyloP cut-off (< 0.1), the AlphaMissense ranking cut-off (≥ 0.564, ranking and
Tier 2 only) and the ranking-score weights.

## Quick start

```bash
cd trio_dnm_pipeline
bin/trio-dnm demo --out demo                         # every stage on a SIMULATED trio; open demo/results/F1.report.html
python -m pytest -q                                  # the test suite; no dependencies beyond pytest

# post-calling only. trio.anno.vcf.gz = joint, normalised trio VCF annotated by vcfanno
# (conf/vcfanno.toml: gnomAD_AF, PON_AF, ClinVar); without gnomAD INFO, `call` warns and
# records population_af_missing. Strand counts come from a pileup of the candidate sites
# (proband carries ALT, both parents hom-ref; same recipe as CANDIDATE_SITES in modules/dnm.nf):
bcftools view -s proband,father,mother trio.anno.vcf.gz -Ou \
  | bcftools view -i 'GT[0]!="RR" && GT[0]!="mis" && GT[1]="RR" && GT[2]="RR"' -Ou \
  | bcftools query -f '%CHROM\t%POS\n' > cand_sites.tsv
# Sample names in the pileup must equal the PED IDs: if the BAM @RG SM tags differ, pass
# -G read_groups.txt (lines "* <bam> <PED id>"), as MPILEUP_CANDIDATES does.
bcftools mpileup -f GRCh38.fa -R cand_sites.tsv -a AD,ADF,ADR,DP -Oz -o FAM001.strand.vcf.gz \
    proband.bam father.bam mother.bam
bin/trio-dnm call --vcf trio.anno.vcf.gz --ped trio.ped --fasta GRCh38.fa \
    --exclude-bed blacklist.bed --exclude-bed GRCh38_alllowmapandsegdup.bed \
    --strand-vcf FAM001.strand.vcf.gz --extra-vcf deeptrio.anno.vcf.gz \
    --caller deeptrio=deeptrio.dnm.vcf.gz --caller triodenovo=triodenovo.vcf --out FAM001
vep ... -i FAM001.dnm.vcf -o FAM001.vep.vcf         # see modules/dnm.nf for the full plugin set
bin/trio-dnm annotate --vcf FAM001.vep.vcf --call-summary FAM001.call_summary.json \
    --gene-table genes.tsv --fasta GRCh38.fa --signatures COSMIC_v3.4_SBS_GRCh38.txt \
    --paternal-age 34 --maternal-age 31 --out FAM001

# workflow: stub dry runs on bundled placeholder inputs (BAM input, then FASTQ input via modules/align.nf),
# then a real run
nextflow run main.nf -stub -profile test_stub
nextflow run main.nf -stub -profile test_stub -params-file tests/nextflow/params_fastq.yaml
nextflow run main.nf -profile docker -params-file params.yaml   # your paths for the params in nextflow.config
```

## Inputs you must supply

* **Samplesheet** (`params.samplesheet`): one row per sample with `family_id, sample_id, role, sex` and either an
  analysis-ready BAM/CRAM or paired FASTQ (aligned by `modules/align.nf`: fastp + FastQC, BWA-MEM2 with a full
  `@RG`, MarkDuplicates, BQSR). An optional `dna_source=lcl` marks lymphoblastoid cell-line DNA (QC WARN).
  `conf/samplesheet.example.csv` shows the columns.
* **Gene table** (`--gene-table`): columns `gene pLI LOEUF mis_z shet hi_score disease inheritance mechanism cancer_role hotspots`.
  Build it from gnomAD v4.1 constraint, GeneBayes s_het, ClinGen dosage, OMIM/GenCC/G2P, and COSMIC CGC/cancerhotspots.
  Hotspots may use one- or three-letter protein notation (`p.R132H` or `p.Arg132His`).
  `conf/gene_table.example.tsv` shows the format only — its values are illustrative, not curated.
* **Signature matrix** (`--signatures`): COSMIC SBS v3.4 GRCh38 TSV. Germline DNMs are refit to SBS1 + SBS5 + SBS40a
  by default. COSMIC v3.4 split SBS40 into SBS40a/b/c; a requested name missing from the matrix is resolved with a
  warning (SBS40 → SBS40a).
* **Resource bundle** for Nextflow: one `params` entry per resource in `nextflow.config` (VEP cache + plugin data,
  gnomAD v4.1 joint sites with `AF_joint`/`AF_grpmax_joint`, ClinVar, exclusion BEDs, Mutect2 germline resource/PoN,
  somalier sites, VerifyBamID2 SVD, MosaicForecast model, ...), set in your `-params-file`. Every resource (with any
  index next to it) is staged into the tasks as a `path` input, so Nextflow makes it visible inside Docker or
  Singularity containers and no bind mounts are needed — do **not** add `-v` mounts to `docker.runOptions`
  (Docker aborts on duplicate mount points). `trio-dnm` tasks stage the `trio_dnm` package into the task
  directory and put it on `PYTHONPATH`, so they run in any python ≥ 3.10 image (default `python:3.11`) or in the
  image built from the provided `Dockerfile` (see the header of `nextflow.config`).

Pipeline provenance (Nextflow trace, timeline, report, DAG and a provenance JSON) is written to
`results/pipeline_info/`; a MultiQC report aggregates the QC at the end of the run.

## Status and caveats

* **Verified**: the test suite (`python -m pytest -q`), a clean-room `pip install` of the package, the
  `trio-dnm demo` run, and an independent static review of the protocol, engine and workflow against current
  tool documentation and the literature. The tests cover a synthetic trio (true DNM, parental leakage, parental
  mosaic, proband mosaic, common variant, masked region, homopolymer, low MQ, inherited, male chrX hemizygous DNM),
  annotation/ACMG scenarios, somatic/CH scenarios and the QC gates.
  The Nextflow wiring is verified by stub dry runs on Nextflow 24.10.4 — BAM input (51 tasks) and FASTQ input via
  `modules/align.nf` (114 tasks) both complete without failures; `tests/test_bin_fixes.py` repeats them whenever
  `nextflow` is on `PATH`.
* **Not verified**: the workflow has **never been executed with the real tools on real data**. Validate it on GIAB
  trio HG002/HG003/HG004 (DNM count and
  spectrum, Mendelian error rate, strand-pileup matching, DeepTrio-only candidates, DeNovoGear on the candidate VCF)
  and on a tumour–normal reference pair before production use.
* Container tags in `nextflow.config` are indicative. Verify each exists and pin by digest.
* The DeNovoGear output parser (`bin/dng2tsv.py`) accepts several field-name variants; confirm against your build.
* SpliceAI scores come from Illumina's pre-computed files, which cover ±50 bp; run SpliceAI locally or use the
  SpliceAI-lookup service for ±500 bp. Pangolin is optional and not in the reference VEP command.
* ACMG classes are **provisional**: only annotation-derivable criteria are automated. Phenotype specificity
  (ClinGen de novo points), segregation, functional evidence and the full PVS1 decision tree need expert curation.
* The default single PP3/BP4 tool is REVEL (Pejaver et al. 2022: BP4 Strong ≤ 0.016, Very strong ≤ 0.003).
  AlphaMissense ClinGen thresholds (Bergquist et al. 2025) are included as configurable values; verify them
  against the publication before clinical use.
