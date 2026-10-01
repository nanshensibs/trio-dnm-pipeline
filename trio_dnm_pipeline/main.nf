#!/usr/bin/env nextflow
/*
 * Trio DNM pipeline v2.0 — germline DNM, mosaic, annotation and somatic tracks.
 *
 *   nextflow run main.nf -profile docker -params-file params.yaml
 *   nextflow run main.nf -stub -profile test_stub        # wiring check on dummy inputs
 *
 * Samplesheet (CSV with header):
 *   family_id,sample_id,role,sex[,dna_source][,library][,lane],bam  |  ...,fastq_1,fastq_2
 *   role       ∈ proband|father|mother|tumor|tissue ; sex ∈ male|female|unknown
 *   dna_source ∈ blood|saliva|lcl|tissue (optional; lcl = lymphoblastoid cell line, flagged
 *                by the Stage-1 gate and excluded from the CH screen)
 *   bam        = analysis-ready (dedup + BQSR) BAM/CRAM with its index alongside
 *                (<bam>.bai or <name>.bai; <cram>.crai or <name>.crai); one row per sample.
 *   fastq_1/2  = gzipped paired-end FASTQ, one row per lane/unit (optional `lane` names the
 *                unit, `library` the @RG LB). These samples go through the ALIGN subworkflow
 *                (modules/align.nf, protocol Stage 2: FastQC/fastp, BWA-MEM2, MarkDuplicates,
 *                BQSR); needs params.known_sites and a BWA-MEM2 index next to params.fasta
 *                (or params.bwamem2_index).
 *   Every trio needs exactly one proband, father and mother; sample_id must match the
 *   @RG SM tag of BAM/CRAM inputs. Relative paths are resolved against the samplesheet's
 *   directory.
 *
 * Reference resources are passed to the processes as staged `path` inputs (see `res`
 * below and nextflow.config), so they are visible inside Docker/Singularity containers.
 */
nextflow.enable.dsl = 2

include { ALIGN                                       } from './modules/align'
include { SAMPLE_QC ; MULTIQC ; PROVENANCE            } from './modules/qc'
include { GERMLINE_CALLING                            } from './modules/germline'
include { DNM_DETECTION ; DNM_ANNOTATION              } from './modules/dnm'
include { MOSAIC_TRACK                                } from './modules/mosaic'
include { SOMATIC_PAIRED ; CH_SCREEN                  } from './modules/somatic'

// --------------------------------------------------------------------------- //
// Inputs

def required(String name, String purpose) {
    if (!params[name]) error "params.${name} is required ${purpose}"
    params[name].toString()
}

// A list param given in a config/params file, or as a comma-separated string on the CLI.
def listParam(v) {
    v instanceof Collection ? (v as List) : (v ? v.toString().tokenize(',')*.trim() : [])
}

// A resource file (or directory) plus the index files that sit next to it.
def withIndex(p) {
    if (!p) return []
    def f = file(p.toString(), checkIfExists: true)
    [f] + ['.tbi', '.csi', '.idx', '.fai', '.gzi'].collect { f.resolveSibling(f.name + it) }.findAll { it.exists() }
}

def optFile(p) {
    p ? file(p.toString(), checkIfExists: true) : []
}

// VerifyBamID2 SVD resource: <prefix>.UD/.mu/.bed (+ .V).
def svdFiles(String prefix) {
    def fs = ['UD', 'mu', 'bed', 'V'].collect { file("${prefix}.${it}") }.findAll { it.exists() }
    if (!(['UD', 'mu', 'bed'].every { ext -> fs.any { it.name.endsWith(".${ext}") } }))
        error "params.verifybamid_svd_prefix: ${prefix}.UD, .mu and .bed are required"
    fs
}

// vcfanno TOML: every `file = "..."` it references is staged (with its index) and the
// TOML text is rewritten to the staged names (vcfanno_res/<name>), so the annotation
// sources are visible in containers and on any executor. Relative paths are resolved
// against the TOML's directory.
def vcfannoConfig(String path) {
    def toml = file(path, checkIfExists: true)
    def staged = []
    def text = toml.text.replaceAll(/(?m)^(\s*file\s*=\s*")([^"]+)(")/) { all, pre, p, post ->
        def src = (p.startsWith('/') || p.contains('://')) ? file(p) : toml.parent.resolve(p)
        staged.addAll(withIndex(src))
        "${pre}vcfanno_res/${src.name}${post}"
    }
    [text, staged]
}

// VEP plugin / --custom data files (with indexes); names resolved again in the VEP process.
def vepFiles() {
    def p = params.vep_plugins ?: [:]
    ['loftee_dir', 'loftee_ancestor', 'loftee_sql', 'cadd_snv', 'cadd_indel', 'revel', 'alphamissense',
     'spliceai_snv', 'spliceai_indel', 'dbnsfp', 'utrannotator', 'gnomad', 'clinvar', 'ccre_bed']
        .findAll { p[it] }.collectMany { withIndex(p[it]) }
}

// BWA-MEM2 index files for params.bwamem2_index (default: the params.fasta prefix).
def bwaIndex() {
    def prefix = (params.bwamem2_index ?: params.fasta).toString()
    ['0123', 'amb', 'ann', 'bwt.2bit.64', 'pac'].collect { file("${prefix}.${it}", checkIfExists: true) }
}

// BAM/CRAM index: <file>.bai|.crai, or the Picard-style <name>.bai|.crai.
def alignmentIndex(bam) {
    def ext = bam.name.endsWith('.cram') ? '.crai' : '.bai'
    def candidates = [bam.resolveSibling(bam.name + ext), bam.resolveSibling(bam.name.replaceAll(/\.(bam|cram)$/, '') + ext)]
    def idx = candidates.find { it.exists() }
    if (!idx) error "no index for ${bam} (expected ${candidates.join(' or ')})"
    idx
}

// Samplesheet -> list of row maps, validated.
def parseSamplesheet(String path) {
    def sheet = file(path, checkIfExists: true)
    def resolve = { String p ->
        file((p.startsWith('/') || p.contains('://')) ? p : sheet.parent.resolve(p).toString(), checkIfExists: true)
    }
    def rows = sheet.splitCsv(header: true, strip: true)
    if (!rows) error "samplesheet ${path}: no rows"
    def missing = ['family_id', 'sample_id', 'role', 'sex'].findAll { !(it in rows[0].keySet()) }
    if (missing) error "samplesheet ${path}: missing column(s) ${missing.join(', ')}"
    def out = rows.withIndex().collect { r, i ->
        def where = "samplesheet ${path} line ${i + 2}"
        def row = [
            fam       : r.family_id, id: r.sample_id, role: r.role?.toLowerCase(),
            sex       : (r.sex ?: 'unknown').toLowerCase(), dna_source: (r.dna_source ?: '').toLowerCase(),
            library   : r.library ?: '', lane: r.lane ?: '',
        ]
        if (!row.fam || !row.id) error "${where}: family_id and sample_id are required"
        if (!(row.role in ['proband', 'father', 'mother', 'tumor', 'tissue'])) error "${where}: role '${r.role}' is not proband|father|mother|tumor|tissue"
        if (!(row.sex in ['male', 'female', 'unknown'])) error "${where}: sex '${r.sex}' is not male|female|unknown"
        if (!(row.dna_source in ['', 'blood', 'saliva', 'lcl', 'tissue'])) error "${where}: dna_source '${r.dna_source}' is not blood|saliva|lcl|tissue"
        if (r.bam && (r.fastq_1 || r.fastq_2)) error "${where}: give either bam or fastq_1/fastq_2, not both"
        if (r.bam) {
            row.bam = resolve(r.bam)
            row.bai = alignmentIndex(row.bam)
        } else if (r.fastq_1 && r.fastq_2) {
            row.fastq_1 = resolve(r.fastq_1)
            row.fastq_2 = resolve(r.fastq_2)
            if (![row.fastq_1, row.fastq_2].every { it.name ==~ /.*\.(fastq|fq)\.gz$/ }) error "${where}: FASTQ files must be gzipped (.fastq.gz / .fq.gz)"
        } else {
            error "${where}: a bam column, or fastq_1 and fastq_2, is required"
        }
        row
    }
    out.groupBy { it.id }.each { id, rs ->
        ['fam', 'role', 'sex', 'dna_source', 'library'].each { k ->
            if (rs*.get(k).unique().size() > 1) error "samplesheet ${path}: sample ${id} has conflicting ${k} values"
        }
        if (rs.any { it.bam } && rs.size() > 1) error "samplesheet ${path}: sample ${id} must have one row when given as BAM/CRAM (or FASTQ rows only)"
        rs.eachWithIndex { row, k -> row.unit = row.lane ?: (k + 1).toString(); row.n_units = rs.size() }
        if (rs*.unit.unique().size() != rs.size()) error "samplesheet ${path}: sample ${id} has duplicate lane values"
    }
    out.groupBy { it.fam }.each { fam, rs ->
        ['proband', 'father', 'mother'].each { role ->
            def n = rs.findAll { it.role == role }*.id.unique().size()
            if (n != 1) error "samplesheet ${path}: family ${fam} needs exactly one ${role} (found ${n})"
        }
    }
    out
}

// --------------------------------------------------------------------------- //

workflow {
    if (!(params.data_type in ['wgs', 'wes', 'panel'])) error "params.data_type must be wgs, wes or panel"
    if (!(params.build in ['GRCh38', 'GRCh37'])) error "params.build must be GRCh38 or GRCh37"
    def fasta_path = required('fasta', '(reference FASTA with .fai and .dict)')
    def sheet = parseSamplesheet(required('samplesheet', '(see the header of main.nf)'))
    def fq_rows = sheet.findAll { it.fastq_1 }
    if (params.data_type != 'wgs' && !params.intervals)
        log.warn "data_type=${params.data_type} without params.intervals: calling and QC run genome-wide"

    def fasta = file(fasta_path, checkIfExists: true)
    def fasta_idx = withIndex(fasta_path).findAll { it.name.endsWith('.fai') || it.name.endsWith('.gzi') }
    if (!fasta_idx.any { it.name.endsWith('.fai') }) error "params.fasta: ${fasta_path}.fai is required"
    def dict = file(fasta_path.replaceAll(/\.(fa|fasta|fna)(\.gz)?$/, '.dict'), checkIfExists: true)
    ref = Channel.value(tuple(fasta, fasta_idx, dict))

    def (toml_text, toml_files) = vcfannoConfig(params.vcfanno_toml.toString())
    def res = [
        pkg              : file("${projectDir}/trio_dnm", checkIfExists: true),   // staged as pylib/trio_dnm
        dnm_config       : optFile(params.dnm_config),
        intervals        : optFile(params.intervals),
        targets          : optFile(params.targets ?: params.intervals),
        somalier_sites   : withIndex(required('somalier_sites', 'for Stage-1 relatedness QC')),
        verifybamid_svd  : svdFiles(required('verifybamid_svd_prefix', 'for Stage-1 contamination QC')),
        vcfanno_toml     : toml_text,
        vcfanno_files    : toml_files,
        exclude_beds     : listParam(params.exclude_beds).collect { file(it, checkIfExists: true) },
        pon              : optFile(params.pon),
        recurrence       : optFile(params.recurrence),
        vep_cache        : file(required('vep_cache', 'for VEP annotation (Stage 8)'), checkIfExists: true),
        vep_plugin_dir   : optFile(params.vep_plugin_dir),
        vep_files        : vepFiles(),
        gene_table       : optFile(params.gene_table),
        cosmic_sbs       : optFile(params.cosmic_sbs),
        validated        : optFile(params.validated),
        germline_resource: [],
        m2_pon           : withIndex(params.m2_pon),
        common_biallelic : [],
        ch_genes_bed     : [],
        mf_umap_bigwig   : [],
        mf_model         : [],
        known_sites      : [],
        bwa_index        : [],
    ]
    if (params.run_mosaic || params.run_somatic || params.run_ch_screen)
        res.germline_resource = withIndex(required('germline_resource', 'for Mutect2 (mosaic, somatic and CH tracks)'))
    if (params.run_mosaic) {
        res.mf_umap_bigwig = file(required('mf_umap_bigwig', 'for MosaicForecast (run_mosaic)'), checkIfExists: true)
        res.mf_model = file(required('mf_model', 'for MosaicForecast (run_mosaic)'), checkIfExists: true)
    }
    if (params.run_somatic) {
        res.common_biallelic = withIndex(required('common_biallelic', 'for GetPileupSummaries (run_somatic)'))
        if (params.data_type != 'wgs' && !res.targets) error "params.targets (or params.intervals) is required for CNVkit on ${params.data_type} data"
    }
    if (params.run_ch_screen)
        res.ch_genes_bed = file(required('ch_genes_bed', 'for the CH screen (run_ch_screen)'), checkIfExists: true)
    if (fq_rows) {
        res.known_sites = listParam(params.known_sites).collectMany { withIndex(it) }
        if (!res.known_sites) error "params.known_sites is required for BQSR of FASTQ samples"
        res.bwa_index = bwaIndex()
    }

    // Stage 2 — FASTQ rows: read QC, alignment, duplicate marking, BQSR
    def bam_samples = sheet.findAll { it.bam }.collect { r ->
        r.subMap(['fam', 'id', 'role', 'sex', 'dna_source']) + [bam: r.bam, bai: r.bai, dt_bam: r.bam, dt_bai: r.bai]
    }
    samples = Channel.fromList(bam_samples)
    mqc = Channel.empty()
    if (fq_rows) {
        aln = ALIGN(
            Channel.fromList(fq_rows).map { r ->
                tuple(r.subMap(['fam', 'id', 'role', 'sex', 'dna_source', 'library', 'unit', 'n_units']), [r.fastq_1, r.fastq_2])
            },
            ref, res)
        samples = samples.mix(aln.samples)
        mqc = mqc.mix(aln.multiqc)
    }

    germ = samples.filter { it.role in ['proband', 'father', 'mother'] }
        .map { s -> tuple(s.fam, s.id, s.role, s.sex, s.bam, s.bai) }

    // family_id -> [proband, father, mother] sample maps
    // (id, sex, dna_source, bam/bai = analysis-ready, dt_bam/dt_bai = BAM for DeepTrio)
    trios = samples.filter { it.role in ['proband', 'father', 'mother'] }
        .map { s -> tuple(s.fam, s) }
        .groupTuple(size: 3)
        .map { fam, members ->
            def by = members.collectEntries { [(it.role): it] }
            tuple(fam, by.proband, by.father, by.mother)
        }

    // Stage 1 — pre-flight QC with hard stop-gates
    qc = SAMPLE_QC(germ, trios, ref, res)
    MULTIQC(mqc.mix(qc.multiqc).collect())
    PROVENANCE(res.pkg, res.dnm_config, ref)

    // Stage 3 — joint trio calling (GATK + DeepTrio), CGP, normalisation, vcfanno
    calls = GERMLINE_CALLING(qc.passed_trios, ref, res)

    // Stages 4/5 — multi-caller DNM detection and the layered filter cascade
    dnm = DNM_DETECTION(calls.trio_vcf, calls.deeptrio_vcf, qc.passed_trios, ref, res)

    // Stage 7 — mosaic-aware track (MosaicForecast on the proband)
    if (params.run_mosaic) {
        MOSAIC_TRACK(qc.passed_trios, ref, res)
    }

    // Stages 6 + 8 — annotation, ACMG, prioritisation, sanity checks, report
    ann = DNM_ANNOTATION(dnm.dnm_vcf, dnm.call_summary, calls.trio_vcf, qc.passed_trios, ref, res)

    // Stage 9 (protocol §11) — somatic analyses vs the proband's blood
    if (params.run_somatic) {
        tumors = samples.filter { it.role in ['tumor', 'tissue'] }
            .map { s -> tuple(s.fam, s.id, s.role, s.bam, s.bai) }
        normals = qc.passed_trios.map { fam, k, d, m, ped -> tuple(fam, k.id, k.bam, k.bai) }
        pairs = tumors.combine(normals, by: 0)
        parent_ids = qc.passed_trios.map { fam, k, d, m, ped -> tuple(fam, d.id, m.id) }
        SOMATIC_PAIRED(pairs, calls.trio_vcf, ann.annotated, parent_ids, ref, res)
    }
    // §11.7 — clonal haematopoiesis screen of the trio members' blood (not LCL / tissue DNA)
    if (params.run_ch_screen) {
        CH_SCREEN(
            qc.passed_trios.flatMap { fam, k, d, m, ped ->
                [k, d, m].findAll { !(it.dna_source in ['lcl', 'tissue']) }.collect { tuple(fam, it.id, it.bam, it.bai) }
            },
            ref, res)
    }
}

// --------------------------------------------------------------------------- //
// Provenance (protocol §13.1): params and run metadata as JSON next to the Nextflow
// trace/timeline/report; PROVENANCE adds the effective trio-dnm config and reference checksums.

def jsonSafe(x) {
    if (x == null || x instanceof Number || x instanceof Boolean) return x
    if (x instanceof Map) return x.collectEntries { k, v -> [(k.toString()): jsonSafe(v)] }
    if (x instanceof Collection) return x.collect { jsonSafe(it) }
    x.toString()
}

workflow.onComplete {
    def info = [
        pipeline  : [name: workflow.manifest.name, version: workflow.manifest.version],
        run       : [
            name        : workflow.runName, session_id: workflow.sessionId, start: workflow.start,
            complete    : workflow.complete, duration: workflow.duration, success: workflow.success,
            exit_status : workflow.exitStatus, error: workflow.errorMessage, resume: workflow.resume,
            stub        : workflow.stubRun, profile: workflow.profile, command_line: workflow.commandLine,
            user        : workflow.userName,
        ],
        source    : [
            project_dir: workflow.projectDir, launch_dir: workflow.launchDir, work_dir: workflow.workDir,
            repository : workflow.repository, revision: workflow.revision, commit_id: workflow.commitId,
            script_id  : workflow.scriptId,
        ],
        nextflow  : [version: workflow.nextflow.version, build: workflow.nextflow.build],
        containers: [engine: workflow.containerEngine, images: params.containers],
        params    : params,
    ]
    def out = file("${params.outdir}/pipeline_info/provenance.json")
    out.parent.mkdirs()
    out.text = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(jsonSafe(info))) + '\n'
}
