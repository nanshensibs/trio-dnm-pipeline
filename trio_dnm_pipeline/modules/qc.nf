// Stage 1 (protocol §3) — pre-flight sample QC with hard stop-gates, plus the
// run-level MultiQC report (§3.1) and provenance records (§13.1).

include { asList ; trioDnmEnv } from './utils'

process MOSDEPTH {
    tag "$id"
    label 'small'
    container params.containers.mosdepth
    input:
    tuple val(fam), val(id), val(role), val(sex), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    path(targets)
    output:
    tuple val(fam), val(id), path("${id}.mosdepth.summary.txt"), emit: summary
    path("${id}.mosdepth.*.dist.txt"), emit: dist
    script:
    def by = targets ? "--by ${targets}" : ''
    """
    mosdepth -t ${task.cpus} -n --fast-mode -f ${fasta} ${by} ${id} ${bam}
    """
    stub:
    """
    printf 'chrom\\tlength\\tbases\\tmean\\tmin\\tmax\\ntotal\\t1\\t30\\t30\\t0\\t60\\n' > ${id}.mosdepth.summary.txt
    touch ${id}.mosdepth.global.dist.txt
    """
}

process VERIFYBAMID2 {
    tag "$id"
    label 'small'
    container params.containers.verifybamid2
    input:
    tuple val(fam), val(id), val(role), val(sex), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    path(svd)   // <prefix>.UD/.mu/.bed(/.V)
    output:
    tuple val(fam), val(id), path("${id}.selfSM"), emit: selfsm
    script:
    def prefix = asList(svd).find { it.name.endsWith('.UD') }.name - ~/\.UD$/
    """
    verifybamid2 --SVDPrefix ${prefix} --Reference ${fasta} \\
        --BamFile ${bam} --NumThread ${task.cpus} --Output ${id}
    """
    stub:
    "printf '#SEQ_ID\\tRG\\tCHIP_ID\\t#SNPS\\t#READS\\tAVG_DP\\tFREEMIX\\n${id}\\tNA\\tNA\\t1\\t1\\t30\\t0.001\\n' > ${id}.selfSM"
}

process SOMALIER_EXTRACT {
    tag "$id"
    label 'small'
    container params.containers.somalier
    input:
    tuple val(fam), val(id), val(role), val(sex), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    path(sites)
    output:
    tuple val(fam), path("extracted/${id}.somalier"), emit: somalier
    script:
    """
    somalier extract -d extracted/ --sites ${sites} -f ${fasta} ${bam}
    """
    stub:
    "mkdir -p extracted && touch extracted/${id}.somalier"
}

process WRITE_PED {
    tag "$fam"
    label 'small'
    container params.containers.trio_dnm
    input:
    tuple val(fam), val(kid), val(dad), val(mom)
    output:
    tuple val(fam), path("${fam}.ped"), emit: ped
    script:
    "printf '%s\\n' ${pedLines(fam, kid, dad, mom).collect { "'${it}'" }.join(' ')} > ${fam}.ped"
    stub:
    "printf '%s\\n' ${pedLines(fam, kid, dad, mom).collect { "'${it}'" }.join(' ')} > ${fam}.ped"
}

process SOMALIER_RELATE {
    tag "$fam"
    label 'small'
    container params.containers.somalier
    publishDir "${params.outdir}/${fam}/qc", mode: params.publish_dir_mode
    input:
    tuple val(fam), path(ped), path(somaliers)
    output:
    tuple val(fam), path("${fam}.somalier.pairs.tsv"), path("${fam}.somalier.samples.tsv"), emit: tsv
    path "${fam}.somalier.html", optional: true
    script:
    """
    somalier relate --ped ${ped} -o ${fam}.somalier ${somaliers}
    """
    stub:
    "touch ${fam}.somalier.pairs.tsv ${fam}.somalier.samples.tsv ${fam}.somalier.html"
}

process QC_GATE {
    tag "$fam"
    label 'small'
    container params.containers.trio_dnm
    errorStrategy 'terminate'   // a failed stop-gate halts the whole run
    publishDir "${params.outdir}/${fam}/qc", mode: params.publish_dir_mode
    input:
    tuple val(fam), path(ped), path(pairs), path(samples), path(selfsms), path(mosdepths), val(kid), val(lcl)
    path(pkg, stageAs: 'pylib/trio_dnm')
    path(cfg)
    output:
    tuple val(fam), path("${fam}.qc_gate.json"), emit: gate
    script:
    def vb = asList(selfsms).collect { "--verifybamid ${it.baseName}=${it}" }.join(' ')
    def md = asList(mosdepths).collect { "--mosdepth ${it.name.replace('.mosdepth.summary.txt', '')}=${it}" }.join(' ')
    def lc = lcl.collect { "--lcl ${it}" }.join(' ')   // dna_source=lcl: proceed with a flag (§3.3)
    def config = cfg ? "--config ${cfg}" : ''
    def halt = params.qc_halt ? '' : '--no-halt'
    """
    ${trioDnmEnv()}
    trio-dnm qc-gate --ped ${ped} --proband ${kid} --data-type ${params.data_type} --build ${params.build} ${config} \\
        --somalier-pairs ${pairs} --somalier-samples ${samples} ${vb} ${md} ${lc} ${halt} --out ${fam}.qc_gate.json
    """
    stub:
    "echo '{\"status\": \"PASS\"}' > ${fam}.qc_gate.json"
}

// Run-level MultiQC over the fastp, FastQC, MarkDuplicates, BQSR, mosdepth, somalier and
// VerifyBamID2 outputs (each file is staged into its own numbered directory).
process MULTIQC {
    label 'small'
    container params.containers.multiqc
    publishDir "${params.outdir}/multiqc", mode: params.publish_dir_mode
    input:
    path(reports, stageAs: '?/*')
    output:
    path "multiqc_report.html", emit: report
    path "multiqc_data", emit: data
    script:
    """
    multiqc --force .
    """
    stub:
    "mkdir -p multiqc_data && touch multiqc_report.html"
}

// Provenance (§13.1): effective trio-dnm thresholds and reference checksums. The
// Nextflow params and run metadata are written by workflow.onComplete (main.nf).
process PROVENANCE {
    label 'small'
    container params.containers.trio_dnm
    publishDir "${params.outdir}/pipeline_info", mode: params.publish_dir_mode
    input:
    path(pkg, stageAs: 'pylib/trio_dnm')
    path(cfg)
    tuple path(fasta), path(fai), path(dict)
    output:
    path "trio_dnm.effective_config.json"
    path "reference.sha256"
    script:
    def config = cfg ? "--config ${cfg}" : ''
    """
    ${trioDnmEnv()}
    trio-dnm config ${config} --data-type ${params.data_type} --build ${params.build} > trio_dnm.effective_config.json
    sha256sum ${fasta} ${fai} ${dict} > reference.sha256
    """
    stub:
    "echo '{}' > trio_dnm.effective_config.json && touch reference.sha256"
}

// PLINK PED lines (tab-separated) of one trio; the proband is affected.
def pedLines(fam, kid, dad, mom) {
    def sx = { s -> s == 'male' ? '1' : (s == 'female' ? '2' : '0') }
    [
        "${fam}\t${kid.id}\t${dad.id}\t${mom.id}\t${sx(kid.sex)}\t2",
        "${fam}\t${dad.id}\t0\t0\t1\t1",
        "${fam}\t${mom.id}\t0\t0\t2\t1",
    ].collect { it.toString() }
}

// Statuses in a qc_gate.json (one object, or a list when several trios were evaluated).
def gateStatuses(json) {
    def j = new groovy.json.JsonSlurper().parseText(json.text)
    (j instanceof List ? j : [j]).collect { it.status }
}

workflow SAMPLE_QC {
    take:
    samples     // (fam, id, role, sex, bam, bai)
    trios       // (fam, proband, father, mother) maps
    ref
    res         // staged resources (main.nf)
    main:
    md = MOSDEPTH(samples, ref, res.targets)
    vb = VERIFYBAMID2(samples, ref, res.verifybamid_svd)
    sm = SOMALIER_EXTRACT(samples, ref, res.somalier_sites)
    ped = WRITE_PED(trios.map { fam, k, d, m -> tuple(fam, k, d, m) })
    rel = SOMALIER_RELATE(ped.ped.join(sm.somalier.groupTuple(size: 3)))
    gate_in = ped.ped
        .join(rel.tsv)
        .join(vb.selfsm.map { fam, id, f -> tuple(fam, f) }.groupTuple(size: 3))
        .join(md.summary.map { fam, id, f -> tuple(fam, f) }.groupTuple(size: 3))
        .join(trios.map { fam, k, d, m -> tuple(fam, k.id, [k, d, m].findAll { it.dna_source == 'lcl' }.collect { it.id }) })
    gate = QC_GATE(gate_in, res.pkg, res.dnm_config)
    // Only trios whose gate is not FAIL proceed (with qc_halt=false a FAIL is reported and the trio skipped).
    passed = trios.join(ped.ped).join(gate.gate)
        .filter { fam, k, d, m, pedf, js ->
            def ok = !('FAIL' in gateStatuses(js))
            if (!ok) log.warn "[${fam}] Stage-1 QC gate FAIL with qc_halt=false: trio skipped (see ${fam}/qc/${fam}.qc_gate.json)"
            ok
        }
        .map { fam, k, d, m, pedf, js -> tuple(fam, k, d, m, pedf) }
    mqc = md.summary.map { fam, id, f -> f }
        .mix(md.dist, vb.selfsm.map { fam, id, f -> f }, rel.tsv.flatMap { fam, p, s -> [p, s] })
    emit:
    passed_trios = passed   // (fam, proband, father, mother, ped)
    multiqc = mqc           // report files for MULTIQC
}
