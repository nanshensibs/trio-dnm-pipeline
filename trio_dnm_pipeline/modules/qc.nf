// Stage 1 — pre-flight sample QC with hard stop-gates.

process MOSDEPTH {
    tag "$id"
    label 'small'
    container params.containers.mosdepth
    input:
    tuple val(fam), val(id), val(role), val(sex), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), val(id), path("${id}.mosdepth.summary.txt"), emit: summary
    script:
    def by = params.targets ? "--by ${params.targets}" : ''
    """
    mosdepth -t ${task.cpus} -n --fast-mode -f ${fasta} ${by} ${id} ${bam}
    """
    stub:
    "printf 'chrom\\tlength\\tbases\\tmean\\tmin\\tmax\\ntotal\\t1\\t30\\t30\\t0\\t60\\n' > ${id}.mosdepth.summary.txt"
}

process VERIFYBAMID2 {
    tag "$id"
    label 'small'
    container params.containers.verifybamid2
    input:
    tuple val(fam), val(id), val(role), val(sex), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), val(id), path("${id}.selfSM"), emit: selfsm
    script:
    """
    verifybamid2 --SVDPrefix ${params.verifybamid_svd_prefix} --Reference ${fasta} \\
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
    output:
    tuple val(fam), path("extracted/${id}.somalier"), emit: somalier
    script:
    """
    somalier extract -d extracted/ --sites ${params.somalier_sites} -f ${fasta} ${bam}
    """
    stub:
    "mkdir -p extracted && touch extracted/${id}.somalier"
}

process WRITE_PED {
    tag "$fam"
    input:
    tuple val(fam), val(kid), val(dad), val(mom)
    output:
    tuple val(fam), path("${fam}.ped"), emit: ped
    exec:
    def sx = { s -> s == 'male' ? '1' : (s == 'female' ? '2' : '0') }
    def ped = task.workDir.resolve("${fam}.ped")
    ped.text = [
        "${fam}\t${kid.id}\t${dad.id}\t${mom.id}\t${sx(kid.sex)}\t2",
        "${fam}\t${dad.id}\t0\t0\t1\t1",
        "${fam}\t${mom.id}\t0\t0\t2\t1",
    ].join('\n') + '\n'
}

process SOMALIER_RELATE {
    tag "$fam"
    label 'small'
    container params.containers.somalier
    publishDir "${params.outdir}/${fam}/qc", mode: 'copy'
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
    "touch ${fam}.somalier.pairs.tsv ${fam}.somalier.samples.tsv"
}

process QC_GATE {
    tag "$fam"
    label 'small'
    errorStrategy 'terminate'   // a failed stop-gate halts the whole run
    publishDir "${params.outdir}/${fam}/qc", mode: 'copy'
    input:
    tuple val(fam), path(ped), path(pairs), path(samples), path(selfsms), path(mosdepths), val(kid)
    output:
    tuple val(fam), path("${fam}.qc_gate.json"), emit: gate
    script:
    def vb = selfsms.collect { "--verifybamid ${it.baseName}=${it}" }.join(' ')
    def md = mosdepths.collect { "--mosdepth ${it.name.replace('.mosdepth.summary.txt', '')}=${it}" }.join(' ')
    def halt = params.qc_halt ? '' : '--no-halt'
    """
    trio-dnm qc-gate --ped ${ped} --proband ${kid} --data-type ${params.data_type} \\
        --somalier-pairs ${pairs} --somalier-samples ${samples} ${vb} ${md} ${halt} --out ${fam}.qc_gate.json
    """
    stub:
    "echo '{\"status\": \"PASS\"}' > ${fam}.qc_gate.json"
}

workflow SAMPLE_QC {
    take:
    samples     // (fam, id, role, sex, bam, bai)
    trios       // (fam, proband, father, mother) maps
    ref
    main:
    md = MOSDEPTH(samples, ref)
    vb = VERIFYBAMID2(samples, ref)
    sm = SOMALIER_EXTRACT(samples, ref)
    ped = WRITE_PED(trios.map { fam, k, d, m -> tuple(fam, k, d, m) })
    rel = SOMALIER_RELATE(ped.ped.join(sm.somalier.groupTuple(size: 3)))
    gate_in = ped.ped
        .join(rel.tsv)
        .join(vb.selfsm.map { fam, id, f -> tuple(fam, f) }.groupTuple(size: 3))
        .join(md.summary.map { fam, id, f -> tuple(fam, f) }.groupTuple(size: 3))
        .join(trios.map { fam, k, d, m -> tuple(fam, k.id) })
    gate = QC_GATE(gate_in)
    // Only trios whose gate is not FAIL proceed (with qc_halt=false a FAIL is reported and the trio skipped).
    passed = trios.join(ped.ped).join(gate.gate)
        .filter { fam, k, d, m, pedf, js -> !js.text.contains('"status": "FAIL"') }
        .map { fam, k, d, m, pedf, js -> tuple(fam, k, d, m, pedf) }
    emit:
    passed_trios = passed   // (fam, proband, father, mother, ped)
}
