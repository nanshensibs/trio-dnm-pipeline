// Stage 2 (protocol §3.1 + §4) — samplesheet rows given as FASTQ: read QC (FastQC on
// the raw reads; fastp adapter auto-detection, poly-G trimming on 2-colour chemistry,
// minimum length 50 bp, low-complexity filter), BWA-MEM2 alignment with a full @RG,
// duplicate marking over all lanes of a sample, and BQSR. Emits the analysis-ready
// (BQSR) BAM used by GATK, QC and the somatic/mosaic tracks, and the pre-BQSR
// duplicate-marked BAM that DeepTrio is given (trained on un-recalibrated qualities, §4.3).

include { asList } from './utils'

process FASTQC {
    tag "${meta.id}:${meta.unit}"
    label 'small'
    container params.containers.fastqc
    input:
    tuple val(meta), path(reads, stageAs: 'raw/*')
    output:
    path "*_fastqc.zip", emit: zip
    path "*_fastqc.html", emit: html
    script:
    def p = "${meta.id}.${meta.unit}"
    """
    # consistent report names: <sample>.<unit>_R1/_R2
    ln -s ${reads[0]} ${p}_R1.fastq.gz
    ln -s ${reads[1]} ${p}_R2.fastq.gz
    fastqc --threads ${task.cpus} --outdir . ${p}_R1.fastq.gz ${p}_R2.fastq.gz
    """
    stub:
    def p = "${meta.id}.${meta.unit}"
    "touch ${p}_R1_fastqc.zip ${p}_R2_fastqc.zip ${p}_R1_fastqc.html ${p}_R2_fastqc.html"
}

process FASTP {
    tag "${meta.id}:${meta.unit}"
    label 'medium'
    container params.containers.fastp
    input:
    tuple val(meta), path(reads, stageAs: 'raw/*')
    output:
    tuple val(meta), path("${meta.id}.${meta.unit}_R{1,2}.trim.fastq.gz"), emit: reads
    path "${meta.id}.${meta.unit}.fastp.json", emit: json
    path "${meta.id}.${meta.unit}.fastp.html", emit: html
    script:
    def p = "${meta.id}.${meta.unit}"
    """
    ln -s ${reads[0]} ${p}_R1.fastq.gz
    ln -s ${reads[1]} ${p}_R2.fastq.gz
    # poly-G trimming is enabled automatically for NovaSeq/NextSeq read names
    fastp --in1 ${p}_R1.fastq.gz --in2 ${p}_R2.fastq.gz \\
        --out1 ${p}_R1.trim.fastq.gz --out2 ${p}_R2.trim.fastq.gz \\
        --json ${p}.fastp.json --html ${p}.fastp.html --report_title '${p}' \\
        --thread ${Math.min(task.cpus as int, 16)} \\
        --detect_adapter_for_pe --length_required 50 --low_complexity_filter
    """
    stub:
    def p = "${meta.id}.${meta.unit}"
    "touch ${p}_R1.trim.fastq.gz ${p}_R2.trim.fastq.gz ${p}.fastp.json ${p}.fastp.html"
}

process BWAMEM2 {
    tag "${meta.id}:${meta.unit}"
    label 'large'
    container params.containers.bwamem2
    input:
    tuple val(meta), path(reads)
    tuple path(fasta), path(fai), path(dict)
    path(index)   // <prefix>.0123 .amb .ann .bwt.2bit.64 .pac
    output:
    tuple val(meta), path("${meta.id}.${meta.unit}.sorted.bam"), emit: bam
    script:
    def p = "${meta.id}.${meta.unit}"
    def prefix = asList(index).find { it.name.endsWith('.bwt.2bit.64') }.name - ~/\.bwt\.2bit\.64$/
    def lb = meta.library ?: meta.id
    """
    # @RG from the first Illumina read name: @instrument:run:flowcell:lane:tile:x:y read:filtered:control:index
    # (ID = flowcell.lane, PU = flowcell.lane.barcode); non-Illumina names fall back to sample/unit.
    hdr=\$(zcat -f ${reads[0]} | head -n 1 || true)
    fc=\$(printf '%s\\n' "\$hdr" | awk '{n = split(substr(\$1, 2), a, ":"); print (n >= 7 ? a[3] : "${meta.id}")}')
    lane=\$(printf '%s\\n' "\$hdr" | awk '{n = split(substr(\$1, 2), a, ":"); print (n >= 7 ? a[4] : "${meta.unit}")}')
    bc=\$(printf '%s\\n' "\$hdr" | awk '{n = split(\$2, b, ":"); print (n >= 4 && b[4] != "" ? b[4] : "NA")}')
    rg="@RG\\tID:\${fc}.\${lane}\\tSM:${meta.id}\\tLB:${lb}\\tPL:ILLUMINA\\tPU:\${fc}.\${lane}.\${bc}"
    bwa-mem2 mem -t ${task.cpus} -K 100000000 -Y -R "\$rg" ${prefix} ${reads[0]} ${reads[1]} \\
        | samtools sort -@ ${task.cpus} -m 1G -T ${p}.sort -o ${p}.sorted.bam -
    """
    stub:
    "touch ${meta.id}.${meta.unit}.sorted.bam"
}

process MARKDUPLICATES {
    // Picard MarkDuplicates (via GATK) over all lanes of a sample; the coordinate-sorted
    // lane BAMs are merged here. Optical distance 2500 for patterned flow cells (§4.2).
    tag "${meta.id}"
    label 'medium'
    container params.containers.gatk
    input:
    tuple val(meta), path(bams)
    output:
    tuple val(meta), path("${meta.id}.md.bam"), path("${meta.id}.md.bam.bai"), emit: bam
    path "${meta.id}.md.metrics.txt", emit: metrics
    script:
    def ins = asList(bams).collect { "-I ${it}" }.join(' ')
    """
    gatk --java-options "-Xmx${(task.memory.toMega() * 0.85) as int}m" MarkDuplicates ${ins} \\
        -O ${meta.id}.md.bam -M ${meta.id}.md.metrics.txt \\
        --OPTICAL_DUPLICATE_PIXEL_DISTANCE 2500 --CREATE_INDEX true --TMP_DIR .
    mv ${meta.id}.md.bai ${meta.id}.md.bam.bai
    """
    stub:
    "touch ${meta.id}.md.bam ${meta.id}.md.bam.bai ${meta.id}.md.metrics.txt"
}

process BQSR {
    // BaseRecalibrator (params.known_sites: dbSNP, Mills/1000G gold indels, 1KG SNVs)
    // + ApplyBQSR (§4.3).
    tag "${meta.id}"
    label 'medium'
    container params.containers.gatk
    publishDir "${params.outdir}/${meta.fam}/alignment", mode: params.publish_dir_mode
    input:
    tuple val(meta), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    path(known_sites, stageAs: 'known_sites/*')
    path(intervals)
    output:
    tuple val(meta), path("${meta.id}.bam"), path("${meta.id}.bam.bai"), emit: bam
    path "${meta.id}.recal.table", emit: table
    script:
    def ks = asList(known_sites).findAll { it.name ==~ /.*\.(vcf|vcf\.gz|vcf\.bgz|bcf)$/ }.collect { "--known-sites ${it}" }.join(' ')
    def iv = intervals ? "-L ${intervals} -ip ${params.interval_padding}" : ''
    def mem = (task.memory.toMega() * 0.85) as int
    """
    gatk --java-options "-Xmx${mem}m" BaseRecalibrator -R ${fasta} -I ${bam} ${ks} ${iv} \\
        -O ${meta.id}.recal.table --tmp-dir .
    gatk --java-options "-Xmx${mem}m" ApplyBQSR -R ${fasta} -I ${bam} --bqsr-recal-file ${meta.id}.recal.table \\
        -O ${meta.id}.bam --tmp-dir .
    mv ${meta.id}.bai ${meta.id}.bam.bai
    """
    stub:
    "touch ${meta.id}.bam ${meta.id}.bam.bai ${meta.id}.recal.table"
}

workflow ALIGN {
    take:
    units   // (meta, [fastq_1, fastq_2]); meta: fam, id, role, sex, dna_source, library, unit, n_units
    ref
    res     // staged resources (main.nf): bwa_index, known_sites, intervals
    main:
    qc = FASTQC(units)
    trimmed = FASTP(units)
    lanes = BWAMEM2(trimmed.reads, ref, res.bwa_index)
    // all units of a sample -> one MarkDuplicates task (groupKey releases each sample as soon as it is complete)
    per_sample = lanes.bam
        .map { meta, bam -> tuple(groupKey(meta.id, meta.n_units), meta.subMap(['fam', 'id', 'role', 'sex', 'dna_source']), bam) }
        .groupTuple()
        .map { key, metas, bams -> tuple(metas[0], bams) }
    md = MARKDUPLICATES(per_sample)
    bq = BQSR(md.bam, ref, res.known_sites, res.intervals)
    samples = bq.bam.map { meta, bam, bai -> tuple(meta.id, meta, bam, bai) }
        .join(md.bam.map { meta, bam, bai -> tuple(meta.id, bam, bai) })
        .map { id, meta, bam, bai, md_bam, md_bai -> meta + [bam: bam, bai: bai, dt_bam: md_bam, dt_bai: md_bai] }
    mqc = qc.zip.mix(trimmed.json, md.metrics, bq.table)
    emit:
    samples   // sample maps: fam, id, role, sex, dna_source, bam, bai, dt_bam, dt_bai
    multiqc = mqc
}
