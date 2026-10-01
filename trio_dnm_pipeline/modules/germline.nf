// Stage 3 (protocol §5) — per-sample GVCFs, joint trio genotyping, site filters, CGP,
// DeepTrio + GLnexus. v2.0 fixes to the v1.0 commands:
//   * no `--output-mode EMIT_VARIANTS_ONLY` with -ERC GVCF (it drops the
//     reference blocks that prove parental hom-ref, the core DNM evidence);
//   * no inline `#` comment inside a backslash-continued command;
//   * `--use-new-qual-calculator` removed (default since GATK 4.1, flag gone);
//   * StrandBiasBySample (FORMAT/SB) requested in HaplotypeCaller AND GenotypeGVCFs
//     (GenotypeGVCFs drops SB unless the annotation is requested there too), so
//     per-sample strand support can be tested in Layer 2; the candidate pileup
//     (MPILEUP_CANDIDATES, FORMAT/ADF,ADR) is the fallback strand source.

process HAPLOTYPECALLER {
    tag "$id"
    label 'medium'
    container params.containers.gatk
    input:
    tuple val(fam), val(id), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    path(intervals)
    output:
    tuple val(fam), val(id), path("${id}.g.vcf.gz"), path("${id}.g.vcf.gz.tbi"), emit: gvcf
    script:
    def pcr = params.pcr_free ? '--pcr-indel-model NONE' : ''
    def iv = intervals ? "-L ${intervals} -ip ${params.interval_padding}" : ''
    """
    gatk --java-options "-Xmx${(task.memory.toMega() * 0.85) as int}m" HaplotypeCaller \\
        -R ${fasta} -I ${bam} -O ${id}.g.vcf.gz \\
        -ERC GVCF ${pcr} ${iv} \\
        -G StandardAnnotation -G StandardHCAnnotation -G AS_StandardAnnotation \\
        -A StrandBiasBySample \\
        --max-alternate-alleles 6
    """
    stub:
    "touch ${id}.g.vcf.gz ${id}.g.vcf.gz.tbi"
}

process GENOTYPE_TRIO {
    tag "$fam"
    label 'medium'
    container params.containers.gatk
    input:
    tuple val(fam), path(gvcfs), path(tbis), path(ped)
    tuple path(fasta), path(fai), path(dict)
    path(intervals)
    output:
    tuple val(fam), path("${fam}.cgp.vcf.gz"), path("${fam}.cgp.vcf.gz.tbi"), emit: vcf
    script:
    def vs = gvcfs.collect { "-V ${it}" }.join(' ')
    def iv = intervals ? "-L ${intervals} -ip ${params.interval_padding}" : params.calling_contigs.collect { "-L ${it}" }.join(' ')
    """
    gatk GenomicsDBImport ${vs} --genomicsdb-workspace-path db ${iv} --merge-input-intervals
    gatk GenotypeGVCFs -R ${fasta} -V gendb://db -O raw.vcf.gz \\
        -G StandardAnnotation -G AS_StandardAnnotation -A StrandBiasBySample \\
        --only-output-calls-starting-in-intervals ${iv}

    # Site-level hard filters (Section 5.3; GATK generic thresholds incl. QUAL < 30 for both classes).
    # VQSR/VETS replaces this when a cohort is available.
    gatk SelectVariants -V raw.vcf.gz -select-type SNP -select-type MNP -O snv.vcf.gz
    gatk SelectVariants -V raw.vcf.gz -select-type INDEL -select-type MIXED -O indel.vcf.gz
    gatk VariantFiltration -V snv.vcf.gz -O snv.f.vcf.gz \\
        -filter "QD < 2.0" --filter-name QD2 -filter "QUAL < 30.0" --filter-name QUAL30 \\
        -filter "FS > 60.0" --filter-name FS60 -filter "MQ < 40.0" --filter-name MQ40 \\
        -filter "MQRankSum < -12.5" --filter-name MQRS -filter "ReadPosRankSum < -8.0" --filter-name RPRS8 \\
        -filter "SOR > 3.0" --filter-name SOR3
    gatk VariantFiltration -V indel.vcf.gz -O indel.f.vcf.gz \\
        -filter "QD < 2.0" --filter-name QD2 -filter "QUAL < 30.0" --filter-name QUAL30 \\
        -filter "FS > 200.0" --filter-name FS200 -filter "ReadPosRankSum < -20.0" --filter-name RPRS20 \\
        -filter "SOR > 10.0" --filter-name SOR10
    gatk MergeVcfs -I snv.f.vcf.gz -I indel.f.vcf.gz -O filtered.vcf.gz

    # Family-aware genotype refinement + PossibleDeNovo (hiConfDeNovo / loConfDeNovo).
    gatk CalculateGenotypePosteriors -V filtered.vcf.gz -ped ${ped} --skip-population-priors -O cgp.tmp.vcf.gz
    gatk VariantAnnotator -R ${fasta} -V cgp.tmp.vcf.gz -ped ${ped} -A PossibleDeNovo -O ${fam}.cgp.vcf.gz
    """
    stub:
    "touch ${fam}.cgp.vcf.gz ${fam}.cgp.vcf.gz.tbi"
}

// Also used (aliased) in dnm.nf for the DeepTrio/GLnexus call set; `label` names the
// call set (trio = GATK, deeptrio) and `samples` fixes the order proband, father, mother.
process NORMALISE {
    tag "${fam}:${label}"
    label 'small'
    container params.containers.bcftools
    input:
    tuple val(fam), val(label), path(vcf), val(samples)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), val(label), path("${fam}.${label}.norm.vcf.gz"), emit: vcf
    script:
    """
    bcftools norm -m -any -f ${fasta} ${vcf} -Ou \\
        | bcftools view -s ${samples.join(',')} -Oz -o ${fam}.${label}.norm.vcf.gz
    """
    stub:
    "touch ${fam}.${label}.norm.vcf.gz"
}

process VCFANNO {
    // gnomAD v4.1 joint AF, ClinVar and in-house PoN (conf/vcfanno.toml). The TOML text
    // arrives with its `file =` paths rewritten to the staged copies under vcfanno_res/.
    tag "${fam}:${label}"
    label 'small'
    container params.containers.vcfanno
    input:
    tuple val(fam), val(label), path(vcf)
    val(toml)
    path(anno_files, stageAs: 'vcfanno_res/*')
    output:
    tuple val(fam), val(label), path("${fam}.${label}.anno.vcf"), emit: vcf
    script:
    """
    cat > vcfanno.toml <<'__VCFANNO_TOML__'
${toml}
__VCFANNO_TOML__
    vcfanno -p ${task.cpus} vcfanno.toml ${vcf} > ${fam}.${label}.anno.vcf
    """
    stub:
    "touch ${fam}.${label}.anno.vcf"
}

process BGZIP_INDEX {
    tag "${fam}:${label}"
    label 'small'
    container params.containers.bcftools
    input:
    tuple val(fam), val(label), path(vcf)
    output:
    tuple val(fam), val(label), path("${fam}.${label}.vcf.gz"), path("${fam}.${label}.vcf.gz.tbi"), emit: vcf
    script:
    """
    bcftools view -Oz -o ${fam}.${label}.vcf.gz ${vcf}
    bcftools index -t ${fam}.${label}.vcf.gz
    """
    stub:
    "touch ${fam}.${label}.vcf.gz ${fam}.${label}.vcf.gz.tbi"
}

process DEEPTRIO {
    // Given the pre-BQSR BAMs when the workflow aligned the reads (DeepTrio is trained
    // on un-recalibrated base qualities, §4.3); otherwise the samplesheet BAM/CRAM.
    tag "$fam"
    label 'large'
    container params.containers.deeptrio
    input:
    tuple val(fam), val(ids), path(bams), path(bais)   // ordered proband, father, mother
    tuple path(fasta), path(fai), path(dict)
    path(intervals)
    output:
    tuple val(fam), path("*.g.vcf.gz"), emit: gvcfs
    script:
    def (kid, dad, mom) = ids
    def model = params.data_type == 'wgs' ? 'WGS' : 'WES'
    def reg = intervals ? "--regions ${intervals}" : ''
    """
    /opt/deepvariant/bin/deeptrio/run_deeptrio --model_type=${model} --ref=${fasta} ${reg} \\
        --reads_child=${bams[0]} --reads_parent1=${bams[1]} --reads_parent2=${bams[2]} \\
        --sample_name_child=${kid} --sample_name_parent1=${dad} --sample_name_parent2=${mom} \\
        --output_vcf_child=${kid}.dt.vcf.gz --output_vcf_parent1=${dad}.dt.vcf.gz --output_vcf_parent2=${mom}.dt.vcf.gz \\
        --output_gvcf_child=${kid}.g.vcf.gz --output_gvcf_parent1=${dad}.g.vcf.gz --output_gvcf_parent2=${mom}.g.vcf.gz \\
        --num_shards=${task.cpus} --intermediate_results_dir=tmp
    """
    stub:
    "touch ${ids.collect { it + '.g.vcf.gz' }.join(' ')}"
}

process GLNEXUS {
    tag "$fam"
    label 'medium'
    container params.containers.glnexus
    input:
    tuple val(fam), path(gvcfs)
    output:
    tuple val(fam), path("${fam}.deeptrio.bcf"), emit: vcf
    script:
    def cfg = params.data_type == 'wgs' ? 'DeepVariant_unfiltered' : 'DeepVariantWES'
    """
    glnexus_cli --config ${cfg} --threads ${task.cpus} --mem-gbytes ${task.memory.toGiga()} ${gvcfs} > ${fam}.deeptrio.bcf
    """
    stub:
    "touch ${fam}.deeptrio.bcf"
}

workflow GERMLINE_CALLING {
    take:
    trios   // (fam, proband, father, mother, ped)
    ref
    res     // staged resources (main.nf)
    main:
    per_sample = trios.flatMap { fam, k, d, m, ped -> [k, d, m].collect { tuple(fam, it.id, it.bam, it.bai) } }
    hc = HAPLOTYPECALLER(per_sample, ref, res.intervals)
    jt = GENOTYPE_TRIO(
        hc.gvcf.map { fam, id, g, t -> tuple(fam, g, t) }.groupTuple(size: 3)
            .join(trios.map { fam, k, d, m, ped -> tuple(fam, ped) }),
        ref, res.intervals)
    norm = NORMALISE(
        jt.vcf.join(trios.map { fam, k, d, m, ped -> tuple(fam, [k.id, d.id, m.id]) })
            .map { fam, v, t, ids -> tuple(fam, 'trio', v, ids) },
        ref)
    anno = VCFANNO(norm.vcf, res.vcfanno_toml, res.vcfanno_files)
    trio_vcf = BGZIP_INDEX(anno.vcf).vcf.map { fam, label, v, t -> tuple(fam, v, t) }

    dt = DEEPTRIO(trios.map { fam, k, d, m, ped -> tuple(fam, [k.id, d.id, m.id], [k.dt_bam, d.dt_bam, m.dt_bam], [k.dt_bai, d.dt_bai, m.dt_bai]) },
        ref, res.intervals)
    gl = GLNEXUS(dt.gvcfs)
    emit:
    trio_vcf                // (fam, vcf, tbi) — GATK: filtered, CGP, normalised, vcfanno-annotated
    deeptrio_vcf = gl.vcf   // (fam, bcf) — raw GLnexus; normalised + annotated in DNM_DETECTION
}
