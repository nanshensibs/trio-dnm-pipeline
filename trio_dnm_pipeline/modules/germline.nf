// Stages 2-3 — per-sample GVCFs, joint trio genotyping, site filters, CGP,
// DeepTrio + GLnexus. v2.0 fixes to the v1.0 commands:
//   * no `--output-mode EMIT_VARIANTS_ONLY` with -ERC GVCF (it drops the
//     reference blocks that prove parental hom-ref, the core DNM evidence);
//   * no inline `#` comment inside a backslash-continued command;
//   * `--use-new-qual-calculator` removed (default since GATK 4.1, flag gone);
//   * StrandBiasBySample (FORMAT/SB) requested so per-sample strand support
//     can be tested in Layer 2.

process HAPLOTYPECALLER {
    tag "$id"
    label 'medium'
    container params.containers.gatk
    input:
    tuple val(fam), val(id), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), val(id), path("${id}.g.vcf.gz"), path("${id}.g.vcf.gz.tbi"), emit: gvcf
    script:
    def pcr = params.pcr_free ? '--pcr-indel-model NONE' : ''
    def iv = params.intervals ? "-L ${params.intervals} -ip ${params.interval_padding}" : ''
    """
    gatk --java-options "-Xmx${task.memory.toGiga() - 2}g" HaplotypeCaller \\
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
    output:
    tuple val(fam), path("${fam}.cgp.vcf.gz"), path("${fam}.cgp.vcf.gz.tbi"), emit: vcf
    script:
    def vs = gvcfs.collect { "-V ${it}" }.join(' ')
    def iv = params.intervals ? "-L ${params.intervals} -ip ${params.interval_padding}" : params.calling_contigs.collect { "-L ${it}" }.join(' ')
    """
    gatk GenomicsDBImport ${vs} --genomicsdb-workspace-path db ${iv} --merge-input-intervals
    gatk GenotypeGVCFs -R ${fasta} -V gendb://db -O raw.vcf.gz \\
        -G StandardAnnotation -G AS_StandardAnnotation --only-output-calls-starting-in-intervals ${iv}

    # Site-level hard filters (Section 5.3). VQSR/VETS replaces this when a cohort is available.
    gatk SelectVariants -V raw.vcf.gz -select-type SNP -select-type MNP -O snv.vcf.gz
    gatk SelectVariants -V raw.vcf.gz -select-type INDEL -select-type MIXED -O indel.vcf.gz
    gatk VariantFiltration -V snv.vcf.gz -O snv.f.vcf.gz \\
        -filter "QD < 2.0" --filter-name QD2 -filter "FS > 60.0" --filter-name FS60 \\
        -filter "MQ < 40.0" --filter-name MQ40 -filter "MQRankSum < -12.5" --filter-name MQRS \\
        -filter "ReadPosRankSum < -8.0" --filter-name RPRS8 -filter "SOR > 3.0" --filter-name SOR3
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

process NORMALISE {
    tag "$fam"
    label 'small'
    container params.containers.bcftools
    input:
    tuple val(fam), path(vcf), path(tbi)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), path("${fam}.norm.vcf.gz"), emit: vcf
    script:
    """
    bcftools norm -m -any -f ${fasta} -Oz -o ${fam}.norm.vcf.gz ${vcf}
    """
    stub:
    "touch ${fam}.norm.vcf.gz"
}

process VCFANNO {
    // gnomAD v4.1 joint AF, ClinVar and in-house PoN (conf/vcfanno.toml)
    tag "$fam"
    label 'small'
    container params.containers.vcfanno
    input:
    tuple val(fam), path(vcf)
    output:
    tuple val(fam), path("${fam}.anno.vcf"), emit: vcf
    script:
    """
    vcfanno -p ${task.cpus} ${params.vcfanno_toml} ${vcf} > ${fam}.anno.vcf
    """
    stub:
    "touch ${fam}.anno.vcf"
}

process BGZIP_INDEX {
    tag "$fam"
    label 'small'
    container params.containers.bcftools
    input:
    tuple val(fam), path(vcf)
    output:
    tuple val(fam), path("${fam}.trio.vcf.gz"), path("${fam}.trio.vcf.gz.tbi"), emit: vcf
    script:
    """
    bcftools view -Oz -o ${fam}.trio.vcf.gz ${vcf}
    bcftools index -t ${fam}.trio.vcf.gz
    """
    stub:
    "touch ${fam}.trio.vcf.gz ${fam}.trio.vcf.gz.tbi"
}

process DEEPTRIO {
    tag "$fam"
    label 'large'
    container params.containers.deeptrio
    input:
    tuple val(fam), val(ids), path(bams), path(bais)   // ordered proband, father, mother
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), path("*.g.vcf.gz"), emit: gvcfs
    script:
    def (kid, dad, mom) = ids
    def model = params.data_type == 'wgs' ? 'WGS' : 'WES'
    def reg = params.intervals ? "--regions ${params.intervals}" : ''
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
    glnexus_cli --config ${cfg} --threads ${task.cpus} ${gvcfs} > ${fam}.deeptrio.bcf
    """
    stub:
    "touch ${fam}.deeptrio.bcf"
}

workflow GERMLINE_CALLING {
    take:
    trios   // (fam, proband, father, mother, ped)
    ref
    main:
    per_sample = trios.flatMap { fam, k, d, m, ped -> [k, d, m].collect { tuple(fam, it.id, it.bam, it.bai) } }
    hc = HAPLOTYPECALLER(per_sample, ref)
    jt = GENOTYPE_TRIO(
        hc.gvcf.map { fam, id, g, t -> tuple(fam, g, t) }.groupTuple(size: 3)
            .join(trios.map { fam, k, d, m, ped -> tuple(fam, ped) }),
        ref)
    trio_vcf = BGZIP_INDEX(VCFANNO(NORMALISE(jt.vcf, ref).vcf).vcf).vcf

    dt = DEEPTRIO(trios.map { fam, k, d, m, ped -> tuple(fam, [k.id, d.id, m.id], [k.bam, d.bam, m.bam], [k.bai, d.bai, m.bai]) }, ref)
    gl = GLNEXUS(dt.gvcfs)
    emit:
    trio_vcf       // (fam, vcf, tbi)
    deeptrio_vcf = gl.vcf   // (fam, bcf) — raw, normalised downstream
}
