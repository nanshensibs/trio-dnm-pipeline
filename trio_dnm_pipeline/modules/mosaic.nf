// Stage 7 — mosaic-aware track. Complements the low-VAF track inside
// `trio-dnm call` (which only sees sites HaplotypeCaller emitted) with a
// sensitive tumour-only Mutect2 scan of the proband classified by
// MosaicForecast, then asserts absence in both parents from their BAMs.

process MUTECT2_PROBAND {
    tag "$fam"
    label 'medium'
    container params.containers.gatk
    input:
    tuple val(fam), val(kid), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), val(kid), path("${kid}.m2.vcf.gz"), emit: vcf
    script:
    def iv = params.intervals ? "-L ${params.intervals}" : ''
    """
    gatk Mutect2 -R ${fasta} -I ${bam} -tumor ${kid} ${iv} \\
        --germline-resource ${params.germline_resource} --panel-of-normals ${params.m2_pon} \\
        -O m2.vcf.gz
    gatk FilterMutectCalls -R ${fasta} -V m2.vcf.gz -O m2.f.vcf.gz
    gatk LeftAlignAndTrimVariants -R ${fasta} -V m2.f.vcf.gz --split-multi-allelics -O ${kid}.m2.vcf.gz
    """
    stub:
    "touch ${kid}.m2.vcf.gz"
}

process MOSAICFORECAST {
    tag "$fam"
    label 'medium'
    container params.containers.mosaicforecast
    input:
    tuple val(fam), val(kid), path(m2_vcf), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), path("${kid}.mf.predictions.txt"), emit: pred
    script:
    // MosaicForecast expects <sample>.bam|cram inside a directory.
    def ext = bam.name.endsWith('.cram') ? 'cram' : 'bam'
    def idx = ext == 'cram' ? 'cram.crai' : 'bam.bai'
    """
    # MosaicForecast input: chrom, start(0-based), end, ref, alt, sample; germline-like VAFs dropped
    vcf2mfbed.py ${m2_vcf} ${kid} --min-vaf ${params.mosaic_min_vaf} --max-vaf 0.40 > ${kid}.m2.bed
    mkdir -p bams && ln -s \$(readlink -f ${bam}) bams/${kid}.${ext} && ln -s \$(readlink -f ${bai}) bams/${kid}.${idx}
    ReadLevel_Features_extraction.py ${kid}.m2.bed features.txt bams ${fasta} ${params.mf_umap_bigwig} ${task.cpus} ${ext}
    Prediction.R features.txt ${params.mf_model} Refine ${kid}.mf.predictions.txt
    """
    stub:
    "touch ${kid}.mf.predictions.txt"
}

process PARENTAL_ABSENCE {
    tag "$fam"
    label 'small'
    container params.containers.bcftools
    publishDir "${params.outdir}/${fam}/mosaic", mode: 'copy'
    input:
    tuple val(fam), path(pred), val(ids), path(bams), path(bais)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), path("${fam}.mosaic_candidates.tsv")
    script:
    """
    # keep MosaicForecast 'mosaic' calls; id column is sample~chrom~pos~ref~alt
    awk -F'\\t' 'NR>1 && \$NF=="mosaic" {split(\$1,a,"~"); print a[2]"\\t"a[3]}' ${pred} | sort -u > sites.tsv
    echo -e "chrom\\tpos\\tref\\talt\\tAD_proband\\tAD_father\\tAD_mother" > ${fam}.mosaic_candidates.tsv
    if [ -s sites.tsv ]; then
      bcftools mpileup -f ${fasta} -T sites.tsv -a AD -d 10000 -q 20 -Q 20 ${bams.join(' ')} -Ou \\
        | bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%AD]\\n' >> ${fam}.mosaic_candidates.tsv
    fi
    """
    stub:
    "touch ${fam}.mosaic_candidates.tsv"
}

workflow MOSAIC_TRACK {
    take:
    trios   // (fam, k, d, m, ped)
    ref
    main:
    m2 = MUTECT2_PROBAND(trios.map { fam, k, d, m, ped -> tuple(fam, k.id, k.bam, k.bai) }, ref)
    mf = MOSAICFORECAST(m2.vcf.join(trios.map { fam, k, d, m, ped -> tuple(fam, k.bam, k.bai) }), ref)
    PARENTAL_ABSENCE(mf.pred.join(trios.map { fam, k, d, m, ped -> tuple(fam, [k.id, d.id, m.id], [k.bam, d.bam, m.bam], [k.bai, d.bai, m.bai]) }), ref)
}
