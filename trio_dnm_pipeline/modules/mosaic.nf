// Stage 7 (protocol §9) — mosaic-aware track. Complements the low-VAF track inside
// `trio-dnm call` (which only sees sites HaplotypeCaller emitted) with a
// sensitive tumour-only Mutect2 scan of the proband classified by
// MosaicForecast, then asserts absence in both parents from their BAMs.

include { asList ; mainFile ; trioDnmEnv } from './utils'

process MUTECT2_PROBAND {
    tag "$fam"
    label 'medium'
    container params.containers.gatk
    input:
    tuple val(fam), val(kid), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    path(intervals)
    path(germline, stageAs: 'germline_resource/*')
    path(pon, stageAs: 'm2_pon/*')
    output:
    tuple val(fam), val(kid), path("${kid}.m2.vcf.gz"), emit: vcf
    script:
    def iv = intervals ? "-L ${intervals}" : ''
    def pn = pon ? "--panel-of-normals ${mainFile(pon)}" : ''
    """
    gatk Mutect2 -R ${fasta} -I ${bam} -tumor ${kid} ${iv} \\
        --germline-resource ${mainFile(germline)} ${pn} \\
        -O m2.vcf.gz
    gatk FilterMutectCalls -R ${fasta} -V m2.vcf.gz -O m2.f.vcf.gz
    gatk LeftAlignAndTrimVariants -R ${fasta} -V m2.f.vcf.gz --split-multi-allelics -O ${kid}.m2.vcf.gz
    """
    stub:
    "touch ${kid}.m2.vcf.gz"
}

process MOSAIC_SITES_BED {
    // MosaicForecast input BED from the Mutect2 calls. Runs in the trio-dnm image
    // (python3 >= 3.10; the MosaicForecast image ships Python 3.6).
    tag "$fam"
    label 'small'
    container params.containers.trio_dnm
    input:
    tuple val(fam), val(kid), path(m2_vcf)
    output:
    tuple val(fam), val(kid), path("${kid}.m2.bed"), emit: bed
    script:
    """
    # chrom, start (0-based), end, ref, alt, sample: PASS biallelic SNVs only (the Refine
    # model is SNV-trained); germline-like VAFs dropped
    vcf2mfbed.py ${m2_vcf} ${kid} --min-vaf ${params.mosaic_min_vaf} --max-vaf 0.40 > ${kid}.m2.bed
    """
    stub:
    "touch ${kid}.m2.bed"
}

process MOSAICFORECAST {
    tag "$fam"
    label 'medium'
    container params.containers.mosaicforecast
    input:
    tuple val(fam), val(kid), path(bed), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    path(umap)
    path(model)
    output:
    tuple val(fam), path("${kid}.mf.predictions.txt"), emit: pred
    script:
    // MosaicForecast expects <sample>.bam|cram inside a directory.
    def ext = bam.name.endsWith('.cram') ? 'cram' : 'bam'
    def idx = ext == 'cram' ? 'cram.crai' : 'bam.bai'
    """
    mkdir -p bams && ln -s \$(readlink -f ${bam}) bams/${kid}.${ext} && ln -s \$(readlink -f ${bai}) bams/${kid}.${idx}
    ReadLevel_Features_extraction.py ${bed} features.txt bams ${fasta} ${umap} ${task.cpus} ${ext}
    Prediction.R features.txt ${model} Refine ${kid}.mf.predictions.txt
    """
    stub:
    "touch ${kid}.mf.predictions.txt"
}

process PARENTAL_ABSENCE {
    tag "$fam"
    label 'small'
    container params.containers.bcftools
    publishDir "${params.outdir}/${fam}/mosaic", mode: params.publish_dir_mode
    input:
    tuple val(fam), path(pred), val(ids), path(bams), path(bais)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), path("${fam}.mosaic_candidates.tsv")
    script:
    def rg = [ids, asList(bams)].transpose().collect { id, b -> "${b} ${id}" }.join(' ')
    """
    # Prediction.R output: find the 'prediction' column by name (the last column is
    # 'chromosome') and keep 'mosaic' calls, including suffixed Refine labels such as
    # 'mosaic;cautious:AF<0.01'. The id column is sample~chrom~pos~ref~alt.
    awk -F'\\t' 'NR == 1 {for (i = 1; i <= NF; i++) if (\$i == "prediction") c = i; next}
        c && \$c ~ /^mosaic/ {split(\$1, a, "~"); print a[2] "\\t" a[3] "\\t" \$c}' ${pred} \\
        | sort -u -k1,1 -k2,2n > sites.tsv
    printf 'chrom\\tpos\\tref\\talt\\tmf_prediction\\tAD_proband\\tAD_father\\tAD_mother\\n' > ${fam}.mosaic_candidates.tsv
    if [ -s sites.tsv ]; then
      cut -f1,2 sites.tsv > regions.tsv
      printf '* %s %s\\n' ${rg} > read_groups.txt
      # -R: index jumps to each site (not a stream through the whole BAM/CRAM)
      bcftools mpileup -f ${fasta} -R regions.tsv -G read_groups.txt -a AD -d 10000 -q 20 -Q 20 ${asList(bams).join(' ')} -Ou \\
        | bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%AD]\\n' \\
        | awk -F'\\t' -v OFS='\\t' 'NR == FNR {lab[\$1 ":" \$2] = \$3; next} {print \$1, \$2, \$3, \$4, lab[\$1 ":" \$2], \$5, \$6, \$7}' sites.tsv - \\
        >> ${fam}.mosaic_candidates.tsv
    fi
    """
    stub:
    "touch ${fam}.mosaic_candidates.tsv"
}

workflow MOSAIC_TRACK {
    take:
    trios   // (fam, k, d, m, ped)
    ref
    res     // staged resources (main.nf)
    main:
    m2 = MUTECT2_PROBAND(trios.map { fam, k, d, m, ped -> tuple(fam, k.id, k.bam, k.bai) }, ref,
        res.intervals, res.germline_resource, res.m2_pon)
    bed = MOSAIC_SITES_BED(m2.vcf)
    mf = MOSAICFORECAST(
        bed.bed.join(trios.map { fam, k, d, m, ped -> tuple(fam, k.bam, k.bai) }), ref, res.mf_umap_bigwig, res.mf_model)
    PARENTAL_ABSENCE(mf.pred.join(trios.map { fam, k, d, m, ped -> tuple(fam, [k.id, d.id, m.id], [k.bam, d.bam, m.bam], [k.bai, d.bai, m.bai]) }), ref)
}
