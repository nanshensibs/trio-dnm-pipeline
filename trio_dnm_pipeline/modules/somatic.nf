// Stage 10 — somatic analyses: tumour/affected-tissue vs matched blood
// (Mutect2 + Strelka2 + DeepSomatic consensus, CNVkit allele-specific CN,
// trio-aware germline subtraction, CCF, TMB, signatures, two-hit analysis)
// and a clonal-haematopoiesis screen of every trio member's blood.

include { VEP as VEP_SOMATIC ; VEP as VEP_CH } from './dnm'

process MUTECT2_PAIRED {
    tag "${tid}"
    label 'large'
    container params.containers.gatk
    input:
    tuple val(fam), val(tid), val(role), path(tbam), path(tbai), val(nid), path(nbam), path(nbai)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), val(tid), path("${tid}.mutect2.vcf.gz"), emit: vcf
    tuple val(fam), val(tid), path("${tid}.contamination.table"), emit: contamination
    script:
    def iv = params.intervals ? "-L ${params.intervals}" : ''
    """
    gatk Mutect2 -R ${fasta} -I ${tbam} -I ${nbam} -normal ${nid} ${iv} \\
        --germline-resource ${params.germline_resource} --panel-of-normals ${params.m2_pon} \\
        --f1r2-tar-gz f1r2.tar.gz -O unfiltered.vcf.gz
    gatk LearnReadOrientationModel -I f1r2.tar.gz -O rom.tar.gz
    gatk GetPileupSummaries -I ${tbam} -V ${params.common_biallelic} -L ${params.common_biallelic} -O t.pileups.table
    gatk GetPileupSummaries -I ${nbam} -V ${params.common_biallelic} -L ${params.common_biallelic} -O n.pileups.table
    gatk CalculateContamination -I t.pileups.table -matched n.pileups.table \\
        -O ${tid}.contamination.table --tumor-segmentation segments.table
    gatk FilterMutectCalls -R ${fasta} -V unfiltered.vcf.gz --contamination-table ${tid}.contamination.table \\
        --tumor-segmentation segments.table --ob-priors rom.tar.gz -O filtered.vcf.gz
    gatk LeftAlignAndTrimVariants -R ${fasta} -V filtered.vcf.gz --split-multi-allelics -O ${tid}.mutect2.vcf.gz
    """
    stub:
    "touch ${tid}.mutect2.vcf.gz ${tid}.contamination.table"
}

process STRELKA2 {
    tag "${tid}"
    label 'large'
    container params.containers.strelka
    input:
    tuple val(fam), val(tid), val(role), path(tbam), path(tbai), val(nid), path(nbam), path(nbai)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), val(tid), path("${tid}.strelka.*.vcf.gz"), emit: vcf
    script:
    def exome = params.data_type == 'wgs' ? '' : '--exome'
    """
    configManta.py --normalBam ${nbam} --tumorBam ${tbam} --referenceFasta ${fasta} ${exome} --runDir manta
    manta/runWorkflow.py -m local -j ${task.cpus}
    configureStrelkaSomaticWorkflow.py --normalBam ${nbam} --tumorBam ${tbam} --referenceFasta ${fasta} ${exome} \\
        --indelCandidates manta/results/variants/candidateSmallIndels.vcf.gz --runDir strelka
    strelka/runWorkflow.py -m local -j ${task.cpus}
    cp strelka/results/variants/somatic.snvs.vcf.gz ${tid}.strelka.snvs.vcf.gz
    cp strelka/results/variants/somatic.indels.vcf.gz ${tid}.strelka.indels.vcf.gz
    """
    stub:
    "touch ${tid}.strelka.snvs.vcf.gz ${tid}.strelka.indels.vcf.gz"
}

process DEEPSOMATIC {
    tag "${tid}"
    label 'large'
    container params.containers.deepsomatic
    input:
    tuple val(fam), val(tid), val(role), path(tbam), path(tbai), val(nid), path(nbam), path(nbai)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), val(tid), path("${tid}.deepsomatic.vcf.gz"), emit: vcf
    script:
    def model = params.data_type == 'wgs' ? 'WGS' : 'WES'
    """
    run_deepsomatic --model_type=${model} --ref=${fasta} \\
        --reads_tumor=${tbam} --reads_normal=${nbam} --sample_name_tumor=${tid} --sample_name_normal=${nid} \\
        --output_vcf=${tid}.deepsomatic.vcf.gz --num_shards=${task.cpus} --intermediate_results_dir=tmp
    """
    stub:
    "touch ${tid}.deepsomatic.vcf.gz"
}

process CNVKIT {
    tag "${tid}"
    label 'medium'
    container params.containers.cnvkit
    input:
    tuple val(fam), val(tid), val(role), path(tbam), path(tbai), val(nid), path(nbam), path(nbai), path(snv_vcf)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), val(tid), path("${tid}.segments.tsv"), emit: segments
    script:
    def method = params.data_type == 'wgs' ? '-m wgs' : "-t ${params.targets}"
    def purity = params.tumor_purity ? "--purity ${params.tumor_purity}" : ''
    """
    cnvkit.py batch ${tbam} --normal ${nbam} ${method} -f ${fasta} --output-dir cnv -p ${task.cpus}
    cnvkit.py call cnv/*.cns -m clonal ${purity} -v ${snv_vcf} -i ${tid} -n ${nid} -o ${tid}.call.cns
    # -> chrom start end total_cn minor_cn (trio-dnm somatic --segments)
    awk -F'\\t' 'NR==1{for(i=1;i<=NF;i++)h[\$i]=i; print "chrom\\tstart\\tend\\ttotal_cn\\tminor_cn"; next}
        {cn1=\$h["cn1"]; cn2=\$h["cn2"]; m=(cn1<cn2?cn1:cn2); if(cn1=="")m="";
         print \$h["chromosome"]"\\t"\$h["start"]"\\t"\$h["end"]"\\t"\$h["cn"]"\\t"m}' ${tid}.call.cns > ${tid}.segments.tsv
    """
    stub:
    "echo -e 'chrom\\tstart\\tend\\ttotal_cn\\tminor_cn' > ${tid}.segments.tsv"
}

process TRIO_DNM_SOMATIC {
    tag "${tid}"
    label 'small'
    publishDir "${params.outdir}/${fam}/somatic", mode: 'copy'
    input:
    tuple val(fam), val(tid), val(role), val(nid), path(m2_vep), path(strelka), path(deepsomatic), path(segments), path(trio_vcf), path(trio_tbi), path(germline_annot), val(dad), val(mom)
    tuple path(fasta), path(fai), path(dict)
    output:
    path "${tid}.*"
    script:
    def mode = role == 'tissue' ? 'tissue' : 'tumor'
    def gt = params.gene_table ? "--gene-table ${params.gene_table}" : ''
    def sig = params.cosmic_sbs ? "--signatures ${params.cosmic_sbs}" : ''
    def pur = params.tumor_purity ? "--purity ${params.tumor_purity}" : ''
    def ffpe = params.ffpe ? '--ffpe' : ''
    def parents = "--father ${dad} --mother ${mom} --trio-vcf ${trio_vcf}"
    """
    trio-dnm somatic --vcf ${m2_vep} --tumor ${tid} --normal ${nid} --mode ${mode} --out ${tid} \\
        --caller strelka2=${strelka.join(',')} --caller deepsomatic=${deepsomatic} --segments ${segments} \\
        --germline-annotated ${germline_annot} --fasta ${fasta} --data-type ${params.data_type} \\
        ${parents} ${gt} ${sig} ${pur} ${ffpe}
    """
    stub:
    "touch ${tid}.somatic.tsv"
}

workflow SOMATIC_PAIRED {
    take:
    pairs          // (fam, tid, role, tbam, tbai, nid, nbam, nbai)
    trio_vcf       // (fam, vcf, tbi)
    germline_annot // (fam, annotated.tsv)
    parent_ids     // (fam, father_id, mother_id)
    ref
    main:
    m2 = MUTECT2_PAIRED(pairs, ref)
    st = STRELKA2(pairs, ref)
    ds = DEEPSOMATIC(pairs, ref)
    cn = CNVKIT(
        pairs.map { fam, tid, role, tb, ti, nid, nb, ni -> tuple(tid, fam, role, tb, ti, nid, nb, ni) }
            .join(m2.vcf.map { fam, tid, v -> tuple(tid, v) })
            .map { tid, fam, role, tb, ti, nid, nb, ni, v -> tuple(fam, tid, role, tb, ti, nid, nb, ni, v) },
        ref)
    vep = VEP_SOMATIC(m2.vcf.map { fam, tid, v -> tuple(tid, v) }, ref)
    keyed = pairs.map { fam, tid, role, tb, ti, nid, nb, ni -> tuple(tid, fam, role, nid) }
        .join(vep.vcf)
        .join(st.vcf.map { fam, tid, v -> tuple(tid, v) })
        .join(ds.vcf.map { fam, tid, v -> tuple(tid, v) })
        .join(cn.segments.map { fam, tid, s -> tuple(tid, s) })
        .map { tid, fam, role, nid, m, s, d, seg -> tuple(fam, tid, role, nid, m, s, d, seg) }
        .combine(trio_vcf, by: 0)
        .combine(germline_annot, by: 0)
        .combine(parent_ids, by: 0)
    TRIO_DNM_SOMATIC(keyed, ref)
}

// --------------------------------------------------------------------------- //

process MUTECT2_CH {
    tag "$id"
    label 'medium'
    container params.containers.gatk
    input:
    tuple val(fam), val(id), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(id), path("${id}.ch.vcf.gz"), emit: vcf
    script:
    """
    gatk Mutect2 -R ${fasta} -I ${bam} -tumor ${id} -L ${params.ch_genes_bed} -ip 10 \\
        --germline-resource ${params.germline_resource} --panel-of-normals ${params.m2_pon} -O u.vcf.gz
    gatk FilterMutectCalls -R ${fasta} -V u.vcf.gz -O f.vcf.gz
    gatk LeftAlignAndTrimVariants -R ${fasta} -V f.vcf.gz --split-multi-allelics -O s.vcf.gz
    gatk SelectVariants -V s.vcf.gz --exclude-filtered -O ${id}.ch.vcf.gz
    """
    stub:
    "touch ${id}.ch.vcf.gz"
}

process TRIO_DNM_CH {
    tag "$id"
    label 'small'
    publishDir "${params.outdir}/ch_screen", mode: 'copy'
    input:
    tuple val(id), path(vcf)
    output:
    path "${id}.ch_*"
    script:
    """
    trio-dnm ch-screen --vcf ${vcf} --samples ${id} --out ${id}
    """
    stub:
    "touch ${id}.ch_screen.tsv"
}

workflow CH_SCREEN {
    take:
    samples   // (fam, id, bam, bai)
    ref
    main:
    m2 = MUTECT2_CH(samples, ref)
    vep = VEP_CH(m2.vcf, ref)
    TRIO_DNM_CH(vep.vcf)
}
