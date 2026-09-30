// Stages 4-8 — independent DNM callers at candidate sites, the `trio-dnm call`
// filter cascade, VEP annotation, parent-of-origin phasing and `trio-dnm annotate`.

process CANDIDATE_SITES {
    // Union of Mendelian-violation sites from GATK and DeepTrio. DeNovoGear and
    // TrioDeNovo run only at these sites, which keeps them tractable on WGS.
    tag "$fam"
    label 'small'
    container params.containers.bcftools
    input:
    tuple val(fam), path(gatk_vcf), path(gatk_tbi), path(dt_bcf), val(kid), val(dad), val(mom)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), path("${fam}.mv_sites.tsv"), path("${fam}.cand.vcf"), path("${fam}.deeptrio.dnm.vcf.gz"), emit: sites
    script:
    def expr = 'GT[0]!="RR" && GT[0]!="mis" && GT[1]="RR" && GT[2]="RR"'
    """
    bcftools norm -m -any -f ${fasta} ${dt_bcf} -Ou | bcftools view -s ${kid},${dad},${mom} -Oz -o dt.vcf.gz
    for v in ${gatk_vcf} dt.vcf.gz; do
        bcftools view -s ${kid},${dad},${mom} \$v -Ou | bcftools view -i '${expr}' -Ou | bcftools query -f '%CHROM\\t%POS\\n'
    done | sort -u -k1,1 -k2,2n > ${fam}.mv_sites.tsv
    bcftools view -T ${fam}.mv_sites.tsv ${gatk_vcf} -Ov -o ${fam}.cand.vcf
    bcftools view -i '${expr}' -f PASS,. dt.vcf.gz -Oz -o ${fam}.deeptrio.dnm.vcf.gz
    """
    stub:
    "touch ${fam}.mv_sites.tsv ${fam}.cand.vcf ${fam}.deeptrio.dnm.vcf.gz"
}

process MPILEUP_CANDIDATES {
    tag "$fam"
    label 'small'
    container params.containers.bcftools
    input:
    tuple val(fam), path(sites), val(ids), path(bams), path(bais)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), path("${fam}.cand.bcf"), emit: bcf
    script:
    """
    bcftools mpileup -f ${fasta} -T ${sites} -a AD,DP -q 20 -Q 20 -Ob -o ${fam}.cand.bcf ${bams.join(' ')}
    """
    stub:
    "touch ${fam}.cand.bcf"
}

process DENOVOGEAR {
    tag "$fam"
    label 'small'
    container params.containers.denovogear
    input:
    tuple val(fam), path(bcf), path(ped)
    output:
    tuple val(fam), path("${fam}.dng.out"), emit: calls
    script:
    """
    dng dnm auto --ped ${ped} --bcf ${bcf} > ${fam}.dng.out
    """
    stub:
    "touch ${fam}.dng.out"
}

process TRIODENOVO {
    tag "$fam"
    label 'small'
    container params.containers.triodenovo
    input:
    tuple val(fam), path(cand_vcf), path(ped)
    output:
    tuple val(fam), path("${fam}.triodenovo.vcf"), emit: calls
    script:
    """
    triodenovo --ped ${ped} --in_vcf ${cand_vcf} --out_vcf ${fam}.triodenovo.vcf --minDQ ${params.triodenovo_min_dq}
    """
    stub:
    "touch ${fam}.triodenovo.vcf"
}

process TRIO_DNM_CALL {
    tag "$fam"
    label 'small'
    publishDir "${params.outdir}/${fam}/dnm", mode: 'copy'
    input:
    tuple val(fam), path(vcf), path(tbi), path(ped), val(kid), path(dng), path(tdn), path(dt)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), path("${fam}.dnm.vcf"), emit: vcf
    tuple val(fam), path("${fam}.call_summary.json"), emit: summary
    path "${fam}.candidates.tsv"
    script:
    def beds = params.exclude_beds.collect { "--exclude-bed ${it}" }.join(' ')
    def pon = params.pon ? "--pon ${params.pon}" : ''
    def rec = params.recurrence ? "--recurrence ${params.recurrence}" : ''
    def cfg = params.dnm_config ? "--config ${params.dnm_config}" : ''
    """
    dng2tsv.py --min-pp ${params.dng_min_pp} ${dng} > ${fam}.dng.tsv
    trio-dnm call --vcf ${vcf} --ped ${ped} --proband ${kid} --out ${fam} --fasta ${fasta} \\
        --data-type ${params.data_type} --build ${params.build} ${cfg} ${beds} ${pon} ${rec} \\
        --caller denovogear=${fam}.dng.tsv --caller triodenovo=${tdn} --caller deeptrio=${dt}
    """
    stub:
    "touch ${fam}.dnm.vcf ${fam}.candidates.tsv; echo '{}' > ${fam}.call_summary.json"
}

workflow DNM_DETECTION {
    take:
    trio_vcf       // (fam, vcf, tbi)
    deeptrio_bcf   // (fam, bcf)
    trios          // (fam, k, d, m, ped)
    ref
    main:
    ids = trios.map { fam, k, d, m, ped -> tuple(fam, k.id, d.id, m.id) }
    peds = trios.map { fam, k, d, m, ped -> tuple(fam, ped) }
    bams = trios.map { fam, k, d, m, ped -> tuple(fam, [k.id, d.id, m.id], [k.bam, d.bam, m.bam], [k.bai, d.bai, m.bai]) }
    cs = CANDIDATE_SITES(trio_vcf.join(deeptrio_bcf).join(ids), ref)
    mp = MPILEUP_CANDIDATES(cs.sites.map { fam, s, c, d -> tuple(fam, s) }.join(bams), ref)
    dng = DENOVOGEAR(mp.bcf.join(peds))
    tdn = TRIODENOVO(cs.sites.map { fam, s, c, d -> tuple(fam, c) }.join(peds))
    call_in = trio_vcf
        .join(trios.map { fam, k, d, m, ped -> tuple(fam, ped, k.id) })
        .join(dng.calls).join(tdn.calls)
        .join(cs.sites.map { fam, s, c, d -> tuple(fam, d) })
    res = TRIO_DNM_CALL(call_in, ref)
    emit:
    dnm_vcf = res.vcf
    call_summary = res.summary
}

// --------------------------------------------------------------------------- //

process VEP {
    tag "$id"
    label 'medium'
    container params.containers.vep
    input:
    tuple val(id), path(vcf)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(id), path("${id}.vep.vcf"), emit: vcf
    script:
    def p = params.vep_plugins
    def merged = params.vep_merged_cache ? '--merged' : ''
    """
    vep --input_file ${vcf} --output_file ${id}.vep.vcf --vcf --force_overwrite \\
        --offline --cache --dir_cache ${params.vep_cache} --dir_plugins ${params.vep_plugin_dir} \\
        --assembly ${params.build} --fasta ${fasta} ${merged} --fork ${task.cpus} \\
        --everything --mane --flag_pick_allele_gene \\
        --pick_order mane_select,mane_plus_clinical,canonical,appris,tsl,biotype,ccds,rank,length \\
        --plugin LoF,loftee_path:${p.loftee_dir},human_ancestor_fa:${p.loftee_ancestor},conservation_file:${p.loftee_sql} \\
        --plugin CADD,snv=${p.cadd_snv},indels=${p.cadd_indel} \\
        --plugin REVEL,file=${p.revel} \\
        --plugin AlphaMissense,file=${p.alphamissense} \\
        --plugin SpliceAI,snv=${p.spliceai_snv},indel=${p.spliceai_indel},cutoff=0.0 \\
        --plugin dbNSFP,${p.dbnsfp},BayesDel_noAF_score,PrimateAI_score,ESM1b_score,EVE_score,MetaRNN_score,phyloP100way_vertebrate,GERP++_RS \\
        --plugin UTRAnnotator,file=${p.utrannotator} \\
        --plugin NMD \\
        --custom file=${p.gnomad},short_name=gnomADv4,format=vcf,type=exact,coords=0,fields=AF_joint%AF_joint_grpmax%nhomalt_joint \\
        --custom file=${p.clinvar},short_name=ClinVar,format=vcf,type=exact,coords=0,fields=CLNSIG%CLNREVSTAT%CLNDN \\
        --custom file=${p.ccre_bed},short_name=ENCODE_cCRE,format=bed,type=overlap
    # expose the gnomAD joint AF under the name trio-dnm looks for
    sed -i 's/gnomADv4_AF_joint/gnomADv4_AF/g' ${id}.vep.vcf
    """
    stub:
    "cp ${vcf} ${id}.vep.vcf"
}

process UNFAZED {
    // Read-backed + transmission phasing of DNMs to parent of origin.
    tag "$fam"
    label 'small'
    container params.containers.unfazed
    input:
    tuple val(fam), path(dnm_vcf), path(trio_vcf), path(tbi), path(ped), val(ids), path(bams), path(bais)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), path("${fam}.poo.bed"), emit: poo
    script:
    """
    unfazed -d ${dnm_vcf} -s ${trio_vcf} -p ${ped} -r ${fasta} -g ${params.build == 'GRCh38' ? 38 : 37} \\
        --bam-pairs ${[ids, bams].transpose().flatten().join(' ')} \\
        -o bed > ${fam}.poo.bed
    """
    stub:
    "touch ${fam}.poo.bed"
}

process TRIO_DNM_ANNOTATE {
    tag "$fam"
    label 'small'
    publishDir "${params.outdir}/${fam}/report", mode: 'copy'
    input:
    tuple val(fam), path(vep_vcf), path(summary), path(poo), val(ages)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), path("${fam}.annotated.tsv"), emit: annotated
    path "${fam}.*"
    script:
    def gt = params.gene_table ? "--gene-table ${params.gene_table}" : ''
    def sig = params.cosmic_sbs ? "--signatures ${params.cosmic_sbs}" : ''
    def ph = params.phenotype_scores ? "--phenotype-scores ${params.phenotype_scores}/${fam}.tsv" : ''
    def val = params.validated ? "--validated ${params.validated}" : ''
    def age = ages ? "--paternal-age ${ages[0]} --maternal-age ${ages[1]}" : ''
    def cfg = params.dnm_config ? "--config ${params.dnm_config}" : ''
    """
    trio-dnm annotate --vcf ${vep_vcf} --out ${fam} --fasta ${fasta} --data-type ${params.data_type} \\
        --call-summary ${summary} --parent-of-origin ${poo} --parentage-confirmed \\
        ${cfg} ${gt} ${sig} ${ph} ${val} ${age}
    """
    stub:
    "touch ${fam}.annotated.tsv ${fam}.report.html"
}

workflow DNM_ANNOTATION {
    take:
    dnm_vcf        // (fam, vcf)
    call_summary   // (fam, json)
    trio_vcf       // (fam, vcf, tbi)
    trios          // (fam, k, d, m, ped)
    ref
    main:
    vep = VEP(dnm_vcf, ref)
    poo = UNFAZED(
        dnm_vcf.join(trio_vcf).join(trios.map { fam, k, d, m, ped -> tuple(fam, ped, [k.id, d.id, m.id], [k.bam, d.bam, m.bam], [k.bai, d.bai, m.bai]) }), ref)
    ages = Channel.value(params.parental_ages ?: [:])
    ann_in = vep.vcf.join(call_summary).join(poo.poo)
        .combine(ages).map { fam, v, s, p, a -> tuple(fam, v, s, p, a[fam]) }
    res = TRIO_DNM_ANNOTATE(ann_in, ref)
    emit:
    annotated = res.annotated
}
