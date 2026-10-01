// Stages 4-5 (protocol §6-§7) — independent DNM callers at candidate sites and the
// `trio-dnm call` filter cascade; Stages 6 + 8 (§8, §10) — VEP annotation,
// parent-of-origin phasing and `trio-dnm annotate` (sanity checks, ACMG, ranking).

include { NORMALISE as NORMALISE_DEEPTRIO ; VCFANNO as VCFANNO_DEEPTRIO ; BGZIP_INDEX as BGZIP_INDEX_DEEPTRIO } from './germline'
include { asList ; trioDnmEnv } from './utils'

process CANDIDATE_SITES {
    // Union of Mendelian-violation sites from GATK and DeepTrio (both normalised, samples
    // ordered proband, father, mother). DeNovoGear and TrioDeNovo run only at these sites,
    // which keeps them tractable on WGS.
    tag "$fam"
    label 'small'
    container params.containers.bcftools
    input:
    tuple val(fam), path(gatk_vcf), path(gatk_tbi), path(dt_vcf), path(dt_tbi), val(kid), val(dad), val(mom)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), path("${fam}.mv_sites.tsv"), path("${fam}.cand.vcf"), path("${fam}.deeptrio.dnm.vcf.gz"), emit: sites
    script:
    def expr = 'GT[0]!="RR" && GT[0]!="mis" && GT[1]="RR" && GT[2]="RR"'
    """
    for v in ${gatk_vcf} ${dt_vcf}; do
        bcftools view -s ${kid},${dad},${mom} \$v -Ou | bcftools view -i '${expr}' -Ou | bcftools query -f '%CHROM\\t%POS\\n'
    done | sort -u -k1,1 -k2,2n > ${fam}.mv_sites.tsv
    # Joint-called GATK records (with PL) at the union sites for DeNovoGear / TrioDeNovo;
    # spanning-deletion '*' alleles are not valid input for either caller.
    if [ -s ${fam}.mv_sites.tsv ]; then
        bcftools view -T ${fam}.mv_sites.tsv -e 'ALT="*"' ${gatk_vcf} -Ov -o ${fam}.cand.vcf
    else
        bcftools view -h ${gatk_vcf} > ${fam}.cand.vcf
    fi
    bcftools view -s ${kid},${dad},${mom} -i '${expr}' -f PASS,. ${dt_vcf} -Oz -o ${fam}.deeptrio.dnm.vcf.gz
    """
    stub:
    "touch ${fam}.mv_sites.tsv ${fam}.cand.vcf ${fam}.deeptrio.dnm.vcf.gz"
}

process MPILEUP_CANDIDATES {
    // Per-strand allele counts (FORMAT/ADF,ADR) at the candidate sites for Layer 2
    // (`trio-dnm call --strand-vcf`): GenotypeGVCFs output has no ADF/ADR. -R uses the
    // BAM/CRAM index to jump to each site; -G names each sample after its samplesheet ID.
    tag "$fam"
    label 'small'
    container params.containers.bcftools
    input:
    tuple val(fam), path(sites), val(ids), path(bams), path(bais)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), path("${fam}.cand.strand.vcf"), emit: vcf
    script:
    def rg = [ids, asList(bams)].transpose().collect { id, b -> "${b} ${id}" }.join(' ')
    """
    printf '* %s %s\\n' ${rg} > read_groups.txt
    bcftools mpileup -f ${fasta} -R ${sites} -G read_groups.txt -a AD,ADF,ADR,DP -q 20 -Q 20 -d 10000 \\
        -Ov -o ${fam}.cand.strand.vcf ${asList(bams).join(' ')}
    """
    stub:
    "touch ${fam}.cand.strand.vcf"
}

process DENOVOGEAR {
    // DNG reads the joint-called candidate VCF (PL + FORMAT/DP) through its --vcf path;
    // its --bcf path only reads the legacy samtools-0.1.x BCF1 format.
    tag "$fam"
    label 'small'
    container params.containers.denovogear
    input:
    tuple val(fam), path(cand_vcf), path(ped)
    output:
    tuple val(fam), path("${fam}.dng.out"), emit: calls
    script:
    """
    dng dnm auto --ped ${ped} --vcf ${cand_vcf} > ${fam}.dng.out
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
    container params.containers.trio_dnm
    // Exit 3 = raw Mendelian error stop-gate failed (§3.3): halt the run. With
    // qc_halt=false `--no-halt` is passed and the gate is reported only.
    errorStrategy { task.exitStatus == 3 ? 'terminate' : (task.exitStatus in [104, 134, 137, 139, 143] ? 'retry' : 'finish') }
    publishDir "${params.outdir}/${fam}/dnm", mode: params.publish_dir_mode
    input:
    tuple val(fam), path(vcf), path(tbi), path(ped), val(kid), path(dng), path(tdn), path(dt_dnm), path(dt_vcf), path(dt_tbi), path(strand)
    tuple path(fasta), path(fai), path(dict)
    path(pkg, stageAs: 'pylib/trio_dnm')
    path(cfg)
    path(beds, stageAs: 'exclude_beds/*')
    path(pon)
    path(rec)
    output:
    tuple val(fam), path("${fam}.dnm.vcf"), emit: vcf
    tuple val(fam), path("${fam}.call_summary.json"), emit: summary
    path "${fam}.candidates.tsv"
    script:
    def bed_args = asList(beds).collect { "--exclude-bed ${it}" }.join(' ')
    def pon_arg = pon ? "--pon ${pon}" : ''
    def rec_arg = rec ? "--recurrence ${rec}" : ''
    def config = cfg ? "--config ${cfg}" : ''
    def halt = params.qc_halt ? '' : '--no-halt'
    """
    ${trioDnmEnv()}
    dng2tsv.py --min-pp ${params.dng_min_pp} ${dng} > ${fam}.dng.tsv
    trio-dnm call --vcf ${vcf} --ped ${ped} --proband ${kid} --out ${fam} --fasta ${fasta} \\
        --data-type ${params.data_type} --build ${params.build} ${config} ${bed_args} ${pon_arg} ${rec_arg} \\
        --caller denovogear=${fam}.dng.tsv --caller triodenovo=${tdn} --caller deeptrio=${dt_dnm} \\
        --extra-vcf ${dt_vcf} --strand-vcf ${strand} ${halt}
    """
    stub:
    "touch ${fam}.dnm.vcf ${fam}.candidates.tsv; echo '{}' > ${fam}.call_summary.json"
}

workflow DNM_DETECTION {
    take:
    trio_vcf       // (fam, vcf, tbi) — GATK, normalised + vcfanno
    deeptrio_bcf   // (fam, bcf)      — GLnexus, raw
    trios          // (fam, k, d, m, ped)
    ref
    res            // staged resources (main.nf)
    main:
    ids = trios.map { fam, k, d, m, ped -> tuple(fam, k.id, d.id, m.id) }
    peds = trios.map { fam, k, d, m, ped -> tuple(fam, ped) }
    bams = trios.map { fam, k, d, m, ped -> tuple(fam, [k.id, d.id, m.id], [k.bam, d.bam, m.bam], [k.bai, d.bai, m.bai]) }

    // The DeepTrio call set gets the same normalisation and vcfanno annotation as the GATK
    // VCF: its Mendelian-violation records missing from the GATK VCF are evaluated by
    // `trio-dnm call --extra-vcf` (flag SECOND_ENGINE_ONLY), so the GATK ∪ DeepTrio union
    // drives candidate discovery (§5.3), not only caller support.
    dt_norm = NORMALISE_DEEPTRIO(deeptrio_bcf.join(ids).map { fam, bcf, k, d, m -> tuple(fam, 'deeptrio', bcf, [k, d, m]) }, ref)
    dt_anno = VCFANNO_DEEPTRIO(dt_norm.vcf, res.vcfanno_toml, res.vcfanno_files)
    dt_vcf = BGZIP_INDEX_DEEPTRIO(dt_anno.vcf).vcf.map { fam, label, v, t -> tuple(fam, v, t) }

    cs = CANDIDATE_SITES(trio_vcf.join(dt_vcf).join(ids), ref)
    cand = cs.sites.map { fam, s, c, d -> tuple(fam, c) }
    mp = MPILEUP_CANDIDATES(cs.sites.map { fam, s, c, d -> tuple(fam, s) }.join(bams), ref)
    dng = DENOVOGEAR(cand.join(peds))
    tdn = TRIODENOVO(cand.join(peds))
    call_in = trio_vcf
        .join(trios.map { fam, k, d, m, ped -> tuple(fam, ped, k.id) })
        .join(dng.calls).join(tdn.calls)
        .join(cs.sites.map { fam, s, c, d -> tuple(fam, d) })
        .join(dt_vcf)
        .join(mp.vcf)
    res_call = TRIO_DNM_CALL(call_in, ref, res.pkg, res.dnm_config, res.exclude_beds, res.pon, res.recurrence)
    emit:
    dnm_vcf = res_call.vcf
    call_summary = res_call.summary
}

// --------------------------------------------------------------------------- //

process VEP {
    tag "$id"
    label 'medium'
    container params.containers.vep
    input:
    tuple val(id), path(vcf)
    tuple path(fasta), path(fai), path(dict)
    path(cache, stageAs: 'vep_cache')
    path(plugin_dir, stageAs: 'vep_plugins')
    path(vep_files, stageAs: 'vep_res/*')   // plugin / --custom data with their indexes
    output:
    tuple val(id), path("${id}.vep.vcf"), emit: vcf
    script:
    def p = params.vep_plugins ?: [:]
    def r = { k -> "vep_res/${p[k].toString().tokenize('/')[-1]}" }   // staged name of a plugin file
    def kv = { Map m -> m.findAll { k, v -> p[v] }.collect { k, v -> "${k}${r(v)}" }.join(',') }
    def plugins = []
    if (p.loftee_dir) plugins << "--plugin LoF," + kv(['loftee_path:': 'loftee_dir', 'human_ancestor_fa:': 'loftee_ancestor', 'conservation_file:': 'loftee_sql'])
    if (p.cadd_snv || p.cadd_indel) plugins << "--plugin CADD," + kv(['snv=': 'cadd_snv', 'indels=': 'cadd_indel'])
    if (p.revel) plugins << "--plugin REVEL,file=${r('revel')}"
    if (p.alphamissense) plugins << "--plugin AlphaMissense,file=${r('alphamissense')}"
    if (p.spliceai_snv || p.spliceai_indel) plugins << "--plugin SpliceAI," + kv(['snv=': 'spliceai_snv', 'indel=': 'spliceai_indel']) + ',cutoff=0.0'
    if (p.dbnsfp) plugins << "--plugin dbNSFP,${r('dbnsfp')},BayesDel_noAF_score,PrimateAI_score,ESM1b_score,EVE_score,MetaRNN_score,phyloP100way_vertebrate,GERP++_RS"
    if (p.utrannotator) plugins << "--plugin UTRAnnotator,file=${r('utrannotator')}"
    plugins << '--plugin NMD'
    // gnomAD v4.1 joint sites: the grpmax field is AF_grpmax_joint. CSQ fields are named
    // <short_name>_<field> (gnomADv4_AF_joint, ...), which trio-dnm annotate reads directly.
    if (p.gnomad) plugins << "--custom file=${r('gnomad')},short_name=gnomADv4,format=vcf,type=exact,coords=0,fields=AF_joint%AF_grpmax_joint%nhomalt_joint"
    if (p.clinvar) plugins << "--custom file=${r('clinvar')},short_name=ClinVar,format=vcf,type=exact,coords=0,fields=CLNSIG%CLNREVSTAT%CLNDN"
    if (p.ccre_bed) plugins << "--custom file=${r('ccre_bed')},short_name=ENCODE_cCRE,format=bed,type=overlap"
    def merged = params.vep_merged_cache ? '--merged' : ''
    def pdir = plugin_dir ? "--dir_plugins ${plugin_dir}" : ''
    """
    vep --input_file ${vcf} --output_file ${id}.vep.vcf --vcf --force_overwrite \\
        --offline --cache --dir_cache ${cache} ${pdir} \\
        --assembly ${params.build} --fasta ${fasta} ${merged} --fork ${task.cpus} \\
        --everything --mane --flag_pick_allele_gene \\
        --pick_order mane_select,mane_plus_clinical,canonical,appris,tsl,biotype,ccds,rank,length \\
        ${plugins.join(' \\\n        ')}
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
    tuple val(fam), path(dnm_vcf), path(trio_vcf), path(tbi), path(ped), val(kid), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    output:
    tuple val(fam), path("${fam}.poo.bed"), emit: poo
    script:
    // --bam-pairs takes {sample_id}:{bam_path} tokens; only the offspring's reads are used.
    """
    unfazed -d ${dnm_vcf} -s ${trio_vcf} -p ${ped} -r ${fasta} -g ${params.build == 'GRCh38' ? 38 : 37} \\
        --bam-pairs ${kid}:${bam} -t ${task.cpus} \\
        -o bed > ${fam}.poo.bed
    """
    stub:
    "touch ${fam}.poo.bed"
}

process TRIO_DNM_ANNOTATE {
    tag "$fam"
    label 'small'
    container params.containers.trio_dnm
    publishDir "${params.outdir}/${fam}/report", mode: params.publish_dir_mode
    input:
    tuple val(fam), path(vep_vcf), path(summary), path(poo), val(ages), path(pheno)
    tuple path(fasta), path(fai), path(dict)
    path(pkg, stageAs: 'pylib/trio_dnm')
    path(cfg)
    path(gene_table)
    path(sbs)
    path(validated)
    output:
    tuple val(fam), path("${fam}.annotated.tsv"), emit: annotated
    path "${fam}.*"
    script:
    def gt = gene_table ? "--gene-table ${gene_table}" : ''
    def sig = sbs ? "--signatures ${sbs}" : ''
    def ph = pheno ? "--phenotype-scores ${pheno}" : ''
    def vld = validated ? "--validated ${validated}" : ''
    def age = ages ? "--paternal-age ${ages[0]} --maternal-age ${ages[1]}" : ''
    def config = cfg ? "--config ${cfg}" : ''
    """
    ${trioDnmEnv()}
    trio-dnm annotate --vcf ${vep_vcf} --out ${fam} --fasta ${fasta} --data-type ${params.data_type} \\
        --build ${params.build} --call-summary ${summary} --parent-of-origin ${poo} --parentage-confirmed \\
        ${config} ${gt} ${sig} ${ph} ${vld} ${age}
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
    res            // staged resources (main.nf)
    main:
    vep = VEP(dnm_vcf, ref, res.vep_cache, res.vep_plugin_dir, res.vep_files)
    poo = UNFAZED(
        dnm_vcf.join(trio_vcf).join(trios.map { fam, k, d, m, ped -> tuple(fam, ped, k.id, k.bam, k.bai) }), ref)
    def ages = params.parental_ages ?: [:]
    // Exomiser/LIRICAL scores: <params.phenotype_scores>/<family>.tsv when present.
    def pheno = { fam ->
        def f = params.phenotype_scores ? file("${params.phenotype_scores}/${fam}.tsv") : null
        f?.exists() ? f : []
    }
    ann_in = vep.vcf.join(call_summary).join(poo.poo)
        .map { fam, v, s, p -> tuple(fam, v, s, p, ages[fam] ?: [], pheno(fam)) }
    res_ann = TRIO_DNM_ANNOTATE(ann_in, ref, res.pkg, res.dnm_config, res.gene_table, res.cosmic_sbs, res.validated)
    emit:
    annotated = res_ann.annotated
}
