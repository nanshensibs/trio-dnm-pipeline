#!/usr/bin/env nextflow
/*
 * Trio DNM pipeline v2.0 — germline DNM, mosaic, annotation and somatic tracks.
 *
 *   nextflow run main.nf -profile docker -params-file params.yaml
 *
 * Samplesheet (CSV): family_id,sample_id,role,sex,bam
 *   role ∈ proband|father|mother|tumor|tissue ; sex ∈ male|female
 *   bam  = analysis-ready (dedup + BQSR) BAM/CRAM with index alongside.
 *   Use the ALIGN subworkflow (modules/align.nf) upstream when starting from FASTQ.
 */
nextflow.enable.dsl = 2

include { SAMPLE_QC                                   } from './modules/qc'
include { GERMLINE_CALLING                            } from './modules/germline'
include { DNM_DETECTION ; DNM_ANNOTATION              } from './modules/dnm'
include { MOSAIC_TRACK                                } from './modules/mosaic'
include { SOMATIC_PAIRED ; CH_SCREEN                  } from './modules/somatic'

workflow {
    ref = Channel.value(tuple(file(params.fasta), file("${params.fasta}.fai"), file(params.fasta.replaceAll(/\.fa(sta)?(\.gz)?$/, '.dict'))))

    rows = Channel.fromPath(params.samplesheet)
        .splitCsv(header: true)
        .map { r -> tuple(r.family_id, r.sample_id, r.role, r.sex, file(r.bam), file(r.bam + (r.bam.endsWith('.cram') ? '.crai' : '.bai'))) }

    germ = rows.filter { it[2] in ['proband', 'father', 'mother'] }

    // family_id -> [proband, father, mother] BAM tuples, ordered
    trios = germ
        .map { fam, id, role, sex, bam, bai -> tuple(fam, [role: role, id: id, sex: sex, bam: bam, bai: bai]) }
        .groupTuple()
        .map { fam, members ->
            def by = members.collectEntries { [(it.role): it] }
            tuple(fam, by.proband, by.father, by.mother)
        }

    // Stage 1 — pre-flight QC with hard stop-gates
    qc = SAMPLE_QC(germ, trios, ref)

    // Stage 2/3 — joint trio calling (GATK + DeepTrio), CGP, normalisation
    calls = GERMLINE_CALLING(qc.passed_trios, ref)

    // Stage 4/5 — multi-caller DNM detection and the layered filter cascade
    dnm = DNM_DETECTION(calls.trio_vcf, calls.deeptrio_vcf, qc.passed_trios, ref)

    // Stage 7 — mosaic-aware track (MosaicForecast on the proband)
    if (params.run_mosaic) {
        MOSAIC_TRACK(qc.passed_trios, ref)
    }

    // Stage 8 — annotation, ACMG, prioritisation, sanity checks, report
    ann = DNM_ANNOTATION(dnm.dnm_vcf, dnm.call_summary, calls.trio_vcf, qc.passed_trios, ref)

    // Stage 10 — somatic analyses
    if (params.run_somatic) {
        tumors = rows.filter { it[2] in ['tumor', 'tissue'] }
            .map { fam, id, role, sex, bam, bai -> tuple(fam, id, role, bam, bai) }
        normals = germ.filter { it[2] == 'proband' }
            .map { fam, id, role, sex, bam, bai -> tuple(fam, id, bam, bai) }
        pairs = tumors.combine(normals, by: 0)
        parent_ids = qc.passed_trios.map { fam, k, d, m, ped -> tuple(fam, d.id, m.id) }
        SOMATIC_PAIRED(pairs, calls.trio_vcf, ann.annotated, parent_ids, ref)
    }
    if (params.run_ch_screen) {
        CH_SCREEN(germ.map { fam, id, role, sex, bam, bai -> tuple(fam, id, bam, bai) }, ref)
    }
}
