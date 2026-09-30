"""`trio-dnm demo`: build a small synthetic trio and run every stage on it.

Everything here is simulated — the reference, reads, VEP-style annotations,
gene table and "signatures" are toy data chosen to exercise each code path
(true DNM, parental leakage, parental mosaic, proband mosaic, common variant,
masked region, homopolymer, low MQ, inherited, male chrX hemizygous DNM).
Use it to see the output formats, not to learn anything about biology.
"""
from __future__ import annotations

import os
import random
from typing import Dict

VCF_HEADER = """##fileformat=VCFv4.2
##source=trio-dnm-demo (SIMULATED DATA)
##contig=<ID=chr1,length=2000>
##contig=<ID=chrX,length=4000000>
##INFO=<ID=MQ,Number=1,Type=Float,Description="RMS mapping quality">
##INFO=<ID=FS,Number=1,Type=Float,Description="Fisher strand">
##INFO=<ID=SOR,Number=1,Type=Float,Description="Strand odds ratio">
##INFO=<ID=gnomAD_AF,Number=A,Type=Float,Description="gnomAD v4.1 joint AF (simulated)">
##INFO=<ID=hiConfDeNovo,Number=1,Type=String,Description="High-confidence possible de novo">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled likelihoods">
##FORMAT=<ID=SB,Number=4,Type=Integer,Description="Per-sample strand bias">
"""

ALT_OF = {"A": "G", "C": "T", "G": "A", "T": "C"}


def write_fasta(path: str, seqs: Dict[str, str]) -> None:
    with open(path, "w") as fh, open(path + ".fai", "w") as fai:
        offset = 0
        for name, seq in seqs.items():
            head = f">{name}\n"
            fh.write(head)
            offset += len(head)
            lines = [seq[i:i + 60] for i in range(0, len(seq), 60)]
            fai.write(f"{name}\t{len(seq)}\t{offset}\t60\t61\n")
            for ln in lines:
                fh.write(ln + "\n")
            offset += sum(len(ln) + 1 for ln in lines)


def genotype(g: str, ref: int, alt: int, gq: int = 99) -> str:
    """FORMAT string with approximately Q30-read PLs and a strand split."""
    dp = ref + alt
    pl = {"0/0": f"0,{3 * dp},{30 * dp}", "0/1": f"{30 * alt},0,{30 * ref}", "1/1": f"{30 * dp},{3 * dp},0"}[g]
    sb = f"{ref // 2},{ref - ref // 2},{alt // 2},{alt - alt // 2}"
    return f"{g}:{ref},{alt}:{dp}:{gq}:{pl}:{sb}"


def build_synthetic_trio(outdir: str, seed: int = 7) -> Dict[str, object]:
    """Write ref.fa(+.fai), trio.vcf, trio.ped and mask.bed; return paths and the
    reference lookup used to build them."""
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)
    chr1 = [rng.choice("ACGT") for _ in range(2000)]
    chr1[98], chr1[99], chr1[100] = "A", "C", "G"  # CpG at 100-101
    for i in range(499, 509):  # A x10 homopolymer at 500-509
        chr1[i] = "A"
    chr1[509] = "C"
    # make the simulated HGVS in SIM_CSQ agree with the reference bases
    chr1[299], chr1[399] = "G", "A"
    chrx = list(rng.choice("ACGT") for _ in range(4000))
    chrx[99] = "C"  # chrX:3,000,100
    chrx = "".join(chrx)
    seqs = {"chr1": "".join(chr1), "chrX": "N" * 3_000_000 + chrx + "N" * (1_000_000 - 4000)}
    fasta = os.path.join(outdir, "ref.fa")
    write_fasta(fasta, seqs)

    def ref(chrom: str, pos: int) -> str:
        return seqs[chrom][pos - 1]

    rows = []

    def add(chrom, pos, info, kid, dad, mom):
        r = ref(chrom, pos)
        rows.append(f"{chrom}\t{pos}\t.\t{r}\t{ALT_OF[r]}\t500\tPASS\t{info}\tGT:AD:DP:GQ:PL:SB\t{kid}\t{dad}\t{mom}")

    good = "MQ=60;FS=1.2;SOR=0.7;hiConfDeNovo=kid"
    add("chr1", 100, good, genotype("0/1", 15, 14), genotype("0/0", 30, 0), genotype("0/0", 32, 0))
    add("chr1", 200, good, genotype("0/1", 16, 14), genotype("0/0", 28, 3, gq=40), genotype("0/0", 30, 0))
    add("chr1", 300, good, genotype("0/1", 15, 15), genotype("0/0", 30, 0), genotype("0/0", 40, 6, gq=30))
    add("chr1", 400, "MQ=60;FS=1;SOR=0.7", genotype("0/1", 45, 8), genotype("0/0", 40, 0), genotype("0/0", 42, 0))
    add("chr1", 505, good, genotype("0/1", 14, 15), genotype("0/0", 30, 0), genotype("0/0", 30, 0))
    add("chr1", 600, good + ";gnomAD_AF=0.01", genotype("0/1", 14, 15), genotype("0/0", 30, 0), genotype("0/0", 30, 0))
    add("chr1", 700, good, genotype("0/1", 14, 15), genotype("0/0", 30, 0), genotype("0/0", 30, 0))
    add("chr1", 800, good, genotype("0/1", 14, 15), genotype("0/1", 15, 15), genotype("0/0", 30, 0))
    add("chr1", 900, "MQ=30;FS=1;SOR=0.7", genotype("0/1", 14, 15), genotype("0/0", 30, 0), genotype("0/0", 30, 0))
    add("chrX", 3_000_100, "MQ=60;FS=1;SOR=0.7", genotype("1/1", 0, 16), genotype("0/0", 18, 0), genotype("0/0", 30, 0))
    vcf = os.path.join(outdir, "trio.vcf")
    with open(vcf, "w") as fh:
        fh.write(VCF_HEADER + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tkid\tdad\tmom\n" + "\n".join(rows) + "\n")
    ped = os.path.join(outdir, "trio.ped")
    with open(ped, "w") as fh:
        fh.write("F1\tkid\tdad\tmom\t1\t2\nF1\tdad\t0\t0\t1\t1\nF1\tmom\t0\t0\t2\t1\n")
    bed = os.path.join(outdir, "mask.bed")
    with open(bed, "w") as fh:
        fh.write("chr1\t690\t710\n")
    return {"vcf": vcf, "ped": ped, "fasta": fasta, "bed": bed, "ref": ref}


# --------------------------------------------------------------------------- #
# Simulated annotation resources
# --------------------------------------------------------------------------- #
CSQ_FIELDS = ["Allele", "Consequence", "IMPACT", "SYMBOL", "Gene", "Feature", "BIOTYPE", "EXON", "HGVSc", "HGVSp",
              "MANE_SELECT", "CANONICAL", "LoF", "LoF_flags", "REVEL", "am_pathogenicity", "am_class", "CADD_PHRED",
              "SpliceAI_pred_DS_AG", "SpliceAI_pred_DS_AL", "SpliceAI_pred_DS_DG", "SpliceAI_pred_DS_DL",
              "gnomADv4_AF", "ClinVar_CLNSIG", "phyloP100way_vertebrate", "DOMAINS"]

# position -> simulated VEP annotation of the demo DNMs
SIM_CSQ = {
    100: dict(Consequence="stop_gained", IMPACT="HIGH", SYMBOL="SCN2A", Gene="ENSG00000136531", Feature="ENST00000375437",
              EXON="12/27", HGVSc="c.1270C>T", HGVSp="p.Arg424Ter", MANE_SELECT="NM_001040142.2", LoF="HC",
              CADD_PHRED="38", SpliceAI_pred_DS_AG="0.01", phyloP100way_vertebrate="5.2"),
    300: dict(Consequence="missense_variant", IMPACT="MODERATE", SYMBOL="KCNQ2", Gene="ENSG00000075043", Feature="ENST00000359125",
              EXON="5/17", HGVSc="c.793G>A", HGVSp="p.Ala265Thr", MANE_SELECT="NM_172107.4", REVEL="0.95",
              am_pathogenicity="0.991", am_class="likely_pathogenic", CADD_PHRED="29.1", SpliceAI_pred_DS_DL="0.02",
              phyloP100way_vertebrate="7.9", DOMAINS="Pfam:PF00520"),
    400: dict(Consequence="missense_variant", IMPACT="MODERATE", SYMBOL="PIK3CA", Gene="ENSG00000121879", Feature="ENST00000263967",
              EXON="21/21", HGVSc="c.3140A>G", HGVSp="p.His1047Arg", MANE_SELECT="NM_006218.4", REVEL="0.93",
              am_pathogenicity="0.97", am_class="likely_pathogenic", CADD_PHRED="27.5", ClinVar_CLNSIG="Pathogenic"),
    3_000_100: dict(Consequence="synonymous_variant", IMPACT="LOW", SYMBOL="CDKL5", Gene="ENSG00000008086", Feature="ENST00000623535",
                    EXON="9/21", HGVSc="c.744C>T", HGVSp="p.Ser248=", MANE_SELECT="NM_001323289.2",
                    SpliceAI_pred_DS_AG="0.03", phyloP100way_vertebrate="-0.4", CADD_PHRED="8.2"),
}

GENE_TABLE = """gene\tpLI\tLOEUF\tmis_z\tshet\thi_score\tdisease\tinheritance\tmechanism\tcancer_role\thotspots
SCN2A\t1.00\t0.11\t5.90\t0.20\t3\tDevelopmental and epileptic encephalopathy (SIMULATED ENTRY)\tAD\tLoF;GoF\t\t
KCNQ2\t1.00\t0.20\t4.80\t0.15\t3\tDevelopmental and epileptic encephalopathy (SIMULATED ENTRY)\tAD\tLoF;DN\t\t
PIK3CA\t1.00\t0.25\t4.20\t\t0\tPIK3CA-related overgrowth (SIMULATED ENTRY)\tsomatic mosaic\tGoF\toncogene\tp.His1047Arg,p.Glu545Lys
CDKL5\t0.99\t0.30\t3.10\t\t3\tCDKL5 deficiency disorder (SIMULATED ENTRY)\tXLD\tLoF\t\t
TP53\t0.99\t0.45\t1.90\t\t30\tLi-Fraumeni syndrome (SIMULATED ENTRY)\tAD\tLoF;DN\tTSG\tp.Arg175His
NF1\t1.00\t0.15\t2.50\t\t3\tNeurofibromatosis type 1 (SIMULATED ENTRY)\tAD\tLoF\tTSG\t
KRAS\t0.73\t0.52\t3.40\t\t0\tNoonan syndrome (SIMULATED ENTRY)\tAD\tGoF\toncogene\tp.Gly12Asp
"""


def toy_signatures(path: str) -> None:
    """Three TOY signatures shaped like SBS1 (CpG C>T), SBS5 and SBS40 (flat-ish).
    Not COSMIC values — download COSMIC v3.4 for real analyses."""
    from .genome import SBS96

    rows = []
    for ch in SBS96:
        cpg_ct = ch[2:5] == "C>T" and ch[-1] == "G"
        s1 = 0.2 if cpg_ct else 0.0005
        s5 = 0.025 if ch[2:5] in ("C>T", "T>C") else 0.004
        s40 = 0.012 + (0.006 if ch[2:5] in ("C>A", "T>C") else 0.0)
        rows.append((ch, s1, s5, s40))
    tot = [sum(r[i] for r in rows) for i in (1, 2, 3)]
    with open(path, "w") as fh:
        fh.write("Type\tTOY_SBS1\tTOY_SBS5\tTOY_SBS40\n")
        for ch, a, b, c in rows:
            fh.write(f"{ch}\t{a / tot[0]:.6f}\t{b / tot[1]:.6f}\t{c / tot[2]:.6f}\n")


def add_simulated_csq(src: str, dst: str) -> None:
    header_line = ('##INFO=<ID=CSQ,Number=.,Type=String,Description="SIMULATED consequence annotations '
                   'in Ensembl VEP format. Format: ' + "|".join(CSQ_FIELDS) + '">')
    with open(src) as fh, open(dst, "w") as out:
        for line in fh:
            if line.startswith("#CHROM"):
                out.write(header_line + "\n")
            if line.startswith("#"):
                out.write(line)
                continue
            f = line.rstrip("\n").split("\t")
            sim = SIM_CSQ.get(int(f[1]))
            if sim:
                vals = dict(sim, Allele=f[4], BIOTYPE="protein_coding", CANONICAL="YES")
                csq = "|".join(vals.get(k, "") for k in CSQ_FIELDS)
                f[7] = (f[7] + ";" if f[7] not in (".", "") else "") + "CSQ=" + csq
            out.write("\t".join(f) + "\n")


# --------------------------------------------------------------------------- #
# Demo driver
# --------------------------------------------------------------------------- #
SOM_HEADER = """##fileformat=VCFv4.2
##source=trio-dnm-demo (SIMULATED DATA)
##INFO=<ID=CSQ,Number=.,Type=String,Description="SIMULATED consequence annotations. Format: Allele|Consequence|IMPACT|SYMBOL|HGVSp">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">
"""


def run_demo(outdir: str) -> Dict[str, object]:
    from .cli import main

    inp = os.path.join(outdir, "inputs")
    res = os.path.join(outdir, "results")
    os.makedirs(res, exist_ok=True)
    t = build_synthetic_trio(inp)
    ref = t["ref"]
    genes = os.path.join(inp, "gene_table.SIMULATED.tsv")
    with open(genes, "w") as fh:
        fh.write(GENE_TABLE)
    sigs = os.path.join(inp, "toy_signatures.tsv")
    toy_signatures(sigs)
    poo = os.path.join(inp, "parent_of_origin.tsv")
    with open(poo, "w") as fh:
        fh.write("chrom\tpos\torigin\nchr1\t100\tpaternal\n")
    p = lambda name: os.path.join(res, name)  # noqa: E731

    main(["qc-gate", "--ped", t["ped"], "--somalier-pairs", _somalier_pairs(inp), "--somalier-samples", _somalier_samples(inp),
          "--verifybamid", f"kid={_selfsm(inp, 'kid', 0.004)}", "--verifybamid", f"dad={_selfsm(inp, 'dad', 0.006)}",
          "--verifybamid", f"mom={_selfsm(inp, 'mom', 0.003)}", "--out", p("F1.qc_gate.json")])
    main(["call", "--vcf", t["vcf"], "--ped", t["ped"], "--fasta", t["fasta"], "--exclude-bed", t["bed"],
          "--mean-depth", "30", "--out", p("F1")])
    add_simulated_csq(p("F1.dnm.vcf"), p("F1.dnm.vep_SIMULATED.vcf"))
    main(["annotate", "--vcf", p("F1.dnm.vep_SIMULATED.vcf"), "--call-summary", p("F1.call_summary.json"),
          "--gene-table", genes, "--fasta", t["fasta"], "--signatures", sigs,
          "--signature-subset", "TOY_SBS1,TOY_SBS5,TOY_SBS40", "--parent-of-origin", poo,
          "--paternal-age", "34", "--maternal-age", "31", "--parentage-confirmed", "--out", p("F1")])

    r800 = ref("chr1", 800)
    r150, r160, r170 = ref("chr1", 150), ref("chr1", 160), ref("chr1", 170)
    a150, a160, a170 = ALT_OF[r150], ALT_OF[r160], ALT_OF[r170]
    som = os.path.join(inp, "tumor.mutect2_SIMULATED.vcf")
    with open(som, "w") as fh:
        fh.write(SOM_HEADER + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTUM\tkid\n")
        fh.write(f"chr1\t150\t.\t{r150}\t{a150}\t.\tPASS\tCSQ={a150}|stop_gained|HIGH|NF1|p.Gln10Ter\tGT:AD:DP\t0/1:40,20:60\t0/0:30,0:30\n")
        fh.write(f"chr1\t160\t.\t{r160}\t{a160}\t.\tPASS\tCSQ={a160}|missense_variant|MODERATE|TP53|p.Arg175His\tGT:AD:DP\t0/1:25,35:60\t0/0:30,0:30\n")
        fh.write(f"chr1\t800\t.\t{r800}\t{ALT_OF[r800]}\t.\tPASS\tCSQ={ALT_OF[r800]}|missense_variant|MODERATE|KRAS|p.Gly12Asp\tGT:AD:DP\t0/1:40,20:60\t0/0:30,0:30\n")
        fh.write(f"chr1\t170\t.\t{r170}\t{a170}\t.\tPASS\tCSQ={a170}|synonymous_variant|LOW|NF1|p.Leu20=\tGT:AD:DP\t0/1:30,30:60\t0/1:15,14:29\n")
    germ = os.path.join(inp, "germline_for_two_hit.tsv")
    with open(germ, "w") as fh:
        fh.write("gene\tchrom\tpos\tref\talt\tpriority_tier\tacmg_class\ttrack\nNF1\tchr1\t50\tC\tT\tTier1\tPathogenic\tgermline\n")
    segs = os.path.join(inp, "tumor.segments_SIMULATED.tsv")
    with open(segs, "w") as fh:
        fh.write("chrom\tstart\tend\ttotal_cn\tminor_cn\nchr1\t1\t155\t2\t0\nchr1\t156\t2000\t2\t1\n")
    main(["somatic", "--vcf", som, "--tumor", "TUM", "--normal", "kid", "--trio-vcf", t["vcf"], "--father", "dad",
          "--mother", "mom", "--purity", "0.6", "--segments", segs, "--gene-table", genes,
          "--germline-annotated", germ, "--fasta", t["fasta"], "--out", p("TUM")])

    ch = os.path.join(inp, "blood.ch_SIMULATED.vcf")
    with open(ch, "w") as fh:
        fh.write(SOM_HEADER + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tdad\tmom\n")
        fh.write("chr2\t25234373\t.\tC\tT\t.\tPASS\tCSQ=T|missense_variant|MODERATE|DNMT3A|p.Arg882His\tGT:AD:DP\t0/1:180,20:200\t0/0:200,0:200\n")
        fh.write("chr4\t105243000\t.\tC\tT\t.\tPASS\tCSQ=T|stop_gained|HIGH|TET2|p.Gln1034Ter\tGT:AD:DP\t0/0:200,0:200\t0/1:190,10:200\n")
    main(["ch-screen", "--vcf", ch, "--samples", "dad,mom", "--out", p("trio")])

    summary = {"outdir": os.path.abspath(outdir), "results": sorted(os.listdir(res))}
    with open(os.path.join(outdir, "README.txt"), "w") as fh:
        fh.write(DEMO_README)
    return summary


def _selfsm(d: str, sample: str, freemix: float) -> str:
    path = os.path.join(d, f"{sample}.selfSM")
    with open(path, "w") as fh:
        fh.write(f"#SEQ_ID\tRG\tCHIP_ID\t#SNPS\t#READS\tAVG_DP\tFREEMIX\n{sample}\tNA\tNA\t10000\t1000000\t30\t{freemix}\n")
    return path


def _somalier_pairs(d: str) -> str:
    path = os.path.join(d, "F1.somalier.pairs.tsv")
    with open(path, "w") as fh:
        fh.write("#sample_a\tsample_b\trelatedness\tibs0\tibs2\thom_concordance\tn\n"
                 "kid\tdad\t0.497\t2\t4210\t0.62\t10000\nkid\tmom\t0.503\t1\t4305\t0.64\t10000\n"
                 "dad\tmom\t0.012\t910\t1480\t0.03\t10000\n")
    return path


def _somalier_samples(d: str) -> str:
    path = os.path.join(d, "F1.somalier.samples.tsv")
    with open(path, "w") as fh:
        fh.write("#family_id\tsample_id\tpaternal_id\tmaternal_id\tsex\tphenotype\tdepth_mean\tX_het\tX_n\tY_depth_mean\n"
                 "F1\tkid\tdad\tmom\t1\t2\t30.1\t3\t1200\t14.8\nF1\tdad\t0\t0\t1\t1\t29.7\t2\t1200\t15.1\n"
                 "F1\tmom\t0\t0\t2\t1\t31.0\t410\t1200\t0.1\n")
    return path


DEMO_README = """trio-dnm demo — ALL DATA IN THIS FOLDER IS SIMULATED
=====================================================

inputs/   synthetic reference, trio VCF/PED, mask BED, toy gene table, TOY signatures,
          simulated somalier/VerifyBamID2 QC files, simulated tumour and CH VCFs.
results/  every output the pipeline produces:

  F1.qc_gate.json          Stage 1 stop-gate verdict (PASS)
  F1.candidates.tsv        every Mendelian-violation candidate with every failure reason
  F1.call_summary.json     filter waterfall, tiers, parental leakage, raw Mendelian error rate
  F1.dnm.vcf               passing DNMs (germline, post-zygotic mosaic, parental mosaic)
  F1.dnm.vep_SIMULATED.vcf the same with simulated VEP CSQ annotations
  F1.annotated.tsv         full annotation, provisional ACMG, tiers, ranking
  F1.shortlist.tsv         Tier 1-3 variants
  F1.qc.json               sanity checks, signatures, counts
  F1.sbs96.tsv             SBS-96 spectrum
  F1.report.html           open in a browser
  TUM.somatic.tsv          somatic calls with trio-aware filtering, CCF, drivers, two-hit
  TUM.somatic_summary.json TMB, drivers, second hits, CH-gene variants, clonality
  TUM.somatic.sbs96.tsv    somatic spectrum
  trio.ch_screen.tsv       clonal haematopoiesis hits in parental blood
  trio.ch_summary.json

The sanity status is REVIEW by design: a 10-site toy trio is far below real WGS DNM counts.
Gene names, HGVS, scores and ClinVar labels are illustrative only.
"""
