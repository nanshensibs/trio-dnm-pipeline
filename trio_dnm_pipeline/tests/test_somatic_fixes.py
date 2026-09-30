"""Regression tests for the somatic / clonal-haematopoiesis review fixes."""
import csv
import json

import pytest

from trio_dnm import somatic
from trio_dnm.cli import main
from trio_dnm.genome import SBS96

FMT = """##FORMAT=<ID=GT,Number=1,Type=String,Description="">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="">
"""
CSQ_HDR = ('##INFO=<ID=CSQ,Number=.,Type=String,Description="Consequence annotations from Ensembl VEP. '
           'Format: Allele|Consequence|IMPACT|SYMBOL|HGVSp|gnomADv4_AF">\n')
HDR = "##fileformat=VCFv4.2\n" + CSQ_HDR + FMT + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTUM\tNORM\n"
NORMAL = "0/0:50,0:50"


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def line(chrom, pos, ref, alt, tum, norm=NORMAL, gene="GENEX", csq_af="", info=None, impact="MODERATE"):
    info = info or f"CSQ={alt}|missense_variant|{impact}|{gene}|p.X{pos}Y|{csq_af}"
    return f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\tGT:AD:DP\t{tum}\t{norm}"


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def run(tmp_path, lines, *extra, header=HDR):
    vcf = write(tmp_path, "som.vcf", header + "\n".join(lines) + "\n")
    out = str(tmp_path / "s")
    main(["somatic", "--vcf", vcf, "--tumor", "TUM", "--normal", "NORM", "--out", out, *extra])
    rows = {(r["chrom"], int(r["pos"])): r for r in read_tsv(out + ".somatic.tsv")}
    return rows, json.load(open(out + ".somatic_summary.json")), out


# --------------------------------------------------------------------------- #
# population AF (cleanroom F6 / protocol F11)
# --------------------------------------------------------------------------- #
def test_population_af_falls_back_to_csq_gnomad(tmp_path):
    rows, _, _ = run(tmp_path, [
        line("chr1", 100, "C", "A", "0/1:60,40:100", csq_af="0.02"),
        line("chr1", 200, "C", "A", "0/1:60,40:100"),
        # an INFO key takes precedence over CSQ
        line("chr1", 300, "C", "A", "0/1:60,40:100",
             info="gnomAD_AF=0.0001;CSQ=A|missense_variant|MODERATE|GENEX|p.X1Y|0.5"),
    ])
    assert rows[("chr1", 100)]["status"] == "FILTERED" and "population_AF" in rows[("chr1", 100)]["reasons"]
    assert rows[("chr1", 100)]["pop_af"] == "2.00e-02"
    assert rows[("chr1", 200)]["status"] == "PASS" and rows[("chr1", 200)]["pop_af"] == ""
    assert rows[("chr1", 300)]["status"] == "PASS" and rows[("chr1", 300)]["pop_af"] == "1.00e-04"


# --------------------------------------------------------------------------- #
# FFPE flag (python PY-01 / protocol F12)
# --------------------------------------------------------------------------- #
def test_ffpe_flags_only_c_to_t_and_g_to_a_and_never_filters(tmp_path):
    low = "0/1:92,8:100"
    changes = {100: ("C", "T"), 200: ("G", "A"), 300: ("T", "C"), 400: ("A", "G"), 500: ("C", "A")}
    lines = [line("chr1", p, r, a, low) for p, (r, a) in changes.items()]
    lines.append(line("chr1", 600, "C", "T", "0/1:70,30:100"))  # C>T above ffpe_max_vaf
    rows, summary, _ = run(tmp_path, lines, "--ffpe")
    for p in changes:
        r = rows[("chr1", p)]
        assert r["status"] == "PASS" and r["reasons"] == "", (p, r["reasons"])
        flagged = p in (100, 200)
        assert r["ffpe_suspect"] == str(flagged), p
        assert r["notes"] == ("FFPE_suspect" if flagged else ""), p
    assert rows[("chr1", 600)]["ffpe_suspect"] == "False"
    assert summary["notes"] == {"FFPE_suspect": 2} and summary["n_pass"] == 6


# --------------------------------------------------------------------------- #
# two-hit analysis (cleanroom F7 / python PY-04 / protocol F03)
# --------------------------------------------------------------------------- #
GERM_HDR = "gene\tchrom\tpos\tref\talt\tpriority_tier\tacmg_class\ttrack\n"


def test_two_hit_germline_plus_loh_without_somatic_point_mutation(tmp_path):
    germ = write(tmp_path, "germ.tsv", GERM_HDR +
                 "RB1\tchr13\t48303900\tG\tA\tTier1\tPathogenic\tgermline\n"
                 "RB1\tchr13\t48303900\tG\tA\tTier1\tPathogenic\tgermline\n"   # duplicate row is listed once
                 "BRCA2\tchr13\t32300000\tC\tT\tTier2\tLikely_pathogenic\tinherited\n"
                 "TP53\tchr1\t5000\tC\tT\tTier1\tPathogenic\tgermline\n"
                 "APC\tchr5\t100\tC\tT\tTier4\tVUS\tgermline\n")
    segs = write(tmp_path, "segs.tsv", "chrom\tstart\tend\ttotal_cn\tminor_cn\n"
                 "chr1\t1\t1000000\t2\t0\nchr13\t40000000\t50000000\t2\t0\nchr13\t30000000\t39999999\t3\t1\n"
                 "chr5\t1\t1000\t2\t0\n")
    rows, summary, out = run(tmp_path, [
        line("chr1", 100, "C", "A", "0/1:60,40:100", gene="GENEB"),   # passenger in LOH, no germline hit
        line("chr1", 6000, "C", "A", "0/1:60,40:100", gene="TP53"),   # germline + somatic + LOH
        line("chr5", 500, "C", "A", "0/1:60,40:100", gene="APC"),     # germline VUS only
    ], "--segments", segs, "--germline-annotated", germ)
    geneb = rows[("chr1", 100)]
    assert geneb["status"] == "PASS" and geneb["second_hit"] == "" and geneb["loh"] == "True"
    assert rows[("chr5", 500)]["second_hit"] == ""
    assert rows[("chr1", 6000)]["second_hit"] == "germline:chr1:5000C>T(germline) | LOH"
    assert summary["second_hits"] == [
        "TP53: germline:chr1:5000C>T(germline) | LOH",
        "RB1: germline:chr13:48303900G>A(germline) | LOH",
    ]
    assert summary["two_hit_genes"] == ["RB1", "TP53"]
    th = {r["gene"]: r for r in read_tsv(out + ".two_hit.tsv")}
    assert set(th) == {"RB1", "TP53"}
    assert th["RB1"]["germline_variant"] == "chr13:48303900G>A(germline)" and th["RB1"]["hit_type"] == "LOH"
    assert th["RB1"]["somatic_evidence"] == "LOH chr13:40000000-50000000 minor_cn=0 total_cn=2"
    assert th["TP53"]["hit_type"] == "somatic_variant+LOH"
    assert th["TP53"]["somatic_evidence"].startswith("chr1:6000C>A p.X6000Y VAF=0.400; LOH chr1:1-1000000")


def test_two_hit_germline_plus_somatic_without_segments(tmp_path):
    germ = write(tmp_path, "germ.tsv", GERM_HDR +
                 "NF1\tchr17\t31000000\tC\tT\tTier1\tPathogenic\tgermline\n"
                 "TP53\tchr17\t7675088\tC\tT\tTier2\tPathogenic\tgermline\n")
    rows, summary, out = run(tmp_path, [
        line("chr17", 31100000, "C", "A", "0/1:60,40:100", gene="NF1"),
        line("chr17", 7675100, "C", "A", "0/1:60,40:100", "0/1:40,10:50", gene="TP53"),  # ALT in normal
    ], "--germline-annotated", germ)
    assert rows[("chr17", 31100000)]["second_hit"] == "germline:chr17:31000000C>T(germline)"
    assert rows[("chr17", 31100000)]["loh"] == ""
    assert rows[("chr17", 7675100)]["status"] == "FILTERED" and rows[("chr17", 7675100)]["second_hit"] == ""
    assert summary["second_hits"] == ["NF1: germline:chr17:31000000C>T(germline)"]
    th = read_tsv(out + ".two_hit.tsv")
    assert len(th) == 1 and th[0]["gene"] == "NF1" and th[0]["hit_type"] == "somatic_variant"


def test_two_hit_table_written_even_when_empty(tmp_path):
    _, summary, out = run(tmp_path, [line("chr1", 100, "C", "A", "0/1:60,40:100")])
    assert read_tsv(out + ".two_hit.tsv") == [] and summary["second_hits"] == []


# --------------------------------------------------------------------------- #
# CCF, multiplicity and depth denominators (manual defect, PY-05, PY-15)
# --------------------------------------------------------------------------- #
def test_multiplicity_estimate():
    assert somatic.estimate_multiplicity(0.5, 0.5, 2, 0) == 2
    assert somatic.estimate_multiplicity(0.5, 0.5, 2, None) == 2
    assert somatic.estimate_multiplicity(0.6, 0.5, 2, 1) == 1          # capped at major_cn = 1
    assert somatic.estimate_multiplicity(0.1, 0.5, 2, 0) == 1          # never below 1
    assert somatic.estimate_multiplicity(0.3, 1.0, 0, 0) == 1          # CN 0 -> major_cn floor of 1
    assert somatic.estimate_multiplicity(0.9, 1.0, 4, 1) == 3          # 0.9*4 = 3.6 -> 4, capped at 3
    assert somatic.cancer_cell_fraction(0.5, 0.5, 2, 2) == pytest.approx(1.0)
    assert somatic.cancer_cell_fraction(0.5, 0.0, 2, 1) == 0.0
    assert somatic.cancer_cell_fraction(0.5, 0.5, 2, 0) == 0.0


def test_load_segments_keeps_homozygous_deletion(tmp_path):
    segs = write(tmp_path, "segs.tsv", "chrom\tstart\tend\ttotal_cn\tminor_cn\n"
                 "chr9\t21900000\t22000000\t0\t0\nchr9\t22000001\t23000000\t\t1\n")
    s = somatic.load_segments(segs)["9"]
    assert s[0]["total_cn"] == 0.0 and s[0]["minor_cn"] == 0.0
    assert s[1]["total_cn"] == 2.0


def test_ccf_uses_multiplicity_cap_and_ad_denominator(tmp_path):
    segs = write(tmp_path, "segs.tsv", "chrom\tstart\tend\ttotal_cn\tminor_cn\n"
                 "chr1\t1\t1000\t2\t0\nchr1\t1001\t2000\t2\t1\nchr2\t1\t1000\t0\t0\n")
    rows, summary, _ = run(tmp_path, [
        line("chr1", 100, "C", "A", "0/1:50,50:100"),      # CN-LOH, both copies mutated
        line("chr1", 1500, "C", "A", "0/1:40,60:100"),     # impossible VAF for m=1 -> capped
        line("chr1", 1600, "C", "A", "0/1:160,40:300"),    # DP > sum(AD)
        line("chr2", 100, "C", "A", "0/1:80,20:100"),      # homozygous deletion segment
    ], "--purity", "0.5", "--segments", segs)
    r = rows[("chr1", 100)]
    assert (r["multiplicity"], r["ccf"], r["ccf_raw"], r["clonality"]) == ("2", "1.00", "1.00", "clonal")
    r = rows[("chr1", 1500)]
    assert (r["multiplicity"], r["ccf"], r["ccf_raw"]) == ("1", "1.00", "2.40")
    r = rows[("chr1", 1600)]
    assert r["tumor_vaf"] == "0.200" and r["tumor_dp"] == "300"
    # The Wilson upper bound uses n = ref+alt = 200, so it can no longer sit below the point estimate.
    assert (r["ccf"], r["clonality"]) == ("0.80", "clonal")
    r = rows[("chr2", 100)]
    assert r["total_cn"] == "0.0" and r["multiplicity"] == "1" and r["ccf"] == "0.40"
    assert all(float(x["ccf"]) <= 1.0 for x in rows.values())


def test_purity_out_of_range_rejected(tmp_path):
    with pytest.raises(SystemExit, match="purity"):
        run(tmp_path, [line("chr1", 100, "C", "A", "0/1:60,40:100")], "--purity", "1.5")


# --------------------------------------------------------------------------- #
# signature refit guard (protocol F13)
# --------------------------------------------------------------------------- #
def _fasta(tmp_path):
    seq = ("ACGTTGCA" * 30)[:240]
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")
    (tmp_path / "ref.fa.fai").write_text(f"chr1\t{len(seq)}\t6\t60\t61\n")
    return str(fa), seq


def test_signature_refit_skipped_below_min_snvs_without_subset(tmp_path):
    fa, seq = _fasta(tmp_path)
    sigs = write(tmp_path, "sigs.tsv", "Type\tSIG_A\tSIG_B\n" + "".join(f"{ch}\t{1 / 96}\t{1 / 96}\n" for ch in SBS96))
    ref = seq[9]
    alt = "A" if ref != "A" else "C"
    lines = [line("chr1", 10, ref, alt, "0/1:60,40:100")]
    _, summary, _ = run(tmp_path, lines, "--fasta", fa, "--signatures", sigs)
    assert summary["signatures"] == {"skipped": "1 SNVs < 50: pass --signature-subset"}
    _, summary, _ = run(tmp_path, lines, "--fasta", fa, "--signatures", sigs, "--signature-subset", "SIG_A")
    assert summary["signatures"]["n_snv"] == 1 and list(summary["signatures"]["exposures"]) == ["SIG_A"]
    conf = write(tmp_path, "c.json", json.dumps({"somatic": {"min_snv_full_refit": 1}}))
    _, summary, _ = run(tmp_path, lines, "--fasta", fa, "--signatures", sigs, "--config", conf)
    assert set(summary["signatures"]["exposures"]) == {"SIG_A", "SIG_B"}


# --------------------------------------------------------------------------- #
# sample-ID validation (cleanroom F9)
# --------------------------------------------------------------------------- #
def test_somatic_rejects_unknown_samples(tmp_path):
    lines = [line("chr1", 100, "C", "A", "0/1:60,40:100")]
    vcf = write(tmp_path, "som.vcf", HDR + "\n".join(lines) + "\n")
    trio = write(tmp_path, "trio.vcf", "##fileformat=VCFv4.2\n" + FMT +
                 "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tkid\tdad\tmom\n")
    out = str(tmp_path / "s")
    base = ["somatic", "--vcf", vcf, "--out", out]
    with pytest.raises(SystemExit, match="TUMOR"):
        main(base + ["--tumor", "TUMOR", "--normal", "NORM"])
    with pytest.raises(SystemExit, match="DAD01"):
        main(base + ["--tumor", "TUM", "--normal", "NORM", "--trio-vcf", trio, "--father", "DAD01", "--mother", "mom"])
    with pytest.raises(SystemExit, match="--father"):
        main(base + ["--tumor", "TUM", "--normal", "NORM", "--trio-vcf", trio])
    assert main(base + ["--tumor", "TUM", "--normal", "NORM", "--trio-vcf", trio, "--father", "dad", "--mother", "mom"]) == 0


# --------------------------------------------------------------------------- #
# CH screen (cleanroom F9 / F20, python PY-05 / PY-14)
# --------------------------------------------------------------------------- #
CH_PLAIN = ("##fileformat=VCFv4.2\n##INFO=<ID=GENE,Number=.,Type=String,Description=\"\">\n" + FMT +
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tdad\tmom\n")


def test_ch_screen_rejects_unknown_samples(tmp_path):
    v = write(tmp_path, "ch.vcf", CH_PLAIN)
    with pytest.raises(SystemExit, match="kid"):
        main(["ch-screen", "--vcf", v, "--samples", "dad,kid", "--out", str(tmp_path / "ch")])


def test_ch_screen_without_csq_uses_info_gene_and_warns(tmp_path, capsys):
    v = write(tmp_path, "ch.vcf", CH_PLAIN + "\n".join([
        "chr2\t25234373\t.\tC\tT\t.\tPASS\tGENE=DNMT3A\tGT:AD:DP\t0/1:90,10:300\t0/0:100,0:100",
        "chr2\t100\t.\tC\tT\t.\tPASS\tGENE=OTHER\tGT:AD:DP\t0/1:90,10:100\t0/0:100,0:100",
        "chr4\t105243000\t.\tC\tT\t.\tPASS\tSYMBOL=TET2;GENE=TET2-AS1,TET2\tGT:AD:DP\t0/0:100,0:100\t0/1:90,10:100",
    ]) + "\n")
    out = str(tmp_path / "ch")
    main(["ch-screen", "--vcf", v, "--samples", "dad,mom", "--out", out])
    err = capsys.readouterr().err
    assert "warning" in err and "impact could not be assessed" in err
    rows = read_tsv(out + ".ch_screen.tsv")
    assert [(r["sample"], r["gene"]) for r in rows] == [("dad", "DNMT3A"), ("mom", "TET2")]
    assert rows[0]["consequence"] == "unannotated" and rows[0]["dp"] == "300"
    # Wilson CI on n = ref+alt (100), not DP (300): it contains the reported VAF.
    lo, hi = map(float, rows[0]["vaf_ci"].split("-"))
    assert lo < float(rows[0]["vaf"]) < hi
    s = json.load(open(out + ".ch_summary.json"))
    assert s["impact_filter"] is False and s["warnings"]
    main(["ch-screen", "--vcf", v, "--samples", "mom", "--out", out, "--gene-info-key", "SYMBOL"])
    assert [r["gene"] for r in read_tsv(out + ".ch_screen.tsv")] == ["TET2"]


def test_ch_screen_with_csq_keeps_impact_filter(tmp_path, capsys):
    hdr = "##fileformat=VCFv4.2\n" + CSQ_HDR + FMT + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tdad\tmom\n"
    v = write(tmp_path, "ch.vcf", hdr + "\n".join([
        "chr2\t25234373\t.\tC\tT\t.\tPASS\tCSQ=T|missense_variant|MODERATE|DNMT3A|p.R882H|\tGT:AD:DP\t0/1:90,10:100\t0/0:100,0:100",
        "chr2\t25234400\t.\tC\tT\t.\tPASS\tCSQ=T|synonymous_variant|LOW|DNMT3A|p.L1=|\tGT:AD:DP\t0/1:90,10:100\t0/0:100,0:100",
    ]) + "\n")
    out = str(tmp_path / "ch")
    main(["ch-screen", "--vcf", v, "--samples", "dad,mom", "--out", out])
    assert "warning" not in capsys.readouterr().err
    rows = read_tsv(out + ".ch_screen.tsv")
    assert len(rows) == 1 and rows[0]["pos"] == "25234373"
    assert json.load(open(out + ".ch_summary.json"))["impact_filter"] is True
