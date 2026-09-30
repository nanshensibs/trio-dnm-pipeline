"""Regression tests for the engine review fixes (FILTER, tiers, mosaic routing,
hemizygous parents, parental-mosaic reachability, AD denominators, external
callers, sample IDs, Mendelian stop-gate, QC gate, config merging, PoN INFO,
supplementary engine, pileup strand evidence, SnpEff ANN header)."""
import copy
import csv
import gzip
import json

import pytest

from trio_dnm import dnm, stats
from trio_dnm.cli import main
from trio_dnm.config import DEFAULTS, load_config
from trio_dnm.vcf import VCFReader

HEAD = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=100000>
##contig=<ID=chrX,length=156040895>
##FILTER=<ID=QD2,Description="QD < 2">
##INFO=<ID=MQ,Number=1,Type=Float,Description="">
##INFO=<ID=FS,Number=1,Type=Float,Description="">
##INFO=<ID=SOR,Number=1,Type=Float,Description="">
##INFO=<ID=gnomAD_AF,Number=A,Type=Float,Description="">
##INFO=<ID=PON_AF,Number=A,Type=Float,Description="">
##INFO=<ID=hiConfDeNovo,Number=1,Type=String,Description="">
##FORMAT=<ID=GT,Number=1,Type=String,Description="">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="">
##FORMAT=<ID=SB,Number=4,Type=Integer,Description="">
"""
GOOD = "MQ=60;FS=1.2;SOR=0.7;hiConfDeNovo=kid;gnomAD_AF=0"
PED = "F1\tkid\tdad\tmom\t{sex}\t2\nF1\tdad\t0\t0\t1\t1\nF1\tmom\t0\t0\t2\t1\n"


def g(gt, ref, alt, gq=99, dp=None, sb=True):
    dp = ref + alt if dp is None else dp
    pl = {"0/0": f"0,{3 * dp},{30 * dp}", "0/1": f"{30 * alt},0,{30 * ref}", "1/1": f"{30 * dp},{3 * dp},0",
          "0": f"0,{30 * dp}", "1": f"{30 * dp},0"}[gt]
    out = f"{gt}:{ref},{alt}:{dp}:{gq}:{pl}"
    if sb:
        out += f":{ref // 2},{ref - ref // 2},{alt // 2},{alt - alt // 2}"
    return out


def row(pos, kid, dad, mom, chrom="chr1", ref="C", alt="T", info=GOOD, filt="PASS", sb=True):
    fmt = "GT:AD:DP:GQ:PL" + (":SB" if sb else "")
    return f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t500\t{filt}\t{info}\t{fmt}\t{kid}\t{dad}\t{mom}"


def write_trio(tmp, rows, sex=1, name="trio", samples=("kid", "dad", "mom"), head=HEAD):
    vcf = tmp / f"{name}.vcf"
    vcf.write_text(head + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n"
                   + "\n".join(rows) + "\n")
    ped = tmp / "trio.ped"
    ped.write_text(PED.format(sex=sex))
    return str(vcf), str(ped)


def call(tmp, vcf, ped, *extra, out="o", rc=0):
    prefix = str(tmp / out)
    assert main(["call", "--vcf", vcf, "--ped", ped, "--out", prefix, "--mean-depth", "30", *extra]) == rc
    with open(prefix + ".candidates.tsv") as fh:
        rows = {int(r["pos"]): r for r in csv.DictReader(fh, delimiter="\t")}
    return rows, json.load(open(prefix + ".call_summary.json"))


# --------------------------------------------------------------------------- #
# FILTER column (cleanroom F1)
# --------------------------------------------------------------------------- #
def test_filter_column_fails_layer2_unless_disabled(tmp_path):
    vcf, ped = write_trio(tmp_path, [row(100, g("0/1", 15, 14), g("0/0", 30, 0), g("0/0", 32, 0), filt="QD2;SOR3"),
                                     row(200, g("0/1", 15, 14), g("0/0", 30, 0), g("0/0", 32, 0), filt=".")])
    rows, _ = call(tmp_path, vcf, ped)
    assert rows[100]["pass"] == "False" and rows[100]["first_fail"] == "L2_read_quality"
    assert "site_filter:QD2;SOR3" in rows[100]["fail_reasons"]
    assert rows[200]["pass"] == "True"
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"site": {"honour_filter": False}}))
    rows, _ = call(tmp_path, vcf, ped, "--config", str(cfg), out="o2")
    assert rows[100]["pass"] == "True"


# --------------------------------------------------------------------------- #
# Tiers (cleanroom F5, python PY-06, protocol F06)
# --------------------------------------------------------------------------- #
def _cand(track="germline", post=0.99, hi=False, callers=()):
    c = dnm.Candidate(rec=None, track=track, ploidy=2)
    c.posterior, c.hi_conf, c.callers = post, hi, list(callers)
    return c


def test_tiers_without_external_callers():
    cfg = load_config()
    assert dnm.assign_tier(_cand(post=0.99, hi=True), 0, cfg) == "HIGH"
    assert dnm.assign_tier(_cand(post=0.999, hi=False), 0, cfg) == "MEDIUM"
    assert dnm.assign_tier(_cand(post=0.90, hi=True), 0, cfg) == "MEDIUM"
    assert dnm.assign_tier(_cand(post=0.90, hi=False), 0, cfg) == "LOW"
    # mosaic tracks are capped at MEDIUM with no independent caller
    for track in ("mosaic", "parental_mosaic"):
        assert dnm.assign_tier(_cand(track, post=0.999, hi=True), 0, cfg) == "MEDIUM"
        assert dnm.assign_tier(_cand(track, post=0.90), 0, cfg) == "LOW"
    failed = _cand(post=0.999, hi=True)
    failed.fails["L1_genotype"].append("x")
    assert dnm.assign_tier(failed, 0, cfg) == "FAIL"


def test_tiers_with_external_callers():
    cfg = load_config()
    three = ("a", "b", "c")
    assert dnm.assign_tier(_cand(post=0.90, hi=True, callers=three), 3, cfg) == "HIGH"
    assert dnm.assign_tier(_cand(post=0.99, callers=three), 3, cfg) == "HIGH"
    assert dnm.assign_tier(_cand(post=0.90, callers=three), 3, cfg) == "MEDIUM"
    assert dnm.assign_tier(_cand(post=0.99, hi=True, callers=("a", "b")), 3, cfg) == "MEDIUM"
    assert dnm.assign_tier(_cand(post=0.99, hi=True, callers=("a",)), 3, cfg) == "LOW"
    assert dnm.assign_tier(_cand(post=0.99, hi=True, callers=("a",)), 1, cfg) == "HIGH"
    # hiConfDeNovo does not apply to the mosaic tracks
    assert dnm.assign_tier(_cand("mosaic", post=0.90, hi=True, callers=three), 3, cfg) == "MEDIUM"
    assert dnm.assign_tier(_cand("mosaic", post=0.99, callers=three), 3, cfg) == "HIGH"


# --------------------------------------------------------------------------- #
# Mosaic routing (protocol F07) and AD denominators (PY-05)
# --------------------------------------------------------------------------- #
def test_mosaic_routing_needs_significant_binomial(tmp_path):
    vcf, ped = write_trio(tmp_path, [
        row(100, g("0/1", 9, 3), g("0/0", 35, 0), g("0/0", 35, 0)),      # VAF 0.25, p = 0.07: germline, fails L1
        row(200, g("0/1", 45, 8), g("0/0", 40, 0), g("0/0", 42, 0)),     # VAF 0.15, significant: mosaic
        row(300, g("0/1", 13, 8, dp=40), g("0/0", 30, 0), g("0/0", 30, 0)),  # AD 13,8 with DP 40: germline
        row(400, g("0/1", 15, 15, dp=60), g("0/0", 30, 0), g("0/0", 30, 0)),
    ])
    rows, _ = call(tmp_path, vcf, ped)
    assert rows[100]["track"] == "germline" and rows[100]["pass"] == "False"
    assert "proband_VAF_out_of_het_range" in rows[100]["fail_reasons"]
    assert "POSSIBLE_MOSAIC_NEEDS_DEPTH" in rows[100]["flags"]
    assert rows[200]["track"] == "mosaic" and rows[200]["pass"] == "True"
    assert rows[300]["track"] == "germline"
    lo, hi = (float(x) for x in rows[400]["proband_vaf_ci"].split("-"))
    assert lo < 0.5 < hi  # CI on n = sum(AD) = 30, not DP = 60


def test_mosaic_posterior_uses_ad_total():
    cfg = load_config()
    vcf_line = row(100, g("0/1", 45, 8, dp=200), g("0/0", 40, 0, dp=200), g("0/0", 42, 0))
    from trio_dnm.vcf import parse_record
    from trio_dnm.genome import Trio
    rec = parse_record(vcf_line, ["kid", "dad", "mom"])
    t = Trio("F1", "kid", "dad", "mom", "male")
    c = dnm.classify_candidate(rec, t, cfg)
    assert c.track == "mosaic"
    assert rec.samples["kid"].ad_total == 53 and rec.samples["kid"].dp == 200
    lo, hi = c.proband_vaf_ci
    assert lo < 8 / 53 < hi


def test_hemizygous_mosaic_routing_and_dp_cap(tmp_path):
    x = dict(chrom="chrX", info="MQ=60;FS=1;SOR=0.7;gnomAD_AF=0")
    vcf, ped = write_trio(tmp_path, [
        row(3_000_100, g("1", 1, 15), g("0/0", 18, 0), g("0/0", 30, 0), **x),   # germline hemizygous
        row(3_000_200, g("0/1", 10, 10), g("0/0", 18, 0), g("0/0", 30, 0), **x),  # VAF 0.5, significant: mosaic
        row(3_000_300, g("1", 1, 5), g("0/0", 18, 0), g("0/0", 30, 0), **x),    # VAF 0.83, n=6: not significant
        row(3_000_400, g("1", 0, 40), g("0/0", 18, 0), g("0/0", 30, 0), **x),   # DP 40 > hemizygous cap 30
    ])
    rows, _ = call(tmp_path, vcf, ped)
    assert rows[3_000_100]["track"] == "germline" and rows[3_000_100]["pass"] == "True"
    assert rows[3_000_200]["track"] == "mosaic"
    r = rows[3_000_300]
    assert r["track"] == "germline" and "proband_VAF<0.85" in r["fail_reasons"]
    assert "POSSIBLE_MOSAIC_NEEDS_DEPTH" in r["flags"]
    assert "proband_DP>cap" in rows[3_000_400]["fail_reasons"]


# --------------------------------------------------------------------------- #
# Hemizygous parental depth (protocol F08)
# --------------------------------------------------------------------------- #
def test_hemizygous_father_depth_floor(tmp_path):
    x = dict(chrom="chrX", info="MQ=60;FS=1;SOR=0.7;gnomAD_AF=0")
    vcf, ped = write_trio(tmp_path, [
        row(3_000_100, g("0/1", 15, 15), g("0/0", 13, 0), g("0/0", 30, 0), **x),
        row(3_000_200, g("0/1", 15, 15), g("0/0", 7, 0), g("0/0", 30, 0), **x),
        row(3_000_300, g("0/1", 40, 8), g("0/0", 16, 0), g("0/0", 40, 0), **x),   # mosaic: floor ceil(30/2)
        row(3_000_400, g("0/1", 40, 8), g("0/0", 14, 0), g("0/0", 40, 0), **x),
        row(100, g("0/1", 15, 15), g("0/0", 13, 0), g("0/0", 30, 0)),             # autosome: diploid floor
    ], sex=2)
    rows, _ = call(tmp_path, vcf, ped)
    assert rows[3_000_100]["pass"] == "True", rows[3_000_100]
    assert "father_DP<8" in rows[3_000_200]["fail_reasons"]
    assert rows[3_000_300]["track"] == "mosaic" and "father_DP" not in rows[3_000_300]["fail_reasons"]
    assert "father_DP<15" in rows[3_000_400]["fail_reasons"]
    assert "father_DP<15" in rows[100]["fail_reasons"]


# --------------------------------------------------------------------------- #
# Parental mosaic called 0/1 (protocol F09)
# --------------------------------------------------------------------------- #
def test_parent_called_het_with_low_vaf_is_parental_mosaic(tmp_path):
    vcf, ped = write_trio(tmp_path, [
        row(100, g("0/1", 15, 15), g("0/1", 25, 5, gq=45), g("0/0", 30, 0)),   # mosaic father called het
        row(200, g("0/1", 15, 15), g("0/1", 15, 15), g("0/0", 30, 0)),         # inherited
    ])
    rows, _ = call(tmp_path, vcf, ped)
    assert 200 not in rows
    r = rows[100]
    assert r["track"] == "parental_mosaic" and "PARENT_CALLED_HET" in r["flags"]
    assert r["pass"] == "True", r


def test_parental_mosaic_still_checks_the_other_parent(tmp_path):
    vcf, ped = write_trio(tmp_path, [row(100, g("0/1", 15, 15), g("0/0", 27, 3, gq=40), g("0/0", 40, 6, gq=30))])
    rows, _ = call(tmp_path, vcf, ped)
    assert rows[100]["track"] == "parental_mosaic"
    assert "father_alt>1" in rows[100]["fail_reasons"] and "mother_alt" not in rows[100]["fail_reasons"]


# --------------------------------------------------------------------------- #
# External callers (cleanroom F13, PY-08)
# --------------------------------------------------------------------------- #
def test_caller_files_bgz_bcf_proband_and_counts(tmp_path, capsys):
    vcf, ped = write_trio(tmp_path, [row(100, g("0/1", 15, 14), g("0/0", 30, 0), g("0/0", 32, 0)),
                                     row(200, g("0/1", 15, 14), g("0/0", 30, 0), g("0/0", 32, 0))])
    body = (HEAD + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tkid\tdad\tmom\n"
            + row(100, "0/1:5,5", "0/0:9,0", "0/0:9,0", info=".").replace("GT:AD:DP:GQ:PL:SB", "GT:AD") + "\n"
            + row(200, "0/0:9,0", "0/0:9,0", "0/0:9,0", info=".").replace("GT:AD:DP:GQ:PL:SB", "GT:AD") + "\n")
    with gzip.open(tmp_path / "dt.vcf.bgz", "wt") as fh:
        fh.write(body)
    other = tmp_path / "other.vcf"
    other.write_text(body.replace("\tkid\tdad\tmom", "\tK\tD\tM"))
    empty = tmp_path / "empty.tsv"
    empty.write_text("chrom\tpos\tref\talt\n")
    rows, summary = call(tmp_path, vcf, ped, "--caller", f"deeptrio={tmp_path / 'dt.vcf.bgz'}",
                         "--caller", f"other={other}", "--caller", f"dng={empty}")
    assert rows[100]["callers"] == "deeptrio,other"
    assert rows[200]["callers"] == "other"  # proband 0/0 in deeptrio: not a deeptrio call
    assert summary["caller_sites"] == {"deeptrio": 1, "other": 2, "dng": 0}
    err = capsys.readouterr().err
    assert "dng" in err and "0 sites" in err
    assert "proband kid is not among its samples" in err
    (tmp_path / "x.bcf").write_bytes(b"\x1f\x8b\x08\x04")
    with pytest.raises(SystemExit, match="convert with bcftools view -Oz"):
        call(tmp_path, vcf, ped, "--caller", f"x={tmp_path / 'x.bcf'}", out="b")
    with pytest.raises(SystemExit, match="convert with bcftools view -Oz"):
        call(tmp_path, str(tmp_path / "x.bcf"), ped, out="b2")


# --------------------------------------------------------------------------- #
# Sample IDs (cleanroom F9)
# --------------------------------------------------------------------------- #
def test_ped_ids_missing_from_vcf_exit(tmp_path):
    vcf, _ = write_trio(tmp_path, [row(100, g("0/1", 15, 14), g("0/0", 30, 0), g("0/0", 32, 0))])
    ped = tmp_path / "bad.ped"
    ped.write_text("F1\tKID01\tDAD01\tmom\t1\t2\nF1\tDAD01\t0\t0\t1\t1\nF1\tmom\t0\t0\t2\t1\n")
    with pytest.raises(SystemExit) as e:
        main(["call", "--vcf", vcf, "--ped", str(ped), "--out", str(tmp_path / "x")])
    msg = str(e.value)
    assert "KID01" in msg and "DAD01" in msg and "VCF samples: kid, dad, mom" in msg
    assert "mom" not in msg.split("not in the VCF header")[0]


# --------------------------------------------------------------------------- #
# Mendelian stop-gate (cleanroom F10, PY-11, protocol F05)
# --------------------------------------------------------------------------- #
def test_mendelian_error_gate_halts_call(trio_data, capsys):
    t = trio_data["tmp"]
    base = ["call", "--vcf", trio_data["vcf"], "--ped", trio_data["ped"], "--mean-depth", "30"]
    assert main(base + ["--out", str(t / "a")]) == 0  # 9 informative sites < qc.mie_min_sites
    s = json.load(open(t / "a.call_summary.json"))
    assert s["mendelian_error_gate"] == "NOT_EVALUATED" and s["raw_mendelian_informative_sites"] == 9
    cfg = t / "qc.json"
    cfg.write_text(json.dumps({"qc": {"mie_min_sites": 5}}))
    assert main(base + ["--config", str(cfg), "--out", str(t / "b")]) == 3
    assert (t / "b.candidates.tsv").exists() and (t / "b.dnm.vcf").exists()
    s = json.load(open(t / "b.call_summary.json"))
    assert s["mendelian_error_gate"] == "FAIL" and s["config"]["qc"]["mie_min_sites"] == 5
    assert "Mendelian error rate" in capsys.readouterr().err
    assert main(base + ["--config", str(cfg), "--out", str(t / "c"), "--no-halt"]) == 0


# --------------------------------------------------------------------------- #
# Population: PON_AF INFO and missing gnomAD annotation (NF-10, protocol F26)
# --------------------------------------------------------------------------- #
def test_pon_info_key_and_missing_population_af(tmp_path, capsys):
    vcf, ped = write_trio(tmp_path, [row(100, g("0/1", 15, 14), g("0/0", 30, 0), g("0/0", 32, 0), info=GOOD + ";PON_AF=0.05"),
                                     row(200, g("0/1", 15, 14), g("0/0", 30, 0), g("0/0", 32, 0))])
    rows, summary = call(tmp_path, vcf, ped)
    assert "panel_of_normals_AF=0.050" in rows[100]["fail_reasons"] and rows[200]["pass"] == "True"
    assert summary["population_af_missing"] is False
    capsys.readouterr()
    vcf, ped = write_trio(tmp_path, [row(100, g("0/1", 15, 14), g("0/0", 30, 0), g("0/0", 32, 0), info="MQ=60")],
                          name="noaf")
    _, summary = call(tmp_path, vcf, ped, out="n")
    assert summary["population_af_missing"] is True
    assert "gnomAD" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Supplementary engine (cleanroom F17)
# --------------------------------------------------------------------------- #
def test_extra_vcf_adds_second_engine_only_candidates(tmp_path):
    vcf, ped = write_trio(tmp_path, [row(100, g("0/1", 15, 14), g("0/0", 30, 0), g("0/0", 32, 0)),
                                     row(300, g("0/1", 15, 14), g("0/0", 30, 0), g("0/0", 32, 0))])
    extra, _ = write_trio(tmp_path, [row(100, g("0/1", 15, 14), g("0/0", 30, 0), g("0/0", 32, 0)),
                                     row(200, g("0/1", 15, 14), g("0/0", 30, 0), g("0/0", 32, 0), info="gnomAD_AF=0"),
                                     row(400, g("0/0", 30, 0), g("0/0", 30, 0), g("0/0", 32, 0))], name="deeptrio")
    rows, summary = call(tmp_path, vcf, ped, "--extra-vcf", extra)
    assert sorted(rows) == [100, 200, 300]
    assert "SECOND_ENGINE_ONLY" in rows[200]["flags"] and rows[200]["pass"] == "True"
    assert "SECOND_ENGINE_ONLY" not in rows[100]["flags"]
    assert summary["second_engine_only_candidates"] == 1 and summary["sites_scanned"] == 2
    assert [r.pos for r in VCFReader(str(tmp_path / "o.dnm.vcf"))] == [100, 200, 300]  # sorted output
    bad, _ = write_trio(tmp_path, [row(200, g("0/1", 15, 14), g("0/0", 30, 0), g("0/0", 32, 0))],
                        name="bad", samples=("K", "dad", "mom"))
    ped2 = tmp_path / "trio.ped"
    ped2.write_text(PED.format(sex=1))
    with pytest.raises(SystemExit, match="kid"):
        call(tmp_path, vcf, str(ped2), "--extra-vcf", bad, out="e")


# --------------------------------------------------------------------------- #
# Pileup strand evidence (protocol F10)
# --------------------------------------------------------------------------- #
MPILEUP_HEAD = """##fileformat=VCFv4.2
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="">
##FORMAT=<ID=ADF,Number=R,Type=Integer,Description="">
##FORMAT=<ID=ADR,Number=R,Type=Integer,Description="">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tkid\tdad\tmom
"""


def test_strand_vcf_supplies_strand_counts(tmp_path):
    rows_ = [row(100, g("0/1", 15, 14, sb=False), g("0/0", 30, 0, sb=False), g("0/0", 32, 0, sb=False), sb=False),
             row(200, g("0/1", 15, 14, sb=False), g("0/0", 30, 0, sb=False), g("0/0", 32, 0, sb=False), sb=False),
             row(300, g("0/1", 15, 14, sb=False), g("0/0", 30, 0, sb=False), g("0/0", 32, 0, sb=False), sb=False,
                 ref="CA", alt="C"),
             row(400, g("0/1", 15, 14, sb=False), g("0/0", 30, 0, sb=False), g("0/0", 32, 0, sb=False), sb=False)]
    vcf, ped = write_trio(tmp_path, rows_)
    rows, _ = call(tmp_path, vcf, ped)
    assert all("NO_STRAND_INFO" in r["flags"] for r in rows.values())
    pile = tmp_path / "pileup.vcf"
    par = "0,0:15,0:15,0"
    pile.write_text(MPILEUP_HEAD + "\n".join([
        f"chr1\t100\t.\tC\tT,<*>\t.\t.\t.\tAD:ADF:ADR\t15,14,0:8,7,0:7,7,0\t{par}\t{par}",
        f"chr1\t200\t.\tC\tG,T,<*>\t.\t.\t.\tAD:ADF:ADR\t30,0,20,0:15,0,0,0:15,0,20,0\t{par}\t{par}",
        f"chr1\t300\t.\tCAAA\tCAA,<*>\t.\t.\t.\tAD:ADF:ADR\t15,14,0:8,7,0:7,7,0\t{par}\t{par}",
    ]) + "\n")
    rows, summary = call(tmp_path, vcf, ped, "--strand-vcf", str(pile), out="s")
    assert "NO_STRAND_INFO" not in rows[100]["flags"] and rows[100]["pass"] == "True"
    assert "alt_single_strand(0F/20R)" in rows[200]["fail_reasons"]
    assert "strand_bias_p" in rows[200]["fail_reasons"]
    assert "NO_STRAND_INFO" not in rows[300]["flags"]  # indel matched after trimming the shared suffix
    assert "NO_STRAND_INFO" in rows[400]["flags"]  # no pileup record
    assert summary["strand_vcf"] == str(pile)


# --------------------------------------------------------------------------- #
# Config merging (cleanroom F8) and summary config
# --------------------------------------------------------------------------- #
def test_full_defaults_copy_keeps_data_type_preset(tmp_path):
    full = copy.deepcopy(DEFAULTS)
    full["proband"]["min_gq"] = 30
    p = tmp_path / "full.json"
    p.write_text(json.dumps(full))
    cfg = load_config(str(p), "wes")
    assert cfg["data_type"] == "wes"
    assert cfg["proband"]["min_dp"] == 20 and cfg["proband"]["min_alt"] == 7
    assert cfg["parent"]["min_dp"] == 20 and cfg["mosaic"]["parent_min_dp"] == 50
    assert cfg["proband"]["min_gq"] == 30
    full["data_type"] = "panel"
    p.write_text(json.dumps(full))
    assert load_config(str(p))["parent"]["min_dp"] == 50
    assert load_config(str(p), "wgs")["parent"]["min_dp"] == 15  # explicit --data-type wins


# --------------------------------------------------------------------------- #
# QC gate (cleanroom F4, PY-02, protocol F04, F14)
# --------------------------------------------------------------------------- #
def _qc(tmp_path, samples_rows, extra_pairs=""):
    pairs = tmp_path / "p.tsv"
    pairs.write_text("#sample_a\tsample_b\trelatedness\tibs0\tibs2\tn\n"
                     "kid\tdad\t0.5\t0\t5000\t10000\nkid\tmom\t0.5\t0\t5000\t10000\ndad\tmom\t0.01\t900\t100\t10000\n"
                     + extra_pairs)
    samples = tmp_path / "s.tsv"
    samples.write_text("#family_id\tsample_id\tdepth_mean\tX_het\tX_n\tY_depth_mean\n" + samples_rows)
    sm = tmp_path / "kid.selfSM"
    sm.write_text("#SEQ_ID\tFREEMIX\nkid\t0.004\n")
    return ["--somalier-pairs", str(pairs), "--somalier-samples", str(samples), "--verifybamid", f"kid={sm}"]


OK_SEX = "F1\tkid\t30\t2\t1000\t14\nF1\tdad\t30\t1\t1000\t15\nF1\tmom\t30\t300\t1000\t0.1\n"


def test_qc_gate_parent_sex_swap_fails(trio_data, tmp_path):
    swapped = "F1\tkid\t30\t2\t1000\t14\nF1\tdad\t30\t300\t1000\t0.1\nF1\tmom\t30\t1\t1000\t15\n"
    out = str(tmp_path / "g.json")
    assert main(["qc-gate", "--ped", trio_data["ped"], "--out", out, *_qc(tmp_path, swapped)]) == 1
    res = json.load(open(out))
    assert res["status"] == "FAIL"
    assert any(f.startswith("dad:") for f in res["fail"]) and any(f.startswith("mom:") for f in res["fail"])


def test_qc_gate_unknown_sex_and_lcl_warn(trio_data, tmp_path):
    xxy = "F1\tkid\t30\t300\t1000\t14\nF1\tdad\t30\t1\t1000\t15\nF1\tmom\t30\t300\t1000\t0.1\n"
    out = str(tmp_path / "g.json")
    assert main(["qc-gate", "--ped", trio_data["ped"], "--out", out, "--lcl", "mom", *_qc(tmp_path, xxy)]) == 0
    res = json.load(open(out))
    assert res["status"] == "WARN" and res["inferred_sex"]["kid"] == "unknown"
    assert any("aneuploidy" in w for w in res["warn"])
    assert any("lymphoblastoid cell-line DNA: expect culture-derived low-VAF artefacts" in w for w in res["warn"])
    assert res["lcl"] == ["mom"]


def test_qc_gate_config_thresholds(trio_data, tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"qc": {"freemix_fail": 0.003}}))
    out = str(tmp_path / "g.json")
    assert main(["qc-gate", "--ped", trio_data["ped"], "--out", out, "--config", str(cfg), *_qc(tmp_path, OK_SEX)]) == 1
    assert any("FREEMIX" in f for f in json.load(open(out))["fail"])


def test_qc_gate_evaluates_every_trio(tmp_path):
    ped = tmp_path / "quad.ped"
    ped.write_text("F1\tkid\tdad\tmom\t1\t2\nF1\tdad\t0\t0\t1\t1\nF1\tmom\t0\t0\t2\t1\n"
                   "F2\tkid2\tdad2\tmom2\t2\t2\nF2\tdad2\t0\t0\t1\t1\nF2\tmom2\t0\t0\t2\t1\n")
    rows = OK_SEX + "F2\tkid2\t30\t300\t1000\t0.1\nF2\tdad2\t30\t1\t1000\t15\nF2\tmom2\t30\t300\t1000\t0.1\n"
    pairs = "kid2\tdad2\t0.02\t2000\t100\t10000\nkid2\tmom2\t0.5\t0\t5000\t10000\n"  # non-paternity in F2
    out = str(tmp_path / "g.json")
    assert main(["qc-gate", "--ped", str(ped), "--out", out, *_qc(tmp_path, rows, pairs)]) == 1
    res = json.load(open(out))
    assert isinstance(res, list) and [r["proband"] for r in res] == ["kid", "kid2"]
    assert res[0]["status"] == "PASS" and res[1]["status"] == "FAIL"
    assert main(["qc-gate", "--ped", str(ped), "--proband", "kid", "--out", out, *_qc(tmp_path, rows, pairs)]) == 0
    assert json.load(open(out))["status"] == "PASS"


# --------------------------------------------------------------------------- #
# VCF header: SnpEff ANN (PY-13)
# --------------------------------------------------------------------------- #
def test_snpeff_ann_header_parsed(tmp_path):
    v = tmp_path / "ann.vcf"
    v.write_text("##fileformat=VCFv4.2\n"
                 "##INFO=<ID=ANN,Number=.,Type=String,Description=\"Functional annotations: 'Allele | Annotation | "
                 "Annotation_Impact | Gene_Name | Gene_ID | Feature_Type | Feature_ID | Transcript_BioType | Rank | "
                 "HGVS.c | HGVS.p | cDNA.pos / cDNA.length | CDS.pos / CDS.length | AA.pos / AA.length | Distance | "
                 "ERRORS / WARNINGS / INFO' \">\n"
                 "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    r = VCFReader(str(v))
    assert r.csq_fields[:4] == ["Allele", "Annotation", "Annotation_Impact", "Gene_Name"]
    assert r.csq_fields[-1] == "ERRORS / WARNINGS / INFO"


def test_binomial_helpers_consistent_with_vaf():
    # the het test on AD 13,8 is not significant on n = sum(AD)
    assert stats.binom_cdf(8, 21, 0.5) > 1e-3
