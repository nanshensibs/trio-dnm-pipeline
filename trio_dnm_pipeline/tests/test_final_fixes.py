"""Regression tests for issues found by the final verification pass."""
import csv
import os

from trio_dnm import somatic
from trio_dnm.cli import main

HDR = """##fileformat=VCFv4.2
##INFO=<ID=gnomAD_AF,Number=A,Type=Float,Description="">
##FORMAT=<ID=GT,Number=1,Type=String,Description="">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="">
"""


def _gt(g, ref, alt):
    return f"{g}:{ref},{alt}:{ref + alt}:99"


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_extra_vcf_is_evaluated_per_trio(tmp_path):
    """A site present in the joint primary VCF only because trio 2 carries it must not
    hide trio 1's second-engine-only call at the same site."""
    cols = "\t".join(["kid1", "dad1", "mom1", "kid2", "dad2", "mom2"])
    head = HDR + f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{cols}\n"
    ref3 = "\t".join([_gt("0/0", 30, 0)] * 3)
    het = "\t".join([_gt("0/1", 15, 15), _gt("0/0", 30, 0), _gt("0/0", 30, 0)])
    primary = tmp_path / "p.vcf"
    primary.write_text(head + f"chr1\t300\t.\tC\tT\t50\tPASS\tgnomAD_AF=0\tGT:AD:DP:GQ\t{ref3}\t{het}\n")
    extra = tmp_path / "e.vcf"
    extra.write_text(head + f"chr1\t300\t.\tC\tT\t50\tPASS\tgnomAD_AF=0\tGT:AD:DP:GQ\t{het}\t{ref3}\n")
    ped = tmp_path / "t.ped"
    ped.write_text("F1\tkid1\tdad1\tmom1\t2\t2\nF1\tdad1\t0\t0\t1\t1\nF1\tmom1\t0\t0\t2\t1\n"
                   "F2\tkid2\tdad2\tmom2\t2\t2\nF2\tdad2\t0\t0\t1\t1\nF2\tmom2\t0\t0\t2\t1\n")
    out = str(tmp_path / "o")
    assert main(["call", "--vcf", str(primary), "--ped", str(ped), "--extra-vcf", str(extra),
                 "--mean-depth", "30", "--out", out]) == 0
    k1 = _rows(out + ".kid1.candidates.tsv")
    assert [r["pos"] for r in k1] == ["300"] and "SECOND_ENGINE_ONLY" in k1[0]["flags"]
    k2 = _rows(out + ".kid2.candidates.tsv")
    assert [r["pos"] for r in k2] == ["300"] and "SECOND_ENGINE_ONLY" not in k2[0]["flags"]


def test_hemizygous_het_like_call_is_not_mosaic(tmp_path):
    head = HDR + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tkid\tdad\tmom\n"
    vcf = tmp_path / "x.vcf"
    vcf.write_text(head + f"chrX\t50000000\t.\tC\tT\t50\tPASS\tgnomAD_AF=0\tGT:AD:DP:GQ\t"
                          f"{_gt('0/1', 15, 15)}\t{_gt('0/0', 30, 0)}\t{_gt('0/0', 30, 0)}\n")
    ped = tmp_path / "t.ped"
    ped.write_text("F1\tkid\tdad\tmom\t1\t2\nF1\tdad\t0\t0\t1\t1\nF1\tmom\t0\t0\t2\t1\n")
    out = str(tmp_path / "o")
    main(["call", "--vcf", str(vcf), "--ped", str(ped), "--mean-depth", "30", "--out", out])
    r = _rows(out + ".candidates.tsv")[0]
    assert r["track"] == "germline" and r["pass"] == "False"
    assert "HEMIZYGOUS_HET_LIKE" in r["flags"]


def test_nan_copy_number_is_missing(tmp_path):
    seg = tmp_path / "seg.tsv"
    seg.write_text("chrom\tstart\tend\ttotal_cn\tminor_cn\nchr1\t1\t5000\tnan\tnan\n")
    s = somatic.segment_at(somatic.load_segments(str(seg)), "chr1", 100)
    assert s["total_cn"] == 2.0 and s["minor_cn"] is None
    assert somatic.estimate_multiplicity(0.3, 0.6, s["total_cn"], s["minor_cn"]) >= 1


def test_annotate_bad_signature_matrix_writes_nothing(tmp_path):
    from trio_dnm import demo
    d = demo.build_synthetic_trio(str(tmp_path / "in"))
    out = str(tmp_path / "t")
    main(["call", "--vcf", d["vcf"], "--ped", d["ped"], "--mean-depth", "30", "--out", out])
    sig = tmp_path / "sig.tsv"
    sig.write_text("Type\tOTHER\nA[C>A]A\t1.0\n")
    try:
        main(["annotate", "--vcf", out + ".dnm.vcf", "--fasta", d["fasta"], "--signatures", str(sig), "--out", out])
    except SystemExit:
        pass
    assert not os.path.exists(out + ".annotated.tsv")


def test_missing_input_is_a_clean_error(tmp_path, capsys):
    rc = main(["call", "--vcf", str(tmp_path / "nope.vcf"), "--ped", str(tmp_path / "nope.ped"), "--out", "x"])
    assert rc == 2 and "trio-dnm:" in capsys.readouterr().err
