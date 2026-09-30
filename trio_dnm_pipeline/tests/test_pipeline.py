import csv
import json
import math

from trio_dnm import annotation as annot
from trio_dnm import genome, signatures, stats
from trio_dnm.cli import main
from trio_dnm.config import load_config
from trio_dnm.vcf import VCFReader


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def test_fisher_and_wilson():
    assert math.isclose(stats.fisher_exact(10, 10, 10, 10), 1.0, rel_tol=1e-6)
    assert stats.fisher_exact(20, 0, 0, 20) < 1e-9
    lo, hi = stats.wilson_interval(15, 30)
    assert lo < 0.5 < hi


def test_trio_posterior_separates_dnm_from_leakage():
    kid = (stats.log_likelihoods([400, 0, 400], 0, 0, 2), 2)
    clean = (stats.log_likelihoods([0, 90, 900], 0, 0, 2), 2)
    shaky = (stats.log_likelihoods([0, 8, 300], 0, 0, 2), 2)
    good = stats.trio_dnm_posterior(kid, clean, clean, 0.0, 1.2e-8)["p_dnm"]
    bad = stats.trio_dnm_posterior(kid, shaky, clean, 0.0, 1.2e-8)["p_dnm"]
    assert good > 0.99
    assert bad < 0.5


def test_hemizygous_posterior():
    kid = (stats.log_likelihoods([], 0, 20, 1), 1)
    mom = (stats.log_likelihoods([], 30, 0, 2), 2)
    assert stats.trio_dnm_posterior(kid, None, mom, 0.0, 1.2e-8)["p_dnm"] > 0.95


def test_par_and_ploidy():
    assert genome.chrom_class("chrX", 100_000) == "PAR"
    assert genome.expected_ploidy("chrX", 3_000_000, "male") == 1
    assert genome.expected_ploidy("chrX", 3_000_000, "female") == 2
    assert genome.expected_ploidy("chrY", 3_000_000, "female") == 0


def test_sbs96_channel_is_pyrimidine_normalised():
    assert genome.sbs96_channel("G", "A", "C", "T") == "A[C>T]G"
    assert genome.sbs96_channel("C", "T", "A", "G") == "A[C>T]G"


def test_nnls_recovers_mixture():
    s1 = [1.0 / 96] * 96
    s2 = [0.0] * 96
    s2[0] = 0.5
    s2[1] = 0.5
    mix = [0.7 * a + 0.3 * b for a, b in zip(s1, s2)]
    h = signatures.nnls([s1, s2], mix)
    assert abs(h[0] - 0.7) < 1e-3 and abs(h[1] - 0.3) < 1e-3


# --------------------------------------------------------------------------- #
# call stage end-to-end
# --------------------------------------------------------------------------- #
def run_call(d):
    out = str(d["tmp"] / "t")
    rc = main(["call", "--vcf", d["vcf"], "--ped", d["ped"], "--out", out, "--fasta", d["fasta"],
               "--exclude-bed", d["bed"], "--mean-depth", "30"])
    assert rc == 0
    return out


def test_call_stage_classifies_every_scenario(trio_data):
    out = run_call(trio_data)
    rows = {int(r["pos"]): r for r in read_tsv(out + ".candidates.tsv")}
    assert 800 not in rows  # inherited from father: not a candidate

    dnm = rows[100]
    assert dnm["pass"] == "True" and dnm["track"] == "germline" and dnm["tier"] == "HIGH"
    assert float(dnm["posterior"]) > 0.99

    assert rows[200]["pass"] == "False" and rows[200]["first_fail"] == "L1_genotype"
    assert "father_alt>1" in rows[200]["fail_reasons"]

    assert rows[300]["track"] == "parental_mosaic"
    assert rows[400]["track"] == "mosaic" and rows[400]["pass"] == "True"

    assert rows[600]["first_fail"] == "L4_population"
    assert rows[700]["first_fail"] == "L3_region" and "in_mask.bed" in rows[700]["fail_reasons"]
    assert "homopolymer" in rows[505]["fail_reasons"]
    assert rows[900]["first_fail"] == "L2_read_quality"

    x = rows[3_000_100]
    assert x["pass"] == "True" and x["track"] == "germline", x

    summary = json.load(open(out + ".call_summary.json"))
    wf = summary["waterfall"]
    assert wf["mendelian_violation_candidates"] == len(rows)
    assert list(wf.values()) == sorted(wf.values(), reverse=True)

    passing = [r.pos for r in VCFReader(out + ".dnm.vcf")]
    assert 100 in passing and 200 not in passing


def test_consensus_tiers_with_external_callers(trio_data):
    t = trio_data["tmp"]
    (t / "dng.tsv").write_text("chrom\tpos\tref\talt\nchr1\t100\tC\tT\n")
    (t / "deeptrio.tsv").write_text("chr1\t100\tC\tT\n")
    out = str(t / "c")
    main(["call", "--vcf", trio_data["vcf"], "--ped", trio_data["ped"], "--out", out, "--mean-depth", "30",
          "--caller", f"dng={t / 'dng.tsv'}", "--caller", f"deeptrio={t / 'deeptrio.tsv'}",
          "--caller", f"triodenovo={t / 'deeptrio.tsv'}"])
    rows = {int(r["pos"]): r for r in read_tsv(out + ".candidates.tsv")}
    assert rows[100]["n_callers"] == "3" and rows[100]["tier"] == "HIGH"
    assert rows[3_000_100]["tier"] == "LOW"


# --------------------------------------------------------------------------- #
# annotation / ACMG
# --------------------------------------------------------------------------- #
CSQ_FIELDS = ["Allele", "Consequence", "IMPACT", "SYMBOL", "Gene", "Feature", "HGVSc", "HGVSp", "MANE_SELECT",
              "CANONICAL", "LoF", "REVEL", "am_pathogenicity", "CADD_PHRED", "SpliceAI_pred_DS_AG",
              "SpliceAI_pred_DS_AL", "SpliceAI_pred_DS_DG", "SpliceAI_pred_DS_DL", "gnomADe_AF", "phyloP100way_vertebrate"]


class _Rec:
    def __init__(self, csq):
        self.info = {"CSQ": "|".join(csq.get(f, "") for f in CSQ_FIELDS)}

    def info_float(self, k):
        return None


def _ann(csq, genes):
    cfg = load_config()
    ann = annot.annotate_record(_Rec(csq), CSQ_FIELDS, genes, cfg)
    codes, pts, cls = annot.acmg(ann, "germline", cfg, parentage_confirmed=False, validated=False)
    return ann, codes, pts, cls, annot.priority_tier(ann, cls, cfg)


GENES = {"SCN2A": {"gene": "SCN2A", "pLI": "1.0", "LOEUF": "0.1", "mis_z": "5.9", "inheritance": "AD", "disease": "DEE11"},
         "TTN": {"gene": "TTN", "pLI": "0.0", "LOEUF": "0.9", "mis_z": "-1.0"}}


def test_lof_in_haploinsufficient_gene_is_pathogenic():
    ann, codes, pts, cls, tier = _ann({"Consequence": "stop_gained", "IMPACT": "HIGH", "SYMBOL": "SCN2A",
                                        "MANE_SELECT": "NM_1", "LoF": "HC"}, GENES)
    assert "PVS1" in codes and "PM6" in codes and "PM2_Supporting" in codes
    assert cls == "Pathogenic" and tier == "Tier1"


def test_damaging_missense_calibrated_pp3():
    ann, codes, pts, cls, tier = _ann({"Consequence": "missense_variant", "IMPACT": "MODERATE", "SYMBOL": "SCN2A",
                                        "REVEL": "0.95", "am_pathogenicity": "0.98"}, GENES)
    assert "PP3_Strong" in codes and "PP2" in codes
    assert cls == "Likely_pathogenic" and tier == "Tier1"


def test_benign_synonymous():
    ann, codes, pts, cls, tier = _ann({"Consequence": "synonymous_variant", "IMPACT": "LOW", "SYMBOL": "TTN",
                                        "SpliceAI_pred_DS_AG": "0.01", "phyloP100way_vertebrate": "-0.5"}, GENES)
    assert "BP4" in codes and "BP7" in codes and cls == "VUS" and tier == "Tier4"


def test_spliceai_flags_tier3():
    ann, codes, pts, cls, tier = _ann({"Consequence": "intron_variant", "IMPACT": "MODIFIER", "SYMBOL": "TTN",
                                        "SpliceAI_pred_DS_DG": "0.45"}, GENES)
    assert "PP3" in codes and tier == "Tier3"


def test_common_variant_ba1():
    ann, codes, pts, cls, tier = _ann({"Consequence": "missense_variant", "SYMBOL": "TTN", "gnomADe_AF": "0.2"}, GENES)
    assert codes == ["BA1"] and cls == "Benign"


# --------------------------------------------------------------------------- #
# annotate stage end-to-end (no VEP: exercises sanity checks + report)
# --------------------------------------------------------------------------- #
def test_annotate_stage_runs_without_vep(trio_data):
    out = run_call(trio_data)
    rc = main(["annotate", "--vcf", out + ".dnm.vcf", "--out", out, "--fasta", trio_data["fasta"],
               "--call-summary", out + ".call_summary.json", "--paternal-age", "30", "--maternal-age", "28"])
    assert rc == 0
    qc = json.load(open(out + ".qc.json"))
    assert qc["sanity"]["n_germline_snv"] == 2
    assert qc["sanity"]["status"] == "REVIEW"  # a 2-DNM toy trio is far below WGS expectations
    assert "age_expected_dnm" in qc["sanity"]
    assert sum(qc["sanity"]["sbs96"].values()) == 2
    html = open(out + ".report.html").read()
    assert "<svg" in html and "Filter waterfall" in html


# --------------------------------------------------------------------------- #
# somatic + CH
# --------------------------------------------------------------------------- #
SOM_HEADER = """##fileformat=VCFv4.2
##INFO=<ID=CSQ,Number=.,Type=String,Description="Consequence annotations from Ensembl VEP. Format: Allele|Consequence|IMPACT|SYMBOL|HGVSp">
##FORMAT=<ID=GT,Number=1,Type=String,Description="">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTUM\tkid
"""


def test_somatic_trio_aware_and_second_hit(trio_data, tmp_path):
    r800 = trio_data["ref"]("chr1", 800)
    a800 = {"A": "G", "C": "T", "G": "A", "T": "C"}[r800]
    som = tmp_path / "som.vcf"
    som.write_text(SOM_HEADER + "\n".join([
        "chr1\t100\t.\tC\tT\t.\tPASS\tCSQ=T|missense_variant|MODERATE|TP53|p.R175H\tGT:AD:DP\t0/1:30,30:60\t0/1:15,14:29",
        "chr1\t150\t.\tA\tG\t.\tPASS\tCSQ=G|stop_gained|HIGH|NF1|p.Q10*\tGT:AD:DP\t0/1:40,20:60\t0/0:30,0:30",
        f"chr1\t800\t.\t{r800}\t{a800}\t.\tPASS\tCSQ=G|missense_variant|MODERATE|KRAS|p.G12D\tGT:AD:DP\t0/1:40,20:60\t0/0:30,0:30",
    ]) + "\n")
    genes = tmp_path / "genes.tsv"
    genes.write_text("gene\tcancer_role\thotspots\nTP53\tTSG\tp.R175H\nNF1\tTSG\t\nKRAS\toncogene\tp.G12D\n")
    germ = tmp_path / "germ.tsv"
    germ.write_text("gene\tchrom\tpos\tref\talt\tpriority_tier\tacmg_class\ttrack\nNF1\tchr1\t5\tA\tT\tTier1\tPathogenic\tgermline\n")
    out = str(tmp_path / "s")
    main(["somatic", "--vcf", str(som), "--tumor", "TUM", "--normal", "kid", "--out", out,
          "--trio-vcf", trio_data["vcf"], "--father", "dad", "--mother", "mom", "--purity", "0.6",
          "--gene-table", str(genes), "--germline-annotated", str(germ)])
    rows = {int(r["pos"]): r for r in read_tsv(out + ".somatic.tsv")}
    assert rows[100]["status"] == "FILTERED" and "ALT_in_normal" in rows[100]["reasons"]
    assert rows[150]["status"] == "PASS" and rows[150]["second_hit"].startswith("germline:")
    assert rows[800]["status"] == "FILTERED" and "inherited" in rows[800]["reasons"]
    s = json.load(open(out + ".somatic_summary.json"))
    assert s["n_pass"] == 1 and s["second_hits"]


def test_ch_screen(tmp_path):
    v = tmp_path / "ch.vcf"
    v.write_text(SOM_HEADER.replace("\tTUM\tkid", "\tdad\tmom") + "\n".join([
        "chr2\t25234373\t.\tC\tT\t.\tPASS\tCSQ=T|missense_variant|MODERATE|DNMT3A|p.R882H\tGT:AD:DP\t0/1:90,10:100\t0/0:100,0:100",
        "chr2\t100\t.\tC\tT\t.\tPASS\tCSQ=T|missense_variant|MODERATE|OTHER|p.A1V\tGT:AD:DP\t0/1:90,10:100\t0/0:100,0:100",
    ]) + "\n")
    out = str(tmp_path / "ch")
    main(["ch-screen", "--vcf", str(v), "--samples", "dad,mom", "--out", out])
    rows = read_tsv(out + ".ch_screen.tsv")
    assert len(rows) == 1 and rows[0]["gene"] == "DNMT3A" and rows[0]["sample"] == "dad"


# --------------------------------------------------------------------------- #
# QC gate and DeNovoGear parser
# --------------------------------------------------------------------------- #
def _qc_files(tmp_path, dad_rel):
    pairs = tmp_path / "f.pairs.tsv"
    pairs.write_text(
        "#sample_a\tsample_b\trelatedness\tibs0\tibs2\tn\n"
        f"kid\tdad\t{dad_rel}\t0\t5000\t10000\nkid\tmom\t0.50\t0\t5000\t10000\ndad\tmom\t0.01\t900\t100\t10000\n")
    samples = tmp_path / "f.samples.tsv"
    samples.write_text(
        "#family_id\tsample_id\tdepth_mean\tX_het\tX_n\tY_depth_mean\n"
        "F1\tkid\t30\t2\t1000\t14\nF1\tdad\t30\t1\t1000\t15\nF1\tmom\t30\t300\t1000\t0.1\n")
    sm = tmp_path / "kid.selfSM"
    sm.write_text("#SEQ_ID\tFREEMIX\nkid\t0.004\n")
    return str(pairs), str(samples), str(sm)


def test_qc_gate_pass_and_nonpaternity(trio_data, tmp_path):
    p, s, sm = _qc_files(tmp_path, 0.49)
    out = str(tmp_path / "g.json")
    args = ["qc-gate", "--ped", trio_data["ped"], "--somalier-pairs", p, "--somalier-samples", s,
            "--verifybamid", f"kid={sm}", "--out", out]
    assert main(args) == 0 and json.load(open(out))["status"] == "PASS"
    p, s, sm = _qc_files(tmp_path, 0.02)
    assert main(args) == 1
    assert any("relatedness" in f for f in json.load(open(out))["fail"])


def test_dng2tsv(tmp_path, capsys):
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "dng2tsv", os.path.join(os.path.dirname(__file__), "..", "bin", "dng2tsv.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = tmp_path / "dng.out"
    f.write_text("DENOVO-SNP id: kid ref_name: chr1 coor: 100 ref_base: C ALT: T,G maxlike_null: 1 pp_null: 0.001 "
                 "pp_dnm: 0.999 tgt_dnm(child/mom/dad): CT/CC/CC\n"
                 "DENOVO-SNP id: kid ref_name: chr1 coor: 200 ref_base: A ALT: G pp_dnm: 0.10\n")
    mod.main([str(f)])
    lines = capsys.readouterr().out.strip().split("\n")
    assert lines[1].startswith("chr1\t100\tC\tT\t") and len(lines) == 2
