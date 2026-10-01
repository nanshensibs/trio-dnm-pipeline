"""Regression tests for the annotation / ACMG / prioritisation / sanity /
signatures / report review fixes."""
import csv
import json

import pytest

from trio_dnm import annotation as annot
from trio_dnm import prioritize, report, sanity, signatures
from trio_dnm.cli import main
from trio_dnm.config import load_config
from trio_dnm.genome import SBS96

FIELDS = ["Allele", "Consequence", "IMPACT", "SYMBOL", "HGVSp", "LoF", "LoF_filter", "LoF_flags", "NMD", "REVEL",
          "REVEL_rankscore", "SpliceAI_pred_DS_AG", "SpliceAI_pred_DS_DG", "gnomADv4_AF_joint", "gnomADe_AF",
          "phyloP100way_vertebrate"]

GENES = {
    "SCN2A": {"gene": "SCN2A", "pLI": "1.0", "LOEUF": "0.1", "mis_z": "5.9", "inheritance": "AD", "disease": "DEE11"},
    "TTN": {"gene": "TTN", "pLI": "0.0", "LOEUF": "0.9", "mis_z": "-1.0"},
    "XGENE": {"gene": "XGENE", "pLI": "0.1", "LOEUF": "0.9", "mis_z": "0.0", "inheritance": "XLR", "disease": "X disorder"},
    "KRAS": {"gene": "KRAS", "hotspots": "p.G12D, p.(G13D)"},
    "TP53": {"gene": "TP53", "hotspots": "p.Arg175His,p.Arg213Ter"},
}


class _Rec:
    def __init__(self, csq):
        self.info = {"CSQ": "|".join(csq.get(f, "") for f in FIELDS)}

    def info_float(self, k):
        return None


def _ann(csq, genes=GENES, cfg=None):
    cfg = cfg or load_config()
    ann = annot.annotate_record(_Rec(csq), FIELDS, genes, cfg)
    codes, pts, cls = annot.acmg(ann, "germline", cfg, parentage_confirmed=False, validated=False)
    return ann, codes, pts, cls, annot.priority_tier(ann, cls, cfg)


def _missense(revel):
    return _ann({"Consequence": "missense_variant", "SYMBOL": "TTN", "REVEL": str(revel)})[1]


# --------------------------------------------------------------------------- #
# REVEL calibration (Pejaver et al. 2022)
# --------------------------------------------------------------------------- #
def test_revel_thresholds_match_pejaver_2022():
    cal = load_config()["annotation"]["calibration"]["REVEL"]
    assert cal["pp3"][:3] == [0.644, 0.773, 0.932]
    assert cal["bp4"] == [0.290, 0.183, 0.016, 0.003]


@pytest.mark.parametrize("revel,code", [
    (0.65, "PP3"), (0.80, "PP3_Moderate"), (0.95, "PP3_Strong"),
    (0.25, "BP4"), (0.10, "BP4_Moderate"), (0.03, "BP4_Moderate"), (0.01, "BP4_Strong"), (0.002, "BP4_VeryStrong"),
])
def test_revel_strengths(revel, code):
    codes = _missense(revel)
    assert code in codes, codes


def test_revel_rankscore_is_not_used_as_raw_score():
    ann, codes, *_ = _ann({"Consequence": "missense_variant", "SYMBOL": "TTN", "REVEL_rankscore": "0.80"})
    assert ann["REVEL"] is None
    assert not any(c.startswith("PP3") or c.startswith("BP4") for c in codes)


# --------------------------------------------------------------------------- #
# PVS1 / NMD escape
# --------------------------------------------------------------------------- #
def test_pvs1_loftee_lc_downgrades_one_level():
    ann, codes, pts, cls, tier = _ann({"Consequence": "stop_gained", "SYMBOL": "SCN2A", "LoF": "LC"})
    assert codes == ["PVS1_Strong", "PM6", "PM2_Supporting"]
    assert pts == 7 and cls == "Likely_pathogenic"
    assert "PVS1_Supporting" in _ann({"Consequence": "start_lost", "SYMBOL": "SCN2A", "LoF": "LC"})[1]
    assert "PVS1_Moderate" in _ann({"Consequence": "stop_gained", "SYMBOL": "SCN2A", "LoF": "LC",
                                    "NMD": "NMD_escaping_variant"})[1]


def test_nmd_escape_from_loftee_filter_end_trunc():
    ann, codes, *_ = _ann({"Consequence": "stop_gained", "SYMBOL": "SCN2A", "LoF": "LC", "LoF_filter": "END_TRUNC"})
    assert ann["nmd_escape"] is True and ann["loftee_filter"] == "END_TRUNC"
    assert "PVS1_Moderate" in codes
    ann, codes, *_ = _ann({"Consequence": "stop_gained", "SYMBOL": "SCN2A", "LoF": "HC", "LoF_filter": "END_TRUNC"})
    assert "PVS1_Strong" in codes
    assert _ann({"Consequence": "stop_gained", "SYMBOL": "SCN2A", "NMD": "NMD_escaping_variant"})[0]["nmd_escape"]
    assert not _ann({"Consequence": "stop_gained", "SYMBOL": "SCN2A", "LoF": "HC"})[0]["nmd_escape"]
    assert prioritize.ANNOT_COLUMNS[-1] == "loftee_filter"


# --------------------------------------------------------------------------- #
# BP7 needs phyloP
# --------------------------------------------------------------------------- #
def test_bp7_requires_phylop():
    syn = {"Consequence": "synonymous_variant", "SYMBOL": "TTN", "SpliceAI_pred_DS_AG": "0.01"}
    assert "BP7" not in _ann(syn)[1]
    assert "BP7" in _ann(dict(syn, phyloP100way_vertebrate="-0.5"))[1]
    assert "BP7" not in _ann(dict(syn, phyloP100way_vertebrate="5.2"))[1]


# --------------------------------------------------------------------------- #
# Tiers
# --------------------------------------------------------------------------- #
def test_canonical_splice_not_tier1_goes_to_tier4():
    tier = _ann({"Consequence": "splice_donor_variant", "SYMBOL": "TTN", "SpliceAI_pred_DS_DG": "0.9"})[4]
    assert tier == "Tier4"
    tier = _ann({"Consequence": "splice_acceptor_variant&intron_variant", "SYMBOL": "TTN", "SpliceAI_pred_DS_AG": "0.9"})[4]
    assert tier == "Tier4"
    assert _ann({"Consequence": "intron_variant", "SYMBOL": "TTN", "SpliceAI_pred_DS_DG": "0.25"})[4] == "Tier3"
    assert _ann({"Consequence": "splice_region_variant&synonymous_variant", "SYMBOL": "TTN",
                 "SpliceAI_pred_DS_DG": "0.6"})[4] == "Tier3"
    # canonical splice in a haploinsufficient gene is still Tier 1
    assert _ann({"Consequence": "splice_donor_variant", "SYMBOL": "SCN2A", "SpliceAI_pred_DS_DG": "0.9"})[4] == "Tier1"


@pytest.mark.parametrize("inh,dominant", [
    ("AD", True), ("ad", True), ("XLD", True), ("Autosomal dominant", True), ("AR;AD", True), ("AR/XLD", True),
    ("AR, AD", True), ("XLR", False), ("XL", False), ("AR", False), ("AR,XLR", False), ("", False),
    ("somatic mosaic", False),
])
def test_dominant_inheritance_tokens(inh, dominant):
    assert annot.gene_constraint({"inheritance": inh}, load_config())["dominant"] is dominant


def test_xlr_lof_is_not_promoted_as_dominant():
    assert _ann({"Consequence": "frameshift_variant", "SYMBOL": "XGENE"})[4] == "Tier4"


# --------------------------------------------------------------------------- #
# Population AF alias
# --------------------------------------------------------------------------- #
def test_gnomad_joint_af_from_vep_custom_is_used_first():
    ann, codes, *_ = _ann({"Consequence": "missense_variant", "SYMBOL": "TTN", "gnomADv4_AF_joint": "0.002",
                           "gnomADe_AF": "0.000001"})
    assert ann["gnomad_af"] == 0.002
    assert "BS1" in codes and "PM2_Supporting" not in codes


# --------------------------------------------------------------------------- #
# Hotspots
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", ["p.Gly12Asp", "p.G12D", "G12D", "ENSP00000308495.3:p.Gly12Asp", "p.(Gly12Asp)", " p.(G12D) "])
def test_norm_pchange_equivalent_notations(raw):
    assert annot.norm_pchange(raw) == "G12D"


def test_norm_pchange_stop_and_synonymous():
    assert annot.norm_pchange("p.Arg424Ter") == annot.norm_pchange("p.R424*") == "R424*"
    assert annot.norm_pchange("ENSP1:p.Ser248%3D") == "S248="
    assert annot.norm_pchange("") == ""


@pytest.mark.parametrize("gene,hgvsp,hit", [
    ("KRAS", "ENSP00000308495.3:p.Gly12Asp", True),
    ("KRAS", "ENSP00000308495.3:p.Gly13Asp", True),
    ("KRAS", "ENSP00000308495.3:p.Gly12Val", False),
    ("TP53", "p.R175H", True),
    ("TP53", "ENSP1:p.Arg213Ter", True),
    ("TP53", "", False),
])
def test_hotspot_matching_across_notations(gene, hgvsp, hit):
    ann = _ann({"Consequence": "missense_variant", "SYMBOL": gene, "HGVSp": hgvsp})[0]
    assert ann["hotspot"] is hit


# --------------------------------------------------------------------------- #
# Signatures (COSMIC v3.4 split SBS40 into SBS40a/b/c)
# --------------------------------------------------------------------------- #
def _sig_matrix(path, names):
    with open(path, "w") as fh:
        fh.write("Type\t" + "\t".join(names) + "\n")
        for ch in SBS96:
            fh.write(ch + "\t" + "\t".join(f"{1 / 96:.6f}" for _ in names) + "\n")
    return str(path)


def test_default_germline_set_matches_cosmic_v34(tmp_path, capsys):
    assert signatures.GERMLINE_SIGNATURES == ["SBS1", "SBS5", "SBS40a"]
    p = _sig_matrix(tmp_path / "v34.tsv", ["SBS1", "SBS2", "SBS5", "SBS40a", "SBS40b", "SBS40c"])
    names, mats = signatures.load_signatures(p, signatures.GERMLINE_SIGNATURES)
    assert names == ["SBS1", "SBS5", "SBS40a"] and set(mats) == set(names)
    assert "warning" not in capsys.readouterr().err


def test_sbs40_resolves_to_sbs40a_with_warning(tmp_path, capsys):
    p = _sig_matrix(tmp_path / "v34.tsv", ["SBS1", "SBS5", "SBS40a", "SBS40b", "SBS40c"])
    names, _ = signatures.load_signatures(p, ["SBS1", "SBS5", "SBS40"])
    assert names == ["SBS1", "SBS5", "SBS40a"]
    assert "SBS40a" in capsys.readouterr().err


def test_sbs40a_falls_back_to_sbs40_in_older_matrix(tmp_path, capsys):
    p = _sig_matrix(tmp_path / "v33.tsv", ["SBS1", "SBS5", "SBS40"])
    names, _ = signatures.load_signatures(p, signatures.GERMLINE_SIGNATURES)
    assert names == ["SBS1", "SBS5", "SBS40"]
    assert "SBS40a" in capsys.readouterr().err


def test_missing_signatures_warn_or_exit(tmp_path, capsys):
    p = _sig_matrix(tmp_path / "m.tsv", ["SBS1", "SBS5"])
    names, _ = signatures.load_signatures(p, ["SBS1", "SBS99"])
    assert names == ["SBS1"] and "SBS99" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        signatures.load_signatures(p, ["SBS98", "SBS99"])
    assert signatures.load_signatures(p, None)[0] == ["SBS1", "SBS5"]


# --------------------------------------------------------------------------- #
# Parent of origin
# --------------------------------------------------------------------------- #
def test_poo_empty_origin_never_matches(tmp_path):
    f = tmp_path / "poo.tsv"
    f.write_text("chrom\tpos\torigin\nchr1\t100\tpaternal\nchr1\t200\t\nchr1\t300\t \nchr1\t400\tmaternal\n")
    assert sanity.load_parent_of_origin(str(f), "", "") == {"1:100": "paternal", "1:400": "maternal"}


UNFAZED = ("#chrom\tstart\tend\tkid\torigin_parent\tother_parent\tevidence_count\tevidence_types\n"
           "chr1\t99\t100\tkid\tdad\tmom\t3\tREADBACKED\n"
           "chrX\t3000099\t3000100\tkid\tmom\tdad\t2\tREADBACKED\n"
           "chr1\t199\t200\tkid\t\t\t0\t\n")


def test_poo_unfazed_ids(tmp_path):
    f = tmp_path / "poo.bed"
    f.write_text(UNFAZED)
    assert sanity.load_parent_of_origin(str(f), "dad", "mom") == {"1:100": "paternal", "X:3000100": "maternal"}
    assert sanity.load_parent_of_origin(str(f), "", "") == {}
    # a mother ID that looks like a keyword still resolves by ID first
    f.write_text(UNFAZED.replace("mom", "patricia"))
    assert sanity.load_parent_of_origin(str(f), "dad", "patricia")["X:3000100"] == "maternal"


def _call(trio_data):
    out = str(trio_data["tmp"] / "t")
    assert main(["call", "--vcf", trio_data["vcf"], "--ped", trio_data["ped"], "--out", out, "--fasta",
                 trio_data["fasta"], "--exclude-bed", trio_data["bed"], "--mean-depth", "30"]) == 0
    return out


@pytest.mark.parametrize("with_summary", [True, False])
def test_annotate_poo_with_and_without_call_summary(trio_data, with_summary):
    out = _call(trio_data)
    poo = trio_data["tmp"] / "poo.bed"
    poo.write_text(UNFAZED)
    args = ["annotate", "--vcf", out + ".dnm.vcf", "--out", out, "--parent-of-origin", str(poo)]
    if with_summary:
        args += ["--call-summary", out + ".call_summary.json"]
    assert main(args) == 0
    san = json.load(open(out + ".qc.json"))["sanity"]
    assert san["phased"] == 2 and san["paternal_fraction"] == 0.5


# --------------------------------------------------------------------------- #
# Sanity expectations
# --------------------------------------------------------------------------- #
def _snvs(n, vaf=0.5):
    return [{"chrom": "1", "pos": i + 1, "ref": "C", "alt": "T", "track": "germline", "proband_vaf": vaf} for i in range(n)]


@pytest.mark.parametrize("dt,n,warn", [
    ("wes", 0, None), ("wes", 1, None), ("wes", 3, None), ("wes", 4, "outside expected [0, 3]"),
    ("wes", 6, "hard range"), ("panel", 0, None), ("panel", 1, None), ("panel", 2, "outside expected [0, 1]"),
    ("panel", 3, "hard range"), ("wgs", 0, "hard range"), ("wgs", 40, "outside expected [45, 95]"), ("wgs", 60, None),
])
def test_count_expectations_per_trio(dt, n, warn):
    cfg = load_config(data_type=dt)
    ws = [w for w in sanity.evaluate(_snvs(n), cfg)["warnings"] if w.startswith("SNV DNM count")]
    if warn is None:
        assert ws == []
    else:
        assert len(ws) == 1 and warn in ws[0]


def test_wes_zero_dnms_passes():
    assert sanity.evaluate([], load_config(data_type="wes"))["status"] == "PASS"
    assert load_config(data_type="panel")["expectations"]["panel"]["hard_high"] == 2


def test_median_vaf_range_from_config():
    cfg = load_config(data_type="wes")
    assert cfg["sanity"]["median_vaf_range"] == [0.42, 0.58]
    variants = _snvs(3, vaf=0.35) + [dict(v, ref="CA", alt="C") for v in _snvs(8, vaf=0.35)]
    cfg["expectations"]["wes"]["indel"] = [0, 10]
    assert any("median germline DNM VAF" in w for w in sanity.evaluate(variants, cfg)["warnings"])
    cfg["sanity"]["median_vaf_range"] = [0.3, 0.7]
    assert not any("median germline DNM VAF" in w for w in sanity.evaluate(variants, cfg)["warnings"])


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def test_report_escapes_every_interpolated_value(tmp_path):
    bad = "<script>alert(1)</script>"
    result = {
        "call_summary": {"proband": "kid", "waterfall": {"step": 1},
                         "parental_leakage": {"father_any_alt_frac": bad, "mother_any_alt_frac": "<b>m</b>", "n": "<i>n</i>"}},
        "sanity": {"sbs96": {ch: 1 for ch in SBS96},
                   "signatures": {"cosine": "<i>c</i>", "exposures": {"<img src=x onerror=alert(1)>": 0.5, "SBS5": "<u>x</u>"}}},
    }
    path = tmp_path / "r.html"
    report.write_html(str(path), result, [])
    html = path.read_text()
    for raw in (bad, "<b>m</b>", "<i>n</i>", "<i>c</i>", "<img src=x", "<u>x</u>"):
        assert raw not in html
    assert "&lt;img src=x onerror=alert(1)&gt; 50%" in html


def test_report_cg_bars_visible_in_dark_mode(tmp_path):
    svg = report.spectrum_svg({ch: 1 for ch in SBS96})
    assert svg.count('class="sbs1"') == 16
    dark = report.CSS.split("prefers-color-scheme:dark")[1].split("\n")[0]
    assert "--sbs-cg:" in dark and "--sbs-cg:#010101" not in dark
    assert ".sbs1{fill:var(--sbs-cg)}" in report.CSS


def test_annotated_tsv_keeps_existing_columns(trio_data):
    out = _call(trio_data)
    assert main(["annotate", "--vcf", out + ".dnm.vcf", "--out", out]) == 0
    with open(out + ".annotated.tsv") as fh:
        header = next(csv.reader(fh, delimiter="\t"))
    assert header == prioritize.ANNOT_COLUMNS
    assert header.index("phenotype_score") == len(header) - 2
