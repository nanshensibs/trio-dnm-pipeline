"""Stage 8: functional annotation, calibrated in-silico evidence, provisional
ACMG/AMP classification and DNM prioritisation.

Reads the CSQ field written by Ensembl VEP (with the plugins listed in the
protocol) plus an optional gene-level table. Field names from VEP plugins and
from dbNSFP are both accepted (see ``ALIASES``).

The ACMG output is *provisional*: it automates only the criteria that can be
evaluated from annotation (PVS1, PS2/PM6, PM2, PP2, PP3/BP4, BA1/BS1, BP7) and
scores them with the Tavtigian et al. (2020) Bayesian point system. Phenotype
specificity, segregation, functional data and PVS1 decision-tree nuances
require curation by a qualified reviewer.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .vcf import open_text, parse_csq, to_float

ALIASES: Dict[str, List[str]] = {
    # Raw REVEL score only: dbNSFP REVEL_rankscore is rank-normalised and is not
    # on the scale of the Pejaver (2022) thresholds.
    "REVEL": ["REVEL", "REVEL_score"],
    "CADD_PHRED": ["CADD_PHRED", "CADD_phred"],
    "AlphaMissense": ["am_pathogenicity", "AlphaMissense_score", "AlphaMissense"],
    "AlphaMissense_class": ["am_class", "AlphaMissense_pred"],
    "BayesDel_noAF": ["BayesDel_noAF_score", "BayesDel_noAF"],
    "PrimateAI": ["PrimateAI", "PrimateAI_score", "PrimateAI-3D", "PrimateAI3D"],
    "ESM1b": ["ESM1b_score", "ESM1b"],
    "EVE": ["EVE_score", "EVE"],
    "MetaRNN": ["MetaRNN_score"],
    "phyloP": ["phyloP100way_vertebrate", "phyloP100way", "phyloP", "phyloP447way", "Conservation", "phyloP241way_mammal"],
    "GERP": ["GERP++_RS", "GERP_91_mammals", "GERP"],
    # gnomADv4_AF_joint: VEP --custom short_name=gnomADv4 with field AF_joint.
    "gnomAD_AF": ["gnomADv4_AF_joint", "gnomADv4_AF", "gnomAD_AF", "gnomADe_AF", "gnomADg_AF", "gnomAD_joint_AF", "MAX_AF"],
    "ClinVar": ["ClinVar_CLNSIG", "CLIN_SIG", "ClinVar"],
    "ClinVar_disease": ["ClinVar_CLNDN"],
    "Pangolin": ["Pangolin", "pangolin"],
    "cCRE": ["ENCODE_cCRE", "cCRE", "SCREEN_cCRE"],
    "NMD": ["NMD"],
}
SPLICEAI_KEYS = ["SpliceAI_pred_DS_AG", "SpliceAI_pred_DS_AL", "SpliceAI_pred_DS_DG", "SpliceAI_pred_DS_DL"]

POINTS = {"Supporting": 1, "Moderate": 2, "Strong": 4, "VeryStrong": 8}
STRENGTHS = ["Supporting", "Moderate", "Strong", "VeryStrong"]

NONCODING = {
    "5_prime_UTR_variant", "3_prime_UTR_variant", "intron_variant", "upstream_gene_variant",
    "downstream_gene_variant", "regulatory_region_variant", "TF_binding_site_variant",
    "intergenic_variant", "non_coding_transcript_exon_variant", "non_coding_transcript_variant",
    "mature_miRNA_variant",
}
SPLICE_REGION = {"splice_region_variant", "splice_donor_5th_base_variant", "splice_donor_region_variant", "splice_polypyrimidine_tract_variant"}
CANONICAL_SPLICE = {"splice_donor_variant", "splice_acceptor_variant"}
DOMINANT_TOKENS = {"AD", "XLD", "DOMINANT"}

AA3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H",
    "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Ter": "*",
}
_AA3_RE = re.compile("|".join(AA3TO1))


def norm_pchange(s: Optional[str]) -> str:
    """Protein change in a comparable form: transcript prefix, 'p.' and
    parentheses removed, three-letter amino acids as one-letter codes, so
    'ENSP0001.3:p.(Gly12Asp)', 'p.G12D' and 'G12D' all give 'G12D'."""
    s = (s or "").strip().split(":")[-1].replace("%3D", "=")
    if s.startswith("p."):
        s = s[2:]
    s = s.replace("(", "").replace(")", "")
    return _AA3_RE.sub(lambda m: AA3TO1[m.group(0)], s)


# --------------------------------------------------------------------------- #
# Gene-level knowledge
# --------------------------------------------------------------------------- #
def load_gene_table(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    """TSV keyed by a ``gene`` (HGNC symbol) column. Recognised optional columns:
    pLI, LOEUF, mis_z, shet, hi_score (ClinGen dosage), disease, inheritance
    (AD/AR/XLD/XLR/...), mechanism (LoF/GoF/DN), cancer_role, hotspots
    (comma-separated protein changes in one- or three-letter notation, e.g.
    p.R132H or p.Arg132His), expression (free text)."""
    out: Dict[str, Dict[str, str]] = {}
    if not path:
        return out
    with open_text(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            if not line.strip():
                continue
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            g = row.get("gene") or row.get("symbol") or row.get("SYMBOL")
            if g:
                out[g] = row
    return out


def load_scores(path: Optional[str]) -> Dict[str, float]:
    """Two-column TSV gene -> score in [0,1] (e.g. Exomiser/LIRICAL phenotype score)."""
    out: Dict[str, float] = {}
    if not path:
        return out
    with open_text(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and to_float(f[1]) is not None:
                out[f[0]] = float(f[1])
    return out


def gene_constraint(g: Dict[str, str], cfg: dict) -> Dict[str, object]:
    a = cfg["annotation"]
    pli, loeuf = to_float(g.get("pLI")), to_float(g.get("LOEUF"))
    misz, shet = to_float(g.get("mis_z")), to_float(g.get("shet"))
    hi = to_float(g.get("hi_score"))
    lof_intol = any([
        pli is not None and pli >= a["pli_constrained"],
        loeuf is not None and loeuf < a["loeuf_constrained"],
        shet is not None and shet >= a["shet_constrained"],
        hi == 3,
    ])
    mis_con = misz is not None and misz >= a["misz_constrained"]
    mech = (g.get("mechanism") or "").lower()
    # Whole tokens only: 'XLR' / 'AR' must not count as dominant.
    inh = set(re.split(r"[;,/|\s]+", (g.get("inheritance") or "").upper()))
    return {
        "lof_intolerant": lof_intol,
        "missense_constrained": mis_con,
        "lof_mechanism": lof_intol or "lof" in mech or "haploinsuff" in mech,
        "dominant": bool(inh & DOMINANT_TOKENS),
        "pLI": pli, "LOEUF": loeuf, "mis_z": misz, "shet": shet, "hi_score": hi,
    }


# --------------------------------------------------------------------------- #
# Transcript selection and field extraction
# --------------------------------------------------------------------------- #
def pick_transcript(csqs: List[Dict[str, str]], allele: Optional[str] = None) -> Optional[Dict[str, str]]:
    if not csqs:
        return None
    pool = csqs
    for key, ok in (("MANE_SELECT", lambda v: bool(v)), ("PICK", lambda v: v == "1"), ("CANONICAL", lambda v: v == "YES")):
        hits = [c for c in pool if ok(c.get(key, ""))]
        if hits:
            return hits[0]
    return pool[0]


def getf(csq: Dict[str, str], name: str) -> Optional[float]:
    """Numeric field via aliases; '&'-joined multi-values -> max."""
    for k in ALIASES.get(name, [name]):
        v = csq.get(k)
        if v not in (None, "", "."):
            vals = [to_float(x) for x in str(v).replace(",", "&").split("&")]
            vals = [x for x in vals if x is not None]
            if vals:
                return max(vals)
    return None


def gets(csq: Dict[str, str], name: str) -> str:
    for k in ALIASES.get(name, [name]):
        v = csq.get(k)
        if v not in (None, "", "."):
            return str(v)
    return ""


def spliceai_max(csq: Dict[str, str]) -> Optional[float]:
    vals = [to_float(csq.get(k)) for k in SPLICEAI_KEYS]
    vals = [v for v in vals if v is not None]
    if vals:
        return max(vals)
    raw = csq.get("SpliceAI_pred", "")
    if raw:  # SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|...
        parts = raw.split("|")
        nums = [to_float(x) for x in parts[1:5]]
        nums = [n for n in nums if n is not None]
        if nums:
            return max(nums)
    pang = getf(csq, "Pangolin")
    return abs(pang) if pang is not None else None


# --------------------------------------------------------------------------- #
# ACMG
# --------------------------------------------------------------------------- #
def calibrated_strength(value: Optional[float], thresholds: List[Optional[float]], higher_is_worse: bool) -> Optional[str]:
    if value is None:
        return None
    best = None
    for strength, t in zip(STRENGTHS, thresholds):
        if t is None:
            continue
        if (higher_is_worse and value >= t) or (not higher_is_worse and value <= t):
            best = strength
    return best


def classify_points(points: int, standalone_benign: bool) -> str:
    if standalone_benign:
        return "Benign"
    if points >= 10:
        return "Pathogenic"
    if points >= 6:
        return "Likely_pathogenic"
    if points >= 0:
        return "VUS"
    if points >= -6:
        return "Likely_benign"
    return "Benign"


def acmg(ann: Dict[str, object], track: str, cfg: dict, parentage_confirmed: bool, validated: bool) -> Tuple[List[str], int, str]:
    a = cfg["annotation"]
    codes: List[Tuple[str, str, int]] = []  # (code, strength, signed points)
    csq_terms = set(str(ann.get("consequence", "")).split("&"))
    gc = ann.get("_constraint") or {}
    af = ann.get("gnomad_af")
    is_lof = bool(csq_terms & set(a["lof_consequences"]))
    is_mis = bool(csq_terms & set(a["missense_consequences"]))

    # PVS1 — simplified Abou Tayoun (2018) logic; requires LoF disease mechanism.
    if is_lof and gc.get("lof_mechanism"):
        strength = "VeryStrong"
        if "start_lost" in csq_terms:
            strength = "Moderate"
        elif ann.get("nmd_escape"):
            strength = "Strong"
        if ann.get("loftee") == "LC":  # one level down
            strength = STRENGTHS[max(0, STRENGTHS.index(strength) - 1)]
        codes.append(("PVS1", strength, POINTS[strength]))

    # De novo evidence.
    if track == "germline":
        if a["de_novo_code"] == "PS2" or (a["de_novo_code"] == "auto" and parentage_confirmed and validated):
            codes.append(("PS2", "Strong", 4))
        else:
            codes.append(("PM6", "Moderate", 2))
    elif track == "mosaic":
        codes.append(("PS2" if validated else "PM6", "Moderate" if validated else "Supporting", 2 if validated else 1))

    # Population.
    if af is not None and af > a["ba1_af"]:
        return ["BA1"], -99, "Benign"
    if af is not None and af > a["bs1_af"]:
        codes.append(("BS1", "Strong", -4))
    elif af is None or af <= a["pm2_max_af"]:
        codes.append(("PM2", "Supporting", 1))

    # PP2: missense in a missense-constrained gene.
    if is_mis and gc.get("missense_constrained"):
        codes.append(("PP2", "Supporting", 1))

    # PP3 / BP4 — single pre-selected tool (missense) or SpliceAI (splicing).
    sai = ann.get("spliceai")
    if not is_lof:
        pp3 = None
        if is_mis:
            tool = a["pp3_tool"]
            cal = a["calibration"][tool]
            val = ann.get(tool)
            s = calibrated_strength(val, cal["pp3"], True)
            if s:
                pp3 = ("PP3", s, POINTS[s])
            else:
                b = calibrated_strength(val, cal["bp4"], False)
                if b and not (sai is not None and sai >= a["spliceai_bp4"]):
                    codes.append(("BP4", b, -POINTS[b]))
        if sai is not None and sai >= a["spliceai_pp3"] and pp3 is None:
            pp3 = ("PP3", "Supporting", 1)
        if pp3:
            codes.append(pp3)
        elif not is_mis and sai is not None and sai <= a["spliceai_bp4"]:
            codes.append(("BP4", "Supporting", -1))
            phylop = ann.get("phyloP")
            if "synonymous_variant" in csq_terms and phylop is not None and phylop < 0.1:
                codes.append(("BP7", "Supporting", -1))

    points = sum(p for _, _, p in codes)
    labels = [c if s == default_strength(c) else f"{c}_{s}" for c, s, _ in codes]
    return labels, points, classify_points(points, False)


def default_strength(code: str) -> str:
    if code.startswith("PVS"):
        return "VeryStrong"
    if code[:2] in ("PS", "BS"):
        return "Strong"
    if code.startswith("PM"):
        return "Moderate"
    return "Supporting"


# --------------------------------------------------------------------------- #
# Per-variant annotation
# --------------------------------------------------------------------------- #
def annotate_record(rec, csq_fields: List[str], genes: Dict[str, Dict[str, str]], cfg: dict) -> Dict[str, object]:
    a = cfg["annotation"]
    csqs = parse_csq(rec.info.get("CSQ") or rec.info.get("ANN"), csq_fields)
    csq = pick_transcript(csqs) or {}
    symbol = csq.get("SYMBOL") or csq.get("Gene_Name") or ""
    g = genes.get(symbol, {})
    gc = gene_constraint(g, cfg)
    af = getf(csq, "gnomAD_AF")
    if af is None:
        for k in cfg["population"]["af_keys"]:
            if rec.info_float(k) is not None:
                af = rec.info_float(k)
                break
    nmd = (gets(csq, "NMD") or "").lower()
    lof_flags = csq.get("LoF_flags", "")
    lof_filter = csq.get("LoF_filter", "")  # LOFTEE END_TRUNC is a filter, not a flag
    hgvsp = csq.get("HGVSp", "")
    protein_change = norm_pchange(hgvsp)
    hotspots = {norm_pchange(h) for h in (g.get("hotspots") or "").split(",") if h.strip()}
    ann: Dict[str, object] = {
        "gene": symbol,
        "gene_id": csq.get("Gene", ""),
        "transcript": csq.get("Feature", ""),
        "mane": csq.get("MANE_SELECT", ""),
        "consequence": csq.get("Consequence", csq.get("Annotation", "")),
        "impact": csq.get("IMPACT", csq.get("Annotation_Impact", "")),
        "hgvsc": csq.get("HGVSc", ""),
        "hgvsp": hgvsp,
        "exon": csq.get("EXON", ""),
        "domains": csq.get("DOMAINS", ""),
        "loftee": csq.get("LoF", ""),
        "loftee_flags": lof_flags,
        "loftee_filter": lof_filter,
        "nmd_escape": "escap" in nmd or "END_TRUNC" in lof_filter or "END_TRUNC" in lof_flags or "NMD" in lof_flags,
        "gnomad_af": af,
        "clinvar": gets(csq, "ClinVar") or str(rec.info.get("CLNSIG", "") or ""),
        "clinvar_disease": gets(csq, "ClinVar_disease") or str(rec.info.get("CLNDN", "") or ""),
        "REVEL": getf(csq, "REVEL"),
        "CADD_PHRED": getf(csq, "CADD_PHRED"),
        "AlphaMissense": getf(csq, "AlphaMissense"),
        "AlphaMissense_class": gets(csq, "AlphaMissense_class"),
        "BayesDel_noAF": getf(csq, "BayesDel_noAF"),
        "PrimateAI": getf(csq, "PrimateAI"),
        "ESM1b": getf(csq, "ESM1b"),
        "EVE": getf(csq, "EVE"),
        "MetaRNN": getf(csq, "MetaRNN"),
        "spliceai": spliceai_max(csq),
        "phyloP": getf(csq, "phyloP"),
        "GERP": getf(csq, "GERP"),
        "cCRE": gets(csq, "cCRE"),
        "utr_annotator": csq.get("5UTR_consequence", "") or csq.get("5UTR_annotation", ""),
        "pLI": gc["pLI"], "LOEUF": gc["LOEUF"], "mis_z": gc["mis_z"], "shet": gc["shet"],
        "hi_score": gc["hi_score"],
        "lof_intolerant": gc["lof_intolerant"],
        "missense_constrained": gc["missense_constrained"],
        "disease": g.get("disease", ""),
        "inheritance": g.get("inheritance", ""),
        "mechanism": g.get("mechanism", ""),
        "cancer_role": g.get("cancer_role", ""),
        "hotspot": bool(protein_change and protein_change in hotspots),
        "_constraint": gc,
    }
    # Predictor concordance: how many missense predictors call damaging.
    votes = []
    for tool in ("REVEL", "AlphaMissense", "CADD_PHRED", "BayesDel_noAF"):
        cal = a["calibration"].get(tool)
        if cal and ann.get(tool) is not None:
            votes.append(calibrated_strength(ann[tool], cal["pp3"], True) is not None)
    ann["predictors_damaging"] = f"{sum(votes)}/{len(votes)}" if votes else ""
    return ann


def priority_tier(ann: Dict[str, object], acmg_class: str, cfg: dict) -> str:
    a = cfg["annotation"]
    terms = set(str(ann.get("consequence", "")).split("&"))
    clin = str(ann.get("clinvar", "")).lower()
    clin_plp = "pathogenic" in clin and "conflicting" not in clin and "benign" not in clin
    is_lof = bool(terms & set(a["lof_consequences"]))
    is_mis = bool(terms & set(a["missense_consequences"]))
    sai = ann.get("spliceai")
    canonical_splice = bool(terms & CANONICAL_SPLICE)
    gc = ann.get("_constraint") or {}
    known_dominant = bool(ann.get("disease")) and gc.get("dominant")
    if acmg_class in ("Pathogenic", "Likely_pathogenic") or clin_plp or ann.get("hotspot"):
        return "Tier1"
    if is_lof and (gc.get("lof_intolerant") or known_dominant):
        return "Tier1"
    if is_mis:
        tool = a["pp3_tool"]
        s = calibrated_strength(ann.get(tool), a["calibration"][tool]["pp3"], True)
        am = ann.get("AlphaMissense")
        damaging = s in ("Moderate", "Strong", "VeryStrong") or (am is not None and am >= 0.564)
        if damaging and (gc.get("missense_constrained") or gc.get("lof_intolerant") or known_dominant):
            return "Tier2"
    # Tier 3 is predicted splicing outside the canonical donor/acceptor sites;
    # canonical splice variants that miss Tier 1 fall through to Tier 4.
    if sai is not None and sai >= a["spliceai_pp3"] and not canonical_splice:
        return "Tier3"
    if is_lof or is_mis or terms & {"synonymous_variant", "stop_lost", "stop_retained_variant"} or terms & SPLICE_REGION:
        return "Tier4"
    return "Tier5"


def priority_score(ann: Dict[str, object], acmg_points: int, tier: str, dnm_tier: str, pheno: Optional[float]) -> float:
    """Heuristic ranking score (not a probability): ACMG points + DNM confidence
    + gene constraint + phenotype match + tier."""
    s = float(max(acmg_points, -10))
    s += {"HIGH": 2, "MEDIUM": 1}.get(dnm_tier, 0)
    s += 1.0 if ann.get("lof_intolerant") or ann.get("missense_constrained") else 0.0
    s += 4.0 * (pheno or 0.0)
    s += {"Tier1": 4, "Tier2": 3, "Tier3": 2, "Tier4": 1}.get(tier, 0)
    return round(s, 2)
