"""Stage 6: biological sanity checks on the final DNM set.

A passing call set must reproduce the aggregate biology of germline DNMs;
deviation indicates a methodological problem, not biology.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from . import stats
from .genome import Fasta, is_cpg_transition, is_transition


def load_parent_of_origin(path: Optional[str], father: str, mother: str) -> Dict[str, str]:
    """Read unfazed output (or a chrom/pos/origin TSV) -> {'chrom:pos': 'paternal'|'maternal'}.

    unfazed reports the origin as a parent sample ID, so ``father``/``mother``
    must be the trio's sample IDs; rows with an empty or unrecognised origin
    are skipped."""
    out: Dict[str, str] = {}
    if not path:
        return out
    with open(path) as fh:
        header = None
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 2:
                continue
            if line.startswith("#") or header is None and not f[1].isdigit():
                header = [x.lstrip("#") for x in f]
                continue
            row = dict(zip(header or ["chrom", "pos", "origin"], f))
            origin = (row.get("origin_parent") or row.get("origin") or "").strip()
            if not origin:
                continue
            low = origin.lower()
            if father and origin == father:
                o = "paternal"
            elif mother and origin == mother:
                o = "maternal"
            elif low.startswith("pat") or low == "father":
                o = "paternal"
            elif low.startswith("mat") or low == "mother":
                o = "maternal"
            else:
                continue
            chrom = row.get("chrom", f[0]).replace("chr", "")
            if "start" in row:  # unfazed: 0-based start
                pos = int(row["start"]) + 1
            else:
                pos = int(row.get("pos", f[1]))
            out[f"{chrom}:{pos}"] = o
    return out


def evaluate(
    variants: List[Dict[str, object]],
    cfg: dict,
    fasta: Optional[Fasta] = None,
    poo: Optional[Dict[str, str]] = None,
    paternal_age: Optional[float] = None,
    maternal_age: Optional[float] = None,
) -> Dict[str, object]:
    """``variants``: dicts with chrom, pos, ref, alt, track, proband_vaf."""
    exp = cfg["expectations"]
    dt = cfg["data_type"]
    germ = [v for v in variants if v.get("track") == "germline"]
    snvs = [v for v in germ if len(v["ref"]) == 1 and len(v["alt"]) == 1]
    indels = [v for v in germ if len(v["ref"]) != len(v["alt"])]
    res: Dict[str, object] = {"n_germline_snv": len(snvs), "n_germline_indel": len(indels),
                              "n_mosaic": sum(1 for v in variants if v.get("track") == "mosaic"),
                              "n_parental_mosaic": sum(1 for v in variants if v.get("track") == "parental_mosaic")}
    warnings: List[str] = []

    # Counts. Exome/panel trios carry ~0-3 coding DNMs, so low counts are
    # expected there and only excess is flagged (``flag_low``: false).
    e = exp.get(dt, exp["wgs"])
    lo, hi = e["snv"]
    flag_low = e.get("flag_low", True)
    n_snv, n_indel = len(snvs), len(indels)
    if n_snv > e["hard_high"] or (flag_low and n_snv < e["hard_low"]):
        warnings.append(f"SNV DNM count {n_snv} outside hard range [{e['hard_low']}, {e['hard_high']}] – revisit filtering")
    elif n_snv > hi or (flag_low and n_snv < lo):
        warnings.append(f"SNV DNM count {n_snv} outside expected [{lo}, {hi}]")
    if n_indel > e["indel"][1] or (flag_low and n_indel < e["indel"][0]):
        warnings.append(f"indel DNM count {n_indel} outside expected {e['indel']}")
    if paternal_age is not None and maternal_age is not None and dt == "wgs":
        am = exp["age_model"]
        expected = am["intercept"] + am["paternal"] * paternal_age + am["maternal"] * maternal_age
        res["age_expected_dnm"] = round(expected, 1)
        res["observed_over_expected"] = round((len(snvs) + len(indels)) / expected, 2) if expected else None

    # Ti/Tv and CpG.
    ti = sum(1 for v in snvs if is_transition(v["ref"], v["alt"]))
    tv = len(snvs) - ti
    res["titv"] = round(ti / tv, 2) if tv else None
    titv_range = exp["titv"].get(dt)
    if res["titv"] is not None and titv_range and len(snvs) >= 20 and not titv_range[0] <= res["titv"] <= titv_range[1]:
        warnings.append(f"Ti/Tv {res['titv']} outside {titv_range}")
    if fasta is not None and snvs:
        cpg = 0
        for v in snvs:
            ctx = fasta.fetch(str(v["chrom"]), int(v["pos"]) - 1, int(v["pos"]) + 1)
            if len(ctx) == 3 and is_cpg_transition(v["ref"], v["alt"], ctx[0], ctx[2]):
                cpg += 1
        res["cpg_transition_fraction"] = round(cpg / len(snvs), 3)
        r = exp["cpg_fraction_snv"]
        if len(snvs) >= 20 and not r[0] <= res["cpg_transition_fraction"] <= r[1]:
            warnings.append(f"CpG transition fraction {res['cpg_transition_fraction']} outside {r}")

    # Indel properties.
    lens = [len(v["alt"]) - len(v["ref"]) for v in indels]
    res["indel_length_hist"] = dict(sorted(Counter(lens).items()))
    ins, dels = sum(1 for x in lens if x > 0), sum(1 for x in lens if x < 0)
    res["ins_del_ratio"] = round(ins / dels, 2) if dels else None
    res["indel_1to4bp_fraction"] = round(sum(1 for x in lens if 1 <= abs(x) <= 4) / len(lens), 2) if lens else None

    # Allele balance.
    vafs = [float(v["proband_vaf"]) for v in germ if v.get("proband_vaf") not in (None, "")]
    res["median_proband_vaf"] = stats.median(vafs)
    vr = cfg["sanity"]["median_vaf_range"]
    if vafs and len(vafs) >= 10 and not vr[0] <= res["median_proband_vaf"] <= vr[1]:
        warnings.append(f"median germline DNM VAF {res['median_proband_vaf']:.2f} deviates from 0.5 (outside {vr})")

    # Parent of origin.
    if poo:
        origins = [poo.get(f"{str(v['chrom']).replace('chr', '')}:{v['pos']}") for v in snvs]
        pat, mat = origins.count("paternal"), origins.count("maternal")
        res["phased"] = pat + mat
        if pat + mat:
            frac = pat / (pat + mat)
            res["paternal_fraction"] = round(frac, 3)
            res["paternal_fraction_ci"] = [round(x, 3) for x in stats.wilson_interval(pat, pat + mat)]
            r = exp["paternal_fraction"]
            if pat + mat >= 10 and not r[0] <= frac <= r[1]:
                warnings.append(f"paternal fraction {frac:.2f} outside {r} – check parental contamination / swap")

    res["warnings"] = warnings
    res["status"] = "PASS" if not warnings else "REVIEW"
    return res
