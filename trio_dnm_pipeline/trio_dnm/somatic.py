"""Stage 10: somatic mutation analysis.

Three use cases share this module:

* **Tumour–normal** (e.g. a proband with a paediatric tumour): consensus of
  Mutect2 / Strelka2 / DeepSomatic calls, trio-aware germline subtraction,
  cancer-cell-fraction and clonality, TMB, SBS-96 signatures, driver/hotspot
  annotation and germline-DNM + somatic "second-hit" detection.
* **Affected tissue vs blood** (brain, skin, vascular lesions): the same
  paired logic, with tissue-restricted variants reported as somatic mosaics.
* **Clonal haematopoiesis (CH) screen** of blood DNA from every trio member,
  so that CH clones are not mistaken for mosaic DNMs or parental leakage.

The trio is exploited as a "super-normal": an ALT allele seen in either
parent is inherited, not somatic, even if the matched normal is shallow.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

from . import annotation as annot
from . import genome, stats
from .dnm import get_pop_af, load_caller_sites, norm_key, write_tsv
from .vcf import VCFReader, open_text, to_float


def load_segments(path: Optional[str]) -> Dict[str, List[dict]]:
    """Allele-specific copy-number segments (FACETS / PURPLE / ASCAT-like TSV):
    columns chrom, start, end, total_cn, minor_cn."""
    segs: Dict[str, List[dict]] = defaultdict(list)
    if not path:
        return segs
    with open_text(path) as fh:
        header = fh.readline().rstrip("\n").lstrip("#").split("\t")
        for line in fh:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            c = genome.bare_chrom(row.get("chrom") or row.get("chromosome", ""))
            segs[c].append({
                "start": int(float(row["start"])), "end": int(float(row["end"])),
                "total_cn": to_float(row.get("total_cn") or row.get("tcn")) or 2.0,
                "minor_cn": to_float(row.get("minor_cn") or row.get("lcn")),
            })
    return segs


def segment_at(segs: Dict[str, List[dict]], chrom: str, pos: int) -> Optional[dict]:
    for s in segs.get(genome.bare_chrom(chrom), []):
        if s["start"] <= pos <= s["end"]:
            return s
    return None


def cancer_cell_fraction(vaf: float, purity: float, total_cn: float = 2.0, multiplicity: float = 1.0) -> float:
    """CCF = VAF * (purity*CN_t + 2*(1-purity)) / (purity * multiplicity)."""
    if purity <= 0:
        return 0.0
    return vaf * (purity * total_cn + 2 * (1 - purity)) / (purity * multiplicity)


def load_germline_hits(path: Optional[str]) -> Dict[str, List[str]]:
    """Annotated germline TSV from `trio-dnm annotate` -> gene -> [descriptions]
    for Tier1/2 or P/LP germline (de novo or inherited) variants."""
    out: Dict[str, List[str]] = defaultdict(list)
    if not path:
        return out
    with open_text(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            if row.get("priority_tier") in ("Tier1", "Tier2") or row.get("acmg_class") in ("Pathogenic", "Likely_pathogenic"):
                out[row.get("gene", "")].append(f"{row.get('chrom')}:{row.get('pos')}{row.get('ref')}>{row.get('alt')}({row.get('track', 'germline')})")
    return out


SOMATIC_COLUMNS = [
    "chrom", "pos", "ref", "alt", "type", "status", "reasons", "n_callers", "callers",
    "tumor_dp", "tumor_alt", "tumor_vaf", "normal_dp", "normal_alt", "normal_vaf",
    "father_alt", "mother_alt", "pop_af", "gene", "consequence", "hgvsp", "impact",
    "REVEL", "AlphaMissense", "CADD_PHRED", "spliceai", "cancer_role", "hotspot",
    "ch_gene", "ffpe_suspect", "total_cn", "minor_cn", "ccf", "clonality",
    "second_hit", "context",
]


def run_somatic(
    vcf: str,
    tumor: str,
    normal: str,
    cfg: dict,
    out_prefix: str,
    trio_vcf: Optional[str] = None,
    father: Optional[str] = None,
    mother: Optional[str] = None,
    extra_callers: Optional[Dict[str, str]] = None,
    purity: Optional[float] = None,
    segments: Optional[str] = None,
    gene_table: Optional[str] = None,
    germline_hits: Optional[str] = None,
    fasta_path: Optional[str] = None,
    signatures_path: Optional[str] = None,
    signature_subset: Optional[List[str]] = None,
    ffpe: bool = False,
    mode: str = "tumor",
) -> dict:
    sc = cfg["somatic"]
    genes = annot.load_gene_table(gene_table)
    segs = load_segments(segments)
    ghits = load_germline_hits(germline_hits)
    caller_sites = {n: load_caller_sites(p) for n, p in (extra_callers or {}).items()}
    ch_genes: Set[str] = set(cfg["ch"]["genes"])

    parent_alt: Dict[str, tuple] = {}
    if trio_vcf and father and mother:
        for r in VCFReader(trio_vcf):
            f, m = r.samples.get(father), r.samples.get(mother)
            if f is None or m is None:
                continue
            if f.alt_depth or m.alt_depth:
                parent_alt[norm_key(r.chrom, r.pos, r.ref, r.alt)] = (f.alt_depth, m.alt_depth)

    reader = VCFReader(vcf)
    rows: List[dict] = []
    fasta = genome.Fasta(fasta_path) if fasta_path else None
    for rec in reader:
        t, n = rec.samples.get(tumor), rec.samples.get(normal)
        if t is None or n is None:
            continue
        k = norm_key(rec.chrom, rec.pos, rec.ref, rec.alt)
        reasons: List[str] = []
        if rec.filter not in ("PASS", "."):
            reasons.append(f"caller_filter:{rec.filter}")
        tv, nv = t.vaf or 0.0, n.vaf or 0.0
        if t.dp < sc["tumor_min_dp"]:
            reasons.append("tumor_DP")
        if t.alt_depth < sc["tumor_min_alt"]:
            reasons.append("tumor_alt")
        if tv < sc["tumor_min_vaf"]:
            reasons.append("tumor_VAF")
        if n.dp < sc["normal_min_dp"]:
            reasons.append("normal_DP")
        if n.alt_depth > sc["normal_max_alt"] or nv > sc["normal_max_vaf"]:
            reasons.append("ALT_in_normal")
        pa = parent_alt.get(k)
        if pa and (pa[0] >= 2 or pa[1] >= 2):
            reasons.append("ALT_in_parent(inherited)")
        af = get_pop_af(rec, cfg["population"]["af_keys"])
        if af is not None and af > sc["max_pop_af"]:
            reasons.append("population_AF")
        callers = ["primary"] + sorted(c for c, s in caller_sites.items() if k in s)
        if caller_sites and len(callers) < sc["min_callers"]:
            reasons.append("single_caller")

        ann = annot.annotate_record(rec, reader.csq_fields, genes, cfg) if reader.csq_fields else {}
        gene = str(ann.get("gene", ""))

        ctx = ""
        if fasta is not None and rec.is_snv:
            ctx = fasta.fetch(rec.chrom, rec.pos - 1, rec.pos + 1)
        ffpe_suspect = bool(ffpe and rec.is_snv and {rec.ref, rec.alt} in ({"C", "T"}, {"G", "A"}) and tv <= sc["ffpe_max_vaf"])
        if ffpe_suspect:
            reasons.append("FFPE_deamination_suspect")

        seg = segment_at(segs, rec.chrom, rec.pos)
        total_cn = seg["total_cn"] if seg else 2.0
        minor = seg["minor_cn"] if seg else None
        ccf = clonality = ""
        if purity:
            _, hi = stats.wilson_interval(t.alt_depth, t.dp)
            ccf_v = cancer_cell_fraction(tv, purity, total_cn)
            ccf_hi = cancer_cell_fraction(hi, purity, total_cn)
            ccf = f"{min(ccf_v, 1.5):.2f}"
            clonality = "clonal" if ccf_hi >= sc["clonal_ccf"] else "subclonal"

        second = []
        if gene and gene in ghits:
            second.append("germline:" + ";".join(ghits[gene]))
        if gene and minor == 0:
            second.append("LOH")
        status = "PASS" if not reasons else "FILTERED"
        if mode == "tissue" and status == "PASS":
            status = "SOMATIC_MOSAIC"
        rows.append({
            "chrom": rec.chrom, "pos": rec.pos, "ref": rec.ref, "alt": rec.alt,
            "type": "SNV" if rec.is_snv else ("MNV" if rec.is_mnv else "INDEL"),
            "status": status, "reasons": ";".join(reasons), "n_callers": len(callers), "callers": ",".join(callers),
            "tumor_dp": t.dp, "tumor_alt": t.alt_depth, "tumor_vaf": f"{tv:.3f}",
            "normal_dp": n.dp, "normal_alt": n.alt_depth, "normal_vaf": f"{nv:.3f}",
            "father_alt": pa[0] if pa else "", "mother_alt": pa[1] if pa else "",
            "pop_af": "" if af is None else f"{af:.2e}",
            "gene": gene, "consequence": ann.get("consequence", ""), "hgvsp": ann.get("hgvsp", ""),
            "impact": ann.get("impact", ""), "REVEL": ann.get("REVEL") or "", "AlphaMissense": ann.get("AlphaMissense") or "",
            "CADD_PHRED": ann.get("CADD_PHRED") or "", "spliceai": ann.get("spliceai") or "",
            "cancer_role": ann.get("cancer_role", ""), "hotspot": ann.get("hotspot", False),
            "ch_gene": gene in ch_genes, "ffpe_suspect": ffpe_suspect,
            "total_cn": total_cn, "minor_cn": "" if minor is None else minor,
            "ccf": ccf, "clonality": clonality, "second_hit": " | ".join(second) if second and status != "FILTERED" else "",
            "context": ctx,
        })

    write_tsv(out_prefix + ".somatic.tsv", rows, SOMATIC_COLUMNS)
    passing = [r for r in rows if r["status"] in ("PASS", "SOMATIC_MOSAIC")]
    coding = [r for r in passing if str(r["impact"]) in ("HIGH", "MODERATE")]
    summary: dict = {
        "tumor": tumor, "normal": normal, "mode": mode,
        "n_candidates": len(rows), "n_pass": len(passing),
        "filter_reasons": dict(Counter(x for r in rows for x in str(r["reasons"]).split(";") if x)),
        "tmb_nonsynonymous_per_mb": round(len(coding) / sc["coding_mb"], 2) if sc["coding_mb"] else None,
        "drivers": [f"{r['gene']} {r['hgvsp']}" for r in passing if r["hotspot"] or (r["cancer_role"] and r["impact"] in ("HIGH", "MODERATE"))],
        "second_hits": [f"{r['gene']}: {r['second_hit']}" for r in passing if r["second_hit"]],
        "ch_gene_variants": [f"{r['gene']} {r['hgvsp']} VAF={r['tumor_vaf']}" for r in passing if r["ch_gene"]],
    }
    if summary["tmb_nonsynonymous_per_mb"] is not None:
        summary["tmb_high"] = summary["tmb_nonsynonymous_per_mb"] >= sc["tmb_high"]
    if purity:
        summary["clonality"] = dict(Counter(r["clonality"] for r in passing))
    if fasta is not None:
        from . import signatures as sig
        counts = sig.spectrum(((r["chrom"], r["pos"], r["ref"], r["alt"]) for r in passing if r["type"] == "SNV"), fasta)
        sig.write_spectrum(out_prefix + ".somatic.sbs96.tsv", counts)
        if signatures_path:
            names, mats = sig.load_signatures(signatures_path, signature_subset)
            summary["signatures"] = sig.refit(counts, names, mats)
    with open(out_prefix + ".somatic_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# --------------------------------------------------------------------------- #
# Clonal haematopoiesis screen
# --------------------------------------------------------------------------- #
CH_COLUMNS = ["sample", "chrom", "pos", "ref", "alt", "gene", "consequence", "hgvsp", "dp", "alt_reads", "vaf", "vaf_ci", "call"]


def run_ch_screen(vcf: str, samples: List[str], cfg: dict, out_prefix: str, gene_field_fallback: Optional[str] = None) -> dict:
    """Screen blood-derived samples for CH-gene variants at 2–35% VAF.

    Input: a VEP-annotated low-VAF call set (e.g. Mutect2 tumour-only with a
    panel of normals, or DeepSomatic tumour-only) containing the samples.
    A CH hit in a proband gene with an apparently mosaic DNM, or in a parent at
    the DNM site, should be treated as CH rather than constitutional.
    """
    ch = cfg["ch"]
    genes = set(ch["genes"])
    reader = VCFReader(vcf)
    rows = []
    for rec in reader:
        ann = annot.annotate_record(rec, reader.csq_fields, {}, cfg) if reader.csq_fields else {}
        gene = str(ann.get("gene") or rec.info.get(gene_field_fallback or "GENE", "") or "")
        if gene not in genes:
            continue
        if str(ann.get("impact", "")) not in ("HIGH", "MODERATE"):
            continue
        for s in samples:
            g = rec.samples.get(s)
            if g is None:
                continue
            v = g.vaf or 0.0
            if g.alt_depth < ch["min_alt"] or not ch["min_vaf"] <= v <= ch["max_vaf"]:
                continue
            lo, hi = stats.wilson_interval(g.alt_depth, g.dp)
            rows.append({
                "sample": s, "chrom": rec.chrom, "pos": rec.pos, "ref": rec.ref, "alt": rec.alt,
                "gene": gene, "consequence": ann.get("consequence", ""), "hgvsp": ann.get("hgvsp", ""),
                "dp": g.dp, "alt_reads": g.alt_depth, "vaf": f"{v:.3f}", "vaf_ci": f"{lo:.3f}-{hi:.3f}",
                "call": "CHIP",
            })
    write_tsv(out_prefix + ".ch_screen.tsv", rows, CH_COLUMNS)
    summary = {"samples": samples, "n_hits": len(rows), "by_sample": dict(Counter(r["sample"] for r in rows)),
               "genes": dict(Counter(r["gene"] for r in rows))}
    with open(out_prefix + ".ch_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
