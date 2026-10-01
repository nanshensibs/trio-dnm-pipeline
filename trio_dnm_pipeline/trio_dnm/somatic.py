"""Stage 9 (protocol §11): somatic mutation analysis; clonal-haematopoiesis screen §11.7.

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
import math
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

from . import annotation as annot
from . import genome, stats
from .dnm import get_pop_af, load_caller_sites, norm_key, write_tsv
from .vcf import Genotype, VCFReader, open_text, to_float, to_int


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
            tc = _finite(to_float(row.get("total_cn") or row.get("tcn")))
            segs[c].append({
                "start": int(float(row["start"])), "end": int(float(row["end"])),
                # total_cn 0 is a homozygous deletion; default to diploid only when missing.
                "total_cn": 2.0 if tc is None else tc,
                "minor_cn": _finite(to_float(row.get("minor_cn") or row.get("lcn"))),
            })
    return segs


def _finite(x: Optional[float]) -> Optional[float]:
    """'nan'/'inf' (as written by pandas/R for missing values) count as missing."""
    return x if x is not None and math.isfinite(x) else None


def segment_at(segs: Dict[str, List[dict]], chrom: str, pos: int) -> Optional[dict]:
    for s in segs.get(genome.bare_chrom(chrom), []):
        if s["start"] <= pos <= s["end"]:
            return s
    return None


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def estimate_multiplicity(vaf: float, purity: float, total_cn: float = 2.0, minor_cn: Optional[float] = None) -> int:
    """Mutant copies per tumour cell: m = round(VAF * (purity*CN_t + 2*(1-purity)) / purity),
    clamped to [1, major_cn] with major_cn = CN_t - minor_cn (CN_t if minor is unknown)."""
    if purity <= 0:
        return 1
    major = max(1, _round_half_up(total_cn if minor_cn is None else total_cn - minor_cn))
    m = _round_half_up(vaf * (purity * total_cn + 2 * (1 - purity)) / purity)
    return min(max(m, 1), major)


def cancer_cell_fraction(vaf: float, purity: float, total_cn: float = 2.0, multiplicity: float = 1.0) -> float:
    """CCF = VAF * (purity*CN_t + 2*(1-purity)) / (purity * multiplicity)."""
    if purity <= 0 or multiplicity <= 0:
        return 0.0
    return vaf * (purity * total_cn + 2 * (1 - purity)) / (purity * multiplicity)


def read_depth(g: Genotype) -> int:
    """Denominator for VAF, Wilson CI and CCF: ref+alt reads from AD (the reads
    the VAF is computed from); FORMAT DP only when AD is absent."""
    return (g.ref_depth + g.alt_depth) or g.dp


def require_samples(reader: VCFReader, ids: List[str], what: str) -> None:
    missing = [s for s in ids if s not in reader.samples]
    if missing:
        raise SystemExit(f"{what}: samples {missing} not in VCF header of {reader.path} (have {reader.samples})")


def load_germline_hits(path: Optional[str]) -> Dict[str, List[dict]]:
    """Annotated germline TSV from `trio-dnm annotate` -> gene -> [{chrom, pos, desc}]
    for Tier1/2 or P/LP germline (de novo, or inherited if such a table is
    supplied) variants."""
    out: Dict[str, List[dict]] = defaultdict(list)
    if not path:
        return out
    with open_text(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            gene = row.get("gene", "")
            if not gene:
                continue
            if row.get("priority_tier") in ("Tier1", "Tier2") or row.get("acmg_class") in ("Pathogenic", "Likely_pathogenic"):
                desc = f"{row.get('chrom')}:{row.get('pos')}{row.get('ref')}>{row.get('alt')}({row.get('track', 'germline')})"
                if all(h["desc"] != desc for h in out[gene]):
                    out[gene].append({"chrom": row.get("chrom", ""), "pos": to_int(row.get("pos")), "desc": desc})
    return out


SOMATIC_COLUMNS = [
    "chrom", "pos", "ref", "alt", "type", "status", "reasons", "n_callers", "callers",
    "tumor_dp", "tumor_alt", "tumor_vaf", "normal_dp", "normal_alt", "normal_vaf",
    "father_alt", "mother_alt", "pop_af", "gene", "consequence", "hgvsp", "impact",
    "REVEL", "AlphaMissense", "CADD_PHRED", "spliceai", "cancer_role", "hotspot",
    "ch_gene", "ffpe_suspect", "total_cn", "minor_cn", "ccf", "clonality",
    "second_hit", "context", "ccf_raw", "multiplicity", "loh", "notes",
]
TWO_HIT_COLUMNS = ["gene", "germline_variant", "hit_type", "somatic_evidence"]


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
    if purity is not None and not 0 < purity <= 1:
        raise SystemExit(f"--purity must be in (0, 1], got {purity}")
    if trio_vcf and not (father and mother):
        raise SystemExit("--trio-vcf requires --father and --mother")
    reader = VCFReader(vcf)
    require_samples(reader, [tumor, normal], "--tumor/--normal")
    genes = annot.load_gene_table(gene_table)
    segs = load_segments(segments)
    ghits = load_germline_hits(germline_hits)
    # LOH (minor_cn == 0) over each germline hit: germline + LOH is a second hit
    # even when the gene carries no somatic SNV/indel.
    for hits in ghits.values():
        for h in hits:
            gs = segment_at(segs, h["chrom"], h["pos"]) if h["pos"] is not None else None
            h["loh_seg"] = gs if gs and gs["minor_cn"] == 0 else None
    caller_sites = {n: load_caller_sites(p) for n, p in (extra_callers or {}).items()}
    ch_genes: Set[str] = set(cfg["ch"]["genes"])

    parent_alt: Dict[str, tuple] = {}
    if trio_vcf:
        trio_reader = VCFReader(trio_vcf)
        require_samples(trio_reader, [father, mother], "--father/--mother (--trio-vcf)")
        for r in trio_reader:
            f, m = r.samples.get(father), r.samples.get(mother)
            if f is None or m is None:
                continue
            if f.alt_depth or m.alt_depth:
                parent_alt[norm_key(r.chrom, r.pos, r.ref, r.alt)] = (f.alt_depth, m.alt_depth)

    rows: List[dict] = []
    fasta = genome.Fasta(fasta_path) if fasta_path else None
    for rec in reader:
        t, n = rec.samples.get(tumor), rec.samples.get(normal)
        if t is None or n is None:
            continue
        k = norm_key(rec.chrom, rec.pos, rec.ref, rec.alt)
        reasons: List[str] = []
        notes: List[str] = []
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
        ann = annot.annotate_record(rec, reader.csq_fields, genes, cfg) if reader.csq_fields else {}
        gene = str(ann.get("gene", ""))
        af = get_pop_af(rec, cfg["population"]["af_keys"])
        if af is None:
            # VEP-annotated Mutect2 output carries gnomAD AF only inside CSQ.
            af = ann.get("gnomad_af")
        if af is not None and af > sc["max_pop_af"]:
            reasons.append("population_AF")
        callers = ["primary"] + sorted(c for c, s in caller_sites.items() if k in s)
        if caller_sites and len(callers) < sc["min_callers"]:
            reasons.append("single_caller")

        ctx = ""
        if fasta is not None and rec.is_snv:
            ctx = fasta.fetch(rec.chrom, rec.pos - 1, rec.pos + 1)
        # FFPE deamination (C>T / G>A at low VAF) is flagged, not filtered.
        ffpe_suspect = bool(ffpe and rec.is_snv and (rec.ref.upper(), rec.alt.upper()) in (("C", "T"), ("G", "A"))
                            and tv <= sc["ffpe_max_vaf"])
        if ffpe_suspect:
            notes.append("FFPE_suspect")

        seg = segment_at(segs, rec.chrom, rec.pos)
        total_cn = seg["total_cn"] if seg else 2.0
        minor = seg["minor_cn"] if seg else None
        ccf = ccf_raw = mult = clonality = ""
        if purity:
            m = estimate_multiplicity(tv, purity, total_cn, minor)
            _, hi = stats.wilson_interval(t.alt_depth, read_depth(t))
            ccf_v = cancer_cell_fraction(tv, purity, total_cn, m)
            ccf_hi = cancer_cell_fraction(hi, purity, total_cn, m)
            ccf, ccf_raw, mult = f"{min(ccf_v, 1.0):.2f}", f"{ccf_v:.2f}", m
            clonality = "clonal" if ccf_hi >= sc["clonal_ccf"] else "subclonal"

        status = "PASS" if not reasons else "FILTERED"
        if mode == "tissue" and status == "PASS":
            status = "SOMATIC_MOSAIC"
        # Second hit = germline Tier1/2 or P/LP variant in this gene plus this
        # somatic variant (and LOH over the germline variant, if any). LOH at a
        # gene without a germline hit is not a second hit: see the loh column.
        second = []
        if gene and gene in ghits and status != "FILTERED":
            second.append("germline:" + ";".join(h["desc"] for h in ghits[gene]))
            if any(h["loh_seg"] for h in ghits[gene]):
                second.append("LOH")
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
            "ccf": ccf, "clonality": clonality, "second_hit": " | ".join(second),
            "context": ctx, "ccf_raw": ccf_raw, "multiplicity": mult,
            "loh": "" if minor is None else minor == 0, "notes": ";".join(notes),
        })

    write_tsv(out_prefix + ".somatic.tsv", rows, SOMATIC_COLUMNS)
    passing = [r for r in rows if r["status"] in ("PASS", "SOMATIC_MOSAIC")]
    coding = [r for r in passing if str(r["impact"]) in ("HIGH", "MODERATE")]

    # Two-hit table: one row per germline hit with a somatic SNV/indel in the
    # same gene and/or LOH over the germline variant.
    somatic_in_gene: Dict[str, List[str]] = defaultdict(list)
    for r in passing:
        if r["gene"] in ghits:
            somatic_in_gene[r["gene"]].append(" ".join(x for x in (
                f"{r['chrom']}:{r['pos']}{r['ref']}>{r['alt']}", str(r["hgvsp"]), f"VAF={r['tumor_vaf']}") if x))
    two_hits: List[dict] = []
    loh_only: List[str] = []
    for gene in sorted(ghits):
        for h in ghits[gene]:
            ev = list(somatic_in_gene.get(gene, []))
            gs = h["loh_seg"]
            if gs:
                ev.append(f"LOH {h['chrom']}:{gs['start']}-{gs['end']} minor_cn=0 total_cn={gs['total_cn']:g}")
            if not ev:
                continue
            kind = "+".join(x for x, on in (("somatic_variant", gene in somatic_in_gene), ("LOH", gs)) if on)
            two_hits.append({"gene": gene, "germline_variant": h["desc"], "hit_type": kind, "somatic_evidence": "; ".join(ev)})
        loh_descs = [h["desc"] for h in ghits[gene] if h["loh_seg"]]
        if loh_descs and gene not in somatic_in_gene:
            loh_only.append(f"{gene}: germline:{';'.join(loh_descs)} | LOH")
    write_tsv(out_prefix + ".two_hit.tsv", two_hits, TWO_HIT_COLUMNS)

    summary: dict = {
        "tumor": tumor, "normal": normal, "mode": mode,
        "n_candidates": len(rows), "n_pass": len(passing),
        "filter_reasons": dict(Counter(x for r in rows for x in str(r["reasons"]).split(";") if x)),
        "notes": dict(Counter(x for r in rows for x in str(r["notes"]).split(";") if x)),
        "tmb_nonsynonymous_per_mb": round(len(coding) / sc["coding_mb"], 2) if sc["coding_mb"] else None,
        "drivers": [f"{r['gene']} {r['hgvsp']}" for r in passing if r["hotspot"] or (r["cancer_role"] and r["impact"] in ("HIGH", "MODERATE"))],
        "second_hits": [f"{r['gene']}: {r['second_hit']}" for r in passing if r["second_hit"]] + loh_only,
        "two_hit_genes": sorted({h["gene"] for h in two_hits}),
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
            n_snv = sum(counts.values())
            if not signature_subset and n_snv < sc["min_snv_full_refit"]:
                # A full-catalogue NNLS on a handful of SNVs over-fits (protocol 11.4).
                summary["signatures"] = {"skipped": f"{n_snv} SNVs < {sc['min_snv_full_refit']}: pass --signature-subset"}
            else:
                names, mats = sig.load_signatures(signatures_path, signature_subset)
                summary["signatures"] = sig.refit(counts, names, mats)
    with open(out_prefix + ".somatic_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# --------------------------------------------------------------------------- #
# Clonal haematopoiesis screen
# --------------------------------------------------------------------------- #
CH_COLUMNS = ["sample", "chrom", "pos", "ref", "alt", "gene", "consequence", "hgvsp", "dp", "alt_reads", "vaf", "vaf_ci", "call"]


def run_ch_screen(vcf: str, samples: List[str], cfg: dict, out_prefix: str, gene_info_key: str = "GENE") -> dict:
    """Screen blood-derived samples for CH-gene variants at 2–35% VAF.

    Input: a VEP-annotated low-VAF call set (e.g. Mutect2 tumour-only with a
    panel of normals, or DeepSomatic tumour-only) containing the samples.
    A CH hit in a proband gene with an apparently mosaic DNM, or in a parent at
    the DNM site, should be treated as CH rather than constitutional.

    Without a CSQ/ANN header the gene is read from INFO/``gene_info_key`` and
    the HIGH/MODERATE impact filter cannot be applied (a warning is printed).
    """
    ch = cfg["ch"]
    genes = set(ch["genes"])
    reader = VCFReader(vcf)
    require_samples(reader, samples, "ch-screen --samples")
    warnings: List[str] = []
    if not reader.csq_fields:
        warnings.append(f"{vcf} has no CSQ/ANN header: CH gene taken from INFO/{gene_info_key}; impact could not be "
                        "assessed, so the HIGH/MODERATE filter was skipped (run VEP for consequence-aware CH calls)")
        print("warning: " + warnings[-1], file=sys.stderr)
    rows = []
    for rec in reader:
        if reader.csq_fields:
            ann = annot.annotate_record(rec, reader.csq_fields, {}, cfg)
            gene = str(ann.get("gene") or "")
            if str(ann.get("impact", "")) not in ("HIGH", "MODERATE"):
                continue
        else:
            ann = {"consequence": "unannotated"}
            names = [x for x in re.split(r"[,&|]", str(rec.info.get(gene_info_key) or "")) if x]
            gene = next((x for x in names if x in genes), "")
        if gene not in genes:
            continue
        for s in samples:
            g = rec.samples.get(s)
            if g is None:
                continue
            v = g.vaf or 0.0
            if g.alt_depth < ch["min_alt"] or not ch["min_vaf"] <= v <= ch["max_vaf"]:
                continue
            lo, hi = stats.wilson_interval(g.alt_depth, read_depth(g))
            rows.append({
                "sample": s, "chrom": rec.chrom, "pos": rec.pos, "ref": rec.ref, "alt": rec.alt,
                "gene": gene, "consequence": ann.get("consequence", ""), "hgvsp": ann.get("hgvsp", ""),
                "dp": g.dp, "alt_reads": g.alt_depth, "vaf": f"{v:.3f}", "vaf_ci": f"{lo:.3f}-{hi:.3f}",
                "call": "CHIP",
            })
    write_tsv(out_prefix + ".ch_screen.tsv", rows, CH_COLUMNS)
    summary = {"samples": samples, "n_hits": len(rows), "by_sample": dict(Counter(r["sample"] for r in rows)),
               "genes": dict(Counter(r["gene"] for r in rows)), "impact_filter": bool(reader.csq_fields)}
    if warnings:
        summary["warnings"] = warnings
    with open(out_prefix + ".ch_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
