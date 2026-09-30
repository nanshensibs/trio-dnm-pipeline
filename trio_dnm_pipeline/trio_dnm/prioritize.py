"""`trio-dnm annotate`: annotate the VEP-annotated DNM VCF from `trio-dnm call`,
classify, prioritise, run sanity checks and signatures, and write reports."""
from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Optional

from . import annotation as annot
from . import sanity, signatures
from .dnm import norm_key, write_tsv
from .genome import Fasta
from .report import write_html
from .vcf import VCFReader, open_text

ANNOT_COLUMNS = [
    "priority_rank", "priority_score", "priority_tier", "acmg_class", "acmg_points", "acmg_codes",
    "proband", "chrom", "pos", "ref", "alt", "track", "dnm_tier", "dnm_posterior", "n_callers", "callers",
    "proband_vaf", "flags", "validated",
    "gene", "transcript", "mane", "consequence", "impact", "hgvsc", "hgvsp", "exon", "domains",
    "loftee", "loftee_flags", "nmd_escape", "gnomad_af", "clinvar", "clinvar_disease",
    "REVEL", "AlphaMissense", "AlphaMissense_class", "CADD_PHRED", "BayesDel_noAF", "PrimateAI", "ESM1b", "EVE",
    "MetaRNN", "predictors_damaging", "spliceai", "phyloP", "GERP", "cCRE", "utr_annotator",
    "pLI", "LOEUF", "mis_z", "shet", "hi_score", "lof_intolerant", "missense_constrained",
    "disease", "inheritance", "mechanism", "hotspot", "phenotype_score",
]


def load_validated(path: Optional[str]) -> set:
    out = set()
    if not path:
        return out
    with open_text(path) as fh:
        for line in fh:
            f = line.strip().split("\t")
            if not f or f[0].startswith("#"):
                continue
            if len(f) >= 4 and f[1].isdigit():
                out.add(norm_key(f[0], int(f[1]), f[2], f[3]))
            elif ":" in f[0]:
                c, p, r, a = f[0].split(":")
                out.add(norm_key(c, int(p), r, a))
    return out


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def run_annotate(
    vcf: str,
    cfg: dict,
    out_prefix: str,
    gene_table: Optional[str] = None,
    phenotype_scores: Optional[str] = None,
    validated_path: Optional[str] = None,
    parentage_confirmed: bool = False,
    fasta_path: Optional[str] = None,
    poo_path: Optional[str] = None,
    paternal_age: Optional[float] = None,
    maternal_age: Optional[float] = None,
    signatures_path: Optional[str] = None,
    signature_subset: Optional[List[str]] = None,
    call_summary: Optional[str] = None,
) -> Dict[str, object]:
    genes = annot.load_gene_table(gene_table)
    pheno = annot.load_scores(phenotype_scores)
    validated = load_validated(validated_path)
    fasta = Fasta(fasta_path) if fasta_path else None
    reader = VCFReader(vcf)
    if not reader.csq_fields:
        print("warning: no CSQ header found – run VEP first; only DNM-level fields will be reported")
    rows: List[Dict[str, object]] = []
    for rec in reader:
        track = str(rec.info.get("DNM_TRACK", "germline"))
        key = norm_key(rec.chrom, rec.pos, rec.ref, rec.alt)
        is_val = key in validated
        ann = annot.annotate_record(rec, reader.csq_fields, genes, cfg)
        codes, points, cls = annot.acmg(ann, track, cfg, parentage_confirmed, is_val)
        tier = annot.priority_tier(ann, cls, cfg)
        ps = pheno.get(str(ann.get("gene")))
        dnm_tier = str(rec.info.get("DNM_TIER", ""))
        row = {k: fmt(v) for k, v in ann.items() if not k.startswith("_")}
        row.update({
            "proband": rec.info.get("DNM_PROBAND", ""), "chrom": rec.chrom, "pos": rec.pos, "ref": rec.ref, "alt": rec.alt,
            "track": track, "dnm_tier": dnm_tier, "dnm_posterior": rec.info.get("DNM_POSTERIOR", ""),
            "n_callers": rec.info.get("N_CALLERS", ""), "callers": rec.info.get("DNM_CALLERS", "") or "",
            "proband_vaf": rec.info.get("PROBAND_VAF", ""), "flags": rec.info.get("DNM_FLAGS", "") or "",
            "validated": is_val, "acmg_codes": ",".join(codes), "acmg_points": points, "acmg_class": cls,
            "priority_tier": tier, "phenotype_score": fmt(ps),
            "priority_score": annot.priority_score(ann, points, tier, dnm_tier, ps),
        })
        rows.append(row)
    rows.sort(key=lambda r: -float(r["priority_score"]))
    for i, r in enumerate(rows, 1):
        r["priority_rank"] = i
    write_tsv(out_prefix + ".annotated.tsv", rows, ANNOT_COLUMNS)
    shortlist = [r for r in rows if r["priority_tier"] in ("Tier1", "Tier2", "Tier3")]
    write_tsv(out_prefix + ".shortlist.tsv", shortlist, ANNOT_COLUMNS)

    father = mother = ""
    summary_in: Dict[str, object] = {}
    if call_summary:
        with open(call_summary) as fh:
            summary_in = json.load(fh)
        father, mother = summary_in.get("father", ""), summary_in.get("mother", "")
    poo = sanity.load_parent_of_origin(poo_path, father, mother)
    san = sanity.evaluate(rows, cfg, fasta, poo, paternal_age, maternal_age)

    sig_result = None
    if fasta is not None:
        snvs = [(r["chrom"], int(r["pos"]), r["ref"], r["alt"]) for r in rows
                if r["track"] == "germline" and len(r["ref"]) == 1 and len(r["alt"]) == 1]
        counts = signatures.spectrum(snvs, fasta)
        signatures.write_spectrum(out_prefix + ".sbs96.tsv", counts)
        if signatures_path:
            names, mats = signatures.load_signatures(signatures_path, signature_subset or signatures.GERMLINE_SIGNATURES)
            sig_result = signatures.refit(counts, names, mats)
        san["sbs96"] = counts
    if sig_result:
        san["signatures"] = sig_result

    result = {
        "call_summary": summary_in,
        "sanity": san,
        "n_annotated": len(rows),
        "by_priority_tier": dict(Counter(r["priority_tier"] for r in rows)),
        "by_acmg_class": dict(Counter(r["acmg_class"] for r in rows)),
    }
    with open(out_prefix + ".qc.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    write_html(out_prefix + ".report.html", result, rows)
    return result
