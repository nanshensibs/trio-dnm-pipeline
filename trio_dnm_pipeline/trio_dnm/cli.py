"""Command-line interface: ``trio-dnm {qc-gate,call,annotate,somatic,ch-screen,signatures,demo,config}``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional

from . import __version__
from .config import load_config
from .genome import read_ped


def _kv(items: Optional[List[str]]) -> Dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected NAME=PATH, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def _cfg(a) -> dict:
    return load_config(a.config, getattr(a, "data_type", None), build=getattr(a, "build", None))


def cmd_call(a) -> int:
    from .dnm import run_call

    cfg = _cfg(a)
    trios = read_ped(a.ped)
    if a.proband:
        trios = [t for t in trios if t.proband == a.proband]
    if not trios:
        raise SystemExit("no complete trio found in pedigree")
    if a.mean_depth:
        cfg["mean_depth"] = a.mean_depth
    res = run_call(a.vcf, trios, cfg, a.out, a.exclude_bed, a.fasta, a.pon, a.recurrence, _kv(a.caller))
    for p, s in res.items():
        print(f"{p}: {s['passing_by_track']} tiers={s['passing_by_tier']}")
    return 0


def cmd_annotate(a) -> int:
    from .prioritize import run_annotate

    cfg = _cfg(a)
    res = run_annotate(
        a.vcf, cfg, a.out, a.gene_table, a.phenotype_scores, a.validated, a.parentage_confirmed,
        a.fasta, a.parent_of_origin, a.paternal_age, a.maternal_age, a.signatures,
        a.signature_subset.split(",") if a.signature_subset else None, a.call_summary,
    )
    print(json.dumps({k: res[k] for k in ("n_annotated", "by_priority_tier", "by_acmg_class")}, indent=2))
    print("sanity:", res["sanity"]["status"], *res["sanity"]["warnings"], sep="\n  ")
    return 0


def cmd_somatic(a) -> int:
    from .somatic import run_somatic

    cfg = _cfg(a)
    res = run_somatic(
        a.vcf, a.tumor, a.normal, cfg, a.out, a.trio_vcf, a.father, a.mother, _kv(a.caller), a.purity,
        a.segments, a.gene_table, a.germline_annotated, a.fasta, a.signatures,
        a.signature_subset.split(",") if a.signature_subset else None, a.ffpe, a.mode,
    )
    print(json.dumps(res, indent=2, default=str))
    return 0


def cmd_ch(a) -> int:
    from .somatic import run_ch_screen

    res = run_ch_screen(a.vcf, a.samples.split(","), _cfg(a), a.out)
    print(json.dumps(res, indent=2))
    return 0


def cmd_signatures(a) -> int:
    from . import signatures as sig
    from .genome import Fasta
    from .vcf import VCFReader

    fasta = Fasta(a.fasta)
    snvs = [(r.chrom, r.pos, r.ref, r.alt) for r in VCFReader(a.vcf) if r.is_snv and r.filter in ("PASS", ".")]
    counts = sig.spectrum(snvs, fasta)
    sig.write_spectrum(a.out + ".sbs96.tsv", counts)
    if a.signatures:
        names, mats = sig.load_signatures(a.signatures, a.signature_subset.split(",") if a.signature_subset else None)
        print(json.dumps(sig.refit(counts, names, mats), indent=2))
    return 0


def cmd_qc_gate(a) -> int:
    from . import qcgate

    trios = [t for t in read_ped(a.ped) if not a.proband or t.proband == a.proband]
    if not trios:
        raise SystemExit("no complete trio found in pedigree")
    t = trios[0]
    reported = {t.proband: t.proband_sex}
    res = qcgate.evaluate(t, a.somalier_pairs, a.somalier_samples, _kv(a.verifybamid), _kv(a.mosdepth),
                          a.data_type or "wgs", reported, a.consanguineous)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    return 1 if res["status"] == "FAIL" and not a.no_halt else 0


def cmd_demo(a) -> int:
    from .demo import run_demo

    res = run_demo(a.out)
    print(f"\nDemo finished. Simulated inputs and all outputs are in {res['outdir']}")
    print("Open results/F1.report.html in a browser; see README.txt for a guide to each file.")
    return 0


def cmd_config(a) -> int:
    print(json.dumps(_cfg(a), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trio-dnm", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, data_type=True):
        sp.add_argument("--config", help="JSON file deep-merged over defaults")
        sp.add_argument("--build", choices=["GRCh38", "GRCh37"])
        if data_type:
            sp.add_argument("--data-type", choices=["wgs", "wes", "panel"])

    c = sub.add_parser("call", help="filter cascade, posterior, mosaic tracks, consensus tiers")
    common(c)
    c.add_argument("--vcf", required=True, help="joint-genotyped, normalised trio VCF (after CGP/PossibleDeNovo)")
    c.add_argument("--ped", required=True)
    c.add_argument("--proband")
    c.add_argument("--out", required=True, help="output prefix")
    c.add_argument("--fasta", help="indexed reference FASTA (homopolymer, N and context checks)")
    c.add_argument("--exclude-bed", action="append", default=[], help="hard-mask BED (repeatable)")
    c.add_argument("--pon", help="panel-of-normals VCF")
    c.add_argument("--recurrence", help="TSV key<TAB>n_trios of project-wide DNM recurrence")
    c.add_argument("--caller", action="append", help="NAME=VCF of an external DNM caller (repeatable)")
    c.add_argument("--mean-depth", type=float)
    c.set_defaults(func=cmd_call)

    a = sub.add_parser("annotate", help="annotation, ACMG, prioritisation, sanity checks, report")
    common(a)
    a.add_argument("--vcf", required=True, help="VEP-annotated *.dnm.vcf from `call`")
    a.add_argument("--out", required=True)
    a.add_argument("--gene-table")
    a.add_argument("--phenotype-scores", help="gene<TAB>score (0-1) from Exomiser/LIRICAL")
    a.add_argument("--validated", help="orthogonally validated variants (chrom pos ref alt)")
    a.add_argument("--parentage-confirmed", action="store_true", help="Stage-1 kinship QC passed")
    a.add_argument("--fasta")
    a.add_argument("--parent-of-origin", help="unfazed output or chrom/pos/origin TSV")
    a.add_argument("--paternal-age", type=float)
    a.add_argument("--maternal-age", type=float)
    a.add_argument("--signatures", help="COSMIC-format SBS matrix")
    a.add_argument("--signature-subset")
    a.add_argument("--call-summary", help="*.call_summary.json from `call`")
    a.set_defaults(func=cmd_annotate)

    s = sub.add_parser("somatic", help="tumour/tissue vs normal somatic analysis")
    common(s)
    s.add_argument("--vcf", required=True, help="primary somatic VCF (e.g. Mutect2 filtered, VEP-annotated)")
    s.add_argument("--tumor", required=True)
    s.add_argument("--normal", required=True)
    s.add_argument("--mode", choices=["tumor", "tissue"], default="tumor")
    s.add_argument("--out", required=True)
    s.add_argument("--caller", action="append", help="NAME=VCF of additional somatic callers")
    s.add_argument("--trio-vcf")
    s.add_argument("--father")
    s.add_argument("--mother")
    s.add_argument("--purity", type=float)
    s.add_argument("--segments", help="allele-specific CN segments TSV")
    s.add_argument("--gene-table")
    s.add_argument("--germline-annotated", help="*.annotated.tsv from `annotate` for two-hit analysis")
    s.add_argument("--fasta")
    s.add_argument("--signatures")
    s.add_argument("--signature-subset")
    s.add_argument("--ffpe", action="store_true")
    s.set_defaults(func=cmd_somatic)

    h = sub.add_parser("ch-screen", help="clonal haematopoiesis screen of blood samples")
    common(h)
    h.add_argument("--vcf", required=True)
    h.add_argument("--samples", required=True, help="comma-separated sample IDs")
    h.add_argument("--out", required=True)
    h.set_defaults(func=cmd_ch)

    g = sub.add_parser("signatures", help="SBS-96 spectrum and signature refit for any VCF")
    g.add_argument("--vcf", required=True)
    g.add_argument("--fasta", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--signatures")
    g.add_argument("--signature-subset")
    g.set_defaults(func=cmd_signatures)

    q = sub.add_parser("qc-gate", help="Stage-1 stop-gates (exit 1 on failure)")
    q.add_argument("--ped", required=True)
    q.add_argument("--proband")
    q.add_argument("--somalier-pairs")
    q.add_argument("--somalier-samples")
    q.add_argument("--verifybamid", action="append", help="SAMPLE=selfSM (repeatable)")
    q.add_argument("--mosdepth", action="append", help="SAMPLE=mosdepth.summary.txt (repeatable)")
    q.add_argument("--data-type", choices=["wgs", "wes", "panel"])
    q.add_argument("--consanguineous", action="store_true")
    q.add_argument("--no-halt", action="store_true")
    q.add_argument("--out", required=True)
    q.set_defaults(func=cmd_qc_gate)

    m = sub.add_parser("demo", help="run every stage on a small SIMULATED trio")
    m.add_argument("--out", default="trio_dnm_demo", help="output directory")
    m.set_defaults(func=cmd_demo)

    k = sub.add_parser("config", help="print the effective configuration")
    common(k)
    k.set_defaults(func=cmd_config)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
