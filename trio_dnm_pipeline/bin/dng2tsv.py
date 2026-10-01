#!/usr/bin/env python3
"""Convert DeNovoGear `dng dnm` text output to chrom/pos/ref/alt/pp_dnm TSV.

DNG writes one line per candidate, e.g.
  DENOVO-SNP id: kid ref_name: chr1 coor: 12345 ref_base: C ALT: T ... pp_dnm: 0.998 tgt_dnm(child/mom/dad): CT/CC/CC ...
Field names have varied between releases, so several aliases are accepted.
"""
import argparse
import re
import sys

TOKEN = re.compile(r"(\S+?):\s+(\S+)")
ALIASES = {
    "chrom": ("ref_name", "chr", "chrom"),
    "pos": ("coor", "pos", "position"),
    "ref": ("ref_base", "ref"),
    "alt": ("ALT", "alt", "alt_base"),
    "pp": ("pp_dnm",),
}


def parse(line):
    kv = dict(TOKEN.findall(line))
    out = {}
    for k, names in ALIASES.items():
        out[k] = next((kv[n] for n in names if n in kv), None)
    # Prefer the child's DNM genotype (e.g. "CT/CC/CC") to pick the ALT allele
    # when DNG lists several candidate ALTs. Indel genotypes are not parsed this way.
    tgt = next((v for k, v in kv.items() if k.startswith("tgt_dnm")), None)
    if tgt and out["ref"] and len(out["ref"]) == 1:
        child = tgt.split("/")[0]
        alts = sorted(set(child) - {out["ref"]})
        if alts and all(len(a) == 1 for a in alts):
            out["alt"] = ",".join(alts)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dng_out")
    ap.add_argument("--min-pp", type=float, default=0.5)
    a = ap.parse_args(argv)
    print("chrom\tpos\tref\talt\tpp_dnm")
    with open(a.dng_out) as fh:
        for line in fh:
            if not line.startswith("DENOVO-"):
                continue
            r = parse(line)
            if None in (r["chrom"], r["pos"], r["ref"], r["alt"], r["pp"]):
                print(f"skipping unparsable line: {line[:80]}", file=sys.stderr)
                continue
            if float(r["pp"]) < a.min_pp:
                continue
            for alt in r["alt"].split(","):
                if alt != r["ref"]:
                    print(f"{r['chrom']}\t{r['pos']}\t{r['ref']}\t{alt}\t{r['pp']}")


if __name__ == "__main__":
    main()
