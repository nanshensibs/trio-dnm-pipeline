#!/usr/bin/env python3
"""Mutect2 (tumour-only) VCF -> MosaicForecast input BED:
chrom, start (0-based), end, ref, alt, sample. Keeps PASS biallelic sites with
the sample's FORMAT/AF inside [min_vaf, max_vaf]. Python 3.6+, stdlib only."""
import argparse
import gzip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vcf")
    ap.add_argument("sample")
    ap.add_argument("--min-vaf", type=float, default=0.03)
    ap.add_argument("--max-vaf", type=float, default=0.40)
    a = ap.parse_args()
    op = gzip.open if a.vcf.endswith(".gz") else open
    col = None
    with op(a.vcf, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            f = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                col = f.index(a.sample)
                continue
            if f[6] not in ("PASS", ".") or "," in f[4]:
                continue
            fmt = dict(zip(f[8].split(":"), f[col].split(":")))
            try:
                af = float(fmt.get("AF", "nan").split(",")[0])
            except ValueError:
                continue
            if a.min_vaf <= af <= a.max_vaf:
                pos = int(f[1])
                print("\t".join([f[0], str(pos - 1), str(pos), f[3], f[4], a.sample]))


if __name__ == "__main__":
    main()
