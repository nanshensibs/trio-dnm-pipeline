#!/usr/bin/env python3
"""Mutect2 (tumour-only) VCF -> MosaicForecast input BED:
chrom, start (0-based), end, ref, alt, sample. Keeps PASS biallelic SNVs with
the sample's FORMAT/AF inside [min_vaf, max_vaf]. Indels are skipped (and
counted on stderr): the MosaicForecast Refine models are SNV-trained, and its
deletion/insertion models need the Phase model type and different labels.
Python 3.10+, stdlib only (runs in the trio-dnm container)."""
import argparse
import gzip
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("vcf")
    ap.add_argument("sample")
    ap.add_argument("--min-vaf", type=float, default=0.03)
    ap.add_argument("--max-vaf", type=float, default=0.40)
    a = ap.parse_args(argv)
    op = gzip.open if a.vcf.endswith((".gz", ".bgz")) else open
    col = None
    n_out = n_indel = 0
    with op(a.vcf, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            f = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                if a.sample not in f[9:]:
                    sys.exit(f"vcf2mfbed.py: sample {a.sample!r} not in {a.vcf} (samples: {', '.join(f[9:])})")
                col = f.index(a.sample, 9)
                continue
            if f[6] not in ("PASS", ".") or "," in f[4]:
                continue
            if len(f[3]) != 1 or len(f[4]) != 1:
                n_indel += 1
                continue
            if f[3].upper() not in "ACGT" or f[4].upper() not in "ACGT":  # e.g. '*' spanning deletion
                continue
            fmt = dict(zip(f[8].split(":"), f[col].split(":")))
            try:
                af = float(fmt.get("AF", "nan").split(",")[0])
            except ValueError:
                continue
            if a.min_vaf <= af <= a.max_vaf:
                pos = int(f[1])
                print("\t".join([f[0], str(pos - 1), str(pos), f[3], f[4], a.sample]))
                n_out += 1
    print(f"vcf2mfbed.py: {n_out} SNV site(s) written; {n_indel} PASS biallelic indel(s) skipped "
          f"(MosaicForecast Refine model is SNV-only)", file=sys.stderr)


if __name__ == "__main__":
    main()
