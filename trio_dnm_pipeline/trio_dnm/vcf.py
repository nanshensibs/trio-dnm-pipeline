"""Minimal, dependency-free VCF reader/writer.

Only what the post-calling stages need: header metadata, per-record INFO and
per-sample FORMAT access, and writing records back out with extra INFO keys.
Input is expected to be left-aligned and split to biallelic records upstream
(``bcftools norm -m -any -f ref.fa``); multi-allelic records are read using
their first ALT allele only.
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, TextIO

MISSING = {".", "", None}


def open_text(path: str, mode: str = "rt") -> TextIO:
    if path.endswith((".gz", ".bgz")):
        return gzip.open(path, mode)  # type: ignore[return-value]
    return open(path, mode)


def to_float(v) -> Optional[float]:
    if v in MISSING:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_int(v) -> Optional[int]:
    f = to_float(v)
    return None if f is None else int(round(f))


def int_list(v) -> List[int]:
    if v in MISSING:
        return []
    out = []
    for x in str(v).split(","):
        out.append(0 if x in MISSING else int(float(x)))
    return out


@dataclass
class Genotype:
    """Per-sample view of the FORMAT columns for one (biallelic) record."""

    fields: Dict[str, str]

    @property
    def gt(self) -> str:
        return self.fields.get("GT", "./.")

    @property
    def alleles(self) -> List[Optional[int]]:
        return [None if a == "." else int(a) for a in re.split(r"[/|]", self.gt)]

    @property
    def ploidy(self) -> int:
        return len(self.alleles)

    @property
    def called(self) -> bool:
        return all(a is not None for a in self.alleles)

    @property
    def alt_count(self) -> Optional[int]:
        """Number of non-reference alleles in GT (0/1/2), None if uncalled."""
        if not self.called:
            return None
        return sum(1 for a in self.alleles if a and a > 0)

    @property
    def gq(self) -> Optional[int]:
        return to_int(self.fields.get("GQ"))

    @property
    def ad(self) -> List[int]:
        return int_list(self.fields.get("AD"))

    @property
    def ref_depth(self) -> int:
        ad = self.ad
        return ad[0] if ad else 0

    @property
    def alt_depth(self) -> int:
        ad = self.ad
        return sum(ad[1:]) if len(ad) > 1 else 0

    @property
    def dp(self) -> int:
        dp = to_int(self.fields.get("DP"))
        if dp is None:
            return sum(self.ad)
        # Some callers report DP < sum(AD) (filtered reads); trust the larger.
        return max(dp, sum(self.ad))

    @property
    def vaf(self) -> Optional[float]:
        ad = self.ad
        tot = sum(ad)
        if tot == 0:
            vf = to_float(self.fields.get("VAF") or self.fields.get("AF"))
            return vf
        return self.alt_depth / tot

    @property
    def pl(self) -> List[int]:
        pl = self.fields.get("PL")
        if pl in MISSING:
            gl = self.fields.get("GL")
            if gl in MISSING:
                return []
            # GL is log10 likelihood; convert to Phred-scaled normalised PL.
            vals = [float(x) for x in gl.split(",")]
            best = max(vals)
            return [int(round(-10 * (v - best))) for v in vals]
        return int_list(pl)

    def strand_alt(self) -> Optional[tuple]:
        """(alt_forward, alt_reverse) read counts if the caller reports them.

        Supports GATK ``SB`` (refF,refR,altF,altR) and bcftools/DeepVariant
        style ``ADF``/``ADR``.
        """
        sb = self.fields.get("SB")
        if sb not in MISSING:
            v = int_list(sb)
            if len(v) >= 4:
                return v[2], v[3]
        adf, adr = self.fields.get("ADF"), self.fields.get("ADR")
        if adf not in MISSING and adr not in MISSING:
            f, r = int_list(adf), int_list(adr)
            if len(f) > 1 and len(r) > 1:
                return sum(f[1:]), sum(r[1:])
        return None


@dataclass
class Record:
    chrom: str
    pos: int
    id: str
    ref: str
    alt: str
    qual: Optional[float]
    filter: str
    info: Dict[str, object]
    format_keys: List[str]
    samples: Dict[str, Genotype]
    extra_alts: List[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"

    @property
    def is_snv(self) -> bool:
        return len(self.ref) == 1 and len(self.alt) == 1 and self.alt != "*"

    @property
    def is_indel(self) -> bool:
        return not self.is_snv and len(self.ref) != len(self.alt)

    @property
    def is_mnv(self) -> bool:
        return len(self.ref) == len(self.alt) > 1

    @property
    def indel_length(self) -> int:
        return len(self.alt) - len(self.ref)

    def info_float(self, key: str) -> Optional[float]:
        v = self.info.get(key)
        if isinstance(v, str) and "," in v:
            v = v.split(",")[0]
        return to_float(v)

    def info_flag(self, key: str) -> bool:
        return self.info.get(key) is True

    def to_line(self, sample_order: List[str]) -> str:
        info_parts = []
        for k, v in self.info.items():
            if v is True:
                info_parts.append(k)
            elif v is False or v is None:
                continue
            else:
                info_parts.append(f"{k}={v}")
        cols = [
            self.chrom,
            str(self.pos),
            self.id,
            self.ref,
            ",".join([self.alt] + self.extra_alts),
            "." if self.qual is None else f"{self.qual:g}",
            self.filter,
            ";".join(info_parts) or ".",
        ]
        if sample_order:
            cols.append(":".join(self.format_keys))
            for s in sample_order:
                g = self.samples[s].fields
                cols.append(":".join(g.get(k, ".") for k in self.format_keys))
        return "\t".join(cols)


class VCFReader:
    def __init__(self, path: str):
        self.path = path
        self.meta: List[str] = []
        self.samples: List[str] = []
        self.csq_fields: List[str] = []
        self._fh = open_text(path)
        for line in self._fh:
            line = line.rstrip("\n")
            if line.startswith("##"):
                self.meta.append(line)
                if line.startswith("##INFO=<ID=CSQ") or line.startswith("##INFO=<ID=ANN"):
                    m = re.search(r'Format: ([^"]+)"', line)
                    if m and not self.csq_fields:
                        self.csq_fields = [x.strip() for x in m.group(1).split("|")]
            elif line.startswith("#CHROM"):
                self.samples = line.split("\t")[9:]
                break

    def __iter__(self) -> Iterator[Record]:
        for line in self._fh:
            if not line.strip() or line.startswith("#"):
                continue
            yield parse_record(line.rstrip("\n"), self.samples)
        self._fh.close()


def parse_info(s: str) -> Dict[str, object]:
    info: Dict[str, object] = {}
    if s in MISSING:
        return info
    for part in s.split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            info[k] = v
        else:
            info[part] = True
    return info


def parse_record(line: str, samples: List[str]) -> Record:
    cols = line.split("\t")
    alts = cols[4].split(",")
    fmt = cols[8].split(":") if len(cols) > 8 else []
    gts: Dict[str, Genotype] = {}
    for name, raw in zip(samples, cols[9:]):
        vals = raw.split(":")
        gts[name] = Genotype({k: (vals[i] if i < len(vals) else ".") for i, k in enumerate(fmt)})
    return Record(
        chrom=cols[0],
        pos=int(cols[1]),
        id=cols[2],
        ref=cols[3],
        alt=alts[0],
        qual=to_float(cols[5]),
        filter=cols[6],
        info=parse_info(cols[7]),
        format_keys=fmt,
        samples=gts,
        extra_alts=alts[1:],
    )


class VCFWriter:
    def __init__(self, path: str, reader: VCFReader, extra_header: List[str], samples: Optional[List[str]] = None):
        self.samples = reader.samples if samples is None else samples
        self._fh = open_text(path, "wt")
        meta = [m for m in reader.meta if not any(m.startswith(h.split(",")[0]) for h in extra_header)]
        for m in meta + extra_header:
            self._fh.write(m + "\n")
        cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
        if self.samples:
            cols += ["FORMAT"] + self.samples
        self._fh.write("\t".join(cols) + "\n")

    def write(self, rec: Record) -> None:
        self._fh.write(rec.to_line(self.samples) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def parse_csq(value: str, fields: List[str]) -> List[Dict[str, str]]:
    """Split a VEP CSQ (or SnpEff ANN) INFO value into per-transcript dicts."""
    out = []
    if not value or value is True:
        return out
    for entry in str(value).split(","):
        parts = entry.split("|")
        out.append({f: (parts[i] if i < len(parts) else "") for i, f in enumerate(fields)})
    return out
