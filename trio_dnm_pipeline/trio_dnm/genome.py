"""Reference-genome helpers: indexed FASTA access, BED interval sets, PAR/ploidy
logic, pedigree parsing and sequence-context features (CpG, homopolymers).
"""
from __future__ import annotations

import bisect
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .vcf import open_text

# Pseudo-autosomal regions (1-based, inclusive).
PAR = {
    "GRCh38": {
        "X": [(10001, 2781479), (155701383, 156030895)],
        "Y": [(10001, 2781479), (56887903, 57217415)],
    },
    "GRCh37": {
        "X": [(60001, 2699520), (154931044, 155260560)],
        "Y": [(10001, 2649520), (59034050, 59363566)],
    },
}


def bare_chrom(chrom: str) -> str:
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def chrom_class(chrom: str, pos: int, build: str = "GRCh38") -> str:
    """Return 'auto', 'PAR', 'X', 'Y' or 'MT'."""
    c = bare_chrom(chrom).upper()
    if c in ("M", "MT"):
        return "MT"
    if c in ("X", "Y"):
        for s, e in PAR[build][c]:
            if s <= pos <= e:
                return "PAR"
        return c
    return "auto"


def expected_ploidy(chrom: str, pos: int, sex: str, build: str = "GRCh38") -> int:
    cls = chrom_class(chrom, pos, build)
    if cls == "X":
        return 1 if sex == "male" else 2
    if cls == "Y":
        return 1 if sex == "male" else 0
    return 2


# --------------------------------------------------------------------------- #
# Pedigree
# --------------------------------------------------------------------------- #
@dataclass
class Trio:
    family: str
    proband: str
    father: str
    mother: str
    proband_sex: str  # 'male' | 'female' | 'unknown'
    affected: bool = True


def read_ped(path: str) -> List[Trio]:
    """Read a PLINK PED/FAM file and return every child with both parents present."""
    people: Dict[str, Tuple[str, str, str, str, str]] = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            f = line.split()
            fam, iid, pat, mat, sex = f[0], f[1], f[2], f[3], f[4]
            aff = f[5] if len(f) > 5 else "0"
            people[iid] = (fam, pat, mat, sex, aff)
    trios = []
    for iid, (fam, pat, mat, sex, aff) in people.items():
        if pat not in ("0", "") and mat not in ("0", "") and pat in people and mat in people:
            trios.append(
                Trio(fam, iid, pat, mat, {"1": "male", "2": "female"}.get(sex, "unknown"), aff == "2")
            )
    return trios


# --------------------------------------------------------------------------- #
# Indexed FASTA (samtools faidx format) – pure Python
# --------------------------------------------------------------------------- #
class Fasta:
    def __init__(self, path: str):
        fai = path + ".fai"
        if not os.path.exists(fai):
            raise FileNotFoundError(f"{fai} not found; run `samtools faidx {path}`")
        self.path = path
        self.index: Dict[str, Tuple[int, int, int, int]] = {}
        with open(fai) as fh:
            for line in fh:
                name, length, offset, lb, lw = line.split("\t")[:5]
                self.index[name] = (int(length), int(offset), int(lb), int(lw))
        self._fh = open(path, "rb")

    def _resolve(self, chrom: str) -> Optional[str]:
        if chrom in self.index:
            return chrom
        alt = bare_chrom(chrom) if chrom.lower().startswith("chr") else "chr" + chrom
        return alt if alt in self.index else None

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """1-based inclusive coordinates; clipped to the contig."""
        name = self._resolve(chrom)
        if name is None:
            return ""
        length, offset, lb, lw = self.index[name]
        start, end = max(1, start), min(length, end)
        if end < start:
            return ""
        s0, e0 = start - 1, end  # half-open 0-based
        b_start = offset + (s0 // lb) * lw + s0 % lb
        b_end = offset + ((e0 - 1) // lb) * lw + (e0 - 1) % lb + 1
        self._fh.seek(b_start)
        raw = self._fh.read(b_end - b_start).decode()
        return raw.replace("\n", "").replace("\r", "").upper()


def homopolymer_run(fasta: Fasta, chrom: str, pos: int, window: int = 20) -> int:
    """Longest single-base run touching the variant position (+/- 1bp)."""
    seq = fasta.fetch(chrom, pos - window, pos + window)
    if not seq:
        return 0
    centre = min(window, pos - 1)  # index of pos in seq
    best = 0
    i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        if i <= centre + 1 and j - 1 >= centre - 1 and seq[i] != "N":
            best = max(best, j - i)
        i = j
    return best


def near_n(fasta: Fasta, chrom: str, pos: int, pad: int = 5) -> bool:
    return "N" in fasta.fetch(chrom, pos - pad, pos + pad)


# --------------------------------------------------------------------------- #
# BED interval sets
# --------------------------------------------------------------------------- #
class IntervalSet:
    """Merged, sorted intervals per contig with O(log n) point/range queries."""

    def __init__(self):
        self._starts: Dict[str, List[int]] = {}
        self._ends: Dict[str, List[int]] = {}
        self.name = ""

    @classmethod
    def from_bed(cls, path: str) -> "IntervalSet":
        raw: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        with open_text(path) as fh:
            for line in fh:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                f = line.split("\t")
                raw[bare_chrom(f[0])].append((int(f[1]), int(f[2])))  # 0-based half-open
        obj = cls()
        obj.name = os.path.basename(path)
        for c, ivs in raw.items():
            ivs.sort()
            merged: List[List[int]] = []
            for s, e in ivs:
                if merged and s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            obj._starts[c] = [m[0] for m in merged]
            obj._ends[c] = [m[1] for m in merged]
        return obj

    def overlaps(self, chrom: str, start1: int, end1: Optional[int] = None) -> bool:
        """1-based inclusive query range."""
        c = bare_chrom(chrom)
        if c not in self._starts:
            return False
        end1 = start1 if end1 is None else end1
        q0, q1 = start1 - 1, end1  # to half-open
        starts, ends = self._starts[c], self._ends[c]
        i = bisect.bisect_right(starts, q1 - 1) - 1
        return i >= 0 and ends[i] > q0

    def total_bp(self) -> int:
        return sum(e - s for c in self._starts for s, e in zip(self._starts[c], self._ends[c]))


# --------------------------------------------------------------------------- #
# Mutation context
# --------------------------------------------------------------------------- #
COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


SBS96 = [
    f"{l}[{m}]{r}"
    for m in ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
    for l in "ACGT"
    for r in "ACGT"
]


def sbs96_channel(ref: str, alt: str, left: str, right: str) -> Optional[str]:
    """Pyrimidine-normalised trinucleotide channel, e.g. 'A[C>T]G'."""
    if len(ref) != 1 or len(alt) != 1 or "N" in (left + ref + right):
        return None
    if ref in "GA":
        ref, alt, left, right = revcomp(ref), revcomp(alt), revcomp(right), revcomp(left)
    return f"{left}[{ref}>{alt}]{right}"


def is_transition(ref: str, alt: str) -> bool:
    return {ref, alt} in ({"A", "G"}, {"C", "T"})


def is_cpg_transition(ref: str, alt: str, left: str, right: str) -> bool:
    """C>T at CpG (C followed by G) or G>A where preceded by C."""
    return (ref == "C" and alt == "T" and right == "G") or (ref == "G" and alt == "A" and left == "C")
