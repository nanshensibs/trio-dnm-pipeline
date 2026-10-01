"""SBS-96 mutational spectra and non-negative signature refitting.

Refitting (not de novo extraction): exposures are estimated for a
user-supplied COSMIC-format signature matrix (first column ``Type`` such as
``A[C>A]A``, one column per signature). With ~60-80 DNMs per trio, restrict
the fit to a small biologically plausible set (default SBS1, SBS5, SBS40a for
germline; COSMIC v3.4 split SBS40 into SBS40a/b/c); a full-catalogue fit on so
few mutations over-fits.
"""
from __future__ import annotations

import math
import sys
from typing import Dict, Iterable, List, Optional, Tuple

from .genome import SBS96, Fasta, sbs96_channel
from .vcf import open_text

GERMLINE_SIGNATURES = ["SBS1", "SBS5", "SBS40a"]


def spectrum(snvs: Iterable[Tuple[str, int, str, str]], fasta: Fasta) -> Dict[str, int]:
    counts = {ch: 0 for ch in SBS96}
    for chrom, pos, ref, alt in snvs:
        ctx = fasta.fetch(chrom, pos - 1, pos + 1)
        if len(ctx) != 3 or ctx[1] != ref.upper():
            continue
        ch = sbs96_channel(ref.upper(), alt.upper(), ctx[0], ctx[2])
        if ch:
            counts[ch] += 1
    return counts


def load_signatures(path: str, keep: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, List[float]]]:
    with open_text(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        names = header[1:]
        rows: Dict[str, List[float]] = {}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 2:
                continue
            rows[f[0]] = [float(x) for x in f[1:]]
    idx = list(range(len(names))) if not keep else _resolve(names, keep, path)
    sigs = {names[i]: [rows[ch][i] if ch in rows else 0.0 for ch in SBS96] for i in idx}
    return [names[i] for i in idx], sigs


def _resolve(names: List[str], keep: List[str], path: str) -> List[int]:
    """Column indices of the requested signatures (matrix order). COSMIC v3.4
    split SBS40 into SBS40a/b/c: a missing 'SBS40' is taken as 'SBS40a' (and a
    missing 'SBS40a' as 'SBS40' in older matrices), with a warning."""
    found = set()
    for want in (k.strip() for k in keep if k.strip()):
        if want in names:
            found.add(want)
        elif want + "a" in names:
            print(f"warning: signature {want} not in {path}; using {want}a (COSMIC v3.4 split)", file=sys.stderr)
            found.add(want + "a")
        elif want[-1:] == "a" and want[:-1] in names:
            print(f"warning: signature {want} not in {path}; using {want[:-1]} (pre-v3.4 matrix)", file=sys.stderr)
            found.add(want[:-1])
        else:
            print(f"warning: signature {want} not in {path}; left out of the refit", file=sys.stderr)
    if not found:
        raise SystemExit(f"none of the requested signatures ({', '.join(keep)}) found in {path}")
    return [i for i, n in enumerate(names) if n in found]


def nnls(W: List[List[float]], v: List[float], iters: int = 5000, tol: float = 1e-10) -> List[float]:
    """min ||W h - v||^2 s.t. h >= 0 by cyclic coordinate descent.
    W is given column-wise: W[j] is signature j (length len(v))."""
    k = len(W)
    h = [1.0 / k] * k
    gram = [[sum(a * b for a, b in zip(W[i], W[j])) for j in range(k)] for i in range(k)]
    wtv = [sum(a * b for a, b in zip(W[i], v)) for i in range(k)]
    for _ in range(iters):
        delta = 0.0
        for j in range(k):
            if gram[j][j] <= 0:
                continue
            grad = sum(gram[j][i] * h[i] for i in range(k)) - wtv[j]
            new = max(0.0, h[j] - grad / gram[j][j])
            delta = max(delta, abs(new - h[j]))
            h[j] = new
        if delta < tol:
            break
    return h


def cosine(a: List[float], b: List[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else sum(x * y for x, y in zip(a, b)) / (na * nb)


def refit(counts: Dict[str, int], sig_names: List[str], sigs: Dict[str, List[float]]) -> Dict[str, object]:
    v = [float(counts.get(ch, 0)) for ch in SBS96]
    total = sum(v)
    if total == 0 or not sig_names:
        return {"n_snv": int(total), "exposures": {}, "cosine": None}
    vp = [x / total for x in v]
    W = [sigs[n] for n in sig_names]
    h = nnls(W, vp)
    s = sum(h) or 1.0
    recon = [sum(W[j][i] * h[j] for j in range(len(W))) for i in range(len(SBS96))]
    return {
        "n_snv": int(total),
        "exposures": {n: round(x / s, 4) for n, x in zip(sig_names, h)},
        "cosine": round(cosine(vp, recon), 4),
    }


def write_spectrum(path: str, counts: Dict[str, int]) -> None:
    with open(path, "w") as fh:
        fh.write("channel\tcount\n")
        for ch in SBS96:
            fh.write(f"{ch}\t{counts.get(ch, 0)}\n")
