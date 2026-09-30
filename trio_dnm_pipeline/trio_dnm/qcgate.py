"""Stage 1 stop-gates: contamination, sex, relatedness and coverage.

Consumes somalier ``relate`` output (``*.pairs.tsv``, ``*.samples.tsv``),
VerifyBamID2 ``*.selfSM`` and mosdepth ``*.mosdepth.summary.txt`` files.
Returns a JSON verdict; the CLI exits non-zero on a hard failure so the
workflow halts before any variant calling.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .genome import Trio
from .vcf import to_float

GATES = {
    "freemix_warn": 0.02,
    "freemix_fail": 0.05,
    "parent_child_relatedness": [0.40, 0.60],  # somalier 'relatedness' (≈ 2 × kinship)
    "parent_child_max_ibs0": 0.005,  # fraction of informative sites
    "unrelated_max_relatedness": 0.10,  # parents, unless consanguinity is declared
    "min_mean_depth": {"wgs": 28.0, "wes": 80.0, "panel": 200.0},
    "male_y_ratio": 0.10,  # Y depth / autosomal depth above which a sample is male
}


def _read_tsv(path: str) -> List[Dict[str, str]]:
    with open(path) as fh:
        header = fh.readline().rstrip("\n").lstrip("#").split("\t")
        return [dict(zip(header, l.rstrip("\n").split("\t"))) for l in fh if l.strip()]


def infer_sex(row: Dict[str, str]) -> str:
    dm = to_float(row.get("depth_mean")) or to_float(row.get("gt_depth_mean")) or 0.0
    y = to_float(row.get("Y_depth_mean")) or 0.0
    xh, xn = to_float(row.get("X_het")) or 0.0, to_float(row.get("X_n")) or 0.0
    if dm <= 0:
        return "unknown"
    if y / dm >= GATES["male_y_ratio"] and (xn == 0 or xh / xn < 0.05):
        return "male"
    if y / dm < GATES["male_y_ratio"] / 2:
        return "female"
    return "unknown"


def evaluate(
    trio: Trio,
    pairs_tsv: Optional[str],
    samples_tsv: Optional[str],
    selfsm: Dict[str, str],
    mosdepth: Dict[str, str],
    data_type: str = "wgs",
    reported_sex: Optional[Dict[str, str]] = None,
    consanguineous: bool = False,
) -> Dict[str, object]:
    fails: List[str] = []
    warns: List[str] = []
    out: Dict[str, object] = {"family": trio.family, "proband": trio.proband}

    # Contamination
    fm: Dict[str, float] = {}
    for s, p in selfsm.items():
        rows = _read_tsv(p)
        if rows:
            fm[s] = to_float(rows[0].get("FREEMIX")) or 0.0
            if fm[s] > GATES["freemix_fail"]:
                fails.append(f"{s}: FREEMIX {fm[s]:.3f} > {GATES['freemix_fail']}")
            elif fm[s] > GATES["freemix_warn"]:
                warns.append(f"{s}: FREEMIX {fm[s]:.3f} – investigate")
    out["freemix"] = fm

    # Coverage
    cov: Dict[str, float] = {}
    for s, p in mosdepth.items():
        with open(p) as fh:
            for line in fh:
                f = line.split("\t")
                if f[0] in ("total", "total_region"):
                    cov[s] = float(f[3])
        need = GATES["min_mean_depth"].get(data_type, 0)
        if s in cov and cov[s] < need:
            warns.append(f"{s}: mean depth {cov[s]:.1f} < {need}")
    out["mean_depth"] = cov

    # Sex
    if samples_tsv:
        sexes = {r.get("sample_id"): infer_sex(r) for r in _read_tsv(samples_tsv)}
        out["inferred_sex"] = sexes
        for s, rep in (reported_sex or {}).items():
            inf = sexes.get(s, "unknown")
            if inf != "unknown" and rep in ("male", "female") and inf != rep:
                fails.append(f"{s}: reported sex {rep} but inferred {inf}")

    # Relatedness
    if pairs_tsv:
        rel = {}
        for r in _read_tsv(pairs_tsv):
            a, b = r.get("sample_a"), r.get("sample_b")
            rel[frozenset((a, b))] = (to_float(r.get("relatedness")) or 0.0, to_float(r.get("ibs0")) or 0.0, to_float(r.get("n")) or 1.0)
        lo, hi = GATES["parent_child_relatedness"]
        for parent in (trio.father, trio.mother):
            k = frozenset((trio.proband, parent))
            if k not in rel:
                fails.append(f"no somalier pair for {trio.proband}/{parent}")
                continue
            r, ibs0, n = rel[k]
            if not lo <= r <= hi:
                fails.append(f"{trio.proband}/{parent}: relatedness {r:.2f} not in [{lo}, {hi}] (non-paternity/maternity or swap)")
            if n and ibs0 / n > GATES["parent_child_max_ibs0"]:
                fails.append(f"{trio.proband}/{parent}: IBS0 fraction {ibs0 / n:.4f} incompatible with parent–child")
        k = frozenset((trio.father, trio.mother))
        if k in rel and rel[k][0] > GATES["unrelated_max_relatedness"] and not consanguineous:
            warns.append(f"parents relatedness {rel[k][0]:.2f} – undeclared consanguinity?")
        out["relatedness"] = {"/".join(sorted(x for x in k if x)): v[0] for k, v in rel.items()}

    out["fail"] = fails
    out["warn"] = warns
    out["status"] = "FAIL" if fails else ("WARN" if warns else "PASS")
    return out
