"""Stage 1 stop-gates: contamination, sex, relatedness and coverage.

Consumes somalier ``relate`` output (``*.pairs.tsv``, ``*.samples.tsv``),
VerifyBamID2 ``*.selfSM`` and mosdepth ``*.mosdepth.summary.txt`` files.
Returns a JSON verdict; the CLI exits non-zero on a hard failure so the
workflow halts before any variant calling. Thresholds come from the ``qc``
section of the configuration (``trio-dnm qc-gate --config``).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .config import DEFAULTS
from .genome import Trio
from .vcf import to_float

GATES = DEFAULTS["qc"]  # default thresholds; pass cfg to evaluate() to override

LCL_WARNING = "lymphoblastoid cell-line DNA: expect culture-derived low-VAF artefacts"


def _gates(cfg: Optional[dict]) -> Dict[str, object]:
    q = dict(GATES)
    q.update((cfg or {}).get("qc") or {})
    return q


def _read_tsv(path: str) -> List[Dict[str, str]]:
    with open(path) as fh:
        header = fh.readline().rstrip("\n").lstrip("#").split("\t")
        return [dict(zip(header, l.rstrip("\n").split("\t"))) for l in fh if l.strip()]


def infer_sex(row: Dict[str, str], gates: Optional[Dict[str, object]] = None) -> str:
    q = gates or GATES
    dm = to_float(row.get("depth_mean")) or to_float(row.get("gt_depth_mean")) or 0.0
    y = to_float(row.get("Y_depth_mean")) or 0.0
    xh, xn = to_float(row.get("X_het")) or 0.0, to_float(row.get("X_n")) or 0.0
    if dm <= 0:
        return "unknown"
    if y / dm >= q["male_y_ratio"] and (xn == 0 or xh / xn < q["male_max_x_het"]):
        return "male"
    if y / dm < q["male_y_ratio"] / 2:
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
    cfg: Optional[dict] = None,
    lcl: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Evaluate the Stage-1 gates for one trio.

    ``reported_sex`` maps sample -> 'male'/'female'/'unknown' (the CLI passes
    all three members: father male, mother female, proband from the PED).
    ``lcl`` lists samples whose DNA comes from a lymphoblastoid cell line.
    """
    q = _gates(cfg)
    members = [trio.proband, trio.father, trio.mother]
    fails: List[str] = []
    warns: List[str] = []
    out: Dict[str, object] = {"family": trio.family, "proband": trio.proband}

    # Contamination
    fm: Dict[str, float] = {}
    for s, p in selfsm.items():
        rows = _read_tsv(p)
        if rows:
            fm[s] = to_float(rows[0].get("FREEMIX")) or 0.0
            if fm[s] > q["freemix_fail"]:
                fails.append(f"{s}: FREEMIX {fm[s]:.3f} > {q['freemix_fail']}")
            elif fm[s] > q["freemix_warn"]:
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
        md = q["min_mean_depth"]
        need = md.get(data_type, 0) if isinstance(md, dict) else md
        if s in cov and cov[s] < need:
            warns.append(f"{s}: mean depth {cov[s]:.1f} < {need}")
    out["mean_depth"] = cov

    # Sex: every member must match the pedigree (a father/mother label swap
    # keeps both parent-child relatedness values at 0.5, so only this catches it).
    if samples_tsv:
        sexes = {r.get("sample_id"): infer_sex(r, q) for r in _read_tsv(samples_tsv)}
        out["inferred_sex"] = sexes
        for s, rep in (reported_sex or {}).items():
            if s not in sexes:
                fails.append(f"{s}: not in somalier samples.tsv (sample IDs must match the PED)")
                continue
            inf = sexes[s]
            if inf == "unknown":
                warns.append(f"{s}: sex could not be inferred from X/Y signal – possible sex-chromosome "
                             f"aneuploidy (e.g. XXY, X0) or low coverage; resolve before interpreting chrX/chrY calls")
            elif rep in ("male", "female") and inf != rep:
                role = {trio.father: " (PED father)", trio.mother: " (PED mother)"}.get(s, "")
                fails.append(f"{s}: reported sex {rep}{role} but inferred {inf} – sample swap or PED error")

    # Relatedness
    if pairs_tsv:
        rel = {}
        for r in _read_tsv(pairs_tsv):
            a, b = r.get("sample_a"), r.get("sample_b")
            rel[frozenset((a, b))] = (to_float(r.get("relatedness")) or 0.0, to_float(r.get("ibs0")) or 0.0, to_float(r.get("n")) or 1.0)
        lo, hi = q["parent_child_relatedness"]
        for parent in (trio.father, trio.mother):
            k = frozenset((trio.proband, parent))
            if k not in rel:
                fails.append(f"no somalier pair for {trio.proband}/{parent}")
                continue
            r, ibs0, n = rel[k]
            if not lo <= r <= hi:
                fails.append(f"{trio.proband}/{parent}: relatedness {r:.2f} not in [{lo}, {hi}] (non-paternity/maternity or swap)")
            if n and ibs0 / n > q["parent_child_max_ibs0"]:
                fails.append(f"{trio.proband}/{parent}: IBS0 fraction {ibs0 / n:.4f} incompatible with parent–child")
        k = frozenset((trio.father, trio.mother))
        if k in rel and rel[k][0] > q["unrelated_max_relatedness"] and not consanguineous:
            warns.append(f"parents relatedness {rel[k][0]:.2f} – undeclared consanguinity?")
        out["relatedness"] = {"/".join(sorted(x for x in k if x)): v[0] for k, v in rel.items()}

    # DNA source
    lcl_members = [s for s in members if s in set(lcl or [])]
    for s in lcl_members:
        warns.append(f"{s}: {LCL_WARNING}")
    out["lcl"] = lcl_members

    out["fail"] = fails
    out["warn"] = warns
    out["status"] = "FAIL" if fails else ("WARN" if warns else "PASS")
    return out
