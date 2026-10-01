"""Small statistics toolkit (stdlib only) and the trio Bayesian DNM model."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple


def log_binom_pmf(k: int, n: int, p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log(1 - p)
    )


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, sum(math.exp(log_binom_pmf(i, n, p)) for i in range(0, k + 1)))


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k)."""
    if k <= 0:
        return 1.0
    return min(1.0, sum(math.exp(log_binom_pmf(i, n, p)) for i in range(k, n + 1)))


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for [[a, b], [c, d]]."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def logp(x: int) -> float:
        return (
            math.lgamma(r1 + 1)
            + math.lgamma(n - r1 + 1)
            + math.lgamma(c1 + 1)
            + math.lgamma(n - c1 + 1)
            - math.lgamma(n + 1)
            - math.lgamma(x + 1)
            - math.lgamma(r1 - x + 1)
            - math.lgamma(c1 - x + 1)
            - math.lgamma(n - r1 - c1 + x + 1)
        )

    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    p_obs = logp(a)
    total = 0.0
    for x in range(lo, hi + 1):
        lp = logp(x)
        if lp <= p_obs + 1e-7:
            total += math.exp(lp)
    return min(1.0, total)


def logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


def median(xs: Sequence[float]) -> Optional[float]:
    s = sorted(xs)
    if not s:
        return None
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# --------------------------------------------------------------------------- #
# Trio DNM posterior
# --------------------------------------------------------------------------- #
def log_likelihoods(pl: List[int], ref_depth: int, alt_depth: int, ploidy: int, err: float = 0.01) -> List[float]:
    """Natural-log genotype likelihoods indexed by alt-allele count.

    Uses the caller's PL when present and of the right length, otherwise a
    binomial read-count model with a symmetric per-read error rate.
    """
    n_gt = ploidy + 1
    if len(pl) == n_gt:
        return [-p / 10.0 * math.log(10) for p in pl]
    n = ref_depth + alt_depth
    if ploidy == 1:
        ps = [err, 1 - err]
    else:
        ps = [err, 0.5, 1 - err]
    return [log_binom_pmf(alt_depth, n, p) for p in ps]


def _hwe(q: float, ploidy: int) -> List[float]:
    if ploidy == 1:
        return [1 - q, q]
    return [(1 - q) ** 2, 2 * q * (1 - q), q * q]


def _transmit(g: int, ploidy: int) -> float:
    """P(transmitted allele is ALT | parent genotype)."""
    return g / ploidy


def _child_given_alleles(p_alt_from: List[float], mu: float) -> List[float]:
    """Distribution of child alt-count given per-allele P(alt) before mutation."""
    dist = [1.0]
    for p in p_alt_from:
        p_post = p * (1 - mu) + (1 - p) * mu
        new = [0.0] * (len(dist) + 1)
        for k, v in enumerate(dist):
            new[k] += v * (1 - p_post)
            new[k + 1] += v * p_post
        dist = new
    return dist


def trio_dnm_posterior(
    child: Tuple[List[float], int],
    father: Optional[Tuple[List[float], int]],
    mother: Optional[Tuple[List[float], int]],
    pop_af: float,
    mu: float,
    af_floor: float = 1e-4,
) -> Dict[str, float]:
    """Posterior probability that the child's ALT allele arose de novo.

    Each argument is ``(log_likelihoods, ploidy)``; a parent is ``None`` when it
    does not contribute a chromosome at this locus (e.g. the father for a male
    proband's chrX outside the PARs). Enumerates every joint genotype
    configuration with HWE parental priors (allele frequency floored at
    ``af_floor`` so absent-from-gnomAD sites stay conservative) and a
    per-transmitted-allele mutation probability ``mu``.
    """
    q = min(max(pop_af or 0.0, af_floor), 0.5)
    parents = [p for p in (father, mother) if p is not None]
    child_ll, _ = child
    terms: List[float] = []
    dnm_terms: List[float] = []

    def rec(idx: int, gts: List[int], log_prior: float, log_lik: float):
        if idx == len(parents):
            p_alt = [_transmit(g, p[1]) for g, p in zip(gts, parents)]
            dist = _child_given_alleles(p_alt, mu)
            for c in range(len(child_ll)):
                pc = dist[c] if c < len(dist) else 0.0
                if pc <= 0:
                    continue
                t = log_prior + log_lik + math.log(pc) + child_ll[c]
                terms.append(t)
                if c > 0 and all(g == 0 for g in gts):
                    dnm_terms.append(t)
            return
        ll, ploidy = parents[idx]
        for g, pr in enumerate(_hwe(q, ploidy)):
            rec(idx + 1, gts + [g], log_prior + math.log(pr), log_lik + ll[g])

    rec(0, [], 0.0, 0.0)
    total = logsumexp(terms)
    p_dnm = math.exp(logsumexp(dnm_terms) - total) if dnm_terms else 0.0
    return {"p_dnm": p_dnm, "phred": -10 * math.log10(max(1e-30, 1 - p_dnm)) if p_dnm < 1 else 300.0}
