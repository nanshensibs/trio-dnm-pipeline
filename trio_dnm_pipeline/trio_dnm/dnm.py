"""Stage 4-7: candidate extraction, layered filter cascade, Bayesian trio
posterior, mosaic / parental-mosaic tracks and multi-caller consensus.

Every candidate Mendelian violation is evaluated against *all* layers so the
TSV records every reason it failed; the waterfall counts each candidate once,
at the first layer it fails.
"""
from __future__ import annotations

import json
import math
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from . import genome, stats
from .genome import Fasta, IntervalSet, Trio
from .vcf import Genotype, Record, VCFReader, VCFWriter, open_text

LAYERS = ["L1_genotype", "L2_read_quality", "L3_region", "L4_population", "L5_posterior"]
TRACKS = ("germline", "mosaic", "parental_mosaic")


@dataclass
class Candidate:
    rec: Record
    track: str
    ploidy: int
    fails: Dict[str, List[str]] = field(default_factory=lambda: OrderedDict((l, []) for l in LAYERS))
    flags: List[str] = field(default_factory=list)
    posterior: Optional[float] = None
    pop_af: Optional[float] = None
    callers: List[str] = field(default_factory=list)
    hi_conf: bool = False
    tier: str = ""
    proband_vaf_ci: Tuple[float, float] = (0.0, 1.0)
    homopolymer: int = 0
    context: str = ""

    @property
    def passed(self) -> bool:
        return not any(self.fails.values())

    @property
    def first_fail(self) -> Optional[str]:
        for l, r in self.fails.items():
            if r:
                return l
        return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def norm_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{genome.bare_chrom(chrom)}:{pos}:{ref}:{alt}"


def load_site_af(path: Optional[str]) -> Dict[str, float]:
    """Panel-of-normals VCF (INFO/AF, or fraction of non-ref samples) -> AF."""
    out: Dict[str, float] = {}
    if not path:
        return out
    reader = VCFReader(path)
    for r in reader:
        af = r.info_float("AF")
        if af is None and r.samples:
            called = [g for g in r.samples.values() if g.called]
            af = sum(1 for g in called if g.alt_count) / max(1, len(called))
        out[norm_key(r.chrom, r.pos, r.ref, r.alt)] = af or 0.0
    return out


def load_recurrence(path: Optional[str]) -> Dict[str, int]:
    """TSV with columns key (chrom:pos:ref:alt) and n_trios."""
    out: Dict[str, int] = {}
    if not path:
        return out
    with open_text(path) as fh:
        for line in fh:
            if line.startswith(("#", "key")) or not line.strip():
                continue
            k, n = line.rstrip("\n").split("\t")[:2]
            c, p, r, a = k.split(":")
            out[norm_key(c, int(p), r, a)] = int(n)
    return out


def load_caller_sites(path: str, proband: Optional[str] = None) -> set:
    """Sites reported as DNMs by an external caller (VCF or chrom/pos/ref/alt TSV).

    ``path`` may be a comma-separated list (e.g. Strelka2 SNV + indel VCFs).

    For VCFs, records with FILTER not PASS/. are ignored; if the VCF has a
    proband column and proband is given, the proband must carry the ALT.
    """
    if "," in path:
        return set().union(*(load_caller_sites(p, proband) for p in path.split(",")))
    sites = set()
    if path.endswith((".vcf", ".vcf.gz", ".bcf")):
        reader = VCFReader(path)
        for r in reader:
            if r.filter not in ("PASS", "."):
                continue
            if proband and proband in r.samples and r.samples[proband].called and not r.samples[proband].alt_count:
                continue
            sites.add(norm_key(r.chrom, r.pos, r.ref, r.alt))
            for a in r.extra_alts:
                sites.add(norm_key(r.chrom, r.pos, r.ref, a))
    else:
        with open_text(path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                f = line.rstrip("\n").split("\t")
                if f[0].lower() in ("chrom", "chr") or not f[1].isdigit():
                    continue
                sites.add(norm_key(f[0], int(f[1]), f[2], f[3]))
    return sites


def estimate_mean_depth(vcf_path: str, samples: Iterable[str], max_records: int = 200000) -> Dict[str, float]:
    depths: Dict[str, List[int]] = {s: [] for s in samples}
    for i, rec in enumerate(VCFReader(vcf_path)):
        if i >= max_records:
            break
        if genome.chrom_class(rec.chrom, rec.pos) != "auto":
            continue
        for s in depths:
            g = rec.samples.get(s)
            if g is not None and g.called and g.dp > 0:
                depths[s].append(g.dp)
    return {s: float(stats.median(v) or 0.0) for s, v in depths.items()}


def get_pop_af(rec: Record, keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in rec.info:
            v = rec.info_float(k)
            if v is not None:
                return v
    return None


def is_clinvar_plp(rec: Record, keys: List[str]) -> bool:
    for k in keys:
        v = rec.info.get(k)
        if isinstance(v, str):
            low = v.lower()
            if "pathogenic" in low and "conflicting" not in low and "benign" not in low:
                return True
    return False


# --------------------------------------------------------------------------- #
# Candidate extraction (Mendelian violation + track assignment)
# --------------------------------------------------------------------------- #
def classify_candidate(rec: Record, trio: Trio, cfg: dict) -> Optional[Candidate]:
    cls = genome.chrom_class(rec.chrom, rec.pos, cfg["build"])
    if cls == "MT" or rec.alt in ("*", "<NON_REF>", "."):
        return None
    ploidy = genome.expected_ploidy(rec.chrom, rec.pos, trio.proband_sex, cfg["build"])
    if ploidy == 0:
        return None
    kid = rec.samples.get(trio.proband)
    dad = rec.samples.get(trio.father)
    mom = rec.samples.get(trio.mother)
    if kid is None or dad is None or mom is None:
        return None

    # Which parents contribute a chromosome here.
    contributing = contributing_parents(cls, trio)
    m = cfg["mosaic"]
    kid_alt, kid_dp = kid.alt_depth, kid.dp
    kid_has_alt = (kid.called and (kid.alt_count or 0) > 0) or (m["enabled"] and kid_alt >= m["min_alt"])
    if not kid_has_alt:
        return None

    # Parents must not be *called* carriers.
    for p in contributing:
        g = rec.samples[p]
        if g.called and (g.alt_count or 0) > 0:
            return None

    track = "germline"
    vaf = kid.vaf or 0.0
    if ploidy == 2 and kid_dp > 0:
        p_low = stats.binom_cdf(kid_alt, kid_dp, 0.5)
        if vaf < cfg["proband"]["het_vaf_min"] or (p_low < m["het_binom_p"] and vaf < 0.40):
            track = "mosaic"
    elif ploidy == 1 and vaf < cfg["proband"]["hemi_vaf_min"]:
        track = "mosaic"

    # Parental mosaicism: significant low-level ALT reads in a parent.
    for p in contributing:
        g = rec.samples[p]
        pv = g.vaf or 0.0
        if (
            g.alt_depth >= m["parental_mosaic_min_alt"]
            and pv <= m["parental_mosaic_max_vaf"]
            and stats.binom_sf(g.alt_depth, g.dp, m["seq_error"]) < m["parental_mosaic_p"]
        ):
            track = "parental_mosaic"
    if track == "mosaic" and not m["enabled"]:
        return None
    c = Candidate(rec=rec, track=track, ploidy=ploidy)
    c.proband_vaf_ci = stats.wilson_interval(kid_alt, kid_dp)
    return c


def contributing_parents(cls: str, trio: Trio) -> List[str]:
    if cls == "X" and trio.proband_sex == "male":
        return [trio.mother]
    if cls == "Y":
        return [trio.father]
    return [trio.father, trio.mother]


# --------------------------------------------------------------------------- #
# Filter layers
# --------------------------------------------------------------------------- #
def layer1_genotype(c: Candidate, trio: Trio, cfg: dict, mean_dp: Dict[str, float]) -> None:
    rec, out = c.rec, c.fails["L1_genotype"]
    pb, pa, mo = cfg["proband"], cfg["parent"], cfg["mosaic"]
    kid = rec.samples[trio.proband]
    cls = genome.chrom_class(rec.chrom, rec.pos, cfg["build"])
    parents = contributing_parents(cls, trio)

    def dp_cap(sample: str, factor: float) -> float:
        md = mean_dp.get(sample) or 0
        # Hemizygous regions have half the diploid depth; cap relative to the autosomal mean.
        return factor * md if md else math.inf

    if c.track == "germline" or c.track == "parental_mosaic":
        if (kid.gq or 0) < pb["min_gq"]:
            out.append(f"proband_GQ<{pb['min_gq']}")
        if c.ploidy == 2:
            v = kid.vaf or 0
            if not (pb["het_vaf_min"] <= v <= pb["het_vaf_max"]):
                out.append("proband_VAF_out_of_het_range")
        if kid.alt_depth < pb["min_alt"]:
            out.append(f"proband_alt<{pb['min_alt']}")
    else:  # mosaic
        if (kid.vaf or 0) < mo["vaf_min"]:
            out.append(f"proband_VAF<{mo['vaf_min']}")
        if kid.alt_depth < mo["min_alt"]:
            out.append(f"proband_alt<{mo['min_alt']}")
    if kid.dp < pb["min_dp"]:
        out.append(f"proband_DP<{pb['min_dp']}")
    if kid.dp > dp_cap(trio.proband, pb["max_dp_factor"]):
        out.append("proband_DP>cap")

    for p in parents:
        g = rec.samples[p]
        role = "father" if p == trio.father else "mother"
        if not g.called:
            out.append(f"{role}_no_call")
            continue
        if (g.gq or 0) < pa["min_gq"]:
            out.append(f"{role}_GQ<{pa['min_gq']}")
        min_dp = mo["parent_min_dp"] if c.track == "mosaic" else pa["min_dp"]
        if g.dp < min_dp:
            out.append(f"{role}_DP<{min_dp}")
        if g.dp > dp_cap(p, pa["max_dp_factor"]):
            out.append(f"{role}_DP>cap")
        if c.track == "parental_mosaic":
            continue  # parental ALT reads are the signal here, not a failure
        max_alt = mo["parent_max_alt"] if c.track == "mosaic" else pa["max_alt"]
        if g.alt_depth > max_alt:
            out.append(f"{role}_alt>{max_alt}")
        if (g.vaf or 0) > pa["max_vaf"]:
            out.append(f"{role}_VAF>{pa['max_vaf']}")


def layer2_read_quality(c: Candidate, trio: Trio, cfg: dict) -> None:
    rec, out, s = c.rec, c.fails["L2_read_quality"], cfg["site"]
    snv = rec.is_snv or rec.is_mnv

    def chk(key: str, bad) -> None:
        v = rec.info_float(key)
        if v is not None and bad(v):
            out.append(f"{key}={v:g}")

    chk("MQ", lambda v: v < s["min_mq"])
    chk("FS", lambda v: v > (s["max_fs_snv"] if snv else s["max_fs_indel"]))
    chk("SOR", lambda v: v > (s["max_sor_snv"] if snv else s["max_sor_indel"]))
    chk("MQRankSum", lambda v: v < s["min_mqranksum"])
    chk("ReadPosRankSum", lambda v: v < (s["min_readposranksum_snv"] if snv else s["min_readposranksum_indel"]))
    chk(s["softclip_info_key"], lambda v: v > s["max_softclip_frac"])

    kid = rec.samples[trio.proband]
    st = kid.strand_alt()
    if st is not None:
        f, r = st
        if f < s["min_alt_per_strand"] or r < s["min_alt_per_strand"]:
            out.append(f"alt_single_strand({f}F/{r}R)")
        sb = kid.fields.get("SB")
        if sb and sb != ".":
            rf, rr, af_, ar = [int(x) for x in sb.split(",")[:4]]
            p = stats.fisher_exact(rf, rr, af_, ar)
            if p < s["min_strand_fisher_p"]:
                out.append(f"strand_bias_p={p:.1e}")
    else:
        c.flags.append("NO_STRAND_INFO")


def layer3_region(c: Candidate, cfg: dict, beds: List[IntervalSet], fasta: Optional[Fasta]) -> None:
    rec, out, rg = c.rec, c.fails["L3_region"], cfg["regions"]
    end = rec.pos + max(len(rec.ref), 1) - 1
    for b in beds:
        if b.overlaps(rec.chrom, rec.pos, end):
            out.append(f"in_{b.name}")
    if fasta is not None:
        hp = genome.homopolymer_run(fasta, rec.chrom, rec.pos + (1 if not rec.is_snv else 0))
        c.homopolymer = hp
        limit = rg["max_homopolymer_snv"] if rec.is_snv else rg["max_homopolymer_indel"]
        if hp > limit:
            out.append(f"homopolymer_{hp}bp")
        if genome.near_n(fasta, rec.chrom, rec.pos, rg["n_pad"]):
            out.append("near_reference_N")
        if rec.is_snv:
            ctx = fasta.fetch(rec.chrom, rec.pos - 1, rec.pos + 1)
            if len(ctx) == 3:
                c.context = ctx


def layer4_population(c: Candidate, cfg: dict, pon: Dict[str, float], recurrence: Dict[str, int]) -> None:
    rec, out, pp = c.rec, c.fails["L4_population"], cfg["population"]
    af = get_pop_af(rec, pp["af_keys"])
    c.pop_af = af
    plp = is_clinvar_plp(rec, pp["clinvar_keys"])
    if plp:
        c.flags.append("CLINVAR_PLP")
    if af is not None and af >= pp["max_af"]:
        if plp:
            c.flags.append("POP_AF_RESCUED_BY_CLINVAR")
        else:
            out.append(f"gnomAD_AF={af:.2e}")
    k = norm_key(rec.chrom, rec.pos, rec.ref, rec.alt)
    if pon.get(k, 0.0) > pp["pon_max_af"]:
        out.append(f"panel_of_normals_AF={pon[k]:.3f}")
    n = recurrence.get(k, 0)
    if n >= pp["recurrence_max"]:
        if plp:
            c.flags.append(f"RECURRENT_KNOWN_PATHOGENIC(n={n})")
        else:
            out.append(f"recurrent_in_{n}_trios")


def layer5_posterior(c: Candidate, trio: Trio, cfg: dict) -> None:
    rec, pc = c.rec, cfg["posterior"]
    cls = genome.chrom_class(rec.chrom, rec.pos, cfg["build"])
    parents = contributing_parents(cls, trio)

    def ll(sample: str, ploidy: int):
        g: Genotype = rec.samples[sample]
        return stats.log_likelihoods(g.pl, g.ref_depth, g.alt_depth, ploidy, pc["read_error"]), ploidy

    kid = ll(trio.proband, c.ploidy)
    dad = ll(trio.father, genome.expected_ploidy(rec.chrom, rec.pos, "male", cfg["build"])) if trio.father in parents else None
    mom = ll(trio.mother, 2) if trio.mother in parents else None
    mu = pc["mu_snv"] if rec.is_snv else pc["mu_indel"]
    if c.track == "germline":
        post = stats.trio_dnm_posterior(kid, dad, mom, c.pop_af or 0.0, mu, pc["af_floor"])["p_dnm"]
    else:
        # The diploid germline model does not describe low-VAF events; use the
        # read-count model with the observed VAF as the alternative hypothesis.
        post = mosaic_posterior(c, trio, parents, cfg)
    c.posterior = post
    hi = rec.info.get("hiConfDeNovo")
    c.hi_conf = isinstance(hi, str) and trio.proband in hi.split(",")
    if post < pc["min_posterior_multicaller"]:
        c.fails["L5_posterior"].append(f"posterior={post:.3f}")


def mosaic_posterior(c: Candidate, trio: Trio, parents: List[str], cfg: dict) -> float:
    """P(proband ALT reads are real & absent in parents) under a two-class
    read model: true event at observed VAF vs sequencing error."""
    err = cfg["mosaic"]["seq_error"]
    kid = c.rec.samples[trio.proband]
    n, k = kid.dp, kid.alt_depth
    if n == 0:
        return 0.0
    v = max(k / n, err * 2)
    l_true = stats.log_binom_pmf(k, n, v)
    l_err = stats.log_binom_pmf(k, n, err)
    prior_true = 1e-3  # candidate already selected on read evidence
    lt = l_true + math.log(prior_true)
    le = l_err + math.log(1 - prior_true)
    p_kid = math.exp(lt - stats.logsumexp([lt, le]))
    if c.track == "parental_mosaic":
        return p_kid
    p_par = 1.0
    for p in parents:
        g = c.rec.samples[p]
        # Probability parent does NOT carry at the proband's VAF.
        l0 = stats.log_binom_pmf(g.alt_depth, g.dp, err)
        l1 = stats.log_binom_pmf(g.alt_depth, g.dp, v)
        p_par *= math.exp(l0 - stats.logsumexp([l0, l1]))
    return p_kid * p_par


# --------------------------------------------------------------------------- #
# Consensus / tiers / clustering
# --------------------------------------------------------------------------- #
def assign_tier(c: Candidate, n_available: int, cfg: dict) -> str:
    if not c.passed:
        return "FAIL"
    cc, pc = cfg["consensus"], cfg["posterior"]
    n = len(c.callers)
    strong = c.posterior is not None and c.posterior >= pc["min_posterior"]
    if n_available == 0:
        if strong and (c.hi_conf or c.track != "germline"):
            return "HIGH"
        return "MEDIUM" if strong else "LOW"
    need_high = min(cc["high_min_callers"], n_available)
    if n >= need_high and (c.hi_conf or strong):
        return "HIGH"
    if n >= min(cc["medium_min_callers"], n_available):
        return "MEDIUM"
    return "LOW"


def flag_clusters(cands: List[Candidate], window: int) -> None:
    by_chrom: Dict[str, List[Candidate]] = defaultdict(list)
    for c in cands:
        by_chrom[genome.bare_chrom(c.rec.chrom)].append(c)
    for lst in by_chrom.values():
        lst.sort(key=lambda x: x.rec.pos)
        for a, b in zip(lst, lst[1:]):
            if b.rec.pos - a.rec.pos <= window:
                for x in (a, b):
                    if "CLUSTERED_DNM" not in x.flags:
                        x.flags.append("CLUSTERED_DNM")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
INFO_HEADER = [
    '##INFO=<ID=DNM_TRACK,Number=1,Type=String,Description="germline | mosaic (post-zygotic in proband) | parental_mosaic">',
    '##INFO=<ID=DNM_TIER,Number=1,Type=String,Description="HIGH | MEDIUM | LOW consensus tier">',
    '##INFO=<ID=DNM_POSTERIOR,Number=1,Type=Float,Description="Trio Bayesian posterior probability of de novo origin">',
    '##INFO=<ID=N_CALLERS,Number=1,Type=Integer,Description="Number of independent DNM callers reporting the site">',
    '##INFO=<ID=DNM_CALLERS,Number=.,Type=String,Description="Callers reporting the site">',
    '##INFO=<ID=DNM_FLAGS,Number=.,Type=String,Description="Review flags">',
    '##INFO=<ID=DNM_PROBAND,Number=1,Type=String,Description="Proband sample ID">',
    '##INFO=<ID=PROBAND_VAF,Number=1,Type=Float,Description="Proband variant allele fraction">',
    '##INFO=<ID=PROBAND_VAF_CI,Number=2,Type=Float,Description="Wilson 95% CI of proband VAF">',
]

TSV_COLUMNS = [
    "proband", "chrom", "pos", "ref", "alt", "type", "track", "tier", "pass", "first_fail",
    "fail_reasons", "flags", "posterior", "n_callers", "callers", "hiConfDeNovo", "pop_af",
    "proband_gt", "proband_dp", "proband_alt", "proband_vaf", "proband_vaf_ci",
    "father_gt", "father_dp", "father_alt", "mother_gt", "mother_dp", "mother_alt",
    "homopolymer", "context",
]


def candidate_row(c: Candidate, trio: Trio) -> Dict[str, object]:
    r = c.rec
    k, f, m = r.samples[trio.proband], r.samples[trio.father], r.samples[trio.mother]
    reasons = [x for v in c.fails.values() for x in v]
    return {
        "proband": trio.proband, "chrom": r.chrom, "pos": r.pos, "ref": r.ref, "alt": r.alt,
        "type": "SNV" if r.is_snv else ("MNV" if r.is_mnv else ("INS" if r.indel_length > 0 else "DEL")),
        "track": c.track, "tier": c.tier, "pass": c.passed, "first_fail": c.first_fail or "",
        "fail_reasons": ";".join(reasons), "flags": ";".join(c.flags),
        "posterior": "" if c.posterior is None else f"{c.posterior:.4f}",
        "n_callers": len(c.callers), "callers": ",".join(c.callers), "hiConfDeNovo": c.hi_conf,
        "pop_af": "" if c.pop_af is None else f"{c.pop_af:.3g}",
        "proband_gt": k.gt, "proband_dp": k.dp, "proband_alt": k.alt_depth,
        "proband_vaf": f"{(k.vaf or 0):.3f}", "proband_vaf_ci": f"{c.proband_vaf_ci[0]:.3f}-{c.proband_vaf_ci[1]:.3f}",
        "father_gt": f.gt, "father_dp": f.dp, "father_alt": f.alt_depth,
        "mother_gt": m.gt, "mother_dp": m.dp, "mother_alt": m.alt_depth,
        "homopolymer": c.homopolymer, "context": c.context,
    }


def write_tsv(path: str, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with open(path, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")


def run_call(
    vcf: str,
    trios: List[Trio],
    cfg: dict,
    out_prefix: str,
    exclude_beds: Optional[List[str]] = None,
    fasta_path: Optional[str] = None,
    pon_path: Optional[str] = None,
    recurrence_path: Optional[str] = None,
    callers: Optional[Dict[str, str]] = None,
) -> Dict[str, dict]:
    beds = [IntervalSet.from_bed(p) for p in (exclude_beds or []) + cfg["regions"]["exclude_beds"]]
    fasta = Fasta(fasta_path) if fasta_path else None
    pon = load_site_af(pon_path)
    recurrence = load_recurrence(recurrence_path)
    samples = {s for t in trios for s in (t.proband, t.father, t.mother)}
    mean_dp = cfg.get("mean_depth") or estimate_mean_depth(vcf, samples)
    if not isinstance(mean_dp, dict):
        mean_dp = {s: float(mean_dp) for s in samples}

    caller_sites = {name: load_caller_sites(path) for name, path in (callers or {}).items()}
    per_trio: Dict[str, List[Candidate]] = {t.proband: [] for t in trios}
    mie: Dict[str, List[int]] = {t.proband: [0, 0] for t in trios}  # [errors, informative]
    n_sites = 0
    reader = VCFReader(vcf)
    for rec in reader:
        n_sites += 1
        for t in trios:
            _count_mie(rec, t, mie[t.proband])
            c = classify_candidate(rec, t, cfg)
            if c is None:
                continue
            layer1_genotype(c, t, cfg, mean_dp)
            layer2_read_quality(c, t, cfg)
            layer3_region(c, cfg, beds, fasta)
            layer4_population(c, cfg, pon, recurrence)
            layer5_posterior(c, t, cfg)
            k = norm_key(rec.chrom, rec.pos, rec.ref, rec.alt)
            c.callers = sorted(n for n, s in caller_sites.items() if k in s)
            per_trio[t.proband].append(c)

    summaries: Dict[str, dict] = {}
    for t in trios:
        cands = per_trio[t.proband]
        for c in cands:
            c.tier = assign_tier(c, len(caller_sites), cfg)
        passing = [c for c in cands if c.passed]
        flag_clusters(passing, cfg["cluster_window_bp"])
        prefix = f"{out_prefix}.{t.proband}" if len(trios) > 1 else out_prefix
        write_tsv(prefix + ".candidates.tsv", [candidate_row(c, t) for c in cands], TSV_COLUMNS)
        with VCFWriter(prefix + ".dnm.vcf", reader, INFO_HEADER, samples=[t.proband, t.father, t.mother]) as w:
            for c in passing:
                r = c.rec
                kid = r.samples[t.proband]
                r.info.update({
                    "DNM_PROBAND": t.proband, "DNM_TRACK": c.track, "DNM_TIER": c.tier,
                    "DNM_POSTERIOR": f"{c.posterior:.4f}", "N_CALLERS": len(c.callers),
                    "DNM_CALLERS": ",".join(c.callers) or None, "DNM_FLAGS": ",".join(c.flags) or None,
                    "PROBAND_VAF": f"{(kid.vaf or 0):.3f}",
                    "PROBAND_VAF_CI": f"{c.proband_vaf_ci[0]:.3f},{c.proband_vaf_ci[1]:.3f}",
                })
                w.write(r)
        waterfall = OrderedDict()
        waterfall["mendelian_violation_candidates"] = len(cands)
        remaining = len(cands)
        first = Counter(c.first_fail for c in cands)
        for layer in LAYERS:
            remaining -= first.get(layer, 0)
            waterfall[f"after_{layer}"] = remaining
        summary = {
            "proband": t.proband,
            "father": t.father,
            "mother": t.mother,
            "proband_sex": t.proband_sex,
            "sites_scanned": n_sites,
            "mean_depth": mean_dp,
            "callers_available": sorted(caller_sites),
            "waterfall": waterfall,
            "passing_by_track": dict(Counter(c.track for c in passing)),
            "passing_by_tier": dict(Counter(c.tier for c in passing)),
            "parental_leakage": parental_leakage(passing, t),
            "raw_mendelian_error_rate": round(mie[t.proband][0] / mie[t.proband][1], 4) if mie[t.proband][1] else None,
        }
        if summary["raw_mendelian_error_rate"] and summary["raw_mendelian_error_rate"] > 0.05:
            summary["qc_warning"] = "raw Mendelian error rate > 5% – suspect sample swap or contamination"
        with open(prefix + ".call_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        summaries[t.proband] = summary
    return summaries


def _count_mie(rec: Record, t: Trio, acc: List[int]) -> None:
    """Raw autosomal Mendelian inconsistency among sites non-ref in the trio."""
    if genome.chrom_class(rec.chrom, rec.pos) != "auto":
        return
    g = [rec.samples.get(s) for s in (t.proband, t.father, t.mother)]
    if any(x is None or not x.called or x.ploidy != 2 for x in g):
        return
    k, f, m = (x.alt_count for x in g)
    if k == f == m == 0:
        return
    acc[1] += 1
    fa = {0: {0}, 1: {0, 1}, 2: {1}}[f]
    ma = {0: {0}, 1: {0, 1}, 2: {1}}[m]
    if k not in {a + b for a in fa for b in ma}:
        acc[0] += 1


def parental_leakage(passing: List[Candidate], t: Trio) -> Dict[str, float]:
    """Replaces the v1.0 'kinship on DNMs' check: if one parent systematically
    shows ALT reads at the child's germline DNMs, that parent's sample is
    likely contaminated, swapped, or under-sequenced."""
    germ = [c for c in passing if c.track == "germline"]
    if not germ:
        return {"n": 0}
    f = sum(1 for c in germ if c.rec.samples[t.father].alt_depth > 0) / len(germ)
    m = sum(1 for c in germ if c.rec.samples[t.mother].alt_depth > 0) / len(germ)
    return {"n": len(germ), "father_any_alt_frac": round(f, 3), "mother_any_alt_frac": round(m, 3)}
