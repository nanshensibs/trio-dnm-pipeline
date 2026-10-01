"""Stage 4-7: candidate extraction, layered filter cascade, Bayesian trio
posterior, mosaic / parental-mosaic tracks and multi-caller consensus.

Every candidate Mendelian violation is evaluated against *all* layers so the
TSV records every reason it failed; the waterfall counts each candidate once,
at the first layer it fails.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from . import genome, stats
from .genome import Fasta, IntervalSet, Trio
from .vcf import (Genotype, Record, VCFReader, VCFWriter, int_list, meta_id, open_text, reject_bcf,
                  require_samples)

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
    mosaic_parents: List[str] = field(default_factory=list)  # parental_mosaic track: the ALT-carrying parent(s)

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


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


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


def _is_vcf(path: str) -> bool:
    if path.endswith((".vcf", ".vcf.gz", ".vcf.bgz")):
        return True
    try:
        with open_text(path) as fh:
            return fh.readline().startswith("##fileformat=VCF")
    except (OSError, UnicodeDecodeError):
        return False


def load_caller_sites(path: str, proband: Optional[str] = None) -> set:
    """Sites reported as DNMs by an external caller (VCF or chrom/pos/ref/alt TSV).

    ``path`` may be a comma-separated list (e.g. Strelka2 SNV + indel VCFs).
    ``.vcf``, ``.vcf.gz`` and ``.vcf.bgz`` (or any file starting with
    ``##fileformat=VCF``) are read as VCF; BCF must be converted first.

    For VCFs, records with FILTER not PASS/. are ignored; if the VCF has a
    proband column and proband is given, the proband must carry the ALT
    (a warning is printed when the VCF has samples but not the proband).
    """
    if "," in path:
        return set().union(*(load_caller_sites(p, proband) for p in path.split(",")))
    reject_bcf(path)
    sites = set()
    if _is_vcf(path):
        reader = VCFReader(path)
        if proband and reader.samples and proband not in reader.samples:
            warn(f"caller file {path}: proband {proband} is not among its samples "
                 f"({', '.join(reader.samples)}); the proband-carries-ALT check is not applied")
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


def has_pop_af(rec: Record, keys: List[str]) -> bool:
    return any(k in rec.info for k in keys)


def _trim_suffix(ref: str, alt: str) -> Tuple[str, str]:
    """Drop the shared trailing bases (bcftools mpileup writes indels with the
    whole repeat in REF/ALT)."""
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    return ref, alt


class StrandPileup:
    """Per-sample strand counts from a ``bcftools mpileup -a AD,ADF,ADR`` VCF.

    GenotypeGVCFs drops FORMAT/SB and HaplotypeCaller writes no ADF/ADR, so
    the joint trio VCF usually carries no per-sample strand evidence; this
    pileup of the candidate sites supplies it to Layer 2. Records are matched
    on chrom/pos and ALT allele (``<*>`` and other symbolic alleles skipped).
    """

    def __init__(self, path: str):
        reader = VCFReader(path)
        self.path = path
        self.samples = reader.samples
        self.sites: Dict[Tuple[str, int], List[Record]] = defaultdict(list)
        for r in reader:
            self.sites[(genome.bare_chrom(r.chrom), r.pos)].append(r)

    def counts(self, rec: Record, sample: str) -> Optional[Tuple[int, int, int, int]]:
        """(ref_fwd, ref_rev, alt_fwd, alt_rev) for ``rec``'s ALT, or None."""
        want = _trim_suffix(rec.ref, rec.alt)
        for r in self.sites.get((genome.bare_chrom(rec.chrom), rec.pos), []):
            g = r.samples.get(sample)
            if g is None:
                continue
            adf, adr = int_list(g.fields.get("ADF")), int_list(g.fields.get("ADR"))
            for i, a in enumerate([r.alt] + r.extra_alts, start=1):
                if a.startswith("<") or a in ("*", "."):
                    continue
                if (a == rec.alt and r.ref == rec.ref) or _trim_suffix(r.ref, a) == want:
                    if len(adf) > i and len(adr) > i:
                        return adf[0], adr[0], adf[i], adr[i]
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
    # Binomial tests and CIs use n = sum(AD), the denominator of the VAF.
    kid_alt, kid_n = kid.alt_depth, kid.ad_total
    kid_has_alt = (kid.called and (kid.alt_count or 0) > 0) or (m["enabled"] and kid_alt >= m["min_alt"])
    if not kid_has_alt:
        return None

    def parental_mosaic(g: Genotype) -> bool:
        """Significant low-level ALT reads (vs sequencing error) in a parent."""
        return (
            g.alt_depth >= m["parental_mosaic_min_alt"]
            and (g.vaf or 0.0) <= m["parental_mosaic_max_vaf"]
            and stats.binom_sf(g.alt_depth, g.ad_total, m["seq_error"]) < m["parental_mosaic_p"]
        )

    # Parents must not be *called* carriers - unless a parent called 0/1 has
    # ALT reads significantly below heterozygous, i.e. a mosaic parent that
    # the joint caller genotyped as het.
    mosaic_parents: List[str] = []
    flags: List[str] = []
    for p in contributing:
        g = rec.samples[p]
        if g.called and (g.alt_count or 0) > 0:
            mosaic_het = (
                g.ploidy == 2 and g.alt_count == 1 and parental_mosaic(g)
                and stats.binom_cdf(g.alt_depth, g.ad_total, 0.5) < m["het_binom_p"]
            )
            if not mosaic_het:
                return None  # inherited
            mosaic_parents.append(p)
            flags.append("PARENT_CALLED_HET")

    track = "germline"
    vaf = kid.vaf or 0.0
    if ploidy == 2 and kid_n > 0:
        # One-sided binomial vs 0.5: only a significant deficit routes to the
        # mosaic track; a low-VAF but non-significant candidate stays germline
        # (and fails the het VAF range in Layer 1).
        p_low = stats.binom_cdf(kid_alt, kid_n, 0.5)
        if p_low < m["het_binom_p"] and vaf < m["max_vaf"]:
            track = "mosaic"
        elif vaf < cfg["proband"]["het_vaf_min"]:
            flags.append("POSSIBLE_MOSAIC_NEEDS_DEPTH")
    elif ploidy == 1 and kid_n > 0 and vaf < cfg["proband"]["hemi_vaf_min"]:
        # A het-like VAF on a hemizygous chromosome suggests aneuploidy (e.g. XXY) or a
        # paralog artefact rather than mosaicism: route only clearly sub-clonal events.
        if stats.binom_cdf(kid_alt, kid_n, 1 - m["seq_error"]) < m["het_binom_p"] and vaf < m["max_vaf"]:
            track = "mosaic"
        elif m["max_vaf"] <= vaf <= cfg["proband"]["het_vaf_max"]:
            flags.append("HEMIZYGOUS_HET_LIKE")
        else:
            flags.append("POSSIBLE_MOSAIC_NEEDS_DEPTH")

    # Parental mosaicism: significant low-level ALT reads in a parent.
    for p in contributing:
        if p not in mosaic_parents and parental_mosaic(rec.samples[p]):
            mosaic_parents.append(p)
    if mosaic_parents:
        track = "parental_mosaic"
    if track == "mosaic" and not m["enabled"]:
        return None
    c = Candidate(rec=rec, track=track, ploidy=ploidy, flags=flags, mosaic_parents=mosaic_parents)
    c.proband_vaf_ci = stats.wilson_interval(kid_alt, kid_n)
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
def parent_ploidy(rec: Record, trio: Trio, sample: str, build: str) -> int:
    return genome.expected_ploidy(rec.chrom, rec.pos, "male" if sample == trio.father else "female", build)


def layer1_genotype(c: Candidate, trio: Trio, cfg: dict, mean_dp: Dict[str, float]) -> None:
    rec, out = c.rec, c.fails["L1_genotype"]
    pb, pa, mo = cfg["proband"], cfg["parent"], cfg["mosaic"]
    kid = rec.samples[trio.proband]
    cls = genome.chrom_class(rec.chrom, rec.pos, cfg["build"])
    parents = contributing_parents(cls, trio)

    def dp_cap(sample: str, factor: float, ploidy: int) -> float:
        md = mean_dp.get(sample) or 0
        # Hemizygous regions have half the diploid depth; cap relative to the autosomal mean.
        return factor * md * ploidy / 2 if md else math.inf

    if c.track == "germline" or c.track == "parental_mosaic":
        if (kid.gq or 0) < pb["min_gq"]:
            out.append(f"proband_GQ<{pb['min_gq']}")
        v = kid.vaf or 0
        if c.ploidy == 2:
            if not (pb["het_vaf_min"] <= v <= pb["het_vaf_max"]):
                out.append("proband_VAF_out_of_het_range")
        elif v < pb["hemi_vaf_min"]:
            out.append(f"proband_VAF<{pb['hemi_vaf_min']}")
        if kid.alt_depth < pb["min_alt"]:
            out.append(f"proband_alt<{pb['min_alt']}")
    else:  # mosaic
        if (kid.vaf or 0) < mo["vaf_min"]:
            out.append(f"proband_VAF<{mo['vaf_min']}")
        if kid.alt_depth < mo["min_alt"]:
            out.append(f"proband_alt<{mo['min_alt']}")
    if kid.dp < pb["min_dp"]:
        out.append(f"proband_DP<{pb['min_dp']}")
    if kid.dp > dp_cap(trio.proband, pb["max_dp_factor"], c.ploidy):
        out.append("proband_DP>cap")

    for p in parents:
        g = rec.samples[p]
        role = "father" if p == trio.father else "mother"
        if not g.called:
            out.append(f"{role}_no_call")
            continue
        if (g.gq or 0) < pa["min_gq"]:
            out.append(f"{role}_GQ<{pa['min_gq']}")
        # A hemizygous parent (father on chrX non-PAR / chrY) has half the depth.
        ploidy = parent_ploidy(rec, trio, p, cfg["build"])
        if c.track == "mosaic":
            min_dp = mo["parent_min_dp"] if ploidy == 2 else math.ceil(mo["parent_min_dp"] / 2)
        else:
            min_dp = pa["min_dp"] if ploidy == 2 else pa["min_dp_hemizygous"]
        if g.dp < min_dp:
            out.append(f"{role}_DP<{min_dp}")
        if g.dp > dp_cap(p, pa["max_dp_factor"], ploidy):
            out.append(f"{role}_DP>cap")
        if c.track == "parental_mosaic" and p in c.mosaic_parents:
            continue  # parental ALT reads are the signal here, not a failure
        max_alt = mo["parent_max_alt"] if c.track == "mosaic" else pa["max_alt"]
        if g.alt_depth > max_alt:
            out.append(f"{role}_alt>{max_alt}")
        if (g.vaf or 0) > pa["max_vaf"]:
            out.append(f"{role}_VAF>{pa['max_vaf']}")


def layer2_read_quality(c: Candidate, trio: Trio, cfg: dict, strand: Optional[StrandPileup] = None) -> None:
    rec, out, s = c.rec, c.fails["L2_read_quality"], cfg["site"]
    snv = rec.is_snv or rec.is_mnv

    # GATK hard filters / VQSR set in the joint VCF (QD, QUAL, ...).
    if s["honour_filter"] and rec.filter not in ("PASS", "."):
        out.append(f"site_filter:{rec.filter}")

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
    table = None  # refF, refR, altF, altR for the Fisher strand-bias test
    sb = int_list(kid.fields.get("SB"))
    if st is not None and len(sb) >= 4:
        table = tuple(sb[:4])
    if st is None and strand is not None:
        # The joint VCF has no per-sample strand counts: use the candidate pileup.
        table = strand.counts(rec, trio.proband)
        if table is not None:
            st = table[2], table[3]
    if st is not None:
        f, r = st
        if f < s["min_alt_per_strand"] or r < s["min_alt_per_strand"]:
            out.append(f"alt_single_strand({f}F/{r}R)")
        if table is not None:
            p = stats.fisher_exact(*table)
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
    # Panel of normals: --pon VCF and/or the vcfanno-added INFO key (PON_AF).
    pon_afs = [v for v in (pon.get(k), rec.info_float(pp["pon_info_key"]) if pp["pon_info_key"] else None)
               if v is not None]
    if pon_afs and max(pon_afs) > pp["pon_max_af"]:
        out.append(f"panel_of_normals_AF={max(pon_afs):.3f}")
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
    n, k = kid.ad_total, kid.alt_depth
    if n == 0:
        return 0.0
    v = max(k / n, err * 2)
    l_true = stats.log_binom_pmf(k, n, v)
    l_err = stats.log_binom_pmf(k, n, err)
    prior_true = cfg["mosaic"]["prior_true"]  # candidate already selected on read evidence
    lt = l_true + math.log(prior_true)
    le = l_err + math.log(1 - prior_true)
    p_kid = math.exp(lt - stats.logsumexp([lt, le]))
    if c.track == "parental_mosaic":
        return p_kid
    p_par = 1.0
    for p in parents:
        g = c.rec.samples[p]
        # Probability parent does NOT carry at the proband's VAF.
        l0 = stats.log_binom_pmf(g.alt_depth, g.ad_total, err)
        l1 = stats.log_binom_pmf(g.alt_depth, g.ad_total, v)
        p_par *= math.exp(l0 - stats.logsumexp([l0, l1]))
    return p_kid * p_par


# --------------------------------------------------------------------------- #
# Consensus / tiers / clustering
# --------------------------------------------------------------------------- #
def assign_tier(c: Candidate, n_available: int, cfg: dict) -> str:
    """Consensus tier of a passing candidate; ``n_available`` = k external callers.

    k == 0: HIGH = posterior >= min_posterior AND hiConfDeNovo (germline only;
    mosaic tracks are capped at MEDIUM without an independent caller);
    MEDIUM = posterior >= min_posterior OR hiConfDeNovo; LOW otherwise.
    k >= 1: HIGH = N_CALLERS >= min(high_min_callers, k) AND (hiConfDeNovo OR
    posterior >= min_posterior; non-germline tracks need the posterior);
    MEDIUM = N_CALLERS >= min(medium_min_callers, k); LOW otherwise.
    """
    if not c.passed:
        return "FAIL"
    cc, pc = cfg["consensus"], cfg["posterior"]
    n = len(c.callers)
    strong = c.posterior is not None and c.posterior >= pc["min_posterior"]
    germline = c.track == "germline"
    if n_available == 0:
        if strong and c.hi_conf and germline:
            return "HIGH"
        return "MEDIUM" if (strong or c.hi_conf) else "LOW"
    need_high = min(cc["high_min_callers"], n_available)
    if n >= need_high and (strong or (c.hi_conf and germline)):
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
    extra_vcfs: Optional[List[str]] = None,
    strand_vcf: Optional[str] = None,
) -> Dict[str, dict]:
    """Run the cascade on ``vcf`` for every trio and write the outputs.

    ``extra_vcfs``: supplementary engine VCFs (e.g. normalised DeepTrio/GLnexus,
    same sample IDs); their records absent from ``vcf`` go through the same
    cascade flagged SECOND_ENGINE_ONLY. ``strand_vcf``: bcftools mpileup VCF
    with FORMAT/ADF,ADR at candidate sites, used by Layer 2 when the primary
    record has no per-sample strand counts.
    """
    reader = VCFReader(vcf)
    members = [s for t in trios for s in (t.proband, t.father, t.mother)]
    require_samples(reader, members, "PED trio member(s)")
    beds = [IntervalSet.from_bed(p) for p in (exclude_beds or []) + cfg["regions"]["exclude_beds"]]
    fasta = Fasta(fasta_path) if fasta_path else None
    pon = load_site_af(pon_path)
    recurrence = load_recurrence(recurrence_path)
    strand = StrandPileup(strand_vcf) if strand_vcf else None
    if strand is not None:
        absent = [t.proband for t in trios if t.proband not in strand.samples]
        if absent:
            warn(f"strand VCF {strand_vcf} has no sample(s) {', '.join(absent)}; strand checks rely on the primary VCF")
    samples = set(members)
    mean_dp = cfg.get("mean_depth") or estimate_mean_depth(vcf, samples)
    if not isinstance(mean_dp, dict):
        mean_dp = {s: float(mean_dp) for s in samples}

    # External callers, per trio so that a caller VCF's proband column is honoured.
    callers = callers or {}
    caller_sites: Dict[str, Dict[str, set]] = {}
    for t in trios:
        caller_sites[t.proband] = {name: load_caller_sites(path, t.proband) for name, path in callers.items()}
        for name, sites in caller_sites[t.proband].items():
            if not sites:
                warn(f"caller {name} ({callers[name]}) yielded 0 sites for {t.proband}; check the file, "
                     f"its FILTER column and sample names")

    # Supplementary engine: keep only its Mendelian-violation candidates; drop
    # those the primary VCF also contains while scanning it below.
    extras: Dict[str, Record] = OrderedDict()
    extra_meta: List[str] = []
    af_keys = cfg["population"]["af_keys"]
    for path in extra_vcfs or []:
        er = VCFReader(path)
        require_samples(er, members, "PED trio member(s)")
        extra_meta += [m for m in er.meta if m.startswith(("##INFO=", "##FORMAT=", "##FILTER="))]
        af_seen = False
        for rec in er:
            af_seen = af_seen or has_pop_af(rec, af_keys)
            k = norm_key(rec.chrom, rec.pos, rec.ref, rec.alt)
            if k not in extras and any(classify_candidate(rec, t, cfg) is not None for t in trios):
                extras[k] = rec
        if not af_seen:
            warn(f"extra VCF {path} carries none of population.af_keys; the gnomAD filter is inactive for its "
                 f"SECOND_ENGINE_ONLY candidates")

    per_trio: Dict[str, List[Candidate]] = {t.proband: [] for t in trios}
    mie: Dict[str, List[int]] = {t.proband: [0, 0] for t in trios}  # [errors, informative]

    def evaluate(rec: Record, t: Trio, extra: bool = False) -> None:
        c = classify_candidate(rec, t, cfg)
        if c is None:
            return
        if extra:
            c.flags.append("SECOND_ENGINE_ONLY")
        layer1_genotype(c, t, cfg, mean_dp)
        layer2_read_quality(c, t, cfg, strand)
        layer3_region(c, cfg, beds, fasta)
        layer4_population(c, cfg, pon, recurrence)
        layer5_posterior(c, t, cfg)
        k = norm_key(rec.chrom, rec.pos, rec.ref, rec.alt)
        c.callers = sorted(n for n, s in caller_sites[t.proband].items() if k in s)
        per_trio[t.proband].append(c)

    n_sites = 0
    pop_af_seen = False
    chrom_rank: Dict[str, int] = {}
    # extra-VCF key -> probands for which the primary VCF already covers the site
    # (evaluated per trio, so a joint cohort VCF does not hide one trio's second-engine call).
    covered: Dict[str, set] = defaultdict(set)
    for rec in reader:
        n_sites += 1
        pop_af_seen = pop_af_seen or has_pop_af(rec, af_keys)
        chrom_rank.setdefault(genome.bare_chrom(rec.chrom), len(chrom_rank))
        keys = [norm_key(rec.chrom, rec.pos, rec.ref, a) for a in [rec.alt] + rec.extra_alts] if extras else []
        for t in trios:
            _count_mie(rec, t, mie[t.proband])
            before = len(per_trio[t.proband])
            evaluate(rec, t)
            kid = rec.samples.get(t.proband)
            # The primary covers this trio at this site if it made a candidate or calls the proband non-ref.
            if len(per_trio[t.proband]) > before or (kid is not None and kid.called and (kid.alt_count or 0) > 0):
                for k in keys:
                    if k in extras:
                        covered[k].add(t.proband)
    for k, rec in extras.items():
        chrom_rank.setdefault(genome.bare_chrom(rec.chrom), len(chrom_rank))
        for t in trios:
            if t.proband not in covered.get(k, ()):
                evaluate(rec, t, extra=True)
    if not pop_af_seen:
        warn(f"no record in {vcf} carries any of population.af_keys ({', '.join(af_keys)}): the gnomAD "
             f"population filter (Layer 4) was inactive; annotate the VCF with vcfanno first")

    # Declare the supplementary engines' INFO/FORMAT/FILTER keys the primary lacks.
    known = {meta_id(m) for m in reader.meta}
    extra_header = []
    for m in extra_meta:
        if meta_id(m) not in known:
            known.add(meta_id(m))
            extra_header.append(m)

    q = cfg["qc"]
    summaries: Dict[str, dict] = {}
    for t in trios:
        cands = per_trio[t.proband]
        if extras:
            cands.sort(key=lambda c: (chrom_rank[genome.bare_chrom(c.rec.chrom)], c.rec.pos))
        for c in cands:
            c.tier = assign_tier(c, len(callers), cfg)
        passing = [c for c in cands if c.passed]
        flag_clusters(passing, cfg["cluster_window_bp"])
        prefix = f"{out_prefix}.{t.proband}" if len(trios) > 1 else out_prefix
        write_tsv(prefix + ".candidates.tsv", [candidate_row(c, t) for c in cands], TSV_COLUMNS)
        with VCFWriter(prefix + ".dnm.vcf", reader, extra_header + INFO_HEADER, samples=[t.proband, t.father, t.mother]) as w:
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
        n_err, n_inf = mie[t.proband]
        rate = round(n_err / n_inf, 4) if n_inf else None
        summary = {
            "proband": t.proband,
            "father": t.father,
            "mother": t.mother,
            "proband_sex": t.proband_sex,
            "sites_scanned": n_sites,
            "mean_depth": mean_dp,
            "callers_available": sorted(callers),
            "caller_sites": {name: len(s) for name, s in caller_sites[t.proband].items()},
            "extra_vcfs": list(extra_vcfs or []),
            "second_engine_only_candidates": sum(1 for c in cands if "SECOND_ENGINE_ONLY" in c.flags),
            "strand_vcf": strand_vcf,
            "population_af_missing": not pop_af_seen,
            "waterfall": waterfall,
            "passing_by_track": dict(Counter(c.track for c in passing)),
            "passing_by_tier": dict(Counter(c.tier for c in passing)),
            "parental_leakage": parental_leakage(passing, t),
            "raw_mendelian_error_rate": rate,
            "raw_mendelian_informative_sites": n_inf,
            # FAIL halts `trio-dnm call` (exit 3) unless --no-halt; NOT_EVALUATED
            # when fewer than qc.mie_min_sites informative sites were seen.
            "mendelian_error_gate": (
                "NOT_EVALUATED" if rate is None or n_inf < q["mie_min_sites"]
                else ("FAIL" if rate > q["mie_max_rate"] else "PASS")
            ),
        }
        if rate and rate > q["mie_max_rate"]:
            summary["qc_warning"] = (f"raw Mendelian error rate > {q['mie_max_rate']:.0%} – suspect sample swap "
                                     f"or contamination")
        summary["config"] = cfg
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
