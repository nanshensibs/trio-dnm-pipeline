"""Default thresholds. Every value can be overridden with a JSON file
(``--config my.json``) whose keys are deep-merged over these defaults; see
``conf/defaults.json`` for the same structure written out in full. Put only
the keys you change in ``--config`` (see ``load_config`` for how a full copy
of the defaults interacts with the data-type presets).

Defaults are tuned for 30x PCR-free WGS. ``DATA_TYPE_PRESETS`` holds the
adjustments applied when ``--data-type wes`` or ``panel`` is given.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "build": "GRCh38",
    "data_type": "wgs",
    "mean_depth": None,  # per-sample; estimated from the VCF when null
    "proband": {
        "min_gq": 20,
        "min_dp": 10,
        "max_dp_factor": 2.0,
        "min_alt": 5,
        "het_vaf_min": 0.30,
        "het_vaf_max": 0.70,
        "hemi_vaf_min": 0.85,
    },
    "parent": {
        "min_gq": 20,
        "min_dp": 15,
        "max_dp_factor": 2.0,
        "max_alt": 1,
        "max_vaf": 0.05,
        # A hemizygous parent (father on chrX non-PAR for a female proband, or on
        # chrY) has half the diploid depth and would show VAF ~1 if a carrier.
        "min_dp_hemizygous": 8,
    },
    "site": {
        "min_mq": 40.0,
        "max_fs_snv": 60.0,
        "max_fs_indel": 200.0,
        "max_sor_snv": 3.0,
        "max_sor_indel": 10.0,
        "min_mqranksum": -12.5,
        "min_readposranksum_snv": -8.0,
        "min_readposranksum_indel": -20.0,
        "min_alt_per_strand": 1,
        "min_strand_fisher_p": 1e-3,
        "max_softclip_frac": 0.30,
        "softclip_info_key": "SCF",
        "honour_filter": True,  # FILTER other than PASS/. fails Layer 2 (site_filter:<FILTER>)
    },
    "regions": {
        "exclude_beds": [],
        "max_homopolymer_snv": 7,  # run >= 8 bp excluded for SNVs
        "max_homopolymer_indel": 5,  # run >= 6 bp excluded for indels
        "n_pad": 5,
        "exclude_mt": True,
    },
    "population": {
        "af_keys": ["gnomAD_AF", "gnomADg_AF", "gnomADe_AF", "gnomad_AF", "AF_gnomad", "gnomAD_joint_AF"],
        "max_af": 1e-4,
        "pon_max_af": 0.01,
        "pon_info_key": "PON_AF",  # vcfanno-added panel-of-normals AF (conf/vcfanno.toml)
        "recurrence_max": 5,
        "clinvar_keys": ["CLNSIG", "ClinVar_CLNSIG"],
    },
    "posterior": {
        "mu_snv": 1.2e-8,
        "mu_indel": 1.0e-9,
        "af_floor": 1e-4,
        "min_posterior": 0.95,
        "min_posterior_multicaller": 0.80,
        "read_error": 0.01,
    },
    "mosaic": {
        "enabled": True,
        "vaf_min": 0.05,
        "min_alt": 3,
        "parent_min_dp": 30,
        "parent_max_alt": 0,
        "het_binom_p": 1e-3,
        "max_vaf": 0.40,  # route to the mosaic track only below this VAF (and binomial p < het_binom_p)
        "prior_true": 1e-3,  # mosaic read-model prior that the ALT reads are real
        "seq_error": 0.005,
        "parental_mosaic_min_alt": 2,
        "parental_mosaic_max_vaf": 0.25,
        "parental_mosaic_p": 1e-4,
    },
    "consensus": {
        "high_min_callers": 3,
        "medium_min_callers": 2,
    },
    "cluster_window_bp": 20000,
    # Stage-1 stop-gates (`trio-dnm qc-gate`) and the post-calling Mendelian gate.
    "qc": {
        "freemix_warn": 0.02,
        "freemix_fail": 0.05,
        "parent_child_relatedness": [0.40, 0.60],  # somalier 'relatedness' (~ 2 x kinship)
        "parent_child_max_ibs0": 0.005,  # fraction of informative sites
        "unrelated_max_relatedness": 0.10,  # parents, unless consanguinity is declared
        "min_mean_depth": {"wgs": 28.0, "wes": 80.0, "panel": 200.0},
        "male_y_ratio": 0.10,  # Y depth / autosomal depth above which a sample is male
        "male_max_x_het": 0.05,  # X_het / X_n below which a Y-positive sample is male
        # `trio-dnm call` halts (exit 3) above this raw autosomal Mendelian
        # error rate once enough informative sites have been seen.
        "mie_max_rate": 0.05,
        "mie_min_sites": 1000,
    },
    # Biological sanity-check expectations (per trio).
    "expectations": {
        "wgs": {"snv": [45, 95], "indel": [3, 12], "hard_low": 30, "hard_high": 150},
        "wes": {"snv": [0.5, 2.5], "indel": [0, 1], "hard_low": 0, "hard_high": 5},
        "panel": {"snv": [0, 1], "indel": [0, 1], "hard_low": 0, "hard_high": 2},
        "titv": {"wgs": [1.7, 2.5], "wes": [2.2, 3.5]},
        "paternal_fraction": [0.65, 0.90],
        "cpg_fraction_snv": [0.10, 0.25],
        # Jonsson et al. 2017: +1.51 DNMs per paternal year, +0.37 per maternal
        # year; intercept is callable-genome dependent — calibrate on your
        # own cohort before using the age-adjusted expectation quantitatively.
        "age_model": {"intercept": 6.0, "paternal": 1.51, "maternal": 0.37},
    },
    "annotation": {
        # Single pre-selected PP3/BP4 tool per ClinGen SVI (Pejaver 2022).
        "pp3_tool": "REVEL",
        "calibration": {
            # thresholds: [supporting, moderate, strong, very_strong] (None = not reached)
            "REVEL": {"pp3": [0.644, 0.773, 0.932, None], "bp4": [0.290, 0.183, 0.052, 0.003]},
            "CADD_PHRED": {"pp3": [25.3, 28.1, None, None], "bp4": [22.7, 17.3, None, None]},
            "BayesDel_noAF": {"pp3": [0.13, 0.27, 0.41, None], "bp4": [-0.18, -0.36, None, None]},
            # ClinGen calibration of AlphaMissense (Bergquist et al. 2025) —
            # verify against the published table before clinical use.
            "AlphaMissense": {"pp3": [0.792, 0.906, 0.990, None], "bp4": [0.169, 0.099, None, None]},
        },
        "spliceai_pp3": 0.20,
        "spliceai_bp4": 0.10,
        "spliceai_high": 0.50,
        "pm2_max_af": 1e-5,
        "ba1_af": 0.05,
        "bs1_af": 1e-3,
        "loeuf_constrained": 0.6,  # gnomAD v4 recommendation
        "pli_constrained": 0.9,
        "misz_constrained": 3.09,
        "shet_constrained": 0.1,
        "lof_consequences": [
            "stop_gained",
            "frameshift_variant",
            "splice_donor_variant",
            "splice_acceptor_variant",
            "start_lost",
            "transcript_ablation",
        ],
        "missense_consequences": ["missense_variant", "inframe_insertion", "inframe_deletion", "protein_altering_variant"],
        "de_novo_code": "auto",  # PS2 when parentage confirmed + validated, else PM6
    },
    "somatic": {
        "tumor_min_dp": 20,
        "tumor_min_alt": 4,
        "tumor_min_vaf": 0.05,
        "normal_min_dp": 10,
        "normal_max_alt": 1,
        "normal_max_vaf": 0.02,
        "max_pop_af": 1e-3,
        "min_callers": 2,
        "ffpe_max_vaf": 0.10,
        "clonal_ccf": 0.8,
        "coding_mb": 30.0,  # callable coding territory used for TMB (Mb)
        "tmb_high": 10.0,
    },
    "ch": {
        "min_vaf": 0.02,
        "max_vaf": 0.35,
        "min_alt": 3,
        "genes": [
            "DNMT3A", "TET2", "ASXL1", "PPM1D", "TP53", "JAK2", "SF3B1", "SRSF2",
            "U2AF1", "ZRSR2", "CBL", "GNB1", "GNAS", "IDH1", "IDH2", "BCOR",
            "BCORL1", "STAG2", "KRAS", "NRAS", "CHEK2", "RAD21", "BRCC3", "CALR",
            "MPL", "SH2B3", "KMT2D", "EZH2", "RUNX1", "PHF6", "DUSP22",
        ],
    },
}

DATA_TYPE_PRESETS: Dict[str, Dict[str, Any]] = {
    "wgs": {},
    # Capture data is deeper but less uniform: require more read support, and
    # deeper parents before asserting absence at low VAF.
    "wes": {"proband": {"min_dp": 20, "min_alt": 7}, "parent": {"min_dp": 20, "min_dp_hemizygous": 10},
            "mosaic": {"parent_min_dp": 50}},
    "panel": {"proband": {"min_dp": 50, "min_alt": 10}, "parent": {"min_dp": 50, "min_dp_hemizygous": 25},
              "mosaic": {"vaf_min": 0.02, "parent_min_dp": 100}},
}


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def diff_from(over: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    """The part of ``over`` whose values differ from ``base`` (recursively)."""
    out: Dict[str, Any] = {}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            sub = diff_from(v, base[k])
            if sub:
                out[k] = sub
        elif k not in base or base[k] != v:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[str] = None, data_type: Optional[str] = None, **overrides) -> Dict[str, Any]:
    """Effective configuration: ``DEFAULTS`` -> data-type preset -> user file -> overrides.

    The data type is the explicit ``data_type`` argument (``--data-type``) if
    given, otherwise the file's ``data_type``, otherwise ``wgs``; an explicit
    ``--data-type`` always wins over the file.

    Only the user-file values that *differ from* ``DEFAULTS`` are applied over
    the preset. A full copy of ``conf/defaults.json`` with a few edits therefore
    keeps the WES/panel preset for every key it did not change, instead of
    silently reverting it to the WGS defaults. The flip side: a value equal to
    the WGS default cannot override a preset; use a different value (or
    ``--data-type wgs``). Keyword ``overrides`` (e.g. ``build``) are set last
    when not None.
    """
    cfg = copy.deepcopy(DEFAULTS)
    user: Dict[str, Any] = {}
    if path:
        with open(path) as fh:
            user = json.load(fh)
    dt = data_type or user.get("data_type") or cfg["data_type"]
    cfg = deep_merge(cfg, DATA_TYPE_PRESETS.get(dt, {}))
    cfg = deep_merge(cfg, diff_from(user, DEFAULTS))
    cfg["data_type"] = dt
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg
