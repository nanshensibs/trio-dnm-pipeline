"""Helpers to coerce loosely-typed LLM JSON into the strict domain enums."""

from __future__ import annotations

from typing import Any

from .evidence import EvidenceTier, NodeType, Polarity, Sign
from .experiment import ExperimentType

_TIER = {
    "human_genetics": EvidenceTier.HUMAN_GENETICS,
    "human genetics": EvidenceTier.HUMAN_GENETICS,
    "genetics": EvidenceTier.HUMAN_GENETICS,
    "perturbation": EvidenceTier.PERTURBATION,
    "crispr": EvidenceTier.PERTURBATION,
    "perturb-seq": EvidenceTier.PERTURBATION,
    "animal": EvidenceTier.ANIMAL,
    "mouse": EvidenceTier.ANIMAL,
    "association": EvidenceTier.ASSOCIATION,
    "observational": EvidenceTier.ASSOCIATION,
    "review": EvidenceTier.REVIEW,
    "opinion": EvidenceTier.REVIEW,
}

_ETYPE = {t.value: t for t in ExperimentType}
_ETYPE.update({
    "lof": ExperimentType.LOSS_OF_FUNCTION,
    "gof": ExperimentType.GAIN_OF_FUNCTION,
    "knockout": ExperimentType.LOSS_OF_FUNCTION,
    "overexpression": ExperimentType.GAIN_OF_FUNCTION,
    "crispr": ExperimentType.CRISPR_SCREEN,
    "mouse": ExperimentType.MOUSE_IN_VIVO,
    "human": ExperimentType.HUMAN_TISSUE,
})


def tier(value: Any, default: EvidenceTier = EvidenceTier.ASSOCIATION) -> EvidenceTier:
    if isinstance(value, EvidenceTier):
        return value
    if value is None:
        return default
    return _TIER.get(str(value).strip().lower(), default)


def experiment_type(value: Any, default: ExperimentType = ExperimentType.HUMAN_TISSUE) -> ExperimentType:
    if isinstance(value, ExperimentType):
        return value
    if value is None:
        return default
    return _ETYPE.get(str(value).strip().lower(), default)


def polarity(value: Any, default: Polarity = Polarity.SUPPORTS) -> Polarity:
    if isinstance(value, Polarity):
        return value
    s = str(value).strip().lower()
    if s in ("supports", "support", "positive", "confirm", "supported"):
        return Polarity.SUPPORTS
    if s in ("contradicts", "contradict", "negative", "refute", "refuted", "null"):
        return Polarity.CONTRADICTS
    if s in ("neutral", "inconclusive"):
        return Polarity.NEUTRAL
    return default


def sign(value: Any, default: Sign = Sign.UNKNOWN) -> Sign:
    if isinstance(value, Sign):
        return value
    s = str(value).strip().lower()
    if s in ("activates", "activate", "up", "positive", "risk", "+"):
        return Sign.ACTIVATES
    if s in ("inhibits", "inhibit", "down", "negative", "protective", "-"):
        return Sign.INHIBITS
    return default


def node_type(value: Any, default: NodeType = NodeType.MOLECULAR_MECHANISM) -> NodeType:
    if isinstance(value, NodeType):
        return value
    s = str(value).strip().lower()
    for nt in NodeType:
        if nt.value == s or nt.name.lower() == s:
            return nt
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("true", "yes", "1", "y")
