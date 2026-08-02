"""Shared research context passed between agents.

Every agent reads from and writes to a single ``ResearchContext``. This is the
blackboard the whole system reasons over: the question, the growing causal
evidence graph, the hypothesis pool, candidate experiments, datasets found, and a
running provenance log so any conclusion can be traced back to the agent and
source that produced it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .evidence import CausalEvidenceGraph, Evidence
from .experiment import Experiment
from .hypothesis import HypothesisPool


@dataclass
class Dataset:
    """A public dataset surfaced by the Public Database Agent."""

    accession: str
    repository: str            # GEO / ArrayExpress / ImmPort / HuBMAP / HCA / ...
    modality: str              # scRNA / scATAC / spatial / proteomics / GWAS / ...
    title: str = ""
    relevance: float = 0.5
    used_for: str = ""         # discovery / validation — enforce leakage separation
    url: str = ""


@dataclass
class LogEntry:
    agent: str
    layer: int
    message: str
    t: float


@dataclass
class ResearchContext:
    question: str
    focus: str = ""            # e.g. "IRF5", "lupus nephritis tissue damage"
    graph: CausalEvidenceGraph = field(default_factory=CausalEvidenceGraph)
    pool: Optional[HypothesisPool] = None
    datasets: list[Dataset] = field(default_factory=list)
    experiments: list[Experiment] = field(default_factory=list)
    log: list[LogEntry] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    # Central registry of every evidence item by id, so ranking / negative-evidence
    # / causal-graph agents can operate on a shared pool and cross-reference it.
    evidence: dict[str, Evidence] = field(default_factory=dict)
    # Which datasets were used to *generate* hypotheses; validation must avoid these.
    discovery_datasets: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.pool is None:
            self.pool = HypothesisPool(self.graph)

    def record(self, agent: str, layer: int, message: str) -> None:
        self.log.append(LogEntry(agent=agent, layer=layer, message=message, t=time.time()))

    def register_evidence(self, ev: Evidence) -> Evidence:
        """Add an evidence item to the shared pool (idempotent by id)."""
        self.evidence[ev.id] = ev
        return ev

    def add_dataset(self, ds: Dataset) -> None:
        self.datasets.append(ds)
        if ds.used_for == "discovery":
            self.discovery_datasets.add(ds.accession)

    def validation_datasets(self) -> list[Dataset]:
        """Datasets safe to validate on: never used in discovery (no leakage)."""
        return [d for d in self.datasets if d.accession not in self.discovery_datasets]

    def add_experiment(self, exp: Experiment) -> None:
        self.experiments.append(exp)

    def store(self, key: str, value: Any) -> None:
        self.artifacts[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.artifacts.get(key, default)
