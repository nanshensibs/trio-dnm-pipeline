"""Causal evidence model for the Lupus Co-Scientist.

The design commitment of this framework is *causal, not descriptive*. Evidence is
never treated as a flat bag of citations. Instead it is:

1.  **Tiered** by how causal it is (human genetics > perturbation > animal >
    association > review), so a single Perturb-seq result outweighs a hundred
    correlative scRNA papers.
2.  **Signed and directed**, so it can be assembled into a causal chain that runs
    from genetic variant -> perturbation -> molecular mechanism -> cell state ->
    tissue pathology -> clinical outcome.
3.  **Polarity-aware**, so *negative* evidence (null perturbations, failed
    replications, boundary conditions) is a first-class citizen used to prune
    hypotheses rather than being silently discarded.

Everything here is pure-Python and deterministic so the reasoning core can be
unit tested without any network or LLM calls.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Iterable, Iterator, Optional


class EvidenceTier(IntEnum):
    """How causal a piece of evidence is. Higher is stronger.

    This encodes the Evidence Ranking Agent's priority explicitly:
    human genetics -> perturbation -> animal -> association -> review.
    """

    REVIEW = 1          # review / opinion / textbook consensus
    ASSOCIATION = 2     # observational, correlative (bulk/scRNA DE, GWAS hit w/o mechanism)
    ANIMAL = 3          # in vivo mouse phenotype
    PERTURBATION = 4    # CRISPR / Perturb-seq / base-edit / knockdown in relevant system
    HUMAN_GENETICS = 5  # human causal genetics: fine-mapped variant + coloc + functional SNP

    @property
    def weight(self) -> float:
        """A convex weight so the top tiers dominate aggregate scores."""
        return {1: 0.15, 2: 0.35, 3: 0.60, 4: 0.85, 5: 1.00}[int(self)]


class Polarity(Enum):
    """Whether evidence supports, contradicts, or is neutral to a causal link."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class Sign(Enum):
    """Direction of a causal edge: does the source activate or inhibit the target?"""

    ACTIVATES = "activates"
    INHIBITS = "inhibits"
    UNKNOWN = "unknown"


class NodeType(Enum):
    """Node classes in the causal chain, ordered from genotype to clinic."""

    VARIANT = "variant"                    # e.g. NCF1-p.Arg90His, IRF5 rs2004640
    GENE = "gene"                          # e.g. ZEB2, MEF2B, IRF5
    PERTURBATION = "perturbation"          # an experimental handle (KO/OE/base-edit)
    MOLECULAR_MECHANISM = "mechanism"      # e.g. bistable switch flip, ISG induction
    CELL_STATE = "cell_state"              # e.g. ABC, pre-ABC, DN2, GC, plasma
    CELL_PHENOTYPE = "cell_phenotype"      # e.g. autoantibody secretion, survival
    TISSUE_PATHOLOGY = "tissue_pathology"  # e.g. lupus nephritis interstitial damage
    CLINICAL_OUTCOME = "clinical_outcome"  # e.g. renal flare, SLEDAI, drug response
    DRUG = "drug"                          # e.g. anifrolumab, C-CAR168
    BIOMARKER = "biomarker"                # e.g. IFN score, anti-dsDNA


# Canonical distance from genotype (0) to clinic (large). Used to check that a
# causal chain actually *runs downhill* toward a clinical outcome.
_NODE_DEPTH = {
    NodeType.VARIANT: 0,
    NodeType.GENE: 1,
    NodeType.PERTURBATION: 1,
    NodeType.MOLECULAR_MECHANISM: 2,
    NodeType.CELL_STATE: 3,
    NodeType.CELL_PHENOTYPE: 4,
    NodeType.TISSUE_PATHOLOGY: 5,
    NodeType.CLINICAL_OUTCOME: 6,
    NodeType.BIOMARKER: 6,
    NodeType.DRUG: 2,  # drugs act as perturbations on mechanism
}


@dataclass
class Evidence:
    """A single traceable observation.

    Attributes:
        id: stable identifier.
        tier: causal tier (see EvidenceTier).
        polarity: does it support or contradict the link it is attached to.
        summary: one-line human-readable claim.
        source: citation, PMID, dataset accession, or trial id.
        strength: 0..1 confidence in the observation itself (effect size,
            replication, sample size baked in by the producing agent).
        agent: which agent contributed it (provenance).
        datasets: accessions backing it (GEO/ImmPort/etc.), if any.
        is_independent: True if this validation used data independent from the
            data that generated the hypothesis (guards against data leakage).
    """

    id: str
    tier: EvidenceTier
    polarity: Polarity
    summary: str
    source: str = ""
    strength: float = 0.7
    agent: str = ""
    datasets: tuple[str, ...] = ()
    is_independent: bool = True

    @property
    def is_negative(self) -> bool:
        return self.polarity is Polarity.CONTRADICTS

    @property
    def signed_weight(self) -> float:
        """Tier weight * observation strength, signed by polarity.

        Neutral evidence contributes nothing to belief; contradicting evidence
        subtracts. Non-independent evidence is discounted to blunt data leakage.
        """
        magnitude = self.tier.weight * max(0.0, min(1.0, self.strength))
        if not self.is_independent:
            magnitude *= 0.4
        if self.polarity is Polarity.SUPPORTS:
            return magnitude
        if self.polarity is Polarity.CONTRADICTS:
            return -magnitude
        return 0.0


@dataclass
class CausalEdge:
    """A directed, signed causal claim between two nodes, backed by evidence."""

    source: str
    target: str
    sign: Sign = Sign.UNKNOWN
    evidence: list[Evidence] = field(default_factory=list)

    def add(self, ev: Evidence) -> None:
        self.evidence.append(ev)

    @property
    def support(self) -> float:
        """Net tier-weighted support for this edge in [-1, 1]-ish range.

        Uses the strongest *supporting* tier as the backbone and lets negative
        evidence erode it. An edge with only review-level support is weak; an
        edge with a fine-mapped human variant plus a Perturb-seq hit is strong.
        """
        if not self.evidence:
            return 0.0
        pos = [e.signed_weight for e in self.evidence if e.signed_weight > 0]
        neg = [e.signed_weight for e in self.evidence if e.signed_weight < 0]
        # Backbone: best single supporting line, then diminishing returns from the rest.
        pos_sorted = sorted(pos, reverse=True)
        backbone = pos_sorted[0] if pos_sorted else 0.0
        corroboration = sum(w * (0.5 ** (i + 1)) for i, w in enumerate(pos_sorted[1:]))
        erosion = sum(neg)  # already negative
        return backbone + corroboration + erosion

    @property
    def best_supporting_tier(self) -> Optional[EvidenceTier]:
        tiers = [e.tier for e in self.evidence if e.polarity is Polarity.SUPPORTS]
        return max(tiers) if tiers else None

    @property
    def is_contested(self) -> bool:
        has_pos = any(e.polarity is Polarity.SUPPORTS for e in self.evidence)
        has_neg = any(e.polarity is Polarity.CONTRADICTS for e in self.evidence)
        return has_pos and has_neg


@dataclass
class CausalChain:
    """An ordered path of edges from an upstream node to a clinical-ish endpoint."""

    nodes: list[str]
    edges: list[CausalEdge]
    graph: "CausalEvidenceGraph"

    @property
    def weakest_link(self) -> float:
        """A chain is only as causal as its weakest edge."""
        if not self.edges:
            return 0.0
        return min(e.support for e in self.edges)

    @property
    def mean_support(self) -> float:
        if not self.edges:
            return 0.0
        return sum(e.support for e in self.edges) / len(self.edges)

    @property
    def spans_to_clinic(self) -> bool:
        """Does the chain terminate at tissue pathology / clinical outcome / biomarker?"""
        if not self.nodes:
            return False
        end_type = self.graph.node_type(self.nodes[-1])
        return end_type in (
            NodeType.TISSUE_PATHOLOGY,
            NodeType.CLINICAL_OUTCOME,
            NodeType.BIOMARKER,
        )

    @property
    def starts_at_genotype(self) -> bool:
        if not self.nodes:
            return False
        return self.graph.node_type(self.nodes[0]) in (NodeType.VARIANT, NodeType.GENE)

    def score(self) -> float:
        """Overall chain quality: weakest link dominates, bonuses for completeness."""
        base = self.weakest_link
        if base <= 0:
            return base
        completeness = 1.0
        if self.spans_to_clinic:
            completeness += 0.25
        if self.starts_at_genotype:
            completeness += 0.25
        return base * completeness

    def describe(self) -> str:
        parts = []
        for i, node in enumerate(self.nodes):
            parts.append(node)
            if i < len(self.edges):
                sign = self.edges[i].sign
                arrow = {Sign.ACTIVATES: "-->", Sign.INHIBITS: "--|", Sign.UNKNOWN: "-?-"}[sign]
                parts.append(arrow)
        return " ".join(parts)


class CausalEvidenceGraph:
    """A directed multigraph of causal claims.

    This is the spine of the whole system: every hypothesis must be expressible
    as a traceable chain of edges in this graph, and the strength of a hypothesis
    is bounded by the weakest edge in its supporting chain.
    """

    def __init__(self) -> None:
        self._node_types: dict[str, NodeType] = {}
        self._edges: dict[tuple[str, str], CausalEdge] = {}
        self._out: dict[str, list[str]] = defaultdict(list)

    # -- construction -------------------------------------------------------
    def add_node(self, name: str, node_type: NodeType) -> None:
        # Keep the most specific type if re-declared; never silently downgrade.
        self._node_types[name] = node_type

    def node_type(self, name: str) -> Optional[NodeType]:
        return self._node_types.get(name)

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        source_type: Optional[NodeType] = None,
        target_type: Optional[NodeType] = None,
        sign: Sign = Sign.UNKNOWN,
        evidence: Optional[Iterable[Evidence]] = None,
    ) -> CausalEdge:
        if source_type is not None:
            self.add_node(source, source_type)
        if target_type is not None:
            self.add_node(target, target_type)
        self._node_types.setdefault(source, NodeType.MOLECULAR_MECHANISM)
        self._node_types.setdefault(target, NodeType.MOLECULAR_MECHANISM)
        key = (source, target)
        edge = self._edges.get(key)
        if edge is None:
            edge = CausalEdge(source=source, target=target, sign=sign)
            self._edges[key] = edge
            self._out[source].append(target)
        if sign is not Sign.UNKNOWN:
            edge.sign = sign
        for ev in evidence or ():
            edge.add(ev)
        return edge

    # -- inspection ---------------------------------------------------------
    @property
    def nodes(self) -> list[str]:
        return list(self._node_types)

    @property
    def edges(self) -> list[CausalEdge]:
        return list(self._edges.values())

    def edge(self, source: str, target: str) -> Optional[CausalEdge]:
        return self._edges.get((source, target))

    def contested_edges(self) -> list[CausalEdge]:
        """Edges with both supporting and contradicting evidence — the live debates."""
        return [e for e in self._edges.values() if e.is_contested]

    def unsupported_edges(self, threshold: float = 0.2) -> list[CausalEdge]:
        """Edges too weak to carry a hypothesis — targets for new experiments."""
        return [e for e in self._edges.values() if e.support < threshold]

    # -- path finding -------------------------------------------------------
    def chains(self, source: str, target: str, max_depth: int = 8) -> list[CausalChain]:
        """All simple directed paths from source to target, as scored chains."""
        results: list[CausalChain] = []
        stack: list[tuple[str, list[str]]] = [(source, [source])]
        while stack:
            node, path = stack.pop()
            if node == target:
                edges = [self._edges[(path[i], path[i + 1])] for i in range(len(path) - 1)]
                results.append(CausalChain(nodes=path, edges=edges, graph=self))
                continue
            if len(path) > max_depth:
                continue
            for nxt in self._out.get(node, ()):
                if nxt not in path:  # simple paths only
                    stack.append((nxt, path + [nxt]))
        return results

    def best_chain(self, source: str, target: str) -> Optional[CausalChain]:
        chains = self.chains(source, target)
        return max(chains, key=CausalChain.score, default=None)

    def chain_from_nodes(self, nodes: list[str]) -> Optional[CausalChain]:
        """Build the specific chain along an explicit ordered node list.

        Returns None if any consecutive edge is missing (a broken chain). This is
        how each hypothesis reasons over *its own* declared mechanism rather than
        any path that happens to connect the same endpoints.
        """
        if len(nodes) < 2:
            return None
        edges: list[CausalEdge] = []
        for i in range(len(nodes) - 1):
            edge = self._edges.get((nodes[i], nodes[i + 1]))
            if edge is None:
                return None
            edges.append(edge)
        return CausalChain(nodes=list(nodes), edges=edges, graph=self)

    def chains_to_clinic(self, source: str) -> list[CausalChain]:
        """Every chain from `source` that reaches a clinical endpoint."""
        endpoints = [
            n for n, t in self._node_types.items()
            if t in (NodeType.TISSUE_PATHOLOGY, NodeType.CLINICAL_OUTCOME, NodeType.BIOMARKER)
        ]
        chains: list[CausalChain] = []
        for end in endpoints:
            chains.extend(self.chains(source, end))
        return sorted(chains, key=CausalChain.score, reverse=True)

    def contradictions(self) -> list[tuple[str, str, str]]:
        """Detect logical contradictions: an activating and an inhibiting edge
        between the same pair of nodes, both with real support."""
        found = []
        for (src, tgt), edge in self._edges.items():
            reverse = self._edges.get((tgt, src))
            if reverse and edge.sign is Sign.ACTIVATES and reverse.sign is Sign.INHIBITS:
                if edge.support > 0.2 and reverse.support > 0.2:
                    found.append((src, tgt, "bidirectional sign conflict"))
        return found

    def coverage(self) -> dict[str, int]:
        """How many nodes of each type exist — a quick completeness readout."""
        counts: dict[str, int] = defaultdict(int)
        for t in self._node_types.values():
            counts[t.value] += 1
        return dict(counts)


def summarize_tiers(evidence: Iterable[Evidence]) -> dict[str, int]:
    """Count evidence by tier name — used in reports."""
    counts: dict[str, int] = defaultdict(int)
    for e in evidence:
        counts[e.tier.name] += 1
    return dict(counts)
