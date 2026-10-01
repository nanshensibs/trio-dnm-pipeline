"""Registry of all 31 agents (28 specialized + 3 novel cross-cutting)."""

from __future__ import annotations

from typing import Optional

from ..agent_base import ScientificAgent
from ..llm import LLMClient
from . import layer1, layer2, layer3, layer4, layer5, layer6, special

# Ordered catalogue: (number, class). Numbers 1-28 match the architecture spec;
# the three novel agents are appended as 29-31 but wired in cross-cuttingly.
AGENT_CLASSES: list[type[ScientificAgent]] = [
    # Layer 1 — Knowledge Acquisition
    layer1.LiteratureAgent,             # 1
    layer1.EvidenceRankingAgent,        # 2
    layer1.PublicDatabaseAgent,         # 3
    layer1.ClinicalGuidelineAgent,      # 4
    # Layer 2 — Mechanism Discovery
    layer2.CellStateAgent,              # 5
    layer2.TrajectoryAgent,             # 6
    layer2.RegulatoryNetworkAgent,      # 7
    layer2.LigandReceptorAgent,         # 8
    layer2.GeneticCausalityAgent,       # 9
    layer2.EvolutionaryAgent,           # 10
    layer2.TissueEcologyAgent,          # 11
    layer2.MetabolismAgent,             # 12
    # Layer 3 — Hypothesis Generation
    layer3.MechanismGeneratorAgent,     # 13
    layer3.CounterfactualAgent,         # 14
    layer3.AlternativeAgent,            # 15
    layer3.ReviewerAgent,               # 16
    layer3.NoveltyAgent,                # 17
    # Layer 4 — Validation
    layer4.PublicDataValidationAgent,   # 18
    layer4.ExperimentalDesignAgent,     # 19
    layer4.StatisticsAgent,             # 20
    # Layer 5 — Translation
    layer5.DrugDiscoveryAgent,          # 21
    layer5.BiomarkerAgent,              # 22
    layer5.PrecisionMedicineAgent,      # 23
    layer5.TrialDesignAgent,            # 24
    # Layer 6 — Meta Reasoning
    layer6.DebateAgent,                 # 25
    layer6.ConsensusAgent,              # 26
    layer6.SkepticAgent,                # 27
    layer6.MetaReviewerAgent,           # 28
    # Novel cross-cutting capabilities
    special.CausalEvidenceGraphAgent,           # 29
    special.NegativeEvidenceAgent,              # 30
    special.DecisionTheoreticExperimentPlannerAgent,  # 31
]


def build_agents(client: Optional[LLMClient] = None) -> dict[str, ScientificAgent]:
    """Instantiate every agent, keyed by agent_id, sharing one LLM client."""
    agents: dict[str, ScientificAgent] = {}
    for cls in AGENT_CLASSES:
        inst = cls(client)
        agents[inst.agent_id] = inst
    return agents


def catalogue() -> list[dict]:
    """Human-readable listing of the roster (used by the CLI `list` command)."""
    rows = []
    for i, cls in enumerate(AGENT_CLASSES, start=1):
        rows.append({
            "n": i,
            "id": cls.agent_id,
            "name": cls.name,
            "layer": cls.layer,
            "purpose": cls.purpose,
            "tools": list(cls.tools),
        })
    return rows
