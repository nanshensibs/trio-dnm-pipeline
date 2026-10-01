"""Lupus Co-Scientist — a mechanism-discovery multi-agent system for SLE.

A biomedical discovery co-scientist (not merely a hypothesis generator) that
progressively eliminates weak hypotheses while accumulating causal evidence.

Quick start (offline / deterministic demo):

    from lupus_coscientist import LupusCoScientist
    report = LupusCoScientist().investigate(
        "What is the causal role of IRF5 in ABC-driven lupus nephritis?",
        focus="IRF5",
    )
    print(report.final_report["headline_mechanism"])

To run with real tools, pass a ``ClaudeAgentClient`` wired to the scientific MCP
servers (see ``lupus_coscientist.mcp_config``).
"""

from .context import Dataset, ResearchContext
from .evidence import (
    CausalChain,
    CausalEdge,
    CausalEvidenceGraph,
    Evidence,
    EvidenceTier,
    NodeType,
    Polarity,
    Sign,
)
from .experiment import (
    DecisionTheoreticPlanner,
    Experiment,
    ExperimentPlan,
    ExperimentType,
    PredictedOutcome,
)
from .hypothesis import Hypothesis, HypothesisPool, HypothesisStatus
from .llm import AnthropicClient, ClaudeAgentClient, LLMRequest, OfflineStubClient
from .orchestrator import LupusCoScientist, RunConfig, RunReport

__version__ = "0.1.0"

__all__ = [
    "LupusCoScientist",
    "RunConfig",
    "RunReport",
    "ResearchContext",
    "Dataset",
    "CausalEvidenceGraph",
    "CausalChain",
    "CausalEdge",
    "Evidence",
    "EvidenceTier",
    "NodeType",
    "Polarity",
    "Sign",
    "Hypothesis",
    "HypothesisPool",
    "HypothesisStatus",
    "Experiment",
    "ExperimentType",
    "ExperimentPlan",
    "PredictedOutcome",
    "DecisionTheoreticPlanner",
    "ClaudeAgentClient",
    "AnthropicClient",
    "OfflineStubClient",
    "LLMRequest",
    "__version__",
]
