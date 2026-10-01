"""Layer 5 — Translation agents (Drug Discovery, Biomarker, Precision Medicine,
Trial Design). These operate on the surviving hypotheses only."""

from __future__ import annotations

from typing import Any

from .. import prompts
from ..agent_base import AgentResult, ScientificAgent
from ..context import ResearchContext


class _TranslationAgent(ScientificAgent):
    """Shared: translation agents act on survivors and just record structured output."""

    store_key = ""
    count_key = ""

    def build_prompt(self, ctx: ResearchContext) -> str:
        survivors = "\n".join(f"- {h.id}: {h.statement}" for h in ctx.pool.ranked()[:5])
        return super().build_prompt(ctx) + "\n\nTop surviving mechanisms:\n" + survivors

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        ctx.store(self.store_key, payload)
        items = payload.get(self.count_key, [])
        return AgentResult(agent=self.agent_id,
                           summary=f"{self.name}: produced {len(items)} {self.count_key}.", payload=payload)


class DrugDiscoveryAgent(_TranslationAgent):
    agent_id = "drug_discovery"
    name = "Drug Discovery Agent"
    layer = 5
    purpose = "Mechanism -> druggable target -> existing drugs -> trials."
    tools = ("mcp__Open_Targets__query_open_targets_graphql", "mcp__Clinical_Trials__search_trials")
    system_prompt = prompts.DRUG_DISCOVERY
    store_key = "drugs"
    count_key = "targets"

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"targets": [
            {"hypothesis_id": "H01", "target": "IRF5", "druggable": False,
             "existing_drugs": ["TLR7/8 antagonists (upstream)", "anifrolumab (IFNAR, parallel)"],
             "trials": ["NCT-illustrative"], "tractability": "TF—target upstream TLR7/IRAK4 or downstream ABC via CAR-T"},
        ]}


class BiomarkerAgent(_TranslationAgent):
    agent_id = "biomarker"
    name = "Biomarker Agent"
    layer = 5
    purpose = "Diagnostic / prognostic / response biomarkers per mechanism."
    system_prompt = prompts.BIOMARKER
    store_key = "biomarkers"
    count_key = "biomarkers"

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"biomarkers": [
            {"hypothesis_id": "H01", "marker": "circulating ABC fraction x IRF5 genotype", "kind": "response",
             "assay": "flow + genotyping", "population": "SLE pre-B-cell-depletion"},
        ]}


class PrecisionMedicineAgent(_TranslationAgent):
    agent_id = "precision_medicine"
    name = "Precision Medicine Agent"
    layer = 5
    purpose = "Define patient endotypes rather than disease categories."
    system_prompt = prompts.PRECISION_MEDICINE
    store_key = "endotypes"
    count_key = "endotypes"

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"endotypes": [
            {"name": "IRF5-switch endotype", "driver_hypothesis": "H01",
             "signature": ["IRF5-high", "ZEB2-high ABC", "IFN-high"],
             "predicted_therapy": "B-cell reset (C-CAR168) or upstream TLR7 blockade"},
        ]}


class TrialDesignAgent(_TranslationAgent):
    agent_id = "trial_design"
    name = "Trial Design Agent"
    layer = 5
    purpose = "Enrichment, endpoints, adaptive design, stratification by endotype."
    tools = ("mcp__Clinical_Trials__analyze_endpoints", "mcp__Clinical_Trials__search_trials")
    system_prompt = prompts.TRIAL_DESIGN
    store_key = "trial"
    count_key = "trial"

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        ctx.store(self.store_key, payload)
        return AgentResult(agent=self.agent_id, summary="Designed a biomarker-gated trial.", payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"trial": {
            "enrichment": "IRF5-switch endotype (IRF5-risk + high ABC fraction)",
            "primary_endpoint": "complete renal response at week 52",
            "design": "adaptive, response-adaptive randomization",
            "stratification": "by baseline ABC fraction and IFN score",
            "biomarker_gate": "ABC fraction > cohort median",
        }}
