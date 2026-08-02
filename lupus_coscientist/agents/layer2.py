"""Layer 2 — Mechanism Discovery agents.

These agents populate the causal evidence graph with nodes (cell states,
mechanisms, genes) and signed edges. Genetic Causality contributes the highest-tier
edges (human genetics); the others contribute mechanism / cell-state structure.
"""

from __future__ import annotations

from typing import Any

from .. import parsing, prompts
from ..agent_base import AgentResult, ScientificAgent
from ..context import ResearchContext
from ..evidence import Evidence, EvidenceTier, NodeType, Polarity, Sign


class CellStateAgent(ScientificAgent):
    agent_id = "cell_state"
    name = "Cell State Agent"
    layer = 2
    purpose = "Identify functional cell states (ABC/pre-ABC/GC/plasma), not clusters."
    system_prompt = prompts.CELL_STATE

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        for s in payload.get("states", []):
            ctx.graph.add_node(s["name"], NodeType.CELL_STATE)
            n += 1
        ctx.store("cell_states", payload)
        return AgentResult(agent=self.agent_id, summary=f"Defined {n} cell states.", payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"states": [
            {"name": "ABC", "markers": ["TBX21", "ITGAX", "ZEB2"], "function": "autoantibody-prone",
             "switch_position": "ZEB2-high"},
            {"name": "pre-ABC", "markers": ["FCRL5"], "function": "transitional",
             "switch_position": "transitional"},
            {"name": "naive B", "markers": ["IGHD"], "function": "resting", "switch_position": "MEF2B-high"},
        ]}


class TrajectoryAgent(ScientificAgent):
    agent_id = "trajectory"
    name = "Development Trajectory Agent"
    layer = 2
    purpose = "Lineage: branching, convergence, irreversibility, plasticity, the fork."
    system_prompt = prompts.TRAJECTORY

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        for t in payload.get("transitions", []):
            ev = Evidence(id=f"traj:{n}", tier=parsing.tier(t.get("evidence_tier"), EvidenceTier.ASSOCIATION),
                          polarity=Polarity.SUPPORTS, summary=f"{t['from']}->{t['to']} transition",
                          agent=self.agent_id, strength=0.55)
            ctx.register_evidence(ev)
            ctx.graph.add_edge(t["from"], t["to"], source_type=NodeType.CELL_STATE,
                               target_type=NodeType.CELL_STATE, sign=Sign.ACTIVATES, evidence=[ev])
            n += 1
        ctx.store("trajectory", payload)
        return AgentResult(agent=self.agent_id, summary=f"Mapped {n} state transitions.",
                           produced_evidence=n, payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"transitions": [
            {"from": "naive B", "to": "pre-ABC", "reversible": True, "fork": False, "evidence_tier": "association"},
            {"from": "pre-ABC", "to": "ABC", "reversible": False, "fork": True, "evidence_tier": "animal"},
        ], "irreversible_points": ["pre-ABC -> ABC commitment"]}


class RegulatoryNetworkAgent(ScientificAgent):
    agent_id = "regulatory_network"
    name = "Regulatory Network Agent"
    layer = 2
    purpose = "TF-enhancer-epigenetic network (SCENIC+/ArchR/CellOracle spirit)."
    system_prompt = prompts.REGULATORY_NETWORK

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        for r in payload.get("regulators", []):
            tier = EvidenceTier.PERTURBATION if r.get("edge_support") == "perturbation" else EvidenceTier.ASSOCIATION
            sign = Sign.ACTIVATES if r.get("role") == "activator" else Sign.INHIBITS
            ctx.graph.add_node(r["tf"], NodeType.GENE)
            for tgt in r.get("targets", []):
                ev = Evidence(id=f"grn:{n}", tier=tier, polarity=Polarity.SUPPORTS,
                              summary=f"{r['tf']} {r.get('role','regulates')} {tgt}",
                              agent=self.agent_id, strength=0.7 if tier is EvidenceTier.PERTURBATION else 0.5)
                ctx.register_evidence(ev)
                ctx.graph.add_edge(r["tf"], tgt, source_type=NodeType.GENE,
                                   target_type=NodeType.MOLECULAR_MECHANISM, sign=sign, evidence=[ev])
                n += 1
        ctx.store("regulatory_network", payload)
        return AgentResult(agent=self.agent_id, summary=f"Built GRN with {n} regulatory edges.",
                           produced_evidence=n, payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"regulators": [
            {"tf": "IRF5", "targets": ["ZEB2-MEF2B switch flip"], "edge_support": "perturbation", "role": "activator"},
            {"tf": "ZEB2", "targets": ["ABC transcriptional program"], "edge_support": "perturbation", "role": "activator"},
        ]}


class LigandReceptorAgent(ScientificAgent):
    agent_id = "ligand_receptor"
    name = "Ligand-Receptor Agent"
    layer = 2
    purpose = "Intercellular communication, correctly separating sender from receiver."
    system_prompt = prompts.LIGAND_RECEPTOR

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        for i in payload.get("interactions", []):
            ev = Evidence(id=f"lr:{n}", tier=parsing.tier(i.get("tier"), EvidenceTier.ASSOCIATION),
                          polarity=Polarity.SUPPORTS,
                          summary=f"{i['sender']}:{i.get('ligand','?')} -> {i['receiver']}:{i.get('receptor','?')}",
                          agent=self.agent_id, strength=0.5)
            ctx.register_evidence(ev)
            ctx.graph.add_edge(i["sender"], i["receiver"], source_type=NodeType.CELL_STATE,
                               target_type=NodeType.CELL_STATE, sign=Sign.ACTIVATES, evidence=[ev])
            n += 1
        ctx.store("ligand_receptor", payload)
        return AgentResult(agent=self.agent_id, summary=f"Inferred {n} sender->receiver interactions.",
                           produced_evidence=n, payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"interactions": [
            {"sender": "pDC", "receiver": "pre-ABC", "ligand": "IFNA", "receptor": "IFNAR",
             "receiver_effect": "primes IRF5 activity", "tier": "animal"},
        ]}


class GeneticCausalityAgent(ScientificAgent):
    agent_id = "genetic_causality"
    name = "Genetic Causality Agent"
    layer = 2
    purpose = "GWAS+eQTL+pQTL+CRISPR+coloc+fine-mapping -> causal genes (top-tier edges)."
    tools = ("mcp__Open_Targets__query_open_targets_graphql", "mcp__Open_Targets__search_entities",
             "mcp__Open_Targets__batch_query_open_targets_graphql")
    system_prompt = prompts.GENETIC_CAUSALITY

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        for g in payload.get("causal_genes", []):
            # Human-genetics tier only when coloc AND fine-mapping support the link.
            coloc = parsing.as_bool(g.get("coloc"))
            fm = parsing.as_bool(g.get("finemapped"))
            tier = EvidenceTier.HUMAN_GENETICS if (coloc and fm) else EvidenceTier.ASSOCIATION
            if parsing.as_bool(g.get("perturbation_confirmed")):
                tier = max(tier, EvidenceTier.PERTURBATION)
            variant = g.get("variant", f"variant_{n}")
            gene = g["gene"]
            sign = Sign.ACTIVATES if g.get("direction") == "risk" else Sign.INHIBITS
            ev = Evidence(id=f"gen:{gene}", tier=tier, polarity=Polarity.SUPPORTS,
                          summary=f"{variant} -> {gene} ({g.get('direction','?')}); coloc={coloc} finemap={fm}",
                          source="OpenTargets", agent=self.agent_id,
                          strength=0.85 if tier is EvidenceTier.HUMAN_GENETICS else 0.5)
            ctx.register_evidence(ev)
            ctx.graph.add_edge(variant, gene, source_type=NodeType.VARIANT,
                               target_type=NodeType.GENE, sign=sign, evidence=[ev])
            n += 1
        ctx.store("genetic_causality", payload)
        return AgentResult(agent=self.agent_id, summary=f"Established {n} variant->gene causal edges.",
                           produced_evidence=n, payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"causal_genes": [
            {"gene": "IRF5", "variant": "rs2004640", "coloc": True, "finemapped": True,
             "perturbation_confirmed": True, "direction": "risk",
             "note": "risk allele increases IRF5 expression; Perturb-seq confirms ABC effect"},
            {"gene": "NCF1", "variant": "p.Arg90His", "coloc": True, "finemapped": True,
             "perturbation_confirmed": False, "direction": "risk", "note": "reduced ROS, ABC expansion"},
        ]}


class EvolutionaryAgent(ScientificAgent):
    agent_id = "evolutionary"
    name = "Evolutionary Biology Agent"
    layer = 2
    purpose = "Why would evolution preserve this pathway? Trade-off / antagonistic pleiotropy."
    system_prompt = prompts.EVOLUTIONARY

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        ctx.store("evolutionary", payload)
        return AgentResult(agent=self.agent_id,
                           summary=f"Evolutionary reframing: {payload.get('conserved_function','')[:80]}",
                           payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"conserved_function": "IRF5/IFN axis drives rapid antiviral and age-associated memory B cells",
                "tradeoff": "The same ABC program that provides durable antiviral memory becomes autoreactive under chronic self-antigen",
                "prediction": "ABCs should be inducible by chronic antigen even without lupus genetics"}


class TissueEcologyAgent(ScientificAgent):
    agent_id = "tissue_ecology"
    name = "Tissue Ecology Agent"
    layer = 2
    purpose = "Niches (vascular/stromal/immune), TLS; which niche sustains the pathogenic state."
    system_prompt = prompts.TISSUE_ECOLOGY

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        for niche in payload.get("niches", []):
            supported = niche.get("supports_state")
            if supported:
                ctx.graph.add_node(niche["name"], NodeType.TISSUE_PATHOLOGY)
                ev = Evidence(id=f"niche:{n}", tier=EvidenceTier.ASSOCIATION, polarity=Polarity.SUPPORTS,
                              summary=f"{niche['name']} niche sustains {supported}", agent=self.agent_id,
                              strength=0.5)
                ctx.register_evidence(ev)
                ctx.graph.add_edge(supported, niche["name"], source_type=NodeType.CELL_STATE,
                                   target_type=NodeType.TISSUE_PATHOLOGY, sign=Sign.ACTIVATES, evidence=[ev])
                n += 1
        ctx.store("tissue_ecology", payload)
        return AgentResult(agent=self.agent_id, summary=f"Characterized {len(payload.get('niches', []))} niches.",
                           produced_evidence=n, payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"niches": [
            {"name": "lupus nephritis tubulointerstitial damage", "cells": ["ABC", "Tph", "myeloid"],
             "supports_state": "ABC", "tls": True},
        ]}


class MetabolismAgent(ScientificAgent):
    agent_id = "metabolism"
    name = "Metabolism Agent"
    layer = 2
    purpose = "Metabolic programs (lipid/SREBP/mito/glycolysis) enabling the pathogenic state."
    system_prompt = prompts.METABOLISM

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        ctx.store("metabolism", payload)
        return AgentResult(agent=self.agent_id,
                           summary=f"Identified {len(payload.get('programs', []))} metabolic programs.",
                           payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"programs": [
            {"pathway": "cholesterol/SREBP2", "role": "enables", "target": "ABC survival", "tier": "association"},
        ]}
