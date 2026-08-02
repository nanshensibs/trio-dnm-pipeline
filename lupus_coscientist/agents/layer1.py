"""Layer 1 — Knowledge Acquisition agents (Literature, Evidence Ranking,
Public Database, Clinical Guideline)."""

from __future__ import annotations

from typing import Any

from .. import parsing, prompts
from ..agent_base import AgentResult, ScientificAgent
from ..context import Dataset, ResearchContext
from ..evidence import Evidence, Polarity


class LiteratureAgent(ScientificAgent):
    agent_id = "literature"
    name = "Literature Agent"
    layer = 1
    purpose = "Comprehensive, structured literature review: consensus, unknowns, contradictions."
    tools = ("mcp__PubMed__search_articles", "mcp__Consensus__search",
             "mcp__Scholar_Gateway__semanticSearch", "mcp__PubMed__get_article_metadata")
    system_prompt = prompts.LITERATURE

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        ctx.store("literature", payload)
        n = 0
        for ref in payload.get("key_references", []):
            ev = Evidence(
                id=f"lit:{ref.get('pmid', n)}",
                tier=parsing.tier(ref.get("tier")),
                polarity=Polarity.SUPPORTS,
                summary=ref.get("claim", ""),
                source=f"PMID:{ref.get('pmid', '?')}",
                agent=self.agent_id,
                strength=0.6,
            )
            ctx.register_evidence(ev)
            n += 1
        # Contradictions become NEGATIVE evidence candidates for the pool.
        for c in payload.get("contradictions", []):
            ev = Evidence(
                id=f"lit-contra:{n}",
                tier=parsing.tier("association"),
                polarity=Polarity.CONTRADICTS,
                summary=f"{c.get('claim','')} — contested: {c.get('camp_b','')}",
                source=";".join(f"PMID:{p}" for p in c.get("pmids", [])),
                agent=self.agent_id,
                strength=0.5,
            )
            ctx.register_evidence(ev)
            n += 1
        return AgentResult(
            agent=self.agent_id, produced_evidence=n,
            summary=(f"Reviewed literature: {len(payload.get('consensus', []))} consensus points, "
                     f"{len(payload.get('unknowns', []))} open questions, "
                     f"{len(payload.get('contradictions', []))} contradictions."),
            payload=payload,
        )

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {
            "consensus": ["IRF5 risk haplotypes associate with SLE across ancestries",
                          "Type I IFN signature marks a large SLE subset"],
            "unknowns": ["Whether IRF5 acts primarily in B cells vs myeloid cells to drive ABC expansion",
                         "Whether IRF5 dosage flips the ZEB2-MEF2B switch"],
            "contradictions": [{"claim": "IRF5 is primarily myeloid-acting",
                                 "camp_a": "myeloid-centric", "camp_b": "B-cell-intrinsic",
                                 "pmids": ["00000001", "00000002"]}],
            "major_hypotheses": [{"statement": "IRF5 hyperactivity biases B cells toward the ABC fate",
                                   "proponents": "B-cell-intrinsic camp", "pmids": ["00000003"]}],
            "key_references": [
                {"pmid": "00000003", "claim": "IRF5 risk allele increases IRF5 expression in B cells",
                 "tier": "human_genetics"},
                {"pmid": "00000004", "claim": "Irf5 knockout reduces ABC formation in lupus mice",
                 "tier": "animal"},
            ],
        }


class EvidenceRankingAgent(ScientificAgent):
    """Deterministic: re-tiers and re-weights the shared evidence pool.

    Rather than trusting each producing agent's self-assigned tier, this agent
    can be given explicit re-rankings; in their absence it applies a guardrail —
    review-tier items are capped in strength so they cannot masquerade as causal.
    """

    agent_id = "evidence_ranking"
    name = "Evidence Ranking Agent"
    layer = 1
    purpose = "Rank evidence by causal tier; prevent treating all papers equally."
    system_prompt = prompts.EVIDENCE_RANKING

    def act(self, ctx: ResearchContext) -> AgentResult:
        payload: dict[str, Any] = {}
        if self.client is not None:
            # Provide the current pool for the model to re-rank.
            req = self.request(ctx)
            req.prompt += "\n\nEvidence pool:\n" + "\n".join(
                f"- {e.id} [{e.tier.name}] {e.summary}" for e in ctx.evidence.values()
            )
            payload = self.client.complete_json(req)

        rerank = {r["id"]: r for r in payload.get("ranked", [])}
        adjusted = 0
        for ev in ctx.evidence.values():
            if ev.id in rerank:
                r = rerank[ev.id]
                ev.tier = parsing.tier(r.get("tier"), ev.tier)
                ev.strength = parsing.as_float(r.get("strength"), ev.strength)
                ev.is_independent = parsing.as_bool(r.get("independent"), ev.is_independent)
                adjusted += 1
            # Guardrail: review-tier evidence cannot carry high strength.
            from ..evidence import EvidenceTier
            if ev.tier is EvidenceTier.REVIEW and ev.strength > 0.4:
                ev.strength = 0.4
                adjusted += 1
        summary = f"Re-ranked {adjusted} evidence items; review-tier capped."
        ctx.record(self.agent_id, self.layer, summary)
        return AgentResult(agent=self.agent_id, summary=summary, payload=payload)


class PublicDatabaseAgent(ScientificAgent):
    agent_id = "public_database"
    name = "Public Database Agent"
    layer = 1
    purpose = "Find relevant public datasets (GEO/ArrayExpress/ImmPort/HuBMAP/HCA)."
    tools = ("mcp__PubMed__search_articles",)
    system_prompt = prompts.PUBLIC_DATABASE

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        for d in payload.get("datasets", []):
            ctx.add_dataset(Dataset(
                accession=d.get("accession", f"DS{n}"),
                repository=d.get("repository", "GEO"),
                modality=d.get("modality", "scRNA"),
                title=d.get("title", ""),
                relevance=parsing.as_float(d.get("relevance"), 0.5),
                used_for=d.get("used_for", "validation"),
                url=d.get("url", ""),
            ))
            n += 1
        return AgentResult(
            agent=self.agent_id, summary=f"Found {n} candidate datasets "
            f"({len(ctx.discovery_datasets)} tagged discovery).", payload=payload,
        )

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"datasets": [
            {"accession": "GSE-DISCOVERY-01", "repository": "GEO", "modality": "scRNA",
             "title": "SLE PBMC atlas (illustrative)", "relevance": 0.9, "used_for": "discovery"},
            {"accession": "GSE-VALID-02", "repository": "ImmPort", "modality": "scRNA",
             "title": "Independent SLE B-cell cohort (illustrative)", "relevance": 0.8,
             "used_for": "validation"},
            {"accession": "EGA-VALID-03", "repository": "ArrayExpress", "modality": "scATAC",
             "title": "SLE B-cell chromatin accessibility (illustrative)", "relevance": 0.7,
             "used_for": "validation"},
        ]}


class ClinicalGuidelineAgent(ScientificAgent):
    agent_id = "clinical_guideline"
    name = "Clinical Guideline Agent"
    layer = 1
    purpose = "Collect EULAR/ACR/KDIGO/FDA/EMA standards and identify clinical gaps."
    tools = ("mcp__Clinical_Trials__search_trials", "mcp__Clinical_Trials__analyze_endpoints")
    system_prompt = prompts.CLINICAL_GUIDELINE

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        ctx.store("clinical", payload)
        return AgentResult(
            agent=self.agent_id,
            summary=f"Collected {len(payload.get('guidelines', []))} guideline points; "
                    f"{len(payload.get('clinical_gaps', []))} clinical gaps identified.",
            payload=payload,
        )

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {
            "guidelines": [{"body": "EULAR", "recommendation": "HCQ backbone + steroid minimization",
                            "year": 2023},
                           {"body": "KDIGO", "recommendation": "MMF or cyclophosphamide for LN class III/IV",
                            "year": 2024}],
            "clinical_gaps": ["No therapy selectively resets the pathogenic ABC compartment",
                              "No biomarker predicts which patients need B-cell depletion vs IFN blockade"],
            "endpoints_in_use": ["SLEDAI-2K", "renal response (complete/partial)", "anti-dsDNA"],
        }
