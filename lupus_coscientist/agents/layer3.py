"""Layer 3 — Hypothesis Generation agents.

Mechanism Generator seeds the pool (and lays each hypothesis's provisional causal
chain into the graph). Counterfactual and Alternative widen it. Reviewer and Novelty
annotate it — Reviewer's fatal-flaw severity becomes negative evidence that erodes
belief during pruning.
"""

from __future__ import annotations

from typing import Any

from .. import parsing, prompts
from ..agent_base import AgentResult, ScientificAgent
from ..context import ResearchContext
from ..evidence import Evidence, EvidenceTier, Polarity, Sign
from ..hypothesis import Hypothesis


def _lay_chain(ctx: ResearchContext, hyp_id: str, chain: list[dict]) -> tuple[str, ...] | None:
    """Add a hypothesis's provisional causal chain to the graph as low-tier edges.

    Returns the full ordered node tuple, or None if the chain is degenerate.
    The provisional edges are REVIEW-tier ('proposed'); real evidence from Layer 2/4
    lands on the same (source, target) keys and upgrades them.
    """
    nodes = [step.get("node") for step in chain if step.get("node")]
    if len(nodes) < 2:
        return None
    for i in range(len(nodes) - 1):
        src, tgt = nodes[i], nodes[i + 1]
        step = chain[i + 1]
        ev = Evidence(
            id=f"proposed:{hyp_id}:{i}", tier=EvidenceTier.REVIEW, polarity=Polarity.SUPPORTS,
            summary=f"proposed link {src}->{tgt} for {hyp_id}", agent="mechanism_generator",
            strength=0.3, is_independent=True,
        )
        ctx.register_evidence(ev)
        ctx.graph.add_edge(
            src, tgt,
            source_type=parsing.node_type(chain[i].get("type")),
            target_type=parsing.node_type(step.get("type")),
            sign=parsing.sign(step.get("sign"), Sign.ACTIVATES),
            evidence=[ev],
        )
    return tuple(nodes)


class MechanismGeneratorAgent(ScientificAgent):
    agent_id = "mechanism_generator"
    name = "Mechanism Generator"
    layer = 3
    purpose = "Produce 20-50 distinct candidate mechanisms, each with a causal chain."
    system_prompt = prompts.MECHANISM_GENERATOR

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        for h in payload.get("hypotheses", []):
            chain_nodes = _lay_chain(ctx, h["id"], h.get("chain", []))
            hyp = Hypothesis(
                id=h["id"], statement=h.get("statement", ""), mechanism=h.get("mechanism", ""),
                chain_nodes=chain_nodes,
                novelty=parsing.as_float(h.get("novelty"), 0.5),
                touches_switch=parsing.as_bool(h.get("touches_switch")),
                touches_ecology=parsing.as_bool(h.get("touches_ecology")),
                touches_fork=parsing.as_bool(h.get("touches_fork")),
                tags=tuple(h.get("tags", [])),
            )
            ctx.pool.add(hyp)
            n += 1
        return AgentResult(agent=self.agent_id, produced_hypotheses=n,
                           summary=f"Generated {n} candidate mechanisms.", payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"hypotheses": [
            {"id": "H01", "statement": "IRF5 risk dosage flips the ZEB2-MEF2B switch in B cells, committing pre-ABCs to the ABC fate that drives lupus nephritis.",
             "mechanism": "IRF5 -> ZEB2 induction -> switch flip -> ABC commitment",
             "chain": [
                 {"node": "rs2004640", "type": "variant", "sign": "activates"},
                 {"node": "IRF5", "type": "gene", "sign": "activates"},
                 {"node": "ZEB2-MEF2B switch flip", "type": "mechanism", "sign": "activates"},
                 {"node": "ABC", "type": "cell_state", "sign": "activates"},
                 {"node": "lupus nephritis tubulointerstitial damage", "type": "tissue_pathology", "sign": "activates"},
                 {"node": "renal flare", "type": "clinical_outcome", "sign": "activates"},
             ],
             "novelty": 0.75, "touches_switch": True, "touches_ecology": False, "touches_fork": True,
             "tags": ["B-cell-intrinsic", "IRF5", "switch"]},
            {"id": "H02", "statement": "IRF5 acts primarily in myeloid cells; ABC expansion is a bystander of myeloid IFN production.",
             "mechanism": "IRF5 -> myeloid IFN -> paracrine ABC induction",
             "chain": [
                 {"node": "rs2004640", "type": "variant", "sign": "activates"},
                 {"node": "IRF5", "type": "gene", "sign": "activates"},
                 {"node": "myeloid IFN production", "type": "mechanism", "sign": "activates"},
                 {"node": "ABC", "type": "cell_state", "sign": "activates"},
                 {"node": "renal flare", "type": "clinical_outcome", "sign": "activates"},
             ],
             "novelty": 0.4, "touches_switch": False, "touches_ecology": False, "touches_fork": False,
             "tags": ["myeloid-centric", "IRF5"]},
            {"id": "H03", "statement": "NCF1-driven low ROS, not IRF5, is the dominant driver of ABC expansion.",
             "mechanism": "NCF1 hypofunction -> low ROS -> ABC survival",
             "chain": [
                 {"node": "p.Arg90His", "type": "variant", "sign": "activates"},
                 {"node": "NCF1", "type": "gene", "sign": "inhibits"},
                 {"node": "low ROS", "type": "mechanism", "sign": "activates"},
                 {"node": "ABC", "type": "cell_state", "sign": "activates"},
                 {"node": "renal flare", "type": "clinical_outcome", "sign": "activates"},
             ],
             "novelty": 0.6, "touches_switch": False, "touches_ecology": False, "touches_fork": False,
             "tags": ["NCF1", "ROS"]},
            {"id": "H04", "statement": "ABC accumulation in SLE is a technical artifact of dissociation stress in scRNA-seq.",
             "mechanism": "dissociation -> stress signature mislabeled as ABC",
             "chain": [
                 {"node": "dissociation stress", "type": "mechanism", "sign": "activates"},
                 {"node": "ABC", "type": "cell_state", "sign": "activates"},
             ],
             "novelty": 0.2, "touches_switch": False, "touches_ecology": False, "touches_fork": False,
             "tags": ["artifact"]},
        ]}


class CounterfactualAgent(ScientificAgent):
    agent_id = "counterfactual"
    name = "Counterfactual Agent"
    layer = 3
    purpose = "Ask what if the opposite were true; surface flipping observations."
    system_prompt = prompts.COUNTERFACTUAL

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        ctx.store("counterfactuals", payload)
        # A counterfactual already supported by data is negative evidence for the hypothesis.
        n = 0
        for cf in payload.get("counterfactuals", []):
            if parsing.as_bool(cf.get("already_supported_by_data")):
                hyp = ctx.pool.get(cf.get("hypothesis_id", ""))
                if hyp:
                    ev = Evidence(id=f"cf:{cf['hypothesis_id']}", tier=EvidenceTier.ASSOCIATION,
                                  polarity=Polarity.CONTRADICTS,
                                  summary=f"counterfactual holds: {cf.get('opposite','')}",
                                  agent=self.agent_id, strength=0.5)
                    ctx.register_evidence(ev)
                    hyp.attach(ev)
                    n += 1
        return AgentResult(agent=self.agent_id, produced_evidence=n,
                           summary=f"Explored {len(payload.get('counterfactuals', []))} counterfactuals.",
                           payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"counterfactuals": [
            {"hypothesis_id": "H02", "opposite": "IRF5 conditional KO in B cells alone abolishes ABCs",
             "already_supported_by_data": True,
             "discriminating_observation": "B-cell-conditional Irf5 KO phenotype"},
        ]}


class AlternativeAgent(ScientificAgent):
    agent_id = "alternative"
    name = "Alternative Explanation Agent"
    layer = 3
    purpose = "Enumerate all plausible explanations (artifact/confound/reverse/rival)."
    system_prompt = prompts.ALTERNATIVE

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        existing = len(ctx.pool.all)
        for i, a in enumerate(payload.get("alternatives", [])):
            hid = f"HA{existing + i:02d}"
            endpoints = None
            if a.get("chain_source") and a.get("chain_target"):
                endpoints = (a["chain_source"], a["chain_target"])
            hyp = Hypothesis(id=hid, statement=a.get("statement", ""), mechanism=a.get("mechanism", ""),
                             chain_endpoints=endpoints, novelty=0.3,
                             tags=(a.get("class", "rival"),))
            ctx.pool.add(hyp)
            n += 1
        ctx.store("alternatives", payload)
        return AgentResult(agent=self.agent_id, produced_hypotheses=n,
                           summary=f"Added {n} alternative explanations.", payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"alternatives": []}  # H04 already covers the artifact class in this demo


class ReviewerAgent(ScientificAgent):
    agent_id = "reviewer"
    name = "Reviewer Agent"
    layer = 3
    purpose = "Nature/Science/Cell reviewer; find the fatal flaw per hypothesis."
    system_prompt = prompts.REVIEWER

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        for r in payload.get("reviews", []):
            hyp = ctx.pool.get(r.get("hypothesis_id", ""))
            if not hyp:
                continue
            sev = parsing.as_float(r.get("severity"), 0.3)
            fixable = parsing.as_bool(r.get("fixable"))
            # A non-fixable fatal flaw usually cites orthogonal contradicting data, so
            # it carries more than mere opinion; a fixable flaw is review-tier.
            flaw_tier = EvidenceTier.REVIEW if fixable else EvidenceTier.ASSOCIATION
            ev = Evidence(id=f"rev:{r['hypothesis_id']}", tier=flaw_tier,
                          polarity=Polarity.CONTRADICTS,
                          summary=f"reviewer flaw ({'fixable' if fixable else 'fatal'}): {r.get('fatal_flaw','')}",
                          agent=self.agent_id, strength=sev)
            ctx.register_evidence(ev)
            hyp.attach(ev)
            hyp.note(f"reviewer: {r.get('fatal_flaw','')}")
            n += 1
        return AgentResult(agent=self.agent_id, produced_evidence=n,
                           summary=f"Reviewed {n} hypotheses; flagged fatal flaws.", payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"reviews": [
            {"hypothesis_id": "H04", "fatal_flaw": "ABC gene program is reproducible across dissociation protocols and in situ (spatial), ruling out pure artifact",
             "fixable": False, "severity": 0.8},
            {"hypothesis_id": "H02", "fatal_flaw": "does not explain B-cell-intrinsic genetic effect on IRF5 expression",
             "fixable": True, "severity": 0.4},
        ]}


class NoveltyAgent(ScientificAgent):
    agent_id = "novelty"
    name = "Novelty Agent"
    layer = 3
    purpose = "Rate novelty (incremental/moderate/major) vs prior literature."
    tools = ("mcp__PubMed__search_articles", "mcp__Consensus__search")
    system_prompt = prompts.NOVELTY

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        for nv in payload.get("novelty", []):
            hyp = ctx.pool.get(nv.get("hypothesis_id", ""))
            if hyp:
                hyp.novelty = parsing.as_float(nv.get("score"), hyp.novelty)
                hyp.note(f"novelty={nv.get('level','?')} vs {nv.get('closest_prior','?')}")
                n += 1
        ctx.store("novelty", payload)
        return AgentResult(agent=self.agent_id, summary=f"Scored novelty for {n} hypotheses.", payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"novelty": [
            {"hypothesis_id": "H01", "level": "major", "score": 0.8,
             "closest_prior": "ZEB2-MEF2B switch (Dai 2024) — not yet linked to IRF5 dosage"},
            {"hypothesis_id": "H03", "level": "moderate", "score": 0.55, "closest_prior": "NCF1/ROS ABC work"},
        ]}
