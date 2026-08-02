"""Layer 4 — Validation agents (Public Data Validation, Experimental Design,
Statistics). Experimental Design produces the likelihood tables the
decision-theoretic planner consumes."""

from __future__ import annotations

from typing import Any

from .. import parsing, prompts
from ..agent_base import AgentResult, ScientificAgent
from ..context import ResearchContext
from ..evidence import Evidence, EvidenceTier, Polarity
from ..experiment import Experiment, PredictedOutcome
from ..hypothesis import HypothesisStatus


class PublicDataValidationAgent(ScientificAgent):
    agent_id = "public_validation"
    name = "Public Data Validation Agent"
    layer = 4
    purpose = "Test each hypothesis on INDEPENDENT public data; never on discovery data."
    tools = ("mcp__PubMed__search_articles",)
    system_prompt = prompts.PUBLIC_VALIDATION

    def build_prompt(self, ctx: ResearchContext) -> str:
        safe = ", ".join(d.accession for d in ctx.validation_datasets()) or "(none available)"
        return (super().build_prompt(ctx) +
                f"\n\nUse ONLY these validation datasets (discovery data is off-limits): {safe}\n"
                f"Discovery datasets you must NOT validate on: {sorted(ctx.discovery_datasets)}")

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        for v in payload.get("validations", []):
            hyp = ctx.pool.get(v.get("hypothesis_id", ""))
            if not hyp:
                continue
            dataset = v.get("dataset", "")
            leaked = dataset in ctx.discovery_datasets
            result = v.get("result", "inconclusive")
            polarity = parsing.polarity(result)
            ev = Evidence(
                id=f"val:{v['hypothesis_id']}:{n}",
                tier=EvidenceTier.ASSOCIATION,
                polarity=polarity,
                summary=f"validation on {dataset}: {v.get('prediction','')} -> {result}"
                        + (" [LEAKAGE FLAGGED]" if leaked else ""),
                source=dataset, agent=self.agent_id,
                strength=parsing.as_float(v.get("strength"), 0.6),
                is_independent=(not leaked) and parsing.as_bool(v.get("independent"), True),
            )
            ctx.register_evidence(ev)
            hyp.attach(ev)
            # Independent validation doesn't just endorse the hypothesis — it
            # accumulates causal evidence on the mechanistic links themselves,
            # upgrading the provisional (proposed) edges of its chain.
            chain = hyp.best_chain(ctx.graph)
            if chain and ev.is_independent:
                for edge in chain.edges:
                    edge.add(ev)
            if polarity is Polarity.SUPPORTS and ev.is_independent:
                hyp.note(f"validated on independent {dataset}")
            n += 1
        return AgentResult(agent=self.agent_id, produced_evidence=n,
                           summary=f"Ran {n} independent validations.", payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"validations": [
            {"hypothesis_id": "H01", "dataset": "GSE-VALID-02",
             "prediction": "IRF5-high B cells enriched for ZEB2 and ABC program",
             "result": "supported", "independent": True, "strength": 0.75},
            {"hypothesis_id": "H01", "dataset": "EGA-VALID-03",
             "prediction": "ABC-committed cells show open ZEB2 / closed MEF2B chromatin",
             "result": "supported", "independent": True, "strength": 0.7},
            {"hypothesis_id": "H03", "dataset": "GSE-VALID-02",
             "prediction": "NCF1 genotype predicts ABC fraction independent of IRF5",
             "result": "inconclusive", "independent": True, "strength": 0.4},
            {"hypothesis_id": "H04", "dataset": "EGA-VALID-03",
             "prediction": "ABC program vanishes in nuclei (no dissociation)",
             "result": "refuted", "independent": True, "strength": 0.7},
        ]}


class ExperimentalDesignAgent(ScientificAgent):
    agent_id = "experimental_design"
    name = "Experimental Design Agent"
    layer = 4
    purpose = "Design discriminating experiments with outcome likelihood tables."
    system_prompt = prompts.EXPERIMENTAL_DESIGN

    def build_prompt(self, ctx: ResearchContext) -> str:
        live = ctx.pool.ranked()
        listing = "\n".join(f"- {h.id}: {h.statement}" for h in live)
        return (super().build_prompt(ctx) +
                "\n\nDesign experiments that discriminate among these LIVE hypotheses. "
                "For each experiment give a likelihood table P(outcome | hypothesis).\n" + listing)

    def integrate(self, ctx: ResearchContext, payload: dict[str, Any]) -> AgentResult:
        n = 0
        for e in payload.get("experiments", []):
            outcomes = [
                PredictedOutcome(label=o.get("label", f"o{i}"),
                                 likelihoods={k: parsing.as_float(v) for k, v in o.get("likelihoods", {}).items()})
                for i, o in enumerate(e.get("outcomes", []))
            ]
            exp = Experiment(
                id=e.get("id", f"E{n}"), title=e.get("title", ""),
                etype=parsing.experiment_type(e.get("type")),
                hypotheses_tested=tuple(e.get("hypotheses_tested", [])),
                outcomes=outcomes,
                cost_months=parsing.as_float(e.get("cost_months")) or None,
                feasibility=parsing.as_float(e.get("feasibility"), 0.8),
                rationale=e.get("rationale", ""),
            )
            ctx.add_experiment(exp)
            n += 1
        return AgentResult(agent=self.agent_id, summary=f"Designed {n} candidate experiments.", payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"experiments": [
            {"id": "E1", "title": "B-cell-conditional Irf5 base-edit knockout, then Perturb-seq for ABC fate",
             "type": "base_edit", "hypotheses_tested": ["H01", "H02", "H03"],
             "cost_months": 6, "feasibility": 0.8,
             "outcomes": [
                 {"label": "ABCs abolished by B-cell IRF5 loss", "likelihoods": {"H01": 0.9, "H02": 0.1, "H03": 0.3}},
                 {"label": "ABCs unchanged", "likelihoods": {"H01": 0.1, "H02": 0.9, "H03": 0.7}},
             ],
             "rationale": "Separates B-cell-intrinsic (H01) from myeloid (H02) and NCF1 (H03) drivers"},
            {"id": "E2", "title": "IRF5 dosage titration and ZEB2/MEF2B reporter readout in primary human B cells",
             "type": "gain_of_function", "hypotheses_tested": ["H01", "H03"],
             "cost_months": 5, "feasibility": 0.75,
             "outcomes": [
                 {"label": "IRF5 dose flips ZEB2>MEF2B switch", "likelihoods": {"H01": 0.85, "H03": 0.2}},
                 {"label": "No switch flip with IRF5 dose", "likelihoods": {"H01": 0.15, "H03": 0.8}},
             ],
             "rationale": "Directly tests the switch-flip mechanism specific to H01"},
            {"id": "E3", "title": "Bulk RNA-seq of sorted ABCs across 3 more cohorts",
             "type": "human_tissue", "hypotheses_tested": ["H01", "H02"],
             "cost_months": 4, "feasibility": 0.9,
             "outcomes": [
                 {"label": "IRF5 correlates with ABC", "likelihoods": {"H01": 0.6, "H02": 0.6}},
                 {"label": "no correlation", "likelihoods": {"H01": 0.4, "H02": 0.4}},
             ],
             "rationale": "Correlative; poor at discriminating H01 vs H02 (low EIG expected)"},
        ]}


class StatisticsAgent(ScientificAgent):
    agent_id = "statistics"
    name = "Statistics Agent"
    layer = 4
    purpose = "Audit batch effects, data leakage, confounding, multiple testing."
    system_prompt = prompts.STATISTICS

    def act(self, ctx: ResearchContext) -> AgentResult:
        # Deterministic leakage check first: any hypothesis validated on discovery data.
        auto_flags = 0
        for hyp in ctx.pool.all:
            leaked = [e for e in hyp.supporting if not e.is_independent]
            if leaked:
                for e in leaked:
                    e.strength = min(e.strength, 0.3)  # discount leaked support
                hyp.note(f"statistics: {len(leaked)} non-independent support items discounted")
                auto_flags += 1

        payload: dict[str, Any] = {}
        if self.client is not None:
            payload = self.client.complete_json(self.request(ctx))
        else:
            payload = self._offline(ctx)

        for f in payload.get("flags", []):
            hyp = ctx.pool.get(f.get("hypothesis_id", ""))
            if not hyp:
                continue
            rec = f.get("recommendation", "ok")
            sev = parsing.as_float(f.get("severity"), 0.3)
            if rec in ("downgrade", "eliminate"):
                ev = Evidence(id=f"stat:{f['hypothesis_id']}", tier=EvidenceTier.ASSOCIATION,
                              polarity=Polarity.CONTRADICTS,
                              summary=f"stat issue: {f.get('issue','')}", agent=self.agent_id, strength=sev)
                ctx.register_evidence(ev)
                hyp.attach(ev)
            if rec == "eliminate":
                hyp.status = HypothesisStatus.ELIMINATED
                hyp.note(f"ELIMINATED by statistics: {f.get('issue','')}")

        summary = f"Statistics audit: {auto_flags} leakage auto-flags, {len(payload.get('flags', []))} model flags."
        ctx.record(self.agent_id, self.layer, summary)
        return AgentResult(agent=self.agent_id, summary=summary, payload=payload)

    def _offline(self, ctx: ResearchContext) -> dict[str, Any]:
        return {"flags": [
            {"hypothesis_id": "H03", "issue": "confound", "severity": 0.4, "recommendation": "downgrade"},
        ]}
