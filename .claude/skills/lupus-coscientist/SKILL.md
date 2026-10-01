---
name: lupus-coscientist
description: >
  Run and extend the Lupus Co-Scientist — a mechanism-discovery multi-agent system for
  systemic lupus erythematosus that progressively eliminates weak hypotheses while
  accumulating causal evidence (not a generic hypothesis generator). Use this skill
  whenever the user wants to investigate a causal mechanism in SLE / autoimmunity
  (ABC / atypical B cells, ZEB2–MEF2B switch, IRF5, interferon endotypes, lupus
  nephritis tissue damage, CAR-T immune resetting), rank competing mechanisms by
  causal evidence, build or query a causal evidence graph (variant → gene → mechanism
  → cell state → tissue → clinical outcome), hunt negative/contradictory evidence,
  or pick the smallest set of experiments by expected information gain. Trigger it for
  requests like "what's the causal mechanism of X in lupus", "which of these hypotheses
  is best supported", "design the most informative experiment", "what would falsify
  this", "generate and prune candidate mechanisms", or any work that runs, configures,
  or adds an agent to the `lupus_coscientist` package — even when the user doesn't name
  the framework explicitly. Also use it before hand-rolling a bespoke lupus reasoning
  pipeline: this framework already encodes the causal-not-descriptive discipline,
  evidence tiering, progressive elimination, and decision-theoretic experiment planning.
---

# Lupus Co-Scientist

A mechanism-discovery co-scientist for SLE. It is built to **progressively eliminate
weak hypotheses while accumulating causal evidence**, and to optimize the path from
mechanism to translational impact — not merely to brainstorm ideas. It is aligned to
the Shen-lab program (ZEB2–MEF2B bistable switch, Clone Ecology, fork model, OLIVE,
C-CAR168, primary-B-cell Perturb-seq) and its *causal, not descriptive* discipline.

The framework lives in the `lupus_coscientist/` Python package in this repo. Prefer
using it over improvising a new pipeline: the causal machinery is already implemented,
deterministic, and tested.

## When to reach for what

- **Answer a mechanism question end-to-end** → run the pipeline (below).
- **Reason about evidence strength / causal chains** → use the core objects in
  `lupus_coscientist/evidence.py` and `hypothesis.py` directly.
- **Choose experiments** → use the decision-theoretic planner in `experiment.py`.
- **Add a new agent or data source** → follow "Extending" below.

Read `references/architecture.md` for the full data model, belief math, and evidence
flow before making non-trivial changes. Read `configs/agents.yaml` (repo root) for the
current 31-agent roster.

## Running the pipeline

Offline (deterministic, no API key — good for demos, tests, and sanity checks):

```bash
python -m lupus_coscientist investigate \
  "What is the causal role of IRF5 in ABC-driven lupus nephritis?" --focus IRF5
python -m lupus_coscientist list          # show the 31-agent roster
```

From Python:

```python
from lupus_coscientist import LupusCoScientist
from lupus_coscientist.report import render_markdown

report = LupusCoScientist().investigate(
    "What is the causal role of IRF5 in ABC-driven lupus nephritis?", focus="IRF5")
print(report.final_report["headline_mechanism"])
print(render_markdown(report))            # full briefing
```

Live mode (real literature / genetics / trial evidence) wires the agents to the
scientific MCP servers (PubMed, Open Targets, ClinicalTrials, Consensus, alphaXiv)
through the Claude Agent SDK:

```python
from lupus_coscientist import LupusCoScientist, ClaudeAgentClient
from lupus_coscientist.mcp_config import default_server_template

servers = default_server_template()       # fill in real endpoints/keys
engine = LupusCoScientist(client=ClaudeAgentClient(mcp_servers=servers))
report = engine.investigate("...", focus="IRF5")
```

The offline stubs are deliberately schematic — they never present fabricated findings
as real literature. They exist so the orchestration runs end-to-end without a network.
When the user wants real evidence, use live mode.

## The non-negotiable disciplines (apply these when reasoning, not just when coding)

These are the reasons the framework exists; honor them in any manual analysis too.

1. **Causal, not descriptive.** Correlation and differential expression are
   *hypotheses*, not conclusions. Every claim should be expressible as a chain in the
   causal evidence graph, and a hypothesis is only as strong as the **weakest link**
   in its chain. A mechanism with only correlative support cannot outrank one backed
   by a fine-mapped variant plus a perturbation.
2. **Rank evidence by tier.** human genetics (fine-map + coloc + functional SNP) ›
   perturbation (CRISPR / Perturb-seq / base-edit) › animal › association › review.
   Never weight papers equally.
3. **Seek negative evidence.** The literature is biased toward positive findings.
   Actively look for contradictory datasets, failed replications, null perturbations,
   and boundary conditions, and let them erode belief.
4. **Guard against leakage.** Data used to *generate* a hypothesis must never be used
   to *validate* it. Discount non-independent support.
5. **Every conclusion needs a falsification signature.** Name the experiment,
   biomarker, or patient phenotype that would break it. If nothing could, it isn't yet
   science.
6. **End by nominating the next perturbation.** Correlation → mechanism happens through
   an experiment; say which one, and why it has the highest expected information gain.

## Reading a result

`report.final_report` gives the headline mechanism, ranked survivors (with belief,
confidence, evidence score, weakest link, independent-support count, and the actual
causal chain), the eliminated hypotheses **with reasons**, and the chosen experiments.
`report.experiment_plan` gives the EIG-ranked experiments and the expected drop in
uncertainty (bits). `report.stages` shows uncertainty falling across the run — that
convergence is the point; if it doesn't fall, something is off (usually too few
distinct hypotheses or missing validation evidence).

## Extending

- **New knowledge/mechanism/translation agent** (LLM-backed): subclass
  `ScientificAgent` (`agent_base.py`), set `agent_id`, `name`, `layer`, `purpose`,
  `tools`, and a `system_prompt` (add it to `prompts.py` with a strict JSON contract),
  implement `integrate()` to fold the parsed JSON into the `ResearchContext` (create
  `Evidence`, `Hypothesis`, `Dataset`, or graph edges), and a deterministic `_offline()`
  stub. Register the class in `agents/registry.py`.
- **New deterministic agent** (operates on the graph/pool): override `act()` instead of
  `integrate()`, like `agents/special.py` (causal-graph audit, negative-evidence,
  experiment planner) and `layer4.StatisticsAgent`.
- **New evidence tier, node type, or experiment type**: extend the enums in
  `evidence.py` / `experiment.py`; keep tier weights convex so top tiers dominate.
- **Regenerate `configs/agents.yaml`** after roster changes (it is auto-generated from
  the registry — see the command in the repo history) and keep the README roster table
  in sync.
- **Always run `python -m pytest -q`** after changes. The suite pins the belief math,
  chain scoring, EIG, and the end-to-end offline convergence; if you change a weight,
  a test should move, and you should understand why.

## Guardrails

- Keep the belief and EIG weights transparent and tunable — they are heuristics meant
  to be inspected and argued with, not hidden. Don't bury them in magic constants.
- Don't let the offline stubs drift into asserting real findings. They are scaffolding.
- In live mode the system is only as good as the evidence its tools return and the
  calibration of the agents' likelihood estimates — that is exactly why the Statistics,
  Negative Evidence, and Skeptic agents exist. Surface that uncertainty; don't launder
  a weak chain into a confident claim.
