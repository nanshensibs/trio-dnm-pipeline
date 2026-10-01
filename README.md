# Lupus Co-Scientist

A **mechanism-discovery** multi-agent system for systemic lupus erythematosus — not
merely a hypothesis *generator*. Where Google's Co-Scientist optimizes for producing
plausible ideas, this system is built to **progressively eliminate weak hypotheses
while accumulating causal evidence**, and to optimize the path from mechanism to
translational impact.

It is designed around the Shen-lab program (ABC / atypical B cells, the ZEB2–MEF2B
bistable switch, IRF5 activity, interferon endotypes, lupus-nephritis tissue damage,
CAR-T immune resetting) and the discipline that program runs on: *causal, not
descriptive*.

```
Research Question
   → Knowledge Integration      (Layer 1)
   → Mechanism Discovery        (Layer 2)   ── populates the Causal Evidence Graph
   → Critical Challenge         (Layer 3)   ── generate • counter • review • novelty
   → Experimental Design        (Layer 4)   ── + independent validation, stats audit
   → Public Data Validation     (Layer 4)
   → Clinical Translation       (Layer 5)
   → Meta Review                (Layer 6)   ── debate • consensus • skeptic • report
```

## What makes it different

Three capabilities, implemented as **real code** in the deterministic reasoning core
(not just prompt text), turn a general hypothesis generator into a biomedical
discovery co-scientist:

1. **Causal Evidence Graph** (`evidence.py`). Evidence is tiered by how causal it is —
   human genetics › perturbation › animal › association › review — and assembled into
   a directed, signed graph running from *variant → gene → mechanism → cell state →
   cell phenotype → tissue pathology → clinical outcome*. **Every hypothesis must be
   expressible as a traceable chain in this graph, and its strength is bounded by the
   weakest link in that chain.** A hypothesis supported only by correlative papers,
   however many, cannot outrank one backed by a fine-mapped variant plus a Perturb-seq
   confirmation.

2. **Negative Evidence Agent** (`agents/special.py`). The literature is biased toward
   positive findings. This agent actively hunts contradictory datasets, failed
   replications, null perturbations, and boundary conditions, and folds them in as
   first-class *negative* evidence that erodes belief. It is run twice — before and
   after validation — so confirmation bias is attacked at both ends.

3. **Decision-Theoretic Experiment Planner** (`experiment.py`). Instead of a wish-list,
   it estimates the **expected information gain (EIG)** of each candidate experiment —
   how much it is expected to reduce Shannon entropy over the competing hypotheses —
   and greedily selects the smallest, cheapest set that most sharply discriminates the
   survivors. This is standard Bayesian experimental design:
   `EIG(e) = H(prior) − E_outcome[ H(posterior | outcome) ]`.

The result is a system that **converges**: uncertainty (in bits) over the live
hypotheses provably decreases across the run, and the output names the single
perturbation most worth doing next.

## The 31 agents

28 specialized scientific agents across 6 layers, plus the 3 novel cross-cutting
capabilities above. Run `python -m lupus_coscientist list` for the full roster, or see
[`configs/agents.yaml`](configs/agents.yaml).

| Layer | Agents |
|-------|--------|
| **1 · Knowledge Acquisition** | Literature · Evidence Ranking · Public Database · Clinical Guideline |
| **2 · Mechanism Discovery** | Cell State · Trajectory · Regulatory Network · Ligand–Receptor · Genetic Causality · Evolutionary · Tissue Ecology · Metabolism |
| **3 · Hypothesis Generation** | Mechanism Generator · Counterfactual · Alternative Explanation · Reviewer · Novelty |
| **4 · Validation** | Public Data Validation · Experimental Design · Statistics |
| **5 · Translation** | Drug Discovery · Biomarker · Precision Medicine · Trial Design |
| **6 · Meta Reasoning** | Debate · Consensus · Skeptic · Meta Reviewer |
| **× · Novel (cross-cutting)** | Causal Evidence Graph · Negative Evidence · Decision-Theoretic Experiment Planner |

Each agent declares its role, the causal-tiering discipline, program alignment
(bistable switch / Clone Ecology / fork model), and a strict JSON output contract
(`prompts.py`). The knowledge, mechanism, and translation agents call an LLM with real
scientific tools; the ranking, causal-graph, negative-evidence, statistics, debate,
skeptic, planner, and meta-review agents run deterministic logic over the shared
evidence graph and hypothesis pool.

## Install & run

```bash
pip install -e .              # core only, zero runtime dependencies
pip install -e ".[dev]"       # + pytest
```

**Offline demo** (deterministic, no API key — exercises the whole orchestration):

```bash
python -m lupus_coscientist investigate \
  "What is the causal role of IRF5 in ABC-driven lupus nephritis?" --focus IRF5
```

or from Python:

```python
from lupus_coscientist import LupusCoScientist
report = LupusCoScientist().investigate(
    "What is the causal role of IRF5 in ABC-driven lupus nephritis?", focus="IRF5")
print(report.final_report["headline_mechanism"])
```

The offline stubs are clearly schematic (they never present fabricated findings as
real literature); they exist so the pipeline, tests, and the worked IRF5 example run
end-to-end with no network. See [`examples/run_irf5.py`](examples/run_irf5.py).

**Live mode** (real literature / genetics / trial evidence) uses the Claude Agent SDK
with the scientific MCP servers (PubMed, Consensus, Open Targets, ClinicalTrials,
alphaXiv, …):

```python
from lupus_coscientist import LupusCoScientist, ClaudeAgentClient
from lupus_coscientist.mcp_config import default_server_template

servers = default_server_template()          # fill in real endpoints/keys
engine = LupusCoScientist(client=ClaudeAgentClient(mcp_servers=servers))
report = engine.investigate("...", focus="IRF5")
```

```bash
python -m lupus_coscientist investigate "..." --focus IRF5 --live
```

## How the reasoning core works

| Module | Responsibility |
|--------|----------------|
| `evidence.py` | `EvidenceTier`, `Evidence`, `CausalEdge`, `CausalChain`, `CausalEvidenceGraph` — tiering, signed edges, weakest-link chain scoring, contradiction detection. |
| `hypothesis.py` | `Hypothesis`, `HypothesisPool` — logistic belief update (chain-dominated), progressive elimination, belief-distribution entropy. |
| `experiment.py` | `Experiment`, `DecisionTheoreticPlanner` — expected information gain and budgeted greedy selection. |
| `context.py` | `ResearchContext` — the shared blackboard (graph, pool, datasets, experiments, provenance log), with discovery/validation dataset separation to block data leakage. |
| `agent_base.py`, `agents/*` | The 31 agents. |
| `orchestrator.py` | The six-layer pipeline with interleaved belief refresh and elimination rounds. |

**Belief.** A hypothesis's belief is updated in log-odds space and is *dominated by its
causal chain*: no traceable chain to a clinical endpoint ⇒ a hard penalty, no matter
how many correlative papers mention it. Independent validation both endorses the
hypothesis *and* accumulates causal evidence on its chain edges (upgrading the
"proposed" links), so surviving mechanisms literally get more causal over the run.

**Elimination is gentle early, decisive late.** Round 1 (post-debate, pre-validation)
keeps the field open (`min_survivors` protection). Round 2 (post-validation) allows the
field to narrow to the single best-supported mechanism — while deliberately retaining
the strongest live *alternative* so the experiment planner has something to
discriminate.

**Guards against the usual failure modes.** Discovery datasets are tracked and barred
from validation (leakage); non-independent support is discounted; review-tier evidence
is strength-capped so opinion cannot masquerade as causation; the Statistics agent
auto-flags leaked support; the Skeptic agent demands a falsification signature for every
surviving conclusion.

## Tests

```bash
python -m pytest -q
```

The suite covers evidence tiering, weakest-link chain scoring, chain strengthening,
contradiction detection, belief monotonicity, pruning/entropy, EIG (a discriminating
experiment beats a correlative one; the planner respects budget and reduces entropy),
and an end-to-end offline run that converges on the switch-flip mechanism and
eliminates the artifact hypothesis.

## Status & honesty

This is a research scaffold. The offline scenario is illustrative, and the belief /
EIG weights are transparent, tunable heuristics — deliberately simple so they can be
inspected and argued with, which is the point. In live mode the framework is only as
good as the evidence its tools return and the calibration of the agents' likelihood
estimates; the Statistics, Negative Evidence, and Skeptic agents exist precisely
because those estimates will sometimes be wrong.
