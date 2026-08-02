# Lupus Co-Scientist — data model & reasoning core

The authoritative diagrams live in `docs/ARCHITECTURE.md` at the repo root. This file
is the condensed version a model needs to reason over or extend the framework safely.

## Modules

| Module | Responsibility |
|--------|----------------|
| `evidence.py` | `EvidenceTier`, `Evidence`, `CausalEdge`, `CausalChain`, `CausalEvidenceGraph` — tiering, signed/directed edges, weakest-link chain scoring, contradiction detection. |
| `hypothesis.py` | `Hypothesis`, `HypothesisPool` — log-odds belief update, progressive elimination, belief-distribution entropy. |
| `experiment.py` | `Experiment`, `PredictedOutcome`, `DecisionTheoreticPlanner` — expected information gain and budgeted greedy selection. |
| `context.py` | `ResearchContext` — shared blackboard (graph, pool, datasets, experiments, evidence registry, provenance log) + discovery/validation dataset separation. |
| `agent_base.py`, `agents/` | `ScientificAgent` base + the 31 agents (`layer1..6`, `special`, `registry`). |
| `orchestrator.py` | Six-layer pipeline with interleaved belief refresh and elimination rounds. |
| `prompts.py`, `parsing.py`, `llm.py`, `report.py`, `mcp_config.py`, `cli.py` | Prompts, JSON→enum coercion, LLM adapters, report renderer, MCP wiring, CLI. |

## Evidence tiers (the ranking discipline, as code)

`EvidenceTier` (IntEnum, higher = more causal) with convex weights so top tiers
dominate aggregate scores:

| Tier | int | weight |
|------|----:|------:|
| HUMAN_GENETICS | 5 | 1.00 |
| PERTURBATION | 4 | 0.85 |
| ANIMAL | 3 | 0.60 |
| ASSOCIATION | 2 | 0.35 |
| REVIEW | 1 | 0.15 |

`Evidence.signed_weight = tier.weight · strength · sign`, ×0.4 if `is_independent`
is False (a leakage guard). `Polarity.CONTRADICTS` makes it negative.

## The causal chain

Node types run downhill from genotype to clinic: `VARIANT → GENE → PERTURBATION/
MOLECULAR_MECHANISM → CELL_STATE → CELL_PHENOTYPE → TISSUE_PATHOLOGY →
CLINICAL_OUTCOME` (plus `DRUG`, `BIOMARKER`). `CausalEdge.support` uses the strongest
supporting line as a backbone with diminishing returns from corroboration, eroded by
negative evidence. `CausalChain.score = weakest_link · completeness` where completeness
adds a bonus for spanning to a clinical endpoint and starting at genotype.

Each `Hypothesis` owns its declared `chain_nodes`; `graph.chain_from_nodes()` scores
that specific path (so hypotheses sharing endpoints don't cross-contaminate). Mechanism
Generator lays each hypothesis's chain as REVIEW-tier "proposed" edges; perturbation,
genetic, and **independent validation** evidence then land on those same edges and
upgrade them — this is how surviving mechanisms literally get more causal over a run.

## Belief update (log-odds, per hypothesis)

```
logit(belief) = logit(prior) + chain_term + Σ signed_evidence + debate_delta
chain_term    = 4·chain_score − 0.5      # chain_score>0
chain_term    = −2.0                       # no traceable chain to a clinical endpoint
```

The causal chain is the dominant *positive* signal; hypothesis-level evidence
(validation up; reviewer/negative/statistics down) adjusts it. No chain ⇒ hard penalty,
regardless of how many correlative papers exist.

## Progressive elimination

`HypothesisPool.prune()` eliminates a hypothesis when it is dominated on the evidence:
belief below floor, OR no traceable chain, OR buried under ≥`max_negative` contradicting
lines with no perturbation/genetic support — while always protecting the top
`min_survivors`. The orchestrator prunes **gently** in round 1 (pre-validation, keep the
field open) and **decisively** in round 2 (post-validation, `min_survivors=1`), while
retaining the strongest live alternative so the experiment planner has something to
discriminate. `HypothesisPool.entropy()` (bits over the normalized belief distribution)
is the convergence metric — it should fall across the run.

## Decision-theoretic experiment planning

For each candidate experiment with a likelihood table `P(outcome | hypothesis)`:

```
EIG(e) = H(prior over live hyps) − Σ_outcome P(outcome)·H(posterior | outcome)     [bits]
```

scaled by feasibility. `DecisionTheoreticPlanner.plan()` ranks by `EIG / lab-months`,
then greedily selects under a budget, Bayes-updating the belief after each pick so it
never buys two experiments that discriminate the same pair. Output: the smallest
discriminating set + expected residual uncertainty.

## Pipeline order (orchestrator)

Layer 1 (Literature → Evidence Ranking → Public Database → Clinical Guideline) →
Layer 2 mechanism agents populate the graph → **Causal Graph audit** → Layer 3
(Mechanism Generator → Alternative → Counterfactual → Reviewer → Novelty) →
**Causal Graph audit + Negative Evidence** → Debate → **Prune #1 (gentle)** →
Layer 4 (Public Validation → Statistics → Experimental Design) →
**Negative Evidence → Prune #2 (decisive)** → **Decision-Theoretic Experiment Planner** →
Layer 5 (Drug Discovery → Biomarker → Precision Medicine → Trial Design) →
Layer 6 (Debate → Consensus → Skeptic → Meta Reviewer).
