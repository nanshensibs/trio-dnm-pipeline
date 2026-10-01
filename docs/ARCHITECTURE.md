# Architecture

## Evidence flow

```mermaid
flowchart TD
    Q[Research question] --> LIT[Literature]
    LIT --> RANK[Evidence Ranking]
    RANK --> INT[Evidence Integration<br/>Layer 2 mechanism agents populate<br/>the Causal Evidence Graph]
    INT --> CG1{{Causal Graph audit}}
    CG1 --> GEN[Mechanism Generator<br/>20–50 candidates]
    GEN --> ALT[Alternative Explanation]
    GEN --> CF[Counterfactual]
    ALT --> CHAL
    CF --> CHAL
    GEN --> CHAL[Reviewer + Novelty challenge]
    CHAL --> NEG1{{Negative Evidence hunt}}
    NEG1 --> DEB1[Scientific Debate]
    DEB1 --> P1[/Prune round 1 · gentle/]
    P1 --> VAL[Public Data Validation<br/>independent data only]
    VAL --> STAT[Statistics audit]
    STAT --> EXP[Experimental Design<br/>likelihood tables]
    EXP --> NEG2{{Negative Evidence hunt}}
    NEG2 --> P2[/Prune round 2 · decisive/]
    P2 --> PLAN{{Decision-Theoretic<br/>Experiment Planner · EIG}}
    PLAN --> TRANS[Clinical Translation<br/>drug · biomarker · endotype · trial]
    TRANS --> META[Meta Review<br/>debate · consensus · skeptic · report]

    classDef novel fill:#ffe8cc,stroke:#e8710a,color:#000;
    class CG1,NEG1,NEG2,PLAN novel;
```

The four orange nodes are the novel cross-cutting capabilities. Everything between
`Prune round 1` and `Prune round 2` is where the system *accumulates causal evidence*
(validation upgrades chain edges) while *eliminating weak hypotheses* (negative
evidence + statistics + debate erode belief).

## The causal chain a hypothesis must own

```mermaid
flowchart LR
    V[variant] -->|human genetics| G[gene]
    G -->|perturbation| M[molecular<br/>mechanism]
    M --> CS[cell state]
    CS --> CP[cell<br/>phenotype]
    CP --> TP[tissue<br/>pathology]
    TP --> CO[clinical<br/>outcome]
```

Belief is bounded by the **weakest edge** in this chain. Proposed links start at
review-tier and are upgraded only when perturbation / genetic / independent-validation
evidence lands on them — so a mechanism becomes believable exactly to the degree it
becomes causal.

## Evidence tiers (Evidence Ranking Agent)

| Tier | Weight | Example |
|------|-------:|---------|
| Human genetics (fine-map + coloc + functional SNP) | 1.00 | `rs2004640 → IRF5`, colocalized eQTL |
| Perturbation (CRISPR / Perturb-seq / base-edit) | 0.85 | B-cell `Irf5` base-edit KO abolishes ABCs |
| Animal (in vivo) | 0.60 | `Irf5⁻/⁻` lupus mouse |
| Association (observational, scRNA DE, bulk) | 0.35 | ABC fraction correlates with IRF5 |
| Review / opinion | 0.15 | narrative review |

## Belief update (per hypothesis, log-odds space)

```
logit(belief) = logit(prior)
              + chain_term          # dominant positive signal; −2.0 if no chain
              + Σ signed evidence    # validation up, reviewer/negative/stats down
              + debate_delta
chain_term    = 4·chain_score − 0.5          (chain_score = weakest-link · completeness)
signed_weight = tier_weight · strength · sign  (×0.4 if not independent → leakage guard)
```

## Decision-theoretic experiment planning

For each candidate experiment with a predicted-outcome likelihood table
`P(outcome | hypothesis)`:

```
EIG(e) = H(prior over live hypotheses) − Σ_outcome P(outcome)·H(posterior | outcome)
```

The planner ranks by `EIG / lab-months`, then greedily selects under a budget,
Bayes-updating the belief after each pick so it never buys two experiments that
discriminate the same pair. Output: the smallest discriminating set and the expected
residual uncertainty in bits.
