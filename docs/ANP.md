# ANP contexts on top of cardinal pairwise ratios

This document defines the minimal Analytic Network Process (ANP) layer in `cardinal-harness`.

Core primitive stays unchanged:

`rater::attribute_prompt::entity_A::entity_B::ratio::confidence`

The ANP layer adds typed context so agents can decompose open-ended prioritization problems without forcing every prompt into a globally composable scale.

## Typed judgment contexts

Each context specifies:

- relation type: `preference` or `influence`
- source cluster: where compared entities come from
- target node: "with respect to" node
- `judgment_kind`:
  - `composable_ratio` for globally propagatable ratio judgments
  - `pairwise_only_ratio` for local comparisons that should not be propagated
- semantics version: explicit scoped assumptions for time horizon, geography, and perspective

## Local fit model

For one context, we fit sparse pairwise judgments in log space:

`log(ratio_ab) ~= s_a - s_b`

using confidence-weighted least squares with light ridge regularization.

Outputs include:

- local priorities for source-cluster nodes (normalized)
- residual diagnostics (`weighted_rmse`, `mean_abs_residual`)
- uncertainty proxy (`diag_cov`) for active pair selection
- suggested kind (`composable_ratio` vs `pairwise_only_ratio`) based on fit consistency

## Global ANP aggregation

`build_weighted_supermatrix` assembles a column-stochastic supermatrix from contexts marked `composable_ratio` and local fit outputs.

Columns without incoming composable contexts default to self-loops, which keeps the matrix well-defined.

`solve_stationary` computes damped stationary priorities:

`v_{t+1} = damping * W * v_t + (1 - damping) * teleport`

Use `cluster_priorities` to extract normalized priorities for a specific cluster (for example, alternatives).

## Active query helpers

Use:

- `rank_contexts_for_query` to score contexts by sensitivity, uncertainty, inconsistency, and exploration
- `select_next_pair` to choose the highest-variance pair in a fitted context
- `propose_next_query` to return context + pair for the next judgment

## Minimal usage

```rust
use std::collections::HashMap;
use cardinal_harness::anp::{
    build_weighted_supermatrix, fit_context, solve_stationary, AnpNetwork, Cluster, JudgmentKind,
    JudgmentContext, LocalFitConfig, Node, PairwiseJudgment, RelationType, StationaryConfig,
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let alt_nodes = vec![
    Node { id: "a".into(), cluster_id: "alts".into(), label: "A".into() },
    Node { id: "b".into(), cluster_id: "alts".into(), label: "B".into() },
];

let ctx = JudgmentContext {
    id: "ctx".into(),
    relation_type: RelationType::Preference,
    target_node_id: "criterion_x".into(),
    source_cluster_id: "alts".into(),
    prompt_text: "Over the next 24 months, how many times more tractable is A than B?".into(),
    semantics_version: 1,
    judgment_kind: JudgmentKind::ComposableRatio,
    incoming_cluster_weight: None,
};

let judgments = vec![
    PairwiseJudgment {
        context_id: "ctx".into(),
        entity_a_id: "a".into(),
        entity_b_id: "b".into(),
        ratio: 2.0,
        confidence: 0.9,
        rater_id: "r1".into(),
        notes: None,
    },
];

let fit = fit_context(&ctx, &alt_nodes, &judgments, &LocalFitConfig::default())?;

let network = AnpNetwork {
    clusters: vec![
        Cluster { id: "alts".into(), label: "Alternatives".into() },
        Cluster { id: "criteria".into(), label: "Criteria".into() },
    ],
    nodes: vec![
        Node { id: "criterion_x".into(), cluster_id: "criteria".into(), label: "Criterion X".into() },
        Node { id: "a".into(), cluster_id: "alts".into(), label: "A".into() },
        Node { id: "b".into(), cluster_id: "alts".into(), label: "B".into() },
    ],
    contexts: vec![ctx],
};

let mut fits = HashMap::new();
fits.insert(fit.context_id.clone(), fit);

let supermatrix = build_weighted_supermatrix(&network, &fits)?;
let stationary = solve_stationary(&supermatrix, &StationaryConfig::default())?;
println!("{:?}", stationary.distribution);
# Ok(())
# }
```

## Design notes

- This is intentionally ANP-lite and diagnostics-first.
- `pairwise_only_ratio` is first-class and preserved in data model.
- Local inconsistency is a signal for prompt rewrite and decomposition, not a reason to force a global projection.

## CLI support

```bash
# Run a full ANP pass from JSON request to JSON output
cargo run --bin cardinal -- anp-demo --input anp_request.json --out anp_output.json

# Run synthetic typed-vs-forced benchmark (JSONL)
cargo run --bin cardinal -- eval-anp --out anp_eval.jsonl
```
