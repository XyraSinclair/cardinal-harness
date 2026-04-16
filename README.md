# cardinal-harness

`cardinal-harness` is the canonical pairwise-ratio elicitation engine for OpenPriors.

It does one job: turn noisy LLM pairwise ratio judgements into globally consistent cardinal scores with uncertainty, then spend the next comparison where it buys the most information.

## Scope

This repo is intentionally narrow. It contains:

- canonical pairwise ratio prompts
- the ratio ladder and JSON judgement contract
- robust score fitting over pairwise observations
- multi-attribute reranking, gating, and stopping
- OpenRouter gateway, pricing, usage, and SQLite cache support
- synthetic evaluation and reporting

Research workflows, training/export code, agent orchestration, and other experimental layers belong in `openpriors-research`, not here.

## Core idea

Instead of asking an LLM for unstable absolute scores, ask:

> How many times more of attribute X does A have than B?

Each answer becomes a noisy log-ratio observation. `cardinal-harness` fits latent scores that best explain the full comparison graph, tracks uncertainty, and stops once top-k is sufficiently certain.

## Prompt

There is one supported prompt template: `canonical_v2`.

- slug: `canonical_v2`
- answer shape: `{"higher_ranked":"A|B","ratio":1.0..26.0,"confidence":0.0..1.0}`
- refusal shape: `{"refused":true}`

Details live in [docs/PROMPTS.md](docs/PROMPTS.md).

## Quickstart

```bash
export OPENROUTER_API_KEY=your_key_here
cargo run --example quickstart
```

Library use:

```rust,no_run
use std::sync::Arc;
use cardinal_harness::{Attribution, ProviderGateway, SqlitePairwiseCache};
use cardinal_harness::gateway::NoopUsageSink;
use cardinal_harness::rerank::{
    ModelLadderPolicy, MultiRerankAttributeSpec, MultiRerankEntity, MultiRerankRequest,
    MultiRerankTopKSpec, RerankRunOptions,
};

# async fn demo() -> Result<(), Box<dyn std::error::Error>> {
let cache = SqlitePairwiseCache::new(SqlitePairwiseCache::default_path())?;
let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;

let req = MultiRerankRequest {
    entities: vec![
        MultiRerankEntity { id: "a".into(), text: "First essay...".into() },
        MultiRerankEntity { id: "b".into(), text: "Second essay...".into() },
        MultiRerankEntity { id: "c".into(), text: "Third essay...".into() },
    ],
    attributes: vec![MultiRerankAttributeSpec {
        id: "clarity".into(),
        prompt: "clarity of explanation".into(),
        prompt_template_slug: Some("canonical_v2".into()),
        weight: 1.0,
    }],
    topk: MultiRerankTopKSpec {
        k: 2,
        ..serde_json::from_str("{}").unwrap()
    },
    gates: vec![],
    comparison_budget: None,
    latency_budget_ms: None,
    model: None,
    rater_id: None,
    comparison_concurrency: None,
    max_pair_repeats: None,
    randomize_presentation_order: true,
};

let resp = cardinal_harness::rerank::multi_rerank(
    Arc::new(gateway),
    Some(&cache),
    Some(Arc::new(ModelLadderPolicy::default())),
    Some(&RerankRunOptions { rng_seed: None, cache_only: false }),
    req,
    Attribution::new("example::quickstart"),
    None,
    None,
    None,
).await?;

for entity in &resp.entities {
    println!("{} {:?}: {:.3} ± {:.3}", entity.id, entity.rank, entity.u_mean, entity.u_std);
}
# Ok(())
# }
```

## CLI

```bash
# Rerank from JSON
cargo run --bin cardinal -- rerank --request input.json --out output.json --trace trace.jsonl

# Generate a markdown or JSON report
cargo run --bin cardinal -- report --request input.json --response output.json --out report.md

# Synthetic evaluation
cargo run --bin cardinal -- eval --out eval.jsonl --curve-csv curves.csv
cargo run --bin cardinal -- eval-likert --out eval_likert.jsonl --curve-csv curves_likert.csv

# Cache management
cargo run --bin cardinal -- cache-export --out cache.jsonl
cargo run --bin cardinal -- cache-prune --max-age-days 30
```

## Architecture

| Module | Purpose |
|--------|---------|
| `rating_engine` | Robust IRLS solver and comparison planning |
| `trait_search` | Multi-attribute utility composition, gating, top-k uncertainty |
| `rerank` | Orchestration loop, comparison execution, stopping, traces, reports |
| `prompts` | Canonical pairwise ratio prompt and ratio ladder |
| `cache` | SQLite-backed memoization for pairwise judgements |
| `gateway` | OpenRouter client, pricing, usage, attribution |
| `text_chunking` | Token-aware chunking helpers |

## Documentation

- [docs/ALGORITHM.md](docs/ALGORITHM.md): scoring, uncertainty, stopping, and evaluation rationale
- [docs/PROMPTS.md](docs/PROMPTS.md): the `canonical_v2` prompt contract

## License

MIT
