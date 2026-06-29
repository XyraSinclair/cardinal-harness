# cardinal-harness

`cardinal-harness` is the canonical pairwise-ratio elicitation engine for OpenPriors.

It does one job: turn noisy LLM pairwise ratio judgements into globally consistent cardinal scores with uncertainty, then spend the next comparison where it buys the most information.

Use it when direct scores are unstable but pairwise comparisons are meaningful: research ideas, prompts, candidate plans, reviewer notes, or any shortlist where "how much better?" carries information. Do not use it for deterministic rankings, scalar metrics, or cases where the attribute itself is incoherent.

The trade is explicit: it costs more than one-shot scoring, saves comparisons versus exhaustive pairwise judging, and returns uncertainty plus receipts instead of only a sorted list.

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

## Prompt surfaces

Two prompt templates are supported:

| Slug | Output shape | Use when |
|------|--------------|----------|
| `canonical_v2` | `{"higher_ranked":"A|B","ratio":1.0..26.0,"confidence":0.0..1.0}` | Default pairwise-ratio judgement. Use this unless you specifically need bucket-token logprobs. |
| `canonical_bucket_v1` | `{"higher_ranked":"A|B","ratio_bucket":0..16,"confidence":0.0..1.0}` | Bucket-index variant for runs that need to map output logprobs onto the fixed ratio ladder. |

Both templates use the same ratio ladder and the same refusal shape: `{"refused":true}`. Unknown `prompt_template_slug` values are rejected. Details live in [docs/PROMPTS.md](docs/PROMPTS.md).

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
    req,
    cardinal_harness::rerank::RerankExecution::new(
        Arc::new(gateway),
        Attribution::new("example::quickstart"),
    )
    .cache(&cache)
    .model_policy(Arc::new(ModelLadderPolicy::default()))
    .run_options(RerankRunOptions { rng_seed: None, cache_only: false }),
).await?;

for entity in &resp.entities {
    println!("{} {:?}: {:.3} ± {:.3}", entity.id, entity.rank, entity.u_mean, entity.u_std);
}
# Ok(())
# }
```

## CLI

The CLI `rerank` command reads a `MultiRerankRequest` JSON file. A copy-pasteable example lives at [`examples/multi-rerank-request.json`](examples/multi-rerank-request.json). Use `validate` first when you want schema and invariant checks without an API key, cache, or network call.

```bash
# Validate request JSON locally before running a model.
cargo run --bin cardinal -- validate \
  --request examples/multi-rerank-request.json

export OPENROUTER_API_KEY=your_key_here

# Rerank from JSON. The example includes both canonical_v2 and canonical_bucket_v1 attributes.
cargo run --bin cardinal -- rerank \
  --request examples/multi-rerank-request.json \
  --out output.json \
  --trace trace.jsonl \
  --report report.md

# Generate a markdown or JSON report later from a saved request + response.
cargo run --bin cardinal -- report \
  --request examples/multi-rerank-request.json \
  --response output.json \
  --out report.md

# Simple single-attribute request shape for library/API callers.
# See examples/simple-rerank-request.json; the current CLI accepts the multi-rerank shape above.
```

Other maintenance commands:

```bash
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


Data flow:

```text
request JSON
  -> rerank manager
  -> per-attribute rating engines
  -> active planner
  -> OpenRouter gateway + SQLite cache
  -> trace JSONL + response JSON + markdown report
```

## Documentation

- [docs/ALGORITHM.md](docs/ALGORITHM.md): scoring, uncertainty, stopping, and evaluation rationale
- [docs/MODEL.md](docs/MODEL.md): compact mathematical contract, assumptions, and failure modes
- [docs/PROMPTS.md](docs/PROMPTS.md): supported prompt templates, output contracts, and JSON request examples
- [docs/WORKED_EXAMPLE.md](docs/WORKED_EXAMPLE.md): concrete rerank walkthrough with request shape, gates, stop reasons, uncertainty, cache, and reproducibility receipts
- [docs/EVALUATION.md](docs/EVALUATION.md): checked-in synthetic evaluation receipt and baseline comparison
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): scaling harness and current dense-solver receipt

## License

MIT
