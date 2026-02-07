# cardinal-harness

The highest quality way to have LLMs put numbers on things.

## The problem

You have a list of items — essays, candidates, proposals, models, anything — and you need quantitative scores, not just a vague ranking. Asking an LLM to "rate this 1–10" is fast but unreliable: scores are miscalibrated, inconsistent across items, and anchored to arbitrary reference points. You can't meaningfully say item A scored 7.2 and item B scored 6.8 and trust the difference.

## The approach

Instead of absolute scores, **cardinal-harness** asks pairwise ratio questions: *"how many times more [attribute] does A have than B?"* These relative judgments are far more reliable than absolute scores — the same way humans can tell which of two objects is heavier much more accurately than they can guess either object's weight.

Each pairwise ratio judgment (e.g., "A is 2.5× clearer than B") becomes a noisy observation of latent scores in log-space. A robust statistical solver (IRLS with Huber loss) combines all pairwise observations into a globally consistent set of scores, automatically downweighting outlier judgments where the LLM was confused or inconsistent.

The system tracks its own uncertainty. It knows which items' scores are well-determined and which aren't, and it selects the next most informative pair to query — so it converges on accurate top-K results with the minimum number of LLM calls. It stops when it's confident enough in the ranking, not after a fixed number of comparisons.

**"Cardinal"** means you get actual numeric scores on a ratio scale — not just ordinal rankings (1st, 2nd, 3rd) but quantitative latent values with uncertainty estimates, so you know both the ranking and how much each item differs from its neighbors.

## Features

- **Pairwise ratio prompts** on a fixed ladder (1.0 .. 26.0) for consistent, calibrated LLM judgments
- **Robust IRLS solver** turns noisy ratio observations into globally consistent latent scores with uncertainty
- **Multi-attribute utility** with gates and weighted top-k focus across multiple dimensions
- **Dynamic query planning** proposes only the most informative pairs, minimizing LLM calls
- **Uncertainty-aware stopping** — stops when top-k is sufficiently certain, not after a budget
- **SQLite cache** for pairwise judgments — repeated runs reuse prior LLM calls
- **OpenRouter integration** for model access with dynamic model-ladder switching

## Quickstart

```bash
export OPENROUTER_API_KEY=your_key_here
cargo run --example quickstart
```

Or use as a library:

```rust,no_run
use std::sync::Arc;
use cardinal_harness::{Attribution, ProviderGateway, SqlitePairwiseCache};
use cardinal_harness::rerank::{
    MultiRerankAttributeSpec, MultiRerankEntity, MultiRerankRequest,
    MultiRerankTopKSpec, ModelLadderPolicy, RerankRunOptions,
};
use cardinal_harness::gateway::NoopUsageSink;

# async fn demo() -> Result<(), Box<dyn std::error::Error>> {
let cache = SqlitePairwiseCache::new(SqlitePairwiseCache::default_path())?;
let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;

let req = MultiRerankRequest {
    // The items to rank — each has a stable id and the text the LLM will see.
    entities: vec![
        MultiRerankEntity { id: "a".into(), text: "First essay...".into() },
        MultiRerankEntity { id: "b".into(), text: "Second essay...".into() },
        MultiRerankEntity { id: "c".into(), text: "Third essay...".into() },
    ],
    // What to score on. Each attribute gets independent pairwise comparisons.
    attributes: vec![
        MultiRerankAttributeSpec {
            id: "clarity".into(),
            prompt: "clarity of explanation".into(),
            prompt_template_slug: Some("canonical_v2".into()),
            weight: 1.0,
        },
    ],
    // Identify the best 2, stop when ~90% confident (tolerated_error=0.1).
    topk: MultiRerankTopKSpec {
        k: 2,
        ..serde_json::from_str("{}").unwrap() // sensible defaults
    },
    gates: vec![],
    comparison_budget: None,
    latency_budget_ms: None,
    model: None,
    rater_id: None,
    comparison_concurrency: None,
    max_pair_repeats: None,
};

let resp = cardinal_harness::rerank::multi_rerank(
    Arc::new(gateway),
    Some(&cache),
    Some(Arc::new(ModelLadderPolicy::default())),
    Some(&RerankRunOptions { rng_seed: None, cache_only: false }),
    req,
    Attribution::new("example::quickstart"),
    None, None, None,
).await?;

// Each entity now has quantitative scores with uncertainty:
for e in &resp.entities {
    println!("{} (rank {:?}): {:.3} ± {:.3}", e.id, e.rank, e.u_mean, e.u_std);
}
# Ok(())
# }
```

See `examples/quickstart.rs` for a fully commented version.

## Pairwise cache

The cache stores judgements keyed on:
- model
- prompt template slug
- prompt template hash
- attribute id + prompt hash
- entity ids + text hashes

This avoids repeated LLM calls across runs. Set a custom path via:

```bash
export CARDINAL_CACHE_PATH=/path/to/cache.sqlite
```

Prune the cache by age or size:

```bash
cargo run --bin cardinal -- cache-prune --max-age-days 30
cargo run --bin cardinal -- cache-prune --max-rows 100000
```

## Models

OpenRouter model ids are accepted directly. Defaults:
- `openai/gpt-5-mini` (default in code)
- `moonshotai/kimi-k2-0905`
- `anthropic/claude-opus-4.5`

### Dynamic model ladder

`ModelLadderPolicy` lets you start high quality and downgrade when uncertainty is low.
Tune thresholds to your domain if you want more (or less) aggressive switching.

## Architecture

- `rating_engine`: robust IRLS solver, diagnostics, and pair planning
- `trait_search`: multi-attribute utility, gating, top-k uncertainty
- `rerank`: orchestration loop + pairwise LLM calls
- `prompts`: ratio ladder prompt templates
- `cache`: SQLite-backed pairwise memoization
- `gateway`: OpenRouter client + usage tracking

See `docs/ALGORITHM.md` for full design rationale — why pairwise ratios, why IRLS, why Huber loss, how the stopping rule works, and more.
See `docs/PROMPTS.md` for prompt template slugs and context placement details.

## CLI

```bash
# Export cache to JSONL
cargo run --bin cardinal -- cache-export --out cache.jsonl

# Prune cache by age or size
cargo run --bin cardinal -- cache-prune --max-age-days 30
cargo run --bin cardinal -- cache-prune --max-rows 100000

# List built-in policies
cargo run --bin cardinal -- policy list

# Load a policy config (JSON)
cargo run --bin cardinal -- policy load --config policy.json

# Run synthetic evals (JSONL) and emit a CSV error curve
cargo run --bin cardinal -- eval --out eval.jsonl --curve-csv curves.csv

# Run synthetic Likert baseline evals (JSONL) for comparison
cargo run --bin cardinal -- eval-likert --out eval_likert.jsonl --curve-csv curves_likert.csv

# Generate a report from request/response JSON
cargo run --bin cardinal -- report --request request.json --response response.json --out report.md

# Run a reproducible rerank with cache locking + seed (no network calls)
cargo run --bin cardinal -- rerank --request request.json --out response.json --lock-cache --cache-only --rng-seed 1337 --report report.md

# Capture per-comparison trace data (JSONL)
cargo run --bin cardinal -- rerank --request request.json --out response.json --trace trace.jsonl
```

## Contributing

See `CONTRIBUTING.md`.

## Security

See `SECURITY.md`.

## Code of Conduct

See `CODE_OF_CONDUCT.md`.

## License

MIT
