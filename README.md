# cardinal-harness

LLMs-as-judge harness for cardinal latents via pairwise ratio comparisons and multi-objective top-k reranking.

## What this is

- Pairwise ratio prompts on a fixed ladder (1.0 .. 26.0)
- Robust IRLS solver to turn noisy ratios into latent scores
- Multi-attribute utility with gates + weighted top-k focus
- Dynamic query planning: propose only the most informative pairs
- Stopping rules based on global top-k uncertainty
- SQLite cache for pairwise judgements
- OpenRouter for model access

The goal is to expose a clean, reusable research-grade core while staying practical for real use.

## Quickstart

```bash
export OPENROUTER_API_KEY=your_key_here
```

```rust
use std::sync::Arc;
use cardinal_harness::{
    Attribution, ProviderGateway, SqlitePairwiseCache,
};
use cardinal_harness::rerank::{
    MultiRerankAttributeSpec, MultiRerankEntity, MultiRerankRequest, MultiRerankTopKSpec,
    ModelLadderPolicy,
};
use cardinal_harness::gateway::NoopUsageSink;

# async fn demo() -> Result<(), Box<dyn std::error::Error>> {
let cache = SqlitePairwiseCache::new(SqlitePairwiseCache::default_path())?;
let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;
let model_policy = Arc::new(ModelLadderPolicy::default());

let req = MultiRerankRequest {
    entities: vec![
        MultiRerankEntity { id: "a".into(), text: "Entity A text".into() },
        MultiRerankEntity { id: "b".into(), text: "Entity B text".into() },
        MultiRerankEntity { id: "c".into(), text: "Entity C text".into() },
    ],
    attributes: vec![
        MultiRerankAttributeSpec {
            id: "clarity".into(),
            prompt: "clarity of explanation".into(),
            prompt_template_slug: Some("canonical_v2".into()),
            weight: 1.0,
        },
    ],
    topk: MultiRerankTopKSpec {
        k: 2,
        weight_exponent: 1.3,
        tolerated_error: 0.1,
        band_size: 5,
        effective_resistance_max_active: 64,
        stop_sigma_inflate: 1.25,
        stop_min_consecutive: 2,
    },
    gates: vec![],
    comparison_budget: None,
    latency_budget_ms: None,
    model: None, // defaults to openai/gpt-5-mini
    rater_id: None,
    comparison_concurrency: None,
    max_pair_repeats: None,
};

let attribution = Attribution::new("example::multi_rerank");
let resp = cardinal_harness::rerank::multi_rerank(
    Arc::new(gateway),
    Some(&cache),
    Some(model_policy),
    req,
    attribution,
    None,
).await?;

println!("stop_reason: {:?}", resp.meta.stop_reason);
# Ok(())
# }
```

## Pairwise cache

The cache stores judgements keyed on:
- model
- prompt template slug
- attribute id + prompt hash
- entity ids + text hashes

This avoids repeated LLM calls across runs. Set a custom path via:

```bash
export CARDINAL_CACHE_PATH=/path/to/cache.sqlite
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

See `docs/ALGORITHM.md` for a short algorithm sketch.

## License

MIT
