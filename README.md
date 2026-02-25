# cardinal-harness

The highest quality way to have LLMs put numbers on things.

## The problem

You have a list of items — essays, candidates, proposals, models, code changes, anything — and you need quantitative scores, not just a vague ranking. Asking an LLM to "rate this 1–10" is fast but unreliable: scores are miscalibrated, inconsistent across items, and anchored to arbitrary reference points. You can't meaningfully say item A scored 7.2 and item B scored 6.8 and trust the difference.

This is the **prior elicitation problem**: LLMs have internal distributions over quality, but extracting those distributions through direct questioning produces noisy, biased point estimates. The challenge is information-theoretic — how do you design queries that maximally extract the model's latent knowledge per token spent?

## The approach

Instead of absolute scores, **cardinal-harness** asks pairwise ratio questions: *"how many times more [attribute] does A have than B?"* These relative judgments are far more reliable than absolute scores — the same way humans can tell which of two objects is heavier much more accurately than they can guess either object's weight.

Each pairwise ratio judgment (e.g., "A is 2.5x clearer than B") becomes a noisy observation of latent scores in log-space. A robust statistical solver (IRLS with Huber loss) combines all pairwise observations into a globally consistent set of scores, automatically downweighting outlier judgments where the LLM was confused or inconsistent.

The system tracks its own uncertainty. It knows which items' scores are well-determined and which aren't, and it selects the next most informative pair to query — so it converges on accurate top-K results with the minimum number of LLM calls. It stops when it's confident enough in the ranking, not after a fixed number of comparisons.

**"Cardinal"** means you get actual numeric scores on a ratio scale — not just ordinal rankings (1st, 2nd, 3rd) but quantitative latent values with uncertainty estimates, so you know both the ranking and how much each item differs from its neighbors.

### Why pairwise ratios specifically?

Three properties make pairwise ratios the right primitive for LLM prior elicitation:

1. **Transitivity in log-space.** If the model judges A/B = 2x and B/C = 3x, then ln(A/B) + ln(B/C) = ln(A/C). Pairwise log-ratios form a linear system that can be solved with standard robust regression. Absolute scores don't compose this way.

2. **Calibration-free.** The model never needs to place items on an absolute scale. Each comparison is self-contained — the model sees two concrete items and judges their relative quality. This sidesteps anchor bias, scale compression, and the well-documented tendency of LLMs to cluster scores around 7/10.

3. **Information density.** A single pairwise ratio judgment carries more information than two independent absolute ratings. The ratio directly encodes the *difference* that matters, rather than requiring the system to infer it from two noisy independent measurements.

## Elicitation methods

### Pairwise ratios (primary)

The core method. Ratio ladder from 1.0 to 26.0 (approximately geometric in log-space), with fine gradations near 1.0 for near-ties. Four prompt template variants (`canonical_v1` through `v3`) optimized for different contexts.

### Likert scale (baseline)

Per-item absolute ratings on configurable scales (5-point, 10-point). Implemented as a comparison baseline — the `eval-likert` CLI command runs head-to-head evaluations showing how pairwise ratios converge faster and more reliably than Likert for the same token budget.

### Planned: logprob extraction

Many models expose token-level log-probabilities. For structured outputs (ratio selection from a fixed ladder), logprobs over the ratio tokens provide a *direct* confidence signal that's cheaper and potentially more calibrated than asking the model to self-report confidence. See [docs/RESEARCH_THREADS.md](docs/RESEARCH_THREADS.md#logprob-extraction).

### Planned: hybrid elicitation

Combine cheap Likert pre-screening (identify obviously dominated items) with expensive pairwise ratios (resolve the top-K boundary). The planner already supports per-comparison cost weighting — extending this to heterogeneous elicitation methods is a natural next step.

## Objective functions

The system optimizes for top-K identification, but "top-K accuracy" decomposes into several distinct metrics, each capturing a different failure mode:

| Metric | What it measures | Failure mode it catches |
|--------|-----------------|------------------------|
| **Top-K precision/recall** | Set overlap with true top-K | Completely wrong items in top-K |
| **Kendall tau-b** | Pairwise concordance (handles ties) | Global ranking disorder |
| **Spearman rho** | Monotonic correlation | Non-linear but monotone distortions |
| **Rank reversals** | Weighted count of adjacent-rank swaps | Local instability at K-boundary |
| **Frontier inversion probability** | P(item K and K+1 are swapped) | The specific failure the planner targets |
| **Coverage @95% CI** | Fraction of true scores inside confidence intervals | Miscalibrated uncertainty |
| **CURL** | Concordance-based utility of ranked lists | Penalizes high-rank errors more than low-rank |

The planner's query selection objective blends **information gain** (from spectral graph theory — effective resistance) with **rank risk** (weighted probability of frontier inversions). The mixing parameter `lambda_risk` controls the blend: pure information gain explores uniformly, pure rank risk focuses narrowly on the K-boundary.

See [docs/RESEARCH_THREADS.md](docs/RESEARCH_THREADS.md#objective-functions) for discussion of additional objectives and their trade-offs.

## Cost model

Cost tracking operates at nanodollar precision (1e-9 USD) with full per-comparison attribution.

### Current: per-token pricing

```
cost = input_tokens * input_price + output_tokens * output_price
```

Provider pricing registry covers Claude, GPT, Kimi, and embedding models. A 20% rerank markup (6/5 ratio) applies on top of provider cost for service billing.

### Planned: cache-aware cost model

Modern inference providers implement **prompt caching** with a paged KV-cache architecture. When multiple requests share a common prefix (e.g., the system prompt + prompt template), cached tokens cost 10-25% of uncached tokens. This fundamentally changes the cost optimization landscape for cardinal-harness because:

- All comparisons for an attribute share the same system prompt + template (~500-2000 tokens)
- Entity text varies per comparison but the prefix is stable
- With prompt caching, the marginal cost of additional comparisons drops significantly after the first

The cost model should respect this structure: `cost = cache_creation_tokens * full_price + cache_hit_tokens * cached_price + uncached_tokens * full_price + output_tokens * output_price`. The planner can then make better cost-benefit decisions about whether an additional comparison is worth its (reduced) marginal cost.

See [docs/RESEARCH_THREADS.md](docs/RESEARCH_THREADS.md#prompt-caching) for the full analysis including paged cache alignment considerations.

### Model ladder

`ModelLadderPolicy` starts with a high-quality model (Claude Opus 4.5 at $5/$25 per 1M tokens) and downgrades to cheaper models (GPT-5-mini at $0.25/$2) when uncertainty is already low. The ladder respects per-attribute thresholds — high-weight attributes stay on expensive models longer.

## Features

- **Pairwise ratio prompts** on a fixed ladder (1.0 .. 26.0) for consistent, calibrated LLM judgments
- **Robust IRLS solver** turns noisy ratio observations into globally consistent latent scores with uncertainty
- **Multi-attribute utility** with gates and weighted top-k focus across multiple dimensions
- **Dynamic query planning** proposes only the most informative pairs, minimizing LLM calls
- **Uncertainty-aware stopping** — stops when top-k is sufficiently certain, not after a budget
- **SQLite cache** for pairwise judgments — repeated runs reuse prior LLM calls
- **OpenRouter integration** for model access with dynamic model-ladder switching
- **Typed ANP contexts** (`composable_ratio` vs `pairwise_only_ratio`) with supermatrix utilities
- **ANP active query helpers** (next context + next pair proposal)
- **12 orthogonal evaluation axes** organized into epistemic, instrumental, and strategic clusters
- **Synthetic evaluation suite** with 6 test scenarios, convergence curves, and head-to-head Likert comparison
- **Nanodollar cost tracking** with per-comparison attribution, usage sinks, and audit trails
- **Commander** strategic agent for codebase-scale evaluation (briefing, decomposition, flywheel, extraction, reflection)

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

## Architecture

```
                  ┌─────────────────────────────────────┐
                  │           rerank::multi              │
                  │  (orchestration loop + stopping)     │
                  └──────┬──────────┬───────────────────┘
                         │          │
              ┌──────────▼──┐  ┌───▼──────────────┐
              │ comparison  │  │   trait_search    │
              │ (LLM calls) │  │ (utility + gates) │
              └──────┬──────┘  └───┬──────────────┘
                     │             │
              ┌──────▼──┐  ┌──────▼──────────┐
              │ prompts  │  │  rating_engine   │
              │ (ladder) │  │ (IRLS + planner) │
              └──────┬──┘  └────────┬─────────┘
                     │              │
              ┌──────▼──────────────▼──┐
              │    gateway + cache     │
              │ (OpenRouter + SQLite)  │
              └───────────────────────┘
```

| Module | Purpose |
|--------|---------|
| `rating_engine` | Robust IRLS solver with Huber loss, effective-resistance planner, diagnostics |
| `trait_search` | Multi-attribute utility composition, MAD normalization, gating, top-K uncertainty |
| `rerank` | Orchestration loop, pairwise LLM comparison, stopping criteria, trace/hooks |
| `anp` | Typed ANP contexts, confidence-weighted local fits, weighted supermatrix |
| `axes` | 12 orthogonal evaluation dimensions (epistemic/instrumental/strategic clusters) |
| `prompts` | Ratio ladder prompt templates with entity context placement |
| `cache` | SQLite-backed pairwise memoization with composite key hashing |
| `gateway` | OpenRouter client, pricing registry, usage tracking, model ladder |
| `pipeline` | Multi-model generate/rank/synthesize pipeline |
| `commander` | Strategic agent: briefing, decomposition, flywheel, extraction, reflection |
| `text_chunking` | Token-aware semantic chunking with overlap |

## Evaluation axes

12 orthogonal dimensions organized into three clusters, designed for both code and idea evaluation:

**Epistemic** (quality of understanding): groundedness, calibration, resolution, causal depth, compositional reach

**Instrumental** (quality of proposed action): leverage, robustness, option value, economy

**Strategic** (meta-judgment): information value, temporal shape, prioritization

Each cluster feeds into the next via typed ANP edges: epistemic enables instrumental design (composable ratio), calibration enables information value assessment (composable ratio), temporal shape locally constrains interventions (pairwise-only ratio — not globally propagatable).

See `src/axes.rs` for the full axis definitions with context-sensitive weight profiles.

## Pairwise cache

The cache stores judgements keyed on (model, prompt template slug, template hash, attribute, entity text hashes). This avoids repeated LLM calls across runs. Set a custom path via:

```bash
export CARDINAL_CACHE_PATH=/path/to/cache.sqlite
```

Prune by age or size:

```bash
cargo run --bin cardinal -- cache-prune --max-age-days 30
cargo run --bin cardinal -- cache-prune --max-rows 100000
```

## Models

OpenRouter model ids are accepted directly. The model ladder starts with high-quality models and downgrades when uncertainty is low:

| Model | Input/Output per 1M tokens | When used |
|-------|---------------------------|-----------|
| `anthropic/claude-opus-4.5` | $5.00 / $25.00 | High-uncertainty, high-weight attributes |
| `openai/gpt-5.2-chat` | $1.75 / $14.00 | Medium-uncertainty comparisons |
| `openai/gpt-5-mini` | $0.25 / $2.00 | Low-uncertainty, confirms existing rankings |
| `moonshotai/kimi-k2-0905` | $0.39 / $1.90 | Cost-effective alternative |

## CLI

```bash
# Rerank from JSON request
cargo run --bin cardinal -- rerank --request input.json --out output.json --trace trace.jsonl

# Synthetic evaluation with convergence curves
cargo run --bin cardinal -- eval --out eval.jsonl --curve-csv curves.csv

# Likert baseline comparison
cargo run --bin cardinal -- eval-likert --out eval_likert.jsonl --curve-csv curves_likert.csv

# ANP demo (typed judgment contexts)
cargo run --bin cardinal -- anp-demo --input examples/anp_demo_request.json --out anp_output.json

# ANP typed-vs-forced benchmark
cargo run --bin cardinal -- eval-anp --out anp_eval.jsonl

# Report generation from request/response JSON
cargo run --bin cardinal -- report --request request.json --response response.json --out report.md

# Cache management
cargo run --bin cardinal -- cache-export --out cache.jsonl
cargo run --bin cardinal -- cache-prune --max-age-days 30

# Policy management
cargo run --bin cardinal -- policy list
cargo run --bin cardinal -- policy load --config policy.json

# Reproducible rerank (no network, cached only)
cargo run --bin cardinal -- rerank --request input.json --out output.json --lock-cache --cache-only --rng-seed 1337
```

## Open research threads

See [docs/RESEARCH_THREADS.md](docs/RESEARCH_THREADS.md) for detailed analysis of:

- **Prompt caching** — exploiting paged KV-cache structure for 75-90% cost reduction on shared prefixes, and how cache-aware cost models change planner behavior
- **Logprob extraction** — using token-level log-probabilities as direct confidence signals, avoiding self-reported confidence calibration issues
- **Likert integration** — hybrid elicitation combining cheap absolute ratings (screening) with expensive pairwise ratios (top-K resolution)
- **Labs coherence metrics** — future opportunities as providers expose internal coherence/consistency measurements as API parameters
- **Objective function zoo** — CURL, weighted discordance, Bayesian regret, and their trade-offs for different use cases
- **Rater calibration** — learning per-model bias/variance profiles from historical judgments
- **Confidence model selection** — beyond power-law mapping (Beta distributions, logistic, learned curves)

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/ALGORITHM.md`](docs/ALGORITHM.md) | Full design rationale: why pairwise ratios, IRLS, Huber loss, stopping rules |
| [`docs/PROMPTS.md`](docs/PROMPTS.md) | Prompt template slugs and context placement details |
| [`docs/ANP.md`](docs/ANP.md) | ANP context typing and supermatrix usage |
| [`docs/RESEARCH_THREADS.md`](docs/RESEARCH_THREADS.md) | Open research directions and opportunities |

## Contributing

See `CONTRIBUTING.md`.

## Security

See `SECURITY.md`.

## Code of Conduct

See `CODE_OF_CONDUCT.md`.

## License

MIT
