# Research Threads

Open research directions for cardinal-harness. Each thread identifies a concrete
opportunity, the current state of the code, what would need to change, and what
the expected payoff is. Ordered roughly by expected impact per unit of effort.

---

## Prompt caching

### The opportunity

Modern inference providers implement **paged KV-cache** architectures where the
key-value activations from previous tokens are stored and reused across requests
that share a common prefix. Anthropic's prompt caching charges ~10% of the normal
input token price for cache hits; OpenAI's automatic caching provides 50% discount
on cached prefix tokens. The cache is typically **paged** — aligned to fixed-size
blocks (e.g., 128 or 256 tokens on Anthropic, implementation-dependent on OpenAI)
— so cache hits require exact prefix matches up to a page boundary.

Cardinal-harness is *structurally ideal* for prompt caching because every
comparison for a given attribute shares:

1. The system prompt (~200-800 tokens)
2. The prompt template body (~300-1000 tokens)
3. The ratio ladder definition (~200 tokens)
4. The attribute description (~50-200 tokens)

This shared prefix is typically 750-2200 tokens. For a rerank job doing 50
comparisons on one attribute, that's 37,500-110,000 tokens that could be served
from cache at 10-25% of full price — a **75-90% cost reduction on input tokens**
for comparisons after the first.

### Interaction with the paged cache structure

The cache is not byte-level — it's paged. This means:

- Prefix sharing works *up to the last complete page boundary* before the content
  diverges. If the shared prefix is 1,847 tokens and pages are 128 tokens wide,
  only floor(1847/128) * 128 = 1792 tokens are cache-eligible.
- **Prompt template design should be cache-page-aware**: pad the shared prefix to
  the nearest page boundary with semantically neutral content (or restructure the
  template so entity-specific content starts at a page boundary).
- Entity text ordering within a comparison matters: if entity A's text always
  precedes entity B's text, and entity A is shared across multiple comparisons
  (e.g., comparing A vs B, A vs C, A vs D), then entity A's tokens *also* become
  part of the cached prefix for those comparisons. The planner could exploit this
  by batching comparisons that share an entity.

### What would need to change

**In `gateway/pricing.rs`:**

```rust
pub struct CacheAwarePricing {
    pub base: ModelPricing,
    /// Price multiplier for cache-hit input tokens (e.g., 0.1 for 90% discount).
    pub cache_hit_multiplier: f64,
    /// Price for cache-write input tokens (typically 1.0 or 1.25x).
    pub cache_write_multiplier: f64,
    /// Page size in tokens for cache alignment.
    pub cache_page_size: usize,
}
```

**In `gateway/types.rs` — `ChatResponse`:**

```rust
pub cache_read_tokens: u32,     // tokens served from cache
pub cache_write_tokens: u32,    // tokens written to cache (first request)
```

**In the planner** (`rating_engine.rs`):

The `cost_per_edge` used by `plan_edges_for_rater()` should be entity-pair
dependent: comparisons that share an entity with a recently-queried comparison
have lower marginal cost (more cache hits). This converts the planner from
uniform-cost to variable-cost edge selection.

**In the orchestration loop** (`rerank/multi.rs`):

Batch comparisons sharing an entity to maximize prefix cache utilization. The
current loop selects pairs independently — a cache-aware variant would cluster
pairs like (A,B), (A,C), (A,D) together.

### Expected payoff

For a typical 30-entity, 2-attribute rerank job (~120 comparisons):
- Current cost: ~120 * 2000 input tokens * $3/1M = $0.72
- Cache-aware cost: ~120 * (500 cached + 1500 uncached) * effective rate ≈ $0.25
- **~65% cost reduction** with no change to judgment quality.

The planner's cost-benefit calculus also shifts: with lower marginal costs,
the system can afford more comparisons before stopping, which means higher
confidence rankings for the same budget.

---

## Logprob extraction

### The opportunity

Many models expose **token-level log-probabilities** in their API responses.
For structured outputs where the model selects from a fixed vocabulary (like our
ratio ladder: 1.0, 1.05, 1.1, ..., 26.0), the logprob distribution over ratio
tokens is a *direct* measurement of the model's uncertainty about the comparison
— richer and cheaper than asking the model to self-report a confidence value.

Currently, cardinal-harness asks the model to output `{"confidence": 0.85}` as
part of its JSON response. This self-reported confidence has two problems:

1. **Calibration**: LLMs are notoriously poorly calibrated when reporting their
   own confidence. A model that says "0.85 confidence" may be right 60% or 95%
   of the time — we don't know without empirical calibration data.

2. **Cost**: The confidence field requires output tokens. With logprobs, the
   confidence signal comes "for free" as metadata on the ratio token itself.

### What logprobs reveal

For a comparison where the model outputs ratio "2.5":

```
Token     Logprob   Probability
"2.5"     -0.22     0.80
"2.1"     -1.61     0.20
"3.1"     -4.61     0.01
...
```

The entropy of this distribution directly measures the model's uncertainty:
- Low entropy (one dominant token): model is confident
- High entropy (probability spread across ratios): model is uncertain
- Bimodal distribution (two peaks): model sees a genuine ambiguity

This is *strictly more informative* than a scalar confidence value because:
- It reveals the *shape* of uncertainty (bimodal vs. diffuse)
- It's grounded in the model's actual token selection, not a post-hoc rationalization
- It can be calibrated empirically without changing the prompt

### Mapping logprobs to observation weight

The confidence mapping `g(c) = eps + (1 - eps) * c^gamma` already exists.
Logprob-derived confidence could feed into this same function:

```
c_logprob = exp(logprob_of_selected_ratio)  // probability of the token chosen
```

Or more sophisticatedly, use the **entropy** of the ratio distribution:

```
H = -sum(p_i * ln(p_i)) for all ratio tokens
c_entropy = 1.0 - H / H_max  // normalized: 0 = maximum entropy, 1 = certain
```

Or use the **probability mass within one ladder step** of the selected ratio:

```
c_neighborhood = sum(p_i for ratios within 1 step of selected)
```

Each of these captures different information. The entropy-based measure is most
principled (information-theoretic), but the neighborhood measure may be more
robust to the long tail of very unlikely ratios.

### What would need to change

**In `gateway/types.rs`:**

```rust
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f64,
}

pub struct ChatResponse {
    // ... existing fields ...
    /// Per-token logprobs for the output, if requested and available.
    pub logprobs: Option<Vec<LogprobEntry>>,
}

pub struct ChatRequest {
    // ... existing fields ...
    /// Whether to request logprobs in the response.
    pub logprobs: bool,
    /// Number of top logprobs to return per token position.
    pub top_logprobs: Option<u32>,
}
```

**In `rerank/comparison.rs`:**

Parse logprobs from the ratio token position. Extract the distribution over
ratio ladder values. Compute confidence using one of the methods above. Use
this *instead of* (or blended with) self-reported confidence.

**In the confidence mapping** (`rating_engine.rs`):

Add a `ConfidenceSource` enum to track whether confidence came from self-report,
logprobs, or a blend. Per-source calibration curves can then be learned from
historical data.

### Availability

- **OpenAI**: `logprobs: true, top_logprobs: 5` in the API request. Available on
  all chat models. Returns top-5 logprobs per output token position.
- **Anthropic**: Not currently available via the standard Messages API (as of
  Feb 2026). May become available in future; monitor API changelog.
- **OpenRouter**: Passes through provider logprob support. Works for OpenAI
  models routed through OpenRouter.

### Expected payoff

- Eliminate the confidence field from the output format (save ~10-20 output tokens
  per comparison, ~5-10% output cost reduction)
- Better-calibrated confidence → better observation weighting → faster convergence
- Bimodal detection → flag genuinely ambiguous comparisons for human review or
  different prompt strategy
- Foundation for learned calibration curves

---

## Labs coherence metrics

### The opportunity

LLM providers have access to internal model state that end-users don't see:
attention patterns, layer activations, internal consistency measures. As the
eval/alignment ecosystem matures, there's a natural path for providers to expose
**coherence and consistency metrics** as first-class API response fields.

Imagine a future API response that includes:

```json
{
  "content": "...",
  "coherence_metrics": {
    "internal_consistency": 0.92,
    "attention_entropy": 0.34,
    "layer_agreement": 0.88,
    "epistemic_uncertainty": 0.15
  }
}
```

These would be provider-computed signals about how "sure" the model is, derived
from its actual internal processing rather than from its text output. They would
be fundamentally more trustworthy than self-reported confidence because they
bypass the model's tendency to confabulate certainty.

### How cardinal-harness should prepare

The architecture already has the right seam: the `confidence` field in
`PairwiseJudgement::Observation` maps to observation weight via `g_of_c()`. When
labs coherence metrics become available, they slot into this same pathway:

1. **ConfidenceSource abstraction**: Replace the scalar `confidence: f64` with
   a richer type that carries provenance:

```rust
pub enum ConfidenceSource {
    SelfReported(f64),
    Logprob { entropy: f64, top_prob: f64 },
    LabsCoherence { internal_consistency: f64, epistemic_uncertainty: f64 },
    Blended { sources: Vec<(ConfidenceSource, f64)> },  // weighted blend
}
```

2. **Per-source calibration**: Different confidence sources have different
   calibration curves. Self-reported confidence might need gamma=2.0 (current
   default), while labs coherence might be nearly linear (gamma≈1.0) if providers
   calibrate it well. The system should learn these curves from historical
   judgment accuracy.

3. **Provider capability detection**: The gateway should probe for available
   confidence signals at connection time and automatically use the highest-quality
   source available for each model.

### Why this matters strategically

If cardinal-harness establishes itself as infrastructure that *consumes*
provider-side coherence metrics, it creates a demand signal that incentivizes
providers to expose those metrics. The framework becomes a natural integration
point for a capability that doesn't exist yet but that multiple labs are likely
developing internally.

The key insight: providers already measure internal consistency for their own
RLHF/RLAIF pipelines. Exposing it via API is a product decision, not a research
problem. Cardinal-harness being ready to consume it removes one barrier to that
product decision.

---

## Objective functions

### Current state

The planner optimizes a blend of information gain (effective resistance) and rank
risk (frontier inversion probability). The evaluation suite reports Kendall tau-b,
Spearman rho, top-K precision/recall, and coverage@95%CI.

### Additional objectives worth tracking

#### CURL (Concordance-based Utility of Ranked Lists)

CURL penalizes high-rank errors more severely than low-rank errors, which aligns
with how ranking errors actually matter in practice (getting #1 wrong is worse
than getting #47 wrong):

```
CURL(sigma, sigma*) = 1 - (2 / (n*(n-1))) * sum_{i<j} w(i,j) * I(sigma(i,j) != sigma*(i,j))
```

where `w(i,j)` is a weight function that emphasizes pairs involving top-ranked
items. Common choices:

- `w(i,j) = 1/(rank_i * rank_j)` — harmonic weighting
- `w(i,j) = max(1/rank_i, 1/rank_j)` — max-rank weighting
- `w(i,j) = exp(-alpha * min(rank_i, rank_j))` — exponential decay

CURL is especially relevant for cardinal-harness because the planner already
focuses on top-K — CURL would measure whether that focus translates into
actual top-K quality.

#### Weighted rank reversals

Count adjacent-rank swaps weighted by their position:

```
WRR = sum_{i: sigma(i) != sigma*(i)} 1/rank(i) * |sigma(i) - sigma*(i)|
```

This is simpler than CURL but captures the same intuition: reversals near the
top matter more. The current `expected_rank_reversals` in `SolveSummary` counts
unweighted — adding position weighting is straightforward.

#### Bayesian regret

If the ranking is used to *select* the top-K for some downstream purpose, the
regret is the expected utility loss from selecting the estimated top-K instead of
the true top-K:

```
Regret = E[U(true_top_k)] - E[U(estimated_top_k)]
```

This requires a utility model over items (e.g., the latent scores themselves).
The advantage of Bayesian regret over simple precision/recall is that it captures
*how bad* the errors are, not just *how many* errors there are.

#### Normalized Discounted Cumulative Gain (nDCG)

Standard IR metric, already well-understood. Worth including because it's the
lingua franca of ranking evaluation:

```
DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
nDCG@k = DCG@k / IDCG@k
```

Where `rel_i` is the ground-truth relevance/score of the item at position i.

### What would need to change

Add a `RankQualityMetrics` struct to `rerank/evaluation.rs`:

```rust
pub struct RankQualityMetrics {
    pub kendall_tau_b: f64,
    pub spearman_rho: f64,
    pub topk_precision: f64,
    pub topk_recall: f64,
    pub coverage_95ci: f64,
    pub ndcg_at_k: f64,
    pub curl_harmonic: f64,
    pub curl_exponential: f64,
    pub weighted_rank_reversals: f64,
    pub bayesian_regret: f64,
}
```

This centralizes all rank quality metrics in one place instead of computing them
ad-hoc. The planner could then be parameterized by which metric to optimize,
and the evaluation suite could report all of them side-by-side.

### Relationship to the planner objective

The planner currently optimizes `delta_info + lambda_risk * delta_rank_risk`.
The `delta_rank_risk` term is closely related to frontier inversion probability.
Alternative planner objectives could target:

- **CURL-optimal**: select pairs that maximally reduce expected CURL error
- **Regret-optimal**: select pairs that maximally reduce Bayesian regret
- **nDCG-optimal**: select pairs that maximally improve expected nDCG@k

Each of these is more expensive to compute than the current effective-resistance
approach (they require forward-simulating the effect of a hypothetical observation
on the full ranking), but they may converge faster for specific use cases.

---

## Likert integration

### Current state

`eval-likert` runs synthetic Likert baseline evaluations for comparison. But
Likert ratings are currently used *only* as a comparison baseline — they're not
integrated into the main reranking pipeline as an elicitation method.

### The hybrid opportunity

Likert and pairwise ratios have complementary strengths:

| | Likert | Pairwise ratio |
|---|--------|---------------|
| Cost per item assessed | 1 LLM call per item | 1 LLM call per *pair* |
| Information per call | Low (noisy absolute score) | High (calibrated relative judgment) |
| Scaling | O(n) calls for n items | O(n log n) to O(n^2) for full tournament |
| Strength | Fast coarse screening | Precise top-K resolution |

A **two-phase hybrid** exploits both:

1. **Phase 1 (Likert screening)**: Rate all n items with cheap Likert calls.
   Identify the bottom ~50% that are clearly dominated. Cost: O(n).
2. **Phase 2 (Pairwise resolution)**: Run pairwise ratios only on the surviving
   top ~50%. Cost: O((n/2) log(n/2)) instead of O(n log n).

For n=100 items, this could reduce total comparisons from ~300 to ~150 while
maintaining the same top-K accuracy — because the Likert phase correctly filters
obviously-bad items even though it can't precisely rank the good ones.

### What would need to change

1. **Extend `PairwiseJudgement`** with a Likert variant:

```rust
pub enum Judgement {
    Pairwise { higher_ranked: HigherRanked, ratio: f64, confidence: f64 },
    Likert { value: f64, scale_max: f64, confidence: f64 },
    Refused,
}
```

2. **Add Likert-to-prior mapping**: Convert Likert scores into prior
   observations for the IRLS solver. A 7/10 Likert score doesn't directly
   translate to a log-ratio, but it *does* provide a weak prior:

```
For items i with Likert score L_i, create synthetic observations:
  ln(s_i / s_anchor) ≈ f(L_i) with high variance
```

   where `f` maps the Likert scale to log-space and the high variance reflects
   Likert's lower precision.

3. **Planner integration**: The planner should know that Likert observations
   exist and have lower information content. It should still propose pairwise
   comparisons for items whose Likert scores are ambiguous (near the cutoff),
   and skip comparisons for items that are clearly dominated.

### Expected payoff

- 30-50% cost reduction for large item sets (n > 50)
- Same or better top-K accuracy (pairwise ratios still resolve the boundary)
- Faster wall-clock time (Likert calls can run in parallel with no pair dependencies)

---

## Cache-aware cost modeling

### Current state

The pricing module (`gateway/pricing.rs`) models cost as a simple linear function
of input and output tokens. The planner uses a uniform `cost_per_edge` for all
comparisons.

### What's missing

Real costs are **non-uniform** across comparisons due to:

1. **Prompt caching** (discussed above): comparisons sharing an entity or
   attribute have lower marginal input cost after the first comparison.

2. **Entity text length variance**: comparing two 50-token entities costs less
   than comparing two 2000-token entities. The planner should prefer cheap
   comparisons when their information gain is similar.

3. **Model switching**: the model ladder changes the per-token price mid-run.
   Cost estimates should reflect the *current* model, not the starting model.

4. **Output token variance**: some comparisons produce longer outputs (e.g.,
   when the model explains its reasoning before the ratio). The planner can't
   predict this, but it can use running averages.

5. **Batch API discounts**: OpenAI's batch API provides 50% discount with
   24-hour turnaround. For non-latency-sensitive use cases, batching comparisons
   could halve costs.

### The cost model cardinal-harness should support

```rust
pub struct ComparisonCostEstimate {
    /// Expected input tokens for this specific comparison.
    pub input_tokens: u32,
    /// Of those, how many are expected to hit prompt cache.
    pub cached_input_tokens: u32,
    /// Expected output tokens (running average or per-model estimate).
    pub output_tokens: u32,
    /// Current model pricing.
    pub pricing: ModelPricing,
    /// Cache pricing (if available).
    pub cache_pricing: Option<CacheAwarePricing>,
    /// Computed cost in nanodollars.
    pub estimated_cost_nanodollars: i64,
}
```

The planner's scoring function then uses `estimated_cost_nanodollars` instead of
the uniform `cost_per_edge`, which lets it naturally prefer cheap comparisons
(cache-hot, short entities, cheap model) when they provide similar information.

### Respecting paged cache structure

The cost model needs to know the **cache page size** for the current provider to
accurately estimate cache hits. If the shared prefix is 1847 tokens and the page
size is 128, only 1792 tokens are cacheable. The remaining 55 tokens of shared
prefix are charged at full price.

Additionally, **entity ordering** within a comparison affects caching: if entity
A's text appears before entity B's text in the prompt, and we run comparisons
(A,B), (A,C), (A,D), then entity A's tokens extend the cached prefix for the
second and third comparisons. The planner could exploit this by:

1. Identifying a "pivot entity" that appears in many comparisons
2. Always placing the pivot entity first in the prompt
3. Batching comparisons involving the pivot entity together

This is similar to how databases optimize join orders to maximize index utilization.

---

## Rater calibration

### The opportunity

Different models (and even different model versions) have systematic biases:
- Some models tend to give higher ratios (optimistic raters)
- Some models have narrower ratio distributions (conservative raters)
- Some models refuse comparisons more often on certain topics

Currently, all raters are treated identically by the solver. The `rater_id` field
exists in `Observation` and per-rater bias/scatter is computed in calibration
evidence, but this information isn't *used* to adjust observation weights.

### What calibration would look like

After accumulating enough historical data (e.g., 100+ judgments from a model),
fit a per-rater calibration model:

```
observed_ratio = bias + scale * true_ratio + noise(scatter)
```

Where:
- `bias`: systematic over/under-estimation (subtract from observations)
- `scale`: ratio compression/expansion (divide observations by scale)
- `scatter`: noise level (feeds into observation variance)

The IRLS solver already accepts per-observation variance. Calibrated raters would
get lower variance (higher weight) and de-biased observations.

### Cross-run learning

The SQLite cache already stores per-model, per-attribute judgment history. A
calibration service could:

1. Identify items with well-established scores (many concordant judgments)
2. Use these as "ground truth" to calibrate new raters
3. Store calibration parameters alongside the model in the pricing registry
4. Apply calibration automatically when a known model is used

This creates a **flywheel**: more judgments → better calibration → more efficient
future judgments → more judgments.

---

## Confidence model selection

### Current state

The confidence mapping `g(c) = eps + (1 - eps) * c^gamma` with eps=0.001,
gamma=2.0 is a power-law with a floor. This was chosen heuristically and works
reasonably well, but there's no theoretical justification for the functional form
or the parameter values.

### Alternative models

**Beta distribution CDF**:
```
g(c) = I_c(alpha, beta)  // regularized incomplete beta function
```
Advantages: flexible shape (U-shaped, J-shaped, symmetric), well-studied,
parameters have clear interpretation (alpha > beta means optimistic about
confidence, beta > alpha means skeptical).

**Logistic**:
```
g(c) = 1 / (1 + exp(-k * (c - c0)))
```
Advantages: sigmoid shape naturally models "threshold" behavior where confidence
below c0 is essentially ignored and confidence above c0 is trusted. The sharpness
k controls how abrupt the transition is.

**Learned calibration curve**: Fit g(c) nonparametrically from historical data.
Bin judgments by reported confidence, compute empirical accuracy per bin, fit an
isotonic regression. This is the most principled approach but requires enough
data to estimate the curve reliably.

### Selection strategy

Rather than picking one confidence model, implement all three and select per-model
based on historical calibration data. The evaluation suite should report
calibration metrics (expected calibration error, Brier score) for each confidence
model on each rater, and the system should automatically use the best-calibrated
model for each rater.

---

## Sparse solvers for scale

### Current state

The IRLS solver uses dense linear algebra (nalgebra Cholesky decomposition) and
is limited to 5,000 items. Memory usage is O(n^2) for the information matrix.

### When this matters

- Corpus-scale evaluation: ranking 50K+ documents
- Model comparison: evaluating outputs from hundreds of models across many prompts
- Tournament systems: ongoing competitions with growing participant pools

### Options

1. **Conjugate gradient (CG)**: Solve the normal equations iteratively without
   forming the full information matrix. The comparison graph is sparse (far fewer
   edges than n^2), so CG with a sparse matrix-vector product would be O(m) per
   iteration where m = number of observations. With a good preconditioner
   (incomplete Cholesky or Jacobi), CG converges in O(sqrt(condition_number))
   iterations.

2. **Sparse Cholesky**: Use a sparse direct solver (e.g., cholmod via sparse
   bindings). This gives exact solutions with O(nnz^{1.5}) cost for typical
   comparison graphs.

3. **Hierarchical decomposition**: For very large problems, decompose into
   clusters (e.g., by topic or initial Likert screening), solve each cluster
   independently, then stitch together with cross-cluster bridge comparisons.

### Recommended path

CG with Jacobi preconditioning is the simplest first step: it works with the
existing observation data structures, doesn't require external solver libraries,
and would extend the practical limit from 5K to ~100K items.

---

## Structured experiment tracking

### The gap

The evaluation suite runs synthetic benchmarks and reports metrics, but there's
no systematic way to track experiments across runs, compare configurations,
or identify regressions. Each run produces a JSONL file that's essentially
standalone.

### What's needed

A lightweight experiment registry that tracks:

- Configuration (model, prompt template, confidence model, planner mode, etc.)
- Per-run metrics (all items from `RankQualityMetrics` above)
- Cost/efficiency metrics (comparisons used, tokens consumed, wall time)
- Convergence curves (error trajectory over comparison count)
- Reproducibility metadata (RNG seed, cache state, model version)

This doesn't need to be a full MLflow — a SQLite table with JSON config and
metrics columns would suffice. The key is being able to answer questions like:

- "Did switching to logprob-based confidence improve convergence speed?"
- "What's the cost-accuracy Pareto frontier across model ladder configurations?"
- "Is the new prompt template v4 actually better than v2?"

The commander store already has SQLite infrastructure for run tracking. Extending
it to cover experiment metadata would be natural.

---

## Reproducibility and auditability

### The compounding insight

The SQLite cache + per-comparison trace data already make individual runs
reproducible. The next step is making them **auditable by third parties**:

- **Judgment receipts**: each comparison produces a signed record containing
  (model, prompt, entities, ratio, confidence, timestamp, cost). These are the
  atomic units of epistemic accountability.
- **Cache sharing**: export/import cache databases between teams so evaluations
  can be replicated without re-running LLM calls.
- **Cache merging**: combine caches from multiple evaluators for consensus
  rankings (each evaluator's judgments become additional observations in the
  IRLS solver).
- **Provenance chains**: link derived rankings to their source judgments, so
  any ranked output can be traced back to the specific LLM calls that produced it.

This matters because cardinal-harness isn't just a measurement tool — it's
infrastructure for *contestable* evaluation. If someone disagrees with a ranking,
they can inspect the specific pairwise judgments that produced it, identify
which ones they dispute, and see how removing those judgments changes the result.

---

## Summary: priority ordering

| Thread | Impact | Effort | Dependencies |
|--------|--------|--------|-------------|
| Prompt caching cost model | High | Medium | Gateway changes |
| Logprob extraction | High | Medium | Provider API support |
| Rank quality metrics (CURL, nDCG) | Medium | Low | None |
| Likert hybrid elicitation | Medium | Medium | Planner changes |
| Rater calibration | Medium | High | Historical data |
| Labs coherence metrics | High | Low (preparation) | Provider APIs (future) |
| Sparse solvers | Medium | High | nalgebra or external dep |
| Confidence model selection | Low | Medium | Calibration data |
| Experiment tracking | Medium | Low | Commander store |
| Reproducibility/auditability | High | Medium | Cache format work |
