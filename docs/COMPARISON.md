# How cardinal-harness relates to the rest of the field

The LLM-ranking ecosystem almost universally elicits **binary or ordinal**
preferences ("is A better than B?", "rank these 20") and then either uses the
order directly or fits an **interval-scale** latent strength from binary
outcomes (Bradley–Terry / Elo). cardinal-harness differs on three axes at
once:

1. It elicits **ratio-magnitude** judgements ("how many times more of X does
   A have than B?") — the Analytic Hierarchy Process / magnitude-estimation
   tradition, not the IR-reranking tradition.
2. It fits **ratio-scale scores with per-item uncertainty**, robustly
   (IRLS + Huber) over the whole comparison graph.
3. It selects the next pair **actively** and stops when the top-k boundary is
   settled within a stated budget.

Lineage in one line: AHP's ratio judgements crossed with LMArena's statistical
discipline (uncertainty + active sampling), delivered as a Rust library/CLI
with a SQLite cache and full cost receipts.

A scale distinction worth being precise about: *ordinal* gives order only;
*interval* (Bradley–Terry, Elo, Thurstone, Rank Centrality) makes differences
meaningful but not ratios — Arena "scores" are interval, you cannot say "A is
twice as good"; *ratio* scales support exactly that claim, and are only
obtainable when the elicitation itself carries magnitude. That is the bet this
repo makes — and note the honest caveat: our checked-in receipts show the bet
paying off on some regimes and losing on others (see
[EVALUATION.md](EVALUATION.md)). Ratio elicitation is strictly more
informative *when the judge can actually provide it*; whether a given model
can, for a given attribute, is an empirical question.

## Prompting regimes

| Regime | Representative work | Shape | Scale | Calls for n items | Characteristic failure |
|---|---|---|---|---|---|
| Pointwise / scalar | "rate 1–10", Likert, query likelihood | item + rubric → score | cardinal in name only | O(n), cheapest | miscalibration; clustering near 7–8; scores not comparable across calls |
| Pairwise preference | PRP ([Qin et al. 2023](https://arxiv.org/abs/2306.17563)) | 2 items → "A"/"B" | ordinal (binary) | naive O(n²); sort/window variants O(n log n)/O(n) | intransitivity; position bias; magnitude discarded |
| Listwise | RankGPT ([Sun et al. 2023](https://arxiv.org/abs/2304.09542)), [RankVicuna](https://arxiv.org/abs/2309.15088), [RankZephyr](https://arxiv.org/abs/2312.02724), [RankLLM](https://arxiv.org/abs/2505.19284) | k items → permutation | ordinal | ~O(n) via sliding window (20/10) | context limits; order sensitivity; no scores, no uncertainty |
| Setwise | [Zhuang et al., SIGIR 2024](https://arxiv.org/abs/2310.09497) | c items → "which is best" | ordinal | between pairwise and listwise | still comparison-only output |
| Tournament / knockout | single-elimination, round-robin + BT | bracket | ordinal → interval if fed to BT | O(n) knockout … O(n²) round-robin | knockout is fragile to a single noisy judgement |
| **Pairwise ratio (this repo)** | AHP tradition, LLM-adapted | 2 items → ratio on a fixed ladder + confidence | **ratio, with posterior std** | default budget 4·n, actively allocated | costs more per call than scalar; requires a coherent attribute |

None of the mainstream regimes ask "how many times more". That question is
this repo's entire reason to exist.

## Aggregation math

| Method | Input | Output scale | Per-item uncertainty | Active selection |
|---|---|---|---|---|
| Bradley–Terry (MLE) | binary wins | interval (log-odds) | bootstrap / Hessian CIs | not intrinsic; [LMArena](https://arxiv.org/abs/2403.04132) bolts it on |
| Elo (online) | sequential binary | interval | weak | no; order-sensitive, superseded by BT in Arena |
| TrueSkill | game outcomes | Gaussian skill (μ, σ) | native σ | matchmaking uses σ (implicit) |
| Thurstone (Case V) | binary | interval | variance | no |
| [Rank Centrality](https://arxiv.org/abs/1209.1688) | comparison graph | stationary distribution | theory bounds, not per-item CIs | no |
| AHP (Saaty eigenvector) | **ratio matrix** | **ratio** | global consistency ratio only | no — wants all n² comparisons |
| **This repo (IRLS + Huber on log-ratios)** | **ratio + confidence** | **ratio (log-space latent)** | posterior std per item + top-k boundary error | effective-resistance planner + certified stop |

The active-ranking literature (LMArena's uncertainty-weighted sampling,
[active top-k aggregation](https://proceedings.mlr.press/v70/mohajer17a.html),
budgeted pairwise ranking) is mature in theory and nearly absent from LLM
tooling; the planner and stopping rule here sit squarely in that line.

## Query-relevance rerankers are a different job

Cohere Rerank, Voyage, Jina, BGE cross-encoders, FlashRank, and the
[`rerankers`](https://github.com/AnswerDotAI/rerankers) library answer *"how
relevant is this document to this query"* in one pass — no criterion-based
magnitudes, no uncertainty, no active selection, and no need for them: it is a
retrieval problem. Same word ("rerank"), different problem. If you need
query→document relevance at scale, use one of those; if you need "sort my
shortlist by how much of X each item has, and tell me how sure you are," use
this.

## The nearest tool: llm-sort

[`llm-sort`](https://github.com/vagos/llm-sort) (an `llm` CLI plugin,
[reviewed by Simon Willison](https://simonwillison.net/2025/Feb/11/llm-sort/))
is the only other general-purpose "sort an arbitrary list by a criterion" CLI
we know of. It feeds binary pairwise judgements into a comparison sort
(`sorted(cmp_to_key(...))`). Same user intent, opposite engineering
philosophy:

| | llm-sort | cardinal sort |
|---|---|---|
| Judgement | binary "which line is better" | ratio ladder + confidence |
| Aggregation | comparison sort trusts every answer | robust global fit; outliers down-weighted |
| Intransitive judge | thrashes (sort assumes transitivity) | modeled — cycles become residuals |
| Output | reordered lines | reordered lines + mean ± std, z, percentile |
| Uncertainty / stop | none | top-k error estimate, certified stop, budget |
| Cost accounting | none | per-run receipt; per-comparison trace; SQLite cache; keyless replay |
| Weight | tiny Python plugin | a Rust engine; heavier by design |

If you just need a quick plausible ordering and don't care about receipts,
llm-sort is less machinery. The moment "how much better?" or "how sure are
we?" matters, the machinery is the point.

## Known pathologies and what this design does about them

| Pathology | Evidence | Design answer here | Status |
|---|---|---|---|
| Position bias | [Judging the Judges](https://arxiv.org/abs/2406.07791) | A/B presentation randomized per comparison | implemented, on by default |
| Scalar miscalibration / clustering | [judgment-distribution work](https://arxiv.org/abs/2503.03064) | ratio elicitation avoids absolute scales entirely | by construction; receipts vs Likert are mixed |
| Context limits | listwise sliding windows | two items per call, always | by construction |
| O(n²) pairwise cost | PRP's own motivation | active planner, default 4·n budget, top-k focus | implemented; receipts do not yet prove early stopping |
| Intransitivity | [non-transitivity](https://arxiv.org/abs/2502.14074), [LLM-RankFusion](https://arxiv.org/abs/2406.00231), [TrustJudge](https://arxiv.org/abs/2509.21117) | latent-score model fits *through* cycles; Huber loss discounts outliers | implemented |
| Baseline / anchor dependence | fixed-baseline comparisons drift | full comparison graph, global fit, gauge pinning | implemented |

"Design answer" is not "proven win" — the honest state of the evidence lives
in [EVALUATION.md](EVALUATION.md) and the checked-in receipt packs under
`artifacts/`.
