# cardinal-harness

[![CI](https://github.com/XyraSinclair/cardinal-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/XyraSinclair/cardinal-harness/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cardinal-harness.svg)](https://crates.io/crates/cardinal-harness)
[![docs.rs](https://img.shields.io/docsrs/cardinal-harness)](https://docs.rs/cardinal-harness)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Does the model actually believe what it just told you — or was it only
echoing how you asked?** cardinal-harness measures the difference and gives
you the number. A preference earns the name *belief* only if it survives the
transformations that shouldn't matter — presentation order, wording,
polarity, who's asking. This engine elicits LLM judgements as noisy
measurements, tests them against exactly that battery, and prices everything
that fails it in nats, with receipts. (Watch one judgement bend under
framing while another refuses to move, on live committed data:
[the receipt viewer](artifacts/live/receipt-viewer-2026-07-08/).)

The everyday verb is sorting:

```console
$ cardinal sort ideas.txt --by "expected impact on retention"
```

`cardinal-harness` turns noisy LLM pairwise **ratio** judgements ("how many
times more of X does A have than B?") into globally consistent **cardinal
scores with uncertainty**, spends each next comparison where it buys the most
information about the order, and stops when the top-k is certain enough — or
the budget runs out. Every run returns receipts: comparisons, tokens, dollar
cost, stop reason, and an optional per-judgement trace.

## Why not just ask the model to sort?

Every obvious way to sort a list with an LLM breaks somewhere:

| Approach | What breaks |
|---|---|
| "Rate each item 1–10" | Miscalibrated, anchor-dependent; scores cluster at 7–8; no error bars |
| "Sort this list" in one prompt | Position bias, context limits, silently dropped or hallucinated items |
| "Which is better, A or B?" over pairs | Ordinal only — throws away *how much* better; naive schedules cost O(n²) |
| Elo / Bradley–Terry over wins | Better aggregation, but still magnitude-blind and usually passive about which pair to ask next |

`cardinal-harness` treats each ratio answer as a noisy log-space measurement,
fits latent scores over the whole comparison graph with a robust solver (IRLS,
Huber loss), reads uncertainty off the posterior, and plans the next
comparison by effective resistance on the graph. Default budget is 4·n
comparisons — O(n), not O(n²). The planner's efficiency is MEASURED, not
assumed: our regret benchmark (`tests/planner_regret.rs`) initially caught
it losing to uniform random pair selection; the fix (anchor-diverse
exploration, replacing a hub-and-spoke geometry) now has it winning at
scarce budgets (where saving comparisons matters), tying at medium ones,
and slightly trailing random at large budgets on global order — all pinned
two-sided, history preserved in the test file and issue #43. The receipts
culture applies to our own planner first.

What you get that the alternatives don't, in one package:

- **Cardinal magnitudes**, not just an order — "A is ~3× better" survives into the output.
- **Uncertainty per item** and a top-k error estimate, honestly reported.
- **Active pair selection** and principled stopping under an explicit budget.
- **Counterbalancing by default**: every planned pair is asked in both presentation orders, and the disagreement rate is reported — position bias measured, not assumed away.
- **Attribute health probes**: judge the opposite side of your criterion (`--two-sided`) and alternate phrasings (`--also-by`), get rank-consistency receipts that tell you whether the attribute even coheres for this judge.
- **Receipts**: JSONL trace per judgement, token/cost accounting, SQLite cache, seeded reproducibility.

## Quickstart

```bash
cargo install cardinal-harness
export OPENROUTER_API_KEY=your_key_here   # any model on OpenRouter

cardinal sort examples/sort-demo.txt --by "usefulness as advice for a software engineer" --scores
```

Real output (preserved, with full receipts, under
[`artifacts/live/sort-demo-2026-07-02/`](artifacts/live/sort-demo-2026-07-02/)):

```text
1.082±0.741	premature optimization is the root of all evil
0.609±0.785	measure twice, cut once
0.514±0.772	a chain is only as strong as its weakest link
0.451±0.768	don't put all your eggs in one basket
0.327±0.758	practice makes perfect
0.258±0.785	if it ain't broke, don't fix it
0.180±0.783	too many cooks spoil the broth
0.000±0.737	a bird in the hand is worth two in the bush

sorted 8 items by "usefulness as advice for a software engineer" · 32 comparisons (1 cached, 0 refused) · $0.0500 · stop: budget_exhausted
```

Note what the receipt admits: at the default 4·n budget those posterior stds
overlap. You get a well-motivated point estimate of the order, not a certified
one — raise `--budget` or focus `--top-k` when you need certainty. The same
run replays offline, keyless, for $0 via `--cache-only`.

`sort` reads newline-delimited items or a JSON array (of strings or
`{"id","text"}` objects) from a file or stdin, and writes plain lines (pipeable),
`--format json|jsonl|csv`, `--scores`, `--reverse`, `--trace trace.jsonl`.
A sort where every comparison fails refuses to print, loudly.

## The evidence path: logprobs as judgements

With `--template ratio_letter_v1`, each comparison asks for ONE letter from
a 52-token alphabet (case = which item, letter = magnitude on the ladder,
`A` = parity, `!` = refuse) — so a single completion position's top-k
logprobs are the model's **full judgement PMF**. The solver then weights
each observation by its **measured variance** instead of a stated
confidence. Rendering, parsing, and mass accounting are delegated to
[seriate](https://github.com/XyraSinclair/seriate); where a provider hides
or rejects logprobs, the path degrades loudly to sampled mode and the run
summary says so (`evidence: 63/63 logprob-mode, visible 0.99`).

Live head-to-head at equal budget and cost on gpt-5.4-mini
([receipts](artifacts/live/evidence-path-2026-07-04/)): top-to-bottom
separation **≈4.0σ vs ≈1.4σ** for the canonical JSON path — roughly 3× the
resolving power per dollar, because each call consumes the model's prior
instead of one sample from it. Caveats in the receipt pack, including that
the two instruments induce correlated but not identical orderings
(Spearman 0.74) — different elicitations tap different priors.

## Healthy elicitation, by default

Most LLM-annotation pathologies are invisible unless you deliberately measure
them. `sort` measures them as part of the run:

- **Both orders, every pair.** LLM judges favor whichever item is presented
  first. Randomizing order (still available via `--no-counterbalance`) only
  averages that bias; the default asks each planned pair in both orders,
  cancels the bias per-pair, and reports the disagreement rate.
- **The opposite side of the attribute.** `--two-sided` also judges
  `lack of <criterion>` (weight −1) and reports whether the two sides mirror
  each other. If they don't, your attribute doesn't mean anything stable to
  this judge — better to learn that before trusting the annotation.
- **Alternate phrasings.** `--also-by "<paraphrase>"` judges the criterion
  under other wordings and reports cross-phrasing rank consistency.

A real run (preserved under
[`artifacts/live/healthy-sort-demo-2026-07-02/`](artifacts/live/healthy-sort-demo-2026-07-02/)):

```console
$ cardinal sort examples/sort-demo.txt \
    --by "usefulness as advice for a software engineer" \
    --two-sided --also-by "how much practical value it offers someone building software" \
    --model anthropic/claude-sonnet-4.6 --budget 120

sorted 8 items · 120 comparisons · $0.3441 · order flips: 11/51 · stop: budget_exhausted
probe [opposite]   "lack of usefulness as advice...": consistency +0.81 — consistent
probe [paraphrase] "how much practical value...":    consistency +0.35 — shaky
```

That run caught the judge reversing itself on **21.6% of pairs** under order
swap, confirmed the criterion is two-sided coherent, and flagged that a
reasonable-sounding paraphrase materially changes the ranking. None of this is
visible in a single-prompt sort or a 1–10 rating pass.

## Building taste: elaborate, judge, explain

Good sorts start with good attribute prompts. Three commands help you get
there, from most magic to most manual:

- **`cardinal elaborate --by "impact"`** — one LLM call expands a terse
  criterion into a precise judging rubric (definition, what counts as more,
  what must not be rewarded), printed to stdout so it composes:
  `cardinal sort list.txt --by "$(cardinal elaborate --by impact)"` — or use
  `cardinal sort --elaborate` to do it inline. The rubric is always shown:
  the magic stays inspectable and editable.
- **`cardinal judge "<A>" "<B>" --by "<criterion>" --show-prompt`** — the
  lowest-level primitive: one pairwise judgement, with the fully rendered
  prompt on stderr and the parsed answer (direction, ratio, confidence,
  cost) on stdout. This is how you develop taste for what a criterion
  actually asks of the judge. Add **`--spin`** for the susceptibility
  probe: the same pair judged under neutral, pro-first, and pro-second
  requester framings (each in both presentation orders, 6 comparisons),
  reporting how far the judgement moves when the asker leans — and whether
  the belief survives the spin at all. A judgement only deserves the name
  *belief* if it is a fixed point of framings that shouldn't matter.
- **`cardinal explain ranking.txt --candidate "clarity" --propose 3`** — the
  inverse problem: you already HAVE a ranking you believe in. Explain
  measures candidate attributes (yours, plus LLM-proposed ones) with the
  normal pairwise machinery and reports which of them — alone and in fitted
  non-negative combination — reconstruct your order.
- **`cardinal weigh --goal "ship v1 fast" --propose 6`** — automated AHP:
  the model decomposes the goal into judgeable considerations (or you pass
  `--attribute name=description` yourself), each pair is judged on
  importance *for that goal*, and the solver's softmaxed log-latents come
  out as a normalized ratio-scale priority vector — weights ready to feed
  back into multi-attribute reranking.
- **`cardinal canonize list.txt --by "depth" --judges m1,m2`** — the
  merge protocol for attribute wordings: the seed plus LLM-proposed
  refinements, each measured over the entities by EVERY judge model, and
  ranked by transmissibility — the mean cross-judge rank agreement of the
  induced latents. An attribute is a communication primitive exactly when
  different minds recover the same cardinal latent from it; this measures
  that, with redundancy receipts against your already-accepted dimensions.
- **`cardinal distinguish list.txt --focus 12`** — the propagation
  primitive: given a set and one focal item, propose (or pass `--by`)
  candidate attributes, measure ALL of them over the whole set, and report
  where the focal item actually lands per attribute — percentile and
  z-score, best direction first. The proposal is a hypothesis; the measured
  profile is the receipt. This is how you find the attribute under which a
  differentiated item deserves to travel.

A real explain receipt (preserved under
[`artifacts/live/taste-tools-demo-2026-07-02/`](artifacts/live/taste-tools-demo-2026-07-02/)),
run against a ranking whose true generating attribute was known:

```text
attribute                                    | alone ρ | weight
---------------------------------------------|---------|-------
usefulness as advice for a software engineer |   +0.98 | 0.85
relevance to software engineering principles |   +0.79 | 0.01
encourages proactive careful planning        |   +0.21 | 0.00
wisdom applicable to technical decision-maki |   +0.81 | 0.15

weighted combination reconstructs your ranking at ρ = +0.98
```

The true attribute is recovered — top standalone correlation and dominant
fitted weight — while three plausible decoys are down-weighted. (One run, one
list; a demonstration of the mechanism, not a benchmark.)

When only the top of a list matters, `--top-k K` focuses the planner on the
K-boundary and `--prune-below <p>` additionally stops spending exploration
comparisons on items whose posterior chance of reaching the top-K drops below
`p` — the pruned count lands in the receipts as `entities_pruned`.

## The Judge Coherence Benchmark

`cardinal bench --models a,b,c` scores models on *judgement* quality with
no ground-truth labels: internal consistency under meaning-preserving
transformations (order swap, reciprocal antisymmetry, cyclic frustration,
framing spin, polarity reversal, paraphrase stability, null calibration)
times a signal axis so a judge cannot ace it by refusing to discriminate.
A genuine belief is a fixed point of the transformations that shouldn't
matter — that's testable without knowing any right answers, which makes it
a benchmark labs can hill-climb without it being memorizable.

The dimensions cross-check (a content-blind hash judge aces order
invariance but can't know the negated attribute must reverse; a sycophant
keeps its correlations and loses spin), and the benchmark validates itself:
five scripted pathological judges — oracle, constant, position-biased,
sycophant, cyclic — run the full battery in the test suite, and each must
be caught by exactly the dimension that names it. 114 comparisons per
model, ~$0.05 on mini-class models, every rate with its denominator and
95% CI. Full argument, formulas, gaming analysis, and honest caveats:
[`docs/BENCHMARK.md`](docs/BENCHMARK.md). Live leaderboard receipts:
[`artifacts/live/judge-bench-2026-07-05/`](artifacts/live/judge-bench-2026-07-05/).

To *feel* what the spin axis measures, open the interactive receipt viewer
([`artifacts/live/receipt-viewer-2026-07-08/`](artifacts/live/receipt-viewer-2026-07-08/),
serve `artifacts/live/` with any static server): a contested pair, a
framing-field slider from insistent-pro-B to insistent-pro-A, and the
judge's measured belief moving under it — gpt-5.4-mini echoing at
+0.200 nats/step while claude-sonnet-4.6 holds direction at every field
point. Every number on the page is selected from committed receipts (never
interpolated), and a regression test
([`tests/live_artifact_pages.rs`](tests/live_artifact_pages.rs)) pins the
page to the receipt bytes so they cannot drift apart.

## Library

```rust,no_run
use std::sync::Arc;
use cardinal_harness::gateway::NoopUsageSink;
use cardinal_harness::rerank::{sort_texts, RerankExecution, SortOptions};
use cardinal_harness::{Attribution, ProviderGateway};

# async fn demo() -> Result<(), Box<dyn std::error::Error>> {
let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;
let execution = RerankExecution::new(Arc::new(gateway), Attribution::new("app::sort"));

let sorted = sort_texts(
    vec![
        "First essay...".into(),
        "Second essay...".into(),
        "Third essay...".into(),
    ],
    "clarity of explanation",
    execution,
    SortOptions::default(),
)
.await?;

for item in &sorted.items {
    println!("{:>2}. {:.3} ± {:.3}  {}", item.rank, item.latent_mean, item.latent_std, item.text);
}
println!("cost: ${:.4}", sorted.meta.provider_cost_nanodollars as f64 / 1e9);
# Ok(())
# }
```

`sort_documents` is the same with caller-owned ids. For multiple weighted
attributes, hard gates ("must be above the 25th percentile on safety"),
top-k-focused stopping, model ladder policies, and caching, use the full
`multi_rerank` API — see [docs/WORKED_EXAMPLE.md](docs/WORKED_EXAMPLE.md).

## Scope

This repo is intentionally narrow. It contains:

- canonical pairwise ratio prompts (the ratio ladder and JSON judgement contract)
- robust score fitting over pairwise observations
- multi-attribute reranking, gating, and top-k stopping
- OpenRouter gateway, pricing, usage, and SQLite cache support
- synthetic evaluation and reporting

Research workflows, training/export code, agent orchestration, and other
experimental layers belong in `openpriors-research`, not here.

Use it for list work where "how much better?" carries information: prompts,
research ideas, candidate plans, reviewer notes, backlog items — any shortlist
where the top cluster matters more than a cheap total order. Do not use it for
deterministic rankings, scalar metrics, or attributes too incoherent to
compare.

## Evidence status

The trade is explicit: this costs more than one-shot scoring, saves
comparisons versus exhaustive pairwise judging, and returns uncertainty plus
receipts instead of only a sorted list. The public evidence is deliberately
reproducible and deliberately narrow — it does **not** show a universal win:

- Offline synthetic evaluation and Likert/scalar comparison receipts live under `artifacts/eval/`.
- Preserved real OpenRouter cardinal-policy receipts live under `artifacts/live/openrouter-benchmark-2026-06-30/`: three policy runs, 459 fresh provider comparisons, 0 cache hits, 0 refusals, $0.994335 provider-reported cost.
- A live structured-judgment method comparison lives under `artifacts/live/method-comparison-2026-06-30-suite-v1/`: scalar matrix vs whole-list sort vs ordinal pairwise vs cardinal pairwise-ratio, judged against a separate LLM reference across six frozen task families.
- A live `sort` demo receipt lives under `artifacts/live/sort-demo-2026-07-02/`.
- `tests/live_method_receipts.rs` guards the live method pack: schema version, frozen suite hash, per-call receipts, usage totals, and absence of provider keys or local paths.
- The compact five-metric offline summary is mixed: 10 cardinal wins, 12 Likert wins, 18 ties. The raw-receipt delta reports 10/12/20 across 42 comparable rows.
- The live method comparison is also mixed: cardinal ties the best regime on two task families, stays close on two, and lags sharply on two.
- All current cardinal synthetic runs stop at `budget_exhausted`; the receipts do not prove early stopping or lower cost.
- Equal call counts are not equal token cost: pairwise prompts carry two items, scalar prompts one.
- An adversarial test battery (266 tests, [docs/TESTING.md](docs/TESTING.md)) pins the solver's mathematical claims — planted-truth recovery, Huber influence bounds, calibration coverage, pathological-judge behavior, method head-to-heads — and its honest negatives: ordinal beats ratio under heavy noise, and the budget-efficiency claim remains unproven.

The next empirical proof target is a larger frozen benchmark with repeated
runs, equalized token or dollar budgets, more held-out task families, and
human or high-budget external reference judgements. Details in
[docs/EVALUATION.md](docs/EVALUATION.md).

## Core idea

Instead of asking an LLM for unstable absolute scores, ask:

> How many times more of attribute X does A have than B?

Each answer becomes a noisy log-ratio observation on a fixed ladder
(1.0 … 26.0, geometric). Log-ratios compose additively, so the full comparison
graph over-determines the latent scores; a robust solver (IRLS + Huber)
downweights outlier judgements, the posterior gives per-item uncertainty, and
the planner targets the pair whose observation most reduces uncertainty about
the top-k boundary. Full rationale in [docs/ALGORITHM.md](docs/ALGORITHM.md)
and the compact mathematical contract in [docs/MODEL.md](docs/MODEL.md).

## Prompt surfaces

Two prompt templates are supported:

| Slug | Output shape | Use when |
|------|--------------|----------|
| `canonical_v2` | `{"higher_ranked":"A|B","ratio":1.0..26.0,"confidence":0.0..1.0}` | Default pairwise-ratio judgement. Use this unless you specifically need bucket-token logprobs. |
| `canonical_bucket_v1` | `{"higher_ranked":"A|B","ratio_bucket":0..16,"confidence":0.0..1.0}` | Bucket-index variant for runs that need to map output logprobs onto the fixed ratio ladder. |
| `ordinal_v1` | `{"higher_ranked":"A|B","confidence":0.0..1.0}` | Natural direction-only judgement; enters the solver as a fixed modest log-ratio. Strictly less informative than ratios — use as a baseline/control or when magnitude questions confuse the judge. |

Both templates use the same ratio ladder and the same refusal shape:
`{"refused":true}`. Unknown `prompt_template_slug` values are rejected.
Details in [docs/PROMPTS.md](docs/PROMPTS.md).

## CLI

Beyond `sort`, the CLI exposes the full request surface:

```bash
# Validate a multi-rerank request locally: no API key, cache, or network.
cargo run --bin cardinal -- validate --request examples/multi-rerank-request.json

# Full multi-attribute rerank from JSON, with trace and markdown report.
cargo run --bin cardinal -- rerank \
  --request examples/multi-rerank-request.json \
  --out output.json --trace trace.jsonl --report report.md

# Model policies: built-in names or JSON files.
cargo run --bin cardinal -- rerank \
  --request examples/multi-rerank-request.json \
  --policy frontier_ladder --out output.json
#   examples/model-policy-quality-only.json    -> anthropic/claude-opus-4.6
#   examples/model-policy-cost-aware-fast.json -> deepseek/deepseek-v4-flash
#   examples/model-policy-frontier-ladder.json -> opus 4.6 -> gemini 3.1 pro preview -> gpt-5.4-mini

# Expand one request across prompt templates and attribute variants.
cargo run --bin cardinal -- experiment-expand \
  --request examples/multi-rerank-request.json \
  --prompt-template canonical_v2 --prompt-template canonical_bucket_v1 \
  --include-negative --variant-json examples/prompt-experiment-variants.json \
  --out expanded-request.json

# Generate a report later from a saved request + response.
cargo run --bin cardinal -- report \
  --request examples/multi-rerank-request.json \
  --response output.json --out report.md

# Offline synthetic evaluation receipts (no API key).
cargo run --bin cardinal -- eval --out artifacts/eval/synthetic_eval.jsonl --curve-csv artifacts/eval/synthetic_curves.csv
cargo run --bin cardinal -- eval-likert --out artifacts/eval/likert_eval.jsonl --curve-csv artifacts/eval/likert_curves.csv
cargo run --bin cardinal -- eval-compare --mode ratio --out artifacts/eval/comparison_summary.json

# Cache management.
cargo run --bin cardinal -- cache-export --out cache.jsonl
cargo run --bin cardinal -- cache-prune --max-age-days 30
```

Live benchmark scripts (`examples/live_openrouter_benchmark.py`,
`examples/live_method_comparison.py`) reproduce the checked-in receipt packs;
both require `OPENROUTER_API_KEY` and spend provider credits.

## Architecture

| Module | Purpose |
|--------|---------|
| `rating_engine` | Robust IRLS solver and comparison planning |
| `trait_search` | Multi-attribute utility composition, gating, top-k uncertainty |
| `rerank` | Orchestration loop, comparison execution, stopping, traces, reports |
| `rerank::sort` | List-in, list-out sorting convenience over the same engine |
| `prompts` | Canonical pairwise ratio prompt and ratio ladder |
| `cache` | SQLite-backed memoization for pairwise judgements |
| `gateway` | OpenRouter client, pricing, usage, attribution |
| `text_chunking` | Token-aware chunking helpers |

```text
list or request JSON
  -> rerank manager
  -> per-attribute rating engines
  -> active planner
  -> OpenRouter gateway + SQLite cache
  -> sorted output + trace JSONL + response JSON + markdown report
```

## Documentation

- [docs/MATH_FRONTIER.md](docs/MATH_FRONTIER.md): the mathematical roadmap for cardinal & stable prior elicitation — Hodge split (shipped), spectral identifiability, elicitation-program equivalence, stochastic transitivity, pooling theory — with rejections recorded as findings
- [docs/PRINCIPLES.md](docs/PRINCIPLES.md): the anti-slop discipline — refutability, scripted-pathology validation, denominators, mathematical register — each rule with the receipt that earned it
- [docs/ALGORITHM.md](docs/ALGORITHM.md): scoring, uncertainty, stopping, and evaluation rationale
- [docs/MODEL.md](docs/MODEL.md): compact mathematical contract, assumptions, and failure modes
- [docs/PROMPTS.md](docs/PROMPTS.md): supported prompt templates, output contracts, and JSON request examples
- [docs/WORKED_EXAMPLE.md](docs/WORKED_EXAMPLE.md): concrete rerank walkthrough with request shape, gates, stop reasons, uncertainty, cache, and reproducibility receipts
- [docs/EVALUATION.md](docs/EVALUATION.md): checked-in synthetic evaluation receipts and an honest cardinal-vs-Likert comparison
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): scaling harness and current dense-solver receipt
- [docs/TESTING.md](docs/TESTING.md): the adversarial test battery — what it attacks, the two solver bugs it found, and the honest negatives it pinned
- [docs/WHAT_WHY_HOW.md](docs/WHAT_WHY_HOW.md): the one-page shareable version — exactly what this is good for, why, and how, with receipts for every claim
- [docs/FIRST_PRINCIPLES.md](docs/FIRST_PRINCIPLES.md): the type system of structured judgement — the instrument grid (arity × scale × output-form), the invariance group of a belief, efficiency theory, and the honest occupancy map of which cells this repo fills
- [docs/COMPARISON.md](docs/COMPARISON.md): how this relates to RankGPT-style listwise ranking, pairwise preference prompting, Bradley–Terry/Elo aggregation, and query-relevance rerankers

## License

MIT
