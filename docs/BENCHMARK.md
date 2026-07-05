# The Judge Coherence Benchmark (JCB)

`cardinal bench --models a,b,c` — score LLM *judgement* quality with **no
ground-truth labels**, purely from internal consistency under
meaning-preserving transformations, plus a signal axis so consistency
cannot be faked by refusing to discriminate.

## The argument

Every LLM-judge eval in circulation scores *agreement with a reference*
(human labels, a stronger model, majority vote). That measures conformity,
not judgement. But a judgement has an internal structure that can be tested
without any reference at all: a genuine belief must be a fixed point of the
transformations that shouldn't matter. Swap which item is shown first — the
direction must hold. Ask the reciprocal question — the answer must invert
exactly. Chain comparisons around a cycle — the ratios must compose. Negate
the criterion — the ranking must reverse. Reword it — the ranking must
survive. Lean on the judge — the belief must not follow the leaning. Show
the same item twice — the judge must tie.

None of these require knowing what the *right* answer is. They are the
measurement-theoretic preconditions for the answers meaning anything at
all. A model that fails them is not a judge with different taste — it is
not a judge. That makes this a canonically shaped benchmark for labs to
hill-climb: improving it cannot be done by memorizing answers, only by
making the model's preference structure more coherent — and coherent
preference structure is upstream of every judging application (reward
models, rankers, evaluators, feed curators).

The honest limit, stated plainly: coherence is necessary, not sufficient. A
judge with coherent but alien taste aces this benchmark. JCB measures
whether the instrument is *rigid*; whether it is *aimed well* is a separate
question that does need references. Labs should treat JCB as the
qualification test that makes reference-based evals meaningful: agreement
scores from an incoherent judge are noise about noise.

## Dimensions

114 comparisons per model on a fixed public corpus (8 short texts, judged
by "depth of insight about living well"), all through the ordinary
counterbalanced pairwise machinery:

| # | Dimension | Measures | Raw unit | Subscore |
|---|---|---|---|---|
| 0 | **signal** | mean fused \|log-ratio\| across 20 pairs | nats | 1 − e^(−x) |
| 1 | **order flip** | direction reversals under slot swap (decisive pairs) | rate + Wilson CI | 1 − rate |
| 2 | **order residual** | \|m_fwd − m_rev\|/2, reciprocal antisymmetry | nats | e^(−x) |
| 3 | **frustration** | Hodge curl fraction of the fused comparison graph | curl fraction | 1 − x |
| 4 | **spin survival** | belief direction stable under pro/con asker framings (3 clear pairs × 6 calls) | rate + Wilson CI | rate |
| 5 | **susceptibility** | mean \|χ\| = \|(m₊ − m₋)/2\| over 5 spin pairs | nats/spin | reported, unscored |
| 6 | **polarity** | Spearman ρ of scores vs the negated attribute | ρ | (1 − ρ)/2 |
| 7 | **paraphrase** | Spearman ρ of scores vs a reworded attribute | ρ | (1 + ρ)/2 |
| 8 | **null bias** | mean \|log-ratio\| on identical-item pairs | nats | e^(−x) |

**Composite**: `coherence` = mean of available consistency subscores
(1,2,3,4,6,7,8); **`JUDGE SCORE` = signal-subscore × coherence.**
Multiplicative on purpose: zero discrimination or zero consistency zeroes
the headline.

## Why it is hard to game

The dimensions cross-check — each obvious exploit of one axis is caught by
another:

| Exploit | Aces | Caught by |
|---|---|---|
| always answer "tie" | every consistency axis | signal = 0 → headline 0 |
| deterministic content hash (order-invariant, decisive) | order, residual, null | polarity (hash can't know the negated attribute must reverse) and paraphrase (or that a rewording must not) |
| always prefer slot A | decisiveness | order flip = 100%, null bias = ln(ratio), fused signal = 0 |
| locally-consistent cyclic preferences | order, residual | frustration (Hodge curl) |
| agree with whoever's asking | signal + correlations | spin survival = 0, χ large |

**These are not hypotheticals — they are the test suite.** Five scripted
judges (oracle, constant, position-biased, sycophant, cyclic) run the full
benchmark in-process (`tests/judge_bench.rs`); each pathology must land in
exactly the dimension that names it, and the oracle must lead the board.
A benchmark that can't separate scripted pathologies has no business
ranking labs.

## Reading a report

```text
model: openai/gpt-5.4-mini · template: canonical_v2
  signal           +0.812 [+0.6, +1.0] nats (n=20)  → 0.556
  order-flip       +0.050 [+0.009, +0.236] rate (n=20)  → 0.950
  ...
  coherence 0.871 · JUDGE SCORE 0.484 · 114 comparisons · $0.05
```

Every rate carries its denominator and a 95% interval (Wilson for rates,
±1.96 se for means). `signal` is in nats: 0.7 ≈ "typically feels a 2×
difference". Raw per-call receipts ship in the JSONL output
(`--out reports.jsonl`) so any number in the table can be recomputed from
the judgements themselves.

## Scope and honest caveats

- **v1 corpus is small and single-domain** (8 aphorisms, 1 attribute
  family). This bounds the CIs — order-flip on 20 pairs has a ±10pt
  interval. It is a standardized instrument, not a final exam; scale the
  corpus before treating gaps under ~0.1 as real.
- **Coherence ≠ correctness** (see above). Pair with reference-based evals;
  JCB is the precondition, not the replacement.
- **Frustration includes quantization curl** from the ratio ladder
  (documented in FIRST_PRINCIPLES §5½); the same floor applies to every
  model, so rankings are unaffected, but absolute curl should not be read
  as pure preference cycling.
- **Susceptibility is one field magnitude**, not a full response curve —
  reported, not scored.
- Spin verdicts on contested pairs are legitimately "undetermined" (a tie
  at zero field has no direction to preserve); only clear pairs enter the
  survival rate, and the denominator is printed.

## Reproduce

```bash
cardinal bench \
  --models "openai/gpt-5.4-mini,google/gemini-2.5-flash,..." \
  --no-cache --out reports.jsonl
```

`OPENROUTER_API_KEY` required; ~$0.05/model on mini-class models. Live
receipts: `artifacts/live/judge-bench-2026-07-05/`.
