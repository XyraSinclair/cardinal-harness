# Pre-registration: weak-judge amplification + cardinal recovery (2026-08-07)

Registered before any benchmark call. Amendments, if any, get dated entries
below this header before the affected cells run.

AMENDMENT (2026-08-07, pre-run): the corpus gap floor is 1.34, not 1.35 —
the Uruguay/Botswana true ratio is 1.345 and truth values are not adjusted
to fit the registered bound. No cell had run.

ERRATUM (2026-08-07, mid-run): the first cardinal-mini invocation received
corpus.json in a `{"items": [...]}` wrapper; the CLI fell back to text-line
parsing and "sorted" 96 JSON syntax lines ($0.1445, 369 comparisons,
discarded, counted toward the $5 cap). corpus.json rewritten as the plain
array the CLI expects; baseline cells were unaffected (they read the same
23 names either way). Cardinal cells re-run on the corrected file.

## Claim under test

"Best library for prior elicitation and LLM list sorting" decomposes into two
falsifiable properties no ordinal baseline shares:

1. **Cardinal recovery.** The pipeline returns calibrated *magnitudes*
   (latent log-scores), not just an ordering. Test: correlation of
   `latent_mean` against ln(true value) across 5.2 orders of magnitude.
2. **Weak-judge amplification.** Robust IRLS over many noisy pairwise ratio
   judgments extracts an ordering better than what the same weak judge can
   state directly. Test: cardinal-vs-baseline accuracy gap as a function of
   judge strength (mini vs nano).

## Corpus

23 countries, attribute "current national population". Truth: UN 2024
estimates (recorded in truth.json). Consecutive true ratios all ≥ 1.34, so
the true ordering is unambiguous even under ±10% truth error. Span:
1.45e9 (India) to 1.0e4 (Tuvalu), 5.2 orders of magnitude. Item text is the
country name only — this is a prior-elicitation task, not retrieval.

## Cells

Judges: `openai/gpt-5.4-mini` (strong), `openai/gpt-5.4-nano` (weak), both
via OpenRouter (vrun key), temperature defaults.

Methods per judge:

- **M1 pointwise direct estimate**: one call per item, "estimate the
  population, answer with a number". Sort by estimate. (23 calls)
- **M2 pointwise 0-100 score**: one call per item, "score how populous on
  0-100". Sort by score. The common-practice baseline. (23 calls)
- **M3 listwise one-shot**: single call, full list in, ordered list out.
  One retry on malformed output; a second malformed output scores the cell
  as failed. (1 call)
- **M4 cardinal sort**: `cardinal sort --by "current national population"
  --seed 20260807 --no-cache`, default plan. (~2n comparisons)

One run per cell (n=1 per cell is a registered limitation; the cardinal run
is seeded, pointwise runs are near-deterministic at temp default).

## Metrics

- Primary: Spearman rho vs truth, per cell. Secondary: Kendall tau.
- Cardinal-recovery: Pearson r of `latent_mean` vs ln(truth), plus fitted
  slope (latent is relative; affine fit, r is primary, slope reported).
  Same computed for M1 (ln estimate vs ln truth) as the comparator —
  M2's 0-100 scale is structurally incapable of spanning 5 orders and its
  compression is itself a result to report.
- All claims carry denominators (23 items, 253 rank pairs).

## Hypotheses (registered)

- **H1 (amplification):** with the nano judge, M4 Spearman exceeds every
  baseline's Spearman. Directional prediction: the M4−M1 gap is larger
  under nano than under mini.
- **H2 (no strong-judge penalty):** with the mini judge, M4 is within 0.05
  Spearman of the best baseline (small famous-entity lists are easy;
  parity is the honest expectation).
- **H3 (cardinal recovery):** M4 latent_mean vs ln(truth) Pearson r ≥ 0.90
  with the mini judge. This is the capability no baseline provides:
  M2's score-vs-ln(truth) rank use is fine but its magnitudes cannot be
  linear in log-space across the span (prediction: visible saturation).

## Abort lines

- Total registered spend cap: $5.00. Estimate: mini cardinal ~60 comparisons
  at small prompts ≈ $0.15; nano cheaper; baselines < $0.05 total. If a
  pre-run estimate or running total exceeds the cap, stop and record.
- Any cell with > 20% failed calls after one retry pass: mark the cell
  failed, report it, do not silently resample.
- If the cardinal CLI errors on this corpus, fix nothing mid-run: record,
  abort the cell, diagnose outside the benchmark.

## What this design does not show (registered honestly)

- n=1 per cell: no run-variance estimate this round.
- One domain: population priors. A "best in the world" claim needs more
  domains; this is the first brick, chosen for unambiguous truth and
  maximal magnitude span.
- Listwise M3 at 23 items fits in context; cardinal's scaling advantage
  (lists larger than context) is not exercised here.
