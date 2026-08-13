# Live A/B round 1 — bin-lp percept-CDF check: the assumption FAILS (2026-08-13)

TOURNAMENT.md named bin-lp (adaptive offset-binary staircase, logprob-read)
the winning synthetic geometry and its percept-CDF assumption the most loaded
and cheapest thing to check live. Checked: `bincdf_probe.py`, 456 calls,
$0.18, OpenRouter/Azure, effort none, byte-identical 2-enum grammar
`{"answer": "yes"|"no"}`, 15-ratio ladder × both presentation orders ×
4 pairs × 3 models, repeats at anchor thresholds. Raw: `bincdf_results.json`;
per-cell fits: `bincdf_summary.json`.

The assumption — p(yes | "is A more than R× B?") = P(percept > log10 R) for
a threshold-independent percept distribution — is **disconfirmed in exactly
the regimes where it would have carried value**:

1. **Step collapse on strong models.** gpt-5.6-sol's fitted logistic scale is
   s ≈ 0.016–0.025 log10 on ALL four pairs — the answer token is a
   deterministic comparator, not a CDF read. On (sol, cat-vs-raccoon) the
   decimal-grammar instrument shows genuine percept width on the same latent
   (max CDF gap between the two instruments 0.555): the decimal grammar
   exposes uncertainty; the binary grammar collapses it to a point answer.
   The logprob read carries ~1 bit, not the 10–30× synthetic information win.

2. **Location disagreement on flat models — instrument anchoring.** Where
   width information would matter most, the binary instrument converges to a
   DIFFERENT location than the decimal instrument on the same (model, pair):
   egg-vs-bowling-ball μ̂ = −2.65 (5.4-mini) / −3.24 (4.1-mini) vs decimal
   E[Z] ≈ −1.7/−1.85 and truth ≈ −1.98; cross-instrument max|ΔF| = 0.82/1.00.
   The in-context threshold drags the percept (same mechanism as census
   finding 1b: the instrument is in the model's context and changes beliefs).

3. **Round-number attractor in the probe's own parameter.** 5.4-mini
   whale-vs-elephant (truth ≈ 25×), AB ladder: p_yes = 0.777 @ R=11,
   0.269 @ 17, **0.500 @ 25**, 0.148 @ 40, 0.182 @ 65, **0.500 @ 100** —
   non-monotone (3 violations), max logistic residual 0.443, with exact-0.5
   returns at the round thresholds. No fixed percept + tie-splitting model
   explains p@17 < p@25; the surface form of R modulates the judgement.
   This is live, direct evidence that code-point surface form contaminates
   measurements — the same risk class as grid16's codebook.

4. **Near-tie framing incoherence.** F(0) estimated from the two orders
   disagrees by 0.94 on (4.1-mini, cat-vs-raccoon): it answers NO to
   "cat > 1.0× raccoon" AND 94% NO to "raccoon > 1.0× cat" — the latter
   factually wrong; "more than 1.0×" reads as "meaningfully more", or the
   boundary nay-biases. Sweden-vs-Portugal shows 0.18–0.23 order gaps on the
   minis, 0.11 on sol. The near-tie regime — the tournament's hard scoring
   region — is where the binary instrument is least coherent.

5. Repeat drift is NOT the story: ≤ 0.12 at most points (worst 0.23,
   5.4-mini sweden interior). The failures above are structural.

## Verdict

**Kill bin-lp as a primary live arm.** What survives of the staircase is a
deterministic direction-finder (bisection on the binary attractor) — but
that attractor is biased up to ~0.9 log10 against the validated decimal
instrument on flat models, carries no width, and is incoherent at
near-ties. The synthetic frontier's transfer condition (real decoder PMFs
faithful to latent judgement) fails for this geometry.

Updated live order:

1. **grid16-class single-call PMF read** — now the primary challenger, and
   finding 3 doubly motivates shipping it WITH the codebook-calibration
   check (round-number surface forms demonstrably modulate mass).
2. **radix4** — code points are abstract digits, plausibly less
   surface-form-loaded than round-number thresholds; worth one cell of the
   same validity probe before trusting.
3. **peel (incumbent decimal)** — still the only geometry with measured
   calibrated width on real providers (SHOOTOUT.md coverage 1.00).

## Caveats

- Single day, single provider route (Azure), 4 pairs, 3 models, n=1–3 per
  ladder point. The step-collapse and round-number findings replicate across
  cells; the exact μ̂ biases are point measurements.
- The stitch assumes order symmetry; its failure at h=0 is itself finding 4,
  and the logistic fits for high-gap cells (4.1-mini cat-vs-raccoon) should
  be read as descriptive, not as percept estimates.
- "More than R times" wording may carry a strict/meaningful ambiguity at
  R=1.0; it cannot explain findings 1–3 (interior thresholds, non-round vs
  round Rs, cross-instrument location gaps).

Reproduce: `python3 bincdf_probe.py run` (~460 calls, ~$0.18, needs
`/tmp/.orkey-decimal-pmf`), then `python3 bincdf_probe.py analyze`.
