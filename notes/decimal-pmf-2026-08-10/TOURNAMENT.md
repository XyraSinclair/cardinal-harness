# Elicitation-geometry tournament — synthetic frontier (2026-08-13)

> **Live rounds 1–2 update (same day): the top two synthetic arms FAILED
> their live validity checks.** Round 1 (BINCDF.md): bin-lp's percept-CDF
> assumption fails — step collapse, threshold anchoring, round-number
> attractors, near-tie incoherence. Round 2 (GRID16.md): grid16 is
> perfect-but-empty on peaked models and contaminated on flat ones
> (codebook TV to 0.55, slot bias flipping near-tie signs, 0.6–0.9 log10
> order effects). Emerging law: native-numeral geometries inherit the
> model's calibration; artificial codebooks measure their own artifact.
> Peel remains the location-faithful incumbent; radix4's validity cell is
> the one remaining challenger question.

Oracle roadmap item #1 (ORACLE-ROADMAP-2026-08-12.md): stop improving the
microscope; make different microscopes compete on whether they improve the
map. This is round one — the SYNTHETIC frontier, where the latent world is
held constant and only the elicitation geometry varies. It ranks which
geometries earn a seat in the live A/B and defines what the live kill test
must measure. It cannot, by construction, answer measurement validity on
real models (see Honest limits).

Instrument: `examples/geometry_tournament.rs`
(cargo run --release --example geometry_tournament — full tables printed;
assertions enforce the load-bearing orderings).

## Setup

- n=8 items, latent nats [0.00, 0.03, 0.10, 0.55, 0.62, 1.60, 1.75, 3.20]:
  a near-tie block (0.03/0.07 gaps — far below single-judgement noise), a
  mid cluster, clear separations above. 28 pairs × 2 counterbalanced
  presentations, evidence → `Observation::from_log_ratio_moments` → the
  production IRLS engine (same path as production multi-rerank).
- Judge percept y ~ Normal(z, 0.25 log10) per call; 1.5% multiplicative
  jitter on every logprob read (census-measured provider floor).
- Cost = calls (input-token-dominated pricing); reason arm 10 units/call.
- 12 reps per (arm, budget); budgets C ∈ {1,2,4,8,16,32} calls per
  presentation.

## Frontier (whole-matrix cost units; RMSE in nats)

| arm     | cost→RMSE≤0.15 | best RMSE | best near-tie | note |
|---------|---------------:|----------:|--------------:|------|
| grid16  |         **56** |    0.0001 |         1.000 | full 16-way PMF in ONE call |
| bin-lp  |         **56** |    0.0036 |         1.000 | ≈exact by 2 probes (σ_z ≈ 4σ_p/β) |
| radix4  |         **56** |    0.0067 |         1.000 | full 4-way conditionals/visited node |
| peel    |            117 |    0.0327 |         0.958 | incumbent decimal kernel |
| mc      |            243 |    0.0377 |         0.938 | same instrument, text only |
| bin-smp |            285 |    0.0443 |         0.917 | staircase, sampled bit |
| reason  |            560 |    0.0502 |         0.917 | σ/2.5 judge, point answer, 10×/call |

Readings:

1. **Channel shape dominates redraw count.** Geometries whose stochastic
   nodes fit inside top-k (grid16: 16 ≤ 20; radix4: 4-way) read the whole
   local PMF per call and saturate almost immediately. The decimal grammar
   spreads mass over ~260 leaves with top-5 sidebands per node, so
   enumeration is the bottleneck: peel needs ~8 calls to reach where
   radix4 is after 1–2. The 47× atom-vs-frequency result survives (peel
   beats mc at every matched budget) but both sit an order of magnitude
   off the single-call-PMF frontier in this world.
2. **The logprob read is worth ~10–30× budget on the same geometry**:
   bin-lp reaches 0.0036 at 2 probes; bin-smp (identical staircase,
   sampled bit) needs >32 to get near 0.04. This is the cleanest isolation
   of "what do logprobs buy" in the tournament.
3. **A stronger no-logprob reasoner loses on cost**: reason plateaus
   around 0.05–0.08 RMSE at 560–2240 units — every logprob arm reaches
   better accuracy at ~10–40× less spend. (Its C=16 coverage dip is the
   2-draw sample-variance artifact — honest small-n variance estimation,
   not a judge failure; mc shows the same at C=2.)
4. **Fusion stays honest end-to-end**: matrix-level cover2 ≈ 1.0 for every
   calibrated logprob arm at every budget — the disagreement-var fix from
   the battery carries to all geometries through the same engine seam.

## Distortion appendix — instrument validity is the dangerous layer

| distortion | endpoint effect |
|---|---|
| peel + 30% snap-to-.0 attractor | transient: near-ties degraded at C≤4, recovered by C=8 (0.070 RMSE), tau 0.994 by C=32 — the credal kernel absorbs it |
| grid16 + codebook edge shift +0.05 log10 | **silent bias floor 0.086 nats at every budget, coverage collapsing 1.00→0.25 as calls grow** — miscalibration is invisible to the instrument and confidence grows around the wrong value |
| bin-lp with σ̂ = 0.75σ or 1.25σ (β miscal) | transient: bad at C=1–2, self-corrects by C=4 (the adaptive staircase drives thresholds to p≈0.5 where β error vanishes to first order) |

This is Oracle's "precision about the wrong thing" made concrete: the most
efficient geometry (grid16) is the most fragile to codebook
misspecification — and fails in the worst possible way, overconfident and
stable. The staircase is the most self-correcting geometry because
adaptivity keeps it operating where its calibration error is
second-order. Decimal peel sits between: distortion costs draws, not
correctness.

## Verdict for the live A/B

Take to real providers, in order:

1. **bin-lp (adaptive offset-binary, logprob-read)** — near-frontier
   efficiency, best distortion robustness, provider-universal fallback
   (same instrument degrades to bin-smp when logprobs vanish), and its
   likelihood is a clean logistic factor for the future
   distribution-native solver.
2. **grid16 (single-token log codebook)** — the efficiency ceiling; must
   ship with a codebook-calibration check (paraphrase/instrument
   invariance probes) because its failure mode is silent.
3. **radix4** — grid16's resolution without the coarse-bin bias, at 1–2
   calls; the natural deep-grammar upgrade of peel.
4. **peel (incumbent)** — the control arm; keeps the validated kernel in
   the race.

Kill test (unchanged from Oracle): on real models across ≥2 domains, does
the best logprob geometry reach target held-out error at ≥2× lower cost
than point/MC/reasoner baselines after every arm gets the same active
selection? The synthetic frontier says the margin available is ~4–10× IF
real decoder PMFs are faithful to latent judgement — which is exactly the
open question the live run exists to answer.

## Honest limits

- The synthetic world equates decoder fidelity with measurement validity:
  every arm's PMF is the exact pushforward of the same latent percept. On
  real models the mapping latent→token-PMF may itself be
  geometry-dependent (codebook effects, surface-form priors). grid16's
  0.0001 is a channel-capacity statement, not a prediction.
- Single fixed judge σ; no instrument×item interaction beyond the
  distortion appendix; no refusals or parse failures modeled.
- bin-lp assumes the probe re-samples the percept each call and that the
  answer-token probability equals the percept CDF — the most
  psychometrically loaded assumption in the set; the live A/B must check
  it first (it is also the cheapest to check: ~2 probes/pair).
