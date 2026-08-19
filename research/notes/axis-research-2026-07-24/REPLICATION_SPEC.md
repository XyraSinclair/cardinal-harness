# Wave-2 replication spec — frozen before any run (2026-07-27)

Guardian panel 2026-07-26, Feynman Move 3 (adopted in SYNTHESIS.md step 5):
convert wave 2 from demonstration to finding — or honestly relabel it.
Wave 2's tier claims currently rest on ONE small model (gpt-5.4-mini), one
run per cell, no test-retest baseline. This spec freezes the replication
design and verdict rules before results exist.

## Design

**Frozen inputs, byte-identical:** the wave-2 probe sets `wave2/<axis>.txt`
and wordings `wave2/prompts.json` for all 8 axes. No rewording — the
wording IS the axis.

**Arm 1 — test-retest (run noise floor).** Repeat all 24 original cells
(8 axes × opus46, gpt56sol, mini54) with the identical command plus
`--no-cache --seed 7`. `--no-cache` is load-bearing: cached judgments
would replay identically and fake a perfect retest. Honest caveat, stated
now: the original runs did not record a seed, so retest deltas include
pair-schedule variance as well as sampling variance — the measured floor
is an upper bound on run noise, which is the conservative direction.

**Arm 2 — small-tier generality.** All 8 axes × three additional
small-tier models, distinct families:
- `anthropic/claude-haiku-4.5`
- `deepseek/deepseek-v4-flash`
- `openai/gpt-5-mini`
Same command shape, `--no-cache --seed 7`, budget 24, `--scores`.

Outputs land in `wave2/replication/` as `sort-<axis>-<modelkey>-rep.json`
(arm 1) and `sort-<axis>-<modelkey>.json` (arm 2). Runner is resumable.
Estimated spend ≈ $2.5 (wave 2 was $1.74/24 runs; arm 2 models are cheap).
Hard abort $6.

## Frozen verdict rules

Let retest ρ = Spearman between original and repeat score vectors over the
12 items of a cell.

1. **Power gate:** if the median retest ρ across the 24 cells < 0.80, the
   probe size itself is under-powered: ALL tier claims are suspended, and
   the pre-committed remedy is more comparisons per cell — not more models,
   not rewording (Feynman's loser condition, verbatim).
2. **Tier-signature test, per admitted axis, per new small model:** the
   wave-2 T3 statistic with the new small model substituted for mini54 —
   primary decoy ranks bottom-half (≥ 7 of 12) for BOTH original frontier
   runs AND the small model places it ≥ 3 ranks higher than the best
   frontier rank. T2 analog computed alongside: (fr↔fr ρ) − mean(fr↔small ρ)
   ≥ 0.20.
3. **Per-axis verdict:** TIER-GENERAL if ≥ 2 of 3 new small models show
   the signature (T3-substitute or T2-analog, whichever admitted the axis
   in wave 2); MINI-SPECIFIC if ≤ 1.
4. **Program verdict:** "tier-divergent axes" is upgraded to a finding iff
   the power gate passes AND ≥ 3 of the 6 admitted axes come back
   TIER-GENERAL. Any other outcome: RESULTS-WAVE2.md gets an erratum on
   top relabeling wave 2 as decoy-instrument validation, and the honest
   state is recorded in the operator queue (Q5).

## Registered predictions

- `scar_tissue_density` and `eschatological_seriousness`: TIER-GENERAL
  (the mechanism — a quality-scaled prior over what genuine operational
  detail looks like — should not be mini-specific).
- No confident prediction for the other four admitted axes; recorded as
  genuinely uncertain.
- Median retest ρ ≥ 0.80 expected but unmeasured — that is the point of
  arm 1.

## Known residual confound (not addressed by this replication)

Probe items and decoys were authored by a frontier-family agent, so
"what frontiers recognize" may partly be "what the author's peer models
recognize" (Feynman seat, 2026-07-26). This replication tests small-tier
generality and run noise only. The author-family confound needs
independently-authored items and is explicitly out of scope here.
