# Probe wave 2 — spec and frozen thresholds (2026-07-25, before any run)

Eight axes from the CATALOG shortlist. Per axis: a purpose-built 12-item
set with planted decoys (engineered high-confounder / low-latent), run
under the same trio as probe 1 — claude-opus-4.6, gpt-5.6-sol (frontier),
gpt-5.4-mini (small) — 24 comparisons each, `cardinal sort`.

The wording under test is the axis key plus its one-sentence definition.
For `eschatological_seriousness` this is explicitly the probe-1 remedy
test: does a definitionally sharp wording create the tier gap that the
bare `connection_to_the_end_of_time` phrase lacked?

## Frozen admission thresholds (per axis)

- **T1 frontier coherence**: opus↔sol Spearman ≥ 0.60. Below this the
  frontiers don't see one latent; axis goes back for rewording regardless
  of anything else.
- **T2 tier gap**: (fr↔fr ρ) − mean(fr↔mini ρ) ≥ 0.20, **or** T3 fires.
- **T3 decoy discrimination**: primary decoy ranks in the bottom half
  (rank ≥ 7 of 12) for BOTH frontiers, and the mini places it ≥ 3 ranks
  higher than the best frontier rank for it.
- **Verdict**: PASS = T1 ∧ (T2 ∨ T3) → advances to battery + phrasing
  family. WEAK = T1 only → keep, reword, re-probe. FAIL = ¬T1.

Roles and per-axis predictions are recorded in each `wave2/<axis>.roles.md`
before results are computed; probe sets land in `wave2/<axis>.txt`.
Runner: `run_wave2.py`; analysis: `analyze_wave2.py`; results:
`RESULTS-WAVE2.md`.
