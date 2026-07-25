# Probe 1 results — scored against the frozen predictions

9 runs (3 axes × 3 models), 24 comparisons each, $0.58 total,
`analyze.py` over the `sort-*.json` evidence in this directory.

## Prediction scorecard

- **P1 (tier divergence ≥0.25) — SPLIT.**
  `ultrawattagedness`: frontier↔frontier ρ=+0.90; frontier↔mini ρ=+0.66/+0.64
  (gap 0.24–0.27) — the signature, right at threshold. `end_of_time`:
  FAILED — mini agrees with frontiers as well as they agree with each
  other (ρ≈0.80–0.87). Bare-phrase end_of_time is mostly TOPIC detection,
  and topic detection is cheap. (corpus_v3 already has `x_risk_futures`
  as a cheap topic axis — same lesson from the other direction.)
- **P2 (decoys rise ≥3 places under mini) — SPLIT.**
  `performed-4:47am` on ultrawattagedness: frontier rank 10, mini rank 6
  (+4) — textbook; the mini bought the performance, both frontiers dumped
  it to the bottom third. `cosmic-SLOP` on end_of_time: shift +0 — worse,
  gpt-5.6-sol itself ranked the slop 4th. Under a neutral phrasing even
  frontier models give cosmic vocabulary partial credit.
- **P3 (clarity control, all pairs ≥0.7) — PASS** (0.80–0.89). The
  divergence on ultrawattagedness is axis-specific, not model noise.
- **P4 (orthogonality |ρ|<0.4 vs clarity) — SPLIT.** opus −0.20 pass;
  gpt-5.6-sol −0.49 — deep items read as less clear to it. Watch this:
  anti-correlation with clarity is still rank information, but it means
  "profound = murky" is a live confounder in one frontier judge.

## What the misses teach (the real yield)

1. **`ultrawattagedness` is a genuine big-model-smell axis** — frontier
   consensus high, small-model decoherence concentrated exactly on the
   performed-intensity decoy. Admit to the next stage (phrasing family +
   invariance battery).
2. **`connection_to_the_end_of_time` under a bare phrasing is a topic
   axis, not a latent.** Its differentiated form has to be the
   slop-punishing version — which means the WORDING carries the axis.
   Next: `cardinal elaborate` variants that explicitly separate
   "genuinely bears on ultimate stakes" from "uses cosmic vocabulary",
   then re-probe; if only steered phrasings discriminate, the coherence
   evidence must come from the steered family, and the axis page must
   publish the exact wording as part of provenance (it already does —
   this is why axis_prompt_hash is in the schema).
3. **Decoy design is load-bearing.** The cosmic-SLOP item arguably HAS
   cosmic content (place-in-cosmos is literally its topic), so frontier
   models weren't strictly wrong. Decoys must be vocabulary-without-
   referent, not vocabulary-with-thin-referent. Build a small decoy
   library per axis family; decoys are reusable instruments.
4. Unplanned observation: `archive-seance` (poetic charge) — opus rank 2
   on ultrawattagedness, mini rank 7. Frontier models read compressed
   poetic force as wattage; minis don't see it. Second divergence site,
   for free — suggests a `compressed_charge` axis probing exactly this.

## Next probes

- Phrasing-family probe for end_of_time (3 wordings × 2 models) — does an
  elaborated rubric create the tier gap the bare phrase lacks?
- Screen the generated catalog (four-lens fan-out, see CATALOG.md):
  paper-kill, then decoy-planted micro-probes for the top ~8 by the same
  method. ~$0.60 per axis at this probe size.
