# Live A/B round 2 — grid16 single-call PMF + codebook calibration (2026-08-13)

TOURNAMENT.md's efficiency ceiling, taken live with the calibration check it
was required to ship with. `grid16_probe.py`: 16-bin signed-log codebook
(letters C..R, bins 0.375 log10 wide over [-3,3]), byte-identical 16-enum
grammar, 3 models × 4 pairs × 2 codebook variants (ascending/descending
letter maps — semantically identical instruments) × 2 presentation orders ×
3 reps = 144 calls, $0.13. Raw: `grid16_results.json`; per-cell:
`grid16_summary.json`.

## Channel finding first

Azure gpt-5.x returns DYNAMIC sidebands — 3–8 entries regardless of
requested `top_logprobs` (5/8/20 all give the same), covering ~0.985–0.999
of legal mass; gpt-4.1-mini honors 20 → true full PMF. So grid16's
one-call-full-PMF premise survives in bounded-residual form: enumerated
mass 0.986–1.000 in every cell. The channel works. The measurement is
where it breaks.

## Results (E[Z] under asc/desc codebooks; TVcode = relabeling distance)

| model | pair | E asc/desc | TVcode | mirror gap (E_AB vs −E_BA) | truth |
|---|---|---|---|---|---|
| 5.6-sol | egg-vs-bowling-ball | −2.06/−2.06 | 0.000 | 0.00 | −1.98 |
| 5.6-sol | cat-vs-raccoon | −0.19/−0.19 | 0.000 | 0.00 | −0.18 |
| 5.6-sol | whale-vs-elephant | +1.31/+1.31 | 0.000 | 0.07 | +1.40 |
| 5.6-sol | sweden-vs-portugal | +0.19/+0.19 | 0.000 | 0.00 | +0.01 |
| 5.4-mini | egg-vs-bowling-ball | −1.43/−1.30 | 0.384 | 0.17/0.39 | −1.98 |
| 5.4-mini | cat-vs-raccoon | −0.17/−0.15 | 0.047 | **sign flip** (−0.17 vs +0.17) | −0.18 |
| 5.4-mini | whale-vs-elephant | +0.29/+0.18 | 0.203 | 0.64/0.91 | +1.40 |
| 5.4-mini | sweden-vs-portugal | −0.18/−0.11 | 0.177 | **sign flip** (−0.18 vs +0.12) | +0.01 |
| 4.1-mini | egg-vs-bowling-ball | −1.52/−1.49 | 0.142 | 0.49/0.22 | −1.98 |
| 4.1-mini | cat-vs-raccoon | −0.27/−0.40 | 0.345 | 0.08/0.21 | −0.18 |
| 4.1-mini | whale-vs-elephant | +1.09/+0.96 | 0.554 | 0.52/0.26 | +1.40 |
| 4.1-mini | sweden-vs-portugal | +0.06/−0.13 | 0.515 | 0.69/0.06 | +0.01 |

## Findings

1. **On the peaked model (sol), grid16 is perfect and empty.** TVcode = 0,
   mirror-exact, point mass (sd = 0.00) in the truth-containing bin in all
   four cells, deterministic across reps. A superb one-call point reader at
   bin resolution (±0.19 log10) — carrying zero width information, because
   sol's effort-none percept is itself deterministic. In this regime peel
   already enumerates 0.98 mass in one call (RESULTS.md); grid16 buys
   nothing peel doesn't.

2. **On flat models — the regime where geometry choice matters — every
   invariance axis fails.**
   - *Codebook relabeling* shifts the measured distribution: TV up to 0.554
     (4.1-mini whale). Letter identity carries its own mass structure —
     the synthetic tournament's "codebook edge shift" distortion is real,
     and it is not a small edge effect.
   - *Slot bias flips near-tie directions*: 5.4-mini judges cat-vs-raccoon
     at E = −0.17 in AB order and +0.17 mirrored — "the second entity is
     bigger" in BOTH orders; sweden-vs-portugal the same. Counterbalancing
     cancels the bias but leaves ≈ 0 net signal: near-tie resolution on
     flat models sits BELOW the slot-bias floor for this geometry.
   - *Presentation gaps* on clear pairs reach 0.64–0.91 log10 (5.4-mini
     whale) — order effects the size of the quantity being measured.
   - *Compression toward Z = 0*: whale +0.29..+1.09 vs truth +1.40; egg
     −1.3..−1.5 vs −1.98. The codebook's coarse outer bins plus letter
     priors drag flat-model mass centerward.
   - *Rep drift* up to ~1.0 relative on tail letters (4.1-mini whale) —
     the PMF tails are not stable objects call-to-call.

3. **Cross-instrument location ranking on the flat-model hard cell (egg,
   truth −1.98)**: decimal peel −1.85/−2.06 (err 0.13/0.08) ≪ grid16
   −1.30..−1.52 (err 0.5–0.7) < binary staircase −2.65/−3.24 (err 0.7–1.3).
   Peel remains the most location-faithful instrument on flat models, by a
   wide margin, in both live rounds.

## The emerging law

Two rounds of live validity probes now point the same way: **geometries
that reuse the model's native representation of magnitude (decimal
numerals) inherit its calibration; artificial codebooks measure their own
artifact.** The binary staircase imported threshold anchoring and
round-number attractors (BINCDF.md); the letter grid imports letter priors,
slot bias, and centerward compression. The decimal grammar's code points
ARE numbers — the surface form and the measurand coincide, and it is the
only geometry whose live PMF has tracked the latent in every test so far
(SHOOTOUT.md coverage 1.00, HARVEST, this round's cross-checks).

## Verdict

- grid16 as primary challenger: **rejected on flat models; redundant on
  peaked ones.** Single-call codebook reads are usable only where the
  model is peaked — where they add nothing over one call of peel.
- Remaining live question for round 3: **radix4** — its code points are
  digits (closer to native numerals than letters). One validity cell
  (relabel + mirror + truth) decides whether deep-grammar decimal peel
  simply wins the program outright.
- The synthetic tournament's transfer condition (decoder PMF faithful to
  latent judgement) is now measured: it holds for the decimal grammar,
  fails for letters and thresholds. Engine work should proceed on peel's
  evidence path.

## Caveats

Same scope limits as BINCDF.md: one day, Azure route, 4 pairs, 3 models,
3 reps. The sol-perfection and mini-contamination patterns replicate
across all cells and both variants; exact TV/E numbers are point
measurements. The whale/sweden pairs lack harvest decimal-instrument
references (harvest covered 2 pairs); the egg/cat comparisons carry the
cross-instrument column.

Reproduce: `python3 grid16_probe.py run` (144 calls, ~$0.13), then
`python3 grid16_probe.py analyze`.
