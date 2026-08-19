# Live A/B round 3 — radix4 validity cell: FAILS; peel wins the program (2026-08-13)

Last challenger from TOURNAMENT.md. `radix4_probe.py`: 3-digit base-4 code
over signed log10 ratio in [-3,3] (64 leaves), each digit a 4-way enum —
even Azure's dynamic sidebands return the full conditional per visited
node, so the deep-grammar channel itself works. 144 calls, $0.13, same
axes as rounds 1-2: relabel (`asc` vs digit-reversed `desc`), mirror,
depth information, cross-instrument. Raw: `radix4_results.json`;
per-cell: `radix4_summary.json`.

## Results (E[Z] by axis; truth in last column)

| model | pair | E asc | E desc | mirror gap | H(d2) bits | truth |
|---|---|---|---|---|---|---|
| 5.6-sol | egg | −2.22 | −1.97 | −0.42 | 0.96 | −1.98 |
| 5.6-sol | cat-raccoon | −0.21 | −0.45 | +0.01 | 0.42 | −0.18 |
| 5.6-sol | whale-elephant | +1.01 | +1.41 | +0.05 | 0.51 | +1.40 |
| 5.6-sol | sweden-portugal | +0.08 | +0.05 | +0.41 | 0.19 | +0.01 |
| 5.4-mini | egg | −1.34 | −0.76 | +1.03 | 1.15 | −1.98 |
| 5.4-mini | cat-raccoon | −0.46 | −0.07 | +0.23 | 1.68 | −0.18 |
| 5.4-mini | whale-elephant | +2.53 | +2.67 | +1.12 | 1.71 | +1.40 |
| 5.4-mini | sweden-portugal | −0.16 | −0.36 | −0.68 | 1.75 | +0.01 |
| 4.1-mini | egg | −0.25 | −1.53 | **+2.50** | 0.33 | −1.98 |
| 4.1-mini | cat-raccoon | −0.33 | −0.22 | +0.36 | 1.09 | −0.18 |
| 4.1-mini | whale-elephant | +1.04 | +1.30 | +0.89 | 1.56 | +1.40 |
| 4.1-mini | sweden-portugal | +0.58 | +0.12 | +1.32 | 1.20 | +0.01 |

## Findings

1. **Worst geometry of the three.** Digit-reversal relabeling moves E[Z] by
   up to 1.28 log10 (4.1-mini egg: −0.25 vs −1.53); mirror gaps reach 2.50.
   Errors vs truth exceed a full decade in several cells (5.4-mini whale
   +2.53 vs +1.40; 4.1-mini egg −0.25 vs −1.98).

2. **The deep digits are prior noise on flat models.** d2 conditional
   entropy runs 1.1–1.75 bits on the minis (uniform = 2): "quarter within
   the range on the log scale" collapses to a digit prior rather than a
   refined percept — and worse, the noisy deep digits CORRUPT the moment
   estimate that d1 alone would have supported.

3. **Even sol loses its invariance.** grid16-sol was exactly invariant;
   radix4-sol wobbles 0.25–0.40 under relabeling and mirroring. The
   hierarchical abstract code is hard even for the strong model at
   effort none.

4. **Digits are not enough — semantics must be native.** radix4's code
   points are numerals, but their MEANING (positional quarters of a log
   scale) is an artificial mapping requiring in-head log arithmetic the
   effort-none pass does not perform. The decimal ratio grammar works
   because its string IS the quantity: "12.5" means 12.5. Nativeness is a
   property of the code-to-measurand mapping, not of the glyph set.

## Program verdict — the live A/B is decided

Three synthetic challengers, three live validity failures, in order of
synthetic promise:

| arm | synthetic frontier | live validity |
|---|---|---|
| bin-lp | 56 (winner) | FAILED — step collapse, threshold anchoring, round-number attractors (BINCDF.md) |
| grid16 | 56 | FAILED flat / empty peaked — letter priors, slot bias, compression (GRID16.md) |
| radix4 | 56 | FAILED — worst of all; abstract positional semantics collapse to digit priors (this note) |
| **peel** | 117 | **holds** — coverage 1.00 (SHOOTOUT.md), best live location fidelity in every cross-check |

The synthetic tournament measured channel capacity; the live rounds
measured validity; only the native-numeral decimal grammar has both. The
law, now supported by three independent failure modes: **an elicitation
geometry is valid iff its code-to-measurand mapping is one the model
already uses natively; every artificial mapping measured its own artifact**
(thresholds → anchors, letters → slot/letter priors, positional digits →
prior collapse).

Consequently: engine work proceeds on peel's evidence path (production
`decimal_ledger` + credal certificate, F5), with counterbalanced
presentation retained (slot bias is real and large). The staircase
survives only as a cheap deterministic direction-finder where a single
bit suffices. Total live-A/B spend: $0.44, one day, question closed.

## Caveats

Same scope as rounds 1-2 (one day, Azure route, 4 pairs, 3 models,
3 reps). The radix4 prompt is one phrasing of the hierarchical semantics;
a better-taught deep code (few-shot, worked examples) might do better —
but at that point the instrument's cost and fragility already concede the
comparison to peel, whose grammar needs no teaching.

Reproduce: `python3 radix4_probe.py run` then `... analyze`.
