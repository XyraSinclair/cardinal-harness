# The first map: 120 corpus entities × 3 attributes × 2 minds (2026-07-06)

The pilot for "Elos for everything": 120 of the operator's own prompts
(stratified across the existing intellectual_ambition range, 40–350
words each), judged under three canonically-worded attributes by the two
top judges from the measured portfolio (deepseek-v4-flash,
gemini-2.5-flash), 960 counterbalanced comparisons per (attribute ×
judge) run. **5,760 judgments, $0.862 total**, $10 cap untouched, every
judgment in the replayable cache. n = 120, one run — a pilot, with the
denominators that word implies.

## The three questions, answered

**1. Do different minds recover the same ordering of real material?**

| attribute | transmissibility (cross-judge ρ) |
|---|---|
| intellectual ambition | **+0.911** |
| epistemic rigor | +0.811 |
| signal density | +0.755 |

Yes — at ρ 0.76–0.91 on the operator's own writing, not aphorisms.
"Intellectual ambition" is the most transmissible dimension measured to
date: two different labs' models nearly agree on what reaching for a
large problem looks like.

**2. Does the stack reproduce the operator's own historical annotations?**

| attribute | fused vs 2026 corpus scores | per-judge |
|---|---|---|
| intellectual ambition | **+0.934** | deepseek +0.891 · gemini +0.934 |
| signal density | +0.743 | +0.702 · +0.723 |
| epistemic rigor | +0.647 | +0.605 · +0.625 |

The full pipeline — pairwise ratio elicitation → counterbalanced O(n)
planning → IRLS solve → z-fused portfolio — recovers annotation scores
that existed before this repo did, at ρ up to 0.93, for under a dollar.
The fused map beats or ties the better single judge on every attribute:
portfolio fusion paying live. (Rigor's lower ρ is honest: the corpus's
epistemic_rigor was scored on a different rubric generation; ρ 0.65
against a years-old rubric is agreement, not noise.)

**3. Are the dimensions independent?**

| pair | fused ρ |
|---|---|
| epistemic rigor × intellectual ambition | **+0.003** |
| epistemic rigor × signal density | +0.402 |
| intellectual ambition × signal density | +0.732 |

Rigor and ambition are measured ORTHOGONAL on this corpus — two genuinely
independent axes of the operator's thought. Signal density is largely
(0.73) subsumed by ambition here; the canonical-attribute loop's
redundancy criterion, appearing at map scale.

Face validity, uncurated: ambition's top = "one of the most important
ideas in the world…", an inference question addressed to Toby Ord;
ambition's bottom = an eyeglasses SKU and a chicken-baking question.
Rigor's top = a versioning spec and hardware benchmarking; rigor's
bottom = venting at ChatGPT. The map reads true.

## Operational findings

- **Order-flip rate ~29% on long prompts** (vs 15–20% on short
  aphorisms): position bias grows with entity length; counterbalancing
  cancels it pair-by-pair, and the receipts show what it cancelled.
- Cost structure: ~$0.07–0.18 per (attribute × judge) run at n = 120 with
  960 comparisons. Extrapolation to the full 14K-entity corpus at the
  same design density: low hundreds of dollars per attribute-pair —
  "Elos for everything" is a budget line, not a moonshot.

## Verdict

The pilot clears its bar on all three questions. Scale-up path: full
corpus, attributes chosen and worded via `canonize`, judges re-weighted
per-attribute from the portfolio geometry, DL-floored repeats on
contested pairs, and the map published as sections (the #46 packet) —
each entity's latent carrying its receipts.
