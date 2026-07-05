# Wording invariance, live (2026-07-05)

`cardinal judge --wordings`: the same pair judged through the three
wordings of one ratio question — "how many times more" (canonical_v2),
"what fraction does the lesser reach" (fraction_v1), "which has LESS and
how many times less" (less_v1) — each in both presentation orders, every
answer lowered to the same signed log-ratio by the slug-aware parser.
Pair: "The obstacle is the way." vs "Live, laugh, love.", criterion
"depth of insight about living well".

| Model | times-more | fraction | times-less | max disagreement | sign |
|---|---|---|---|---|---|
| claude-sonnet-4.6 | +1.131 | +1.642 | +1.246 | 0.510 | consistent |
| gemini-2.5-flash | +1.380 | +2.303 | +1.246 | **1.056** | consistent |
| gpt-5.4-mini | +1.773 | +2.120 | +2.229 | 0.456 | consistent |

## Findings

1. **Inversion works at the frontier**: no model flips sign when asked
   "which has less" — the failure mode the scripted inversion-blind judge
   pins in tests does not appear in these three.
2. **Numerical framing bias is real and directional**: the fraction
   wording elicits systematically LARGER separations than the ratio
   wording in all three models (+0.35 to +0.92 nats). gemini says "4×
   more" when asked multiplicatively and "reaches a tenth" (10×) when
   asked fractionally — a 2.5× multiplicative inconsistency from wording
   alone. This is the classic human ratio-bias (fractions feel smaller
   than the equivalent ratio), reproduced in machine judges.
3. Practical consequence for elicitation: magnitudes are
   wording-calibrated, not absolute. Mixing wordings in one solve without
   a per-template calibration term would inject systematic magnitude
   noise; directions are safe to mix.

Cost: $0.04 across three models (6 calls each). Raw JSON per model in
this directory.
