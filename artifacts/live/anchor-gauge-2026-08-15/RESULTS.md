# Grok gauge calibration on anchors (E2)

**Errata:** (1) The WST/MST/SST transitivity criterion in the PROGRAM.md band
definition is UNMEASURED — this protocol has no repeat draws; bands here use
the four measured criteria (order, curl, polarity, paraphrase). (2) The first
run of this harness was interrupted mid-flight; everything was re-run cleanly
in one pass (this pack), nothing hand-stitched. (3) Cities truth uses UN
urban-agglomeration figures; "metro area" diverges for London/Paris/Chicago/
Vienna. Amazon length contested (6,400 km used).

Protocol: 3 anchor pools × 16 entities (countries/population, rivers/length,
cities/metro population), canonical_v2 pairwise ratio, budget 4n
counterbalanced + 2n paraphrase + 2n negation, models gpt-4.1-mini and
gpt-5.4-nano. 768/768 calls parsed, 0 refused, 0 errors, $0.107 (cap $5).
Offline synthetic judge recovered planted truth and flipped sign under
negation before any live call. Instrument: `examples/anchor_gauge.rs`.

| pool · model | order agr (pairs) | curl hcr | polarity ρ | paraphrase ρ | truth ρ | truth slope | band |
|---|---|---|---|---|---|---|---|
| countries · mini | 0.75 (32) | 0.039 | −0.53 | 0.84 | **0.965** | 0.65 | partial 2/4 |
| cities · mini | 0.69 (32) | 0.060 | −0.35 | 0.93 | 0.832 | 0.68 | partial 2/4 |
| rivers · mini | 0.63 (32) | 0.129 | −0.81 | 0.93 | 0.350 | 0.03 | partial 2/4 |
| countries · nano | 0.55 (31) | 0.170 | +0.58 | 0.80 | 0.809 | 0.55 | not 0/4 |
| cities · nano | 0.63 (32) | 0.193 | +0.16 | 0.44 | 0.453 | 0.15 | not 0/4 |
| rivers · nano | 0.41 (32) | 0.228 | +0.01 | 0.27 | 0.106 | −0.01 | not 0/4 |

**What the truth axis says about the provisional bands.** The bands separate
models (mini partial everywhere, nano not everywhere — matching truth ρ
ordering in every pool) but not pools within mini: countries (truth ρ 0.965)
and rivers (0.350) both land "partial 2/4". Individual readouts DO
discriminate: curl 0.039 vs 0.129, order 0.75 vs 0.63 — the composite loses
what the components see. **No cell reached "grokked", including one with
truth ρ 0.965**: order ≥ 0.90 and polarity ≤ −0.80 are too strict at this
budget (64 comparisons, most pairs asked once per orientation — agreement
carries single-draw noise). Magnitude calibration: truth slopes 0.55–0.68
where rank is good — judges compress true log-ratios ~⅓, everywhere.

**Proposed (not adopted) with this evidence:** gate bands on curl + order
jointly (curl ≤ 0.10 AND order ≥ 0.70 separates every truth-good cell from
every truth-bad cell here, 6/6); polarity as a reported diagnostic, not a
gate (negation wordings measure their own construct: truth ρ under negation
was weak even for mini). One run, two models, three pools — evidence to
carry into the next calibration round, not a retuning.
