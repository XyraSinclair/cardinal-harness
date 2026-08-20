# JCB scaled-battery live smoke — 2026-08-19

First live run of the battery-as-data machinery (commit a66758e): pool →
seeded generation → full JCB axes on real entities. ONE model, ONE seed —
a machinery smoke, not a board. Do not quote these numbers as a ranking.

- Battery: `anchors-country-population-v1/n16/s1` (spec: `battery.json`,
  regenerable from the pool + seed 1) — 16 countries, 56 core pairs,
  480 calls.
- Model: `openai/gpt-5.4-mini`, template `canonical_v2`, no cache.
- Cost: $0.1825 · 480 comparisons · 0 refusals.
- Raw per-call records: `reports.jsonl`.

## Board line

JUDGE 0.580 · signal 1.366 nats · coherence 0.779 (harmonic 0.519) ·
order-flip 0.250 [0.155, 0.377] (n=56) · curl 0.109 (n=41 cycles) ·
spin 1.000 (n=4) · polarity ρ −0.697 · truth-slope 0.658 (n=56).

## Observations (single-model, single-seed — hypotheses, not findings)

1. **The meaningful-entity tier is harder than the v1 aphorisms.**
   Same model class on v1.2 sat near order-flip ~0.05 and higher orbit
   coherence; here order-flip is 0.250 and orbit coherence 0.130 with
   interaction share 0.530. If this holds across models, the public tiers
   discriminate better than the demo corpus — the property a leaderboard
   needs.
2. **truth-slope 0.658**: judged log-ratios are ~2/3 of true population
   log-ratios — direction is right, magnitudes compressed. This is the
   max-ratio-inflation sidebar doing its job on first contact; it is
   reported, never scored.
3. **null-bias exactly 0.000 (n=8)** replicates the identical-null floor
   effect already flagged in BENCHMARK.md's open attacks (byte-equal
   pairs are free to pass); near-identical null pairs remain v2 work.
4. Nuisance drift is dominated by `bullet` (0.855 nats, n=17) and `halo`
   (0.648) — format edits move this judge on real entities far more than
   on aphorisms.

Next: multi-model run over the three pools (spend gate with Xyra), then
the pairwiseratio.org refresh.
