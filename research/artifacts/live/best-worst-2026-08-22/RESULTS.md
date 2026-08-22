# E6 — best–worst vs listwise vs pairwise, DeepSeek V4 Flash (k-wise · ordinal · point)

**Errata:** none yet.

Question (the consumer's): reranking ~24 items under a custom user prompt
where an adequate quality adjustment — not a certified order — is the bar,
which instrument gives adequate agreement with the canonical pairwise sort
at the lowest dollars per item? Instrument: `experiments/examples/setwise_cached.rs`
`--answer {bw,order}` (design: PROGRAM.md E6; climbed 2026-08-22).

Setup. `deepseek/deepseek-v4-flash` via OpenRouter, pinned to the five
providers of the manifund-deepseek lane, reasoning disabled, no logprobs.
n = 24 Manifund items (1600 chars each), k = 8, three attributes — two
rubric files (`impact_per_dollar`, `theory_of_change`) and one plain
user-prompt string ("fit for a funder who wants cheap high-leverage AI safety
field-building"). Chunk design: `--presentations` m rounds of seeded shuffle
→ 3 groups of 8; m = 3 ⇒ 9 calls per attribute, m = 6 ⇒ 18. Pools: seed 17
(m = 3 and m = 6, both modes) and seed 23 (m = 3, both modes; a different
24-item pool). Baseline each run: canonical_v2 `sort_documents`, default
budget = 96 comparisons per attribute. Six live runs, 54 + 108 + 54 = 216
setwise calls + 18 pairwise sorts, **$0.2735 total** (cap $1/run). Every
setwise call parsed (0 malformed, 0 refused, 0 errors); every observation
graph connected (1 component). Offline synthetic-judge dry run first
(`offline/`): `order` m = 3 recovered planted truth ρ 0.93–0.96 vs pairwise
0.93–0.96; `bw` 0.72–0.88; the untouched `ratio` arm reproduces.

## Agreement with the pairwise sort (Spearman ρ over 24 items)

| run | impact_per_dollar | theory_of_change | user prompt | $/item (setwise) | $/item (pairwise, 96 cmp) |
|---|---|---|---|---|---|
| order m=3 s17 | 0.84 | 0.78 | 0.85 | 1.6e-4 / 1.2e-4 / 1.1e-4 | 5.8e-4 / 5.1e-4 / 4.4e-4 |
| order m=6 s17 | 0.85 | 0.76 | 0.92 | 3.2e-4 / 2.3e-4 / 1.8e-4 | 5.4e-4 / 5.0e-4 / 4.3e-4 |
| order m=3 s23 | 0.64 | 0.73 | 0.80 | 1.7e-4 / 1.2e-4 / 0.9e-4 | 5.8e-4 / 5.1e-4 / 4.4e-4 |
| bw m=3 s17 | 0.46 | 0.41 | −0.08 | 1.6e-4 / 1.1e-4 / 1.1e-4 | 4.6e-4 / 4.5e-4 / 4.1e-4 |
| bw m=6 s17 | 0.32 | 0.76 | −0.18 | 3.2e-4 / 2.0e-4 / 1.8e-4 | 4.7e-4 / 4.6e-4 / 4.0e-4 |
| bw m=3 s23 | 0.12 | −0.02 | 0.19 | 1.7e-4 / 1.0e-4 / 0.9e-4 | 4.6e-4 / 4.4e-4 / 4.0e-4 |

Denominator for "adequate": the pairwise sort's own test–retest across the
same-seed runs (independent calls, same pool and prompt) is ρ 0.90 / 0.91 /
0.91 (seed 17, 6 pairs each; min 0.83) and 0.84 / 0.94 / 0.83 (seed 23, one
pair). `order` test–retest m = 3 vs m = 6 on the same pool: 0.88 / 0.90 /
0.94 — as stable as pairwise. `bw` m = 3 vs m = 6: 0.85 / 0.53 / 0.46.
`order` vs `bw` in the same run (m = 6): 0.20 / 0.72 / 0.04 — they are not
measuring the same thing on two of three attributes.

Cost. The first attribute of each run pays the prefix once more (no provider
cache evidence on this lane); later attributes ≈ 1.0e-4 $/item for 9 calls
(input-dominated: ~3.0k tokens in, 9 tokens out per `order` call, 3 per
`bw`). The pairwise arm at 96 comparisons ≈ 4.4e-4 $/item. So `order` at
m = 3 costs **~¼** of the pairwise sort and lands at or just under the
pairwise test–retest ceiling on two attributes (0.84 vs 0.90; 0.85 vs 0.91)
and below it on one (0.78 vs 0.91; seed-23 pool 0.64–0.80 vs 0.83–0.94).
`bw` at the same price is not an adequate instrument here.

## Position bias (measured, pooled over the three runs per mode, 108 calls, 13.5 expected per slot)

- `order` first-rank picks by slot A..H: 3, 9, 13, 16, 24, 15, 19, 9 —
  slots A–B under-picked (12/27 expected); last-rank picks: 7, 5, 12, 7, 13,
  20, 14, **30** — the last slot is ranked last 2.2× its fair share.
- `bw` best picks: 8, 6, 5, 20, 15, 22, 19, 13 — the first three slots
  under-picked (19/40.5 expected); worst picks: 8, 16, 6, 9, 14, 10, 19,
  **26** — last-slot again 1.9×.

A primacy-disfavour + recency-worst shape on both. Randomized slot order per
call keeps it from becoming an item effect, but it costs precision; a
slot-offset term or the paired presentation this design deleted would be
the fix if a later regime needs it.

## Reading

1. For the consumer's regime (adequate adjustment, custom prompt, ~24
   items), **plain listwise at k = 8 lowered into the solver is the
   instrument**: 9 calls, ~¼ the cost of the 96-comparison pairwise sort,
   agreement with pairwise at or near pairwise's own reliability, and
   test–retest as good as pairwise. The E5/E6 question "is a new best–worst
   instrument needed" answers **no** in this regime — the listwise arm that
   the climb kept as the efficiency denominator is the winner.
2. Best–worst — the "highest-value missing cell" of the instrument grid
   (docs/FIRST_PRINCIPLES.md §2) — is refuted as built: 13 observations per
   call vs 28 for `order` at the same input cost, and the worst pick is a
   weak, biased signal. The grid entry should carry this pack, not the
   prior.
3. Caveats, honestly: one model, one corpus family, n = 24, k = 8, m ≤ 6;
   the pairwise baseline is itself at ρ ≈ 0.9 reliability so agreement above
   that is not measurable here; no logprob/PMF arm (deleted in the climb —
   E7's question); the user-prompt attribute is one string.

Replay: `report.json` + `trace.jsonl` (+ `pairwise_trace.jsonl`) per run
under `live/<mode>-m<m>[-s23]/`; offline packs under `offline/`. Runner
script mirrored in `live/run.sh`.
