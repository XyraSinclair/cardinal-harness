# Rail fitness: gemini-cli subscription rail vs local judges (2026-08-15/16)

Question: does `--model gemini-cli/<model>` — structured judgements shelled
through Google's gemini CLI under a Google AI Pro OAuth session, with the
lean-harness trim (README § gemini-cli rail) — behave like a real judge, or
does the subscription rail add noise?

## Design

Same shape as the local `manifund-relentless-2026-08-15` sweeps: 40 Manifund
proposals (`data/manifund.txt`) x 32 attributes
(`batteries/manifund_attributes.txt`), canonical_v2, seed 1, budget 240 per
attribute, landed to `ratiometer.judgments` under
`manifund-gemini-flash-2026-08-15`. Judge: `gemini-cli/gemini-2.5-flash`
(served `gemini-3.5-flash` per the CLI's model table). Metric:
`scripts/cross_judge_agreement.py` — per-attribute direction agreement over
co-judged decisive pairs, cross-run via comma-separated run_tags.

## Result: the rail is a real third judge

7,390 judgements, 32/32 attributes, **0 refusals, 0 errors, $0 marginal
cost**, ~4.1M input tokens (median ~570/judgement under the lean harness;
the stock CLI harness would have been ~7,600/judgement).

| judge pair | mean direction agreement (32 attrs) |
|---|---|
| gemini-flash vs gemma4-31b | **0.781** |
| gemma4-26b-a4b vs gemma4-31b (local baseline) | 0.776 |
| gemini-flash vs gemma4-26b-a4b | 0.726 |

Cross-family agreement (gemini vs 31b) lands *at* the within-family local
baseline — the subscription rail loses nothing. The weakest pair is
gemini vs a4b, consistent with a4b (the smallest judge) being the odd one
out rather than the rail being noisy.

Attribute quality replicates across families: the same attributes floor
every pairing — `room for more funding in this exact niche` (0.45–0.60),
`counterfactual impact of a marginal dollar of funding` (0.46–0.67),
`probability of a net-negative outcome`, `replaceability by work others
would do anyway`. These are construct problems, not judge problems:
rephrase candidates. Well-posed attributes (`empirical testability`,
`quality of quantitative reasoning`, `potential to become financially
self-sustaining`) clear 0.81+ against a judge from a different model
family, different serving stack, different harness.

## Operational scar: concurrency 2, not 8

At the default 8-wide comparison concurrency the subscription's per-minute
limit turns each burst into a multi-minute backoff stall: per-cell
wall-clock degraded 196s → 552s → 1235s → 2195s over the first four
attributes (completion histograms: 32-judgement bursts separated by 6–13
min of silence, while a lone probe returned in 2s). At `--concurrency 2`
(flag added for this run) cells hold a steady ~700s median — ~28
judgements/min, 4x the effective throughput of 8-wide, and the politer
traffic shape for a subscription account. Full battery: ~5.6h wall-clock.

Raw tables: `agree-gemini-31b.txt`, `agree-gemini-a4b.txt`,
`agree-baseline.txt` (this directory).
