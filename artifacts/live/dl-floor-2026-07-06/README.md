# The repeat-elicitation stack, live end-to-end (2026-07-06)

`cargo run --example dl_floor_live -- deepseek/deepseek-v4-flash 6`:
nonce draws (6 per pair, temperature 0) on a 7-pair cycle-bearing graph
over 5 corpus items → DerSimonian–Laird pooling → the
structural/contextual variance split and the floored solve. 42
judgements, $0.0043, 29% of input tokens billed cached.

| quantity | value | reading |
|---|---|---|
| σ_w (contextual, per draw) | 0.195 nats | one irrelevant suffix token moves a deepseek judgement ~0.2 nats on these pairs |
| σ_b (structural floor) | 0.030 nats | the graph explains the pair means almost perfectly |
| Q on df | 3.70 on 3 | ≈ 1.2 per df: textbook homogeneity |
| floored vs naive scores | identical to 2dp | the no-phantom-floor property, holding on live data |

The null result is the point: on pairs the judge is structurally
consistent about, the DL floor changes nothing — it only bites when
frustration is real (pinned with planted cases in
tests/repeat_pooling.rs). An estimator that alters clean data would be
worse than no estimator.

Operational notes: per-pair refusal/parse losses of 0–3 of 6 draws on
short prompts (deepseek; the long-prompt run in nonce-draws-2026-07-06
had 0/8) — refusal rates are prompt-length-dependent and now visible per
pair; cache hit rate 29% here vs 95% on the long-prompt pack — prefix
caching pays proportionally to prefix length, exactly as priced.
