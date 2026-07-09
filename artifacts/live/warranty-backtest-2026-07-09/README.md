# Warranty Pricing Backtest - 2026-07-09

This receipt is the disk-only backtest requested by GitHub issue #50's
red-team follow-up. It uses only checked-in SQLite caches and makes no network
or API calls.

## Pre-Committed Method

The backtest prices one-unit reversal warranties from held-out pairwise
judgment draws, then adjudicates against draws not used for pricing.

Two data paths are probed:

1. `judge-bench-2026-07-05/bench-cache.sqlite`: within-run counterbalance
   split. The pricing side is every usable draw where the canonical first
   entity is in slot A; the claim side is every usable draw where that same
   entity is in slot B. This prices presentation-order noise, not session
   noise.
2. `corpus-map-500-2026-07-08/map-cache.sqlite`: repeat-draw split. Eligible
   warranties require at least four usable draws for the same
   `(model, attribute, unordered pair)`. Rows are ordered by
   `(created_at, key_hash)`; the first half prices the warranty and the second
   half adjudicates it. In this cache the supported k>=4 subset tests
   attribute-prompt/repeat noise, with presentation orders present inside the
   halves when the rows supply them.

For every usable draw, the signed log ratio follows the repo reflection rule:
positive means the canonical first entity in the unordered pair is judged
higher, negative means the canonical second entity is judged higher.

For each analysis group, pooled within-pair variance is

`sigma_w^2 = sum_p sum_t (x_pt - mean_p)^2 / sum_p (k_p - 1)`.

When the eligible pair graph has residual cycle degrees of freedom, the script
also estimates the DerSimonian-Laird heterogeneity floor across pair means:

`sigma_b^2 = max(0, (Q - df) / c)`.

If `sigma_b` is unidentifiable, the primary price uses no heterogeneity floor
and the output includes a sensitivity range with `sigma_b` set to
`0`, `0.5 * sigma_w`, `1.0 * sigma_w`, and `2.0 * sigma_w`.

For a warranted pair:

`P(reversal) = Phi(-abs(Delta_hat) / sqrt(sigma_b^2 + sigma_w^2/k_claim + se(Delta_hat)^2))`

where `Delta_hat` is the pricing-half mean, `se(Delta_hat)^2 =
sigma_w^2/k_price`, and `k_claim` is the number of held-out adjudication
draws. Premium is `P(reversal)` times a unit payout.

Two adjudication events are reported:

- Raw flip: the claim-half pooled mean has the opposite sign from the pricing
  mean.
- Certified flip: the claim-half pooled mean has the opposite sign and is more
  than `2 * sigma_w / sqrt(k_claim)` from zero. This is the actual payout
  trigger under the red-team correction.

Calibration is evaluated with equal-count predicted-probability deciles. For
both raw and certified events, the script reports:

- realized event frequency by decile,
- Brier score,
- Brier score for a base-rate-only predictor,
- weighted least-squares reliability slope of decile observed frequency on
  decile mean predicted probability.

Pre-committed loss criteria:

- The model loses an event calibration if its Brier score is not strictly
  lower than the base-rate-only Brier score.
- The model loses an event calibration if its reliability slope is outside
  `[0.5, 1.5]`, or if the slope is unidentifiable.
- The primary issue #50 gate is the certified flip, because that is the actual
  trigger. Raw flips are still reported to expose the zero-crossing formula's
  behavior.

## Results

Run:

```bash
python3 artifacts/live/warranty-backtest-2026-07-09/analysis.py
```

The script writes `results.json` with per-pair records, variance components,
decile reliability tables, exclusions, and sensitivity prices.

### Ideal Issue #50 Backtest

The originally proposed judge-bench v1 -> retest claims stream is blocked on
disk. The retest pack exists at
`artifacts/live/judge-bench-retest-2026-07-05/`, but it contains only:

- `README.md`
- `comparison.txt`
- `leaderboard.txt`
- `reports.jsonl`
- `run.stderr`

There is no pair-level SQLite cache and no per-pair held-out claim stream.
That missing stream is the one that would test session/test-retest noise under
the leaderboard protocol. Recreating it is estimated at about `$0.46`.

### Primary Path: Map Repeat Split

Selected because `corpus-map-500-2026-07-08/map-cache.sqlite` has a k>=4
repeat subset.

Noise class priced: attribute-prompt/repeat-draw noise on the k>=4 subset.
This is closer to the DerSimonian-Laird repeat substrate than pure order
counterbalancing, but it is still not the missing v1 -> retest session stream.

Denominators:

- input rows: 10,762
- usable draws: 10,751
- input exclusions: 11 refused rows
- unordered `(model, attribute, pair)` groups: 5,308
- k>=4 groups: 72
- excluded from warranty pricing: 5,236 with fewer than 4 draws; 4 with zero
  pricing-half mean
- pairs priced: 68
- pricing draws: 136
- claim draws: 136
- all draws on priced pairs: 272

DL floor status: unidentifiable in both model groups. The eligible k>=4 graph
has zero cycle degrees of freedom, so the primary price uses `sigma_b = 0` and
the sensitivity range in `results.json` varies unidentifiable `sigma_b` from
`0` to `2 * sigma_w`.

Collected premium and losses, unit payout:

| event | claims | collected premium | loss ratio |
| --- | ---: | ---: | ---: |
| raw flip | 30 | 12.390 | 2.421 |
| certified flip | 2 | 12.390 | 0.161 |

Calibration:

| event | Brier | base-rate Brier | reliability slope | verdict |
| --- | ---: | ---: | ---: | --- |
| raw flip | 0.3265 | 0.2465 | 0.182 | LOSES |
| certified flip | 0.0805 | 0.0285 | -0.151 | LOSES |

Primary issue #50 gate: **LOSES**. The certified trigger is the actual payout
trigger, and it fails both pre-committed calibration tests: Brier is worse than
base-rate-only and reliability slope is outside `[0.5, 1.5]`.

Primary reliability table:

| decile | n | mean predicted P | raw freq | certified freq |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 0.009 | 0.333 | 0.000 |
| 2 | 7 | 0.019 | 0.286 | 0.143 |
| 3 | 7 | 0.029 | 0.429 | 0.000 |
| 4 | 7 | 0.066 | 0.429 | 0.000 |
| 5 | 7 | 0.132 | 0.571 | 0.143 |
| 6 | 6 | 0.213 | 0.333 | 0.000 |
| 7 | 7 | 0.249 | 0.571 | 0.000 |
| 8 | 7 | 0.306 | 0.714 | 0.000 |
| 9 | 7 | 0.346 | 0.429 | 0.000 |
| 10 | 7 | 0.433 | 0.286 | 0.000 |

Sensitivity when unidentifiable `sigma_b` is varied:

| sigma_b multiplier | premium | raw loss ratio | certified loss ratio |
| ---: | ---: | ---: | ---: |
| 0.0 | 12.390 | 2.421 | 0.161 |
| 0.5 | 13.583 | 2.209 | 0.147 |
| 1.0 | 16.233 | 1.848 | 0.123 |
| 2.0 | 21.354 | 1.405 | 0.094 |

### Supplemental Path: Judge-Bench Order Split

Noise class priced: within-run presentation-order noise only. This does not
price session noise.

Denominators:

- input rows: 684, not the 828 rows expected in the issue note
- usable draws: 683
- input exclusions: 1 refused row
- unordered `(model, attribute, pair)` groups: 149
- groups with both orders: 126
- excluded from warranty pricing: 23 missing the counter order
- pairs priced: 126
- pricing draws: 450
- claim draws: 210
- all draws on priced pairs: 660

Collected premium and losses, unit payout:

| event | claims | collected premium | loss ratio |
| --- | ---: | ---: | ---: |
| raw flip | 29 | 41.343 | 0.701 |
| certified flip | 0 | 41.343 | 0.000 |

Calibration:

| event | Brier | base-rate Brier | reliability slope | verdict |
| --- | ---: | ---: | ---: | --- |
| raw flip | 0.1814 | 0.1772 | 0.627 | LOSES |
| certified flip | 0.1264 | 0.0000 | 0.000 | LOSES |

Order-split reliability table:

| decile | n | mean predicted P | raw freq | certified freq |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 0.047 | 0.083 | 0.000 |
| 2 | 13 | 0.146 | 0.154 | 0.000 |
| 3 | 12 | 0.233 | 0.083 | 0.000 |
| 4 | 13 | 0.303 | 0.154 | 0.000 |
| 5 | 13 | 0.349 | 0.308 | 0.000 |
| 6 | 12 | 0.397 | 0.167 | 0.000 |
| 7 | 13 | 0.420 | 0.385 | 0.000 |
| 8 | 12 | 0.437 | 0.250 | 0.000 |
| 9 | 13 | 0.454 | 0.385 | 0.000 |
| 10 | 13 | 0.480 | 0.308 | 0.000 |

## Interpretation

The pricing formula can lose against checked-in claims history, and it does
lose here under the pre-committed calibration gates.

The raw zero-crossing event and the certified payout event behave differently.
On the primary repeat split, raw flips are underpriced in loss-ratio terms
(`2.421`), while certified flips are overcollected (`0.161`) but still badly
miscalibrated. That is exactly the red-team warning: pricing a raw crossing is
not the same as pricing the actual 2se-certified trigger.

The remaining blocker for the exact issue #50 test is data, not code: the
v1 -> retest pair-level stream is absent from disk. Creating that stream would
test the session-noise class that neither substitute path fully prices.
