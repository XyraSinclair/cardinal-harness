# Confidence Knob Receipt, 2026-07-09

## Pre-committed Verdict Rule

This rule was written before running `analysis.py` for this receipt.

Primary live-data exercise gate:

- If any model/template confidence distribution is degenerate, meaning more
  than 90% of non-refused observations place mass on one exact confidence
  value, the primary verdict is **knob unexercised on live data**. The
  histogram is the receipt. Steps 2-4 are still reported on whatever variation
  exists.

Confidence-map verdict rule:

- If the flat model is not rejected at `p < 0.05`, or the bootstrap confidence
  interval for `gamma` covers `0`, the verdict is **DELETE THE KNOB supported**.
- If the flat model is rejected and the fitted `(eps, gamma)` is far from
  `(1e-3, 2.0)`, the verdict is **knob wrong: refit or estimate jointly**.
- If the flat model is rejected and the fitted curve is consistent with the
  defaults, the verdict is **knob survives**.

Operationalization of "far from defaults" for this receipt: the defaults are
called consistent only if `eps = 1e-3` and `gamma = 2.0` both fall inside the
pair-bootstrap 95% percentile intervals and the fixed-default curve has
`delta_aic <= 10` against the best fitted confidence curve. Otherwise, after a
flat-model rejection, the fitted curve is called far from defaults.

## Method

Data sources:

- `artifacts/live/corpus-map-500-2026-07-08/map-cache.sqlite`
- `artifacts/live/judge-bench-2026-07-05/bench-cache.sqlite`

The script uses the `pairwise_cache` table and opens SQLite files in immutable
read-only mode so untracked WAL/SHM sidecars are not counted as committed data.

Signed log-ratio convention follows `src/rerank/comparison.rs` and
`src/rerank/types.rs`: `higher_ranked = A` maps to `+ln(ratio)` in presented
A-over-B coordinates, `higher_ranked = B` maps to `-ln(ratio)`. For residual
groups, each row is pulled back to the lexicographically first entity in the
unordered pair; rows where that entity was presented as slot B receive the sign
flip used by `signed_log_ratio_toward_first`.

Residual groups are keyed by source cache, model, prompt template, attribute,
and unordered entity pair. The extra source/template terms prevent accidental
mixing between receipt packs while preserving the requested within-pair
control. Refused and malformed rows are excluded from residual analysis and
reported with denominators.

For each residual-eligible group with at least two valid draws, the per-pair
mean signed log-ratio is subtracted and the resulting per-draw residuals are
used for:

- confidence histograms by source/model/template;
- empirical residual mean-square by confidence bin;
- a Gaussian residual likelihood with
  `Var(residual | confidence c) = sigma0^2 / g(c)`, where
  `g(c) = eps + (1 - eps) * c^gamma`.

The likelihood is fitted over residual draws after pair-mean subtraction.
Pair residuals are not independent because their group mean was estimated, so
bootstrap intervals resample whole residual groups, not individual rows.

Important caveats:

- Confidence and pair difficulty are confounded in raw data; subtracting the
  within-pair mean controls stable pair magnitude/difficulty, but it does not
  prove confidence was randomized within each pair.
- Residuals are generated from cached point judgments. The logprob evidence
  path's measured variance columns are intentionally not substituted for the
  stated-confidence channel being tested here.
- The likelihood-ratio p-value uses the standard chi-square approximation with
  two added parameters. The flat null is partly on a boundary/non-identifiable
  ridge (`gamma = 0` or `eps = 1`), so the bootstrap CI is treated as an
  equally important part of the verdict.

## Results

Run:

```bash
python3 artifacts/live/confidence-knob-2026-07-09/analysis.py
```

Results are written to `results.json`. The summarized verdict section below is
filled after running the script.

### Verdict

**Primary verdict: DELETE THE KNOB supported.**

No source/model/template confidence histogram crossed the >90% single-value
degeneracy gate, so this is not a "knob unexercised" result. The residual
likelihood then selected the flat model exactly: fitted `eps = 1e-9`,
`gamma = 0.0`, `sigma0 = 0.6170`. The likelihood-ratio statistic against flat
was `0.0` with chi-square-df-2 `p = 1.0`; the 500-pair-bootstrap 95% interval
for `gamma` was `[0.0, 0.0000061]`, covering zero.

The configured default curve was not merely unnecessary here; it was a much
worse fixed curve on these residuals: fixed-default AIC `23130.78` versus flat
AIC `21346.99` (`delta = +1783.79`).

### Denominators

SQLite rows read as committed immutable database files:

| source | rows | refused | malformed non-refused | valid observations |
|---|---:|---:|---:|---:|
| corpus-map-500-2026-07-08 | 10,762 | 11 | 0 | 10,751 |
| judge-bench-2026-07-05 | 684 | 1 | 0 | 683 |
| total | 11,446 | 12 | 0 | 11,434 |

Residual analysis:

| quantity | count |
|---|---:|
| valid unordered-pair groups | 5,457 |
| residual groups with `n >= 2` | 5,424 |
| residual draws | 11,401 |
| pair-mean-removed residual df | 5,977 |
| singleton groups excluded | 33 |
| singleton draws excluded | 33 |
| residual groups with both presentation orders | 5,424 |
| residual groups with only one presentation order | 0 |

Residual group-size histogram: `n=1`: 33, `n=2`: 5,225, `n=3`: 1, `n=4`:
192, `n=30`: 6.

### Confidence Histograms

The exact confidence histograms are in `results.json`. This table reports each
source/model/template's largest exact confidence mass and decile histogram
ordered as `[0,.1), [.1,.2), ..., [.9,1]`.

| source / model / template | valid n | refused | top exact mass | decile histogram |
|---|---:|---:|---:|---|
| corpus / deepseek-v4-flash / canonical_v2 | 5,310 | 6 | `0.85`: 1,496 (0.282) | `0,3,21,156,46,110,1191,827,1883,1073` |
| corpus / gemini-2.5-flash / canonical_v2 | 5,441 | 5 | `0.8`: 2,348 (0.432) | `2,0,0,0,0,1,63,175,2368,2832` |
| bench / claude-haiku-4.5 / canonical_v2 | 113 | 1 | `0.72`: 69 (0.611) | `0,0,0,0,0,0,8,80,21,4` |
| bench / claude-sonnet-4.6 / canonical_v2 | 114 | 0 | `0.62`: 17 (0.149) | `0,1,0,2,1,15,23,23,28,21` |
| bench / deepseek-v4-flash / canonical_v2 | 114 | 0 | `0.85`: 27 (0.237) | `0,0,0,0,0,1,25,18,38,32` |
| bench / gemini-2.5-flash / canonical_v2 | 114 | 0 | `0.9`: 39 (0.342) | `1,0,0,0,0,3,0,0,51,59` |
| bench / gpt-5.4-mini / canonical_v2 | 114 | 0 | `0.86`: 12 (0.105) | `0,0,0,0,0,8,13,24,37,32` |
| bench / gpt-5.4-nano / canonical_v2 | 114 | 0 | `0.62`: 45 (0.395) | `1,1,0,3,4,1,75,21,7,1` |

No row exceeded the 0.90 single-value degeneracy threshold.

### Empirical Precision Shape

Empirical relative precision is normalized to the top confidence bin. The
default weight column is the configured `eps=1e-3, gamma=2.0` shape, likewise
normalized to the top bin. The fitted curve is flat, so its relative weight is
`1.0` in every non-empty bin.

| confidence bin | residual draws | residual MSE | empirical rel. precision | default rel. weight |
|---|---:|---:|---:|---:|
| `[0.0,0.1)` | 2 | 1.367767 | 0.409 | 0.001 |
| `[0.1,0.2)` | 3 | 0.151735 | 3.686 | 0.018 |
| `[0.2,0.3)` | 21 | 0.043576 | 12.835 | 0.053 |
| `[0.3,0.4)` | 160 | 0.086160 | 6.491 | 0.112 |
| `[0.4,0.5)` | 50 | 0.123555 | 4.527 | 0.194 |
| `[0.5,0.6)` | 131 | 0.102903 | 5.435 | 0.340 |
| `[0.6,0.7)` | 1,397 | 0.171911 | 3.253 | 0.455 |
| `[0.7,0.8)` | 1,167 | 0.241180 | 2.319 | 0.592 |
| `[0.8,0.9)` | 4,429 | 0.343503 | 1.628 | 0.782 |
| `[0.9,1.0]` | 4,041 | 0.559298 | 1.000 | 1.000 |

On these cached point judgments, higher stated confidence does not correspond
to lower within-pair residual variance. The observed binned shape is roughly
opposite the configured monotone-up precision curve; because this receipt only
tests the shipped nonnegative-`gamma` family, the best admissible curve is flat.

### ML Fit

| model | eps | gamma | sigma0 | NLL | AIC |
|---|---:|---:|---:|---:|---:|
| flat | `1e-9` | `0.0` | 0.6170 | 10672.49 | 21346.99 |
| fitted confidence curve | `1e-9` | `0.0` | 0.6170 | 10672.49 | 21350.99 |
| fixed default curve | `1e-3` | `2.0` | 0.5306 | 11564.39 | 23130.78 |

Bootstrap over 5,424 residual pair groups, 500 replicates, seed `20260709`:

| parameter | p2.5 | p50 | p97.5 |
|---|---:|---:|---:|
| eps | `1e-9` | `1e-9` | 0.138885 |
| gamma | 0.0 | 0.0 | 0.0000061 |
| sigma0 | 0.5953 | 0.6174 | 0.6379 |
| LRT statistic vs flat | 0.0 | 0.0 | 10.1328 |

Subgroup sensitivity in `results.json` reaches the same conclusion: every
source/model/template either lands exactly on `gamma = 0` or an effectively
zero `gamma` with a non-rejected flat model.
