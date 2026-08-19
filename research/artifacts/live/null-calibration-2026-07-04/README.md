# Null-pair calibration receipt — 2026-07-04

`cardinal calibrate`: byte-identical text presented in BOTH slots of the
ratio-letter instrument, 8 diverse null texts per model. A perfect judge
answers parity (`A`); any directional probability mass is pure elicitation
artifact — position prior plus letter prior, with no content to hide behind.

| model | P(slot A) | P(parity) | P(slot B) | mean bias (nats) | cost |
|---|---|---|---|---|---|
| openai/gpt-5.4-mini | 0.000 | 1.000 | 0.000 | 0.0000 | $0.0032 |
| openai/gpt-5.4-nano | 0.000 | 1.000 | 0.000 | 0.0000 | $0.0008 |
| deepseek/deepseek-v4-flash | 0.000 | 1.000 | 0.000 | 0.0000 | $0.0004 |
| anthropic/claude-haiku-4.5 | 0.000 | 1.000 | 0.000 | 0.0000 | $0.0044 |

## Reading it

- **At the exact null point, all four judges are clean**: full mass on
  parity, zero directional artifact. The single-letter alphabet does not,
  by itself, induce a measurable letter or position prior on these models.
- **This does not clear position bias in general.** The identical-text null
  is the easiest case; real position bias appears on near-tie DISTINCT
  pairs — which is what the order-residual receipt measures per run
  (e.g. 13/27 order flips, residuals ~0.03 nats, in the live evidence-path
  receipt). The two instruments are complementary: `calibrate` isolates the
  pure prior; the residual measures bias under content load.
- Total cost of the sweep: under one cent. Run it on any judge before
  trusting it: `cardinal calibrate --models <slugs> --nulls 8`.
