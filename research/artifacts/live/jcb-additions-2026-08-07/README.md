# JCB board additions: claude-sonnet-5 + claude-opus-4.8 (2026-08-07)

Same-week coverage of the two Anthropic releases the 2026-07-18 board
predates. Same frozen battery as `../kimi-k3-bench-2026-07-18/`
(canonical_v2, 194 comparisons/model, temperature 0, isolated `--no-cache`
runs through OpenRouter), so rows are within-version comparable with the
existing board.

## Results

| Model | JUDGE | signal (nats) | coherence (harm) | flip [95% CI] | residual | curl | spin | χ | pol ρ | para ρ | refusals | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | **0.537** | 0.817 | 0.963 (0.961) | 0.10 [.03,.30] | 0.131 | 0.030 | 3/3 | 0.07 | −0.93 | 0.93 | 0 | $0.666 |
| anthropic/claude-sonnet-5 | **0.450** | 0.667 | 0.925 (0.919) | 0.15 [.05,.36] | 0.077 | 0.040 | 3/3 | 0.14 | −0.93 | 0.93 | 1 | $0.478 |

Board placement (by JUDGE, within the 15-row board): opus-4.8 ranks 5th —
between gpt-5.6-terra (0.576) and gpt-5.6-sol (0.527). sonnet-5 ranks 11th —
between claude-haiku-4.5 (0.461) and deepseek-v4-flash (0.4496 exact;
sonnet-5 is 0.4505, a 0.0009 gap far inside the 0.022 retest floor —
a statistical tie displayed as equal 0.450).

## Readings (denominators as printed by the harness)

- **opus-4.8 slots below opus-4.6 (0.597)**: slightly lower signal
  (0.817 vs 1.003 nats) at higher coherence (0.963 vs 0.943). Its
  orbit-coherence 0.975 and interaction 0.008 are among the cleanest
  measured. Gap to opus-4.6 is 0.060 — just under the max observed retest
  delta (0.064), so "below" is directionally real but near the noise edge.
- **sonnet-5 lands below sonnet-4.6 (0.509)**: the composite drop is
  signal-driven (0.667 vs 0.833 nats — the newer model hedges more on this
  battery) while coherence improved (0.925 vs 0.901) and order-residual
  0.077 is best-in-family. One refusal (sonnet-4.6 had zero), which also
  dropped the null-bias cell to n=3.
- Null bias exactly 0.000 nats for both — the all-frontier convergence on
  that axis now holds at 15/15.

## Files

- `sonnet5-bench.log`, `sonnet5-leaderboard.json`, `sonnet5-report.jsonl`
- `opus48-bench.log`, `opus48-leaderboard.json`, `opus48-report.jsonl`

## Reproduce

```
vrun ./target/debug/cardinal bench --models anthropic/claude-sonnet-5 \
  --no-cache --json --out sonnet5-report.jsonl > sonnet5-leaderboard.json
vrun ./target/debug/cardinal bench --models anthropic/claude-opus-4.8 \
  --no-cache --json --out opus48-report.jsonl > opus48-leaderboard.json
```
