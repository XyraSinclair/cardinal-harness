# JCB same-day board + Kimi K3 debut (2026-07-18)

Kimi K3 (released this week, `moonshotai/kimi-k3`, $3/M in · $15/M out,
1M ctx) benched on the Judge Coherence Benchmark, then the full board
re-run same-day so the comparison is within-run, not cross-date (the
battery grew from 138 to 194 calls since the 2026-07-05 packs — old
JUDGE scores are not comparable to these).

## Verdict

**Kimi K3 debuts at #1 with JUDGE 0.626, a 0.100 lead over
gemini-2.5-flash** — well above the documented test–retest floor
(mean |ΔJUDGE| 0.022, max 0.064, `../judge-bench-retest-2026-07-05/`),
so the lead is real, not sampling luck. Its profile is the cleanest
measured to date: order-flip 0/20, null bias exactly 0, spin survival
3/3 with the lowest susceptibility on the board (0.146 nats/spin),
curl 0.018, zero refusals.

## Board (canonical_v2, 194 comparisons/model, temperature 0)

| # | Model | JUDGE | signal (nats) | coherence (harm) | flip [95% CI] | residual | curl | spin | χ | pol ρ | para ρ | refusals | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | moonshotai/kimi-k3 | **0.626** | **1.133** | 0.923 (0.915) | **0.00** [0,.16] | 0.144 | 0.018 | 3/3 | **0.15** | −0.86 | **0.98** | 0 | $1.494 |
| 2 | google/gemini-2.5-flash | 0.526 | 1.015 | 0.825 (0.751) | 0.15 [.05,.36] | 0.274 | 0.039 | 3/3 | 0.22 | −0.81 | **0.98** | 0 | $0.036 |
| 3 | anthropic/claude-sonnet-4.6 | 0.509 | 0.833 | 0.901 (0.793) | **0.00** [0,.16] | 0.101 | **0.008** | 3/3 | 0.32 | **−0.93** | 0.93 | 0 | $0.382 |
| 4 | anthropic/claude-haiku-4.5 | 0.461 | 0.756 | 0.870 (0.687) | 0.10 [.03,.30] | **0.084** | 0.032 | 3/3 | 0.32 | −0.81 | 0.83 | 4 | $0.282 |
| 5 | deepseek/deepseek-v4-flash | 0.450 | 0.865 | 0.776 (0.699) | 0.15 [.05,.36] | 0.154 | 0.143 | 2/3 | 0.47 | −0.55 | **0.98** | 0 | $0.010² |
| 6 | openai/gpt-5.4-mini | 0.363 | 0.655 | 0.756 (0.437) | 0.20 [.08,.42] | 0.292 | 0.076 | 2/3 | 0.55 | −0.62 | 0.93 | 0 | $0.077 |
| 7 | openai/gpt-5.4-nano | 0.246 | 0.522 | 0.605 (0.000) | 0.40 [.22,.61] | 0.238 | 0.217 | **0/2** | 0.96 | −0.02 | 0.93 | 0 | $0.021 |

² deepseek's isolated retry hit cache for 118 of 194 calls (paid in the
aborted board run); its first attempt aborted mid-battery on an empty
completion — see Harness findings.

Null bias was exactly 0.000 nats for every model on the board.

## Notable structure

- **K3 wins on signal × coherence jointly**: sonnet is more coherent on
  residual/curl but separates items 27% less (0.833 vs 1.133 nats); gemini matches K3's
  paraphrase stability but flips order 15% of the time. K3 is the first
  model measured with a perfect flip record *and* >1 nat of signal.
- **gpt-5.4-nano's harmonic coherence is 0.000** — driven by spin
  survival 0/2: its direction follows the asker's lean. Its polarity ρ
  of −0.02 means negating the attribute barely changes its answers
  (inversion-blind), consistent with the 07-05 finding that nano cannot
  produce a reproducible polarity relation.
- **K3 lacks logprobs via OpenRouter**, so `calibrate` (single-token
  ratio-letter instrument) returned no parseable null judgements —
  sampled-mode JSON is its only channel. The bench's sampled null axis
  still measured 0.000 nats bias.

## Orbit probe (Z₂³, GiveDirectly vs AMF)

`orbit-givedirectly-amf.json`: K3 judged "expected lives saved per
marginal $1M donated" over the full order × polarity × wording orbit
(8 comparisons, $0.093). Invariant coefficient **−2.386 nats** (AMF
≈ 10.9× GiveDirectly — direction consistent with GiveWell's published
cost-effectiveness ordering), orbit coherence **0.986** (98.6% of
Parseval energy in the invariant component), all 8 framings agree in
sign, Parseval residual 2e-15, zero refusals.

## Harness findings (for the public benchmark)

1. **One flaky model aborts the whole multi-model bench run**:
   deepseek-v4-flash returned an empty completion mid-battery
   ("Invalid JSON: EOF", the exact failure mode documented in
   `../../notes/manifund-campaign-2026-07-13/p1-results.md`) and the
   remaining models never ran. The public harness needs per-model
   isolation: one model's provider flake must cost one row, not the
   board. (This run's workaround: separate invocations, see
   `oai-bench.log` / `deepseek-bench.log`.)
2. **Bench re-runs are fresh by design** (procedural nonce rotation), so
   same-day re-invocation of an already-benched model hits cache only
   within the same battery version — K3's rows were served from cache in
   the board run (194/194 cached, $0 marginal).

## Files

- `bench.log` / `report.jsonl` / `leaderboard.json` — K3 solo run (first
  contact, $1.4935).
- `full-bench.log` / `full-report.jsonl` — board run (K3 cached, gemini,
  sonnet, haiku complete; aborted at deepseek).
- `oai-bench.log` / `oai-report.jsonl` / `oai-leaderboard.json` — mini +
  nano.
- `deepseek-bench.log` / `deepseek-report.jsonl` — isolated retry.
- `orbit-givedirectly-amf.json` / `orbit.log` — the Z₂³ probe.
- `calibrate.jsonl` — empty (no logprobs channel on K3; see above).

Total spend this pack: ≈ $2.39 (K3 $1.49 + board $0.70 + mini/nano
$0.10 + orbit $0.09 + deepseek retry $0.01), on the `scry-chambers1`
vault key ($150 headroom; the shell-env `epistack-competition1` key is
exhausted at its $150 cap — use `vrun` for all cardinal runs).
