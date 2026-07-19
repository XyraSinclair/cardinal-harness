# Logprob unlock census + adaptive instrument design (2026-07-19)

Session goal: make the harness use logprob PMFs wherever a provider can emit them, and
design the adaptive layer that chooses between effort-none+logprobs, resampling, reasoning
escalation, and model escalation per pair. Everything below is dated, denominated, and
rerunnable (`probe_openai_direct.py`, `probe_openrouter_unlock.py` in this directory).

## 1. Headline measurements

**The `reasoning: {"effort": "none"}` unlock works through OpenRouter.** GPT-5.5 and
GPT-5.6 reject `logprobs` at default reasoning (HTTP 400) and serve top-5 logprobs when the
request carries `effort: "none"` or `enabled: false`.

**`model_supports_logprobs` is stale for the GPT-5.4/5.6 families.** The gate
(`src/rerank/comparison.rs:947-953`) returns false for `openai/gpt-5.4*` and
`openai/gpt-5.6*` citing upstream 502/400 dated 2026-07-18. Measured 2026-07-19 via
OpenRouter, provider=Azure on every call:

| cell | n | logprobs returned | cost total |
|---|---|---|---|
| openai/gpt-5.4-mini, control (no reasoning field) | 10 | 10/10, top-5 | ~$0.001 |
| openai/gpt-5.4-mini, effort=none | 10 | 10/10, top-5 | ~$0.001 |
| openai/gpt-5.5, effort=none | 10 | 10/10, top-5 | ~$0.007 |
| openai/gpt-5.6-sol, effort=none | 10 | 10/10, top-5 | ~$0.007 |
| openai/gpt-5.5, control | 1 | 0/1 (HTTP 400) | — |
| openai/gpt-5.6-sol, control | 1 | 0/1 (HTTP 400) | — |

Total session probe spend: $0.021 (12 + 40 calls). Noise class: single day, single
OpenRouter provider (Azure). The 2026-07-18 gate note and the 2026-07-19 measurements can
both be true — provider routing varies by day. That volatility is itself the finding: a
hard-coded gate rots in under 24 h; capability must be probed, dated data.

Direct OpenAI API (2026-07-18, `probe_openai_direct.py`, key outside this repo):

- gpt-5.5, gpt-5.6-luna/sol/terra: logprobs **iff** `reasoning_effort: "none"`; hard cap
  `top_logprobs ≤ 5` (6/10/20 all → 400); works identically on Chat Completions and
  Responses API (`include: ["message.output_text.logprobs"]`).
- Any real effort (low…xhigh) → 400 "logprobs is not supported". No reasoning-on PMF
  exists anywhere in the 5.x line.
- gpt-5.5-pro: no path (chat 403; Responses forbids effort=none; efforts forbid logprobs).
- o3/o4 family, gpt-5, gpt-5-mini, gpt-5-chat-latest: 403.
- gpt-4.1/4o family: unconditional logprobs, top-20 cap.
- Strict `response_format: json_schema` preserves logprobs at effort=none and pins token
  positions. Observed ratio-token PMF (gpt-5.6-sol, n=1 illustration): 120@0.52, 100@0.24,
  130@0.10, 140@0.04, 115@0.02. Loose `json_object` let the model invent keys — not a
  valid instrument shape.

## 2. What this changes in the code (anchors from a full 2026-07-19 source map)

1. **The default judge is logprob-dark.** `openai/gpt-5.4-mini` is the default everywhere
   (`src/rerank/multi.rs:63`, `src/rerank/model_policy.rs:60`) and the gate returns false
   for it, so the default JSON path runs point-only with unit precision
   (`multi.rs:1168`). Measured today it serves top-5 logprobs at ~$0.0001/call.
2. **`ReasoningConfig` is dead in production.** The only live setter is the Kimi
   disable fallback (`src/gateway/openrouter.rs:333-341`). `ReasoningEffort::None` is never
   constructed on any request. The unlock maneuver is unimplemented; the gate blanket-refuses
   reasoning-class models instead (`comparison.rs:935-941`).
3. **Effort is instrument identity, not transport.** Switching the default judge from
   default-reasoning-sampled to effort-none-logprob changes the measurement, not just the
   wire format (prompt bytes are physics; so is decode mode). The flip needs its own
   instrument row and a benchmark comparison, not a drive-by default change.
4. **Truncation handling is already honest.** `collect_distribution` keeps residual mass
   `(1 − Σp).max(0)` explicit (`src/gateway/types.rs:947-949`); the seriate path marks
   `logprob_mode = false` on degrade (`comparison.rs:1135`). The censored-likelihood solver
   (`src/censored_likelihood.rs`) is the natural consumer of truncated PMFs and is built
   but unwired.

## 3. Design: the adaptive instrument layer

Instrument identity = (model, provider, reasoning mode, response format, top_logprobs,
slot order, cache state). Capability and trust attach to that tuple, never to a model name.

**Capability registry, probed not declared.** Replace the hard-coded
`model_supports_logprobs` string-matching with a small dated registry:
`{instrument → LogprobGate ∈ {Unconditional, EffortNoneOnly, Never}, top_k_cap,
probed_at, n, provider}`. A probe binary reproduces the census above (~$0.02 for the
full matrix). The gate function becomes a lookup with the current hard-coded table as the
seed/fallback. Advertised support never enters the registry; only measured support does.
Stale entries (> some TTL) degrade to the existing loud-retry path
(`comparison.rs:1054-1058`) rather than silently gating.

**Escalation ladder per pair, cheap → expensive.** All rungs already have machinery;
what is missing is the triggers wiring diagnostics to actions:

1. *Logprob read* (effort-none where the gate requires): default rung for every capable
   instrument. ~$0.0001–0.0007/call measured.
2. *Reverse order*: already counterbalanced (`multi.rs:785-799`).
3. *Resample* (nonce draws, `src/rerank/sampling.rs`): trigger when visible logprob mass
   is low or the PMF is near-flat — the roadmap doctrine that today exists only as prose.
4. *Reasoning escalation* (effort low/medium, sampled, no logprobs): trigger when a pair's
   readings disagree beyond what their variance explains — high studentized LOO |z|
   (`rating_engine.rs:1119-1166`) on a pair with nonzero frontier weight. Its evidential
   value is bias correction: reasoning-vs-none disagreement is a per-instrument offset to
   estimate, not noise to average away.
5. *Model escalation at the boundary*: the current ladder only cheapens easy pairs
   (`model_policy.rs:84-109`); add the upward rung — critical pairs
   (`trait_search.rs:1357-1388`) that stay unstable after rungs 1–4 go to the high model.

Every trigger emits a candidate into the existing planner and competes on the existing
score `(delta_info + λ·delta_rank_risk)/cost` (`rating_engine.rs:2314-2442`). The ladder
adds candidates; the scorer stays.

**Per-instrument gain/bias joins the solve.** `gain_calibration.rs` already fits
per-template multiplicative gains by bilinear alternation. Reasoning-vs-none is the same
shape: model `y = g_m·(s_i − s_j) + b_m + ε` per instrument, reference instrument pinned.
Escalation observations then correct the cheap instrument globally instead of only fixing
one pair. This is MATH_FRONTIER §3⅞ (judge portfolio) plus §5¾ (template gains) meeting
the capability registry.

**Censored-likelihood cutover consumes truncation.** Top-5 slices on flat PMFs censor
real mass; `censored_likelihood.rs` treats rungs as intervals and is the principled sink
for `visible_mass < 1`. Wiring it stays a separate tranche gated on its own benchmark.

## 4. Proposed tranches (issue-contract shaped, each independently bounded)

1. **Effort-none unlock in the gateway path.** When a logprob-gated request targets an
   `EffortNoneOnly` instrument, attach `ReasoningConfig { effort: None }`. Acceptance:
   seriate + bucket paths obtain logprob posteriors on `openai/gpt-5.5` and
   `openai/gpt-5.6-sol`; degrade path still loud. Smallest slice, unblocks everything.
2. **Probed capability registry** seeded with today's table; gate becomes lookup;
   probe bin + dated fixture. Acceptance: `model_supports_logprobs` truth table derives
   from the fixture; stale-TTL degrade covered by a scripted-pathology test.
3. **Resample + reasoning escalation triggers** in the rerank loop, competing on the
   existing planner score. Acceptance: planted flat-PMF pathology triggers resample;
   planted high-residual pathology triggers reasoning escalation; neither fires on
   clean fixtures.
4. **Instrument gain/bias estimation** extending `gain_calibration.rs` to
   reasoning-vs-none channels. Acceptance: planted-bias fixture recovers the offset.
5. **Boundary model-escalation rung** in `ModelPolicy`. Acceptance: critical-pair
   instability routes upward on a planted near-tie fixture.
6. **Instrument benchmark** before any default flip: effort-none+logprob vs current
   default sampled mode on a frozen corpus (accuracy per dollar, test–retest). The
   default changes only if this wins.

## 5. Open questions

- Does the Azure-served logprob path for gpt-5.4-mini hold across days/providers?
  (Yesterday's gate note says no; today says yes, 20/20. Registry TTL answers this
  structurally.)
- Reasoning-effort sweep on sampled mode (FIRST_PRINCIPLES §6, "never swept") — needed
  before rung 4's bias model has priors.
- Multi-token ratio atoms under decimal ladders: the strict-schema numeric PMF observed
  above is rich; continuation rescoring remains the known blocker
  (`gateway/types.rs:220-224`).
