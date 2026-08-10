# Measured census + revised kernel (2026-08-10, OpenRouter provider=Azure throughout)

Probes run the same night as DESIGN.md; every cell below is from `probe_peel.py` or the
inline follow-ups recorded in the session. Spend: ~$0.05 total on the vault openpriors
key. Noise class: single day, single provider (Azure via OpenRouter), n small —
per-token masses drift a few percent between calls on the mini (provider numeric
nondeterminism); treat masses as measurements with noise, not constants.

## Census results

| # | question | verdict | evidence |
|---|---|---|---|
| 1 | grammar-mask semantics | **logprobs are masked AND renormalized to grammar-legal tokens** | schema forces `.` after int: unconstrained pos shows `"}`/`.`/`0` at .38/.38/.23; under pattern the same pos shows `.` at exactly 1.0. 4.1-mini pads top-5 with vocab-start junk (`!`,`"`,`#`) at ~0 — mask artifact. |
| 1b | grammar visibility | **the schema is ALSO in the model's context** — changing the grammar changes beliefs, not just the mask | enum excluding the mode (`120`, raw p=.79): `140` (raw p<.006) jumps to .396. Only context-visibility explains it. |
| 2 | logit_bias peeling | **DEAD on GPT-5.x/Azure at effort=none** — accepted syntactically, ignored at sampler and in logprobs | banned `120` (bias −100) still sampled 3/4 draws at its natural p≈.79. |
| 2b | chosen-token exact mass | **ALIVE — the load-bearing mechanism** | sampled sub-top-5 tokens (`500`, `55`) return their own exact logprob (`55` → p=.0110). |
| 2c | `n` batched sampling | Azure silently collapses n=8 → 1 choice; OpenAI-proper route 401 on this key | sequential prompt-cached calls are the rail for now. |
| — | determinism | 5.6-sol logprobs bit-identical across 4 calls; 5.4-mini masses drift ~±10% relative | probes are (near-)reproducible; ledger stores provenance. |

## The revised kernel: resampling IS peeling

logit_bias is dead and grammar mutation changes the instrument (1b), so DESIGN.md's
`peel` move is replaced by something better that needs neither:

**Every temperature-1 draw under the FIXED grammar returns its own exact mass** (2b),
even when the drawn token is far below top-5. So width-peeling at a node is just
resampling at that node: distinct draws reveal residual tokens with exact masses;
discovery rate is proportional to undiscovered mass (coupon-collector) — heavy hidden
tokens surface fast, light ones stay honestly inside the residual slack. Discovery is
stochastic; measurement is exact. No renormalization arithmetic, no provider-fragile
parameters — any logprob-serving provider supports it.

**Structural collapse for the ratio grammar.** Under pattern `^[0-9]{1,3}\.[0-9]$`,
every JSON-format position is forced (measured p=1.0): the entire distribution lives at
TWO stochastic positions (integer group, fraction digit). The trie collapses to a small
product; root resampling IS node resampling; **prefill is not needed for the flagship
instrument**. The prefill census now only gates deeper grammars (multi-field, long
decimals).

## End-to-end mini-kernel run (worst case: flat gpt-5.4-mini)

Egg vs bowling ball by mass, 25 sequential draws (~$0.003, prompt-cached):

- 25 distinct integer tokens enumerated with exact masses; enumerated mass 0.408,
  residual 0.592 (this model is genuinely flat: mode `100` @ .044).
- E[log10 r] point (renormalized) = 1.80; truth ≈ 1.98 (5450g/57g); credal envelope
  [0.73, 2.51] with residual placed adversarially across the full grammar support.
- Same instrument on gpt-5.6-sol: ONE call enumerates 0.98 mass (`120` @ .79 — and 120x
  is factually right); envelope closes to ~±0.01 for ~$0.001.

Reading: the envelope is honest about what top-5 + flat models cost; the point estimate
is already sane at 25 calls on the worst case; on peaked models the full calibrated PMF
is essentially free. Where the judge is uncertain, that uncertainty is now measured
structure (a wide exact PMF) instead of sampling noise.

## Instrument-identity doctrine (hardened by 1b)

The grammar is part of the prompt physics. A kernel run is identified by
(prompt, grammar, model, effort, decode mode); all probes within a run keep the grammar
byte-identical. Cross-grammar comparisons (ladder vs free-form) are comparisons of
DIFFERENT instruments and must be labeled as such. This is the same doctrine as
effort-none vs sampled (adaptive-logprobs 2026-07-19 §2.3), now with direct evidence.

## What remains open

1. Prefill census (probe #3) — now only for deep grammars, demoted in priority.
2. `n` batching via a provider that honors it (would cut resample cost further).
3. Cross-provider replication of census #1/#1b/#2 (single-provider, single-day so far).
4. Validation study (probe #5): kernel PMF vs large-sample frequencies, one model.
5. Oracle consult in flight; reconcile its answer against this note when it lands.
