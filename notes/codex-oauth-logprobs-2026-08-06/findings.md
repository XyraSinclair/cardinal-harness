# Codex-oauth logprobs probe — findings (2026-08-06)

Question: can the ChatGPT-subscription Responses backend
(`chatgpt.com/backend-api/codex/responses`, reached through the cxp pool
proxy) serve logprobs, and does the two-phase reason-then-read shape from
docs/LOGPROBS.md survive on it?

## Answers

1. **Yes, chosen-token logprobs.** `include: ["message.output_text.logprobs"]`
   at `reasoning: {"effort": "none"}` returns per-token logprobs on the
   answer text. gpt-5.6-sol: 10/10 calls. gpt-5.4-mini: 10/10 calls.
2. **No alternatives.** The `top_logprobs` parameter is rejected at the
   gateway with FastAPI-style `{"detail": "Unsupported parameter:
   top_logprobs"}` at every effort (11/11 rejections across two runs). Each
   served logprob entry has `top_logprobs: []`. The PMF visible per call is
   the sampled token's mass only.
3. **`effort: "none"` is officially supported on the wire** even though
   models_cache advertises only low…ultra: the 400 for `minimal` enumerates
   `'none', 'low', 'medium', 'high', 'xhigh', 'max'` as supported values for
   gpt-5.6-sol.
4. **The reasoning gate reproduces.** Any reasoning effort (or unset, which
   defaults to reasoning) + logprobs include → 400 "logprobs are not
   supported with reasoning models" — same gate as the official API.
5. **Two-phase works subscription-billed.** Phase 1 at `medium` (reasoning
   tokens observed), phase 2 fresh call at `none` with the analysis as
   assistant context + logprobs include → logprobs served (n=1 shape check;
   the phase-2 request is the same request class as the 10/10 cell).

## Cross-model two-phase (added 2026-08-07)

`probe_crossmodel.py`, uncertain pair "1L solid ice (A) vs 1L liquid water
(B) by mass" (correct: B — ice is less dense, so a liter of it is lighter):

| Cell | n | Answer | Sampled mass |
|---|---|---|---|
| A: sol@none baseline | 5 | B 5/5 | 1.0000 |
| D: mini@none baseline | 5 | **A (wrong) 5/5** | 0.4294 mean, 0.15 sd |
| B: sol ← sol@medium analysis | 5 | B 5/5 | 1.0000 |
| C: mini ← sol@medium analysis | 5 | B 5/5 | 1.0000 |

Cell C is the payload: **reason on one model, read logprobs on another.**
mini alone answers wrong at 0.43 confidence; given sol's 314-char
`medium`-effort analysis (84 reasoning tokens) as assistant context, mini
flips to correct at 1.0 mass, logprobs served 5/5. The reasoning model does
the thinking; the cheap non-reasoning read gives the calibrated PMF.

Verdict-leak trap (caught live): phase 1 initially inherited the one-letter
system prompt, so the "analysis" was literally the verdict letter
(analysis_chars=1) and phase 2 measured verdict-copying, not reasoning
transfer. The fixed probe overrides `instructions` with a no-verdict analyst
prompt and asserts analysis length > 80.

## Prompt cache on this backend (added 2026-08-07, small n)

`probe_cache_codex.py`: ~3700-token prompt, 2816-token stable prefix, nonce
at the tail, 6 calls per run.

- Unkeyed: 1/6 then 2/6 calls returned `cached_tokens: 2816`.
- With `prompt_cache_key`: 2/6. Parameter accepted, no visible lift at n=6.
- Proxy log confirms every call in these runs was served by ONE account —
  so the miss scatter is upstream OpenAI-side cache routing, not pool
  rotation.
- Side discovery: cxp-agent's `thread_key()` uses `prompt_cache_key` (then
  conversation/session id) for account-affinity routing. Keyed traffic
  therefore pins one pool account automatically — the right behavior for
  cache coherence, since each account is its own cache namespace.

Contrast: the official API measured 12/12 warm hits with the same shape
(docs/LOGPROBS.md). Codex-rail calls are $0 marginal anyway, so cache here
is a latency optimization only; the cache-economics case for nonce-perturbed
resampling lives on the API rail.

## Robustness probes (added 2026-08-07, second pass)

Two refutation attempts against the headline claims, both survived:

**Does analysis context mechanically saturate the phase-2 read?** No — the
read tracks evidence content (`probe_spread.py` + inline follow-up). On the
near-tie pair "all ants vs all humans by total mass" (mini baseline: mixed
tokens, mass 0.66, sd 0.23):

| Context given to mini@none | Tokens (n=5) | Mass |
|---|---|---|
| none (baseline) | A,B,A,A,A | 0.66 sd 0.23 |
| sol@medium real analysis (contains decisive biomass numbers) | B ×5 | 0.999 |
| deliberately balanced analysis (resolves nothing) | A,A,A,A,B | 0.63 sd 0.20 |

Saturation follows the *decisiveness of the analysis*, not the presence of
context. A first attempt used an "undecidable" beauty pair (aurora vs
eclipse) and misfired: mini's baseline was already 0.995, so it could not
discriminate — refutation pairs need measured baseline spread.

**Does the harness's real rail (OpenRouter) deliver cache hits?** Yes —
`probe_cache_openrouter.py`, same nonce-at-tail shape, openai/gpt-5.4-mini
served by Azure: 5/5 warm hits both with and without `prompt_cache_key`
(3328 of ~3658 tokens cached; warm cost $0.00052 vs cold $0.00277, an 81%
discount). The key was not load-bearing for serial single-stream calls; per
OpenAI docs it aids cache routing under concurrent load, so the threaded
keys are harmless insurance there. Caveat: pinning
`provider: {"only": ["openai"]}` 401s through the account's stale OpenAI
BYOK integration — unpinned routing (the harness default) is what was
measured.

## Parser trap (cost us the first two runs)

On this backend `response.completed` carries an **empty `output` list**.
The real output items — including logprobs — arrive only in
`response.output_item.done` events. Any client that reads only the completed
event concludes "no output, no logprobs" and is wrong. `probe_codex_oauth.py`
`parse_stream` rebuilds output from item-done events.

## Consequences for the harness

- Subscription-billed confidence weighting is available: single-token answer
  alphabets (the letter ladder) give P(sampled letter) at zero marginal cost
  through the codex oauth rail.
- Without alternatives, distribution spread needs repeated sampling at
  temperature; each sample still reports its own token's true mass, so the
  visible-mass diagnostic (is the sampled mass low?) survives per call.
- Easy-pair pin: both models put ≈1.0 on 'B' (bowling ball heavier than egg),
  the correct answer, in all 20 calls.

## Scripts

- `probe_codex_oauth.py` — effort ladder + two-phase (guarded main).
- `probe_variants.py` — include-only vs top_logprobs discrimination.
- `probe_raw_sse.py` — raw SSE event dump (found the empty-output trap).
- `probe_repeats.py` — n=10 repeat cells per model.
- `probe_crossmodel.py` — cross-model two-phase cells A–D.
- `probe_cache_codex.py` — prompt-cache hit-rate probe (`--key` adds
  `prompt_cache_key`).
- `probe_spread.py` — saturation-vs-evidence-tracking refutation probe.
- `probe_cache_openrouter.py` — end-to-end cache hits on the OpenRouter
  rail (`--no-key` control; needs `vrun`).

All calls route through the cxp proxy; no tokens touched. Cost: subscription
quota only, ~60 small calls total including the two wasted unguarded runs.
