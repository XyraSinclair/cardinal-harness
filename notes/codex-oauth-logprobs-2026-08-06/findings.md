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

All calls route through the cxp proxy; no tokens touched. Cost: subscription
quota only, ~60 small calls total including the two wasted unguarded runs.
