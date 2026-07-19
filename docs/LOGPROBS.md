# Logprobs Reference

This document is the reference for token logprobs in the judge gateway. All facts come
from live probes with dates and sample counts. The probe scripts are in
`notes/adaptive-logprobs-2026-07-19/`. When a provider changes, run the scripts again
and update the tables.

## Alternative counts for each model

The provider returns a list of alternative tokens for each output token position. The
`top_logprobs` parameter sets the requested list length. Each model serves a fixed
maximum count. The counts below come from the official OpenAI API (2026-07-18) and from
OpenRouter (2026-07-19, provider Azure, n=10 for each cell).

| Model | Count | Necessary condition |
|---|---|---|
| gpt-5.1, gpt-5.2 | 5 | `reasoning_effort` unset or `"none"` |
| gpt-5.4, gpt-5.4-mini, gpt-5.4-nano | 5 | `reasoning_effort` unset or `"none"` |
| gpt-5.5 | 5 | `reasoning_effort: "none"` |
| gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra | 5 | `reasoning_effort: "none"` |
| gpt-4.1, gpt-4.1-mini, gpt-4o family | 20 | none |
| gpt-5, gpt-5-mini, gpt-5-chat-latest | 0 | no path exists |
| gpt-5.5-pro, o3, o3-mini, o4-mini | 0 | no path exists |

Each 5.x model that serves logprobs serves 5 alternatives, and not more. A request
for 6 or more gets HTTP 400 from OpenAI. Many OpenRouter hosts do not reject an
over-cap request. They return HTTP 200 with `logprobs: null`. Thus the request layer
must clamp `top_logprobs` to the known cap for each route. The 2026-04 census
(`diamond` archive) measured caps of 5 for Alibaba hosts and 20 for Cerebras.

## The reasoning gate

OpenAI blocks logprobs when reasoning is on. A request with `reasoning_effort` set to
`low`, `medium`, `high`, or `xhigh` and `logprobs: true` gets HTTP 400. This applies to
Chat Completions and to the Responses API. The only path to a 5.x PMF is
`reasoning_effort: "none"`.

Through OpenRouter, the equivalent unlock is `reasoning: {"effort": "none"}` or
`reasoning: {"enabled": false}`. Measured 2026-07-19: gpt-5.5 and gpt-5.6-sol served
logprobs in 10 of 10 calls with the unlock, and 0 of 1 without it. The gpt-5.4 family
served logprobs with and without the unlock (20 of 20 calls, provider Azure).

NOTE: Provider behavior through OpenRouter changes from day to day. The
`model_supports_logprobs` gate recorded 400 errors for gpt-5.4 on 2026-07-18. The same
route served logprobs on 2026-07-19. Capability data must carry a probe date.

## Structured outputs

Strict `json_schema` response format keeps logprobs at `effort: "none"`. The schema
also pins each field to a stable token position. This makes PMF extraction
deterministic. Loose `json_object` mode lets the model select its own keys. Do not use
`json_object` mode as an instrument.

## PMF quality cautions

CAUTION: A numeric `ratio` field spreads its probability mass across many magnitude
tokens. Measured on gpt-5.4-mini (2026-07-19, n=13): the top-5 alternatives of the
first ratio token held only 0.23 mean visible mass, and sampled answers ranged from 20
to 5000. Single-token answer alphabets (the letter ladder) do not have this problem.

CAUTION: A visible analysis field before the answer collapses the answer PMF. Measured
on gpt-5.6-sol (2026-07-19): the ratio token PMF went from 5 alternatives with 0.81
top-1 mass to a single token with 1.0 mass. The model commits during its own visible
text. Do not put free-text analysis before the answer tokens in a PMF instrument.

## Two-phase elicitation: reasoning context, then a logprob read

There is no direct way to get a PMF from a reasoning pass. There is a two-phase way.
Phase 1 asks the model for an analysis at `reasoning_effort: "medium"` without a
verdict. Phase 2 sends a new request at `effort: "none"` with the analysis in the
context, and reads logprobs on the verdict tokens.

Measured on gpt-5.6-sol (2026-07-19, Chat Completions): the phase-2 PMF kept its
spread (0.81 and 0.19 on two ladder-adjacent tokens) and moved relative to the
one-shot PMF. The Responses API also accepts a `previous_response_id` continuation
from an `effort: medium` response into an `effort: "none"` request with logprobs.
But that path returned only 1 alternative per position (n=1). A fresh Responses call
returned 5. Use the Chat Completions two-phase shape until more probes explain this.

## Prompt cache and nonce perturbation

The OpenAI prompt cache makes distribution-stability measurement cheap. Put the stable
system prompt and entity text in a long prefix. Put the nonce at the end. Send
`prompt_cache_key` to help the cache route repeated prefixes to the same servers.

Measured on gpt-5.4-mini (2026-07-19, 1562 prompt tokens, 13 calls): 12 of 12 warm
calls hit the cache with 1280 cached tokens. Cached input tokens cost 10 percent of
the fresh price on this model family.

NOTE: Same-prompt repeats do not return identical logprobs. Three repeats of one nonce
gave top-1 ratio-token mass of 0.14, 0.12, and 0.26. Ten different nonces gave a mean
top-1 mass of 0.14 with a standard deviation of 0.06. Thus server noise and nonce
sensitivity had the same size in this measurement. A stability instrument must
average over repeats before it attributes variance to the nonce.

## Rerun commands

```
OPENROUTER_API_KEY=... python3 notes/adaptive-logprobs-2026-07-19/probe_openrouter_unlock.py
OPENAI_API_KEY=...     python3 notes/adaptive-logprobs-2026-07-19/probe_openai_direct.py
OPENAI_API_KEY=...     python3 notes/adaptive-logprobs-2026-07-19/probe_twophase.py
OPENAI_API_KEY=...     python3 notes/adaptive-logprobs-2026-07-19/probe_cache.py
```
