# Live openpriors.com arXiv board (2026-08-06)

The same 15-abstract corpus, run through the production judgement rail the
same evening as the rail comparison — the first arXiv lens on openpriors.com.

- POST `/v1/judgements/runs` (serve on colo2, loopback via SSH tunnel; the
  public front currently 403s under the operator's concealment posture, so
  visibility follows whenever that lifts — not changed here).
- lens `arxiv`, axis_key `practical-applicability`, axis_prompt identical to
  the rail-comparison attribute, model `anthropic/claude-sonnet-4.6`,
  privacy public, BYO openpriors OpenRouter key.
- run_ref `jrun_44a50701647041e3978568a3f75597ad`: 120 comparisons,
  $0.7039 (cost_is_estimate false), 128,612 input + 21,203 output tokens,
  completed 2026-08-07T04:17Z.
- Landing verified: `GET /v1/judgements?lens=arxiv&axis_key=practical-applicability`
  returns 15 rows from `scry_judgements.scores_current`.
- Cross-check: production rank 1 = `2608.06366v1`, the same item rail A
  ranked 1 locally (independent run, same model family, different pair plan).
- Board URL once visible: `/l/arxiv/practical-applicability`; per-run view
  `/j/jrun_44a50701647041e3978568a3f75597ad`.
