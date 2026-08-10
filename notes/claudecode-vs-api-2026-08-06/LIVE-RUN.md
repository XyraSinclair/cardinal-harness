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

## Board #2: arxiv/interestingness (2026-08-10) — the reusable recipe

48 freshest August cs.AI/LG/CL abstracts, `anthropic/claude-haiku-4.5`,
`requested_k` 12, privacy public. `jrun_b61f63f83cef4f46bf41d90d70194688`:
384 comparisons, $1.274 (`cost_is_estimate` false), 427k in + 170k out,
~5 min wall. Live at openpriors.com/l/arxiv/interestingness.

Recipe deltas vs the pilot (these are the parts worth keeping):

- **Abstract source**: `scry_pg.arxiv_papers.payload` (String) = title +
  full text with `\nAbstract\n\n` marker; `metadata` JSON has NO abstract
  (checked 2026-08-10 — `arxiv_tex_hf.abstract` is a false memory). Select
  `argMax(substring(payload,1,6000), updated_at)` grouped by `arxiv_id`,
  parse title = first line, abstract = paragraph after the marker, cut at
  `1 Introduction`-style headings, cap ~2.4k chars.
- **No tunnel needed**: on colo2 the API front serves loopback
  `127.0.0.1:8080` (`/v1/judgements*`; GET run status at
  `/v1/judgements/runs/<ref>`). cardinald itself binds `CARDINALD_ADDR`
  (127.0.0.1:8093); talk to 8080.
- **No BYO key transit**: source `CARDINALD_OPENROUTER_KEY` from
  `/etc/cardinald/env` inside a server-side python (`sudo python3`, read
  the one line by `startswith`), POST with `x-provider-key` on loopback.
  Key bytes never leave colo2 or enter a transcript.
- Request schema (serve `routes/judgements.rs`, `deny_unknown_fields`):
  `entities[{id,text}] ≤200 × ≤8KB`, `axis_key`, `axis_prompt ≤4KB`,
  `requested_k` (1..=n, top-k target — all entities still get scores),
  `model`, `privacy`, `lens`. Ship the JSON by scp + sha256 parity.
