# Results: Claude Code rail vs Claude API rail (2026-08-06)

Design pre-registered in DESIGN.md (with its one pre-run amendment). Both
rails ran to completion the same evening, same corpus (15 cs.LG abstracts),
same attribute, same seed, model pinned `claude-sonnet-4-6` both sides,
`--no-cache`, 60 comparisons each (0 cached, 0 refused on both).

## Headline

The subscription rail reproduces the API rail. **Every undirected pair on
which both rails were internally decisive agreed: 21/21 (100%).** Board-level
Spearman 0.850, Kendall tau 0.676 (n=15 items, 105 rank pairs). The residual
rank divergence comes from sparse pair coverage (each rail judged 30 of 105
undirected pairs; 25 shared), 4 internally-tied pairs, and solver posterior
uncertainty — not from contradicting judgments.

## Per-rail instrument health

| | API (OpenRouter) | Claude Code (subscription) |
|---|---|---|
| comparisons | 60 (0 refused) | 60 (0 refused) |
| order flips | 4/30 | 1/28 |
| frustration | 0.015 | 0.025 |
| syst order | 0.192 nats/pair | 0.270 nats/pair |
| provider cost | $0.343 | $0.000 marginal |
| wall time | ~1.5 min | ~14 min |

The subscription rail's position-bias diagnostic (order flips 1/28) is no
worse than the API's — the CLI scaffolding did not add measurable
position sensitivity in this run (n=1 run; instrument demonstration per
PRINCIPLES.md §3, not a model property).

## Verdict for practice

Claude Code print mode is fit for provenanced judgment runs where wall time
is not binding: same-model pair verdicts matched the API perfectly in this
run at zero marginal cost. The API rail remains the choice for latency
(~10× faster here) and for logprob evidence (the CLI serves none).

## Caveats

- ComparisonTrace records the requested slug (`claude-code/claude-sonnet-4-6`)
  per row; the adapter's per-response `served_model` is not yet threaded into
  traces. Tonight's pin is evidenced by the pre-registered smoke calls
  (modelUsage `claude-sonnet-4-6`) and by the CLI erroring on unknown model
  ids. Threading served_model into traces is a noted follow-up.
- Prompt bytes differ across rails by construction (CLI scaffolding); this
  measured the rails as deployed.
- Board-level agreement at 30/105 pair coverage mixes rail difference with
  plan divergence; a shared-plan (fixed pair schedule) variant would isolate
  the rail effect if 0.85 ever needs tightening.

## Reproduce

```
# rail A
vrun ./target/debug/cardinal sort corpus.json --by "practical real-world applicability of the paper's contribution, as evidenced by the abstract" \
  --model anthropic/claude-sonnet-4.6 --seed 20260806 --no-cache --format json --trace api-trace.jsonl > api.json
# rail B
CARDINAL_CLAUDE_CODE_CONFIG_DIR=~/.claude-judge ./target/debug/cardinal sort corpus.json --by "..." \
  --model claude-code/claude-sonnet-4-6 --seed 20260806 --no-cache --format json --trace cc-trace.jsonl > cc.json
python3 analyze.py api.json api-trace.jsonl cc.json cc-trace.jsonl
```
