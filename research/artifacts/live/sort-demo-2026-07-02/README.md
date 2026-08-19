# Live `cardinal sort` demo receipt — 2026-07-02

A real, unedited run of the `sort` verb against OpenRouter, preserved as a
reproducibility receipt for the README quickstart.

## Commands

```bash
export OPENROUTER_API_KEY=your_key_here

cargo run --release --bin cardinal -- sort examples/sort-demo.txt \
  --by "usefulness as advice for a software engineer" \
  --model anthropic/claude-sonnet-4.6 \
  --seed 7 \
  --cache artifacts/live/sort-demo-2026-07-02/cache.sqlite \
  --trace artifacts/live/sort-demo-2026-07-02/trace.jsonl \
  --format json > artifacts/live/sort-demo-2026-07-02/output.json

# Keyless offline replay of the identical run from the pairwise cache:
cargo run --release --bin cardinal -- sort examples/sort-demo.txt \
  --by "usefulness as advice for a software engineer" \
  --model anthropic/claude-sonnet-4.6 \
  --seed 7 \
  --cache artifacts/live/sort-demo-2026-07-02/cache.sqlite \
  --cache-only --scores
```

## Files

| File | Contents |
|------|----------|
| `output.json` | Full structured result: per-item rank/mean/std/z/percentile plus run meta |
| `sorted-scores.txt` | Text-mode `--scores` output from the keyless cache-only replay |
| `trace.jsonl` | All 32 comparison events: pair, model, tokens, cost, ratio, confidence, cache status |
| `cache-export.jsonl` | The 31 cached pairwise judgements (exported via `cardinal cache-export`) |
| `stderr.txt` / `replay-stderr.txt` | The one-line run summaries |

The `cache.sqlite` file itself is gitignored; `cache-export.jsonl` is its
portable content.

## What this run shows — and what it does not

- 8 items, single attribute, default budget (4·n = 32 comparisons), model
  `anthropic/claude-sonnet-4.6`, provider-reported cost **$0.049995**,
  12,480 input / 837 output tokens, 0 refusals.
- The replay run served **32/32 comparisons from cache at $0** and ran without
  an API key in the environment.
- The run stopped at `budget_exhausted` with `topk_error ≈ 6.09`, i.e. the
  default budget was **not** enough to certify the middle boundary of this
  list to the default 0.1 tolerance. The posterior stds (~0.74–0.79) overlap
  heavily. That is the honest reading: with 32 judgements you get a
  well-motivated point estimate of the order, not a certified one. Raise
  `--budget` (or lower `--top-k`) when you need certainty.
