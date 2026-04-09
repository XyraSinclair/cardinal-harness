# GX10 Pairwise SFT

This directory is the shortest path from cardinal-harness rerank traces to a
real fine-tuning run.

## First Hypothesis

The first experiment should test one thing:

> A swap-aware SFT model trained on exported pairwise rerank traces will reduce
> cost to certified stop by at least 25%, or comparisons to stop by at least
> 20%, relative to the untuned baseline judge, without materially hurting top-k
> quality or increasing false refusals.

That hypothesis is:

- interesting because it targets end-to-end efficiency, not just agreement
- falsifiable because it has a numeric threshold
- cheap enough to test in a few days

## Stack Choice

For v1, use the boring stack:

- data export: `cardinal dataset-export`
- replay prompt export: `cardinal prompt-grid-export`
- SFT: `transformers` + `peft`
- optional dataset handling: `datasets`

Do not start with RL.

Why:

- the immediate blocker is data plumbing and a supervised baseline
- the output space is tiny and structured
- most of the win should come from cleaner local measurements and better
  antisymmetry, not policy optimization

If and only if the supervised baseline is clearly promising, the next serious
RL candidate should be `verl` for larger-scale post-training. Do not make
`prime-rl` or similar libraries the first dependency unless you already know
you need rollout-heavy optimization.

## End-to-End Flow

1. Run a normal rerank with trace capture:

```bash
cargo run --bin cardinal -- rerank \
  --request rerank.request.json \
  --out rerank.response.json \
  --trace rerank.trace.jsonl
```

2. Export fine-tuning records:

```bash
cargo run --bin cardinal -- dataset-export \
  --request rerank.request.json \
  --response rerank.response.json \
  --trace rerank.trace.jsonl \
  --out gx10-train.jsonl
```

3. Smoke-test the dataset and rendering path:

```bash
python training/gx10_pairwise/train_sft.py \
  --train-data gx10-train.jsonl \
  --model YOUR_GX10_MODEL \
  --output-dir runs/gx10-sft \
  --dry-run
```

4. Run a LoRA SFT pass:

```bash
python training/gx10_pairwise/train_sft.py \
  --train-data gx10-train.jsonl \
  --model YOUR_GX10_MODEL \
  --output-dir runs/gx10-sft \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4
```

5. Export the full pair grid for replay. This is important: trace-only pairs are
not enough once the tuned judge changes the planner path.

```bash
cargo run --bin cardinal -- prompt-grid-export \
  --request rerank.request.json \
  --model gx10/local-sft \
  --out replay.prompts.jsonl
```

6. Seed the cache by running the tuned model over the full prompt grid:

```bash
python training/gx10_pairwise/seed_cache.py \
  --prompt-grid replay.prompts.jsonl \
  --model runs/gx10-sft \
  --cache-db replay.cache.sqlite
```

7. Replay the rerank fully from cache:

```bash
cargo run --bin cardinal -- rerank \
  --request rerank.request.json \
  --cache replay.cache.sqlite \
  --cache-only \
  --out replay.response.json
```

8. Compare baseline vs replay:

```bash
python training/gx10_pairwise/compare_replay.py \
  --baseline rerank.response.json \
  --candidate replay.response.json
```

Compare:

- top-k precision / recall
- comparisons attempted
- refusals
- cost to hit the same tolerated error

## Notes

- `train_sft.py` intentionally supports `--dry-run` without importing the heavy
  training libraries. Use that first.
- The exporter emits chat-style `messages` plus canonicalized targets, so a
  later structural-loss trainer can reuse the same dataset.
- The exporter snaps off-ladder float ratios onto the configured ladder before
  writing training targets.
- `seed_cache.py` can also score labeled JSONL records, not just prompt-grid
  exports, so you can inspect exact winner / ratio accuracy before replay.
