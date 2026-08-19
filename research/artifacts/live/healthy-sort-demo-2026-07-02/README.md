# Live healthy-elicitation demo receipt — 2026-07-02

A real, unedited run of the healthy sort path: both-orders counterbalancing
(default), the opposite side of the attribute (`--two-sided`), and a
paraphrase probe (`--also-by`), against `anthropic/claude-sonnet-4.6`.

## Command

```bash
export OPENROUTER_API_KEY=your_key_here

cargo run --release --bin cardinal -- sort examples/sort-demo.txt \
  --by "usefulness as advice for a software engineer" \
  --two-sided \
  --also-by "how much practical value it offers someone building software" \
  --model anthropic/claude-sonnet-4.6 \
  --seed 7 --budget 120 \
  --cache artifacts/live/healthy-sort-demo-2026-07-02/cache.sqlite \
  --trace artifacts/live/healthy-sort-demo-2026-07-02/trace.jsonl \
  --format json > artifacts/live/healthy-sort-demo-2026-07-02/output.json
```

## What the receipts measured, live

```text
sorted 8 items · 120 comparisons (18 cached, 0 refused) · $0.3441 · order flips: 11/51 · stop: budget_exhausted
probe [opposite]   "lack of usefulness as advice for a software engineer": consistency +0.81 — consistent
probe [paraphrase] "how much practical value it offers someone building software": consistency +0.35 — shaky
```

Three findings this run would have silently absorbed without the receipts:

1. **Position bias is real and measurable**: the judge reversed direction on
   **11 of 51 counterbalanced pairs (21.6%)** when A/B presentation was
   swapped. Counterbalancing cancels this per-pair; the flip rate tells you
   how much the judge's answers depended on ordering rather than content.
2. **The attribute is two-sided coherent** (+0.81): scoring "usefulness" and
   "lack of usefulness" produced consistent (mirrored) rankings, so the
   criterion means something stable to this judge.
3. **The paraphrase is NOT interchangeable** (+0.35): rephrasing the criterion
   as "practical value for someone building software" shifted the ranking
   substantially. Anyone annotating with a single phrasing would never see
   this.

Costs and caveats: 41,504 input tokens, $0.344127 provider-reported. The run
stopped at `budget_exhausted`; probes and counterbalancing spend more of the
budget on health measurement and less on depth — that is the explicit trade.

## Files

| File | Contents |
|------|----------|
| `output.json` | Items with combined-utility scores, meta (flips, tokens, cost), probe consistencies |
| `trace.jsonl` | All 120 comparison events incl. presentation order (`swapped`) per call |
| `cache-export.jsonl` | Cached pairwise judgements (portable form of the gitignored `cache.sqlite`) |
| `stderr.txt` | The run summary and probe lines exactly as printed |
