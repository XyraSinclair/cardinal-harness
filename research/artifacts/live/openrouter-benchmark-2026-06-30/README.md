# Live OpenRouter Benchmark Pack — 2026-06-30

This directory is a preserved real-provider receipt pack for cardinal-harness. It was generated through `examples/live_openrouter_benchmark.py`, which shells out to `cardinal rerank` with `OPENROUTER_API_KEY` and exports each run's response, trace, report, and cache rows.

## Aggregate receipt

- Policy runs: 3
- Case runs: 9
- Fresh provider comparisons used: 459
- Cached comparisons: 0
- Refusals: 0
- Provider input tokens: 234299
- Provider output tokens: 80983
- Provider cost: $0.994335
- Trace rows using estimated local fallback cost: 0

## Policy run summary

| Policy | Comparisons | Cached | Refused | Cost USD | Top public evidence | Top routing policy | Top release risks |
|---|---:|---:|---:|---:|---|---|---|
| `quality_only_opus_46` | 153 | 0 | 0 | $0.482300 | worked_example, readme, evaluation_doc | frontier_ladder_2026_06, qwen_3_7_max | no_large_live_receipt, baseline_breadth, provider_metadata_drift |
| `frontier_ladder_2026_06` | 153 | 0 | 0 | $0.482175 | readme, worked_example, evaluation_doc | frontier_ladder_2026_06, qwen_3_7_max | no_large_live_receipt, baseline_breadth, provider_metadata_drift |
| `cost_aware_fast_deepseek_v4_flash` | 153 | 0 | 0 | $0.029860 | evaluation_doc, worked_example, readme | frontier_ladder_2026_06, quality_only_opus_46 | no_large_live_receipt, cache_provenance, provider_metadata_drift |

## What this proves

- The OpenRouter gateway, current policy JSON, prompt parsing, trace writing, markdown report generation, and cache export all work against live provider traffic on the three preserved case families.
- The quality-only and frontier-ladder receipts agree that the largest release gap is `no_large_live_receipt`, and that `frontier_ladder_2026_06` is the strongest routing policy candidate in the tested set.
- The cheaper DeepSeek route completed the same case families with zero cache hits and zero refusals for about three cents, making it useful for smoke and iteration runs.

## What this does not prove

- It is not a universal benchmark against scalar ratings or all ordinal methods.
- It does not replace a larger task-family suite with external baselines and held-out prompts.
- It does not freeze OpenRouter model availability, pricing, or parameter support; refresh model metadata before reusing these policy files for a new public claim.
