# Live OpenRouter Benchmark Receipt

This receipt is generated from real OpenRouter calls through `cardinal rerank`.
It is not produced by the synthetic evaluator.

Policy: `cost_aware_fast_deepseek_v4_flash`
Run directory: `/Users/xyra/Documents/labs/cardinal-harness/artifacts/live/openrouter-benchmark-2026-06-30/cost_aware_fast`
Cases: 3
Comparisons used: 153
Cached comparisons: 0
Refusals: 0
Provider input tokens: 82739
Provider output tokens: 72716
Provider cost: $0.029860
Trace cost estimate rows: 0

| Case | Model used | Stop | Used/Budget | Cached | Refused | Cost USD | Top entities |
|---|---|---|---:|---:|---:|---:|---|
| `public_evidence_surfaces` | `deepseek/deepseek-v4-flash` | `no_new_pairs` | 45/72 | 0 | 0 | $0.008822 | evaluation_doc, worked_example, readme |
| `model_policy_live_routing` | `deepseek/deepseek-v4-flash` | `no_new_pairs` | 45/72 | 0 | 0 | $0.006776 | frontier_ladder_2026_06, quality_only_opus_46 |
| `public_release_risks` | `deepseek/deepseek-v4-flash` | `no_new_pairs` | 63/84 | 0 | 0 | $0.014263 | no_large_live_receipt, cache_provenance, provider_metadata_drift |

## Interpretation guardrails

- `comparisons_cached = 0` means this run made fresh provider calls for every comparison in the receipt.
- A budget-exhausted stop is still a valid receipt; it means the run spent the configured live-call budget before proving the tolerated top-k error bound.
- This suite tests real provider integration, parsing, trace/cost accounting, and project-relevant ranking surfaces. It does not by itself prove global superiority over every scalar or ordinal baseline.
