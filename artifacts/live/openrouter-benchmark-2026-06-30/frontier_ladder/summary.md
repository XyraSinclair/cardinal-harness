# Live OpenRouter Benchmark Receipt

This receipt is generated from real OpenRouter calls through `cardinal rerank`.
It is not produced by the synthetic evaluator.

Policy: `frontier_ladder_2026_06`
Run directory: `/Users/xyra/Documents/labs/cardinal-harness/artifacts/live/openrouter-benchmark-2026-06-30/frontier_ladder`
Cases: 3
Comparisons used: 153
Cached comparisons: 0
Refusals: 0
Provider input tokens: 75780
Provider output tokens: 4131
Provider cost: $0.482175
Trace cost estimate rows: 0

| Case | Model used | Stop | Used/Budget | Cached | Refused | Cost USD | Top entities |
|---|---|---|---:|---:|---:|---:|---|
| `public_evidence_surfaces` | `anthropic/claude-opus-4.6` | `no_new_pairs` | 45/72 | 0 | 0 | $0.143400 | readme, worked_example, evaluation_doc |
| `model_policy_live_routing` | `anthropic/claude-opus-4.6` | `no_new_pairs` | 45/72 | 0 | 0 | $0.143625 | frontier_ladder_2026_06, qwen_3_7_max |
| `public_release_risks` | `anthropic/claude-opus-4.6` | `no_new_pairs` | 63/84 | 0 | 0 | $0.195150 | no_large_live_receipt, baseline_breadth, provider_metadata_drift |

## Interpretation guardrails

- `comparisons_cached = 0` means this run made fresh provider calls for every comparison in the receipt.
- A budget-exhausted stop is still a valid receipt; it means the run spent the configured live-call budget before proving the tolerated top-k error bound.
- This suite tests real provider integration, parsing, trace/cost accounting, and project-relevant ranking surfaces. It does not by itself prove global superiority over every scalar or ordinal baseline.
