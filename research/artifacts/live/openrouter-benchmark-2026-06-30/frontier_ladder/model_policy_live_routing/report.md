# Rerank Report

- Request hash: `c542293c728ac68058401baa248bed53ccad89c2c3c096a29e362c1c8b563ef3`
- Stop reason: `no_new_pairs`
- k: 2
- Global top-k error: 2.9548
- Tolerated error: 0.0800
- Comparisons used/attempted/refused/cached: 45/45/0/0
- Comparison budget: 72
- Model used: anthropic/claude-opus-4.6
- Rater ID: live-openrouter-model_policy_live_routing
- Latency: 27442 ms
- Provider tokens input/output/total: 22650/1215/23865
- Provider cost: $0.143625000
- RNG seed: 20260630
- Model policy: ModelLadderPolicy(high=anthropic/claude-opus-4.6, mid=google/gemini-3.1-pro-preview, low=openai/gpt-5.4-mini, switch_error=0.1, similarity_ln_ratio=0.12, max_pair_std=0.6, min_comparisons=12)

## Warnings / Degraded State

- Run stopped with non-converged stop reason `no_new_pairs`; inspect uncertainty before sharing this as a settled ordering.
- Global top-k error 2.9548 exceeds tolerated error 0.0800.

## Run Status

The planner found candidate comparisons, but all eligible pairs were already known or blocked.

## Attributes

- `public_benchmark_reliability` (weight 0.450) — reliability as a judge for public benchmark receipts where brittle or unserious judgments would damage credibility
- `cost_discipline` (weight 0.300) — cost discipline for repeated real OpenRouter benchmark runs without sacrificing the validity of the evidence
- `freshness_and_capability` (weight 0.250) — current-generation capability and parameter support fit for cardinal pairwise ratio judging on OpenRouter

## Top Entities

- frontier_ladder_2026_06 (rank Some(1), feasible true, u_mean 3.296, u_std 1.242, p_flip 0.315)
  - `cost_discipline`: latent 2.013 ± 0.811, z 0.440, min_norm 3.013, percentile 0.583
  - `freshness_and_capability`: latent 1.028 ± 1.147, z 0.773, min_norm 2.028, percentile 0.917
  - `public_benchmark_reliability`: latent 1.582 ± 0.756, z 0.762, min_norm 2.582, percentile 0.750
- qwen_3_7_max (rank Some(2), feasible true, u_mean 2.101, u_std 1.242, p_flip 0.500)
  - `cost_discipline`: latent 1.406 ± 0.768, z -0.440, min_norm 2.406, percentile 0.417
  - `freshness_and_capability`: latent 0.636 ± 1.104, z -0.146, min_norm 1.636, percentile 0.417
  - `public_benchmark_reliability`: latent 0.921 ± 0.881, z 0.068, min_norm 1.921, percentile 0.583
- cost_aware_deepseek_v4_flash (rank Some(3), feasible true, u_mean 2.020, u_std 1.196, p_flip 0.513)
  - `cost_discipline`: latent 2.508 ± 0.750, z 1.155, min_norm 3.508, percentile 0.917
  - `freshness_and_capability`: latent 0.228 ± 1.081, z -1.100, min_norm 1.228, percentile 0.250
  - `public_benchmark_reliability`: latent 0.297 ± 0.805, z -0.587, min_norm 1.297, percentile 0.250
- kimi_k2_thinking (rank Some(4), feasible true, u_mean 1.991, u_std 1.275, p_flip 0.517)
  - `cost_discipline`: latent 1.208 ± 0.749, z -0.725, min_norm 2.208, percentile 0.250
  - `freshness_and_capability`: latent 0.760 ± 1.175, z 0.146, min_norm 1.760, percentile 0.583
  - `public_benchmark_reliability`: latent 0.790 ± 0.850, z -0.068, min_norm 1.790, percentile 0.417
- quality_only_opus_46 (rank Some(5), feasible true, u_mean 1.986, u_std 1.301, p_flip 0.518)
  - `cost_discipline`: latent 0.000 ± 0.811, z -2.473, min_norm 1.000, percentile 0.083
  - `freshness_and_capability`: latent 0.944 ± 1.175, z 0.575, min_norm 1.944, percentile 0.750
  - `public_benchmark_reliability`: latent 1.667 ± 0.881, z 0.852, min_norm 2.667, percentile 0.917
- glm_5_2 (rank Some(6), feasible true, u_mean 1.377, u_std 1.188, p_flip 0.617)
  - `cost_discipline`: latent 2.141 ± 0.776, z 0.624, min_norm 3.141, percentile 0.750
  - `freshness_and_capability`: latent 0.000 ± 1.044, z -1.634, min_norm 1.000, percentile 0.083
  - `public_benchmark_reliability`: latent 0.000 ± 0.834, z -0.898, min_norm 1.000, percentile 0.083
