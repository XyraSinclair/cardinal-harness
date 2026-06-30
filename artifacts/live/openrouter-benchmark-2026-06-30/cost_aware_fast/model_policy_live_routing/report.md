# Rerank Report

- Request hash: `c542293c728ac68058401baa248bed53ccad89c2c3c096a29e362c1c8b563ef3`
- Stop reason: `no_new_pairs`
- k: 2
- Global top-k error: 2.8784
- Tolerated error: 0.0800
- Comparisons used/attempted/refused/cached: 45/45/0/0
- Comparison budget: 72
- Model used: deepseek/deepseek-v4-flash
- Rater ID: live-openrouter-model_policy_live_routing
- Latency: 84518 ms
- Provider tokens input/output/total: 24085/14294/38379
- Provider cost: $0.006775553
- RNG seed: 20260630
- Model policy: FixedPolicy(model=deepseek/deepseek-v4-flash)

## Warnings / Degraded State

- Run stopped with non-converged stop reason `no_new_pairs`; inspect uncertainty before sharing this as a settled ordering.
- Global top-k error 2.8784 exceeds tolerated error 0.0800.

## Run Status

The planner found candidate comparisons, but all eligible pairs were already known or blocked.

## Attributes

- `public_benchmark_reliability` (weight 0.450) — reliability as a judge for public benchmark receipts where brittle or unserious judgments would damage credibility
- `cost_discipline` (weight 0.300) — cost discipline for repeated real OpenRouter benchmark runs without sacrificing the validity of the evidence
- `freshness_and_capability` (weight 0.250) — current-generation capability and parameter support fit for cardinal pairwise ratio judging on OpenRouter

## Top Entities

- frontier_ladder_2026_06 (rank Some(1), feasible true, u_mean 3.029, u_std 1.755, p_flip 0.433)
  - `cost_discipline`: latent 0.510 ± 0.703, z 0.535, min_norm 1.510, percentile 0.750
  - `freshness_and_capability`: latent 0.776 ± 0.823, z 1.034, min_norm 1.776, percentile 0.917
  - `public_benchmark_reliability`: latent 1.082 ± 0.779, z 0.566, min_norm 2.082, percentile 0.750
- quality_only_opus_46 (rank Some(2), feasible true, u_mean 2.427, u_std 1.818, p_flip 0.500)
  - `cost_discipline`: latent 0.238 ± 0.749, z -0.416, min_norm 1.238, percentile 0.417
  - `freshness_and_capability`: latent 0.365 ± 0.843, z -0.341, min_norm 1.365, percentile 0.417
  - `public_benchmark_reliability`: latent 1.363 ± 0.787, z 1.061, min_norm 2.363, percentile 0.917
- glm_5_2 (rank Some(3), feasible true, u_mean 1.819, u_std 1.803, p_flip 0.567)
  - `cost_discipline`: latent 0.476 ± 0.749, z 0.416, min_norm 1.476, percentile 0.583
  - `freshness_and_capability`: latent 0.569 ± 0.837, z 0.341, min_norm 1.569, percentile 0.583
  - `public_benchmark_reliability`: latent 0.317 ± 0.768, z -0.783, min_norm 1.317, percentile 0.250
- qwen_3_7_max (rank Some(4), feasible true, u_mean 1.807, u_std 1.778, p_flip 0.568)
  - `cost_discipline`: latent 0.000 ± 0.724, z -1.247, min_norm 1.000, percentile 0.083
  - `freshness_and_capability`: latent 0.715 ± 0.821, z 0.832, min_norm 1.715, percentile 0.750
  - `public_benchmark_reliability`: latent 0.780 ± 0.787, z 0.033, min_norm 1.780, percentile 0.583
- kimi_k2_thinking (rank Some(5), feasible true, u_mean 1.291, u_std 1.732, p_flip 0.625)
  - `cost_discipline`: latent 0.019 ± 0.693, z -1.181, min_norm 1.019, percentile 0.250
  - `freshness_and_capability`: latent 0.313 ± 0.843, z -0.517, min_norm 1.313, percentile 0.250
  - `public_benchmark_reliability`: latent 0.742 ± 0.735, z -0.033, min_norm 1.742, percentile 0.417
- cost_aware_deepseek_v4_flash (rank Some(6), feasible true, u_mean 0.917, u_std 1.642, p_flip 0.669)
  - `cost_discipline`: latent 0.590 ± 0.686, z 0.814, min_norm 1.590, percentile 0.917
  - `freshness_and_capability`: latent 0.000 ± 0.754, z -1.566, min_norm 1.000, percentile 0.083
  - `public_benchmark_reliability`: latent 0.000 ± 0.703, z -1.342, min_norm 1.000, percentile 0.083
