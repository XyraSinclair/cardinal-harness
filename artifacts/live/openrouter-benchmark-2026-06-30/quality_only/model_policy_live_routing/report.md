# Rerank Report

- Request hash: `c542293c728ac68058401baa248bed53ccad89c2c3c096a29e362c1c8b563ef3`
- Stop reason: `no_new_pairs`
- k: 2
- Global top-k error: 2.9279
- Tolerated error: 0.0800
- Comparisons used/attempted/refused/cached: 45/45/0/0
- Comparison budget: 72
- Model used: anthropic/claude-opus-4.6
- Rater ID: live-openrouter-model_policy_live_routing
- Latency: 40922 ms
- Provider tokens input/output/total: 22650/1220/23870
- Provider cost: $0.143750000
- RNG seed: 20260630
- Model policy: FixedPolicy(model=anthropic/claude-opus-4.6)

## Warnings / Degraded State

- Run stopped with non-converged stop reason `no_new_pairs`; inspect uncertainty before sharing this as a settled ordering.
- Global top-k error 2.9279 exceeds tolerated error 0.0800.

## Run Status

The planner found candidate comparisons, but all eligible pairs were already known or blocked.

## Attributes

- `public_benchmark_reliability` (weight 0.450) — reliability as a judge for public benchmark receipts where brittle or unserious judgments would damage credibility
- `cost_discipline` (weight 0.300) — cost discipline for repeated real OpenRouter benchmark runs without sacrificing the validity of the evidence
- `freshness_and_capability` (weight 0.250) — current-generation capability and parameter support fit for cardinal pairwise ratio judging on OpenRouter

## Top Entities

- frontier_ladder_2026_06 (rank Some(1), feasible true, u_mean 3.044, u_std 1.136, p_flip 0.317)
  - `cost_discipline`: latent 2.030 ± 0.954, z 0.422, min_norm 3.030, percentile 0.583
  - `freshness_and_capability`: latent 1.145 ± 1.142, z 0.805, min_norm 2.145, percentile 0.917
  - `public_benchmark_reliability`: latent 1.405 ± 0.771, z 0.667, min_norm 2.405, percentile 0.750
- qwen_3_7_max (rank Some(2), feasible true, u_mean 1.917, u_std 1.232, p_flip 0.500)
  - `cost_discipline`: latent 1.393 ± 0.795, z -0.422, min_norm 2.393, percentile 0.417
  - `freshness_and_capability`: latent 0.687 ± 1.295, z -0.065, min_norm 1.687, percentile 0.417
  - `public_benchmark_reliability`: latent 0.826 ± 0.919, z 0.022, min_norm 1.826, percentile 0.583
- quality_only_opus_46 (rank Some(3), feasible true, u_mean 1.891, u_std 1.270, p_flip 0.504)
  - `cost_discipline`: latent 0.000 ± 0.954, z -2.265, min_norm 1.000, percentile 0.083
  - `freshness_and_capability`: latent 1.007 ± 1.295, z 0.544, min_norm 2.007, percentile 0.750
  - `public_benchmark_reliability`: latent 1.592 ± 0.919, z 0.874, min_norm 2.592, percentile 0.917
- kimi_k2_thinking (rank Some(4), feasible true, u_mean 1.798, u_std 1.212, p_flip 0.519)
  - `cost_discipline`: latent 1.159 ± 0.800, z -0.730, min_norm 2.159, percentile 0.250
  - `freshness_and_capability`: latent 0.755 ± 1.286, z 0.065, min_norm 1.755, percentile 0.583
  - `public_benchmark_reliability`: latent 0.786 ± 0.879, z -0.022, min_norm 1.786, percentile 0.417
- cost_aware_deepseek_v4_flash (rank Some(5), feasible true, u_mean 1.669, u_std 1.125, p_flip 0.542)
  - `cost_discipline`: latent 2.262 ± 0.856, z 0.728, min_norm 3.262, percentile 0.917
  - `freshness_and_capability`: latent 0.277 ± 1.141, z -0.844, min_norm 1.277, percentile 0.250
  - `public_benchmark_reliability`: latent 0.192 ± 0.815, z -0.682, min_norm 1.192, percentile 0.250
- glm_5_2 (rank Some(6), feasible true, u_mean 1.283, u_std 1.103, p_flip 0.607)
  - `cost_discipline`: latent 2.181 ± 0.801, z 0.621, min_norm 3.181, percentile 0.750
  - `freshness_and_capability`: latent 0.000 ± 1.096, z -1.371, min_norm 1.000, percentile 0.083
  - `public_benchmark_reliability`: latent 0.000 ± 0.850, z -0.896, min_norm 1.000, percentile 0.083
