# Rerank Report

- Request hash: `2df9d30916c4e74d909113f74d5f17051acf896d5a4e6dc49d5cca84471c3496`
- Stop reason: `no_new_pairs`
- k: 3
- Global top-k error: 3.2176
- Tolerated error: 0.0800
- Comparisons used/attempted/refused/cached: 45/45/0/0
- Comparison budget: 72
- Model used: deepseek/deepseek-v4-flash
- Rater ID: live-openrouter-public_evidence_surfaces
- Latency: 335979 ms
- Provider tokens input/output/total: 23718/21154/44872
- Provider cost: $0.008822069
- RNG seed: 20260630
- Model policy: FixedPolicy(model=deepseek/deepseek-v4-flash)

## Warnings / Degraded State

- Run stopped with non-converged stop reason `no_new_pairs`; inspect uncertainty before sharing this as a settled ordering.
- Global top-k error 3.2176 exceeds tolerated error 0.0800.

## Run Status

The planner found candidate comparisons, but all eligible pairs were already known or blocked.

## Attributes

- `public_readiness` (weight 0.400) — public-readiness for a serious technical reader deciding whether this artifact is worth sharing widely
- `evidence_value` (weight 0.350) — strength of concrete, reproducible evidence rather than narrative assertion
- `operator_reproducibility` (weight 0.250) — ability for another engineer to rerun, audit, and falsify the claim from the documented commands and artifacts

## Top Entities

- evaluation_doc (rank Some(1), feasible true, u_mean 3.211, u_std 1.710, p_flip 0.451)
  - `evidence_value`: latent 1.080 ± 0.693, z 0.494, min_norm 2.080, percentile 0.583
  - `operator_reproducibility`: latent 1.328 ± 0.728, z 0.907, min_norm 2.328, percentile 0.750
  - `public_readiness`: latent 0.665 ± 0.785, z 0.349, min_norm 1.665, percentile 0.583
- worked_example (rank Some(2), feasible true, u_mean 3.136, u_std 1.620, p_flip 0.458)
  - `evidence_value`: latent 1.234 ± 0.713, z 0.782, min_norm 2.234, percentile 0.917
  - `operator_reproducibility`: latent 1.609 ± 0.733, z 1.418, min_norm 2.609, percentile 0.917
  - `public_readiness`: latent 0.447 ± 0.725, z -0.349, min_norm 1.447, percentile 0.417
- readme (rank Some(3), feasible true, u_mean 2.782, u_std 1.761, p_flip 0.500)
  - `evidence_value`: latent 0.000 ± 0.713, z -1.531, min_norm 1.000, percentile 0.083
  - `operator_reproducibility`: latent 0.947 ± 0.837, z 0.214, min_norm 1.947, percentile 0.583
  - `public_readiness`: latent 1.127 ± 0.797, z 1.831, min_norm 2.127, percentile 0.917
- prompts_doc (rank Some(4), feasible true, u_mean 2.498, u_std 1.761, p_flip 0.532)
  - `evidence_value`: latent 0.553 ± 0.712, z -0.494, min_norm 1.553, percentile 0.417
  - `operator_reproducibility`: latent 0.586 ± 0.837, z -0.442, min_norm 1.586, percentile 0.250
  - `public_readiness`: latent 0.822 ± 0.797, z 0.853, min_norm 1.822, percentile 0.750
- benchmarks_doc (rank Some(5), feasible true, u_mean 1.652, u_std 1.617, p_flip 0.631)
  - `evidence_value`: latent 1.205 ± 0.697, z 0.728, min_norm 2.205, percentile 0.750
  - `operator_reproducibility`: latent 0.712 ± 0.829, z -0.214, min_norm 1.712, percentile 0.417
  - `public_readiness`: latent 0.000 ± 0.713, z -1.785, min_norm 1.000, percentile 0.083
- model_doc (rank Some(6), feasible true, u_mean 1.236, u_std 1.714, p_flip 0.672)
  - `evidence_value`: latent 0.485 ± 0.691, z -0.621, min_norm 1.485, percentile 0.250
  - `operator_reproducibility`: latent 0.000 ± 0.738, z -1.508, min_norm 1.000, percentile 0.083
  - `public_readiness`: latent 0.402 ± 0.786, z -0.496, min_norm 1.402, percentile 0.250
