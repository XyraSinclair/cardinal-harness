# Rerank Report

- Request hash: `2df9d30916c4e74d909113f74d5f17051acf896d5a4e6dc49d5cca84471c3496`
- Stop reason: `no_new_pairs`
- k: 3
- Global top-k error: 3.7149
- Tolerated error: 0.0800
- Comparisons used/attempted/refused/cached: 45/45/0/0
- Comparison budget: 72
- Model used: anthropic/claude-opus-4.6
- Rater ID: live-openrouter-public_evidence_surfaces
- Latency: 41307 ms
- Provider tokens input/output/total: 22605/1215/23820
- Provider cost: $0.143400000
- RNG seed: 20260630
- Model policy: ModelLadderPolicy(high=anthropic/claude-opus-4.6, mid=google/gemini-3.1-pro-preview, low=openai/gpt-5.4-mini, switch_error=0.1, similarity_ln_ratio=0.12, max_pair_std=0.6, min_comparisons=12)

## Warnings / Degraded State

- Run stopped with non-converged stop reason `no_new_pairs`; inspect uncertainty before sharing this as a settled ordering.
- Global top-k error 3.7149 exceeds tolerated error 0.0800.

## Run Status

The planner found candidate comparisons, but all eligible pairs were already known or blocked.

## Attributes

- `public_readiness` (weight 0.400) — public-readiness for a serious technical reader deciding whether this artifact is worth sharing widely
- `evidence_value` (weight 0.350) — strength of concrete, reproducible evidence rather than narrative assertion
- `operator_reproducibility` (weight 0.250) — ability for another engineer to rerun, audit, and falsify the claim from the documented commands and artifacts

## Top Entities

- readme (rank Some(1), feasible true, u_mean 3.672, u_std 3.179, p_flip 0.420)
  - `evidence_value`: latent 0.000 ± 0.789, z -1.338, min_norm 1.000, percentile 0.083
  - `operator_reproducibility`: latent 0.424 ± 0.795, z -0.727, min_norm 1.424, percentile 0.250
  - `public_readiness`: latent 0.973 ± 0.837, z 3.996, min_norm 1.973, percentile 0.917
- worked_example (rank Some(2), feasible true, u_mean 3.159, u_std 3.033, p_flip 0.451)
  - `evidence_value`: latent 0.919 ± 0.771, z 0.813, min_norm 1.919, percentile 0.917
  - `operator_reproducibility`: latent 1.711 ± 0.725, z 0.622, min_norm 2.711, percentile 0.750
  - `public_readiness`: latent 0.382 ± 0.797, z 0.406, min_norm 1.382, percentile 0.583
- evaluation_doc (rank Some(3), feasible true, u_mean 2.418, u_std 2.995, p_flip 0.500)
  - `evidence_value`: latent 0.769 ± 0.769, z 0.463, min_norm 1.769, percentile 0.583
  - `operator_reproducibility`: latent 1.817 ± 0.746, z 0.733, min_norm 2.817, percentile 0.917
  - `public_readiness`: latent 0.216 ± 0.786, z -0.607, min_norm 1.216, percentile 0.250
- model_doc (rank Some(4), feasible true, u_mean 1.902, u_std 3.178, p_flip 0.533)
  - `evidence_value`: latent 0.267 ± 0.789, z -0.712, min_norm 1.267, percentile 0.250
  - `operator_reproducibility`: latent 0.000 ± 0.771, z -1.172, min_norm 1.000, percentile 0.083
  - `public_readiness`: latent 0.438 ± 0.837, z 0.742, min_norm 1.438, percentile 0.750
- prompts_doc (rank Some(5), feasible true, u_mean 1.665, u_std 3.083, p_flip 0.549)
  - `evidence_value`: latent 0.374 ± 0.779, z -0.463, min_norm 1.374, percentile 0.417
  - `operator_reproducibility`: latent 0.809 ± 0.773, z -0.323, min_norm 1.809, percentile 0.417
  - `public_readiness`: latent 0.249 ± 0.810, z -0.406, min_norm 1.249, percentile 0.417
- benchmarks_doc (rank Some(6), feasible true, u_mean 1.579, u_std 3.023, p_flip 0.555)
  - `evidence_value`: latent 0.843 ± 0.759, z 0.637, min_norm 1.843, percentile 0.750
  - `operator_reproducibility`: latent 1.426 ± 0.795, z 0.323, min_norm 2.426, percentile 0.583
  - `public_readiness`: latent 0.000 ± 0.794, z -1.918, min_norm 1.000, percentile 0.083
