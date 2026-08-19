# Rerank Report

- Request hash: `2df9d30916c4e74d909113f74d5f17051acf896d5a4e6dc49d5cca84471c3496`
- Stop reason: `no_new_pairs`
- k: 3
- Global top-k error: 3.5843
- Tolerated error: 0.0800
- Comparisons used/attempted/refused/cached: 45/45/0/0
- Comparison budget: 72
- Model used: anthropic/claude-opus-4.6
- Rater ID: live-openrouter-public_evidence_surfaces
- Latency: 35439 ms
- Provider tokens input/output/total: 22605/1215/23820
- Provider cost: $0.143400000
- RNG seed: 20260630
- Model policy: FixedPolicy(model=anthropic/claude-opus-4.6)

## Warnings / Degraded State

- Run stopped with non-converged stop reason `no_new_pairs`; inspect uncertainty before sharing this as a settled ordering.
- Global top-k error 3.5843 exceeds tolerated error 0.0800.

## Run Status

The planner found candidate comparisons, but all eligible pairs were already known or blocked.

## Attributes

- `public_readiness` (weight 0.400) — public-readiness for a serious technical reader deciding whether this artifact is worth sharing widely
- `evidence_value` (weight 0.350) — strength of concrete, reproducible evidence rather than narrative assertion
- `operator_reproducibility` (weight 0.250) — ability for another engineer to rerun, audit, and falsify the claim from the documented commands and artifacts

## Top Entities

- worked_example (rank Some(1), feasible true, u_mean 3.412, u_std 2.839, p_flip 0.460)
  - `evidence_value`: latent 0.789 ± 0.796, z 0.747, min_norm 1.789, percentile 0.917
  - `operator_reproducibility`: latent 1.703 ± 0.742, z 0.684, min_norm 2.703, percentile 0.917
  - `public_readiness`: latent 0.461 ± 0.809, z 0.714, min_norm 1.461, percentile 0.750
- readme (rank Some(2), feasible true, u_mean 3.013, u_std 2.954, p_flip 0.488)
  - `evidence_value`: latent 0.000 ± 0.861, z -1.722, min_norm 1.000, percentile 0.083
  - `operator_reproducibility`: latent 0.243 ± 0.801, z -0.939, min_norm 1.243, percentile 0.250
  - `public_readiness`: latent 0.939 ± 0.832, z 3.215, min_norm 1.939, percentile 0.917
- evaluation_doc (rank Some(3), feasible true, u_mean 2.839, u_std 2.844, p_flip 0.500)
  - `evidence_value`: latent 0.701 ± 0.793, z 0.470, min_norm 1.701, percentile 0.583
  - `operator_reproducibility`: latent 1.685 ± 0.776, z 0.665, min_norm 2.685, percentile 0.750
  - `public_readiness`: latent 0.325 ± 0.811, z 0.002, min_norm 1.325, percentile 0.583
- benchmarks_doc (rank Some(4), feasible true, u_mean 1.740, u_std 2.866, p_flip 0.576)
  - `evidence_value`: latent 0.743 ± 0.794, z 0.602, min_norm 1.743, percentile 0.750
  - `operator_reproducibility`: latent 1.294 ± 0.801, z 0.230, min_norm 2.294, percentile 0.583
  - `public_readiness`: latent 0.000 ± 0.819, z -1.696, min_norm 1.000, percentile 0.083
- prompts_doc (rank Some(5), feasible true, u_mean 1.642, u_std 2.865, p_flip 0.583)
  - `evidence_value`: latent 0.400 ± 0.821, z -0.470, min_norm 1.400, percentile 0.417
  - `operator_reproducibility`: latent 0.880 ± 0.801, z -0.230, min_norm 1.880, percentile 0.417
  - `public_readiness`: latent 0.203 ± 0.811, z -0.635, min_norm 1.203, percentile 0.250
- model_doc (rank Some(6), feasible true, u_mean 1.411, u_std 2.953, p_flip 0.597)
  - `evidence_value`: latent 0.251 ± 0.861, z -0.938, min_norm 1.251, percentile 0.250
  - `operator_reproducibility`: latent 0.000 ± 0.797, z -1.209, min_norm 1.000, percentile 0.083
  - `public_readiness`: latent 0.324 ± 0.832, z -0.002, min_norm 1.324, percentile 0.417
