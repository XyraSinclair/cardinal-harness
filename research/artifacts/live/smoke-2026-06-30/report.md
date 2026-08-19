# Rerank Report

- Request hash: `4784a8080abd967a2a85ae0dfd49d42c0976906f9647eb4a6cbf1964c29c46bf`
- Stop reason: `budget_exhausted`
- k: 2
- Global top-k error: 1.0000
- Tolerated error: 0.1000
- Comparisons used/attempted/refused/cached: 2/2/0/0
- Comparison budget: 2
- Model used: openai/gpt-5.4-mini
- Rater ID: docs-example
- Latency: 2952 ms
- Provider tokens input/output/total: 723/52/775
- Provider cost: $0.000776250
- RNG seed: 20260630

## Warnings / Degraded State

- Run stopped with non-converged stop reason `budget_exhausted`; inspect uncertainty before sharing this as a settled ordering.
- Global top-k error 1.0000 exceeds tolerated error 0.1000.

## Run Status

The run used the configured comparison budget before meeting the stopping tolerance; inspect the top-k error before treating the frontier as settled.

## Attributes

- `clarity` (weight 0.600) — clarity of explanation
- `evidence` (weight 0.400) — strength of concrete evidence and examples

## Top Entities

- essay_b (rank Some(1), feasible true, u_mean 4.671, u_std 400000.000, p_flip 0.500)
  - `clarity`: latent 0.742 ± 1.145, z 4.576, min_norm 1.742, percentile 0.833
  - `evidence`: latent 0.000 ± 1.000, z 0.000, min_norm 1.000, percentile 0.500
- essay_c (rank Some(2), feasible true, u_mean 0.600, u_std 400000.000, p_flip 0.500)
  - `clarity`: latent 0.095 ± 1.492, z 0.000, min_norm 1.095, percentile 0.500
  - `evidence`: latent 0.000 ± 1.000, z 0.000, min_norm 1.000, percentile 0.833
- essay_a (rank Some(3), feasible true, u_mean 0.000, u_std 400000.000, p_flip 0.500)
  - `clarity`: latent 0.000 ± 1.492, z -0.674, min_norm 1.000, percentile 0.167
  - `evidence`: latent 0.000 ± 1.000, z 0.000, min_norm 1.000, percentile 0.167
