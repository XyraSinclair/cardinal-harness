# Rerank Report

- Request hash: `1719e25c45d22883177defb878b6382c67b3628ff5d701ff6f4dc22260d86441`
- Stop reason: `no_new_pairs`
- k: 3
- Global top-k error: 3.8491
- Tolerated error: 0.0800
- Comparisons used/attempted/refused/cached: 63/63/0/0
- Comparison budget: 84
- Model used: anthropic/claude-opus-4.6
- Rater ID: live-openrouter-public_release_risks
- Latency: 46102 ms
- Provider tokens input/output/total: 30525/1701/32226
- Provider cost: $0.195150000
- RNG seed: 20260630
- Model policy: FixedPolicy(model=anthropic/claude-opus-4.6)

## Warnings / Degraded State

- Run stopped with non-converged stop reason `no_new_pairs`; inspect uncertainty before sharing this as a settled ordering.
- Global top-k error 3.8491 exceeds tolerated error 0.0800.

## Run Status

The planner found candidate comparisons, but all eligible pairs were already known or blocked.

## Attributes

- `credibility_risk` (weight 0.450) — risk to public credibility if cardinal-harness is shared widely as a serious empirical artifact
- `falsifiability_gap` (weight 0.350) — how much this issue blocks concrete falsifiable live evidence rather than merely improving polish
- `fix_leverage` (weight 0.200) — leverage of fixing this issue for making the project safer to share widely

## Top Entities

- no_large_live_receipt (rank Some(1), feasible true, u_mean 5.222, u_std 2.384, p_flip 0.232)
  - `credibility_risk`: latent 0.817 ± 0.785, z 1.987, min_norm 1.817, percentile 0.929
  - `falsifiability_gap`: latent 1.538 ± 0.784, z 3.481, min_norm 2.538, percentile 0.929
  - `fix_leverage`: latent 0.639 ± 0.825, z 0.000, min_norm 1.639, percentile 0.500
- baseline_breadth (rank Some(2), feasible true, u_mean 3.904, u_std 2.383, p_flip 0.327)
  - `credibility_risk`: latent 0.780 ± 0.785, z 1.865, min_norm 1.780, percentile 0.786
  - `falsifiability_gap`: latent 0.997 ± 0.784, z 1.483, min_norm 1.997, percentile 0.786
  - `fix_leverage`: latent 0.351 ± 0.825, z -0.674, min_norm 1.351, percentile 0.357
- provider_metadata_drift (rank Some(3), feasible true, u_mean 1.812, u_std 2.277, p_flip 0.500)
  - `credibility_risk`: latent 0.225 ± 0.749, z 0.000, min_norm 1.225, percentile 0.500
  - `falsifiability_gap`: latent 0.682 ± 0.750, z 0.320, min_norm 1.682, percentile 0.643
  - `fix_leverage`: latent 0.000 ± 0.787, z -1.495, min_norm 1.000, percentile 0.071
- cost_estimate_semantics (rank Some(4), feasible true, u_mean 1.733, u_std 2.281, p_flip 0.507)
  - `credibility_risk`: latent 0.257 ± 0.774, z 0.109, min_norm 1.257, percentile 0.643
  - `falsifiability_gap`: latent 0.540 ± 0.718, z -0.207, min_norm 1.540, percentile 0.357
  - `fix_leverage`: latent 0.175 ± 0.795, z -1.086, min_norm 1.175, percentile 0.214
- cache_provenance (rank Some(5), feasible true, u_mean 1.698, u_std 2.306, p_flip 0.510)
  - `credibility_risk`: latent 0.024 ± 0.775, z -0.674, min_norm 1.024, percentile 0.214
  - `falsifiability_gap`: latent 0.596 ± 0.734, z 0.000, min_norm 1.596, percentile 0.500
  - `fix_leverage`: latent 0.724 ± 0.813, z 0.198, min_norm 1.724, percentile 0.786
- api_surface_complexity (rank Some(6), feasible true, u_mean 1.549, u_std 2.312, p_flip 0.523)
  - `credibility_risk`: latent 0.040 ± 0.785, z -0.623, min_norm 1.040, percentile 0.357
  - `falsifiability_gap`: latent 0.413 ± 0.725, z -0.674, min_norm 1.413, percentile 0.214
  - `fix_leverage`: latent 0.963 ± 0.810, z 0.758, min_norm 1.963, percentile 0.929
- dependency_audit_warnings (rank Some(7), feasible true, u_mean 0.444, u_std 2.333, p_flip 0.617)
  - `credibility_risk`: latent 0.000 ± 0.770, z -0.755, min_norm 1.000, percentile 0.071
  - `falsifiability_gap`: latent 0.000 ± 0.764, z -2.201, min_norm 1.000, percentile 0.071
  - `fix_leverage`: latent 0.640 ± 0.809, z 0.001, min_norm 1.640, percentile 0.643
