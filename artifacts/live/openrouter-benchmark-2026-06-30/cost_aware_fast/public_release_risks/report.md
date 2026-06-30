# Rerank Report

- Request hash: `1719e25c45d22883177defb878b6382c67b3628ff5d701ff6f4dc22260d86441`
- Stop reason: `no_new_pairs`
- k: 3
- Global top-k error: 4.6365
- Tolerated error: 0.0800
- Comparisons used/attempted/refused/cached: 63/63/0/0
- Comparison budget: 84
- Model used: deepseek/deepseek-v4-flash
- Rater ID: live-openrouter-public_release_risks
- Latency: 208420 ms
- Provider tokens input/output/total: 34936/37268/72204
- Provider cost: $0.014262637
- RNG seed: 20260630
- Model policy: FixedPolicy(model=deepseek/deepseek-v4-flash)

## Warnings / Degraded State

- Run stopped with non-converged stop reason `no_new_pairs`; inspect uncertainty before sharing this as a settled ordering.
- Global top-k error 4.6365 exceeds tolerated error 0.0800.

## Run Status

The planner found candidate comparisons, but all eligible pairs were already known or blocked.

## Attributes

- `credibility_risk` (weight 0.450) — risk to public credibility if cardinal-harness is shared widely as a serious empirical artifact
- `falsifiability_gap` (weight 0.350) — how much this issue blocks concrete falsifiable live evidence rather than merely improving polish
- `fix_leverage` (weight 0.200) — leverage of fixing this issue for making the project safer to share widely

## Top Entities

- no_large_live_receipt (rank Some(1), feasible true, u_mean 4.837, u_std 3.869, p_flip 0.412)
  - `credibility_risk`: latent 0.637 ± 0.891, z 1.111, min_norm 1.637, percentile 0.786
  - `falsifiability_gap`: latent 0.663 ± 0.704, z 2.532, min_norm 1.663, percentile 0.786
  - `fix_leverage`: latent 0.813 ± 0.771, z 1.223, min_norm 1.813, percentile 0.786
- cache_provenance (rank Some(2), feasible true, u_mean 3.942, u_std 3.495, p_flip 0.457)
  - `credibility_risk`: latent 0.440 ± 0.776, z 0.000, min_norm 1.440, percentile 0.500
  - `falsifiability_gap`: latent 0.666 ± 0.704, z 2.548, min_norm 1.666, percentile 0.929
  - `fix_leverage`: latent 0.634 ± 0.731, z 0.674, min_norm 1.634, percentile 0.643
- provider_metadata_drift (rank Some(3), feasible true, u_mean 3.194, u_std 3.507, p_flip 0.500)
  - `credibility_risk`: latent 0.773 ± 0.784, z 1.877, min_norm 1.773, percentile 0.929
  - `falsifiability_gap`: latent 0.036 ± 0.698, z -0.576, min_norm 1.036, percentile 0.357
  - `fix_leverage`: latent 0.217 ± 0.708, z -0.605, min_norm 1.217, percentile 0.214
- dependency_audit_warnings (rank Some(4), feasible true, u_mean 2.771, u_std 3.624, p_flip 0.524)
  - `credibility_risk`: latent 0.535 ± 0.830, z 0.534, min_norm 1.535, percentile 0.643
  - `falsifiability_gap`: latent 0.152 ± 0.676, z 0.000, min_norm 1.152, percentile 0.500
  - `fix_leverage`: latent 0.409 ± 0.696, z -0.016, min_norm 1.409, percentile 0.357
- baseline_breadth (rank Some(5), feasible true, u_mean 2.546, u_std 3.624, p_flip 0.536)
  - `credibility_risk`: latent 0.380 ± 0.816, z -0.336, min_norm 1.380, percentile 0.357
  - `falsifiability_gap`: latent 0.288 ± 0.701, z 0.674, min_norm 1.288, percentile 0.643
  - `fix_leverage`: latent 0.414 ± 0.771, z 0.000, min_norm 1.414, percentile 0.500
- api_surface_complexity (rank Some(6), feasible true, u_mean 2.078, u_std 3.857, p_flip 0.560)
  - `credibility_risk`: latent 0.320 ± 0.891, z -0.674, min_norm 1.320, percentile 0.214
  - `falsifiability_gap`: latent 0.033 ± 0.699, z -0.593, min_norm 1.033, percentile 0.214
  - `fix_leverage`: latent 0.872 ± 0.732, z 1.403, min_norm 1.872, percentile 0.929
- cost_estimate_semantics (rank Some(7), feasible true, u_mean 0.000, u_std 3.519, p_flip 0.675)
  - `credibility_risk`: latent 0.000 ± 0.801, z -2.475, min_norm 1.000, percentile 0.071
  - `falsifiability_gap`: latent 0.000 ± 0.664, z -0.755, min_norm 1.000, percentile 0.071
  - `fix_leverage`: latent 0.000 ± 0.717, z -1.270, min_norm 1.000, percentile 0.071
