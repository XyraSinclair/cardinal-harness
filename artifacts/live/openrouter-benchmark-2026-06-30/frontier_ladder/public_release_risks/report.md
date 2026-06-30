# Rerank Report

- Request hash: `1719e25c45d22883177defb878b6382c67b3628ff5d701ff6f4dc22260d86441`
- Stop reason: `no_new_pairs`
- k: 3
- Global top-k error: 3.8585
- Tolerated error: 0.0800
- Comparisons used/attempted/refused/cached: 63/63/0/0
- Comparison budget: 84
- Model used: anthropic/claude-opus-4.6
- Rater ID: live-openrouter-public_release_risks
- Latency: 43430 ms
- Provider tokens input/output/total: 30525/1701/32226
- Provider cost: $0.195150000
- RNG seed: 20260630
- Model policy: ModelLadderPolicy(high=anthropic/claude-opus-4.6, mid=google/gemini-3.1-pro-preview, low=openai/gpt-5.4-mini, switch_error=0.1, similarity_ln_ratio=0.12, max_pair_std=0.6, min_comparisons=12)

## Warnings / Degraded State

- Run stopped with non-converged stop reason `no_new_pairs`; inspect uncertainty before sharing this as a settled ordering.
- Global top-k error 3.8585 exceeds tolerated error 0.0800.

## Run Status

The planner found candidate comparisons, but all eligible pairs were already known or blocked.

## Attributes

- `credibility_risk` (weight 0.450) — risk to public credibility if cardinal-harness is shared widely as a serious empirical artifact
- `falsifiability_gap` (weight 0.350) — how much this issue blocks concrete falsifiable live evidence rather than merely improving polish
- `fix_leverage` (weight 0.200) — leverage of fixing this issue for making the project safer to share widely

## Top Entities

- no_large_live_receipt (rank Some(1), feasible true, u_mean 5.486, u_std 2.823, p_flip 0.231)
  - `credibility_risk`: latent 0.928 ± 0.789, z 3.563, min_norm 1.928, percentile 0.929
  - `falsifiability_gap`: latent 1.347 ± 0.806, z 2.662, min_norm 2.347, percentile 0.929
  - `fix_leverage`: latent 0.858 ± 0.850, z 0.788, min_norm 1.858, percentile 0.786
- baseline_breadth (rank Some(2), feasible true, u_mean 4.032, u_std 2.809, p_flip 0.318)
  - `credibility_risk`: latent 0.836 ± 0.789, z 3.126, min_norm 1.836, percentile 0.786
  - `falsifiability_gap`: latent 0.770 ± 0.787, z 0.988, min_norm 1.770, percentile 0.786
  - `fix_leverage`: latent 0.372 ± 0.839, z -0.206, min_norm 1.372, percentile 0.357
- provider_metadata_drift (rank Some(3), feasible true, u_mean 1.438, u_std 2.685, p_flip 0.500)
  - `credibility_risk`: latent 0.177 ± 0.758, z 0.000, min_norm 1.177, percentile 0.500
  - `falsifiability_gap`: latent 0.525 ± 0.734, z 0.276, min_norm 1.525, percentile 0.643
  - `fix_leverage`: latent 0.143 ± 0.794, z -0.674, min_norm 1.143, percentile 0.214
- cost_estimate_semantics (rank Some(4), feasible true, u_mean 1.291, u_std 2.690, p_flip 0.511)
  - `credibility_risk`: latent 0.243 ± 0.763, z 0.310, min_norm 1.243, percentile 0.643
  - `falsifiability_gap`: latent 0.347 ± 0.718, z -0.239, min_norm 1.347, percentile 0.357
  - `fix_leverage`: latent 0.000 ± 0.796, z -0.968, min_norm 1.000, percentile 0.071
- api_surface_complexity (rank Some(5), feasible true, u_mean 1.011, u_std 2.748, p_flip 0.531)
  - `credibility_risk`: latent 0.035 ± 0.783, z -0.674, min_norm 1.035, percentile 0.214
  - `falsifiability_gap`: latent 0.197 ± 0.718, z -0.674, min_norm 1.197, percentile 0.214
  - `fix_leverage`: latent 0.995 ± 0.799, z 1.068, min_norm 1.995, percentile 0.929
- cache_provenance (rank Some(6), feasible true, u_mean 0.933, u_std 2.760, p_flip 0.537)
  - `credibility_risk`: latent 0.000 ± 0.783, z -0.840, min_norm 1.000, percentile 0.071
  - `falsifiability_gap`: latent 0.430 ± 0.733, z 0.000, min_norm 1.430, percentile 0.500
  - `fix_leverage`: latent 0.473 ± 0.832, z 0.000, min_norm 1.473, percentile 0.500
- dependency_audit_warnings (rank Some(7), feasible true, u_mean 0.656, u_std 2.715, p_flip 0.558)
  - `credibility_risk`: latent 0.097 ± 0.750, z -0.379, min_norm 1.097, percentile 0.357
  - `falsifiability_gap`: latent 0.000 ± 0.806, z -1.246, min_norm 1.000, percentile 0.071
  - `fix_leverage`: latent 0.574 ± 0.850, z 0.207, min_norm 1.574, percentile 0.643
