# Rerank Report

- Request hash: `f49a2bf7b1a77aedb31424ed85cb619a98633c31b3f9cd514a5c6103c64ca366`
- Stop reason: `no_new_pairs`
- k: 41
- Global top-k error: 11.3901
- Tolerated error: 0.1500
- Comparisons used/attempted/refused/cached: 987/987/0/0
- Comparison budget: 1400
- Model used: deepseek/deepseek-v4-flash
- Rater ID: manifund-p2-acx2024-2026-07-19
- Latency: 1746370 ms
- Provider tokens input/output/total: 2671422/496784/3168206
- Provider cost: $0.493752915
- RNG seed: 7

## Warnings / Degraded State

- Run stopped with non-converged stop reason `no_new_pairs`; inspect uncertainty before sharing this as a settled ordering.
- Global top-k error 11.3901 exceeds tolerated error 0.1500.

## Run Status

The planner found candidate comparisons, but all eligible pairs were already known or blocked.

## Attributes

- `theory_of_change` (weight 0.300) — plausibility of the causal path from the proposed activities to the claimed impact
- `impact_per_dollar` (weight 0.300) — expected impact per marginal dollar at the stated minimum funding ask
- `team_evidence` (weight 0.250) — strength of verifiable track-record evidence that this team can execute this plan
- `epistemic_integrity` (weight 0.150) — epistemic integrity of the write-up: honest failure modes, quantified claims, falsifiable milestones

## Top Entities

- improving-fish-w (rank Some(1), feasible true, u_mean 8.918, u_std 0.884, p_flip 0.076)
  - `epistemic_integrity`: latent 1.954 ± 1.299, z 0.847, min_norm 2.954, percentile 0.825
  - `impact_per_dollar`: latent 5.820 ± 1.270, z 2.035, min_norm 6.820, percentile 0.994
  - `team_evidence`: latent 3.947 ± 0.519, z 1.624, min_norm 4.947, percentile 0.982
  - `theory_of_change`: latent 3.996 ± 0.374, z 1.685, min_norm 4.996, percentile 0.958
- apart-incubates- (rank Some(2), feasible true, u_mean 8.508, u_std 1.064, p_flip 0.139)
  - `epistemic_integrity`: latent 2.890 ± 1.609, z 2.829, min_norm 3.890, percentile 0.982
  - `impact_per_dollar`: latent 4.899 ± 0.915, z 1.301, min_norm 5.899, percentile 0.934
  - `team_evidence`: latent 3.008 ± 0.794, z 0.383, min_norm 4.008, percentile 0.633
  - `theory_of_change`: latent 3.932 ± 0.549, z 1.541, min_norm 4.932, percentile 0.922
- regrant-to-chari (rank Some(3), feasible true, u_mean 8.286, u_std 1.107, p_flip 0.170)
  - `epistemic_integrity`: latent 2.081 ± 1.297, z 1.116, min_norm 3.081, percentile 0.886
  - `impact_per_dollar`: latent 5.135 ± 1.447, z 1.489, min_norm 6.135, percentile 0.970
  - `team_evidence`: latent 3.134 ± 0.927, z 0.549, min_norm 4.134, percentile 0.693
  - `theory_of_change`: latent 3.946 ± 0.619, z 1.572, min_norm 4.946, percentile 0.946
- run-a-self-help- (rank Some(4), feasible true, u_mean 8.255, u_std 1.183, p_flip 0.182)
  - `epistemic_integrity`: latent 1.704 ± 1.583, z 0.319, min_norm 2.704, percentile 0.608
  - `impact_per_dollar`: latent 4.635 ± 1.652, z 1.091, min_norm 5.635, percentile 0.910
  - `team_evidence`: latent 3.802 ± 1.135, z 1.432, min_norm 4.802, percentile 0.922
  - `theory_of_change`: latent 3.942 ± 0.438, z 1.563, min_norm 4.942, percentile 0.934
- african-school-o (rank Some(5), feasible true, u_mean 8.195, u_std 0.784, p_flip 0.140)
  - `epistemic_integrity`: latent 1.826 ± 1.000, z 0.576, min_norm 2.826, percentile 0.693
  - `impact_per_dollar`: latent 5.119 ± 0.902, z 1.476, min_norm 6.119, percentile 0.958
  - `team_evidence`: latent 3.836 ± 0.589, z 1.477, min_norm 4.836, percentile 0.934
  - `theory_of_change`: latent 3.637 ± 0.455, z 0.877, min_norm 4.637, percentile 0.801
- promote-georgism (rank Some(6), feasible true, u_mean 8.150, u_std 0.985, p_flip 0.173)
  - `epistemic_integrity`: latent 3.879 ± 1.613, z 4.921, min_norm 4.879, percentile 0.994
  - `impact_per_dollar`: latent 4.071 ± 0.818, z 0.641, min_norm 5.071, percentile 0.741
  - `team_evidence`: latent 3.000 ± 0.692, z 0.372, min_norm 4.000, percentile 0.620
  - `theory_of_change`: latent 3.407 ± 0.441, z 0.359, min_norm 4.407, percentile 0.657
- develop-an-acces (rank Some(7), feasible true, u_mean 8.032, u_std 0.999, p_flip 0.192)
  - `epistemic_integrity`: latent 1.947 ± 1.406, z 0.832, min_norm 2.947, percentile 0.813
  - `impact_per_dollar`: latent 4.844 ± 1.033, z 1.257, min_norm 5.844, percentile 0.922
  - `team_evidence`: latent 2.735 ± 0.738, z 0.021, min_norm 3.735, percentile 0.512
  - `theory_of_change`: latent 4.054 ± 0.543, z 1.816, min_norm 5.054, percentile 0.982
- scaling-legal-im (rank Some(8), feasible true, u_mean 7.921, u_std 1.125, p_flip 0.223)
  - `epistemic_integrity`: latent 1.663 ± 1.356, z 0.231, min_norm 2.663, percentile 0.560
  - `impact_per_dollar`: latent 3.888 ± 1.441, z 0.496, min_norm 4.888, percentile 0.681
  - `team_evidence`: latent 2.960 ± 0.701, z 0.319, min_norm 3.960, percentile 0.608
  - `theory_of_change`: latent 4.305 ± 0.693, z 2.379, min_norm 5.305, percentile 0.994
- build-a-new-nonp (rank Some(9), feasible true, u_mean 7.761, u_std 1.038, p_flip 0.238)
  - `epistemic_integrity`: latent 1.921 ± 1.410, z 0.778, min_norm 2.921, percentile 0.801
  - `impact_per_dollar`: latent 3.935 ± 1.365, z 0.533, min_norm 4.935, percentile 0.729
  - `team_evidence`: latent 3.125 ± 0.853, z 0.537, min_norm 4.125, percentile 0.681
  - `theory_of_change`: latent 3.926 ± 0.478, z 1.527, min_norm 4.926, percentile 0.910
- support-early-ca (rank Some(10), feasible true, u_mean 7.754, u_std 1.209, p_flip 0.257)
  - `epistemic_integrity`: latent 1.554 ± 1.616, z 0.000, min_norm 2.554, percentile 0.500
  - `impact_per_dollar`: latent 4.190 ± 1.616, z 0.737, min_norm 5.190, percentile 0.777
  - `team_evidence`: latent 3.408 ± 0.816, z 0.910, min_norm 4.408, percentile 0.849
  - `theory_of_change`: latent 3.863 ± 0.629, z 1.385, min_norm 4.863, percentile 0.886
