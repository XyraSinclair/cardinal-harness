# Rerank Report

- Request hash: `140f1c9a29f2ea956f438ad5992512647914d7880c157c6ebd4854fcb94d89fd`
- Stop reason: `no_new_pairs`
- k: 57
- Global top-k error: 11.0650
- Tolerated error: 0.1500
- Comparisons used/attempted/refused/cached: 1296/1296/0/0
- Comparison budget: 1300
- Model used: deepseek/deepseek-v4-flash
- Rater ID: manifund-p2-eacc-2026-07-19
- Latency: 2474745 ms
- Provider tokens input/output/total: 3067612/752669/3820281
- Provider cost: $0.607567132
- RNG seed: 7

## Warnings / Degraded State

- Run stopped with non-converged stop reason `no_new_pairs`; inspect uncertainty before sharing this as a settled ordering.
- Global top-k error 11.0650 exceeds tolerated error 0.1500.

## Run Status

The planner found candidate comparisons, but all eligible pairs were already known or blocked.

## Attributes

- `theory_of_change` (weight 0.300) — plausibility of the causal path from the proposed activities to the claimed impact
- `impact_per_dollar` (weight 0.300) — expected impact per marginal dollar at the stated minimum funding ask
- `team_evidence` (weight 0.250) — strength of verifiable track-record evidence that this team can execute this plan
- `epistemic_integrity` (weight 0.150) — epistemic integrity of the write-up: honest failure modes, quantified claims, falsifiable milestones

## Top Entities

- lightcone-infrastructure (rank Some(1), feasible true, u_mean 8.179, u_std 0.826, p_flip 0.008)
  - `epistemic_integrity`: latent 2.943 ± 0.755, z 2.408, min_norm 3.943, percentile 0.968
  - `impact_per_dollar`: latent 2.617 ± 0.756, z 1.667, min_norm 3.617, percentile 0.994
  - `team_evidence`: latent 3.836 ± 0.551, z 1.535, min_norm 4.836, percentile 0.955
  - `theory_of_change`: latent 2.719 ± 0.399, z 1.282, min_norm 3.719, percentile 0.917
- 80000-hours (rank Some(2), feasible true, u_mean 8.154, u_std 0.820, p_flip 0.008)
  - `epistemic_integrity`: latent 2.398 ± 0.764, z 1.030, min_norm 3.398, percentile 0.878
  - `impact_per_dollar`: latent 2.037 ± 0.570, z 0.785, min_norm 3.037, percentile 0.763
  - `team_evidence`: latent 4.598 ± 0.689, z 2.717, min_norm 5.598, percentile 0.994
  - `theory_of_change`: latent 2.970 ± 0.455, z 1.812, min_norm 3.970, percentile 0.981
- animal-advocacy-strategy-forum (rank Some(3), feasible true, u_mean 7.661, u_std 1.245, p_flip 0.063)
  - `epistemic_integrity`: latent 2.448 ± 0.714, z 1.157, min_norm 3.448, percentile 0.904
  - `impact_per_dollar`: latent 2.261 ± 1.466, z 1.126, min_norm 3.261, percentile 0.904
  - `team_evidence`: latent 3.859 ± 0.770, z 1.571, min_norm 4.859, percentile 0.968
  - `theory_of_change`: latent 2.706 ± 0.489, z 1.255, min_norm 3.706, percentile 0.904
- impact-accelerator-program-biggest-career-program-for-experienced-professionals-u20ul2jdzhq (rank Some(4), feasible true, u_mean 7.651, u_std 1.090, p_flip 0.047)
  - `epistemic_integrity`: latent 2.089 ± 0.513, z 0.248, min_norm 3.089, percentile 0.622
  - `impact_per_dollar`: latent 2.616 ± 1.051, z 1.666, min_norm 3.616, percentile 0.981
  - `team_evidence`: latent 3.866 ± 1.148, z 1.582, min_norm 4.866, percentile 0.981
  - `theory_of_change`: latent 2.651 ± 0.430, z 1.139, min_norm 3.651, percentile 0.878
- using-monitoring--evaluation-to-enhance-impact-in-the-animal-cause-area (rank Some(5), feasible true, u_mean 7.636, u_std 0.858, p_flip 0.026)
  - `epistemic_integrity`: latent 3.298 ± 0.978, z 3.306, min_norm 4.298, percentile 0.994
  - `impact_per_dollar`: latent 2.156 ± 0.605, z 0.966, min_norm 3.156, percentile 0.853
  - `team_evidence`: latent 3.430 ± 0.444, z 0.906, min_norm 4.430, percentile 0.827
  - `theory_of_change`: latent 2.508 ± 0.478, z 0.839, min_norm 3.508, percentile 0.801
- mats-funding (rank Some(6), feasible true, u_mean 7.474, u_std 1.157, p_flip 0.067)
  - `epistemic_integrity`: latent 2.485 ± 0.768, z 1.249, min_norm 3.485, percentile 0.917
  - `impact_per_dollar`: latent 2.391 ± 0.417, z 1.323, min_norm 3.391, percentile 0.955
  - `team_evidence`: latent 3.530 ± 0.607, z 1.061, min_norm 4.530, percentile 0.865
  - `theory_of_change`: latent 2.592 ± 1.040, z 1.016, min_norm 3.592, percentile 0.865
- putting-animal-advocacy-research-into-action (rank Some(7), feasible true, u_mean 7.458, u_std 1.046, p_flip 0.057)
  - `epistemic_integrity`: latent 2.132 ± 0.717, z 0.356, min_norm 3.132, percentile 0.647
  - `impact_per_dollar`: latent 2.363 ± 1.054, z 1.282, min_norm 3.363, percentile 0.942
  - `team_evidence`: latent 3.738 ± 0.441, z 1.383, min_norm 4.738, percentile 0.917
  - `theory_of_change`: latent 2.679 ± 0.639, z 1.199, min_norm 3.679, percentile 0.891
- ai-safety-camp-south-africa-condor-camp (rank Some(8), feasible true, u_mean 7.341, u_std 1.181, p_flip 0.084)
  - `epistemic_integrity`: latent 2.347 ± 1.188, z 0.902, min_norm 3.347, percentile 0.853
  - `impact_per_dollar`: latent 1.999 ± 1.071, z 0.727, min_norm 2.999, percentile 0.737
  - `team_evidence`: latent 3.568 ± 0.520, z 1.119, min_norm 4.568, percentile 0.878
  - `theory_of_change`: latent 2.793 ± 0.616, z 1.439, min_norm 3.793, percentile 0.968
- ceealar (rank Some(9), feasible true, u_mean 7.236, u_std 0.823, p_flip 0.047)
  - `epistemic_integrity`: latent 2.614 ± 0.703, z 1.576, min_norm 3.614, percentile 0.955
  - `impact_per_dollar`: latent 1.757 ± 0.501, z 0.359, min_norm 2.757, percentile 0.647
  - `team_evidence`: latent 3.490 ± 0.671, z 0.998, min_norm 4.490, percentile 0.840
  - `theory_of_change`: latent 2.743 ± 0.540, z 1.334, min_norm 3.743, percentile 0.929
- building-bridges-effective-animal-advocacy-in-the-global-south (rank Some(10), feasible true, u_mean 7.217, u_std 1.045, p_flip 0.077)
  - `epistemic_integrity`: latent 2.299 ± 1.054, z 0.779, min_norm 3.299, percentile 0.788
  - `impact_per_dollar`: latent 2.084 ± 0.605, z 0.858, min_norm 3.084, percentile 0.827
  - `team_evidence`: latent 3.671 ± 1.102, z 1.279, min_norm 4.671, percentile 0.891
  - `theory_of_change`: latent 2.564 ± 0.443, z 0.957, min_norm 3.564, percentile 0.827
