use cardinal_harness::censored_likelihood::{
    fit_ordered_probit, CensoredFitConfig, CensoredObservation, CensoredPartition,
};
use cardinal_harness::prompts::RATIO_LADDER;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;

fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.gen_range(0.0..1.0_f64).max(f64::MIN_POSITIVE);
    let u2 = rng.gen_range(0.0..1.0_f64);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn point_fit(
    n: usize,
    partition: &CensoredPartition,
    observations: &[(usize, usize, usize)],
    prior_variance: f64,
) -> Vec<f64> {
    let mut information = DMatrix::identity(n, n) / prior_variance;
    let mut rhs = DVector::zeros(n);
    for &(i, j, cell) in observations {
        let center = partition.representative(cell).unwrap();
        information[(i, i)] += 1.0;
        information[(j, j)] += 1.0;
        information[(i, j)] -= 1.0;
        information[(j, i)] -= 1.0;
        rhs[i] += center;
        rhs[j] -= center;
    }
    let mut scores: Vec<f64> = information
        .cholesky()
        .unwrap()
        .solve(&rhs)
        .iter()
        .copied()
        .collect();
    let mean = scores.iter().sum::<f64>() / n as f64;
    for score in &mut scores {
        *score -= mean;
    }
    scores
}

fn mse(estimated: &[f64], truth: &[f64]) -> f64 {
    estimated
        .iter()
        .zip(truth)
        .map(|(estimated, truth)| (estimated - truth).powi(2))
        .sum::<f64>()
        / truth.len() as f64
}

#[derive(Default)]
struct CellResult {
    point_mse: f64,
    censored_mse: f64,
    runs: usize,
}

#[test]
fn ordered_probit_beats_or_matches_point_centers_on_its_planted_channel() {
    let full = CensoredPartition::from_ratio_ladder(RATIO_LADDER).unwrap();
    let coarse = CensoredPartition::from_centers(&[-2.5f64.ln(), 0.0, 2.5f64.ln()]).unwrap();
    let mut cells: BTreeMap<(bool, u64, usize), CellResult> = BTreeMap::new();

    for (is_coarse, partition) in [(false, &full), (true, &coarse)] {
        for sigma in [0.15_f64, 0.6] {
            for n in [8_usize, 24] {
                for seed in 0..20_u64 {
                    let mut rng = StdRng::seed_from_u64(
                        seed + 10_000 * n as u64 + 1_000_000 * u64::from(is_coarse),
                    );
                    let mut truth: Vec<f64> =
                        (0..n).map(|_| 0.8 * standard_normal(&mut rng)).collect();
                    let mean = truth.iter().sum::<f64>() / n as f64;
                    for score in &mut truth {
                        *score -= mean;
                    }

                    let mut raw = Vec::new();
                    let mut counts: BTreeMap<(usize, usize, usize), usize> = BTreeMap::new();
                    for i in 0..n {
                        for j in (i + 1)..n {
                            for _ in 0..3 {
                                let latent =
                                    truth[i] - truth[j] + sigma * standard_normal(&mut rng);
                                let cell = partition.cell_for_value(latent);
                                raw.push((i, j, cell));
                                *counts.entry((i, j, cell)).or_default() += 1;
                            }
                        }
                    }
                    let observations: Vec<CensoredObservation> = counts
                        .into_iter()
                        .map(|((i, j, cell), count)| CensoredObservation { i, j, cell, count })
                        .collect();
                    let config = CensoredFitConfig {
                        sigma,
                        prior_variance: 1.0,
                        ..CensoredFitConfig::default()
                    };
                    let censored = fit_ordered_probit(n, partition, &observations, config).unwrap();
                    assert!(censored.converged);
                    let point = point_fit(n, partition, &raw, config.prior_variance);
                    let result = cells.entry((is_coarse, sigma.to_bits(), n)).or_default();
                    result.point_mse += mse(&point, &truth);
                    result.censored_mse += mse(&censored.scores, &truth);
                    result.runs += 1;
                }
            }
        }
    }

    let mut wins = 0usize;
    let mut coarse_noisy_ratios = Vec::new();
    let mut fine_clean_ratios = Vec::new();
    for (&(is_coarse, sigma_bits, _), result) in &cells {
        let point = result.point_mse / result.runs as f64;
        let censored = result.censored_mse / result.runs as f64;
        let ratio = censored / point;
        wins += usize::from(ratio <= 1.0);
        let sigma = f64::from_bits(sigma_bits);
        if is_coarse && sigma == 0.6 {
            coarse_noisy_ratios.push(ratio);
        }
        if !is_coarse && sigma == 0.15 {
            fine_clean_ratios.push(ratio);
        }
    }
    coarse_noisy_ratios.sort_by(f64::total_cmp);
    fine_clean_ratios.sort_by(f64::total_cmp);
    let coarse_median = coarse_noisy_ratios[coarse_noisy_ratios.len() / 2];
    let fine_worst = fine_clean_ratios.iter().copied().fold(0.0, f64::max);

    assert!(wins >= 6, "censored model won only {wins}/8 cells");
    assert!(
        coarse_median <= 0.8,
        "coarse-channel median MSE ratio was {coarse_median:.3}, expected at least 20% improvement"
    );
    assert!(
        fine_worst <= 1.05,
        "fine/low-noise MSE ratio was {fine_worst:.3}, expected no more than 5% regression"
    );
}
