//! Experimental ordered-probit observation model for ladder-valued judgements.
//!
//! This module is deliberately separate from the production rating engine. It
//! exists to test whether treating a reported rung as an interval, rather than
//! an exact point, improves recovery and calibration before any solver cutover.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
const MIN_PROBABILITY: f64 = 1e-300;

#[derive(Debug, thiserror::Error)]
pub enum CensoredLikelihoodError {
    #[error("a partition requires at least two finite, strictly increasing centers")]
    InvalidCenters,
    #[error("fit requires at least two entities")]
    TooFewEntities,
    #[error("sigma, prior_variance, tolerances, and iteration limits must be positive and finite")]
    InvalidConfig,
    #[error("observation references an invalid entity or cell")]
    InvalidObservation,
    #[error("observed-information matrix is not positive definite")]
    SingularInformation,
    #[error("line search could not improve the penalized likelihood")]
    LineSearchFailed,
    #[error("ordered-probit probability underflowed; increase sigma or rescale the data")]
    NumericalUnderflow,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CensoredPartition {
    centers: Vec<f64>,
    cuts: Vec<f64>,
}

impl CensoredPartition {
    pub fn from_centers(centers: &[f64]) -> Result<Self, CensoredLikelihoodError> {
        if centers.len() < 2
            || centers.iter().any(|value| !value.is_finite())
            || centers.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(CensoredLikelihoodError::InvalidCenters);
        }
        let cuts = centers
            .windows(2)
            .map(|pair| 0.5 * (pair[0] + pair[1]))
            .collect();
        Ok(Self {
            centers: centers.to_vec(),
            cuts,
        })
    }

    pub fn from_ratio_ladder(ratios: &[f64]) -> Result<Self, CensoredLikelihoodError> {
        if ratios.len() < 2
            || (ratios[0] - 1.0).abs() > 1e-12
            || ratios
                .iter()
                .any(|ratio| !ratio.is_finite() || *ratio < 1.0)
            || ratios.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(CensoredLikelihoodError::InvalidCenters);
        }
        let logs: Vec<f64> = ratios.iter().map(|ratio| ratio.ln()).collect();
        let mut centers = Vec::with_capacity(2 * logs.len() - 1);
        centers.extend(logs[1..].iter().rev().map(|value| -*value));
        centers.push(0.0);
        centers.extend(logs[1..].iter().copied());
        Self::from_centers(&centers)
    }

    pub fn ordinal() -> Self {
        Self::from_centers(&[-1.0, 1.0]).expect("fixed ordinal centers are valid")
    }

    #[must_use]
    pub fn cells(&self) -> usize {
        self.cuts.len() + 1
    }

    pub fn representative(&self, cell: usize) -> Option<f64> {
        self.centers.get(cell).copied()
    }

    pub fn bounds(&self, cell: usize) -> Option<(f64, f64)> {
        if cell >= self.cells() {
            return None;
        }
        let lower = cell
            .checked_sub(1)
            .map_or(f64::NEG_INFINITY, |index| self.cuts[index]);
        let upper = self.cuts.get(cell).copied().unwrap_or(f64::INFINITY);
        Some((lower, upper))
    }

    #[must_use]
    pub fn cell_for_value(&self, value: f64) -> usize {
        self.cuts.partition_point(|cut| value >= *cut)
    }

    pub fn reflected_cell(&self, cell: usize) -> Option<usize> {
        (cell < self.cells()).then(|| self.cells() - 1 - cell)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellStats {
    pub probability: f64,
    pub log_probability: f64,
    pub score: f64,
    pub observed_information: f64,
    pub truncated_mean: f64,
}

fn normal_cdf(value: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-value / SQRT_2)
}

fn normal_pdf(value: f64) -> f64 {
    if value.is_finite() {
        INV_SQRT_2PI * (-0.5 * value * value).exp()
    } else {
        0.0
    }
}

fn scaled_pdf(value: f64, density: f64) -> f64 {
    if value.is_finite() {
        value * density
    } else {
        0.0
    }
}

fn interval_probability(alpha: f64, beta: f64) -> f64 {
    if alpha == f64::NEG_INFINITY {
        normal_cdf(beta)
    } else if beta == f64::INFINITY {
        normal_cdf(-alpha)
    } else if alpha > 0.0 {
        normal_cdf(-alpha) - normal_cdf(-beta)
    } else {
        normal_cdf(beta) - normal_cdf(alpha)
    }
}

pub fn cell_stats(
    partition: &CensoredPartition,
    cell: usize,
    delta: f64,
    sigma: f64,
) -> Result<CellStats, CensoredLikelihoodError> {
    if !delta.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
        return Err(CensoredLikelihoodError::InvalidConfig);
    }
    let (lower, upper) = partition
        .bounds(cell)
        .ok_or(CensoredLikelihoodError::InvalidObservation)?;
    let alpha = (lower - delta) / sigma;
    let beta = (upper - delta) / sigma;
    let phi_alpha = normal_pdf(alpha);
    let phi_beta = normal_pdf(beta);
    let probability = interval_probability(alpha, beta);
    if !probability.is_finite() || probability <= MIN_PROBABILITY {
        return Err(CensoredLikelihoodError::NumericalUnderflow);
    }
    let density_difference = phi_alpha - phi_beta;
    let score = density_difference / (sigma * probability);
    let second_probability_ratio =
        (scaled_pdf(alpha, phi_alpha) - scaled_pdf(beta, phi_beta)) / probability;
    let raw_information =
        ((density_difference / probability).powi(2) - second_probability_ratio) / (sigma * sigma);
    if !raw_information.is_finite() || raw_information < -1e-9 {
        return Err(CensoredLikelihoodError::NumericalUnderflow);
    }
    Ok(CellStats {
        probability,
        log_probability: probability.ln(),
        score,
        observed_information: raw_information.max(0.0),
        truncated_mean: delta + sigma * density_difference / probability,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CensoredObservation {
    pub i: usize,
    pub j: usize,
    pub cell: usize,
    pub count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CensoredFitConfig {
    pub sigma: f64,
    pub prior_variance: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub min_information: f64,
}

impl Default for CensoredFitConfig {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            prior_variance: 1.0,
            max_iterations: 100,
            tolerance: 1e-9,
            min_information: 1e-12,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CensoredFit {
    pub scores: Vec<f64>,
    pub diag_cov: Vec<f64>,
    pub log_posterior: f64,
    pub objective_history: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

fn validate_fit_inputs(
    n: usize,
    partition: &CensoredPartition,
    observations: &[CensoredObservation],
    config: CensoredFitConfig,
) -> Result<(), CensoredLikelihoodError> {
    if n < 2 {
        return Err(CensoredLikelihoodError::TooFewEntities);
    }
    if !config.sigma.is_finite()
        || config.sigma <= 0.0
        || !config.prior_variance.is_finite()
        || config.prior_variance <= 0.0
        || config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= 0.0
        || !config.min_information.is_finite()
        || config.min_information <= 0.0
    {
        return Err(CensoredLikelihoodError::InvalidConfig);
    }
    if observations.iter().any(|observation| {
        observation.i >= n
            || observation.j >= n
            || observation.i == observation.j
            || observation.cell >= partition.cells()
            || observation.count == 0
    }) {
        return Err(CensoredLikelihoodError::InvalidObservation);
    }
    Ok(())
}

fn point_initialization(
    n: usize,
    partition: &CensoredPartition,
    observations: &[CensoredObservation],
    prior_variance: f64,
) -> Result<Vec<f64>, CensoredLikelihoodError> {
    let mut information = DMatrix::identity(n, n) / prior_variance;
    let mut rhs = DVector::zeros(n);
    for observation in observations {
        let count = observation.count as f64;
        let center = partition
            .representative(observation.cell)
            .ok_or(CensoredLikelihoodError::InvalidObservation)?;
        information[(observation.i, observation.i)] += count;
        information[(observation.j, observation.j)] += count;
        information[(observation.i, observation.j)] -= count;
        information[(observation.j, observation.i)] -= count;
        rhs[observation.i] += count * center;
        rhs[observation.j] -= count * center;
    }
    let Some(cholesky) = information.cholesky() else {
        return Err(CensoredLikelihoodError::SingularInformation);
    };
    let mut scores: Vec<f64> = cholesky.solve(&rhs).iter().copied().collect();
    let mean = scores.iter().sum::<f64>() / n as f64;
    for score in &mut scores {
        *score -= mean;
    }
    Ok(scores)
}

fn objective(
    scores: &[f64],
    partition: &CensoredPartition,
    observations: &[CensoredObservation],
    config: CensoredFitConfig,
) -> Result<f64, CensoredLikelihoodError> {
    let mut value =
        -scores.iter().map(|score| score * score).sum::<f64>() / (2.0 * config.prior_variance);
    for observation in observations {
        let delta = scores[observation.i] - scores[observation.j];
        value += observation.count as f64
            * cell_stats(partition, observation.cell, delta, config.sigma)?.log_probability;
    }
    Ok(value)
}

fn gradient_and_information(
    scores: &[f64],
    partition: &CensoredPartition,
    observations: &[CensoredObservation],
    config: CensoredFitConfig,
) -> Result<(DVector<f64>, DMatrix<f64>), CensoredLikelihoodError> {
    let n = scores.len();
    let prior_precision = 1.0 / config.prior_variance;
    let mut gradient =
        DVector::from_iterator(n, scores.iter().map(|score| -score * prior_precision));
    let mut information = DMatrix::identity(n, n) * prior_precision;

    for observation in observations {
        let delta = scores[observation.i] - scores[observation.j];
        let stats = cell_stats(partition, observation.cell, delta, config.sigma)?;
        let count = observation.count as f64;
        let score = count * stats.score;
        let edge_information = count * stats.observed_information.max(config.min_information);
        gradient[observation.i] += score;
        gradient[observation.j] -= score;
        information[(observation.i, observation.i)] += edge_information;
        information[(observation.j, observation.j)] += edge_information;
        information[(observation.i, observation.j)] -= edge_information;
        information[(observation.j, observation.i)] -= edge_information;
    }
    Ok((gradient, information))
}

pub fn fit_ordered_probit(
    n: usize,
    partition: &CensoredPartition,
    observations: &[CensoredObservation],
    config: CensoredFitConfig,
) -> Result<CensoredFit, CensoredLikelihoodError> {
    validate_fit_inputs(n, partition, observations, config)?;
    let mut scores = point_initialization(n, partition, observations, config.prior_variance)?;
    let mut current = objective(&scores, partition, observations, config)?;
    let mut objective_history = vec![current];
    let mut converged = false;
    let mut iterations = 0usize;

    for iteration in 0..config.max_iterations {
        let (gradient, information) =
            gradient_and_information(&scores, partition, observations, config)?;
        let Some(cholesky) = information.cholesky() else {
            return Err(CensoredLikelihoodError::SingularInformation);
        };
        let step = cholesky.solve(&gradient);
        if step.amax() <= config.tolerance {
            converged = true;
            iterations = iteration;
            break;
        }

        let mut scale = 1.0;
        let mut accepted = None;
        while scale >= 2f64.powi(-24) {
            let mut candidate: Vec<f64> = scores
                .iter()
                .zip(step.iter())
                .map(|(score, update)| score + scale * update)
                .collect();
            let mean = candidate.iter().sum::<f64>() / n as f64;
            for score in &mut candidate {
                *score -= mean;
            }
            let candidate_objective = objective(&candidate, partition, observations, config)?;
            if candidate_objective + 1e-12 >= current {
                accepted = Some((candidate, candidate_objective));
                break;
            }
            scale *= 0.5;
        }
        let Some((candidate, candidate_objective)) = accepted else {
            return Err(CensoredLikelihoodError::LineSearchFailed);
        };
        let improvement = candidate_objective - current;
        scores = candidate;
        current = candidate_objective;
        objective_history.push(current);
        iterations = iteration + 1;
        if improvement.abs() <= config.tolerance * (1.0 + current.abs()) {
            converged = true;
            break;
        }
    }

    let (_, information) = gradient_and_information(&scores, partition, observations, config)?;
    let Some(cholesky) = information.cholesky() else {
        return Err(CensoredLikelihoodError::SingularInformation);
    };
    let covariance = cholesky.inverse();
    // Scores are reported in the mean-zero gauge. Project the inverse
    // information through P = I - 11ᵀ/n so marginal variances do not retain
    // the prior-only common mode that the reported scores omit.
    let grand_mean = covariance.iter().sum::<f64>() / (n * n) as f64;
    let diag_cov = (0..n)
        .map(|index| {
            let row_mean = (0..n)
                .map(|column| covariance[(index, column)])
                .sum::<f64>()
                / n as f64;
            (covariance[(index, index)] - 2.0 * row_mean + grand_mean).max(0.0)
        })
        .collect();

    Ok(CensoredFit {
        scores,
        diag_cov,
        log_posterior: current,
        objective_history,
        iterations,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompts::RATIO_LADDER;

    #[test]
    fn ratio_partition_has_symmetric_midpoint_cuts() {
        let partition = CensoredPartition::from_ratio_ladder(RATIO_LADDER).unwrap();
        assert_eq!(partition.cells(), 2 * RATIO_LADDER.len() - 1);
        let center = RATIO_LADDER.len() - 1;
        let half_first_step = 0.5 * RATIO_LADDER[1].ln();
        assert_eq!(
            partition.bounds(center),
            Some((-half_first_step, half_first_step))
        );
        assert_eq!(partition.reflected_cell(0), Some(partition.cells() - 1));
        assert_eq!(partition.reflected_cell(center), Some(center));
    }

    #[test]
    fn channel_reflects_under_slot_swap() {
        let partition = CensoredPartition::from_ratio_ladder(RATIO_LADDER).unwrap();
        for cell in 0..partition.cells() {
            let reflected = partition.reflected_cell(cell).unwrap();
            for delta in [-2.0, -0.3, 0.0, 0.8, 2.5] {
                let left = cell_stats(&partition, cell, delta, 0.7).unwrap();
                let right = cell_stats(&partition, reflected, -delta, 0.7).unwrap();
                assert!((left.probability - right.probability).abs() < 1e-13);
                assert!((left.score + right.score).abs() < 1e-11);
                assert!((left.observed_information - right.observed_information).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn analytic_score_and_information_match_finite_differences() {
        let partition = CensoredPartition::from_ratio_ladder(RATIO_LADDER).unwrap();
        let cases = [
            (partition.cell_for_value(2.5f64.ln()), 0.6),
            (RATIO_LADDER.len() - 1, 0.0),
        ];
        let sigma = 0.8;
        let h = 1e-4;
        for (cell, delta) in cases {
            let at = cell_stats(&partition, cell, delta, sigma).unwrap();
            let plus = cell_stats(&partition, cell, delta + h, sigma)
                .unwrap()
                .log_probability;
            let minus = cell_stats(&partition, cell, delta - h, sigma)
                .unwrap()
                .log_probability;
            let numeric_score = (plus - minus) / (2.0 * h);
            let numeric_information = -(plus - 2.0 * at.log_probability + minus) / (h * h);
            assert!((at.score - numeric_score).abs() < 1e-7);
            assert!((at.observed_information - numeric_information).abs() < 1e-6);
        }
    }

    #[test]
    fn vanishing_noise_concentrates_on_the_point_cell() {
        let partition = CensoredPartition::from_ratio_ladder(RATIO_LADDER).unwrap();
        let center = 2.5f64.ln();
        let cell = partition.cell_for_value(center);
        let selected = cell_stats(&partition, cell, center, 1e-2).unwrap();
        let adjacent = cell_stats(&partition, cell - 1, center, 1e-2).unwrap();
        assert!(selected.probability > 1.0 - 1e-12);
        assert!(adjacent.probability < 1e-12);
    }

    #[test]
    fn small_noise_fit_recovers_the_planted_cells() {
        let partition = CensoredPartition::from_ratio_ladder(RATIO_LADDER).unwrap();
        let truth = [2.1f64.ln(), 0.0, -2.1f64.ln()];
        let mut observations = Vec::new();
        for i in 0..truth.len() {
            for j in (i + 1)..truth.len() {
                observations.push(CensoredObservation {
                    i,
                    j,
                    cell: partition.cell_for_value(truth[i] - truth[j]),
                    count: 4,
                });
            }
        }
        let fit = fit_ordered_probit(
            truth.len(),
            &partition,
            &observations,
            CensoredFitConfig {
                sigma: 0.03,
                ..CensoredFitConfig::default()
            },
        )
        .unwrap();
        assert!(fit.converged);
        for observation in observations {
            let fitted_delta = fit.scores[observation.i] - fit.scores[observation.j];
            assert_eq!(partition.cell_for_value(fitted_delta), observation.cell);
        }
    }

    #[test]
    fn deep_tail_underflow_is_an_error_not_a_flat_likelihood() {
        let partition = CensoredPartition::from_ratio_ladder(RATIO_LADDER).unwrap();
        let error = cell_stats(&partition, 0, 100.0, 0.01).unwrap_err();
        assert!(matches!(error, CensoredLikelihoodError::NumericalUnderflow));
    }

    #[test]
    fn weak_prior_keeps_an_extreme_cell_fit_finite() {
        let partition = CensoredPartition::from_ratio_ladder(RATIO_LADDER).unwrap();
        let observation = CensoredObservation {
            i: 0,
            j: 1,
            cell: partition.cells() - 1,
            count: 1,
        };
        let fit = fit_ordered_probit(2, &partition, &[observation], CensoredFitConfig::default())
            .unwrap();
        assert!(fit.converged);
        assert!(fit.scores.iter().all(|score| score.is_finite()));
        assert!(fit
            .diag_cov
            .iter()
            .all(|variance| variance.is_finite() && *variance > 0.0));
    }

    #[test]
    fn covariance_uses_the_reported_mean_zero_gauge() {
        let partition = CensoredPartition::ordinal();
        let fit = fit_ordered_probit(
            2,
            &partition,
            &[],
            CensoredFitConfig {
                prior_variance: 4.0,
                ..CensoredFitConfig::default()
            },
        )
        .unwrap();
        assert_eq!(fit.scores, vec![0.0, 0.0]);
        assert_eq!(fit.diag_cov, vec![2.0, 2.0]);
    }

    #[test]
    fn graph_fit_improves_monotonically_and_recovers_order() {
        let partition = CensoredPartition::from_ratio_ladder(RATIO_LADDER).unwrap();
        let truth = [0.9, 0.3, -0.2, -1.0];
        let mut observations = Vec::new();
        for i in 0..truth.len() {
            for j in (i + 1)..truth.len() {
                observations.push(CensoredObservation {
                    i,
                    j,
                    cell: partition.cell_for_value(truth[i] - truth[j]),
                    count: 8,
                });
            }
        }
        let fit = fit_ordered_probit(
            truth.len(),
            &partition,
            &observations,
            CensoredFitConfig {
                sigma: 0.3,
                ..CensoredFitConfig::default()
            },
        )
        .unwrap();
        assert!(fit.converged);
        assert!(fit
            .objective_history
            .windows(2)
            .all(|pair| pair[1] + 1e-12 >= pair[0]));
        assert!(fit.scores.windows(2).all(|pair| pair[0] > pair[1]));
    }
}
