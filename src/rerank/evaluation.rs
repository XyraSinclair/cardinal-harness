//! Offline evaluation harness for the multi-objective reranker.
//!
//! Runs synthetic cases through the actual TraitSearchManager loop,
//! replacing LLM comparisons with a deterministic simulator.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::rating_engine::{
    AttributeParams, Config as EngineConfig, Observation, PlannerMode, RaterParams, RatingEngine,
};
use crate::rerank::gates::{validate_gate_specs, ParsedGateSpec};
use crate::rerank::types::{
    HigherRanked, MultiRerankGateSpec, MultiRerankTopKSpec, PairwiseJudgement, RerankStopReason,
};
use crate::trait_search::{
    compute_attribute_units, AttributeConfig, GateSpec, TopKConfig, TraitSearchConfig,
    TraitSearchManager,
};

type AttrUnits = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
type LikertInference = (Vec<f64>, Vec<f64>, Vec<bool>);

#[derive(Debug, thiserror::Error)]
pub enum EvaluationError {
    #[error("invalid synthetic gate spec: {0}")]
    InvalidGate(#[from] crate::rerank::MultiRerankError),
}

/// Synthetic pairwise signal used by the cardinal evaluation harness.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticPairwiseMode {
    /// Pairwise comparisons carry the true noisy ratio magnitude.
    #[default]
    Ratio,
    /// Pairwise comparisons carry only direction plus a fixed small ratio.
    Ordinal,
}

/// Configuration for the cardinal pairwise simulator.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct PairwiseEvalConfig {
    pub mode: SyntheticPairwiseMode,
}

impl Default for PairwiseEvalConfig {
    fn default() -> Self {
        Self {
            mode: SyntheticPairwiseMode::Ratio,
        }
    }
}

const ORDINAL_SYNTHETIC_RATIO: f64 = 2.1;

fn parse_evaluation_gates<'a>(
    attributes: &'a [SyntheticAttribute],
    gates: &'a [MultiRerankGateSpec],
) -> Result<Vec<ParsedGateSpec<'a>>, EvaluationError> {
    let attribute_ids: HashSet<&str> = attributes.iter().map(|attr| attr.id).collect();
    validate_gate_specs(gates, &attribute_ids).map_err(EvaluationError::from)
}

// =============================================================================
// Rank quality metrics
// =============================================================================

/// Comprehensive rank quality metrics computed against ground truth.
///
/// These go beyond simple precision/recall to capture different failure modes:
/// - **CURL** penalizes high-rank errors more than low-rank errors
/// - **nDCG** is the standard IR metric (lingua franca of ranking evaluation)
/// - **Weighted rank reversals** captures local instability weighted by position
/// - **Bayesian regret** measures expected utility loss from selecting estimated top-K
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RankQualityMetrics {
    /// Kendall tau-b (pairwise concordance, handles ties).
    pub kendall_tau_b: f64,
    /// Spearman rho (monotonic rank correlation).
    pub spearman_rho: f64,
    /// Top-K set precision.
    pub topk_precision: f64,
    /// Top-K set recall.
    pub topk_recall: f64,
    /// Fraction of true scores within 95% confidence intervals.
    pub coverage_95ci: f64,
    /// Normalized Discounted Cumulative Gain at K.
    /// DCG@k = sum_{i=1}^{k} (2^rel_i - 1) / log2(i+1), normalized by ideal DCG.
    pub ndcg_at_k: f64,
    /// CURL with harmonic weights: w(i,j) = 1/(rank_i * rank_j).
    /// Penalizes high-rank errors more severely.
    pub curl_harmonic: f64,
    /// CURL with exponential decay: w(i,j) = exp(-0.1 * min(rank_i, rank_j)).
    pub curl_exponential: f64,
    /// Weighted rank reversals: sum of 1/rank * |displacement| for misranked items.
    pub weighted_rank_reversals: f64,
    /// Bayesian regret: E[U(true_top_k)] - E[U(estimated_top_k)].
    /// Captures how bad the errors are, not just how many.
    pub bayesian_regret: f64,
    /// Number of pairwise discordances in the top-K region.
    pub topk_discordance_count: usize,
}

impl RankQualityMetrics {
    /// Compute all rank quality metrics from predicted and true scores.
    ///
    /// `pred_scores` and `true_scores` must have the same length.
    /// `feasible_indices` restricts evaluation to feasible items.
    /// `k` is the top-K threshold.
    /// `pred_vars` provides variance estimates for coverage computation.
    pub fn compute(
        pred_scores: &[f64],
        true_scores: &[f64],
        pred_vars: &[f64],
        feasible_indices: &[usize],
        k: usize,
    ) -> Self {
        let k = k.min(feasible_indices.len());

        let pred_feas: Vec<f64> = feasible_indices.iter().map(|&i| pred_scores[i]).collect();
        let true_feas: Vec<f64> = feasible_indices.iter().map(|&i| true_scores[i]).collect();

        let kt = if pred_feas.len() >= 2 {
            kendall_tau_b(&pred_feas, &true_feas)
        } else {
            0.0
        };
        let sr = if pred_feas.len() >= 2 {
            spearman_rho(&pred_feas, &true_feas)
        } else {
            0.0
        };

        let tp = topk_precision(pred_scores, true_scores, feasible_indices, k);
        let tr = topk_recall(pred_scores, true_scores, feasible_indices, k);
        let cov = coverage_95(pred_scores, pred_vars, true_scores, feasible_indices);

        let ndcg = compute_ndcg_at_k(pred_scores, true_scores, feasible_indices, k);
        let curl_h = compute_curl(
            pred_scores,
            true_scores,
            feasible_indices,
            CurlWeight::Harmonic,
        );
        let curl_e = compute_curl(
            pred_scores,
            true_scores,
            feasible_indices,
            CurlWeight::Exponential(0.1),
        );
        let wrr = compute_weighted_rank_reversals(pred_scores, true_scores, feasible_indices, k);
        let regret = compute_bayesian_regret(pred_scores, true_scores, feasible_indices, k);
        let disc = compute_topk_discordance(pred_scores, true_scores, feasible_indices, k);

        Self {
            kendall_tau_b: kt,
            spearman_rho: sr,
            topk_precision: tp,
            topk_recall: tr,
            coverage_95ci: cov,
            ndcg_at_k: ndcg,
            curl_harmonic: curl_h,
            curl_exponential: curl_e,
            weighted_rank_reversals: wrr,
            bayesian_regret: regret,
            topk_discordance_count: disc,
        }
    }
}

/// Weighting scheme for CURL computation.
enum CurlWeight {
    /// w(i,j) = 1 / (rank_i * rank_j)
    Harmonic,
    /// w(i,j) = exp(-alpha * min(rank_i, rank_j))
    Exponential(f64),
}

/// Compute CURL (Concordance-based Utility of Ranked Lists).
///
/// Measures rank agreement with position-dependent weighting so that
/// errors among top-ranked items are penalized more heavily.
fn compute_curl(
    pred_scores: &[f64],
    true_scores: &[f64],
    indices: &[usize],
    weight: CurlWeight,
) -> f64 {
    if indices.len() < 2 {
        return 1.0;
    }

    // Sort indices by predicted score (descending) to get predicted ranks.
    let mut pred_order: Vec<usize> = indices.to_vec();
    pred_order.sort_by(|&a, &b| {
        pred_scores[b]
            .partial_cmp(&pred_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Sort indices by true score (descending) to get true ranks.
    let mut true_order: Vec<usize> = indices.to_vec();
    true_order.sort_by(|&a, &b| {
        true_scores[b]
            .partial_cmp(&true_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build rank maps (1-based).
    let mut pred_rank: HashMap<usize, usize> = HashMap::new();
    let mut true_rank: HashMap<usize, usize> = HashMap::new();
    for (r, &idx) in pred_order.iter().enumerate() {
        pred_rank.insert(idx, r + 1);
    }
    for (r, &idx) in true_order.iter().enumerate() {
        true_rank.insert(idx, r + 1);
    }

    let mut weighted_concordant = 0.0;
    let mut weighted_total = 0.0;

    for (ii, &a) in indices.iter().enumerate() {
        for &b in indices.iter().skip(ii + 1) {
            let pr_a = pred_rank[&a] as f64;
            let pr_b = pred_rank[&b] as f64;
            let tr_a = true_rank[&a] as f64;
            let tr_b = true_rank[&b] as f64;

            let w = match weight {
                CurlWeight::Harmonic => 1.0 / (pr_a.min(tr_a) * pr_b.min(tr_b)),
                CurlWeight::Exponential(alpha) => {
                    (-alpha * pr_a.min(tr_a).min(pr_b.min(tr_b))).exp()
                }
            };

            weighted_total += w;

            let pred_diff = pred_scores[a] - pred_scores[b];
            let true_diff = true_scores[a] - true_scores[b];
            let pred_cmp = if pred_diff > 0.0 {
                1
            } else if pred_diff < 0.0 {
                -1
            } else {
                0
            };
            let true_cmp = if true_diff > 0.0 {
                1
            } else if true_diff < 0.0 {
                -1
            } else {
                0
            };
            let concordance_credit = if pred_cmp == 0 && true_cmp == 0 {
                1.0
            } else if pred_cmp == 0 || true_cmp == 0 {
                0.5
            } else if pred_cmp == true_cmp {
                1.0
            } else {
                0.0
            };
            weighted_concordant += w * concordance_credit;
        }
    }

    if weighted_total == 0.0 {
        1.0
    } else {
        weighted_concordant / weighted_total
    }
}

/// Compute nDCG@k.
fn compute_ndcg_at_k(pred_scores: &[f64], true_scores: &[f64], indices: &[usize], k: usize) -> f64 {
    if indices.is_empty() || k == 0 {
        return 0.0;
    }

    // Sort by predicted score (descending) to get the predicted ranking.
    let mut pred_order: Vec<usize> = indices.to_vec();
    pred_order.sort_by(|&a, &b| {
        pred_scores[b]
            .partial_cmp(&pred_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Sort by true score (descending) for ideal ranking.
    let mut ideal_order: Vec<usize> = indices.to_vec();
    ideal_order.sort_by(|&a, &b| {
        true_scores[b]
            .partial_cmp(&true_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Normalize true scores to [0, 1] for relevance.
    let min_true = indices
        .iter()
        .map(|&i| true_scores[i])
        .fold(f64::INFINITY, f64::min);
    let max_true = indices
        .iter()
        .map(|&i| true_scores[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let range = (max_true - min_true).max(1e-9);

    let relevance = |idx: usize| -> f64 { ((true_scores[idx] - min_true) / range).clamp(0.0, 1.0) };

    let k_eff = k.min(pred_order.len());

    let dcg: f64 = pred_order
        .iter()
        .take(k_eff)
        .enumerate()
        .map(|(i, &idx)| {
            let rel = relevance(idx);
            (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2()
        })
        .sum();

    let idcg: f64 = ideal_order
        .iter()
        .take(k_eff)
        .enumerate()
        .map(|(i, &idx)| {
            let rel = relevance(idx);
            (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2()
        })
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        (dcg / idcg).clamp(0.0, 1.0)
    }
}

/// Compute weighted rank reversals in the top-K region.
///
/// For each item in the top-K (by true rank), compute the displacement from its
/// predicted rank, weighted by 1/true_rank. Items that should be in top-K but
/// aren't get maximum displacement.
fn compute_weighted_rank_reversals(
    pred_scores: &[f64],
    true_scores: &[f64],
    indices: &[usize],
    k: usize,
) -> f64 {
    if indices.is_empty() || k == 0 {
        return 0.0;
    }

    let mut pred_order: Vec<usize> = indices.to_vec();
    pred_order.sort_by(|&a, &b| {
        pred_scores[b]
            .partial_cmp(&pred_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut true_order: Vec<usize> = indices.to_vec();
    true_order.sort_by(|&a, &b| {
        true_scores[b]
            .partial_cmp(&true_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = indices.len();
    let k_eff = k.min(n);

    let mut pred_rank_map: HashMap<usize, usize> = HashMap::new();
    for (r, &idx) in pred_order.iter().enumerate() {
        pred_rank_map.insert(idx, r + 1);
    }

    let mut wrr = 0.0;
    for (true_r, &idx) in true_order.iter().take(k_eff).enumerate() {
        let pred_r = pred_rank_map[&idx];
        let displacement = (pred_r as f64 - (true_r + 1) as f64).abs();
        let weight = 1.0 / (true_r + 1) as f64;
        wrr += weight * displacement;
    }

    wrr
}

/// Compute Bayesian regret: the expected utility loss from selecting the
/// predicted top-K instead of the true top-K.
fn compute_bayesian_regret(
    pred_scores: &[f64],
    true_scores: &[f64],
    indices: &[usize],
    k: usize,
) -> f64 {
    if indices.is_empty() || k == 0 {
        return 0.0;
    }

    let pred_topk = topk_set(pred_scores, indices, k, false);
    let true_topk = topk_set(true_scores, indices, k, true);

    let true_utility: f64 = true_topk.iter().map(|&i| true_scores[i]).sum();
    let pred_utility: f64 = pred_topk.iter().map(|&i| true_scores[i]).sum();

    (true_utility - pred_utility).max(0.0)
}

/// Count pairwise discordances among the top-K items.
fn compute_topk_discordance(
    pred_scores: &[f64],
    true_scores: &[f64],
    indices: &[usize],
    k: usize,
) -> usize {
    if indices.is_empty() || k == 0 {
        return 0;
    }

    // Items in the top-K by either predicted or true ranking.
    let pred_topk = topk_set(pred_scores, indices, k, false);
    let true_topk = topk_set(true_scores, indices, k, true);
    let relevant: Vec<usize> = pred_topk.union(&true_topk).copied().collect();

    let mut discordances = 0;
    for (i, &a) in relevant.iter().enumerate() {
        for &b in relevant.iter().skip(i + 1) {
            let pred_cmp = pred_scores[a].partial_cmp(&pred_scores[b]);
            let true_cmp = true_scores[a].partial_cmp(&true_scores[b]);
            if let (Some(p), Some(t)) = (pred_cmp, true_cmp) {
                if p != t && p != std::cmp::Ordering::Equal && t != std::cmp::Ordering::Equal {
                    discordances += 1;
                }
            }
        }
    }

    discordances
}

// =============================================================================
// Synthetic case definitions
// =============================================================================

#[derive(Debug, Clone)]
pub struct SyntheticAttribute {
    pub id: &'static str,
    pub weight: f64,
    pub scores: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SyntheticCase {
    pub name: &'static str,
    pub attributes: Vec<SyntheticAttribute>,
    pub gates: Vec<MultiRerankGateSpec>,
    pub topk: MultiRerankTopKSpec,
    pub comparison_budget: Option<usize>,
    pub latency_budget_ms: Option<u64>,
    pub max_pair_repeats: Option<usize>,
    pub prewarm_pairs_per_attr: usize,
    pub noise_sigma: f64,
    pub refusal_rate: f64,
    pub outlier_rate: f64,
    pub seed: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EvaluationMetrics {
    pub kendall_tau: f64,
    pub spearman_rho: f64,
    pub kendall_tau_all: f64,
    pub spearman_rho_all: f64,
    pub topk_precision: f64,
    pub topk_recall: f64,
    pub coverage_95ci: f64,
    pub gate_precision: Option<f64>,
    pub gate_recall: Option<f64>,
    pub comparisons_attempted: usize,
    pub comparisons_used: usize,
    pub comparisons_refused: usize,
    pub stop_reason: RerankStopReason,
    pub latency_ms: u128,
    /// Extended rank quality metrics (populated when ground truth is available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank_quality: Option<RankQualityMetrics>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EvaluationResult {
    pub case_name: String,
    pub pairwise_mode: SyntheticPairwiseMode,
    pub metrics: EvaluationMetrics,
    pub error_trajectory: Vec<f64>,
}

// =============================================================================
// Likert baseline evaluation
// =============================================================================

/// Configuration for the Likert baseline simulator.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct LikertEvalConfig {
    /// Number of discrete levels in the Likert scale (e.g. 5 or 10).
    pub levels: usize,
    /// Multiplies the synthetic `comparison_budget` when allocating Likert ratings.
    ///
    /// Use this to explore fairness regimes, e.g.:
    /// - `1.0`: equal number of model calls (pairwise comparisons vs per-item ratings)
    /// - `2.0`: rough proxy for equal "entity reads" (pairwise prompts contain 2 entities)
    pub budget_multiplier: f64,
}

impl Default for LikertEvalConfig {
    fn default() -> Self {
        Self {
            levels: 10,
            budget_multiplier: 1.0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LikertEvaluationMetrics {
    pub kendall_tau: f64,
    pub spearman_rho: f64,
    pub kendall_tau_all: f64,
    pub spearman_rho_all: f64,
    pub topk_precision: f64,
    pub topk_recall: f64,
    pub coverage_95ci: f64,
    pub gate_precision: Option<f64>,
    pub gate_recall: Option<f64>,
    pub ratings_attempted: usize,
    pub ratings_used: usize,
    pub ratings_refused: usize,
    pub latency_ms: u128,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LikertEvaluationResult {
    pub case_name: String,
    /// Final metrics at the end of the allocated rating budget.
    pub metrics: LikertEvaluationMetrics,
    /// Trajectory of `1 - topk_precision` as more ratings are collected.
    pub error_trajectory: Vec<f64>,
}

#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonOutcome {
    CardinalWin,
    LikertWin,
    Tie,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ComparisonMetricDelta {
    pub cardinal: f64,
    pub likert: f64,
    /// Cardinal minus Likert. Positive is better for quality metrics; for
    /// `comparisons_used`, lower resource use is better and `outcome` applies
    /// that direction explicitly.
    pub delta: f64,
    pub higher_is_better: bool,
    pub outcome: ComparisonOutcome,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EvaluationComparisonDeltas {
    pub topk_precision: ComparisonMetricDelta,
    pub topk_recall: ComparisonMetricDelta,
    pub kendall_tau_b: ComparisonMetricDelta,
    pub coverage_95ci: ComparisonMetricDelta,
    pub comparisons_used: ComparisonMetricDelta,
}

#[derive(Debug, Clone, serde::Serialize, Default)]
pub struct EvaluationWinLossTie {
    pub cardinal_wins: usize,
    pub likert_wins: usize,
    pub ties: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EvaluationComparisonMetricValues {
    pub topk_precision: f64,
    pub topk_recall: f64,
    pub kendall_tau_b: f64,
    pub coverage_95ci: f64,
    pub comparisons_used: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EvaluationComparisonCase {
    pub case_name: String,
    pub cardinal: EvaluationComparisonMetricValues,
    pub likert: EvaluationComparisonMetricValues,
    pub cardinal_minus_likert: EvaluationComparisonDeltas,
    pub win_loss_tie: EvaluationWinLossTie,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EvaluationComparisonSummary {
    pub pairwise_config: PairwiseEvalConfig,
    pub likert_config: LikertEvalConfig,
    pub metric_names: [&'static str; 5],
    pub aggregate_win_loss_tie: EvaluationWinLossTie,
    pub cases: Vec<EvaluationComparisonCase>,
}

// =============================================================================
// Public API
// =============================================================================

pub fn synthetic_cases() -> Vec<SyntheticCase> {
    let topk = default_topk(5);

    vec![
        SyntheticCase {
            name: "clean_ordering_10",
            attributes: vec![SyntheticAttribute {
                id: "attr_0",
                weight: 1.0,
                scores: (0..10).map(|i| 10.0 - i as f64).collect(),
            }],
            gates: vec![],
            topk: topk.clone(),
            comparison_budget: None,
            latency_budget_ms: None,
            max_pair_repeats: None,
            prewarm_pairs_per_attr: 0,
            noise_sigma: 0.0,
            refusal_rate: 0.0,
            outlier_rate: 0.0,
            seed: 42,
        },
        SyntheticCase {
            name: "noisy_ordering_50",
            attributes: vec![SyntheticAttribute {
                id: "attr_0",
                weight: 1.0,
                scores: (0..50).map(|i| 50.0 - i as f64).collect(),
            }],
            gates: vec![],
            topk: default_topk(10),
            comparison_budget: None,
            latency_budget_ms: None,
            max_pair_repeats: None,
            prewarm_pairs_per_attr: 0,
            noise_sigma: 0.5,
            refusal_rate: 0.0,
            outlier_rate: 0.0,
            seed: 43,
        },
        SyntheticCase {
            name: "multi_attr_weighted_20",
            attributes: vec![
                SyntheticAttribute {
                    id: "attr_linear",
                    weight: 1.0,
                    scores: (0..20).map(|i| 20.0 - i as f64).collect(),
                },
                SyntheticAttribute {
                    id: "attr_quadratic",
                    weight: 0.6,
                    scores: (0..20)
                        .map(|i| {
                            let v = 20.0 - i as f64;
                            v * v
                        })
                        .collect(),
                },
                SyntheticAttribute {
                    id: "attr_random",
                    weight: 0.4,
                    scores: seeded_random_scores(20, 123),
                },
            ],
            gates: vec![],
            topk: default_topk(5),
            comparison_budget: None,
            latency_budget_ms: None,
            max_pair_repeats: None,
            prewarm_pairs_per_attr: 0,
            noise_sigma: 0.3,
            refusal_rate: 0.0,
            outlier_rate: 0.0,
            seed: 44,
        },
        SyntheticCase {
            name: "clustered_scores_30",
            attributes: vec![SyntheticAttribute {
                id: "attr_cluster",
                weight: 1.0,
                scores: [vec![10.0; 10], vec![5.0; 10], vec![0.0; 10]].concat(),
            }],
            gates: vec![],
            topk: default_topk(10),
            comparison_budget: None,
            latency_budget_ms: None,
            max_pair_repeats: None,
            prewarm_pairs_per_attr: 0,
            noise_sigma: 0.2,
            refusal_rate: 0.0,
            outlier_rate: 0.0,
            seed: 45,
        },
        SyntheticCase {
            name: "scale_compression_40",
            attributes: vec![SyntheticAttribute {
                id: "attr_compressed",
                weight: 1.0,
                // One extreme item makes 10-level direct ratings collapse the
                // remaining candidates into the same bucket.  The best
                // non-outliers are deliberately placed late so tie-breaking by
                // input order is not accidentally correct.
                scores: [
                    vec![1000.0],
                    (0..31).map(|i| 1.0 + (i as f64 * 0.03)).collect::<Vec<_>>(),
                    vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
                ]
                .concat(),
            }],
            gates: vec![],
            topk: default_topk(5),
            comparison_budget: Some(160),
            latency_budget_ms: None,
            max_pair_repeats: None,
            prewarm_pairs_per_attr: 0,
            noise_sigma: 0.0,
            refusal_rate: 0.0,
            outlier_rate: 0.0,
            seed: 49,
        },
        // This is intentionally adversarial: sparse active comparisons plus
        // high-confidence flipped pairwise observations.  It is a regression
        // target, not evidence that the robust solver already beats direct
        // scoring under outliers.
        SyntheticCase {
            name: "outlier_robustness_25",
            attributes: vec![SyntheticAttribute {
                id: "attr_0",
                weight: 1.0,
                scores: (0..25).map(|i| 25.0 - i as f64).collect(),
            }],
            gates: vec![],
            topk: default_topk(5),
            comparison_budget: None,
            latency_budget_ms: None,
            max_pair_repeats: None,
            prewarm_pairs_per_attr: 0,
            noise_sigma: 0.2,
            refusal_rate: 0.0,
            outlier_rate: 0.1,
            seed: 46,
        },
        SyntheticCase {
            name: "gated_feasibility_30",
            attributes: vec![SyntheticAttribute {
                id: "attr_gate",
                weight: 1.0,
                scores: (0..30).map(|i| 30.0 - i as f64).collect(),
            }],
            gates: vec![MultiRerankGateSpec {
                attribute_id: "attr_gate".to_string(),
                unit: "percentile".to_string(),
                op: ">=".to_string(),
                threshold: 0.6,
            }],
            topk: default_topk(5),
            comparison_budget: None,
            latency_budget_ms: None,
            max_pair_repeats: None,
            prewarm_pairs_per_attr: 80,
            noise_sigma: 0.2,
            refusal_rate: 0.0,
            outlier_rate: 0.0,
            seed: 47,
        },
        SyntheticCase {
            name: "inconsistent_cycle_12",
            attributes: vec![SyntheticAttribute {
                id: "attr_cycle",
                weight: 1.0,
                scores: (0..12).map(|i| 12.0 - i as f64).collect(),
            }],
            gates: vec![],
            topk: default_topk(5),
            comparison_budget: Some(200),
            latency_budget_ms: None,
            max_pair_repeats: None,
            prewarm_pairs_per_attr: 0,
            noise_sigma: 0.8,
            refusal_rate: 0.0,
            outlier_rate: 0.15,
            seed: 48,
        },
    ]
}

pub fn run_synthetic_suite(filter: Option<&str>) -> Result<Vec<EvaluationResult>, EvaluationError> {
    run_synthetic_suite_with_config(filter, PairwiseEvalConfig::default())
}

pub fn run_synthetic_suite_with_config(
    filter: Option<&str>,
    cfg: PairwiseEvalConfig,
) -> Result<Vec<EvaluationResult>, EvaluationError> {
    let cases = synthetic_cases();
    let selected: Vec<SyntheticCase> = match filter {
        Some(name) => cases.into_iter().filter(|c| c.name == name).collect(),
        None => cases,
    };

    selected
        .into_iter()
        .map(|case| run_synthetic_case_with_config(&case, cfg))
        .collect()
}

pub fn run_synthetic_case(case: &SyntheticCase) -> Result<EvaluationResult, EvaluationError> {
    run_synthetic_case_with_config(case, PairwiseEvalConfig::default())
}

pub fn run_synthetic_case_with_config(
    case: &SyntheticCase,
    cfg: PairwiseEvalConfig,
) -> Result<EvaluationResult, EvaluationError> {
    let mut rng = StdRng::seed_from_u64(case.seed);

    let n_entities = case.attributes.first().map(|a| a.scores.len()).unwrap_or(0);

    let n_attributes = case.attributes.len();
    let comparison_budget = case
        .comparison_budget
        .unwrap_or_else(|| default_comparison_budget(n_entities, n_attributes));
    let latency_budget = case.latency_budget_ms.map(std::time::Duration::from_millis);

    let attributes_cfg: Vec<AttributeConfig> = case
        .attributes
        .iter()
        .map(|attr| AttributeConfig::new(attr.id, attr.weight))
        .collect();

    let topk_cfg = TopKConfig {
        k: case.topk.k,
        weight_exponent: case.topk.weight_exponent,
        tolerated_error: case.topk.tolerated_error,
        band_size: case.topk.band_size,
        effective_resistance_max_active: case.topk.effective_resistance_max_active,
        stop_sigma_inflate: case.topk.stop_sigma_inflate,
        stop_min_consecutive: case.topk.stop_min_consecutive,
        min_explore_degree: case.topk.min_explore_degree,
    };

    let parsed_gates = parse_evaluation_gates(&case.attributes, &case.gates)?;
    let gates_cfg: Vec<GateSpec> = parsed_gates
        .iter()
        .map(|gate| {
            GateSpec::new(
                gate.attribute_id,
                gate.unit.as_str(),
                gate.op.as_str(),
                gate.threshold,
            )
        })
        .collect();

    let config = TraitSearchConfig::new(n_entities, attributes_cfg, topk_cfg.clone(), gates_cfg);

    let mut engines: HashMap<String, RatingEngine> = HashMap::new();
    let mut raters: HashMap<String, RaterParams> = HashMap::new();
    let rater_id = "sim";
    raters.insert(rater_id.to_string(), RaterParams::default());

    let mut engine_cfg = EngineConfig {
        rank_weight_exponent: topk_cfg.weight_exponent,
        top_k: Some(topk_cfg.k),
        ..Default::default()
    };
    if topk_cfg.k > 0 {
        let tail_weight =
            (1.0 / (topk_cfg.k as f64).powf(topk_cfg.weight_exponent)).clamp(0.05, 1.0);
        engine_cfg.tail_weight = tail_weight;
    }

    for attr in &case.attributes {
        let engine = RatingEngine::new(
            n_entities,
            AttributeParams::default(),
            raters.clone(),
            Some(engine_cfg.clone()),
        )
        .expect("initializing rating engine with valid config should succeed");
        engines.insert(attr.id.to_string(), engine);
    }

    let mut manager = TraitSearchManager::new(config, engines)
        .expect("initializing trait search manager should succeed");

    let start_time = Instant::now();

    let mut pair_repeats: HashMap<(usize, usize, usize), f64> = HashMap::new();
    let mut comparisons_attempted = 0usize;
    let mut comparisons_used = 0usize;
    let mut comparisons_refused = 0usize;
    let mut error_trajectory: Vec<f64> = Vec::new();

    let mut refused_pairs: HashSet<(usize, usize, usize)> = HashSet::new();

    let attr_id_to_index: HashMap<&str, usize> = case
        .attributes
        .iter()
        .enumerate()
        .map(|(idx, a)| (a.id, idx))
        .collect();

    // Gates can self-starve if applied before any scores exist. Prewarm comparisons
    // give the solver enough signal so gating reflects actual structure.
    if case.prewarm_pairs_per_attr > 0 {
        for (attr_idx, attr_truth) in case.attributes.iter().enumerate() {
            for _ in 0..case.prewarm_pairs_per_attr {
                let i = rng.gen_range(0..n_entities);
                let mut j = rng.gen_range(0..n_entities);
                if i == j {
                    j = (j + 1) % n_entities;
                }
                let judgement = simulate_pairwise(
                    &mut rng,
                    &attr_truth.scores,
                    i,
                    j,
                    case.noise_sigma,
                    case.outlier_rate,
                    cfg.mode,
                );
                if let PairwiseJudgement::Observation {
                    higher_ranked,
                    ratio,
                    confidence,
                } = judgement
                {
                    let (obs_i, obs_j) = match higher_ranked {
                        HigherRanked::A => (i, j),
                        HigherRanked::B => (j, i),
                    };
                    let obs = Observation::new(obs_i, obs_j, ratio, confidence, rater_id, 1.0);
                    if manager.add_observation(attr_truth.id, obs).is_ok() {
                        comparisons_used += 1;
                    }
                    *pair_repeats
                        .entry((attr_idx, i.min(j), i.max(j)))
                        .or_insert(0.0) += 1.0;
                } else {
                    comparisons_refused += 1;
                }
                comparisons_attempted += 1;
            }
        }
    }

    #[derive(Clone, Copy)]
    struct CompareTask {
        key: (usize, usize, usize),
        attr_idx: usize,
        i: usize,
        j: usize,
    }

    let stop_reason = 'rerank: loop {
        manager
            .recompute_global_state()
            .expect("recomputing global state should succeed with valid data");
        error_trajectory.push(manager.estimate_topk_error());

        if manager.certified_stop() {
            break 'rerank RerankStopReason::CertifiedStop;
        }
        if manager.estimate_topk_error() <= topk_cfg.tolerated_error {
            break 'rerank RerankStopReason::ToleratedErrorMet;
        }
        if comparisons_attempted >= comparison_budget {
            break 'rerank RerankStopReason::BudgetExhausted;
        }
        if let Some(limit) = latency_budget {
            if start_time.elapsed() >= limit {
                break 'rerank RerankStopReason::LatencyBudgetExceeded;
            }
        }

        let remaining_budget = comparison_budget.saturating_sub(comparisons_attempted);
        if remaining_budget == 0 {
            break 'rerank RerankStopReason::BudgetExhausted;
        }

        let batch_size = DEFAULT_BATCH_SIZE.min(remaining_budget);
        // Ask for the number of proposals we will actually execute.  Oversampling
        // here changes the planner policy because forced exploration is budgeted
        // inside `propose_batch`; a 3x request turns the first executed batch into
        // mostly exploration rather than the configured frontier/exploration mix.
        let proposals = manager
            .propose_batch(rater_id, batch_size, PlannerMode::Hybrid)
            .expect("generating trait search proposals should succeed");

        if proposals.is_empty() {
            break 'rerank RerankStopReason::NoProposals;
        }

        let mut batch_seen: HashSet<(usize, usize, usize)> = HashSet::new();
        let mut tasks: Vec<CompareTask> = Vec::with_capacity(batch_size);

        for proposal in proposals {
            let attr_id = proposal.attribute_id.as_str();
            let Some(&attr_idx) = attr_id_to_index.get(attr_id) else {
                continue;
            };

            let i = proposal.i;
            let j = proposal.j;
            if i >= n_entities || j >= n_entities {
                continue;
            }

            let (a, b) = if i <= j { (i, j) } else { (j, i) };
            let key = (attr_idx, a, b);

            if refused_pairs.contains(&key) {
                continue;
            }
            if !batch_seen.insert(key) {
                continue;
            }
            if let Some(max) = case.max_pair_repeats {
                if pair_repeats.get(&key).copied().unwrap_or(0.0) >= max as f64 {
                    continue;
                }
            }

            tasks.push(CompareTask {
                key,
                attr_idx,
                i,
                j,
            });
            if tasks.len() >= batch_size {
                break;
            }
        }

        if tasks.is_empty() {
            break 'rerank RerankStopReason::NoNewPairs;
        }

        for task in tasks {
            comparisons_attempted += 1;

            if rng.gen::<f64>() < case.refusal_rate {
                comparisons_refused += 1;
                refused_pairs.insert(task.key);
                continue;
            }

            let attr_truth = &case.attributes[task.attr_idx];
            let judgement = simulate_pairwise(
                &mut rng,
                &attr_truth.scores,
                task.i,
                task.j,
                case.noise_sigma,
                case.outlier_rate,
                cfg.mode,
            );

            match judgement {
                PairwiseJudgement::Refused => {
                    comparisons_refused += 1;
                    refused_pairs.insert(task.key);
                }
                PairwiseJudgement::Observation {
                    higher_ranked,
                    ratio,
                    confidence,
                } => {
                    let (obs_i, obs_j) = match higher_ranked {
                        HigherRanked::A => (task.i, task.j),
                        HigherRanked::B => (task.j, task.i),
                    };
                    let obs = Observation::new(obs_i, obs_j, ratio, confidence, rater_id, 1.0);
                    if manager.add_observation(attr_truth.id, obs).is_ok() {
                        comparisons_used += 1;
                    }

                    *pair_repeats.entry(task.key).or_insert(0.0) += 1.0;
                }
            }
        }
    };

    manager
        .recompute_global_state()
        .expect("final recomputation of global state should succeed");

    let n = n_entities;
    let mut pred_scores = vec![0.0; n];
    let mut pred_vars = vec![0.0; n];
    let mut pred_feasible = vec![true; n];
    for (i, (score, (var, feasible))) in pred_scores
        .iter_mut()
        .zip(pred_vars.iter_mut().zip(pred_feasible.iter_mut()))
        .enumerate()
    {
        let state = manager.entity_state(i);
        *score = state.u_mean;
        *var = state.u_var;
        *feasible = state.feasible;
    }

    let true_scores = compute_ground_truth_scores(&case.attributes);
    let true_feasible = compute_true_feasible(&case.attributes, &case.gates)?;

    let eval_indices: Vec<usize> = (0..n).filter(|&i| pred_feasible[i]).collect();

    let kendall_tau_all = kendall_tau_b(&pred_scores, &true_scores);
    let spearman_rho_all = spearman_rho(&pred_scores, &true_scores);

    // Gate-aware top-k metrics intentionally compare the predicted feasible
    // frontier against the true feasible frontier.  Otherwise a false
    // infeasible gate decision would remove a missed true top-k item from the
    // denominator and overstate performance on gated synthetic cases.
    let true_eval_indices: Vec<usize> = (0..n).filter(|&i| true_feasible[i]).collect();

    let (kendall_tau, spearman_rho, topk_precision, topk_recall) = if eval_indices.len() >= 2 {
        let pred_eval: Vec<f64> = eval_indices.iter().map(|&i| pred_scores[i]).collect();
        let true_eval: Vec<f64> = eval_indices.iter().map(|&i| true_scores[i]).collect();
        (
            kendall_tau_b(&pred_eval, &true_eval),
            spearman_rho(&pred_eval, &true_eval),
            topk_precision_against_truth(
                &pred_scores,
                &true_scores,
                &eval_indices,
                &true_eval_indices,
                case.topk.k,
            ),
            topk_recall_against_truth(
                &pred_scores,
                &true_scores,
                &eval_indices,
                &true_eval_indices,
                case.topk.k,
            ),
        )
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    let coverage_95ci = coverage_95(&pred_scores, &pred_vars, &true_scores, &eval_indices);

    let (gate_precision, gate_recall) = if case.gates.is_empty() {
        (None, None)
    } else {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;
        for (&pred, &truth) in pred_feasible.iter().zip(true_feasible.iter()) {
            match (pred, truth) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }
        let precision = if tp + fp > 0 {
            Some(tp as f64 / (tp + fp) as f64)
        } else {
            Some(0.0)
        };
        let recall = if tp + fn_ > 0 {
            Some(tp as f64 / (tp + fn_) as f64)
        } else {
            Some(0.0)
        };
        (precision, recall)
    };

    let rank_quality = if eval_indices.len() >= 2 {
        Some(RankQualityMetrics::compute(
            &pred_scores,
            &true_scores,
            &pred_vars,
            &eval_indices,
            case.topk.k,
        ))
    } else {
        None
    };

    Ok(EvaluationResult {
        case_name: case.name.to_string(),
        pairwise_mode: cfg.mode,
        metrics: EvaluationMetrics {
            kendall_tau,
            spearman_rho,
            kendall_tau_all,
            spearman_rho_all,
            topk_precision,
            topk_recall,
            coverage_95ci,
            gate_precision,
            gate_recall,
            comparisons_attempted,
            comparisons_used,
            comparisons_refused,
            stop_reason,
            latency_ms: 0,
            rank_quality,
        },
        error_trajectory,
    })
}

/// Run a single synthetic case through a simulated per-item Likert baseline.
///
/// This baseline estimates each attribute score directly for each item on a bounded
/// integer scale, then combines attributes with the same weighting and gating logic
/// used in pairwise evaluation metrics.
pub fn run_likert_baseline_case(
    case: &SyntheticCase,
    cfg: LikertEvalConfig,
) -> Result<LikertEvaluationResult, EvaluationError> {
    let mut rng = StdRng::seed_from_u64(case.seed ^ 0x9E37_79B9_7F4A_7C15);
    let n = case.attributes.first().map(|a| a.scores.len()).unwrap_or(0);
    let levels = cfg.levels.max(2);

    let n_attributes = case.attributes.len();
    let pairwise_budget = case
        .comparison_budget
        .unwrap_or_else(|| default_comparison_budget(n, n_attributes));
    let rating_budget = ((pairwise_budget as f64) * cfg.budget_multiplier)
        .round()
        .max(1.0) as usize;

    let mut sums: HashMap<&str, Vec<f64>> = HashMap::new();
    let mut counts: HashMap<&str, Vec<u32>> = HashMap::new();
    for attr in &case.attributes {
        sums.insert(attr.id, vec![0.0; n]);
        counts.insert(attr.id, vec![0; n]);
    }

    let mut ratings_attempted = 0usize;
    let mut ratings_used = 0usize;
    let mut ratings_refused = 0usize;
    let mut error_trajectory = Vec::new();

    let stride = n.saturating_mul(n_attributes).max(1);

    for step in 0..rating_budget {
        let slot = step % stride;
        let attr_idx = (slot / n.max(1)).min(n_attributes.saturating_sub(1));
        let entity_idx = if n == 0 { 0 } else { slot % n };
        let attr = &case.attributes[attr_idx];

        ratings_attempted += 1;
        if rng.gen::<f64>() < case.refusal_rate {
            ratings_refused += 1;
        } else {
            let maybe = simulate_likert_rating(
                &mut rng,
                &attr.scores,
                entity_idx,
                levels,
                case.noise_sigma,
                case.outlier_rate,
            );
            if let Some(r) = maybe {
                let s = sums
                    .get_mut(attr.id)
                    .expect("sum map invariant violated for likert baseline");
                let c = counts
                    .get_mut(attr.id)
                    .expect("count map invariant violated for likert baseline");
                s[entity_idx] += r as f64;
                c[entity_idx] += 1;
                ratings_used += 1;
            } else {
                ratings_refused += 1;
            }
        }

        // Track the same gate-aware error used by the final metrics: predicted
        // top-k comes from the current predicted-feasible set, but the target
        // top-k comes from the true feasible set.  Using one shared feasible
        // set here would let gate false-negatives erase items from the target
        // denominator and make gated synthetic cases look better than they are.
        if (step + 1) % stride == 0 || step + 1 == rating_budget {
            let (pred_scores, _, pred_feasible) =
                infer_scores_from_likert(case, levels, &sums, &counts, false)?;
            let true_scores = compute_ground_truth_scores(&case.attributes);
            let true_feasible = compute_true_feasible(&case.attributes, &case.gates)?;
            let pred_indices: Vec<usize> = (0..n).filter(|&i| pred_feasible[i]).collect();
            let true_indices: Vec<usize> = (0..n).filter(|&i| true_feasible[i]).collect();
            let p = topk_precision_against_truth(
                &pred_scores,
                &true_scores,
                &pred_indices,
                &true_indices,
                case.topk.k,
            );
            error_trajectory.push((1.0 - p).clamp(0.0, 1.0));
        }
    }

    let (pred_scores, pred_vars, pred_feasible) =
        infer_scores_from_likert(case, levels, &sums, &counts, true)?;
    let true_scores = compute_ground_truth_scores(&case.attributes);
    let true_feasible = compute_true_feasible(&case.attributes, &case.gates)?;
    let eval_indices: Vec<usize> = (0..n).filter(|&i| pred_feasible[i]).collect();

    let kendall_tau_all = kendall_tau_b(&pred_scores, &true_scores);
    let spearman_rho_all = spearman_rho(&pred_scores, &true_scores);

    let true_eval_indices: Vec<usize> = (0..n).filter(|&i| true_feasible[i]).collect();

    let (kendall_tau, spearman_rho, topk_precision_v, topk_recall_v) = if eval_indices.len() >= 2 {
        let pred_eval: Vec<f64> = eval_indices.iter().map(|&i| pred_scores[i]).collect();
        let true_eval: Vec<f64> = eval_indices.iter().map(|&i| true_scores[i]).collect();
        (
            kendall_tau_b(&pred_eval, &true_eval),
            spearman_rho(&pred_eval, &true_eval),
            topk_precision_against_truth(
                &pred_scores,
                &true_scores,
                &eval_indices,
                &true_eval_indices,
                case.topk.k,
            ),
            topk_recall_against_truth(
                &pred_scores,
                &true_scores,
                &eval_indices,
                &true_eval_indices,
                case.topk.k,
            ),
        )
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    let coverage_95ci = coverage_95(&pred_scores, &pred_vars, &true_scores, &eval_indices);

    let (gate_precision, gate_recall) = if case.gates.is_empty() {
        (None, None)
    } else {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;
        for (&pred, &truth) in pred_feasible.iter().zip(true_feasible.iter()) {
            match (pred, truth) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }
        let precision = if tp + fp > 0 {
            Some(tp as f64 / (tp + fp) as f64)
        } else {
            Some(0.0)
        };
        let recall = if tp + fn_ > 0 {
            Some(tp as f64 / (tp + fn_) as f64)
        } else {
            Some(0.0)
        };
        (precision, recall)
    };

    Ok(LikertEvaluationResult {
        case_name: case.name.to_string(),
        metrics: LikertEvaluationMetrics {
            kendall_tau,
            spearman_rho,
            kendall_tau_all,
            spearman_rho_all,
            topk_precision: topk_precision_v,
            topk_recall: topk_recall_v,
            coverage_95ci,
            gate_precision,
            gate_recall,
            ratings_attempted,
            ratings_used,
            ratings_refused,
            latency_ms: 0,
        },
        error_trajectory,
    })
}

/// Run the Likert baseline across selected synthetic cases.
pub fn run_likert_baseline_suite(
    filter: Option<&str>,
    cfg: LikertEvalConfig,
) -> Result<Vec<LikertEvaluationResult>, EvaluationError> {
    let cases = synthetic_cases();
    let selected: Vec<SyntheticCase> = match filter {
        Some(name) => cases.into_iter().filter(|c| c.name == name).collect(),
        None => cases,
    };

    selected
        .into_iter()
        .map(|case| run_likert_baseline_case(&case, cfg))
        .collect()
}

/// Run pairwise cardinal and Likert baselines side-by-side and emit mechanical deltas.
pub fn run_evaluation_comparison_summary(
    filter: Option<&str>,
    likert_cfg: LikertEvalConfig,
) -> Result<EvaluationComparisonSummary, EvaluationError> {
    run_evaluation_comparison_summary_with_config(filter, PairwiseEvalConfig::default(), likert_cfg)
}

pub fn run_evaluation_comparison_summary_with_config(
    filter: Option<&str>,
    pairwise_cfg: PairwiseEvalConfig,
    likert_cfg: LikertEvalConfig,
) -> Result<EvaluationComparisonSummary, EvaluationError> {
    let cardinal_results = run_synthetic_suite_with_config(filter, pairwise_cfg)?;
    let likert_results = run_likert_baseline_suite(filter, likert_cfg)?;
    let likert_by_case: HashMap<&str, &LikertEvaluationResult> = likert_results
        .iter()
        .map(|result| (result.case_name.as_str(), result))
        .collect();

    let mut aggregate_win_loss_tie = EvaluationWinLossTie::default();
    let mut cases = Vec::with_capacity(cardinal_results.len());

    for cardinal in cardinal_results {
        let likert = likert_by_case
            .get(cardinal.case_name.as_str())
            .expect("synthetic and Likert suites should select identical case names");
        let cardinal_metrics = cardinal.metrics;
        let likert_metrics = likert.metrics.clone();
        let cardinal_values = EvaluationComparisonMetricValues {
            topk_precision: cardinal_metrics.topk_precision,
            topk_recall: cardinal_metrics.topk_recall,
            kendall_tau_b: cardinal_metrics.kendall_tau,
            coverage_95ci: cardinal_metrics.coverage_95ci,
            comparisons_used: cardinal_metrics.comparisons_used as f64,
        };
        let likert_values = EvaluationComparisonMetricValues {
            topk_precision: likert_metrics.topk_precision,
            topk_recall: likert_metrics.topk_recall,
            kendall_tau_b: likert_metrics.kendall_tau,
            coverage_95ci: likert_metrics.coverage_95ci,
            comparisons_used: likert_metrics.ratings_used as f64,
        };
        let cardinal_minus_likert = EvaluationComparisonDeltas {
            topk_precision: compare_metric_delta(
                cardinal_values.topk_precision,
                likert_values.topk_precision,
                true,
            ),
            topk_recall: compare_metric_delta(
                cardinal_values.topk_recall,
                likert_values.topk_recall,
                true,
            ),
            kendall_tau_b: compare_metric_delta(
                cardinal_values.kendall_tau_b,
                likert_values.kendall_tau_b,
                true,
            ),
            coverage_95ci: compare_metric_delta(
                cardinal_values.coverage_95ci,
                likert_values.coverage_95ci,
                true,
            ),
            comparisons_used: compare_metric_delta(
                cardinal_values.comparisons_used,
                likert_values.comparisons_used,
                false,
            ),
        };
        let win_loss_tie = win_loss_tie_for_deltas(&cardinal_minus_likert);
        add_win_loss_tie(&mut aggregate_win_loss_tie, &win_loss_tie);
        cases.push(EvaluationComparisonCase {
            case_name: cardinal.case_name,
            cardinal: cardinal_values,
            likert: likert_values,
            cardinal_minus_likert,
            win_loss_tie,
        });
    }

    Ok(EvaluationComparisonSummary {
        pairwise_config: pairwise_cfg,
        likert_config: likert_cfg,
        metric_names: [
            "topk_precision",
            "topk_recall",
            "kendall_tau_b",
            "coverage_95ci",
            "comparisons_used",
        ],
        aggregate_win_loss_tie,
        cases,
    })
}

// =============================================================================
// Helpers
const COMPARISON_TIE_EPSILON: f64 = 1e-12;

fn compare_metric_delta(
    cardinal: f64,
    likert: f64,
    higher_is_better: bool,
) -> ComparisonMetricDelta {
    let delta = cardinal - likert;
    let outcome = if delta.abs() <= COMPARISON_TIE_EPSILON {
        ComparisonOutcome::Tie
    } else if (delta > 0.0) == higher_is_better {
        ComparisonOutcome::CardinalWin
    } else {
        ComparisonOutcome::LikertWin
    };

    ComparisonMetricDelta {
        cardinal,
        likert,
        delta,
        higher_is_better,
        outcome,
    }
}

fn win_loss_tie_for_deltas(deltas: &EvaluationComparisonDeltas) -> EvaluationWinLossTie {
    let mut counts = EvaluationWinLossTie::default();
    for outcome in [
        deltas.topk_precision.outcome,
        deltas.topk_recall.outcome,
        deltas.kendall_tau_b.outcome,
        deltas.coverage_95ci.outcome,
        deltas.comparisons_used.outcome,
    ] {
        match outcome {
            ComparisonOutcome::CardinalWin => counts.cardinal_wins += 1,
            ComparisonOutcome::LikertWin => counts.likert_wins += 1,
            ComparisonOutcome::Tie => counts.ties += 1,
        }
    }
    counts
}

fn add_win_loss_tie(total: &mut EvaluationWinLossTie, case: &EvaluationWinLossTie) {
    total.cardinal_wins += case.cardinal_wins;
    total.likert_wins += case.likert_wins;
    total.ties += case.ties;
}

// =============================================================================

const DEFAULT_BATCH_SIZE: usize = 32;

fn default_comparison_budget(n_entities: usize, n_attributes: usize) -> usize {
    4usize
        .saturating_mul(n_entities.max(1))
        .saturating_mul(n_attributes.max(1))
}

fn default_topk(k: usize) -> MultiRerankTopKSpec {
    MultiRerankTopKSpec {
        k,
        weight_exponent: 1.3,
        tolerated_error: 0.1,
        band_size: 5,
        effective_resistance_max_active: 64,
        stop_sigma_inflate: 1.25,
        stop_min_consecutive: 2,
        min_explore_degree: 2,
    }
}

fn seeded_random_scores(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(0.0..1.0)).collect()
}

fn infer_scores_from_likert(
    case: &SyntheticCase,
    levels: usize,
    sums: &HashMap<&str, Vec<f64>>,
    counts: &HashMap<&str, Vec<u32>>,
    strict_for_gates: bool,
) -> Result<LikertInference, EvaluationError> {
    let n = case.attributes.first().map(|a| a.scores.len()).unwrap_or(0);
    let levels = levels.max(2);
    let mid = (levels as f64 + 1.0) / 2.0;
    let range = (levels as f64 - 1.0).max(1.0);

    // Use a crude uncertainty model:
    // - variance of a single Likert draw scales with the assumed noise level
    // - missing ratings get a broad prior variance
    let noise_std = case.noise_sigma.abs() * range / 2.0;
    let noise_var = noise_std * noise_std;
    let prior_var = (range * range) / 4.0;

    let mut per_attr_units: HashMap<&str, AttrUnits> = HashMap::new();
    let mut per_attr_vars: HashMap<&str, Vec<f64>> = HashMap::new();
    let mut per_attr_scales: HashMap<&str, f64> = HashMap::new();

    for attr in &case.attributes {
        let s = sums
            .get(attr.id)
            .expect("sum map invariant violated for likert baseline");
        let c = counts
            .get(attr.id)
            .expect("count map invariant violated for likert baseline");

        let mut means = vec![mid; n];
        let mut mean_vars = vec![prior_var; n];

        for ((mean, mean_var), (&sum, &count)) in means
            .iter_mut()
            .zip(mean_vars.iter_mut())
            .zip(s.iter().zip(c.iter()))
        {
            if count > 0 {
                *mean = sum / count as f64;
                let denom = count as f64;
                let v = if noise_var > 0.0 {
                    noise_var / denom
                } else {
                    0.0
                };
                *mean_var = v.max(0.0);
            }
        }

        let (scale, z, min_norm, pct) = compute_attribute_units(&means);
        per_attr_units.insert(attr.id, (means, z, min_norm, pct));
        per_attr_vars.insert(attr.id, mean_vars);
        per_attr_scales.insert(attr.id, scale.max(1e-6));
    }

    let mut u_mean = vec![0.0; n];
    let mut u_var = vec![0.0; n];
    for attr in &case.attributes {
        let (means, _, _, _) = per_attr_units
            .get(attr.id)
            .expect("units map invariant violated for likert baseline");
        let mean_vars = per_attr_vars
            .get(attr.id)
            .expect("var map invariant violated for likert baseline");
        let scale = *per_attr_scales
            .get(attr.id)
            .expect("scale map invariant violated for likert baseline");
        let inv = 1.0 / scale;
        let inv2 = inv * inv;
        let w = attr.weight;
        let w2 = w * w;
        for ((u, var), (&mean, &mean_var)) in u_mean
            .iter_mut()
            .zip(u_var.iter_mut())
            .zip(means.iter().zip(mean_vars.iter()))
        {
            *u += w * (mean * inv);
            *var += w2 * (mean_var.max(0.0) * inv2);
        }
    }

    let mut feasible = vec![true; n];
    let parsed_gates = parse_evaluation_gates(&case.attributes, &case.gates)?;
    for gate in parsed_gates {
        let (latent, z, min_norm, pct) = per_attr_units
            .get(gate.attribute_id)
            .expect("gate attribute missing in likert baseline");
        let c = counts
            .get(gate.attribute_id)
            .expect("gate count missing in likert baseline");
        for (i, feasible_i) in feasible.iter_mut().enumerate() {
            if strict_for_gates && c[i] == 0 {
                *feasible_i = false;
                continue;
            }
            let value = gate.unit.select(latent[i], z[i], min_norm[i], pct[i]);
            let pass = gate.op.passes(value, gate.threshold);
            *feasible_i &= pass;
        }
    }

    Ok((u_mean, u_var, feasible))
}

fn simulate_pairwise(
    rng: &mut impl Rng,
    truth_scores: &[f64],
    i: usize,
    j: usize,
    noise_sigma: f64,
    outlier_rate: f64,
    mode: SyntheticPairwiseMode,
) -> PairwiseJudgement {
    if i >= truth_scores.len() || j >= truth_scores.len() {
        return PairwiseJudgement::Refused;
    }

    let (scale, _, _, _) = compute_attribute_units(truth_scores);
    let min_val = truth_scores
        .iter()
        .fold(f64::INFINITY, |acc, v| acc.min(*v));
    let shift = if min_val <= 0.0 { -min_val + 1e-3 } else { 0.0 };

    let score_i = truth_scores[i] + shift;
    let score_j = truth_scores[j] + shift;

    let mut ratio_raw = if score_j <= 0.0 {
        26.0
    } else {
        (score_i / score_j).max(1e-6)
    };

    let noise = sample_normal(rng, 0.0, noise_sigma.max(1e-6));
    ratio_raw *= noise.exp();

    let mut higher_ranked = if ratio_raw >= 1.0 {
        HigherRanked::A
    } else {
        HigherRanked::B
    };

    let mut ratio = if ratio_raw >= 1.0 {
        ratio_raw
    } else {
        1.0 / ratio_raw
    };

    let mut confidence = confidence_from_signal((ratio.ln()).abs(), scale.max(1e-6), noise_sigma);

    if outlier_rate > 0.0 && rng.gen::<f64>() < outlier_rate {
        higher_ranked = match higher_ranked {
            HigherRanked::A => HigherRanked::B,
            HigherRanked::B => HigherRanked::A,
        };
        confidence = 0.9;
    }

    if mode == SyntheticPairwiseMode::Ordinal {
        ratio = ORDINAL_SYNTHETIC_RATIO;
        confidence = 0.9;
    }
    ratio = ratio.clamp(1.0, 26.0);

    PairwiseJudgement::Observation {
        higher_ranked,
        ratio,
        confidence,
    }
}

fn simulate_likert_rating(
    rng: &mut impl Rng,
    truth_scores: &[f64],
    i: usize,
    levels: usize,
    noise_sigma: f64,
    outlier_rate: f64,
) -> Option<u32> {
    if i >= truth_scores.len() || truth_scores.is_empty() || levels < 2 {
        return None;
    }

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in truth_scores {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    if !min.is_finite() || !max.is_finite() {
        return None;
    }
    let denom = (max - min).max(1e-9);
    let v = truth_scores[i];
    let norm = if v.is_finite() {
        ((v - min) / denom).clamp(0.0, 1.0)
    } else {
        0.5
    };

    let mut expected = 1.0 + norm * (levels as f64 - 1.0);
    let noise_std = noise_sigma.abs() * (levels as f64 - 1.0) / 2.0;
    expected += sample_normal(rng, 0.0, noise_std.max(1e-9));

    if outlier_rate > 0.0 && rng.gen::<f64>() < outlier_rate {
        expected = rng.gen_range(1.0..=(levels as f64));
    }

    let rating = expected.round().clamp(1.0, levels as f64) as u32;
    Some(rating)
}

fn sample_normal(rng: &mut impl Rng, mean: f64, std: f64) -> f64 {
    if std <= 0.0 {
        return mean;
    }
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mean + z0 * std
}

fn confidence_from_signal(diff_abs: f64, scale: f64, noise_sigma: f64) -> f64 {
    let signal = diff_abs / scale.max(1e-6);
    let denom = signal + noise_sigma.max(1e-6);
    let raw = if denom > 0.0 { signal / denom } else { 0.5 };
    raw.clamp(0.05, 0.98)
}

fn compute_ground_truth_scores(attributes: &[SyntheticAttribute]) -> Vec<f64> {
    let n = attributes.first().map(|a| a.scores.len()).unwrap_or(0);
    let mut u = vec![0.0; n];

    for attr in attributes {
        let (scale, _, _, _) = compute_attribute_units(&attr.scores);
        let inv_scale = 1.0 / scale.max(1e-6);
        for (u_i, &score) in u.iter_mut().zip(attr.scores.iter()) {
            *u_i += attr.weight * (score * inv_scale);
        }
    }

    u
}

fn compute_true_feasible(
    attributes: &[SyntheticAttribute],
    gates: &[MultiRerankGateSpec],
) -> Result<Vec<bool>, EvaluationError> {
    let n = attributes.first().map(|a| a.scores.len()).unwrap_or(0);
    let mut feasible = vec![true; n];

    let mut attr_units: HashMap<&str, AttrUnits> = HashMap::new();
    for attr in attributes {
        let (_scale, z, min_norm, pct) = compute_attribute_units(&attr.scores);
        attr_units.insert(attr.id, (attr.scores.clone(), z, min_norm, pct));
    }

    let parsed_gates = parse_evaluation_gates(attributes, gates)?;
    for gate in parsed_gates {
        let Some((latent, z, min_norm, pct)) = attr_units.get(gate.attribute_id) else {
            continue;
        };
        for (i, feasible_i) in feasible.iter_mut().enumerate() {
            let value = gate.unit.select(latent[i], z[i], min_norm[i], pct[i]);
            let pass = gate.op.passes(value, gate.threshold);
            *feasible_i &= pass;
        }
    }

    Ok(feasible)
}

fn kendall_tau_b(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n != y.len() || n < 2 {
        return 0.0;
    }

    let mut concordant = 0f64;
    let mut discordant = 0f64;
    let mut ties_x = 0f64;
    let mut ties_y = 0f64;

    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        for (&xj, &yj) in x.iter().zip(y.iter()).skip(i + 1) {
            let dx = xi - xj;
            let dy = yi - yj;

            if dx == 0.0 && dy == 0.0 {
                continue;
            } else if dx == 0.0 {
                ties_x += 1.0;
            } else if dy == 0.0 {
                ties_y += 1.0;
            } else if (dx > 0.0 && dy > 0.0) || (dx < 0.0 && dy < 0.0) {
                concordant += 1.0;
            } else {
                discordant += 1.0;
            }
        }
    }

    let denom = ((concordant + discordant + ties_x) * (concordant + discordant + ties_y)).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (concordant - discordant) / denom
    }
}

fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n != y.len() || n < 2 {
        return 0.0;
    }
    let rx = ranks_with_ties(x);
    let ry = ranks_with_ties(y);

    let mean_x = rx.iter().sum::<f64>() / n as f64;
    let mean_y = ry.iter().sum::<f64>() / n as f64;

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;

    for (&x_rank, &y_rank) in rx.iter().zip(ry.iter()) {
        let dx = x_rank - mean_x;
        let dy = y_rank - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    if den_x == 0.0 || den_y == 0.0 {
        0.0
    } else {
        num / (den_x.sqrt() * den_y.sqrt())
    }
}

fn ranks_with_ties(scores: &[f64]) -> Vec<f64> {
    let n = scores.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks = vec![0.0; n];
    let mut i = 0usize;
    while i < n {
        let score = scores[indices[i]];
        let mut j = i + 1;
        while j < n && scores[indices[j]] == score {
            j += 1;
        }
        let avg_rank = (i + j - 1) as f64 / 2.0;
        for k in i..j {
            ranks[indices[k]] = avg_rank;
        }
        i = j;
    }

    ranks
}

fn topk_precision(pred_scores: &[f64], true_scores: &[f64], indices: &[usize], k: usize) -> f64 {
    topk_precision_against_truth(pred_scores, true_scores, indices, indices, k)
}

fn topk_recall(pred_scores: &[f64], true_scores: &[f64], indices: &[usize], k: usize) -> f64 {
    topk_recall_against_truth(pred_scores, true_scores, indices, indices, k)
}

fn topk_precision_against_truth(
    pred_scores: &[f64],
    true_scores: &[f64],
    pred_indices: &[usize],
    true_indices: &[usize],
    k: usize,
) -> f64 {
    let pred_set = topk_set(pred_scores, pred_indices, k, false);
    let true_set = topk_set(true_scores, true_indices, k, true);
    if pred_set.is_empty() {
        return 0.0;
    }
    let inter = pred_set.intersection(&true_set).count();
    inter as f64 / pred_set.len() as f64
}

fn topk_recall_against_truth(
    pred_scores: &[f64],
    true_scores: &[f64],
    pred_indices: &[usize],
    true_indices: &[usize],
    k: usize,
) -> f64 {
    let pred_set = topk_set(pred_scores, pred_indices, k, false);
    let true_set = topk_set(true_scores, true_indices, k, true);
    if true_set.is_empty() {
        return 0.0;
    }
    let inter = pred_set.intersection(&true_set).count();
    inter as f64 / true_set.len() as f64
}

fn topk_set(scores: &[f64], indices: &[usize], k: usize, include_ties: bool) -> HashSet<usize> {
    if indices.is_empty() || k == 0 {
        return HashSet::new();
    }

    let mut sorted: Vec<usize> = indices.to_vec();
    sorted.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let k_eff = k.min(sorted.len());

    if !include_ties {
        return sorted.into_iter().take(k_eff).collect();
    }

    let threshold = scores[sorted[k_eff - 1]];
    sorted
        .into_iter()
        .filter(|&i| scores[i] >= threshold)
        .collect()
}

fn coverage_95(
    pred_scores: &[f64],
    pred_vars: &[f64],
    true_scores: &[f64],
    indices: &[usize],
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }

    let z = 1.96;
    let mut covered = 0usize;
    let mut total = 0usize;

    for &i in indices {
        let var = pred_vars[i];
        if !var.is_finite() || var < 0.0 {
            continue;
        }
        let std = var.sqrt();
        let lower = pred_scores[i] - z * std;
        let upper = pred_scores[i] + z * std;
        let truth = true_scores[i];
        if truth.is_finite() {
            total += 1;
            if truth >= lower && truth <= upper {
                covered += 1;
            }
        }
    }

    if total == 0 {
        0.0
    } else {
        covered as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Rank quality metrics
    // =========================================================================

    #[test]
    fn test_rank_quality_perfect_ranking() {
        // Predicted scores perfectly match true scores.
        let pred = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let truth = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let vars = vec![0.1; 5];
        let indices: Vec<usize> = (0..5).collect();

        let rq = RankQualityMetrics::compute(&pred, &truth, &vars, &indices, 3);

        assert!(
            (rq.kendall_tau_b - 1.0).abs() < 1e-6,
            "perfect tau should be 1.0"
        );
        assert!(
            (rq.spearman_rho - 1.0).abs() < 1e-6,
            "perfect rho should be 1.0"
        );
        assert!((rq.topk_precision - 1.0).abs() < 1e-6);
        assert!((rq.topk_recall - 1.0).abs() < 1e-6);
        assert!(
            (rq.ndcg_at_k - 1.0).abs() < 1e-6,
            "perfect nDCG should be 1.0"
        );
        assert!(
            (rq.curl_harmonic - 1.0).abs() < 1e-6,
            "perfect CURL should be 1.0"
        );
        assert!(rq.weighted_rank_reversals < 1e-6, "no reversals expected");
        assert!(rq.bayesian_regret < 1e-6, "no regret expected");
        assert_eq!(rq.topk_discordance_count, 0);
    }

    #[test]
    fn test_rank_quality_reversed_ranking() {
        // Predicted scores are completely reversed.
        let pred = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let truth = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let vars = vec![0.1; 5];
        let indices: Vec<usize> = (0..5).collect();

        let rq = RankQualityMetrics::compute(&pred, &truth, &vars, &indices, 2);

        assert!(
            rq.kendall_tau_b < -0.9,
            "reversed tau should be negative: {}",
            rq.kendall_tau_b
        );
        assert!(rq.spearman_rho < -0.9, "reversed rho should be negative");
        assert!(rq.topk_precision < 0.01, "top-K should be completely wrong");
        assert!(rq.bayesian_regret > 0.0, "should have positive regret");
        assert!(rq.topk_discordance_count > 0, "should have discordances");
    }

    #[test]
    fn test_rank_quality_partial_error() {
        // Items 0 and 1 are swapped, rest correct.
        let pred = vec![8.0, 10.0, 6.0, 4.0, 2.0];
        let truth = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let vars = vec![0.1; 5];
        let indices: Vec<usize> = (0..5).collect();

        let rq = RankQualityMetrics::compute(&pred, &truth, &vars, &indices, 2);

        // Top-2 set is still {0, 1}, just in wrong order.
        assert!(
            (rq.topk_precision - 1.0).abs() < 1e-6,
            "top-2 set should be correct"
        );
        assert!((rq.topk_recall - 1.0).abs() < 1e-6);
        // But there should be rank reversals and discordances within top-K.
        assert!(rq.weighted_rank_reversals > 0.0);
        assert!(rq.topk_discordance_count > 0);
        // CURL should be less than perfect.
        assert!(rq.curl_harmonic < 1.0);
    }

    #[test]
    fn test_ndcg_perfect_vs_random() {
        let truth = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let indices: Vec<usize> = (0..5).collect();

        // Perfect ranking.
        let ndcg_perfect = compute_ndcg_at_k(&truth, &truth, &indices, 3);
        assert!((ndcg_perfect - 1.0).abs() < 1e-6);

        // Worst-case: reverse ranking.
        let pred_worst = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let ndcg_worst = compute_ndcg_at_k(&pred_worst, &truth, &indices, 3);
        assert!(ndcg_worst < ndcg_perfect);
    }

    #[test]
    fn test_curl_monotonicity() {
        // CURL should decrease as ranking quality degrades.
        let truth = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let indices: Vec<usize> = (0..10).collect();

        let curl_perfect = compute_curl(&truth, &truth, &indices, CurlWeight::Harmonic);

        // Swap top 2.
        let pred_swap2 = vec![9.0, 10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let curl_swap2 = compute_curl(&pred_swap2, &truth, &indices, CurlWeight::Harmonic);

        // Swap items 5 and 6 (lower-ranked, should matter less).
        let pred_swap_low = vec![10.0, 9.0, 8.0, 7.0, 6.0, 4.0, 5.0, 3.0, 2.0, 1.0];
        let curl_swap_low = compute_curl(&pred_swap_low, &truth, &indices, CurlWeight::Harmonic);

        assert!((curl_perfect - 1.0).abs() < 1e-6);
        // Swapping top items should hurt CURL more than swapping low items.
        assert!(
            curl_swap2 < curl_swap_low,
            "top swap ({curl_swap2:.4}) should be worse than low swap ({curl_swap_low:.4})"
        );
    }

    #[test]
    fn test_curl_penalizes_one_sided_ties() {
        let truth = vec![4.0, 3.0, 2.0, 1.0];
        let tied_pred = vec![1.0; 4];
        let reversed = vec![1.0, 2.0, 3.0, 4.0];
        let indices: Vec<usize> = (0..4).collect();

        let tied_curl = compute_curl(&tied_pred, &truth, &indices, CurlWeight::Harmonic);
        let reversed_curl = compute_curl(&reversed, &truth, &indices, CurlWeight::Harmonic);

        assert!(
            tied_curl > reversed_curl,
            "ties should be less bad than reversals: tied={tied_curl}, reversed={reversed_curl}"
        );
        assert!(
            tied_curl < 1.0,
            "one-sided predicted ties must not receive full concordance credit"
        );
    }

    #[test]
    fn rank_quality_catches_frontier_order_errors_when_topk_set_is_perfect() {
        let truth = vec![100.0, 90.0, 80.0, 70.0, 60.0, 50.0];
        // Same top-3 set as the truth, but the frontier itself is reversed.
        // Plain top-k precision/recall would call this perfect; rank-quality
        // metrics should still expose the ordering failure a maintainer cares
        // about when a rerank is used as an ordered shortlist.
        let pred_reversed_frontier = vec![80.0, 90.0, 100.0, 70.0, 60.0, 50.0];
        let variances = vec![1.0; truth.len()];
        let indices: Vec<usize> = (0..truth.len()).collect();

        assert!((topk_precision(&pred_reversed_frontier, &truth, &indices, 3) - 1.0).abs() < 1e-6);
        assert!((topk_recall(&pred_reversed_frontier, &truth, &indices, 3) - 1.0).abs() < 1e-6);

        let metrics =
            RankQualityMetrics::compute(&pred_reversed_frontier, &truth, &variances, &indices, 3);
        assert!(
            metrics.ndcg_at_k < 1.0,
            "nDCG must penalize an internally reversed top-k frontier"
        );
        assert!(
            metrics.curl_harmonic < 1.0,
            "CURL must penalize pairwise reversals even with perfect top-k membership"
        );
        assert!(
            metrics.weighted_rank_reversals > 0.0,
            "weighted rank reversals should expose displacement of the true best item"
        );
        assert_eq!(
            metrics.topk_discordance_count, 3,
            "all three pairs inside the reversed top-3 frontier are discordant"
        );
        assert!(
            metrics.bayesian_regret.abs() < 1e-6,
            "same top-k membership has zero set-utility regret; this guard is specifically about order quality"
        );
    }

    #[test]
    fn test_weighted_rank_reversals() {
        let truth = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let indices: Vec<usize> = (0..5).collect();

        // Perfect ranking: no reversals.
        let wrr = compute_weighted_rank_reversals(&truth, &truth, &indices, 3);
        assert!(wrr < 1e-6);

        // Swap items 0 and 4: item 0 (true rank 1) predicted at rank 5.
        let pred_bad = vec![2.0, 8.0, 6.0, 4.0, 10.0];
        let wrr_bad = compute_weighted_rank_reversals(&pred_bad, &truth, &indices, 3);
        assert!(wrr_bad > 0.0, "should have nonzero weighted reversals");
    }

    #[test]
    fn test_bayesian_regret_bounds() {
        let truth = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let indices: Vec<usize> = (0..5).collect();

        // Perfect: zero regret.
        let regret_perfect = compute_bayesian_regret(&truth, &truth, &indices, 2);
        assert!(regret_perfect < 1e-6);

        // Worst case: select bottom-2 instead of top-2.
        let pred_worst = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let regret_worst = compute_bayesian_regret(&pred_worst, &truth, &indices, 2);
        // True top-2: {10, 8} = 18. Predicted top-2: {4, 2} gets true scores = 4+2 = 6. Wait,
        // predicted top-2 by pred_worst are indices 3,4 (scores 8,10 in pred), so true scores
        // for those are 4,2 = 6. True top-2 utility = 10+8 = 18. Regret = 18-6 = 12.
        assert!(
            (regret_worst - 12.0).abs() < 1e-6,
            "regret={regret_worst}, expected 12.0"
        );
    }

    // =========================================================================
    // Synthetic evaluation suite
    // =========================================================================

    #[test]
    fn test_synthetic_suite_runs_all_cases() {
        let results = run_synthetic_suite(None).expect("synthetic suite should run");
        assert!(results.len() >= 6, "should have at least 6 test cases");

        for result in &results {
            assert!(!result.case_name.is_empty());
            assert!(
                result.metrics.comparisons_used > 0,
                "{}: no comparisons used",
                result.case_name
            );
            assert!(result.metrics.kendall_tau_all.is_finite());
            assert!(result.metrics.spearman_rho_all.is_finite());
        }
    }

    #[test]
    fn test_synthetic_clean_ordering_is_accurate() {
        let results =
            run_synthetic_suite(Some("clean_ordering_10")).expect("synthetic suite should run");
        assert_eq!(results.len(), 1);
        let r = &results[0];

        // With no noise, solver should get perfect or near-perfect ranking.
        assert!(
            r.metrics.kendall_tau >= 0.8,
            "clean ordering tau={}, expected >=0.8",
            r.metrics.kendall_tau
        );
        assert!(r.metrics.topk_precision >= 0.8);
    }

    #[test]
    fn test_synthetic_rank_quality_populated() {
        let results =
            run_synthetic_suite(Some("noisy_ordering_50")).expect("synthetic suite should run");
        assert_eq!(results.len(), 1);
        let r = &results[0];

        let rq = r
            .metrics
            .rank_quality
            .as_ref()
            .expect("rank_quality should be populated");
        assert!(rq.ndcg_at_k.is_finite());
        assert!(rq.curl_harmonic.is_finite());
        assert!(rq.curl_exponential.is_finite());
        assert!(rq.bayesian_regret.is_finite());
        assert!(rq.weighted_rank_reversals.is_finite());
    }

    #[test]
    fn test_likert_baseline_runs() {
        let results =
            run_likert_baseline_suite(Some("clean_ordering_10"), LikertEvalConfig::default())
                .expect("likert baseline suite should run");
        assert_eq!(results.len(), 1);
        let r = &results[0];

        assert!(r.metrics.ratings_used > 0);
        assert!(r.metrics.kendall_tau_all.is_finite());
    }

    // =========================================================================
    // Helper function tests
    // =========================================================================

    #[test]
    fn test_kendall_tau_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((kendall_tau_b(&x, &x) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_kendall_tau_reversed() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((kendall_tau_b(&x, &y) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_spearman_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((spearman_rho(&x, &x) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ranks_with_ties() {
        let scores = vec![3.0, 1.0, 3.0, 2.0];
        let ranks = ranks_with_ties(&scores);
        // Sorted: 1.0(idx1)=rank0, 2.0(idx3)=rank1, 3.0(idx0,idx2)=avg(rank2,rank3)=2.5
        assert!((ranks[0] - 2.5).abs() < 1e-6);
        assert!((ranks[1] - 0.0).abs() < 1e-6);
        assert!((ranks[2] - 2.5).abs() < 1e-6);
        assert!((ranks[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_topk_set_basic() {
        let scores = vec![10.0, 5.0, 8.0, 3.0, 7.0];
        let indices: Vec<usize> = (0..5).collect();
        let top2 = topk_set(&scores, &indices, 2, false);
        assert!(top2.contains(&0)); // score 10
        assert!(top2.contains(&2)); // score 8
        assert_eq!(top2.len(), 2);
    }

    #[test]
    fn gate_aware_topk_recall_penalizes_false_infeasible_items() {
        let pred_scores = vec![100.0, 90.0, 80.0, 70.0];
        let true_scores = vec![100.0, 90.0, 80.0, 70.0];
        let pred_feasible = vec![1, 2, 3];
        let true_feasible = vec![0, 1, 2, 3];

        // A gated run must not get to erase a true top-k item from the target
        // set merely because the gate predicted that item infeasible.
        assert!(
            (topk_precision_against_truth(
                &pred_scores,
                &true_scores,
                &pred_feasible,
                &true_feasible,
                2,
            ) - 0.5)
                .abs()
                < 1e-6
        );
        assert!(
            (topk_recall_against_truth(
                &pred_scores,
                &true_scores,
                &pred_feasible,
                &true_feasible,
                2,
            ) - 0.5)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn test_coverage_95_all_within() {
        let pred = vec![5.0, 3.0, 7.0];
        let truth = vec![5.0, 3.0, 7.0]; // exact match
        let vars = vec![1.0, 1.0, 1.0]; // generous variance
        let indices = vec![0, 1, 2];
        let cov = coverage_95(&pred, &vars, &truth, &indices);
        assert!((cov - 1.0).abs() < 1e-6);
    }

    #[test]
    fn synthetic_case_rejects_invalid_gate_units() {
        let case = SyntheticCase {
            name: "invalid_gate",
            attributes: vec![SyntheticAttribute {
                id: "quality",
                weight: 1.0,
                scores: vec![3.0, 2.0, 1.0],
            }],
            gates: vec![MultiRerankGateSpec {
                attribute_id: "quality".to_string(),
                unit: "bogus".to_string(),
                op: ">=".to_string(),
                threshold: 0.5,
            }],
            topk: default_topk(1),
            comparison_budget: Some(3),
            latency_budget_ms: None,
            max_pair_repeats: None,
            prewarm_pairs_per_attr: 0,
            noise_sigma: 0.0,
            refusal_rate: 0.0,
            outlier_rate: 0.0,
            seed: 7,
        };

        let error = run_synthetic_case(&case)
            .expect_err("invalid gate specs should not be silently ignored");
        assert!(matches!(error, EvaluationError::InvalidGate(_)));
    }
}
