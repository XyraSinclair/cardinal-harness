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
use crate::rerank::types::{
    HigherRanked, MultiRerankGateSpec, MultiRerankTopKSpec, PairwiseJudgement, RerankStopReason,
};
use crate::trait_search::{
    compute_attribute_units, AttributeConfig, GateSpec, TopKConfig, TraitSearchConfig,
    TraitSearchManager,
};

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
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EvaluationResult {
    pub case_name: String,
    pub metrics: EvaluationMetrics,
    pub error_trajectory: Vec<f64>,
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
                scores: [
                    vec![10.0; 10],
                    vec![5.0; 10],
                    vec![0.0; 10],
                ]
                .concat(),
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

pub fn run_synthetic_suite(filter: Option<&str>) -> Vec<EvaluationResult> {
    let cases = synthetic_cases();
    let selected: Vec<SyntheticCase> = match filter {
        Some(name) => cases.into_iter().filter(|c| c.name == name).collect(),
        None => cases,
    };

    selected
        .into_iter()
        .map(|case| run_synthetic_case(&case))
        .collect()
}

pub fn run_synthetic_case(case: &SyntheticCase) -> EvaluationResult {
    let mut rng = StdRng::seed_from_u64(case.seed);

    let n_entities = case
        .attributes
        .first()
        .map(|a| a.scores.len())
        .unwrap_or(0);

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
    };

    let gates_cfg: Vec<GateSpec> = case
        .gates
        .iter()
        .map(|g| GateSpec::new(&g.attribute_id, g.unit.to_ascii_lowercase(), &g.op, g.threshold))
        .collect();

    let config = TraitSearchConfig::new(n_entities, attributes_cfg, topk_cfg.clone(), gates_cfg);

    let mut engines: HashMap<String, RatingEngine> = HashMap::new();
    let mut raters: HashMap<String, RaterParams> = HashMap::new();
    let rater_id = "sim";
    raters.insert(rater_id.to_string(), RaterParams::default());

    let mut engine_cfg = EngineConfig::default();
    engine_cfg.rank_weight_exponent = topk_cfg.weight_exponent;
    engine_cfg.top_k = Some(topk_cfg.k);
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
        .expect("rating engine init");
        engines.insert(attr.id.to_string(), engine);
    }

    let mut manager = TraitSearchManager::new(config, engines).expect("trait search init");

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
                    *pair_repeats.entry((attr_idx, i.min(j), i.max(j))).or_insert(0.0) += 1.0;
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
        manager.recompute_global_state().expect("recompute");
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
        let proposal_request_size = (batch_size.saturating_mul(3)).max(batch_size);
        let proposals = manager
            .propose_batch(rater_id, proposal_request_size, PlannerMode::Hybrid)
            .expect("proposals");

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

            tasks.push(CompareTask { key, attr_idx, i, j });
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

    manager.recompute_global_state().expect("final recompute");

    let n = n_entities;
    let mut pred_scores = vec![0.0; n];
    let mut pred_vars = vec![0.0; n];
    let mut pred_feasible = vec![true; n];
    for i in 0..n {
        let state = manager.entity_state(i);
        pred_scores[i] = state.u_mean;
        pred_vars[i] = state.u_var;
        pred_feasible[i] = state.feasible;
    }

    let true_scores = compute_ground_truth_scores(&case.attributes);
    let true_feasible = compute_true_feasible(&case.attributes, &case.gates);

    let eval_indices: Vec<usize> = (0..n).filter(|&i| pred_feasible[i]).collect();

    let kendall_tau_all = kendall_tau_b(&pred_scores, &true_scores);
    let spearman_rho_all = spearman_rho(&pred_scores, &true_scores);

    let (kendall_tau, spearman_rho, topk_precision, topk_recall) = if eval_indices.len() >= 2 {
        let pred_eval: Vec<f64> = eval_indices.iter().map(|&i| pred_scores[i]).collect();
        let true_eval: Vec<f64> = eval_indices.iter().map(|&i| true_scores[i]).collect();
        let k = case.topk.k.min(eval_indices.len());
        (
            kendall_tau_b(&pred_eval, &true_eval),
            spearman_rho(&pred_eval, &true_eval),
            topk_precision(&pred_scores, &true_scores, &eval_indices, k),
            topk_recall(&pred_scores, &true_scores, &eval_indices, k),
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
        for i in 0..n {
            match (pred_feasible[i], true_feasible[i]) {
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


    EvaluationResult {
        case_name: case.name.to_string(),
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
            latency_ms: start_time.elapsed().as_millis(),
        },
        error_trajectory,
    }
}

// =============================================================================
// Helpers
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
    }
}

fn seeded_random_scores(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(0.0..1.0)).collect()
}

fn simulate_pairwise(
    rng: &mut impl Rng,
    truth_scores: &[f64],
    i: usize,
    j: usize,
    noise_sigma: f64,
    outlier_rate: f64,
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

    let mut confidence =
        confidence_from_signal((ratio.ln()).abs(), scale.max(1e-6), noise_sigma);

    if outlier_rate > 0.0 && rng.gen::<f64>() < outlier_rate {
        higher_ranked = match higher_ranked {
            HigherRanked::A => HigherRanked::B,
            HigherRanked::B => HigherRanked::A,
        };
        confidence = 0.9;
    }

    ratio = ratio.clamp(1.0, 26.0);

    PairwiseJudgement::Observation {
        higher_ranked,
        ratio,
        confidence,
    }
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
        for i in 0..n {
            u[i] += attr.weight * (attr.scores[i] * inv_scale);
        }
    }

    u
}

fn compute_true_feasible(attributes: &[SyntheticAttribute], gates: &[MultiRerankGateSpec]) -> Vec<bool> {
    let n = attributes.first().map(|a| a.scores.len()).unwrap_or(0);
    let mut feasible = vec![true; n];

    let mut attr_units: HashMap<&str, (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = HashMap::new();
    for attr in attributes {
        let (_scale, z, min_norm, pct) = compute_attribute_units(&attr.scores);
        attr_units.insert(attr.id, (attr.scores.clone(), z, min_norm, pct));
    }

    for gate in gates {
        let Some((latent, z, min_norm, pct)) = attr_units.get(gate.attribute_id.as_str()) else {
            continue;
        };
        let unit = gate.unit.to_ascii_lowercase();
        for i in 0..n {
            let value = match unit.as_str() {
                "latent" => latent[i],
                "z" => z[i],
                "percentile" => pct[i],
                "min_norm" => min_norm[i],
                _ => latent[i],
            };
            let pass = match gate.op.as_str() {
                ">=" => value >= gate.threshold,
                "<=" => value <= gate.threshold,
                _ => true,
            };
            feasible[i] = feasible[i] && pass;
        }
    }

    feasible
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

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];

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

    for i in 0..n {
        let dx = rx[i] - mean_x;
        let dy = ry[i] - mean_y;
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
    indices.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());

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

fn topk_precision(
    pred_scores: &[f64],
    true_scores: &[f64],
    indices: &[usize],
    k: usize,
) -> f64 {
    let pred_set = topk_set(pred_scores, indices, k, false);
    let true_set = topk_set(true_scores, indices, k, true);
    if pred_set.is_empty() {
        return 0.0;
    }
    let inter = pred_set.intersection(&true_set).count();
    inter as f64 / pred_set.len() as f64
}

fn topk_recall(
    pred_scores: &[f64],
    true_scores: &[f64],
    indices: &[usize],
    k: usize,
) -> f64 {
    let pred_set = topk_set(pred_scores, indices, k, false);
    let true_set = topk_set(true_scores, indices, k, true);
    if true_set.is_empty() {
        return 0.0;
    }
    let inter = pred_set.intersection(&true_set).count();
    inter as f64 / true_set.len() as f64
}

fn topk_set(
    scores: &[f64],
    indices: &[usize],
    k: usize,
    include_ties: bool,
) -> HashSet<usize> {
    if indices.is_empty() || k == 0 {
        return HashSet::new();
    }

    let mut sorted: Vec<usize> = indices.to_vec();
    sorted.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

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
