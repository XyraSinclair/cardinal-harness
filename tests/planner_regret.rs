//! The efficiency question, answered empirically: does the active planner
//! actually save comparisons versus a random-pair baseline on the SAME
//! judge? "We built an effective-resistance planner" is a claim; this file
//! is the receipt. Planted truth, simulated judge, comparisons counted
//! until the recovered order reaches a target Kendall tau.

use std::collections::HashMap;

use cardinal_harness::rating_engine::{
    AttributeParams, Config, Observation, PlannerMode, RaterParams, RatingEngine,
};
use cardinal_harness::trait_search::{
    AttributeConfig, TopKConfig, TraitSearchConfig, TraitSearchManager,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N: usize = 20;
const TARGET_TAU: f64 = 0.85;
const MAX_COMPARISONS: usize = 400;

fn planted_truth(seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..N).map(|_| rng.gen_range(0.0..3.0)).collect()
}

/// Simulated ratio judge: true log-gap plus Gaussian noise.
fn judge(rng: &mut StdRng, truth: &[f64], i: usize, j: usize) -> Observation {
    let gap = truth[i] - truth[j] + rng.gen_range(-0.6..0.6);
    Observation::new(i, j, gap.exp().clamp(0.04, 26.0), 0.8, "sim", 1.0)
}

fn kendall_tau(truth: &[f64], scores: &[f64]) -> f64 {
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    for i in 0..truth.len() {
        for j in (i + 1)..truth.len() {
            let t = (truth[i] - truth[j]).signum();
            let s = (scores[i] - scores[j]).signum();
            if t == s {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    (concordant - discordant) as f64 / (concordant + discordant).max(1) as f64
}

fn manager(k: usize) -> TraitSearchManager {
    let mut raters = HashMap::new();
    raters.insert("sim".to_string(), RaterParams::default());
    let engine = RatingEngine::new(N, AttributeParams::default(), raters, None).unwrap();
    let mut engines = HashMap::new();
    engines.insert("q".to_string(), engine);
    let config = TraitSearchConfig::new(
        N,
        vec![AttributeConfig::new("q", 1.0)],
        TopKConfig::new(k),
        vec![],
    );
    TraitSearchManager::new(config, engines).unwrap()
}

/// True top-k set from the planted truth.
fn true_top_k(truth: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..truth.len()).collect();
    idx.sort_by(|&a, &b| truth[b].total_cmp(&truth[a]));
    let mut top: Vec<usize> = idx.into_iter().take(k).collect();
    top.sort_unstable();
    top
}

/// Recovered top-k set from scores.
fn recovered_top_k(scores: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let mut top: Vec<usize> = idx.into_iter().take(k).collect();
    top.sort_unstable();
    top
}

/// Objectives a pair-selection policy can be scored on.
#[derive(Clone, Copy)]
enum Objective {
    /// Global order: Kendall tau over all items reaches TARGET_TAU.
    GlobalTau,
    /// The planner's actual objective: the top-k SET is exactly right.
    TopKSet(usize),
}

fn objective_met(objective: Objective, truth: &[f64], scores: &[f64]) -> bool {
    match objective {
        Objective::GlobalTau => kendall_tau(truth, scores) >= TARGET_TAU,
        Objective::TopKSet(k) => recovered_top_k(scores, k) == true_top_k(truth, k),
    }
}

/// Comparisons spent until the objective holds, driving pairs from the
/// planner (k matches the objective for TopKSet, middle otherwise).
fn planner_cost(seed: u64, objective: Objective) -> usize {
    let truth = planted_truth(seed);
    let k = match objective {
        Objective::TopKSet(k) => k,
        Objective::GlobalTau => N.div_ceil(2),
    };
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);
    let mut manager = manager(k);
    let mut used = 0usize;
    while used < MAX_COMPARISONS {
        manager.recompute_global_state().unwrap();
        if let Some(scores) = manager.attribute_scores("q") {
            if used > 0 && objective_met(objective, &truth, scores) {
                return used;
            }
        }
        // Mirror the real orchestrator: over-request proposals, dedup by
        // pair key within the batch, execute up to 8 distinct pairs.
        let proposals = manager
            .propose_batch("sim", 24, PlannerMode::Hybrid)
            .unwrap();
        if proposals.is_empty() {
            break;
        }
        let mut seen = std::collections::HashSet::new();
        let mut executed = 0usize;
        for proposal in proposals {
            let key = (proposal.i.min(proposal.j), proposal.i.max(proposal.j));
            if !seen.insert(key) {
                continue;
            }
            let obs = judge(&mut rng, &truth, proposal.i, proposal.j);
            manager.add_observation("q", obs).unwrap();
            used += 1;
            executed += 1;
            if executed >= 8 {
                break;
            }
        }
        if executed == 0 {
            break;
        }
    }
    used.max(MAX_COMPARISONS)
}

/// Comparisons spent until the objective holds, choosing pairs uniformly.
fn random_cost(seed: u64, objective: Objective) -> usize {
    let truth = planted_truth(seed);
    let k = match objective {
        Objective::TopKSet(k) => k,
        Objective::GlobalTau => N.div_ceil(2),
    };
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);
    let mut pair_rng = StdRng::seed_from_u64(seed ^ 0xa11);
    let mut manager = manager(k);
    let mut used = 0usize;
    while used < MAX_COMPARISONS {
        manager.recompute_global_state().unwrap();
        if let Some(scores) = manager.attribute_scores("q") {
            if used > 0 && objective_met(objective, &truth, scores) {
                return used;
            }
        }
        for _ in 0..8 {
            let i = pair_rng.gen_range(0..N);
            let mut j = pair_rng.gen_range(0..N);
            if i == j {
                j = (j + 1) % N;
            }
            let obs = judge(&mut rng, &truth, i, j);
            manager.add_observation("q", obs).unwrap();
            used += 1;
        }
    }
    used.max(MAX_COMPARISONS)
}

/// FIX-CYCLE HISTORY (issue #43), honestly kept:
/// - 2026-07-04 pre-fix: planner LOST to random on first-hit top-5
///   (≈134.7 vs ≈86.7) and global tau (≈51.3 vs ≈47.3); hub-anchor
///   exploration geometry was the lead suspect.
/// - 2026-07-04 anchor-diversity fix (quantile-rotating exploration
///   anchors): global-tau FLIPPED to a planner win (≈43.3 vs ≈47.3,
///   ratio 0.92); first-hit top-5 improved to ratio ≈1.36.
/// - ALSO: first-hit-time of a flickering exact-set state is itself a
///   biased metric favoring high-variance estimators — see the
///   fixed-budget benchmark below for the artifact-free picture, where
///   the planner WINS at scarce budgets (60), ties at 120, and slightly
///   trails at 180.
///
/// The pin is two-sided so both silent regressions and silent
/// improvements surface.
#[test]
fn honest_negative_planner_loses_to_random_on_top_k_identification() {
    let seeds: Vec<u64> = (0..12).collect();
    let objective = Objective::TopKSet(5);
    let planner_mean = seeds
        .iter()
        .map(|&s| planner_cost(s, objective))
        .sum::<usize>() as f64
        / seeds.len() as f64;
    let random_mean = seeds
        .iter()
        .map(|&s| random_cost(s, objective))
        .sum::<usize>() as f64
        / seeds.len() as f64;
    let ratio = planner_mean / random_mean;
    eprintln!(
        "REGRET[top5-set]: planner {planner_mean:.1} vs random {random_mean:.1} (ratio {ratio:.2})"
    );
    assert!(
        (1.05..=2.2).contains(&ratio),
        "planner/random top-5 ratio left the measured band: {ratio:.2}          (planner {planner_mean:.1}, random {random_mean:.1}) — below 1.05          means the planner now WINS: celebrate, fix issue #43's status, and          re-pin; above 2.2 means it got even worse"
    );
}

/// Post anchor-fix: the planner now WINS global-tau first-hit (≈43.3 vs
/// ≈47.3, ratio 0.92 — it lost pre-fix at 51.3). Two-sided pin so both
/// regressions and further improvements surface. History in the doc
/// comment above.
#[test]
fn global_tau_honest_negative_random_is_competitive() {
    let seeds: Vec<u64> = (0..12).collect();
    let objective = Objective::GlobalTau;
    let planner_mean = seeds
        .iter()
        .map(|&s| planner_cost(s, objective))
        .sum::<usize>() as f64
        / seeds.len() as f64;
    let random_mean = seeds
        .iter()
        .map(|&s| random_cost(s, objective))
        .sum::<usize>() as f64
        / seeds.len() as f64;
    let ratio = planner_mean / random_mean;
    eprintln!(
        "REGRET[global-tau]: planner {planner_mean:.1} vs random {random_mean:.1} (ratio {ratio:.2})"
    );
    assert!(
        (0.75..=1.45).contains(&ratio),
        "global-tau planner/random ratio drifted outside the measured band:          {ratio:.2} (planner {planner_mean:.1}, random {random_mean:.1}) —          if this IMPROVED below 0.75, celebrate and re-pin; if it degraded          above 1.45, the planner got worse at global order"
    );
}

#[test]
fn config_default_batch_is_deterministic_given_seeded_judge() {
    let a = planner_cost(3, Objective::GlobalTau);
    let b = planner_cost(3, Objective::GlobalTau);
    assert_eq!(a, b, "planner cost must be reproducible under fixed seeds");
    let _ = Config::default();
}

// =========================================================================
// Fixed-budget accuracy: the artifact-free comparison.
//
// First-hit-time of a FLICKERING state (exact-set correctness) is biased
// toward high-variance estimators — a policy whose scores jitter can cross
// "correct" by luck. Fixed-budget accuracy asks the honest question: after
// exactly B comparisons, how good is the answer?
// =========================================================================

fn run_policy_to_budget(seed: u64, planner: bool, budget: usize, k: usize) -> Vec<f64> {
    let truth = planted_truth(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);
    let mut pair_rng = StdRng::seed_from_u64(seed ^ 0xa11);
    let mut manager = manager(k);
    let mut used = 0usize;
    while used < budget {
        manager.recompute_global_state().unwrap();
        if planner {
            let proposals = manager
                .propose_batch("sim", 24, PlannerMode::Hybrid)
                .unwrap();
            if proposals.is_empty() {
                break;
            }
            let mut seen = std::collections::HashSet::new();
            let mut executed = 0usize;
            for proposal in proposals {
                let key = (proposal.i.min(proposal.j), proposal.i.max(proposal.j));
                if !seen.insert(key) {
                    continue;
                }
                let obs = judge(&mut rng, &truth, proposal.i, proposal.j);
                manager.add_observation("q", obs).unwrap();
                used += 1;
                executed += 1;
                if executed >= 8 || used >= budget {
                    break;
                }
            }
            if executed == 0 {
                break;
            }
        } else {
            for _ in 0..8 {
                if used >= budget {
                    break;
                }
                let i = pair_rng.gen_range(0..N);
                let mut j = pair_rng.gen_range(0..N);
                if i == j {
                    j = (j + 1) % N;
                }
                let obs = judge(&mut rng, &truth, i, j);
                manager.add_observation("q", obs).unwrap();
                used += 1;
            }
        }
    }
    manager.recompute_global_state().unwrap();
    manager.attribute_scores("q").unwrap().to_vec()
}

#[test]
fn fixed_budget_accuracy_planner_vs_random() {
    let seeds: Vec<u64> = (0..16).collect();
    let k = 5usize;
    for &budget in &[60usize, 120, 180] {
        let mut planner_tau = 0.0;
        let mut random_tau = 0.0;
        let mut planner_top5 = 0usize;
        let mut random_top5 = 0usize;
        for &seed in &seeds {
            let truth = planted_truth(seed);
            let ps = run_policy_to_budget(seed, true, budget, k);
            let rs = run_policy_to_budget(seed, false, budget, k);
            planner_tau += kendall_tau(&truth, &ps);
            random_tau += kendall_tau(&truth, &rs);
            planner_top5 += (recovered_top_k(&ps, k) == true_top_k(&truth, k)) as usize;
            random_top5 += (recovered_top_k(&rs, k) == true_top_k(&truth, k)) as usize;
        }
        let n = seeds.len() as f64;
        eprintln!(
            "REGRET[budget {budget}]: tau planner {:.3} vs random {:.3} · top5-rate planner {}/{} vs random {}/{}",
            planner_tau / n,
            random_tau / n,
            planner_top5,
            seeds.len(),
            random_top5,
            seeds.len(),
        );
        // Pin: at every measured budget the planner's mean tau must be at
        // least random's minus a small tolerance (and the printed numbers
        // are the receipt for the top-5 rates).
        assert!(
            planner_tau >= random_tau - 0.02 * n,
            "planner tau fell below random at budget {budget}"
        );
    }
}
