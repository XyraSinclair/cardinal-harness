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

/// HONEST NEGATIVE #1, measured and pinned: the active planner currently
/// LOSES to uniform random pair selection even on its own objective —
/// comparisons until the top-5 set is exactly right (n=20, noise ±0.6,
/// 12 seeds): planner ≈134.7 vs random ≈86.7. At noise ±1.2 the gap
/// widens (≈200.7 vs ≈136.7). Suspected mechanisms (issue #43): forced
/// exploration builds a hub graph against a single top anchor (fragile
/// geometry); cross-boundary candidates over-anchor to extreme items
/// (acknowledged in trait_search's own comments); repeated critical-pair
/// hammering under noise. The pin is two-sided: a fix that makes the
/// planner actually WIN will trip the upper bound and force a re-pin —
/// that is the desired failure.
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
    assert!(
        (1.05..=2.2).contains(&ratio),
        "planner/random top-5 ratio left the measured band: {ratio:.2}          (planner {planner_mean:.1}, random {random_mean:.1}) — below 1.05          means the planner now WINS: celebrate, fix issue #43's status, and          re-pin; above 2.2 means it got even worse"
    );
}

/// HONEST NEGATIVE, measured and pinned: on GLOBAL order recovery the
/// boundary-focused planner is NOT better than uniform random pairs —
/// boundary focus underserves the tail, while random spread is near-optimal
/// for global tau. First measured 2026-07-04: planner ≈51.3 vs random
/// ≈47.3 comparisons to tau 0.85 (n=20, 12 seeds). Whole-list sorts (which
/// default to a middle boundary) inherit this; see issue #38. The pin is
/// two-sided so silent regressions AND silent improvements both surface.
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
