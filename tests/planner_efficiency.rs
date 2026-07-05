//! Active planner and stopping-efficiency tests, exercised at the
//! `TraitSearchManager` level (multi-attribute wrapper around `RatingEngine`).
//!
//! Every scenario below is a *planted-truth* world: we fix a ground-truth
//! latent score for each of 24 items, build an engine from a known
//! observation topology, then drive `propose_batch` in a loop where each
//! proposed pair is "judged" by consulting the planted scores directly
//! (optionally with injected noise). Because the planted truth and the
//! initial graph are fully deterministic (fixed seeds, fixed RNG-free
//! topologies), every non-statistical test here reproduces bit-for-bit
//! across runs. The one genuinely statistical claim (noisy-judge recovery)
//! is validated by an ensemble over many fixed `StdRng` seeds rather than a
//! single draw.

use std::collections::HashMap;
use std::collections::HashSet;

use cardinal_harness::rating_engine::{
    AttributeParams, Observation, PlannerMode, RaterParams, RatingEngine,
};
use cardinal_harness::trait_search::{
    AttributeConfig, TopKConfig, TraitSearchConfig, TraitSearchManager,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const ATTR: &str = "quality";
const N: usize = 24;
const K: usize = 8;

fn raters() -> HashMap<String, RaterParams> {
    let mut r = HashMap::new();
    r.insert("sim".to_string(), RaterParams::default());
    r
}

/// Standard-normal draw via Box-Muller, built on `rand`'s uniform sampler
/// (no `rand_distr` dependency available, and we must not touch Cargo.toml).
fn std_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen_range(1e-12..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------
// Planted worlds
// ---------------------------------------------------------------------

/// Planted score: linear descent, `gap` per rank step (item 0 is truth-best).
fn planted_scores(n: usize, gap: f64) -> Vec<f64> {
    (0..n).map(|i| -(i as f64) * gap).collect()
}

/// Planted score with a mild gap through `elbow`, then a much steeper drop
/// for the tail -- late-ranked items are unambiguously out of contention.
fn planted_scores_clear_tail(n: usize, gap: f64, elbow: usize, tail_gap: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i <= elbow {
            out.push(-(i as f64) * gap);
        } else {
            let head = -(elbow as f64) * gap;
            out.push(head - ((i - elbow) as f64) * tail_gap);
        }
    }
    out
}

/// Planted score with a uniform gap everywhere except a single distinguished
/// step between item `m` and item `m+1`, which uses `special_gap` instead.
/// Used both to engineer a single genuinely-ambiguous boundary pair (tiny
/// `special_gap`) and to engineer a rock-solid boundary separation (huge
/// `special_gap`).
fn planted_scores_kink(n: usize, gap: f64, m: usize, special_gap: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let mut cur = 0.0;
    out.push(cur);
    for i in 1..n {
        let step = if i - 1 == m { special_gap } else { gap };
        cur -= step;
        out.push(cur);
    }
    out
}

fn planted_ratio(scores: &[f64], i: usize, j: usize) -> f64 {
    (scores[i] - scores[j]).exp()
}

// ---------------------------------------------------------------------
// Observation topologies
// ---------------------------------------------------------------------

/// "Wheel" topology: hub item 0 compared directly against every other item
/// (coarse global order in one hop, bounded uncertainty growth), plus a rim
/// chain across the remaining items so every non-hub item has degree >= 2
/// and forced min-degree exploration does not dominate the batch.
fn wheel_engine(n: usize, scores: &[f64]) -> RatingEngine {
    let mut engine = RatingEngine::new(n, AttributeParams::default(), raters(), None).unwrap();
    let mut obs = Vec::new();
    for i in 1..n {
        obs.push(Observation::new(
            0,
            i,
            planted_ratio(scores, 0, i),
            1.0,
            "sim",
            1.0,
        ));
    }
    for i in 1..(n - 1) {
        obs.push(Observation::new(
            i,
            i + 1,
            planted_ratio(scores, i, i + 1),
            1.0,
            "sim",
            1.0,
        ));
    }
    engine.ingest(&obs);
    engine
}

fn manager_wheel(n: usize, k: usize, scores: &[f64], prune: Option<f64>) -> TraitSearchManager {
    let mut engines = HashMap::new();
    engines.insert(ATTR.to_string(), wheel_engine(n, scores));
    let mut topk = TopKConfig::new(k);
    topk.prune_p_topk_below = prune;
    let config = TraitSearchConfig::new(n, vec![AttributeConfig::new(ATTR, 1.0)], topk, vec![]);
    TraitSearchManager::new(config, engines).unwrap()
}

/// Band-3 topology: each item is directly compared against its next three
/// neighbors. The redundant paths keep effective resistance (hence
/// posterior variance) low and roughly uniform across the chain, so planted
/// score gaps translate directly into resolved/unresolved pairs without a
/// single high-distance hub edge (which would clamp under `max_log_ratio`).
fn band3_engine(n: usize, scores: &[f64]) -> RatingEngine {
    let mut engine = RatingEngine::new(n, AttributeParams::default(), raters(), None).unwrap();
    let mut obs = Vec::new();
    for i in 0..n {
        for d in 1..=3usize {
            if i + d < n {
                obs.push(Observation::new(
                    i,
                    i + d,
                    planted_ratio(scores, i, i + d),
                    1.0,
                    "sim",
                    1.0,
                ));
            }
        }
    }
    engine.ingest(&obs);
    engine
}

fn manager_band3(n: usize, k: usize, scores: &[f64], prune: Option<f64>) -> TraitSearchManager {
    let mut engines = HashMap::new();
    engines.insert(ATTR.to_string(), band3_engine(n, scores));
    let mut topk = TopKConfig::new(k);
    topk.prune_p_topk_below = prune;
    let config = TraitSearchConfig::new(n, vec![AttributeConfig::new(ATTR, 1.0)], topk, vec![]);
    TraitSearchManager::new(config, engines).unwrap()
}

/// Plain chain: only adjacent items are compared. Endpoints have degree 1
/// (below `min_explore_degree`), so this is the topology that exercises
/// forced exploration and, when configured, exploration pruning.
fn chain_engine(n: usize, scores: &[f64]) -> RatingEngine {
    let mut engine = RatingEngine::new(n, AttributeParams::default(), raters(), None).unwrap();
    let mut obs = Vec::new();
    for i in 0..(n - 1) {
        obs.push(Observation::new(
            i,
            i + 1,
            planted_ratio(scores, i, i + 1),
            1.0,
            "sim",
            1.0,
        ));
    }
    engine.ingest(&obs);
    engine
}

fn manager_chain(n: usize, k: usize, scores: &[f64], prune: Option<f64>) -> TraitSearchManager {
    let mut engines = HashMap::new();
    engines.insert(ATTR.to_string(), chain_engine(n, scores));
    let mut topk = TopKConfig::new(k);
    topk.prune_p_topk_below = prune;
    let config = TraitSearchConfig::new(n, vec![AttributeConfig::new(ATTR, 1.0)], topk, vec![]);
    TraitSearchManager::new(config, engines).unwrap()
}

/// Drive `propose_batch` for up to `rounds` rounds, answering every proposed
/// pair with the (noiseless) planted truth. Returns the number of
/// observations actually ingested (stops early once no proposals remain).
fn run_loop(
    manager: &mut TraitSearchManager,
    rounds: usize,
    batch_size: usize,
    scores: &[f64],
) -> usize {
    let mut total_obs = 0usize;
    for _ in 0..rounds {
        let proposals = manager
            .propose_batch("sim", batch_size, PlannerMode::Hybrid)
            .unwrap();
        if proposals.is_empty() {
            break;
        }
        for p in &proposals {
            let obs = Observation::new(p.i, p.j, planted_ratio(scores, p.i, p.j), 1.0, "sim", 1.0);
            manager.add_observation(&p.attribute_id, obs).unwrap();
            total_obs += 1;
        }
        manager.recompute_global_state().unwrap();
    }
    total_obs
}

// ---------------------------------------------------------------------
// 1. BAND FOCUS
// ---------------------------------------------------------------------

/// Claim: on a planted 24-item problem with coarse order already
/// established and a clear top-k boundary, `propose_batch` concentrates on
/// the boundary band -- at least 70% of proposed pairs must involve an item
/// whose current rank is within `band_size` of the boundary rank k.
#[test]
fn band_focus_concentrates_near_boundary() {
    let scores = planted_scores(N, 0.35);
    let mut manager = manager_wheel(N, K, &scores, None);
    manager.recompute_global_state().unwrap();

    let band_size = 5; // TopKConfig::new default
    let proposals = manager
        .propose_batch("sim", 16, PlannerMode::Hybrid)
        .unwrap();
    assert!(!proposals.is_empty(), "expected a non-empty batch");

    let near_boundary = |idx: usize| -> bool {
        let rank = manager.entity_state(idx).rank.unwrap();
        (rank as i64 - K as i64).unsigned_abs() as usize <= band_size
    };

    let hits = proposals
        .iter()
        .filter(|p| near_boundary(p.i) || near_boundary(p.j))
        .count();
    let frac = hits as f64 / proposals.len() as f64;
    assert!(
        frac >= 0.70,
        "expected >=70% of proposals within band_size of the boundary, got {frac:.2} ({hits}/{})",
        proposals.len()
    );
}

/// Claim (regression guard): the boundary-focused batch must not be
/// dominated by forced min-degree exploration edges reaching all the way to
/// the extremes of the ranking. Concretely, the best (rank 1) and worst
/// (rank n) items -- both already well outside the uncertainty band -- must
/// not appear in the batch at all when the graph already gives them
/// sufficient degree.
#[test]
fn band_focus_excludes_ranking_extremes() {
    let scores = planted_scores(N, 0.35);
    let mut manager = manager_wheel(N, K, &scores, None);
    manager.recompute_global_state().unwrap();

    let proposals = manager
        .propose_batch("sim", 16, PlannerMode::Hybrid)
        .unwrap();
    assert!(!proposals.is_empty());

    let touches_extreme = proposals
        .iter()
        .any(|p| p.i == 0 || p.j == 0 || p.i == N - 1 || p.j == N - 1);
    assert!(
        !touches_extreme,
        "batch should not reach for the best/worst items when the boundary band is elsewhere: {proposals:?}"
    );
}

// ---------------------------------------------------------------------
// 2. PRUNING SAFETY
// ---------------------------------------------------------------------

/// Claim: running the active-planning loop with `prune_p_topk_below` set
/// yields the exact same top-k SET as running it unpruned, while pruning at
/// least one hopeless tail entity from forced exploration along the way.
#[test]
fn pruning_preserves_topk_while_skipping_hopeless_tail() {
    // n=24 >= 16, chain topology (single-order observations), steep tail
    // drop past item 13 so the far tail is unambiguously out of contention.
    let scores = planted_scores_clear_tail(N, 0.35, 13, 3.0);

    let mut m_pruned = manager_chain(N, K, &scores, Some(0.05));
    m_pruned.recompute_global_state().unwrap();
    run_loop(&mut m_pruned, 12, 8, &scores);

    let mut m_plain = manager_chain(N, K, &scores, None);
    m_plain.recompute_global_state().unwrap();
    run_loop(&mut m_plain, 12, 8, &scores);

    let topk_pruned: HashSet<usize> = m_pruned.ranked_indices()[..K].iter().copied().collect();
    let topk_plain: HashSet<usize> = m_plain.ranked_indices()[..K].iter().copied().collect();
    assert_eq!(
        topk_pruned, topk_plain,
        "pruning must not change which entities end up in the top-k"
    );

    assert!(
        m_pruned.explore_pruned_count() > 0,
        "this tail-heavy problem should trigger at least one exploration prune"
    );
    assert_eq!(
        m_plain.explore_pruned_count(),
        0,
        "pruning must stay off when prune_p_topk_below is None"
    );
}

/// Claim (companion): a pruned entity is not dropped from the problem -- it
/// keeps a rank and stays feasible even after several rounds of active
/// planning built on top of the pruning decision.
#[test]
fn pruned_entity_keeps_rank_after_loop() {
    let scores = planted_scores_clear_tail(N, 0.35, 13, 3.0);
    let mut manager = manager_chain(N, K, &scores, Some(0.05));
    manager.recompute_global_state().unwrap();
    run_loop(&mut manager, 12, 8, &scores);

    assert!(manager.explore_pruned_count() > 0);
    let tail = manager.entity_state(N - 1);
    assert!(tail.feasible, "pruned entity must remain feasible");
    let rank = tail.rank.expect("pruned entity must still hold a rank");
    assert!(
        rank > K,
        "pruned entity should remain firmly outside the top-k, got rank {rank}"
    );
    assert!(
        rank >= N - 2,
        "pruned entity is planted deep in the steep tail and should stay near the very bottom, got rank {rank}"
    );
}

// ---------------------------------------------------------------------
// 3. EFFECTIVE INFORMATION
// ---------------------------------------------------------------------

/// Claim: answering a proposed batch consistently with planted truth must
/// weakly reduce `estimate_topk_error()` relative to before the batch.
#[test]
fn effective_information_batch_reduces_topk_error() {
    let scores = planted_scores(N, 0.35);
    let mut manager = manager_wheel(N, K, &scores, None);
    manager.recompute_global_state().unwrap();

    let err_before = manager.estimate_topk_error();
    assert!(err_before.is_finite() && err_before > 0.0);

    let obs_used = run_loop(&mut manager, 1, 16, &scores);
    assert!(
        obs_used > 0,
        "batch should not be empty on a genuinely ambiguous boundary"
    );

    let err_after = manager.estimate_topk_error();
    assert!(
        err_after <= err_before + 1e-9,
        "a truthfully-answered batch must not increase estimated top-k error: before={err_before:.4} after={err_after:.4}"
    );
    assert!(
        err_after < err_before,
        "expected a real (strict) improvement from 16 informative comparisons: before={err_before:.4} after={err_after:.4}"
    );
}

/// Claim (finer grain): each of the top few individually-proposed pairs,
/// answered on its own from the same starting state, is independently
/// non-harmful -- weakly reduces estimated top-k error by itself. This
/// isolates the "measured over the batch" claim down to single edges.
#[test]
fn effective_information_top_proposals_are_individually_non_harmful() {
    let scores = planted_scores(N, 0.35);

    let mut base = manager_wheel(N, K, &scores, None);
    base.recompute_global_state().unwrap();
    let err_before = base.estimate_topk_error();

    let proposals = base.propose_batch("sim", 16, PlannerMode::Hybrid).unwrap();
    assert!(proposals.len() >= 5);

    for p in proposals.iter().take(5) {
        // Fresh, independent copy of the starting state for each proposal.
        let mut manager = manager_wheel(N, K, &scores, None);
        manager.recompute_global_state().unwrap();
        let obs = Observation::new(p.i, p.j, planted_ratio(&scores, p.i, p.j), 1.0, "sim", 1.0);
        manager.add_observation(&p.attribute_id, obs).unwrap();
        manager.recompute_global_state().unwrap();
        let err_after = manager.estimate_topk_error();
        assert!(
            err_after <= err_before + 1e-9,
            "proposal ({}, {}) alone increased error: before={err_before:.4} after={err_after:.4}",
            p.i,
            p.j
        );
    }
}

// ---------------------------------------------------------------------
// 4. CRITICAL PAIR
// ---------------------------------------------------------------------

/// Claim: when exactly two items straddle the top-k boundary within noise
/// (everything else confidently resolved), the first proposal in the
/// exploitation batch must involve at least one of those two items.
#[test]
fn critical_pair_is_prioritized_first() {
    // Uniform big gap (2.0) everywhere except a tiny 0.02 gap right at the
    // k / k+1 boundary -- item (K-1) and item K are the only genuinely
    // ambiguous pair.
    let scores = planted_scores_kink(N, 2.0, K - 1, 0.02);
    let mut manager = manager_band3(N, K, &scores, None);
    manager.recompute_global_state().unwrap();

    // Sanity-check the planted setup itself: the two items closest to
    // p_flip = 0.5 (maximal boundary ambiguity) must be exactly the kink
    // pair, and must be much closer to 0.5 than the next-closest item.
    let mut by_ambiguity: Vec<(usize, f64)> = (0..N)
        .map(|i| (i, (manager.entity_state(i).p_flip - 0.5).abs()))
        .collect();
    by_ambiguity.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let top_two: HashSet<usize> = by_ambiguity[..2].iter().map(|(i, _)| *i).collect();
    assert_eq!(
        top_two,
        HashSet::from([K - 1, K]),
        "expected items {}/{} to be the uniquely ambiguous boundary pair, got {by_ambiguity:?}",
        K - 1,
        K
    );
    assert!(
        by_ambiguity[2].1 > 10.0 * by_ambiguity[1].1.max(1e-9),
        "the third-most-ambiguous item should be far less ambiguous than the boundary pair: {by_ambiguity:?}"
    );

    let proposals = manager
        .propose_batch("sim", 8, PlannerMode::Hybrid)
        .unwrap();
    assert!(!proposals.is_empty());
    let first = &proposals[0];
    assert!(
        first.i == K - 1 || first.j == K - 1 || first.i == K || first.j == K,
        "first exploitation proposal must touch the critical pair, got ({}, {})",
        first.i,
        first.j
    );
}

// ---------------------------------------------------------------------
// 5. STOPPING SOUNDNESS
// ---------------------------------------------------------------------

/// Claim: with a huge planted boundary gap and enough consistent
/// observations gathered through the active-planning loop, `certified_stop`
/// permits stopping (with `estimate_topk_error` under 0.1) using far fewer
/// than an exhaustive number of pairwise observations.
#[test]
fn stopping_soundness_certifies_before_exhaustive() {
    // Small step (0.5) almost everywhere, huge (8.0) step right at the
    // k/k+1 boundary: items 0..K are truly, overwhelmingly the top-k.
    let scores = planted_scores_kink(N, 0.5, K - 1, 8.0);
    let mut manager = manager_chain(N, K, &scores, None);
    manager.recompute_global_state().unwrap();

    let seed_obs = N - 1; // chain_engine edge count
    let exhaustive = N * (N - 1) / 2;

    let mut loop_obs = 0usize;
    let mut stopped = false;
    for _round in 0..40 {
        // Mirror the orchestrator's stop semantics: EITHER the certified
        // separation bound fires OR the estimated top-k error falls under
        // the tolerated threshold (multi.rs checks the latter first). The
        // anchor-diversity exploration (issue #43 fix) can drive the error
        // to zero with proposals drying up before the certified streak
        // accumulates — which is a stop, not a stall.
        manager.recompute_global_state().unwrap();
        if manager.certified_stop() || manager.estimate_topk_error() <= 0.1 {
            stopped = true;
            break;
        }
        let proposals = manager
            .propose_batch("sim", 8, PlannerMode::Hybrid)
            .unwrap();
        if std::env::var("PLANNER_DEBUG").is_ok() {
            eprintln!(
                "round {_round}: proposals {} err {:.4}",
                proposals.len(),
                manager.estimate_topk_error()
            );
        }
        for p in &proposals {
            let obs = Observation::new(p.i, p.j, planted_ratio(&scores, p.i, p.j), 1.0, "sim", 1.0);
            manager.add_observation(&p.attribute_id, obs).unwrap();
            loop_obs += 1;
        }
        if !proposals.is_empty() {
            manager.recompute_global_state().unwrap();
        }
    }

    assert!(
        stopped,
        "certified_stop should trigger on this overwhelming planted gap"
    );
    assert!(
        manager.estimate_topk_error() < 0.1,
        "certified stop must coincide with low estimated top-k error, got {:.4}",
        manager.estimate_topk_error()
    );
    let total_obs = seed_obs + loop_obs;
    assert!(
        total_obs < exhaustive,
        "expected far fewer than exhaustive ({exhaustive}) observations, used {total_obs}"
    );
}

/// Claim (companion): once certified_stop fires under the huge-gap
/// scenario, the recovered top-k set is exactly the true planted top-k --
/// stopping early does not come at the cost of correctness.
///
/// Adversarial note: a `certified_stop` that trivially returns `true` from
/// round 0 (never actually certifying anything) would still leave this
/// planted chain's *seed* graph alone giving the right top-k, since the
/// scenario is deliberately overwhelming -- so a correctness-only check is
/// not by itself proof the stopping mechanism did any work. We therefore
/// also require that the loop actually ran for a nontrivial number of
/// rounds and ingested a nontrivial number of observations beyond the seed
/// chain before certifying, so a stub `certified_stop() -> true` is caught
/// here too (independently of `stopping_soundness_certifies_before_exhaustive`,
/// which catches it via the error-bound check instead).
#[test]
fn stopping_soundness_recovers_correct_topk() {
    let scores = planted_scores_kink(N, 0.5, K - 1, 8.0);
    let mut manager = manager_chain(N, K, &scores, None);
    manager.recompute_global_state().unwrap();

    let mut rounds_run = 0usize;
    let mut loop_obs = 0usize;
    for _round in 0..40 {
        if manager.certified_stop() {
            break;
        }
        rounds_run += 1;
        let proposals = manager
            .propose_batch("sim", 8, PlannerMode::Hybrid)
            .unwrap();
        for p in &proposals {
            let obs = Observation::new(p.i, p.j, planted_ratio(&scores, p.i, p.j), 1.0, "sim", 1.0);
            manager.add_observation(&p.attribute_id, obs).unwrap();
            loop_obs += 1;
        }
        if !proposals.is_empty() {
            manager.recompute_global_state().unwrap();
        }
    }

    manager.recompute_global_state().unwrap();
    assert!(
        manager.certified_stop() || manager.estimate_topk_error() <= 0.1,
        "must actually reach a stop condition before checking correctness"
    );
    assert!(
        rounds_run >= 3 && loop_obs >= 8,
        "certification should require real active-planning work beyond the seed chain \
         (a stub certified_stop that fires immediately would trivially pass the topk \
         check below on this overwhelming-gap scenario without exercising anything): \
         rounds_run={rounds_run} loop_obs={loop_obs}"
    );
    let topk: HashSet<usize> = manager.ranked_indices()[..K].iter().copied().collect();
    let truth: HashSet<usize> = (0..K).collect();
    assert_eq!(
        topk, truth,
        "certified top-k must match the planted top-k exactly"
    );
}

/// Claim: with an already-obvious boundary (huge separation, redundant
/// band-3 topology so no forced exploration is pending), the planner should
/// see essentially zero remaining top-k error and have nothing useful left
/// to propose -- no wasted comparisons past the point of certainty.
#[test]
fn no_wasted_proposals_when_boundary_is_already_obvious() {
    let scores = planted_scores_kink(N, 0.5, K - 1, 8.0);
    let mut manager = manager_band3(N, K, &scores, None);
    manager.recompute_global_state().unwrap();

    assert!(
        manager.estimate_topk_error() < 0.01,
        "boundary is overwhelming; error should already be ~0, got {:.4}",
        manager.estimate_topk_error()
    );
    let proposals = manager
        .propose_batch("sim", 8, PlannerMode::Hybrid)
        .unwrap();
    assert!(
        proposals.is_empty(),
        "no comparisons should be proposed once the boundary is already certain: {proposals:?}"
    );
}

// ---------------------------------------------------------------------
// Determinism & statistical robustness
// ---------------------------------------------------------------------

/// Claim: with fixed engine RNG seeds and no state change between calls,
/// `propose_batch` is fully deterministic -- a regression guard against
/// accidental nondeterminism (e.g. hash-map iteration order or thread-local
/// RNG use) leaking into planner output.
#[test]
fn propose_batch_is_deterministic_given_fixed_state() {
    let scores = planted_scores(N, 0.35);
    let mut manager = manager_wheel(N, K, &scores, None);
    manager.recompute_global_state().unwrap();

    let first = manager
        .propose_batch("sim", 16, PlannerMode::Hybrid)
        .unwrap();
    let second = manager
        .propose_batch("sim", 16, PlannerMode::Hybrid)
        .unwrap();

    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.attribute_id, b.attribute_id);
        assert_eq!((a.i, a.j), (b.i, b.j));
        assert!((a.global_score - b.global_score).abs() < 1e-12);
    }
}

/// Claim (statistical): under realistic judge noise (sigma=0.6 log-ratio,
/// confidence 0.85), an active-planning loop of modest length (8 rounds x 8
/// pairs = 64 observations, far short of the 276-pair exhaustive count)
/// recovers the exact planted top-k set in the large majority of runs.
/// This is validated over an ensemble of 60 independent, fixed `StdRng`
/// seeds -- never on a single noisy draw -- and the assertion is
/// deliberately two-sided: high recovery is required, but recovery must
/// not be 100% (otherwise the noise level would not be doing real work and
/// the test would be a tautology).
#[test]
fn noisy_judge_ensemble_recovers_topk_most_of_the_time() {
    let base_scores = planted_scores(N, 0.35);
    let truth: HashSet<usize> = (0..K).collect();

    let sigma = 0.6;
    let seeds = 60usize;
    let mut recovered = 0usize;

    for seed in 0..seeds as u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut manager = manager_wheel(N, K, &base_scores, None);
        manager.recompute_global_state().unwrap();

        for _round in 0..8 {
            if manager.certified_stop() {
                break;
            }
            let proposals = manager
                .propose_batch("sim", 8, PlannerMode::Hybrid)
                .unwrap();
            if proposals.is_empty() {
                break;
            }
            for p in &proposals {
                let true_log = base_scores[p.i] - base_scores[p.j];
                let noisy_log = true_log + sigma * std_normal(&mut rng);
                let obs = Observation::new(p.i, p.j, noisy_log.exp(), 0.85, "sim", 1.0);
                manager.add_observation(&p.attribute_id, obs).unwrap();
            }
            manager.recompute_global_state().unwrap();
        }

        let final_topk: HashSet<usize> = manager.ranked_indices()[..K].iter().copied().collect();
        if final_topk == truth {
            recovered += 1;
        }
    }

    let rate = recovered as f64 / seeds as f64;
    assert!(
        rate >= 0.80,
        "expected robust top-k recovery under noise, got {rate:.2} ({recovered}/{seeds})"
    );
    assert!(
        recovered < seeds,
        "sigma=0.6 noise should occasionally cause a miss -- 100% recovery would suggest the noise is not actually exercised (got {recovered}/{seeds})"
    );
}
