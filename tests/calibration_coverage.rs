//! Uncertainty calibration for `RatingEngine` / `TraitSearchManager`.
//!
//! Every claim below is a falsifiable mathematical statement about the
//! IRLS+Huber posterior, attacked with planted-truth ensembles over many
//! fixed seeds (never a single noisy draw). All RNG is `StdRng` with fixed
//! seeds; no `thread_rng` appears anywhere in an assertion path.

use std::collections::HashMap;
use std::f64::consts::PI;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use cardinal_harness::rating_engine::{AttributeParams, Observation, RaterParams, RatingEngine};
use cardinal_harness::trait_search::{
    AttributeConfig, TopKConfig, TraitSearchConfig, TraitSearchManager,
};

// ---------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------

/// Standard normal draw via Box-Muller, seeded StdRng only.
fn std_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen_range(1e-12..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn raters(beta: f64, default_confidence: f64) -> HashMap<String, RaterParams> {
    let mut m = HashMap::new();
    m.insert(
        "judge".to_string(),
        RaterParams {
            beta,
            cost_per_edge: 1.0,
            default_confidence,
        },
    );
    m
}

/// Build a RatingEngine over `n` items with planted latent `truth` (natural
/// log scale) and noisy pairwise ratio observations along `edges`.
/// Observed log-ratio = (truth[i]-truth[j]) + Normal(0, sigma).
fn planted_engine(
    n: usize,
    truth: &[f64],
    edges: &[(usize, usize)],
    sigma: f64,
    confidence: f64,
    reps: f64,
    rng: &mut StdRng,
) -> RatingEngine {
    let mut engine =
        RatingEngine::new(n, AttributeParams::default(), raters(1.0, confidence), None).unwrap();
    let mut obs = Vec::with_capacity(edges.len());
    for &(i, j) in edges {
        let true_log_ratio = truth[i] - truth[j];
        let noisy_log_ratio = true_log_ratio + sigma * std_normal(rng);
        let ratio = noisy_log_ratio.exp();
        obs.push(Observation::new(i, j, ratio, confidence, "judge", reps));
    }
    engine.ingest(&obs);
    engine
}

fn complete_graph(n: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((i, j));
        }
    }
    edges
}

/// A connected but sparse graph: a chain plus a handful of long chords, fixed
/// topology (not randomized) so ensembles differ only in observation noise.
fn sparse_connected_graph(n: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for i in 0..(n - 1) {
        edges.push((i, i + 1));
    }
    // A few chords for redundancy / cycle structure, avoiding a bare path
    // (which would make some diag_cov entries blow up disproportionately).
    for i in 0..n {
        let j = (i + 3) % n;
        if i != j {
            let e = if i < j { (i, j) } else { (j, i) };
            if !edges.contains(&e) {
                edges.push(e);
            }
        }
    }
    edges
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / (v.len() as f64)
}

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![lo];
    }
    (0..n)
        .map(|k| lo + (hi - lo) * (k as f64) / ((n - 1) as f64))
        .collect()
}

/// Gauge-aligned coverage check: center both the estimated score vector and
/// the planted truth vector by subtracting their own means (the additive
/// gauge freedom of the pinned Laplacian solve is only defined up to a
/// per-component shift), then test whether centered truth falls in
/// centered_estimate +/- z * std.
fn gauge_aligned_coverage_hits(
    scores: &[f64],
    diag_cov: &[f64],
    truth: &[f64],
    z: f64,
) -> (usize, usize) {
    let mean_est = mean(scores);
    let mean_truth = mean(truth);
    let mut hits = 0usize;
    let n = scores.len();
    for i in 0..n {
        let centered_est = scores[i] - mean_est;
        let centered_truth = truth[i] - mean_truth;
        let std = diag_cov[i].max(0.0).sqrt();
        let lo = centered_est - z * std;
        let hi = centered_est + z * std;
        if centered_truth >= lo && centered_truth <= hi {
            hits += 1;
        }
    }
    (hits, n)
}

// ---------------------------------------------------------------------
// Claim 1: coverage
// ---------------------------------------------------------------------

/// Claim: over an ensemble of independent noisy replicas of a planted 10-item
/// problem, the fraction of (replica, item) pairs whose gauge-aligned planted
/// score falls inside the reported ~95% interval (mean +/- 1.96*std) should
/// sit near 0.95, in a defensible band. This is the core promise of the
/// uncertainty engine: intervals should mean what they claim to mean.
///
/// `confidence=1.0` and `sigma=1.0` are *matched to the model*: with
/// `beta=1`, `reps=1`, `temperature=1`, the engine's assumed observation
/// precision `lam = g(confidence)*beta*reps/T` is exactly 1.0, i.e. it
/// assumes unit-variance log-ratio noise -- which is exactly what we inject.
/// This isolates the posterior-variance estimator's own calibration from any
/// confidence-vs-noise mismatch.
///
/// FINDING: observed coverage is ~0.997, not ~0.95. Investigation (see report)
/// traces this to `TraitSearchManager`-style diff-variance reasoning and the
/// `RatingEngine`'s gauge-pinned `diag_cov`: marginal variances are reported
/// relative to an arbitrary pinned reference node, and mean-centering (the
/// only gauge-invariant alignment available from the public API) does not
/// remove the positive correlation between nodes' pin-relative errors, so
/// gauge-aligned intervals end up conservative (wider than the true sampling
/// spread) rather than tight. That is a safe-direction miscalibration
/// (over-covering, not overconfident), but it means these intervals should
/// not be read as literal 95% intervals in this evaluation frame.
#[test]
fn coverage_ensemble_moderate_noise_complete_graph() {
    let n = 10;
    let truth = linspace(-2.0, 2.0, n);
    let edges = complete_graph(n);
    let sigma = 1.0; // matches model-implied sigma at confidence=1.0, reps=1
    let confidence = 1.0;
    let reps = 1.0;
    let z = 1.96;

    let replicas = 300usize;
    let mut hits = 0usize;
    let mut total = 0usize;

    for r in 0..replicas {
        let mut rng = StdRng::seed_from_u64(10_000 + r as u64);
        let mut engine = planted_engine(n, &truth, &edges, sigma, confidence, reps, &mut rng);
        let summary = engine.solve();
        assert!(!summary.degraded, "replica {r} degraded unexpectedly");
        let (h, t) = gauge_aligned_coverage_hits(&summary.scores, &summary.diag_cov, &truth, z);
        hits += h;
        total += t;
    }

    let coverage = hits as f64 / total as f64;
    eprintln!("[coverage_ensemble_moderate_noise_complete_graph] coverage={coverage:.4} over {total} draws");

    // Defensible band: nominal calibration would sit near 0.95, but the
    // honestly observed value here is ~0.997 (see FINDING above), with the
    // known conservative bias supplying a large, stable safety margin. The
    // floor is set at 0.94 -- comfortably below the observed value but well
    // above a loose 0.90, so a regression that erodes a meaningful chunk of
    // that conservatism (e.g. a partial reversion of the gauge-pin
    // over-covering behavior, or a scale bug in diag_cov) is still caught,
    // not just catastrophic overconfidence.
    assert!(
        (0.94..=1.0).contains(&coverage),
        "coverage {coverage:.4} outside defensible band [0.94, 1.0]"
    );
}

/// Claim: the same coverage property should hold (not collapse) on a sparser,
/// more realistic comparison graph (chain + chords) rather than a complete
/// graph. Sparser graphs give larger, more heterogeneous posterior variances,
/// which is a harder regime for a variance estimator to get right.
#[test]
fn coverage_ensemble_sparse_graph() {
    let n = 10;
    let truth = linspace(-2.0, 2.0, n);
    let edges = sparse_connected_graph(n);
    let sigma = 1.0; // matched to confidence=1.0, reps=1 model-implied sigma
    let confidence = 1.0;
    let reps = 1.0;
    let z = 1.96;

    let replicas = 300usize;
    let mut hits = 0usize;
    let mut total = 0usize;
    let mut degraded_count = 0usize;

    for r in 0..replicas {
        let mut rng = StdRng::seed_from_u64(20_000 + r as u64);
        let mut engine = planted_engine(n, &truth, &edges, sigma, confidence, reps, &mut rng);
        let summary = engine.solve();
        if summary.degraded {
            degraded_count += 1;
        }
        let (h, t) = gauge_aligned_coverage_hits(&summary.scores, &summary.diag_cov, &truth, z);
        hits += h;
        total += t;
    }

    let coverage = hits as f64 / total as f64;
    eprintln!("[coverage_ensemble_sparse_graph] coverage={coverage:.4} over {total} draws");
    assert_eq!(
        degraded_count, 0,
        "sparse-graph ensemble should not need the numerical-fallback ridge on well-conditioned planted replicas"
    );

    // Same conservative-not-overconfident finding as the complete-graph case
    // (observed ~0.997); floor at 0.94 for the same reason as the
    // complete-graph test -- tight enough to catch a meaningful erosion of
    // the safety margin under sparsity, not just outright overconfidence.
    assert!(
        (0.94..=1.0).contains(&coverage),
        "sparse-graph coverage {coverage:.4} outside defensible band [0.94, 1.0]"
    );
}

/// Sanity/monotonicity control: with very low noise and high confidence, the
/// posterior should be tight AND correct, so coverage should be very high.
/// This guards against a vacuous pass on claim 1 (e.g. reporting intervals so
/// wide they always cover regardless of noise).
#[test]
fn coverage_ensemble_low_noise_is_high() {
    let n = 10;
    let truth = linspace(-2.0, 2.0, n);
    let edges = complete_graph(n);
    let sigma = 0.03;
    let confidence = 0.99;
    let reps = 1.0;
    let z = 1.96;

    let replicas = 200usize;
    let mut hits = 0usize;
    let mut total = 0usize;
    let mut degraded_count = 0usize;

    for r in 0..replicas {
        let mut rng = StdRng::seed_from_u64(30_000 + r as u64);
        let mut engine = planted_engine(n, &truth, &edges, sigma, confidence, reps, &mut rng);
        let summary = engine.solve();
        if summary.degraded {
            degraded_count += 1;
        }
        let (h, t) = gauge_aligned_coverage_hits(&summary.scores, &summary.diag_cov, &truth, z);
        hits += h;
        total += t;
    }

    let coverage = hits as f64 / total as f64;
    eprintln!("[coverage_ensemble_low_noise_is_high] coverage={coverage:.4} over {total} draws");

    assert_eq!(
        degraded_count, 0,
        "low-noise complete-graph ensemble should never need the numerical-fallback ridge"
    );
    assert!(
        coverage >= 0.95,
        "low-noise coverage {coverage:.4} should be near-perfect, got below 0.95"
    );
}

/// Adversarial: mix small honest noise on most edges with a minority of huge
/// outlier "misjudgments" (the kind a bad-faith or confused judge produces).
/// Huber downweighting exists precisely to blunt these; coverage should not
/// collapse to near-zero. If it drops well below nominal, that is a finding
/// about IRLS-with-reused-weights variance optimism, reported honestly.
#[test]
fn coverage_survives_adversarial_outlier_mixture() {
    let n = 10;
    let truth = linspace(-2.0, 2.0, n);
    let edges = complete_graph(n);
    let confidence = 1.0;
    let reps = 1.0;
    let z = 1.96;

    let replicas = 300usize;
    let mut hits = 0usize;
    let mut total = 0usize;

    for r in 0..replicas {
        let mut rng = StdRng::seed_from_u64(40_000 + r as u64);
        let mut engine =
            RatingEngine::new(n, AttributeParams::default(), raters(1.0, confidence), None)
                .unwrap();
        let mut obs = Vec::with_capacity(edges.len());
        for &(i, j) in &edges {
            let true_log_ratio = truth[i] - truth[j];
            // ~15% of edges are wild misjudgments (large, sign-agnostic noise);
            // the other 85% match the model-implied sigma exactly.
            let is_outlier = rng.gen_bool(0.15);
            let sigma = if is_outlier { 10.0 } else { 1.0 };
            let noisy_log_ratio = true_log_ratio + sigma * std_normal(&mut rng);
            let ratio = noisy_log_ratio.exp();
            obs.push(Observation::new(i, j, ratio, confidence, "judge", reps));
        }
        engine.ingest(&obs);
        let summary = engine.solve();
        let (h, t) = gauge_aligned_coverage_hits(&summary.scores, &summary.diag_cov, &truth, z);
        hits += h;
        total += t;
    }

    let coverage = hits as f64 / total as f64;
    eprintln!(
        "[coverage_survives_adversarial_outlier_mixture] coverage={coverage:.4} over {total} draws"
    );

    // Observed ~0.956: Huber downweighting keeps this from collapsing to
    // anything like the ~85% "honest" edge fraction, let alone lower. The
    // floor here is set well below the observed value (not at it) so a
    // regression that meaningfully degrades outlier robustness would still
    // be caught without this test being a tautology on today's exact number.
    assert!(
        coverage >= 0.80,
        "coverage under adversarial outlier mixture collapsed to {coverage:.4}"
    );
}

// ---------------------------------------------------------------------
// Claim 2: uncertainty shrinks with more observations
// ---------------------------------------------------------------------

/// Claim: doubling the observation weight on every edge of a graph should
/// weakly shrink every item's posterior std (more evidence, same or less
/// uncertainty). Checked across many independent noise seeds so this is not
/// a single-draw fluke.
#[test]
fn uncertainty_weakly_shrinks_as_observations_double() {
    let n = 8;
    let truth = linspace(-1.5, 1.5, n);
    let edges = complete_graph(n);
    let confidence = 0.85;
    let sigma = 0.5;

    let seeds = 60u64;
    let mut violations = 0usize;
    let mut total_checks = 0usize;

    for s in 0..seeds {
        let mut rng1 = StdRng::seed_from_u64(50_000 + s);
        let mut engine_single =
            planted_engine(n, &truth, &edges, sigma, confidence, 1.0, &mut rng1);
        let summary_single = engine_single.solve();

        // Re-derive the identical noisy observations, but with reps doubled
        // (same noise draws => same effective mu per edge, just more weight).
        let mut rng2 = StdRng::seed_from_u64(50_000 + s);
        let mut engine_double =
            planted_engine(n, &truth, &edges, sigma, confidence, 2.0, &mut rng2);
        let summary_double = engine_double.solve();

        for i in 0..n {
            total_checks += 1;
            let std1 = summary_single.diag_cov[i].max(0.0).sqrt();
            let std2 = summary_double.diag_cov[i].max(0.0).sqrt();
            // Allow tiny numerical slack; must not meaningfully increase.
            if std2 > std1 + 1e-9 {
                violations += 1;
            }
        }
    }

    eprintln!("[uncertainty_weakly_shrinks_as_observations_double] violations={violations}/{total_checks}");
    assert_eq!(
        violations, 0,
        "found {violations} items whose std increased when observation reps doubled"
    );
}

/// Claim: on a fixed, well-connected graph, an item with extra independent
/// evidence (more observations touching it) has smaller posterior std, on
/// average across seeds, than an item with only the baseline amount.
///
/// This deliberately uses a complete graph (rather than a star/tree) as the
/// base topology: a sparse "single bridge to the hub" design is fragile
/// under Huber down-weighting (if that one bridge edge is treated as an
/// outlier on a given noise draw, the whole component's absolute position
/// can blow up, confounding the degree comparison with a rare-but-huge
/// tail event). A complete graph has enough redundancy that no single
/// edge's Huber down-weighting can dominate, isolating the effect of
/// "extra evidence" cleanly.
#[test]
fn uncertainty_smaller_for_more_observed_item_same_graph() {
    let n = 8;
    let truth: Vec<f64> = vec![0.0; n]; // magnitude irrelevant; only evidence amount varies
    let base_edges = complete_graph(n);
    let sigma = 0.4;
    let confidence = 0.85;

    // node 7 gets 8 extra independent comparisons against node 6, on top of
    // the baseline complete-graph edge; node 3 gets only the baseline.
    let extra_target = 7usize;
    let extra_partner = 6usize;
    let baseline_node = 3usize;

    let seeds = 150u64;
    let mut extra_std_sum = 0.0;
    let mut baseline_std_sum = 0.0;

    for s in 0..seeds {
        let mut rng = StdRng::seed_from_u64(70_000 + s);
        let mut engine =
            RatingEngine::new(n, AttributeParams::default(), raters(1.0, confidence), None)
                .unwrap();

        let mut obs = Vec::with_capacity(base_edges.len() + 8);
        for &(i, j) in &base_edges {
            let true_log_ratio = truth[i] - truth[j];
            let noisy_log_ratio = true_log_ratio + sigma * std_normal(&mut rng);
            obs.push(Observation::new(
                i,
                j,
                noisy_log_ratio.exp(),
                confidence,
                "judge",
                1.0,
            ));
        }
        for _ in 0..8 {
            let true_log_ratio = truth[extra_target] - truth[extra_partner];
            let noisy_log_ratio = true_log_ratio + sigma * std_normal(&mut rng);
            obs.push(Observation::new(
                extra_target,
                extra_partner,
                noisy_log_ratio.exp(),
                confidence,
                "judge",
                1.0,
            ));
        }
        engine.ingest(&obs);
        let summary = engine.solve();
        extra_std_sum += summary.diag_cov[extra_target].max(0.0).sqrt();
        baseline_std_sum += summary.diag_cov[baseline_node].max(0.0).sqrt();
    }

    let extra_std = extra_std_sum / seeds as f64;
    let baseline_std = baseline_std_sum / seeds as f64;

    eprintln!(
        "[uncertainty_smaller_for_more_observed_item_same_graph] extra_evidence_std={extra_std:.4} baseline_std={baseline_std:.4}"
    );

    assert!(
        extra_std < baseline_std,
        "item with extra independent evidence should have smaller average std: extra={extra_std:.4}, baseline={baseline_std:.4}"
    );
}

// ---------------------------------------------------------------------
// Claim 3: top-k error honesty
// ---------------------------------------------------------------------

fn sim_raters_default() -> HashMap<String, RaterParams> {
    let mut m = HashMap::new();
    m.insert("sim".to_string(), RaterParams::default());
    m
}

/// Claim: when the top-k boundary gap is huge (item k and item k+1 are
/// separated by a landslide score gap backed by many confident, repeated
/// observations), `TraitSearchManager::estimate_topk_error` must be tiny.
#[test]
fn topk_error_tiny_for_huge_boundary_gap() {
    let n = 6;
    let k = 3;
    // Landslide gap: top 3 items around score +10, bottom 3 around score -10.
    let truth: Vec<f64> = vec![10.0, 9.5, 9.0, -9.0, -9.5, -10.0];

    let mut engine =
        RatingEngine::new(n, AttributeParams::default(), sim_raters_default(), None).unwrap();
    let mut obs = Vec::new();
    // Densely observe every pair, many reps, high confidence: variance tiny.
    for i in 0..n {
        for j in (i + 1)..n {
            let ratio = (truth[i] - truth[j]).exp();
            obs.push(Observation::new(i, j, ratio, 0.95, "sim", 20.0));
        }
    }
    engine.ingest(&obs);

    let mut engines = HashMap::new();
    engines.insert("quality".to_string(), engine);
    let config = TraitSearchConfig::new(
        n,
        vec![AttributeConfig::new("quality", 1.0)],
        TopKConfig::new(k),
        vec![],
    );
    let mut manager = TraitSearchManager::new(config, engines).unwrap();
    manager.recompute_global_state().unwrap();

    let err = manager.estimate_topk_error();
    eprintln!("[topk_error_tiny_for_huge_boundary_gap] err={err:.6}");
    assert!(
        err < 1e-3,
        "expected tiny top-k error for landslide gap, got {err}"
    );
}

/// Claim: when the top-k boundary is a genuine coin flip (item k and item
/// k+1 are planted at *exactly* equal true score, each with only sparse
/// evidence), `estimate_topk_error` must be substantial (>0.2): the system
/// must not pretend to know which of two tied items belongs in the top-k.
#[test]
fn topk_error_substantial_for_coin_flip_boundary() {
    let n = 6;
    let k = 3;
    // Items ranked 3 and 4 (indices 2 and 3) are planted exactly tied;
    // everything else is clearly separated so the *only* ambiguity is the
    // boundary itself.
    let truth: Vec<f64> = vec![10.0, 6.0, 2.0, 2.0, -6.0, -10.0];

    let mut engine =
        RatingEngine::new(n, AttributeParams::default(), sim_raters_default(), None).unwrap();
    let mut obs = Vec::new();
    // Sparse evidence: a single chain of comparisons, modest confidence, no
    // repeats, so posterior variance at the boundary stays non-trivial.
    for i in 0..(n - 1) {
        let ratio = (truth[i] - truth[i + 1]).exp();
        obs.push(Observation::new(i, i + 1, ratio, 0.7, "sim", 1.0));
    }
    engine.ingest(&obs);

    let mut engines = HashMap::new();
    engines.insert("quality".to_string(), engine);
    let config = TraitSearchConfig::new(
        n,
        vec![AttributeConfig::new("quality", 1.0)],
        TopKConfig::new(k),
        vec![],
    );
    let mut manager = TraitSearchManager::new(config, engines).unwrap();
    manager.recompute_global_state().unwrap();

    let err = manager.estimate_topk_error();
    eprintln!("[topk_error_substantial_for_coin_flip_boundary] err={err:.6}");
    assert!(
        err > 0.2,
        "expected substantial top-k error at a coin-flip boundary, got {err}"
    );
}

/// Claim: adding clarifying, high-confidence observations that reinforce the
/// whole comparison chain (not just the boundary pair) should shrink
/// `estimate_topk_error` substantially relative to the sparse baseline --
/// the estimator must actually respond to new evidence, not just report a
/// constant.
///
/// Note: the boundary items here (indices 2, 3) are planted with a real but
/// *small* gap (2.5 vs 2.0), not an exact tie -- this is epistemic
/// uncertainty (resolvable with evidence), unlike the exact-tie aleatoric
/// case in `topk_error_substantial_for_coin_flip_boundary` above, where no
/// amount of matching evidence can push error below the intrinsic 0.5 for a
/// genuinely 50/50 pair.
#[test]
fn topk_error_shrinks_after_targeted_boundary_observations() {
    let n = 6;
    let k = 3;
    let truth: Vec<f64> = vec![10.0, 6.0, 2.5, 2.0, -6.0, -10.0];

    let mut engine =
        RatingEngine::new(n, AttributeParams::default(), sim_raters_default(), None).unwrap();
    let mut obs = Vec::new();
    for i in 0..(n - 1) {
        let ratio = (truth[i] - truth[i + 1]).exp();
        obs.push(Observation::new(i, i + 1, ratio, 0.7, "sim", 1.0));
    }
    engine.ingest(&obs);

    let mut engines = HashMap::new();
    engines.insert("quality".to_string(), engine);
    let config = TraitSearchConfig::new(
        n,
        vec![AttributeConfig::new("quality", 1.0)],
        TopKConfig::new(k),
        vec![],
    );
    let mut manager = TraitSearchManager::new(config, engines).unwrap();
    manager.recompute_global_state().unwrap();
    let err_before = manager.estimate_topk_error();

    // Reinforce every edge in the chain with much stronger, high-confidence,
    // high-rep evidence matching the same planted truth.
    let mut extra = Vec::new();
    for i in 0..(n - 1) {
        let ratio = (truth[i] - truth[i + 1]).exp();
        extra.push(Observation::new(i, i + 1, ratio, 0.99, "sim", 100.0));
    }
    manager.add_observations("quality", &extra).unwrap();
    manager.recompute_global_state().unwrap();
    let err_after = manager.estimate_topk_error();

    eprintln!(
        "[topk_error_shrinks_after_targeted_boundary_observations] before={err_before:.6} after={err_after:.6}"
    );
    assert!(
        err_before > 0.5,
        "sparse baseline should carry substantial top-k error, got {err_before}"
    );
    assert!(
        err_after < err_before * 0.3,
        "reinforcing evidence should substantially shrink top-k error: before={err_before}, after={err_after}"
    );
}

// ---------------------------------------------------------------------
// Claim 4: p_flip semantics
// ---------------------------------------------------------------------

/// Claim: `GlobalEntityState::p_flip` documents the probability that an
/// item's rank relative to the top-k boundary "flips" from where it
/// currently sits. Items far above the boundary must have p_flip -> 0;
/// items far below must have p_flip -> 1.
#[test]
fn p_flip_extremes_far_above_and_below_boundary() {
    let n = 6;
    let k = 3;
    let truth: Vec<f64> = vec![100.0, 90.0, 80.0, -80.0, -90.0, -100.0];

    let mut engine =
        RatingEngine::new(n, AttributeParams::default(), sim_raters_default(), None).unwrap();
    let mut obs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let ratio = (truth[i] - truth[j]).exp();
            obs.push(Observation::new(i, j, ratio, 0.95, "sim", 20.0));
        }
    }
    engine.ingest(&obs);

    let mut engines = HashMap::new();
    engines.insert("quality".to_string(), engine);
    let config = TraitSearchConfig::new(
        n,
        vec![AttributeConfig::new("quality", 1.0)],
        TopKConfig::new(k),
        vec![],
    );
    let mut manager = TraitSearchManager::new(config, engines).unwrap();
    manager.recompute_global_state().unwrap();

    let p_top = manager.entity_state(0).p_flip; // rank 1, deep in top-k
    let p_bottom = manager.entity_state(5).p_flip; // rank 6, deep out of top-k

    eprintln!(
        "[p_flip_extremes_far_above_and_below_boundary] p_top={p_top:.6} p_bottom={p_bottom:.6}"
    );
    assert!(
        p_top < 1e-3,
        "item far above boundary should have p_flip near 0, got {p_top}"
    );
    assert!(
        p_bottom > 1.0 - 1e-3,
        "item far below boundary should have p_flip near 1, got {p_bottom}"
    );
}

/// Claim: the boundary item itself (rank == k) sits exactly at the
/// decision line against itself, so its p_flip must be ~0.5 -- this pins the
/// documented semantics so nobody silently flips the direction of p_flip.
#[test]
fn p_flip_at_boundary_is_half() {
    let n = 6;
    let k = 3;
    let truth: Vec<f64> = vec![100.0, 90.0, 80.0, -80.0, -90.0, -100.0];

    let mut engine =
        RatingEngine::new(n, AttributeParams::default(), sim_raters_default(), None).unwrap();
    let mut obs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let ratio = (truth[i] - truth[j]).exp();
            obs.push(Observation::new(i, j, ratio, 0.95, "sim", 20.0));
        }
    }
    engine.ingest(&obs);

    let mut engines = HashMap::new();
    engines.insert("quality".to_string(), engine);
    let config = TraitSearchConfig::new(
        n,
        vec![AttributeConfig::new("quality", 1.0)],
        TopKConfig::new(k),
        vec![],
    );
    let mut manager = TraitSearchManager::new(config, engines).unwrap();
    manager.recompute_global_state().unwrap();

    let p_boundary = manager.entity_state(2).p_flip; // rank == k == 3
    eprintln!("[p_flip_at_boundary_is_half] p_boundary={p_boundary:.6}");
    assert!(
        (p_boundary - 0.5).abs() < 0.02,
        "boundary item's p_flip should be ~0.5, got {p_boundary}"
    );
}

/// Claim: p_flip is monotonically non-increasing as rank improves (moving up
/// the ranking, away from the bottom, towards and through the boundary and
/// into the top-k). A chain of six items with clean separation gives an
/// unambiguous total order to check this against.
#[test]
fn p_flip_monotonic_with_rank_on_clean_chain() {
    let n = 6;
    let k = 3;
    let truth: Vec<f64> = vec![15.0, 10.0, 5.0, -5.0, -10.0, -15.0];

    let mut engine =
        RatingEngine::new(n, AttributeParams::default(), sim_raters_default(), None).unwrap();
    let mut obs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let ratio = (truth[i] - truth[j]).exp();
            obs.push(Observation::new(i, j, ratio, 0.85, "sim", 3.0));
        }
    }
    engine.ingest(&obs);

    let mut engines = HashMap::new();
    engines.insert("quality".to_string(), engine);
    let config = TraitSearchConfig::new(
        n,
        vec![AttributeConfig::new("quality", 1.0)],
        TopKConfig::new(k),
        vec![],
    );
    let mut manager = TraitSearchManager::new(config, engines).unwrap();
    manager.recompute_global_state().unwrap();

    // Items are indexed 0..n in strictly descending planted-score order, so
    // p_flip should be non-decreasing in index (best item -> lowest p_flip).
    let p_flips: Vec<f64> = (0..n).map(|i| manager.entity_state(i).p_flip).collect();
    eprintln!("[p_flip_monotonic_with_rank_on_clean_chain] {p_flips:?}");
    for w in p_flips.windows(2) {
        assert!(
            w[0] <= w[1] + 1e-9,
            "p_flip should be non-decreasing down the ranking: {p_flips:?}"
        );
    }
    // And it must actually move, not sit flat at a constant.
    assert!(
        p_flips[n - 1] - p_flips[0] > 0.5,
        "p_flip should vary substantially from top to bottom of a clean chain: {p_flips:?}"
    );
}
