//! Property-based recovery tests for the IRLS+Huber solver in `rating_engine`.
//!
//! Mathematical claims under attack (see doc comment on each `#[test]`):
//!   1. Planted recovery: noisy pairwise log-ratio observations consistent with a
//!      known latent ranking are aggregated back into (nearly) the correct order.
//!   2. Huber robustness: a bounded fraction of adversarial (reversed,
//!      high-confidence) observations should not destroy the recovered ranking,
//!      and the robust (Huber) fit should out-perform a naive weighted-least-
//!      squares fit (Huber effectively disabled) under the same corruption.
//!   3. Gauge invariance: only *relative* differences between recovered scores
//!      are meaningful. Relabeling items (which changes which node gets pinned
//!      to the gauge origin) must not change recovered pairwise differences,
//!      and translating the origin of the latent truth must not change the
//!      observations generated from it (so recovery quality is unaffected).
//!   4. Stated-confidence invariance: the point path treats model-reported
//!      confidence as metadata; otherwise uncalibrated self-assessment can
//!      manufacture solver precision.
//!   5. Ratio-ladder sanity: a bigger elicited ratio must produce a bigger
//!      recovered gap, monotonically, all the way up to the ladder cap (26.0).
//!
//! All tests are deterministic: every source of randomness is a seeded
//! `StdRng`, and every statistical claim is evaluated as an aggregate over an
//! ensemble of many fixed seeds rather than a single noisy draw.

#![allow(clippy::field_reassign_with_default)] // Config is #[non_exhaustive]-style; reassign keeps intent obvious

use std::collections::{HashMap, HashSet};

use cardinal_harness::prompts::RATIO_LADDER;
use cardinal_harness::rating_engine::{
    AttributeParams, Config, Observation, RaterParams, RatingEngine,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------
// Shared test harness
// ---------------------------------------------------------------------

const RATER: &str = "sim";

fn raters() -> HashMap<String, RaterParams> {
    let mut m = HashMap::new();
    m.insert(RATER.to_string(), RaterParams::default());
    m
}

fn engine_with_cfg(n: usize, cfg: Config) -> RatingEngine {
    RatingEngine::new(n, AttributeParams::default(), raters(), Some(cfg)).unwrap()
}

fn engine(n: usize) -> RatingEngine {
    engine_with_cfg(n, Config::default())
}

/// Standard-normal deviate via Box-Muller, driven entirely by a seeded RNG.
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen_range(1e-12..1.0f64);
    let u2: f64 = rng.gen_range(0.0..1.0f64);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Draw `n` planted latent scores uniformly in `[-spread, spread]`.
fn planted_scores(n: usize, spread: f64, rng: &mut StdRng) -> Vec<f64> {
    (0..n).map(|_| rng.gen_range(-spread..spread)).collect()
}

/// A spanning chain (guarantees connectivity) plus `extra` random pairs.
fn edge_topology(n: usize, extra: usize, rng: &mut StdRng) -> Vec<(usize, usize)> {
    let mut edges: Vec<(usize, usize)> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
    for _ in 0..extra {
        let i = rng.gen_range(0..n);
        let mut j = rng.gen_range(0..n);
        while j == i {
            j = rng.gen_range(0..n);
        }
        edges.push((i.min(j), i.max(j)));
    }
    edges
}

/// Build a noisy log-ratio observation consistent with `planted[i] - planted[j]`.
fn noisy_observation(
    i: usize,
    j: usize,
    planted: &[f64],
    noise_sd: f64,
    confidence: f64,
    rng: &mut StdRng,
) -> Observation {
    let true_diff = planted[i] - planted[j];
    let noisy_diff = true_diff + noise_sd * gaussian(rng);
    let ratio = noisy_diff.exp();
    Observation::new(i, j, ratio, confidence, RATER, 1.0)
}

/// Reversed, over-confident observation: claims the opposite of the truth,
/// inflated so it actively fights the true direction, presented with maximum
/// confidence (an adversarial rater lying boldly and loudly).
fn adversarial_observation(i: usize, j: usize, planted: &[f64]) -> Observation {
    let true_diff = planted[i] - planted[j];
    let sign = if true_diff >= 0.0 { 1.0 } else { -1.0 };
    let flipped = -true_diff - sign * 2.5;
    Observation::new(i, j, flipped.exp(), 1.0, RATER, 1.0)
}

/// Evenly separated planted truth over `n` items: an unambiguous total order.
fn evenly_spaced_truth(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 * 0.6).collect()
}

/// Planted truth with a wide gap isolating the top 3 items from the rest,
/// so "the top-3" is unambiguous even under substantial per-edge noise —
/// internal order within each cluster still varies by a modest 0.6 spacing.
fn clustered_top3_truth(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let base = i as f64 * 0.6;
            if i + 3 >= n {
                base + 3.0
            } else {
                base
            }
        })
        .collect()
}

/// Build one run (clean when `adversarial_frac == 0.0`, corrupted otherwise)
/// against a given planted truth. The set of corrupted edges is a genuine
/// random subset (not a fixed prefix), and the per-edge noise draws are
/// frozen up front so that a clean run and a corrupted run sharing the same
/// seed agree on every *non*-corrupted edge — isolating the effect of the
/// corruption itself.
fn build_run(
    n: usize,
    planted: &[f64],
    extra: usize,
    adversarial_frac: f64,
    seed: u64,
    huber_k: f64,
) -> (Vec<f64>, Vec<f64>) {
    let planted = planted.to_vec();

    let mut topo_rng = StdRng::seed_from_u64(seed);
    let edges = edge_topology(n, extra, &mut topo_rng);
    let m = edges.len();

    let mut noise_rng = StdRng::seed_from_u64(seed ^ 0xA5A5_A5A5);
    let noises: Vec<f64> = edges.iter().map(|_| gaussian(&mut noise_rng)).collect();

    let mut select_rng = StdRng::seed_from_u64(seed ^ 0x5A5A_5A5A);
    let n_adv = ((m as f64) * adversarial_frac).round() as usize;
    let mut order: Vec<usize> = (0..m).collect();
    for k in 0..n_adv.min(m) {
        let r = select_rng.gen_range(k..m);
        order.swap(k, r);
    }
    let adversarial_set: HashSet<usize> = order[..n_adv.min(m)].iter().copied().collect();

    let mut obs = Vec::with_capacity(m);
    for (k, (&(i, j), &z)) in edges.iter().zip(noises.iter()).enumerate() {
        if adversarial_set.contains(&k) {
            obs.push(adversarial_observation(i, j, &planted));
        } else {
            let ratio = (planted[i] - planted[j] + 0.15 * z).exp();
            obs.push(Observation::new(i, j, ratio, 0.85, RATER, 1.0));
        }
    }

    let mut cfg = Config::default();
    cfg.huber_k = huber_k;
    let mut eng = engine_with_cfg(n, cfg);
    eng.ingest(&obs);
    (eng.solve().scores, planted)
}

/// Kendall tau-a (concordant - discordant) / total pairs, ignoring ties.
/// Falsifiable by any implementation that inverts sign conventions, swaps i/j,
/// or otherwise scrambles relative order.
fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let da = a[i] - a[j];
            let db = b[i] - b[j];
            if da == 0.0 || db == 0.0 {
                continue;
            }
            if (da > 0.0) == (db > 0.0) {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let total = concordant + discordant;
    if total == 0 {
        0.0
    } else {
        (concordant - discordant) as f64 / total as f64
    }
}

/// Indices of the top-`k` items by score, descending, ties broken by index.
fn top_k_indices(scores: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
    idx.into_iter().take(k).collect()
}

/// Rank (0 = highest score) of every item.
fn ranks_desc(scores: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
    let mut rank = vec![0usize; scores.len()];
    for (r, &i) in idx.iter().enumerate() {
        rank[i] = r;
    }
    rank
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / (xs.len() as f64)
}

/// Run one planted-recovery replica and return (tau, recovered_scores, planted_scores).
fn run_planted_replica(
    n: usize,
    extra_edges: usize,
    noise_sd: f64,
    confidence: f64,
    seed: u64,
) -> (f64, Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let planted = planted_scores(n, 2.0, &mut rng);
    let edges = edge_topology(n, extra_edges, &mut rng);
    let obs: Vec<Observation> = edges
        .iter()
        .map(|&(i, j)| noisy_observation(i, j, &planted, noise_sd, confidence, &mut rng))
        .collect();

    let mut eng = engine(n);
    eng.ingest(&obs);
    let summary = eng.solve();
    let tau = kendall_tau(&summary.scores, &planted);
    (tau, summary.scores, planted)
}

// =======================================================================
// 1. PLANTED RECOVERY
// =======================================================================

/// Claim: for a moderate-noise pairwise ratio graph consistent with a known
/// latent ranking, the recovered scores' Kendall tau against the planted
/// order exceeds a demanding threshold, on average over many seeded replicas.
fn planted_recovery_ensemble(n: usize, min_mean_tau: f64, min_worst_tau: f64) {
    const SEEDS: u64 = 40;
    let extra = 2 * n;
    let mut taus = Vec::with_capacity(SEEDS as usize);
    for seed in 0..SEEDS {
        let (tau, _, _) = run_planted_replica(n, extra, 0.25, 0.85, seed ^ (n as u64) << 32);
        taus.push(tau);
    }
    let avg = mean(&taus);
    let worst = taus.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        avg > min_mean_tau,
        "n={n}: mean tau {avg:.4} did not exceed {min_mean_tau} across {SEEDS} seeds (taus={taus:?})"
    );
    assert!(
        worst > min_worst_tau,
        "n={n}: worst-case tau {worst:.4} fell below {min_worst_tau} across {SEEDS} seeds (taus={taus:?})"
    );
}

#[test]
fn planted_recovery_tau_n8() {
    planted_recovery_ensemble(8, 0.90, 0.55);
}

#[test]
fn planted_recovery_tau_n16() {
    planted_recovery_ensemble(16, 0.90, 0.60);
}

#[test]
fn planted_recovery_tau_n32() {
    planted_recovery_ensemble(32, 0.90, 0.65);
}

/// Claim: mean Kendall tau improves (does not regress) as the number of
/// observations doubles, holding everything else fixed. This is the
/// information-monotonicity property any sane aggregator must satisfy.
#[test]
fn planted_recovery_tau_improves_as_observations_double() {
    const SEEDS: u64 = 50;
    let n = 16usize;
    // Observation counts: (n-1) spanning edges + extra, so extra doubles: 4, 8, 16, 32.
    let extras = [4usize, 8, 16, 32];
    let mut mean_taus = Vec::new();

    for &extra in &extras {
        let mut taus = Vec::with_capacity(SEEDS as usize);
        for seed in 0..SEEDS {
            let (tau, _, _) = run_planted_replica(n, extra, 0.35, 0.8, seed ^ 0xABCD);
            taus.push(tau);
        }
        mean_taus.push(mean(&taus));
    }

    // Monotone non-decreasing on average, with a small slack for noise floor
    // effects near tau=1 (a correct solver can plateau, but must not regress).
    for w in mean_taus.windows(2) {
        assert!(
            w[1] >= w[0] - 0.02,
            "tau should not regress as observation count grows: {:?} (extras={:?})",
            mean_taus,
            extras
        );
    }
    // And the overall improvement from fewest to most observations must be real.
    let first = *mean_taus.first().unwrap();
    let last = *mean_taus.last().unwrap();
    assert!(
        last - first > 0.05,
        "expected a real improvement in mean tau from {first:.4} (extra={}) to {last:.4} (extra={}), got delta={:.4}",
        extras[0],
        extras[extras.len() - 1],
        last - first
    );
}

// =======================================================================
// 2. OUTLIER ROBUSTNESS (Huber)
// =======================================================================

/// Claim: with 5% of observations replaced by adversarial, maximally
/// confident, reversed judgements, the recovered top-3 (by score) is
/// unchanged from the clean run in the overwhelming majority of seeded
/// replicas. Huber downweighting must catch these outliers.
#[test]
fn outlier_top3_stable_at_5pct_adversarial() {
    const SEEDS: u64 = 120;
    let n = 14usize;
    let extra = n; // sparser graph => each edge (including outliers) has real leverage
    let planted = clustered_top3_truth(n); // a wide, unambiguous gap isolates "the top-3"
    let mut unchanged = 0u32;
    for seed in 0..SEEDS {
        let (clean_scores, _) = build_run(n, &planted, extra, 0.0, seed, Config::default().huber_k);
        let (corrupt_scores, _) =
            build_run(n, &planted, extra, 0.05, seed, Config::default().huber_k);
        // Compare as sets: the claim is about *membership* in the top-3, not
        // the internal order among those three, which a single fair edge can
        // legitimately perturb even with no corruption at all.
        let mut clean_top3 = top_k_indices(&clean_scores, 3);
        let mut corrupt_top3 = top_k_indices(&corrupt_scores, 3);
        clean_top3.sort_unstable();
        corrupt_top3.sort_unstable();
        if clean_top3 == corrupt_top3 {
            unchanged += 1;
        }
    }
    let rate = unchanged as f64 / SEEDS as f64;
    assert!(
        rate >= 0.90,
        "top-3 should survive 5% adversarial corruption in >=90% of seeds, got {rate:.2} ({unchanged}/{SEEDS})"
    );
}

/// Claim: with 15% adversarial corruption (a harder attack, past the
/// classical ~30-50% Huber breakdown margin is not expected but 15% should
/// still be substantially absorbed), rank displacement stays bounded rather
/// than scrambling the ranking arbitrarily, and the single top item stays
/// within a generous band of the top.
#[test]
fn outlier_rank_displacement_bounded_at_15pct_adversarial() {
    const SEEDS: u64 = 120;
    let n = 14usize;
    let extra = 2 * n;
    let planted = evenly_spaced_truth(n);

    let mut total_mean_displacement = 0.0;
    let mut top1_within_band = 0u32;

    for seed in 0..SEEDS {
        let (clean_scores, _) = build_run(n, &planted, extra, 0.0, seed, Config::default().huber_k);
        let (corrupt_scores, _) =
            build_run(n, &planted, extra, 0.15, seed, Config::default().huber_k);

        let clean_ranks = ranks_desc(&clean_scores);
        let corrupt_ranks = ranks_desc(&corrupt_scores);

        let displacement: f64 = clean_ranks
            .iter()
            .zip(corrupt_ranks.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / n as f64;
        total_mean_displacement += displacement;

        let true_top1 = top_k_indices(&clean_scores, 1)[0];
        let corrupt_rank_of_top1 = corrupt_ranks[true_top1];
        if corrupt_rank_of_top1 < 5 {
            top1_within_band += 1;
        }
    }

    let avg_displacement = total_mean_displacement / SEEDS as f64;
    // A random shuffle of n=14 has expected per-item rank displacement ~ n/3 ≈ 4.7.
    // Bounded robustness means we stay well under that even at 15% corruption.
    assert!(
        avg_displacement < 3.0,
        "average per-item rank displacement under 15% adversarial corruption too high: {avg_displacement:.3} (random-shuffle baseline ~4.7)"
    );

    // Random-shuffle baseline for landing in the top-5 of n=14 is 5/14 ≈ 0.357;
    // a bounded-robustness solver should clear that by a wide margin.
    let band_rate = top1_within_band as f64 / SEEDS as f64;
    assert!(
        band_rate >= 0.75,
        "the true top item should stay within the top-5 in >=75% of seeds under 15% corruption (random-shuffle baseline ~0.36), got {band_rate:.2}"
    );
}

/// Claim: the robust (Huber) fit recovers the planted order measurably
/// better than a naive weighted-least-squares fit under the same adversarial
/// corruption. We approximate "Huber disabled" by setting `huber_k` to a
/// huge value, which makes the Huber weight-clip threshold effectively
/// unreachable so every IRLS iteration keeps `z[k] == 1.0` (see
/// `solve_irls_huber`) — i.e. plain weighted least squares.
///
/// `huber_k` is 1e15 rather than 1e6 here: on graphs with continuous,
/// generic per-edge noise (as generated below) 1e6 is already enough
/// headroom over any residual scale, but 1e15 costs nothing and stays safe
/// even if a run happens to produce several near-tied residuals. The
/// `point_confidence_is_metadata_in_an_anchored_triangle` and
/// `huber_mad_scale_collapses_on_near_tied_residuals` tests below explain why
/// a merely-large `huber_k` is not always enough to disable Huber.
#[test]
fn huber_robust_fit_beats_naive_least_squares_under_outliers() {
    const SEEDS: u64 = 40;
    let n = 14usize;
    let extra = n;
    let planted = evenly_spaced_truth(n);
    let mut robust_taus = Vec::with_capacity(SEEDS as usize);
    let mut naive_taus = Vec::with_capacity(SEEDS as usize);

    for seed in 0..SEEDS {
        let (robust_scores, planted_out) =
            build_run(n, &planted, extra, 0.15, seed, Config::default().huber_k);
        let (naive_scores, _) = build_run(n, &planted, extra, 0.15, seed, 1.0e15);
        debug_assert_eq!(planted_out, planted);
        robust_taus.push(kendall_tau(&robust_scores, &planted));
        naive_taus.push(kendall_tau(&naive_scores, &planted));
    }

    let robust_mean = mean(&robust_taus);
    let naive_mean = mean(&naive_taus);
    assert!(
        robust_mean > naive_mean + 0.05,
        "robust (Huber) mean tau {robust_mean:.4} should beat naive least-squares mean tau {naive_mean:.4} by a real margin under 15% adversarial corruption"
    );
}

// =======================================================================
// 3. GAUGE INVARIANCE
// =======================================================================

/// Claim: relabeling items (a permutation of indices) changes which node is
/// pinned to the internal gauge origin, but must not change the recovered
/// *pairwise* score differences between corresponding items, nor the
/// recovered order. This directly stresses `pin_nodes` (which always pins
/// the minimum-index node per component) for hidden asymmetries.
#[test]
fn gauge_invariance_permutation_preserves_pairwise_deltas() {
    const SEEDS: u64 = 20;
    let n = 12usize;

    for seed in 0..SEEDS {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x9EED);
        let planted = planted_scores(n, 2.0, &mut rng);
        let edges = edge_topology(n, 2 * n, &mut rng);

        // Fixed noise draws shared between the two runs (frozen up front) so the
        // only difference between the two engines is the relabeling itself.
        let mut noise_rng = StdRng::seed_from_u64(seed ^ 0x1234);
        let noises: Vec<f64> = edges.iter().map(|_| gaussian(&mut noise_rng)).collect();
        let noise_sd = 0.2;
        let confidence = 0.8;

        // Baseline (identity permutation).
        let base_obs: Vec<Observation> = edges
            .iter()
            .zip(noises.iter())
            .map(|(&(i, j), &z)| {
                let ratio = (planted[i] - planted[j] + noise_sd * z).exp();
                Observation::new(i, j, ratio, confidence, RATER, 1.0)
            })
            .collect();
        let mut base_eng = engine(n);
        base_eng.ingest(&base_obs);
        let base_scores = base_eng.solve().scores;

        // A derangement-ish permutation: reverse the index order. Item `i`'s
        // truth now lives at slot `perm[i]` in the relabeled engine.
        let perm: Vec<usize> = (0..n).rev().collect();

        let perm_obs: Vec<Observation> = edges
            .iter()
            .zip(noises.iter())
            .map(|(&(i, j), &z)| {
                let (pi, pj) = (perm[i], perm[j]);
                let ratio = (planted[i] - planted[j] + noise_sd * z).exp();
                Observation::new(pi, pj, ratio, confidence, RATER, 1.0)
            })
            .collect();
        let mut perm_eng = engine(n);
        perm_eng.ingest(&perm_obs);
        let perm_scores = perm_eng.solve().scores;

        for i in 0..n {
            for j in (i + 1)..n {
                let base_delta = base_scores[i] - base_scores[j];
                let perm_delta = perm_scores[perm[i]] - perm_scores[perm[j]];
                assert!(
                    (base_delta - perm_delta).abs() < 1e-6,
                    "seed {seed}: pairwise delta for ({i},{j}) not preserved under relabeling: base={base_delta:.6} perm={perm_delta:.6}"
                );
            }
        }
    }
}

/// Claim: the log-ratio observation model only ever depends on *differences*
/// of the latent truth. Translating the origin of the (hypothetical) planted
/// truth by an arbitrary constant must leave the generated observations, and
/// therefore the recovered fit, unchanged — and the recovered pairwise
/// differences must track the planted pairwise differences within a noise
/// tolerance that improves as we average across an ensemble.
#[test]
fn gauge_invariance_shift_of_latent_origin_leaves_recovery_unchanged() {
    const SEEDS: u64 = 25;
    let n = 10usize;
    let shift = 137.0; // arbitrary, "meaningless" choice of where zero is

    let mut errs = Vec::new();
    for seed in 0..SEEDS {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5EED);
        let planted = planted_scores(n, 1.5, &mut rng);
        let shifted: Vec<f64> = planted.iter().map(|s| s + shift).collect();
        let edges = edge_topology(n, 2 * n, &mut rng);

        let mut noise_rng = StdRng::seed_from_u64(seed ^ 0xF00D);
        let noises: Vec<f64> = edges.iter().map(|_| gaussian(&mut noise_rng)).collect();

        let build = |truth: &[f64]| -> Vec<Observation> {
            edges
                .iter()
                .zip(noises.iter())
                .map(|(&(i, j), &z)| {
                    let ratio = (truth[i] - truth[j] + 0.2 * z).exp();
                    Observation::new(i, j, ratio, 0.85, RATER, 1.0)
                })
                .collect()
        };

        let obs_unshifted = build(&planted);
        let obs_shifted = build(&shifted);

        // The generated observations must agree to near machine precision:
        // only differences of the latent truth matter to the model (a large
        // additive shift only introduces ordinary floating-point cancellation
        // error, not a structural dependence on the shift itself).
        for (a, b) in obs_unshifted.iter().zip(obs_shifted.iter()) {
            assert!(
                (a.ratio - b.ratio).abs() < 1e-9,
                "seed {seed}: shifting the latent origin changed a generated ratio beyond fp noise ({} vs {})",
                a.ratio,
                b.ratio
            );
        }

        let mut eng = engine(n);
        eng.ingest(&obs_shifted);
        let scores = eng.solve().scores;

        // Recovered pairwise deltas should track planted pairwise deltas
        // (the shift cancels identically in both).
        let mut sq_err = 0.0;
        let mut count = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let recovered_delta = scores[i] - scores[j];
                let planted_delta = planted[i] - planted[j]; // == shifted[i]-shifted[j]
                sq_err += (recovered_delta - planted_delta).powi(2);
                count += 1.0;
            }
        }
        errs.push((sq_err / count).sqrt());
    }

    let avg_rmse = mean(&errs);
    assert!(
        avg_rmse < 0.35,
        "average RMSE of recovered vs planted pairwise deltas too high: {avg_rmse:.4} across {SEEDS} seeds"
    );
}

// =======================================================================
// 4. STATED CONFIDENCE IS METADATA
// =======================================================================

#[test]
fn point_confidence_does_not_change_a_two_item_fit() {
    let ratio = 4.0;
    let mut lo_eng = engine(2);
    lo_eng.ingest(&[Observation::new(0, 1, ratio, 0.1, RATER, 1.0)]);
    let lo = lo_eng.solve();

    let mut hi_eng = engine(2);
    hi_eng.ingest(&[Observation::new(0, 1, ratio, 0.9, RATER, 1.0)]);
    let hi = hi_eng.solve();

    assert_eq!(
        lo.scores.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        hi.scores.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        lo.diag_cov.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        hi.diag_cov.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn point_confidence_is_metadata_in_an_anchored_triangle() {
    let build = |conflict_confidence: f64| -> Vec<f64> {
        let mut cfg = Config::default();
        cfg.huber_k = 1.0e18;
        let mut eng = engine_with_cfg(3, cfg);
        eng.ingest(&[
            Observation::new(0, 1, 1.0, 1.0, RATER, 50.0),
            Observation::new(1, 2, 1.0, 1.0, RATER, 50.0),
            Observation::new(2, 0, 6.0, conflict_confidence, RATER, 1.0),
        ]);
        eng.solve().scores
    };

    let lo = build(0.05);
    let hi = build(0.95);
    assert_eq!(
        lo.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        hi.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

// =======================================================================
// 5. RATIO LADDER SANITY
// =======================================================================

/// Claim: on a bare two-item graph, the recovered score gap is a strictly
/// increasing function of the elicited ratio, all the way up the canonical
/// prompt ratio ladder (capped at 26.0). A solver that clamped too
/// aggressively, mis-clamped `max_log_ratio`, or applied a non-monotone
/// transform to the ratio would fail this.
#[test]
fn ratio_ladder_monotone_gap_recovery() {
    let mut gaps = Vec::with_capacity(RATIO_LADDER.len());
    for &ratio in RATIO_LADDER {
        let mut eng = engine(2);
        eng.ingest(&[Observation::new(0, 1, ratio, 0.9, RATER, 1.0)]);
        let scores = eng.solve().scores;
        gaps.push((scores[0] - scores[1]).abs());
    }

    for w in gaps.windows(2) {
        assert!(
            w[1] > w[0] + 1e-9,
            "recovered gap must strictly increase along the ratio ladder: {:?} for ladder {:?}",
            gaps,
            RATIO_LADDER
        );
    }
}

/// Claim: an observation at the ladder cap (26.0) produces a materially
/// larger recovered gap than one at the ladder's midpoint (1.5x).
#[test]
fn ratio_ladder_cap_exceeds_midladder_gap() {
    let cap_ratio = *RATIO_LADDER.last().unwrap();
    assert_eq!(cap_ratio, 26.0, "expected ladder cap to still be 26.0");
    let mid_ratio = 1.5;
    assert!(RATIO_LADDER.contains(&mid_ratio));

    let gap_for = |ratio: f64| -> f64 {
        let mut eng = engine(2);
        eng.ingest(&[Observation::new(0, 1, ratio, 0.9, RATER, 1.0)]);
        let scores = eng.solve().scores;
        (scores[0] - scores[1]).abs()
    };

    let cap_gap = gap_for(cap_ratio);
    let mid_gap = gap_for(mid_ratio);
    assert!(
        cap_gap > mid_gap * 3.0,
        "ladder cap (26.0) gap {cap_gap:.4} should be much larger than midladder (1.5x) gap {mid_gap:.4}"
    );
}

// =======================================================================
// REGRESSION: real solver bug found while reviewing this suite
// =======================================================================

/// Claim under attack: setting `huber_k` to a "large enough to disable
/// Huber" value (1e6, the value this suite originally used for that purpose
/// in two now-fixed tests) reliably reduces IRLS+Huber to plain weighted
/// least squares, i.e. every edge keeps `z[k] == 1.0` in `solve_irls_huber`.
///
/// This is FALSE for graphs where several edges agree closely enough that
/// their residuals are tied up to floating-point noise (~1e-9 to 1e-16),
/// which is exactly what happens whenever two or more strong, mutually
/// consistent "anchor" observations are present (a completely ordinary
/// real-world situation, not a contrived edge case). `solve_irls_huber`
/// estimates the outlier scale as `mad(residuals)`, with a guard that only
/// falls back to `max(|residuals|)` when `mad <= cfg.tiny` (1e-18). But a
/// MAD computed from near-tied *non-exactly-equal* floating-point residuals
/// lands around 1e-9..1e-16 — well above `cfg.tiny`, so the guard never
/// fires — and `delta = huber_k * scale` collapses to a similarly tiny
/// number. Every residual, including the legitimately large one from a
/// genuine disagreement, then gets clipped by Huber (`z[k] = delta /
/// |residual|`), and the fit collapses toward all-scores-near-zero
/// regardless of the actual evidence. Increasing `huber_k` by another ~9-12
/// orders of magnitude (to ~1e15-1e18) works around it, which is what this
/// suite's other tests now do — but the underlying MAD-collapse is a real
/// robustness defect in `solve_irls_huber`'s "is MAD effectively zero"
/// check, independent of any test's chosen `huber_k`.
///
/// Concretely: two anchor edges (0,1) and (1,2), each declaring exact
/// equality (ratio 1.0) with reps=50, plus one conflicting edge (2,0) with
/// ratio 6.0 at confidence 0.95. With `huber_k = 1e6` the naive
/// weighted-least-squares answer (which `huber_k = 1e18` correctly
/// recovers, and which plain algebra on the reduced 2x2 normal equations
/// confirms independently) is `|score[2] - score[0]| ≈ 0.0624`. The actual
/// engine instead returns something on the order of 1e-5 — off by roughly
/// three orders of magnitude — because the anchors' own near-zero residuals
/// get treated as "the outlier scale" and used to clip everything,
/// including the conflicting edge.
///
/// Regression test: this bug was found by adversarial review of this suite
/// and FIXED in `solve_irls_huber` with a relative degeneracy floor — a MAD
/// below 1e-8 of the max-abs residual now falls back to the max-abs scale,
/// exactly as for MAD == 0.
#[test]
fn huber_mad_scale_collapses_on_near_tied_residuals() {
    let mut cfg = Config::default();
    cfg.huber_k = 1.0e6; // "should" disable Huber per its own doc comment; does not.
    let mut eng = engine_with_cfg(3, cfg);
    eng.ingest(&[
        Observation::new(0, 1, 1.0, 1.0, RATER, 50.0),
        Observation::new(1, 2, 1.0, 1.0, RATER, 50.0),
        Observation::new(2, 0, 6.0, 0.95, RATER, 1.0),
    ]);
    let scores = eng.solve().scores;
    let dev = (scores[2] - scores[0]).abs();

    // Ground truth from the reduced 2x2 normal equations (node 0 pinned):
    // the point path now gives the conflicting edge unit weight, so
    // s2 = ln(6) / (25 + 1).
    let expected = 6.0f64.ln() / 26.0;
    assert!(
        (dev - expected).abs() < 0.01,
        "expected naive-WLS deviation ~{expected:.6} (huber nominally disabled at huber_k=1e6), \
got {dev:.10} — the MAD-from-floating-point-noise bug is crushing the fit"
    );
}
