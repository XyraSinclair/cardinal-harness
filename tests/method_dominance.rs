//! Method head-to-heads on planted regimes.
//!
//! These tests pin *today's measured* relationships between cardinal ratio
//! mode, cardinal ordinal mode, and the per-item Likert baseline, using the
//! built-in synthetic evaluation harness (`src/rerank/evaluation.rs`). They
//! are evidence, not universal-superiority claims: several tests below
//! *also* pin regimes where ratio mode loses, because the repo's culture is
//! honest evidence over marketing. If a number here drifts, that is either a
//! regression or a genuine improvement in the solver -- either way, it
//! should surface as a diff in this file, not silently.
//!
//! Four claims are attacked:
//!
//! 1. SCALE-COMPRESSION REGIME: on `scale_compression_40` (the suite's
//!    documented strong regime for cardinal ratio mode -- one outlier
//!    collapses a 10-level Likert scale onto the remaining items), ratio
//!    mode must beat the Likert baseline on Kendall tau.
//! 2. RATIO vs ORDINAL INFORMATION: averaged across the whole built-in
//!    suite, ratio mode's mean Kendall tau must be at least ordinal mode's
//!    (magnitude information must not hurt *in aggregate*). The companion
//!    high-noise/high-outlier fixture marks wrong edges as highly confident;
//!    it now pins that ignoring that uncalibrated label restores ratio mode's
//!    advantage under robust fitting.
//! 3. BUDGET EFFICIENCY: search the checked-in cases where ratio mode at half
//!    budget matches or beats Likert at full budget on tau. Three cases clear
//!    this bar after removing the confidence transform: `clean_ordering_10`,
//!    `inconsistent_cycle_12`, and `outlier_robustness_25`. Pin the exact set
//!    so future losses and gains both require an explicit accounting.
//! 4. TRAJECTORY MONOTONICITY: `estimate_topk_error` trajectories emitted by
//!    the harness must be weakly improving in aggregate: final error <=
//!    initial error, for every built-in case in ratio mode, and (via a
//!    seeded ensemble) under adversarial noise/outlier pressure too.

use std::collections::HashSet;

use cardinal_harness::rerank::evaluation::{
    run_likert_baseline_case, run_synthetic_case_with_config, synthetic_cases, LikertEvalConfig,
    PairwiseEvalConfig, SyntheticAttribute, SyntheticCase, SyntheticPairwiseMode,
};
use cardinal_harness::rerank::types::MultiRerankTopKSpec;

// =============================================================================
// Local synthetic-case builders (independent of the checked-in suite, so
// these tests do not merely re-derive numbers already pinned elsewhere).
// =============================================================================

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
        prune_p_topk_below: None,
    }
}

/// A scale-compression case in the spirit of `scale_compression_40`: one
/// extreme outlier collapses a tightly packed cluster onto a single Likert
/// bucket, with the true top-k hidden late in input order so tie-breaking by
/// position cannot accidentally look correct. Parameterized by seed and
/// noise so we can run it as a seeded ensemble rather than a single draw.
fn compressed_case(seed: u64, noise_sigma: f64) -> SyntheticCase {
    SyntheticCase {
        name: "ensemble_compressed",
        attributes: vec![SyntheticAttribute {
            id: "attr_compressed",
            weight: 1.0,
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
        noise_sigma,
        refusal_rate: 0.0,
        outlier_rate: 0.0,
        seed,
    }
}

/// An adversarial regime: small n, heavy log-normal noise (sigma=0.8), and a
/// 15% flipped-with-high-confidence outlier rate. This is deliberately
/// hostile to cardinal ratio mode, whose observations carry a magnitude that
/// a flipped high-confidence outlier can badly mis-set; ordinal mode's fixed
/// small ratio is comparatively insulated from that failure mode.
fn adversarial_case(seed: u64) -> SyntheticCase {
    SyntheticCase {
        name: "ensemble_adversarial",
        attributes: vec![SyntheticAttribute {
            id: "attr_adversarial",
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
        seed,
    }
}

fn ratio_cfg() -> PairwiseEvalConfig {
    PairwiseEvalConfig {
        mode: SyntheticPairwiseMode::Ratio,
    }
}

fn ordinal_cfg() -> PairwiseEvalConfig {
    PairwiseEvalConfig {
        mode: SyntheticPairwiseMode::Ordinal,
    }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

// =============================================================================
// Claim 1: scale-compression regime.
// =============================================================================

/// On the suite's documented strong regime, cardinal ratio mode must
/// materially beat the 10-level Likert baseline on Kendall tau (all items,
/// gate-free) and on top-k precision. This is a solver regression pin: if
/// the IRLS+Huber fit stops exploiting ratio magnitude here, this test
/// catches it immediately.
#[test]
fn scale_compression_case_ratio_beats_likert_on_tau_and_topk() {
    let cardinal = run_synthetic_case_with_config(
        &synthetic_cases()
            .into_iter()
            .find(|c| c.name == "scale_compression_40")
            .expect("scale_compression_40 case must exist in the built-in suite"),
        ratio_cfg(),
    )
    .expect("cardinal ratio run should succeed");
    let likert = run_likert_baseline_case(
        &synthetic_cases()
            .into_iter()
            .find(|c| c.name == "scale_compression_40")
            .expect("scale_compression_40 case must exist in the built-in suite"),
        LikertEvalConfig::default(),
    )
    .expect("likert baseline run should succeed");

    let ratio_tau = cardinal.metrics.kendall_tau_all;
    let likert_tau = likert.metrics.kendall_tau_all;

    assert!(
        ratio_tau >= 0.65,
        "ratio tau should stay strong on the compression regime, got {ratio_tau}"
    );
    assert!(
        likert_tau <= 0.40,
        "10-level Likert should stay weak on the compression regime, got {likert_tau}"
    );
    assert!(
        ratio_tau - likert_tau >= 0.35,
        "ratio should materially beat Likert on the compression regime: ratio {ratio_tau}, likert {likert_tau}"
    );

    assert!(
        cardinal.metrics.topk_precision >= 0.99,
        "ratio mode should recover the compressed top-k exactly, got precision {}",
        cardinal.metrics.topk_precision
    );
    assert!(
        likert.metrics.topk_precision <= 0.35,
        "Likert should lose the compressed top-k, got precision {}",
        likert.metrics.topk_precision
    );
}

/// The scale-compression win should not be an artifact of one lucky seed.
/// Sweep 25 seeds at a fixed moderate noise level and require ratio mode to
/// win on every single draw, with a healthy mean margin.
#[test]
fn scale_compression_regime_ratio_beats_likert_ensemble() {
    let seeds: Vec<u64> = (5_000..5_025).collect();
    let noise_sigma = 0.15;

    let mut margins = Vec::with_capacity(seeds.len());
    let mut wins = 0usize;

    for &seed in &seeds {
        let case = compressed_case(seed, noise_sigma);
        let ratio =
            run_synthetic_case_with_config(&case, ratio_cfg()).expect("ratio run should succeed");
        let likert = run_likert_baseline_case(&case, LikertEvalConfig::default())
            .expect("likert run should succeed");

        let ratio_tau = ratio.metrics.kendall_tau_all;
        let likert_tau = likert.metrics.kendall_tau_all;
        if ratio_tau > likert_tau {
            wins += 1;
        }
        margins.push(ratio_tau - likert_tau);
    }

    let mean_margin = mean(&margins);
    let min_margin = margins.iter().cloned().fold(f64::INFINITY, f64::min);

    assert_eq!(
        wins,
        seeds.len(),
        "ratio mode should beat Likert on every seeded replica of the compression regime, won {wins}/{}",
        seeds.len()
    );
    assert!(
        min_margin > 0.0,
        "every replica should show a strictly positive margin, worst was {min_margin}"
    );
    assert!(
        mean_margin >= 0.25,
        "mean margin across {} seeds should stay comfortably positive, got {mean_margin}",
        seeds.len()
    );
}

/// A planted-truth compression case independent of the checked-in suite: we
/// know the exact top-5 by construction (item 0 is the extreme outlier and
/// items 32..=39 form a strictly descending tail above the compressed
/// cluster). Ratio mode must recover that exact set; Likert must not.
#[test]
fn planted_truth_independent_compressed_case_ratio_recovers_exact_topk() {
    let case = compressed_case(777, 0.1);
    let ratio =
        run_synthetic_case_with_config(&case, ratio_cfg()).expect("ratio run should succeed");
    let likert = run_likert_baseline_case(&case, LikertEvalConfig::default())
        .expect("likert run should succeed");

    assert!(
        ratio.metrics.topk_precision >= 0.99,
        "ratio mode should exactly recover the planted top-5, got {}",
        ratio.metrics.topk_precision
    );
    assert!(
        ratio.metrics.topk_recall >= 0.99,
        "ratio mode should exactly recover the planted top-5 (recall), got {}",
        ratio.metrics.topk_recall
    );
    assert!(
        likert.metrics.topk_precision <= 0.5,
        "Likert should not recover the planted top-5 under 10-level quantization, got {}",
        likert.metrics.topk_precision
    );
}

// =============================================================================
// Claim 2: ratio vs ordinal information.
// =============================================================================

/// Averaged across the whole built-in suite, cardinal ratio mode's mean
/// Kendall tau must be at least ordinal mode's: magnitude information must
/// not hurt in aggregate. This is a suite-average claim, not a per-case one
/// -- individual cases can and do flip (see the adversarial ensemble below).
#[test]
fn ratio_mode_beats_ordinal_mode_on_average_tau_across_built_in_suite() {
    let cases = synthetic_cases();
    assert!(
        cases.len() >= 4,
        "suite should have a meaningful case count"
    );

    let mut ratio_taus = Vec::with_capacity(cases.len());
    let mut ordinal_taus = Vec::with_capacity(cases.len());

    for case in &cases {
        let ratio =
            run_synthetic_case_with_config(case, ratio_cfg()).expect("ratio run should succeed");
        let ordinal = run_synthetic_case_with_config(case, ordinal_cfg())
            .expect("ordinal run should succeed");
        ratio_taus.push(ratio.metrics.kendall_tau_all);
        ordinal_taus.push(ordinal.metrics.kendall_tau_all);
    }

    let ratio_avg = mean(&ratio_taus);
    let ordinal_avg = mean(&ordinal_taus);

    assert!(
        ratio_avg >= 0.5 && ordinal_avg >= 0.5,
        "both modes should be well above chance on this suite: ratio {ratio_avg}, ordinal {ordinal_avg}"
    );
    // MEASUREMENT HISTORY: before the exploration anchor-diversity fix
    // (2026-07-04, issue #43), ratio measured 0.703 vs ordinal 0.650.
    // The anchor fix flipped the relationship to ratio 0.648 vs ordinal
    // 0.726 under the old confidence transform. Removing that
    // anti-calibrated transform restores the expected direction without
    // changing the observations: ratio 0.808 vs ordinal 0.726 on this
    // deterministic suite. The pin below would fail under either the old
    // confidence-weighted path or a regression that discards magnitudes.
    assert!(
        ratio_avg - ordinal_avg >= 0.02,
        "ratio magnitude should beat direction-only observations on this suite: ratio {ratio_avg}, ordinal {ordinal_avg}"
    );
    assert!(
        ratio_avg - ordinal_avg <= 0.3,
        "the ratio/ordinal gap should be a real but bounded effect, got {}",
        ratio_avg - ordinal_avg
    );
}

/// Regression for the deleted confidence transform: this fixture marks
/// flipped outliers as highly confident. The former solver amplified those
/// wrong edges and ratio mode lost almost every seed. Confidence is now
/// metadata, so robust fitting—not uncalibrated self-assessment—controls the
/// damage, and ratio magnitude should recover a majority of seeds.
#[test]
fn ratio_information_survives_high_confidence_outlier_labels() {
    let seeds: Vec<u64> = (6_000..6_030).collect();

    let mut ratio_taus = Vec::with_capacity(seeds.len());
    let mut ordinal_taus = Vec::with_capacity(seeds.len());
    let mut ratio_wins = 0usize;

    for &seed in &seeds {
        let case = adversarial_case(seed);
        let ratio =
            run_synthetic_case_with_config(&case, ratio_cfg()).expect("ratio run should succeed");
        let ordinal = run_synthetic_case_with_config(&case, ordinal_cfg())
            .expect("ordinal run should succeed");
        let rt = ratio.metrics.kendall_tau_all;
        let ot = ordinal.metrics.kendall_tau_all;
        if rt >= ot {
            ratio_wins += 1;
        }
        ratio_taus.push(rt);
        ordinal_taus.push(ot);
    }

    let ratio_avg = mean(&ratio_taus);
    let ordinal_avg = mean(&ordinal_taus);

    assert!(
        ratio_wins >= 20,
        "ratio mode should recover most seeds once stated confidence is metadata, won {ratio_wins}/{}",
        seeds.len()
    );
    assert!(
        ratio_avg > ordinal_avg,
        "ratio mode should beat ordinal on average here: ordinal {ordinal_avg}, ratio {ratio_avg}"
    );
}

// =============================================================================
// Claim 3: budget efficiency.
// =============================================================================

/// Search the checked-in suite for cases where cardinal ratio mode at half
/// budget matches or beats Likert at full budget on Kendall tau. Removing
/// the anti-calibrated confidence transform expands the measured set from
/// only `clean_ordering_10` to three named cases. Pin the exact set so either
/// a regression or another real efficiency gain demands an explicit update.
#[test]
fn budget_efficiency_half_budget_pin_matches_measured_cases() {
    const TIE_EPS: f64 = 1e-9;
    let cases = synthetic_cases();

    let mut qualifying: Vec<&'static str> = Vec::new();
    let mut clean_ordering_ratio_half_attempted = None;
    let mut clean_ordering_likert_full_attempted = None;

    for case in &cases {
        let full_budget = case.comparison_budget.unwrap_or_else(|| {
            4 * case.attributes[0].scores.len().max(1) * case.attributes.len().max(1)
        });
        let mut half_case = case.clone();
        half_case.comparison_budget = Some((full_budget / 2).max(1));

        let ratio_half = run_synthetic_case_with_config(&half_case, ratio_cfg())
            .expect("half-budget ratio run should succeed");
        let likert_full = run_likert_baseline_case(case, LikertEvalConfig::default())
            .expect("full-budget likert run should succeed");

        let ratio_half_tau = ratio_half.metrics.kendall_tau_all;
        let likert_full_tau = likert_full.metrics.kendall_tau_all;

        if case.name == "clean_ordering_10" {
            clean_ordering_ratio_half_attempted = Some(ratio_half.metrics.comparisons_attempted);
            clean_ordering_likert_full_attempted = Some(likert_full.metrics.ratings_attempted);
        }

        if ratio_half_tau + TIE_EPS >= likert_full_tau {
            qualifying.push(case.name);
        }
    }

    let qualifying_set: HashSet<&'static str> = qualifying.into_iter().collect();
    assert_eq!(
        qualifying_set,
        HashSet::from([
            "clean_ordering_10",
            "inconsistent_cycle_12",
            "outlier_robustness_25",
        ]),
        "the half-budget-matches-full-budget-Likert cases changed; inspect whether this is a regression or a real efficiency change"
    );

    let ratio_half_attempted =
        clean_ordering_ratio_half_attempted.expect("clean_ordering_10 must be in the suite");
    let likert_full_attempted =
        clean_ordering_likert_full_attempted.expect("clean_ordering_10 must be in the suite");
    assert!(
        ratio_half_attempted * 2 <= likert_full_attempted,
        "the qualifying case should actually use materially fewer queries: ratio_half {ratio_half_attempted}, likert_full {likert_full_attempted}"
    );
}

/// Budget sanity: at the suite's stock configuration, the main planning
/// loop's budget accounting must never overrun. This is what makes the
/// efficiency claim above meaningful rather than an artifact of a leaky stop
/// condition.
#[test]
fn comparisons_attempted_never_exceeds_configured_budget() {
    for case in &synthetic_cases() {
        let full_budget = case.comparison_budget.unwrap_or_else(|| {
            4 * case.attributes[0].scores.len().max(1) * case.attributes.len().max(1)
        });
        let full = run_synthetic_case_with_config(case, ratio_cfg())
            .expect("full-budget ratio run should succeed");
        assert!(
            full.metrics.comparisons_attempted <= full_budget,
            "case {} attempted {} comparisons against a budget of {}",
            case.name,
            full.metrics.comparisons_attempted,
            full_budget
        );
        assert!(
            full.metrics.comparisons_used <= full.metrics.comparisons_attempted,
            "case {} used more comparisons ({}) than it attempted ({})",
            case.name,
            full.metrics.comparisons_used,
            full.metrics.comparisons_attempted
        );
    }
}

/// Halving the budget must not overrun it. Cases without gate prewarm are
/// checked here to isolate the planner loop; the dedicated regression below
/// checks that gate prewarm also stops exactly at the same budget.
#[test]
fn comparisons_attempted_never_exceeds_half_budget_for_non_prewarm_cases() {
    for case in &synthetic_cases() {
        if case.prewarm_pairs_per_attr > 0 {
            continue;
        }
        let full_budget = case.comparison_budget.unwrap_or_else(|| {
            4 * case.attributes[0].scores.len().max(1) * case.attributes.len().max(1)
        });
        let mut half_case = case.clone();
        half_case.comparison_budget = Some((full_budget / 2).max(1));
        let half = run_synthetic_case_with_config(&half_case, ratio_cfg())
            .expect("half-budget ratio run should succeed");
        assert!(
            half.metrics.comparisons_attempted <= half_case.comparison_budget.unwrap(),
            "case {} (half budget) attempted {} comparisons against a budget of {}",
            case.name,
            half.metrics.comparisons_attempted,
            half_case.comparison_budget.unwrap()
        );
    }
}

/// Regression for a fixed bug in `run_synthetic_case_with_config`'s gate
/// prewarm: the old loop ran `prewarm_pairs_per_attr * n_attributes`
/// comparisons before the planner's budget check. A caller-supplied budget
/// smaller than that total was already overrun before planning began.
///
/// `gated_feasibility_30` has `prewarm_pairs_per_attr: 80` on one attribute.
/// Its default budget is 120, so the defect was silent at stock settings.
/// With budget 60, the prewarm phase must stop at 60 rather than attempting
/// all 80 comparisons.
///
/// Regression test: found by this suite and FIXED — the prewarm loop now
/// checks `comparisons_attempted >= comparison_budget` before every prewarm
/// comparison and stops when the budget is spent.
#[test]
fn prewarm_respects_comparison_budget() {
    let mut case = synthetic_cases()
        .into_iter()
        .find(|c| c.name == "gated_feasibility_30")
        .expect("gated_feasibility_30 case must exist in the built-in suite");
    assert!(
        case.prewarm_pairs_per_attr > 0,
        "this repro requires a case with nonzero gate-prewarm"
    );

    let small_budget = 60usize;
    assert!(
        case.prewarm_pairs_per_attr * case.attributes.len() > small_budget,
        "the repro's whole point is that prewarm alone exceeds the configured budget"
    );
    case.comparison_budget = Some(small_budget);

    let result = run_synthetic_case_with_config(&case, ratio_cfg())
        .expect("budget-limited prewarm should complete");

    assert!(
        result.metrics.comparisons_attempted <= small_budget,
        "comparisons_attempted ({}) should never exceed the configured budget ({}), \
         but the gate-prewarm phase ignores it entirely",
        result.metrics.comparisons_attempted,
        small_budget
    );
}

// =============================================================================
// Claim 4: trajectory monotonicity.
// =============================================================================

/// `estimate_topk_error` trajectories must be weakly improving in aggregate:
/// for every built-in case in ratio mode, the final recorded error must not
/// exceed the initial one. The path in between need not be monotonic (mid-run
/// bumps from newly-discovered structure are fine); only the net direction is
/// pinned.
#[test]
fn ratio_mode_trajectory_final_error_never_exceeds_initial_across_suite() {
    const EPS: f64 = 1e-9;
    let cases = synthetic_cases();
    assert!(!cases.is_empty());

    for case in &cases {
        let result =
            run_synthetic_case_with_config(case, ratio_cfg()).expect("ratio run should succeed");
        let traj = &result.error_trajectory;
        assert!(
            traj.len() >= 2,
            "case {} should record more than one trajectory point",
            case.name
        );
        let first = traj[0];
        let last = *traj.last().unwrap();
        assert!(
            last <= first + EPS,
            "case {}: final error {last} should not exceed initial error {first}; trajectory: {traj:?}",
            case.name
        );
    }
}

/// Beyond "not worse", the trajectory should show a real net reduction on
/// average across the suite -- otherwise "weakly improving" would be
/// satisfied trivially by a flat line.
#[test]
fn ratio_mode_trajectory_shows_net_reduction_on_average_across_suite() {
    let cases = synthetic_cases();
    let mut reductions = Vec::with_capacity(cases.len());

    for case in &cases {
        let result =
            run_synthetic_case_with_config(case, ratio_cfg()).expect("ratio run should succeed");
        let traj = &result.error_trajectory;
        let first = traj[0];
        let last = *traj.last().unwrap();
        reductions.push((first - last) / first.max(1e-9));
    }

    let mean_reduction = mean(&reductions);
    assert!(
        mean_reduction >= 0.05,
        "mean fractional error reduction across the suite should be a real effect, got {mean_reduction}"
    );
    assert!(
        mean_reduction <= 0.95,
        "mean fractional error reduction should stay below near-total collapse, got {mean_reduction}"
    );
}

/// The monotonicity claim should survive adversarial noise and outliers, not
/// just the checked-in cases' fixed seeds. Sweep 40 seeds of the adversarial
/// regime used in the ratio-vs-ordinal counter-evidence test above and
/// require zero violations.
#[test]
fn ratio_mode_trajectory_monotonicity_holds_under_adversarial_noise_ensemble() {
    const EPS: f64 = 1e-9;
    let seeds: Vec<u64> = (7_000..7_040).collect();

    let mut violations = 0usize;
    let mut reductions = Vec::with_capacity(seeds.len());

    for &seed in &seeds {
        let case = adversarial_case(seed);
        let result =
            run_synthetic_case_with_config(&case, ratio_cfg()).expect("ratio run should succeed");
        let traj = &result.error_trajectory;
        assert!(traj.len() >= 2, "seed {seed} should record a trajectory");
        let first = traj[0];
        let last = *traj.last().unwrap();
        if last > first + EPS {
            violations += 1;
        }
        reductions.push((first - last) / first.max(1e-9));
    }

    assert_eq!(
        violations, 0,
        "no seeded replica of the adversarial regime should end worse than it started, saw {violations}/{}",
        seeds.len()
    );
    let mean_reduction = mean(&reductions);
    assert!(
        mean_reduction >= 0.1,
        "the adversarial ensemble should still show a real average reduction, got {mean_reduction}"
    );
}
