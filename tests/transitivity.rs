//! The stochastic-transitivity hierarchy, pinned — including the diagnostic
//! this instrument exists for: a judge whose MEAN log-ratios telescope
//! exactly (zero curl, invisible to every Hodge diagnostic) while its choice
//! probabilities violate strong stochastic transitivity.

use std::collections::HashMap;

use cardinal_harness::rating_engine::{
    AttributeParams, Config, Observation, RaterParams, RatingEngine,
};
use cardinal_harness::repeat_pooling::RepeatDraws;
use cardinal_harness::rerank::stochastic_transitivity;

fn draws(i: usize, j: usize, d: Vec<f64>) -> RepeatDraws {
    RepeatDraws { i, j, draws: d }
}

#[test]
fn a_coherent_random_utility_judge_satisfies_sst() {
    // Latent gaps large vs noise: p's near 1, ordered — no violations.
    let mk = |gap: f64| -> Vec<f64> {
        (0..20)
            .map(|t| gap + if t % 5 == 0 { -0.1 } else { 0.05 })
            .collect()
    };
    let report = stochastic_transitivity(&[
        draws(0, 1, mk(0.5)),
        draws(1, 2, mk(0.5)),
        draws(0, 2, mk(1.0)),
    ]);
    assert_eq!(report.testable_triads, 1);
    assert_eq!(report.sst_violations, 0, "{report:?}");
    assert_eq!(report.wst_violations, 0);
}

#[test]
fn zero_curl_judge_with_sst_violation_is_caught_here_and_only_here() {
    // The construction: pair (0,2)'s draws mix a 55% majority of small
    // positives with large-magnitude minorities tuned so the MEAN equals
    // exactly the telescoped sum of the other two means — zero cycle
    // residual — while P(0>2) = 0.55 < max(0.75, 0.75): an SST (and MST)
    // violation with no curl signature at all.
    let ab: Vec<f64> = (0..20)
        .map(|t| if t < 15 { 0.4 } else { -0.2 }) // mean 0.25, p = .75
        .collect();
    let bc = ab.clone();
    let mean_ab = ab.iter().sum::<f64>() / 20.0;
    // (0,2): 11 wins of +w, 9 losses of −l, mean forced to 2·mean_ab.
    let target = 2.0 * mean_ab;
    let w = 1.0f64;
    let l = (11.0 * w - 20.0 * target) / 9.0;
    let ac: Vec<f64> = (0..20).map(|t| if t < 11 { w } else { -l }).collect();
    assert!((ac.iter().sum::<f64>() / 20.0 - target).abs() < 1e-12);

    let report = stochastic_transitivity(&[
        draws(0, 1, ab.clone()),
        draws(1, 2, bc.clone()),
        draws(0, 2, ac.clone()),
    ]);
    assert_eq!(report.testable_triads, 1);
    let t = &report.triads[0];
    assert!(!t.wst_violated, "p_ac = .55 ≥ .5: {t:?}");
    assert!(t.mst_violated && t.sst_violated, "{t:?}");

    // And the Hodge side is BLIND to it: means telescope exactly, so the
    // solver's cyclic residual vanishes.
    let mut raters = HashMap::new();
    raters.insert("sim".to_string(), RaterParams::default());
    let mut engine = RatingEngine::new(
        3,
        AttributeParams::default(),
        raters,
        Some(Config::default()),
    )
    .unwrap();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    engine.ingest(&[
        Observation::from_log_ratio_moments(0, 1, mean(&ab), 1.0, "sim", 1.0),
        Observation::from_log_ratio_moments(1, 2, mean(&bc), 1.0, "sim", 1.0),
        Observation::from_log_ratio_moments(0, 2, mean(&ac), 1.0, "sim", 1.0),
    ]);
    let summary = engine.solve();
    assert!(
        summary.hcr < 1e-9,
        "zero curl by construction — the mean-level diagnostics see nothing: {}",
        summary.hcr
    );
}

#[test]
fn a_cyclic_tournament_is_exactly_a_wst_violation() {
    // p(0>1), p(1>2), p(2>0) all 0.75: rock-paper-scissors in
    // probability. The first implementation had an "unorientable"
    // branch here — dead code, because every 3-tournament has a
    // Hamiltonian path: the orientation exists and the reversed edge
    // fails WST. Cyclic majority ⟺ WST violation, pinned as such.
    // k = 60: at k = 20 a 0.25-deep violation across three binomial p̂'s
    // is only ~1.5 combined SE — the margin machinery correctly refused
    // to call it (the first version of this test asserted otherwise and
    // lost the argument to its own instrument).
    let win: Vec<f64> = (0..60).map(|t| if t < 45 { 0.3 } else { -0.3 }).collect();
    let report = stochastic_transitivity(&[
        draws(0, 1, win.clone()),
        draws(1, 2, win.clone()),
        draws(2, 0, win.clone()),
    ]);
    assert_eq!(report.testable_triads, 1);
    assert!(report.triads[0].cyclic, "{report:?}");
    assert_eq!(report.wst_violations, 1);
    assert_eq!(report.wst_violations_2se, 1, "{report:?}");
}

#[test]
fn noise_margins_separate_deep_violations_from_shallow_ones() {
    // A shallow SST violation (p_ac just under max premise) at k = 20 is
    // within noise; the report must say so via the 2-SE counters.
    let p75: Vec<f64> = (0..20).map(|t| if t < 15 { 0.3 } else { -0.3 }).collect();
    let p70: Vec<f64> = (0..20).map(|t| if t < 14 { 0.3 } else { -0.3 }).collect();
    let report = stochastic_transitivity(&[
        draws(0, 1, p75.clone()),
        draws(1, 2, p75.clone()),
        draws(0, 2, p70),
    ]);
    assert_eq!(report.sst_violations, 1, "raw violation exists: {report:?}");
    assert_eq!(
        report.sst_violations_2se, 0,
        "but 0.05 below threshold at k=20 is sampling noise, and the \
         margin counter must not cry wolf: {report:?}"
    );
}
