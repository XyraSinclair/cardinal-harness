use std::collections::HashMap;

use cardinal_harness::rating_engine::{
    plan_edges_for_rater, AttributeParams, Config, EngineSpec, Observation, PlannerMode,
    RaterParams, RatingEngine,
};

fn sim_raters() -> HashMap<String, RaterParams> {
    let mut raters = HashMap::new();
    raters.insert("sim".to_string(), RaterParams::default());
    raters
}

#[test]
fn rating_engine_solve_orders_simple_chain() {
    let mut engine = RatingEngine::new(
        3,
        AttributeParams::default(),
        sim_raters(),
        Some(Config::default()),
    )
    .unwrap();

    // 0 > 1 > 2
    let obs = vec![
        Observation::new(0, 1, 3.0, 1.0, "sim", 1.0),
        Observation::new(1, 2, 3.0, 1.0, "sim", 1.0),
    ];
    engine.ingest(&obs);
    let summary = engine.solve();

    assert_eq!(summary.scores.len(), 3);
    assert_eq!(summary.diag_cov.len(), 3);
    assert!(summary.scores.iter().all(|v| v.is_finite()));
    assert!(summary.diag_cov.iter().all(|v| v.is_finite() && *v >= 0.0));

    assert!(summary.scores[0] > summary.scores[1]);
    assert!(summary.scores[1] > summary.scores[2]);

    assert!(engine.same_component(0, 2));
    assert!(engine.diff_var_for(0, 2).unwrap() > 0.0);
}

#[test]
fn rating_engine_diff_var_for_cross_component_equals_diag_sum() {
    let mut engine = RatingEngine::new(4, AttributeParams::default(), sim_raters(), None).unwrap();

    // Only connect {0,1}; {2} and {3} remain disconnected.
    engine.ingest(&[Observation::new(0, 1, 2.0, 0.9, "sim", 1.0)]);
    let _ = engine.solve();

    assert!(engine.same_component(0, 1));
    assert!(!engine.same_component(0, 2));

    let diag = engine.diag_cov().unwrap();
    let dv = engine.diff_var_for(0, 2).unwrap();
    assert!((dv - (diag[0] + diag[2])).abs() <= 1e-9);
}

#[test]
fn plan_edges_requires_solve_state() {
    let engine = RatingEngine::new(3, AttributeParams::default(), sim_raters(), None).unwrap();
    let candidates = vec![(0, 1), (1, 2)];
    assert!(plan_edges_for_rater(&engine, &candidates, "sim", PlannerMode::Hybrid, false).is_err());
}

#[test]
fn plan_edges_returns_sorted_proposals_and_skips_invalid_candidates() {
    let mut engine = RatingEngine::new(4, AttributeParams::default(), sim_raters(), None).unwrap();
    engine.ingest(&[
        Observation::new(0, 1, 2.0, 0.9, "sim", 1.0),
        Observation::new(1, 2, 2.0, 0.9, "sim", 1.0),
        Observation::new(2, 3, 2.0, 0.9, "sim", 1.0),
    ]);
    let _ = engine.solve();

    let candidates = vec![
        (0, 0),  // invalid
        (10, 1), // invalid
        (0, 1),
        (3, 2),
        (1, 3),
    ];
    let proposals =
        plan_edges_for_rater(&engine, &candidates, "sim", PlannerMode::Hybrid, false).unwrap();

    // Only the 3 valid candidates should survive.
    assert_eq!(proposals.len(), 3);
    assert!(proposals.iter().all(|p| p.i != p.j && p.i < 4 && p.j < 4));

    for w in proposals.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "proposals not sorted: {:?} < {:?}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn rating_engine_cycle_dim_positive_for_simple_triangle() {
    let mut engine = RatingEngine::new(3, AttributeParams::default(), sim_raters(), None).unwrap();

    // A simple inconsistent cycle: 0 > 1 > 2 > 0.
    engine.ingest(&[
        Observation::new(0, 1, 2.0, 1.0, "sim", 1.0),
        Observation::new(1, 2, 2.0, 1.0, "sim", 1.0),
        Observation::new(2, 0, 2.0, 1.0, "sim", 1.0),
    ]);
    let summary = engine.solve();

    assert_eq!(summary.components, 1);
    assert!(summary.cycle_dim >= 1);
    assert!(summary.scores.iter().all(|v| v.is_finite()));
    assert!(summary.diag_cov.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn explicit_precision_dominates_stated_confidence_in_conflict() {
    // Two conflicting observations on the same pair: a stated-confidence
    // claim that i wins 4x, and a PMF-derived claim (tight variance,
    // precision 1000) that i LOSES 4x. The measured claim must win the
    // fused edge; with symmetric stated confidences instead, the claims
    // cancel to a near-tie. This pins the precision channel as a real,
    // separate weighting path — not a relabeled g(c).
    let mut engine = RatingEngine::new(2, AttributeParams::default(), sim_raters(), None).unwrap();
    engine.ingest(&[
        Observation::new(0, 1, 4.0, 0.5, "sim", 1.0),
        Observation::from_log_ratio_moments(0, 1, 0.25f64.ln(), 0.001, "sim", 1.0),
    ]);
    let scores = engine.solve().scores;
    assert!(
        scores[1] > scores[0] + 0.5,
        "tight PMF evidence must dominate: {scores:?}"
    );

    let mut tie_engine =
        RatingEngine::new(2, AttributeParams::default(), sim_raters(), None).unwrap();
    tie_engine.ingest(&[
        Observation::new(0, 1, 4.0, 0.5, "sim", 1.0),
        Observation::new(0, 1, 0.25, 0.5, "sim", 1.0),
    ]);
    let tie = tie_engine.solve().scores;
    assert!(
        (tie[0] - tie[1]).abs() < 1e-9,
        "equal stated confidence cancels exactly: {tie:?}"
    );
}

#[test]
fn moments_constructor_sign_semantics() {
    // Negative mean log-ratio: j has more; the solve must respect it.
    let mut engine = RatingEngine::new(2, AttributeParams::default(), sim_raters(), None).unwrap();
    engine.ingest(&[Observation::from_log_ratio_moments(
        0, 1, -1.0, 0.05, "sim", 1.0,
    )]);
    let scores = engine.solve().scores;
    assert!(
        scores[1] > scores[0],
        "negative mean means j wins: {scores:?}"
    );
}

#[test]
fn degenerate_precision_is_skipped_not_poisonous() {
    let mut engine = RatingEngine::new(2, AttributeParams::default(), sim_raters(), None).unwrap();
    let mut bad = Observation::from_log_ratio_moments(0, 1, 1.0, 0.05, "sim", 1.0);
    bad.precision = Some(f64::NAN);
    engine.ingest(&[bad]);
    let scores = engine.solve().scores;
    assert!(
        (scores[0] - scores[1]).abs() < 1e-12,
        "NaN precision must contribute nothing: {scores:?}"
    );
}

fn identity_spec() -> EngineSpec {
    EngineSpec {
        n: 7,
        attribute: AttributeParams { temperature: 1.25 },
        raters: vec![
            (
                "zeta".into(),
                RaterParams {
                    beta: 0.8,
                    cost_per_edge: 1.75,
                    default_confidence: 0.61,
                },
            ),
            (
                "alpha".into(),
                RaterParams {
                    beta: 1.3,
                    cost_per_edge: 2.25,
                    default_confidence: 0.84,
                },
            ),
        ],
        config: Config {
            eps_confidence: 0.02,
            gamma_confidence: 1.7,
            huber_k: 1.2,
            irls_max_iters: 9,
            irls_tol: 1e-7,
            ridge_lambda: 1e-8,
            tiny: 1e-15,
            max_log_ratio: 8.0,
            hutch_probes: 7,
            rank_weight_exponent: 1.4,
            rank_band_window: 4,
            small_gap_threshold: 0.3,
            max_rank_pairs: Some(1_000),
            top_k: Some(2),
            tail_weight: 0.2,
            lambda_risk: 0.6,
            rng_seed: 42,
        },
    }
}

fn assert_spec_id_changes(base: &EngineSpec, mutate: impl FnOnce(&mut EngineSpec)) {
    let mut changed = base.clone();
    mutate(&mut changed);
    assert_ne!(base.id(), changed.id());
}

fn next_f64(value: f64) -> f64 {
    f64::from_bits(value.to_bits() + 1)
}

#[test]
fn engine_spec_identity_is_order_canonical_and_covers_every_policy_field() {
    let base = identity_spec();
    let mut reordered = base.clone();
    reordered.raters.reverse();
    assert_eq!(base.canonical_bytes(), reordered.canonical_bytes());
    assert_eq!(base.id(), reordered.id());

    assert_spec_id_changes(&base, |spec| spec.n += 1);
    assert_spec_id_changes(&base, |spec| {
        spec.attribute.temperature = next_f64(spec.attribute.temperature)
    });
    assert_spec_id_changes(&base, |spec| spec.raters[0].0.push_str("-changed"));
    assert_spec_id_changes(&base, |spec| {
        spec.raters[0].1.beta = next_f64(spec.raters[0].1.beta)
    });
    assert_spec_id_changes(&base, |spec| {
        spec.raters[0].1.cost_per_edge = next_f64(spec.raters[0].1.cost_per_edge)
    });
    assert_spec_id_changes(&base, |spec| {
        spec.raters[0].1.default_confidence = next_f64(spec.raters[0].1.default_confidence)
    });
    assert_spec_id_changes(&base, |spec| {
        spec.config.eps_confidence = next_f64(spec.config.eps_confidence)
    });
    assert_spec_id_changes(&base, |spec| {
        spec.config.gamma_confidence = next_f64(spec.config.gamma_confidence)
    });
    assert_spec_id_changes(&base, |spec| {
        spec.config.huber_k = next_f64(spec.config.huber_k)
    });
    assert_spec_id_changes(&base, |spec| spec.config.irls_max_iters += 1);
    assert_spec_id_changes(&base, |spec| {
        spec.config.irls_tol = next_f64(spec.config.irls_tol)
    });
    assert_spec_id_changes(&base, |spec| {
        spec.config.ridge_lambda = next_f64(spec.config.ridge_lambda)
    });
    assert_spec_id_changes(&base, |spec| spec.config.tiny = next_f64(spec.config.tiny));
    assert_spec_id_changes(&base, |spec| {
        spec.config.max_log_ratio = next_f64(spec.config.max_log_ratio)
    });
    assert_spec_id_changes(&base, |spec| spec.config.hutch_probes += 1);
    assert_spec_id_changes(&base, |spec| {
        spec.config.rank_weight_exponent = next_f64(spec.config.rank_weight_exponent)
    });
    assert_spec_id_changes(&base, |spec| spec.config.rank_band_window += 1);
    assert_spec_id_changes(&base, |spec| {
        spec.config.small_gap_threshold = next_f64(spec.config.small_gap_threshold)
    });
    assert_spec_id_changes(&base, |spec| {
        spec.config.max_rank_pairs = spec.config.max_rank_pairs.map(|value| value + 1)
    });
    assert_spec_id_changes(&base, |spec| {
        spec.config.top_k = spec.config.top_k.map(|value| value + 1)
    });
    assert_spec_id_changes(&base, |spec| {
        spec.config.tail_weight = next_f64(spec.config.tail_weight)
    });
    assert_spec_id_changes(&base, |spec| {
        spec.config.lambda_risk = next_f64(spec.config.lambda_risk)
    });
    assert_spec_id_changes(&base, |spec| spec.config.rng_seed += 1);
}
