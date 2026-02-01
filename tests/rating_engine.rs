use std::collections::HashMap;

use cardinal_harness::rating_engine::{
    plan_edges_for_rater, AttributeParams, Config, Observation, PlannerMode, RaterParams,
    RatingEngine,
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
