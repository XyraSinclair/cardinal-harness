use cardinal_harness::rerank::evaluation::{run_synthetic_suite, synthetic_cases};

fn assert_prob(x: f64) {
    assert!(
        x.is_finite() && (0.0..=1.0).contains(&x),
        "expected probability in [0,1], got {x}"
    );
}

#[test]
fn synthetic_suite_filter_selects_exact_name() {
    let all = synthetic_cases();
    assert!(all.iter().any(|c| c.name == "clean_ordering_10"));

    let selected = run_synthetic_suite(Some("clean_ordering_10"));
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].case_name, "clean_ordering_10");
}

#[test]
fn clean_ordering_case_is_perfect_topk() {
    let result = run_synthetic_suite(Some("clean_ordering_10"));
    let metrics = &result[0].metrics;

    assert!(metrics.kendall_tau >= 0.99);
    assert!(metrics.spearman_rho >= 0.99);
    assert_prob(metrics.topk_precision);
    assert_prob(metrics.topk_recall);
    assert!(metrics.topk_precision >= 0.99);
    assert!(metrics.topk_recall >= 0.99);

    assert_eq!(metrics.comparisons_refused, 0);
    assert!(metrics.comparisons_attempted > 0);
    assert!(metrics.comparisons_used > 0);
}

#[test]
fn gated_case_produces_reasonable_gate_metrics() {
    let result = run_synthetic_suite(Some("gated_feasibility_30"));
    let metrics = &result[0].metrics;

    assert_prob(metrics.coverage_95ci);
    assert_prob(metrics.topk_precision);
    assert_prob(metrics.topk_recall);

    let gate_precision = metrics
        .gate_precision
        .expect("gate_precision should be present");
    let gate_recall = metrics.gate_recall.expect("gate_recall should be present");
    assert_prob(gate_precision);
    assert_prob(gate_recall);

    // Regression guards: gating should work on this synthetic.
    assert!(gate_precision >= 0.8);
    assert!(gate_recall >= 0.8);
}
