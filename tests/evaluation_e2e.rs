use cardinal_harness::rerank::evaluation::{
    run_evaluation_comparison_summary, run_evaluation_comparison_summary_with_config,
    run_likert_baseline_suite, run_synthetic_suite, run_synthetic_suite_with_config,
    synthetic_cases, ComparisonOutcome, LikertEvalConfig, PairwiseEvalConfig,
    SyntheticPairwiseMode,
};

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

    let selected =
        run_synthetic_suite(Some("clean_ordering_10")).expect("synthetic suite should run");
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].case_name, "clean_ordering_10");
}

#[test]
fn clean_ordering_case_is_perfect_topk() {
    let result =
        run_synthetic_suite(Some("clean_ordering_10")).expect("synthetic suite should run");
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
    let result =
        run_synthetic_suite(Some("gated_feasibility_30")).expect("synthetic suite should run");
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

#[test]
fn scale_compression_case_exposes_likert_quantization_loss() {
    let cardinal = run_synthetic_suite(Some("scale_compression_40"))
        .expect("cardinal synthetic suite should run");
    let likert =
        run_likert_baseline_suite(Some("scale_compression_40"), LikertEvalConfig::default())
            .expect("likert baseline suite should run");

    assert_eq!(cardinal.len(), 1);
    assert_eq!(likert.len(), 1);

    let cardinal_metrics = &cardinal[0].metrics;
    let likert_metrics = &likert[0].metrics;

    assert_eq!(cardinal_metrics.comparisons_refused, 0);
    assert_eq!(likert_metrics.ratings_refused, 0);
    assert!(
        cardinal_metrics.topk_precision >= 0.99,
        "cardinal should recover the compressed top-k, got precision {}",
        cardinal_metrics.topk_precision
    );
    assert!(
        likert_metrics.topk_precision <= 0.4,
        "10-level Likert should lose the compressed non-outlier ordering, got precision {}",
        likert_metrics.topk_precision
    );
    assert!(
        cardinal_metrics.kendall_tau_all > likert_metrics.kendall_tau_all + 0.5,
        "cardinal tau {} should materially exceed Likert tau {}",
        cardinal_metrics.kendall_tau_all,
        likert_metrics.kendall_tau_all
    );
}

#[test]
fn ordinal_pairwise_mode_exposes_scale_loss_control() {
    let ratio = run_synthetic_suite(Some("scale_compression_40"))
        .expect("ratio cardinal synthetic suite should run");
    let ordinal = run_synthetic_suite_with_config(
        Some("scale_compression_40"),
        PairwiseEvalConfig {
            mode: SyntheticPairwiseMode::Ordinal,
        },
    )
    .expect("ordinal cardinal synthetic suite should run");

    assert_eq!(ratio.len(), 1);
    assert_eq!(ordinal.len(), 1);
    assert_eq!(ordinal[0].pairwise_mode, SyntheticPairwiseMode::Ordinal);

    let ratio_precision = ratio[0].metrics.topk_precision;
    let ordinal_precision = ordinal[0].metrics.topk_precision;
    assert!(
        ordinal_precision < ratio_precision,
        "ordinal pairwise mode should expose lost ratio magnitude on the compressed frontier: ordinal {}, ratio {}",
        ordinal_precision,
        ratio_precision
    );
    assert!(
        ordinal_precision <= 0.8,
        "ordinal pairwise mode should remain a visible control, got precision {}",
        ordinal_precision
    );
}

#[test]
fn comparison_summary_makes_cardinal_minus_likert_receipts_explicit() {
    let summary = run_evaluation_comparison_summary(None, LikertEvalConfig::default())
        .expect("comparison summary should run");

    assert_eq!(
        summary.metric_names,
        [
            "topk_precision",
            "topk_recall",
            "kendall_tau_b",
            "coverage_95ci",
            "comparisons_used",
        ]
    );
    assert_eq!(summary.cases.len(), synthetic_cases().len());
    assert_eq!(summary.pairwise_config.mode, SyntheticPairwiseMode::Ratio);

    let mut observed_counts = (0usize, 0usize, 0usize);
    for case in &summary.cases {
        let deltas = &case.cardinal_minus_likert;

        assert_eq!(
            deltas.topk_precision.delta,
            deltas.topk_precision.cardinal - deltas.topk_precision.likert
        );
        assert_eq!(
            deltas.topk_recall.delta,
            deltas.topk_recall.cardinal - deltas.topk_recall.likert
        );
        assert_eq!(
            deltas.kendall_tau_b.delta,
            deltas.kendall_tau_b.cardinal - deltas.kendall_tau_b.likert
        );
        assert_eq!(
            deltas.coverage_95ci.delta,
            deltas.coverage_95ci.cardinal - deltas.coverage_95ci.likert
        );
        assert_eq!(
            deltas.comparisons_used.delta,
            deltas.comparisons_used.cardinal - deltas.comparisons_used.likert
        );

        for outcome in [
            deltas.topk_precision.outcome,
            deltas.topk_recall.outcome,
            deltas.kendall_tau_b.outcome,
            deltas.coverage_95ci.outcome,
            deltas.comparisons_used.outcome,
        ] {
            match outcome {
                ComparisonOutcome::CardinalWin => observed_counts.0 += 1,
                ComparisonOutcome::LikertWin => observed_counts.1 += 1,
                ComparisonOutcome::Tie => observed_counts.2 += 1,
            }
        }

        assert_eq!(
            case.win_loss_tie.cardinal_wins
                + case.win_loss_tie.likert_wins
                + case.win_loss_tie.ties,
            summary.metric_names.len()
        );
    }

    assert_eq!(
        summary.aggregate_win_loss_tie.cardinal_wins,
        observed_counts.0
    );
    assert_eq!(
        summary.aggregate_win_loss_tie.likert_wins,
        observed_counts.1
    );
    assert_eq!(summary.aggregate_win_loss_tie.ties, observed_counts.2);
}

#[test]
fn comparison_summary_accepts_ordinal_pairwise_config() {
    let summary = run_evaluation_comparison_summary_with_config(
        Some("scale_compression_40"),
        PairwiseEvalConfig {
            mode: SyntheticPairwiseMode::Ordinal,
        },
        LikertEvalConfig::default(),
    )
    .expect("comparison summary should run");

    assert_eq!(summary.pairwise_config.mode, SyntheticPairwiseMode::Ordinal);
    assert_eq!(summary.cases.len(), 1);
    assert_eq!(summary.cases[0].case_name, "scale_compression_40");
}
