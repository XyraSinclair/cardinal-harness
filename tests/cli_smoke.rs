use std::collections::HashMap;
use std::process::Command;

use cardinal_harness::rerank::{
    AttributeScoreSummary, MultiRerankAttributeSpec, MultiRerankEntity, MultiRerankEntityResult,
    MultiRerankMeta, MultiRerankRequest, MultiRerankResponse, MultiRerankTopKSpec,
    RerankStopReason,
};
use tempfile::tempdir;

#[derive(Debug, serde::Deserialize)]
struct EvalMetrics {
    kendall_tau: f64,
    spearman_rho: f64,
    topk_precision: f64,
    topk_recall: f64,
    comparisons_attempted: usize,
    comparisons_used: usize,
    comparisons_refused: usize,
    stop_reason: String,
    latency_ms: u128,
}

#[derive(Debug, serde::Deserialize)]
struct EvalResult {
    case_name: String,
    metrics: EvalMetrics,
    error_trajectory: Vec<f64>,
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn run_cli_eval(case: &str) -> EvalResult {
    let dir = tempdir().unwrap();
    let out_path = dir.path().join("eval.jsonl");

    let status = Command::new(env!("CARGO_BIN_EXE_cardinal"))
        .args(["eval", "--case", case])
        .arg("--out")
        .arg(&out_path)
        .status()
        .unwrap();
    assert!(status.success());

    let raw = std::fs::read_to_string(&out_path).unwrap();
    let first_line = raw.lines().next().unwrap();
    serde_json::from_str(first_line).unwrap()
}

#[test]
fn cli_eval_smoke_and_determinism() {
    let a = run_cli_eval("clean_ordering_10");
    let b = run_cli_eval("clean_ordering_10");

    assert_eq!(a.case_name, "clean_ordering_10");
    assert_eq!(b.case_name, "clean_ordering_10");

    // Smoke checks (quality).
    assert!(a.metrics.kendall_tau >= 0.99);
    assert!(a.metrics.spearman_rho >= 0.99);
    assert!(a.metrics.topk_precision >= 0.99);
    assert!(a.metrics.topk_recall >= 0.99);
    assert_eq!(a.metrics.comparisons_refused, 0);
    assert!(a.metrics.comparisons_attempted > 0);
    assert!(a.metrics.comparisons_used > 0);
    assert!(!a.metrics.stop_reason.is_empty());

    // Determinism across separate processes: ignore latency, compare all else.
    assert!(approx_eq(
        a.metrics.kendall_tau,
        b.metrics.kendall_tau,
        1e-12
    ));
    assert!(approx_eq(
        a.metrics.spearman_rho,
        b.metrics.spearman_rho,
        1e-12
    ));
    assert!(approx_eq(
        a.metrics.topk_precision,
        b.metrics.topk_precision,
        1e-12
    ));
    assert!(approx_eq(
        a.metrics.topk_recall,
        b.metrics.topk_recall,
        1e-12
    ));
    assert_eq!(
        a.metrics.comparisons_attempted,
        b.metrics.comparisons_attempted
    );
    assert_eq!(a.metrics.comparisons_used, b.metrics.comparisons_used);
    assert_eq!(a.metrics.comparisons_refused, b.metrics.comparisons_refused);
    assert_eq!(a.metrics.stop_reason, b.metrics.stop_reason);
    assert_ne!(a.metrics.latency_ms, 0);
    assert_ne!(b.metrics.latency_ms, 0);

    assert_eq!(a.error_trajectory.len(), b.error_trajectory.len());
    for (x, y) in a.error_trajectory.iter().zip(b.error_trajectory.iter()) {
        assert!(approx_eq(*x, *y, 1e-12));
    }
}

#[test]
fn cli_report_json_smoke() {
    let dir = tempdir().unwrap();

    let request_path = dir.path().join("request.json");
    let response_path = dir.path().join("response.json");
    let out_path = dir.path().join("report.json");

    let req = MultiRerankRequest {
        entities: vec![
            MultiRerankEntity {
                id: "a".into(),
                text: "Entity A text".into(),
            },
            MultiRerankEntity {
                id: "b".into(),
                text: "Entity B text".into(),
            },
        ],
        attributes: vec![MultiRerankAttributeSpec {
            id: "clarity".into(),
            prompt: "clarity of explanation".into(),
            prompt_template_slug: Some("canonical_v2".into()),
            weight: 1.0,
        }],
        topk: MultiRerankTopKSpec {
            k: 1,
            weight_exponent: 1.0,
            tolerated_error: 0.1,
            band_size: 5,
            effective_resistance_max_active: 64,
            stop_sigma_inflate: 1.25,
            stop_min_consecutive: 2,
        },
        gates: vec![],
        comparison_budget: Some(1),
        latency_budget_ms: None,
        model: Some("openai/gpt-5-mini".into()),
        rater_id: None,
        comparison_concurrency: Some(1),
        max_pair_repeats: Some(1),
    };

    let mut a_scores = HashMap::new();
    a_scores.insert(
        "clarity".to_string(),
        AttributeScoreSummary {
            latent_mean: 1.0,
            latent_std: 0.1,
            z_score: 0.5,
            min_normalized: 2.0,
            percentile: 0.75,
        },
    );
    let mut b_scores = HashMap::new();
    b_scores.insert(
        "clarity".to_string(),
        AttributeScoreSummary {
            latent_mean: 0.0,
            latent_std: 0.2,
            z_score: -0.5,
            min_normalized: 1.0,
            percentile: 0.25,
        },
    );

    let resp = MultiRerankResponse {
        entities: vec![
            MultiRerankEntityResult {
                id: "a".into(),
                rank: Some(1),
                feasible: true,
                u_mean: 1.0,
                u_std: 0.1,
                p_flip: 0.01,
                attribute_scores: a_scores,
            },
            MultiRerankEntityResult {
                id: "b".into(),
                rank: Some(2),
                feasible: true,
                u_mean: 0.0,
                u_std: 0.2,
                p_flip: 0.02,
                attribute_scores: b_scores,
            },
        ],
        meta: MultiRerankMeta {
            global_topk_error: 0.0,
            tolerated_error: req.topk.tolerated_error,
            k: req.topk.k,
            band_size: req.topk.band_size,
            comparisons_attempted: 1,
            comparisons_used: 1,
            comparisons_refused: 0,
            comparisons_cached: 0,
            comparison_budget: 1,
            latency_ms: 1,
            model_used: "openai/gpt-5-mini".into(),
            rater_id_used: "openai/gpt-5-mini".into(),
            provider_input_tokens: 0,
            provider_output_tokens: 0,
            provider_cost_nanodollars: 0,
            stop_reason: RerankStopReason::BudgetExhausted,
        },
    };

    std::fs::write(&request_path, serde_json::to_string_pretty(&req).unwrap()).unwrap();
    std::fs::write(&response_path, serde_json::to_string_pretty(&resp).unwrap()).unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_cardinal"))
        .args(["report", "--format", "json"])
        .arg("--request")
        .arg(&request_path)
        .arg("--response")
        .arg(&response_path)
        .arg("--out")
        .arg(&out_path)
        .status()
        .unwrap();
    assert!(status.success());

    let raw = std::fs::read_to_string(&out_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();

    assert!(
        v.get("request_hash")
            .and_then(|h| h.as_str())
            .unwrap()
            .len()
            >= 16
    );
    assert_eq!(
        v.pointer("/summary/stop_reason")
            .and_then(|s| s.as_str())
            .unwrap(),
        "budget_exhausted"
    );
    assert_eq!(
        v.pointer("/attributes/0/id")
            .and_then(|s| s.as_str())
            .unwrap(),
        "clarity"
    );
    assert_eq!(
        v.pointer("/top_entities/0/id")
            .and_then(|s| s.as_str())
            .unwrap(),
        "a"
    );
}
