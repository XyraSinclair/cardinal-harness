use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use cardinal_harness::rerank::{
    AttributeScoreSummary, MultiRerankAttributeSpec, MultiRerankEntity, MultiRerankEntityResult,
    MultiRerankMeta, MultiRerankRequest, MultiRerankResponse, MultiRerankTopKSpec,
    RerankStopReason,
};
use tempfile::tempdir;

#[allow(dead_code)]
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
    pairwise_mode: String,
    metrics: EvalMetrics,
    error_trajectory: Vec<f64>,
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn cardinal_bin() -> PathBuf {
    let cargo_bin = option_env!("CARGO_BIN_EXE_cardinal").filter(|path| !path.is_empty());
    if let Some(path) = cargo_bin {
        let path = PathBuf::from(path);
        if path.exists() {
            return path;
        }
    }

    let test_exe =
        std::env::current_exe().expect("failed to resolve current integration test binary path");
    let deps_dir = test_exe.parent().unwrap_or_else(|| {
        panic!(
            "integration test binary path has no parent directory: {}",
            test_exe.display()
        )
    });
    let target_dir = deps_dir.parent().unwrap_or_else(|| {
        panic!(
            "integration test binary parent has no target directory: {}",
            deps_dir.display()
        )
    });
    let fallback = target_dir.join(format!("cardinal{}", std::env::consts::EXE_SUFFIX));

    if fallback.exists() {
        return fallback;
    }

    panic!(
        "failed to locate compiled cardinal binary; CARGO_BIN_EXE_cardinal={:?}; \
         integration test binary={}; fallback path={}. Run `cargo test --test cli_smoke` \
         so Cargo builds the cardinal binary before the smoke tests run",
        cargo_bin,
        test_exe.display(),
        fallback.display()
    );
}

fn run_cli_eval_with_mode(case: &str, mode: &str) -> EvalResult {
    let dir = tempdir().unwrap();
    let out_path = dir.path().join("eval.jsonl");

    let bin = cardinal_bin();
    let status = Command::new(&bin)
        .args(["eval", "--case", case, "--mode", mode])
        .arg("--out")
        .arg(&out_path)
        .status()
        .unwrap_or_else(|err| panic!("failed to run cardinal eval at {}: {err}", bin.display()));
    assert!(status.success(), "cardinal eval exited with {status}");

    let raw = std::fs::read_to_string(&out_path).unwrap();
    let first_line = raw.lines().next().unwrap();
    serde_json::from_str(first_line).unwrap()
}

fn run_cli_eval(case: &str) -> EvalResult {
    run_cli_eval_with_mode(case, "ratio")
}

#[derive(Debug, serde::Deserialize)]
struct LikertEvalMetrics {
    kendall_tau: f64,
    spearman_rho: f64,
    topk_precision: f64,
    topk_recall: f64,
    ratings_attempted: usize,
    ratings_used: usize,
    ratings_refused: usize,
}

#[derive(Debug, serde::Deserialize)]
struct LikertEvalResult {
    case_name: String,
    metrics: LikertEvalMetrics,
    error_trajectory: Vec<f64>,
}

fn run_cli_eval_likert(case: &str) -> LikertEvalResult {
    let dir = tempdir().unwrap();
    let out_path = dir.path().join("eval_likert.jsonl");

    let bin = cardinal_bin();
    let status = Command::new(&bin)
        .args(["eval-likert", "--case", case])
        .arg("--out")
        .arg(&out_path)
        .status()
        .unwrap_or_else(|err| {
            panic!(
                "failed to run cardinal eval-likert at {}: {err}",
                bin.display()
            )
        });
    assert!(
        status.success(),
        "cardinal eval-likert exited with {status}"
    );

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
    assert_eq!(a.pairwise_mode, "ratio");
    assert_eq!(b.pairwise_mode, "ratio");

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
    // Latency is environment-dependent and can legitimately be 0 in fast/synthetic runs.

    assert_eq!(a.error_trajectory.len(), b.error_trajectory.len());
    for (x, y) in a.error_trajectory.iter().zip(b.error_trajectory.iter()) {
        assert!(approx_eq(*x, *y, 1e-12));
    }
}

#[test]
fn cli_eval_accepts_ordinal_pairwise_mode() {
    let result = run_cli_eval_with_mode("scale_compression_40", "ordinal");

    assert_eq!(result.case_name, "scale_compression_40");
    assert_eq!(result.pairwise_mode, "ordinal");
    assert_eq!(result.metrics.comparisons_refused, 0);
    assert!(result.metrics.comparisons_attempted > 0);
    assert!(result.metrics.comparisons_used > 0);
}

#[test]
fn cli_eval_likert_smoke_and_determinism() {
    let a = run_cli_eval_likert("clean_ordering_10");
    let b = run_cli_eval_likert("clean_ordering_10");

    assert_eq!(a.case_name, "clean_ordering_10");
    assert_eq!(b.case_name, "clean_ordering_10");

    assert!(a.metrics.kendall_tau >= 0.99);
    assert!(a.metrics.spearman_rho >= 0.99);
    assert!(a.metrics.topk_precision >= 0.99);
    assert!(a.metrics.topk_recall >= 0.99);

    assert!(a.metrics.ratings_attempted > 0);
    assert!(a.metrics.ratings_used > 0);
    assert!(a.metrics.ratings_refused <= a.metrics.ratings_attempted);

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
    assert_eq!(a.metrics.ratings_attempted, b.metrics.ratings_attempted);
    assert_eq!(a.metrics.ratings_used, b.metrics.ratings_used);
    assert_eq!(a.metrics.ratings_refused, b.metrics.ratings_refused);

    assert_eq!(a.error_trajectory.len(), b.error_trajectory.len());
    for (x, y) in a.error_trajectory.iter().zip(b.error_trajectory.iter()) {
        assert!(approx_eq(*x, *y, 1e-12));
    }
}

#[test]
fn cli_validate_example_request_smoke() {
    let request_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/multi-rerank-request.json");

    let bin = cardinal_bin();
    let output = Command::new(&bin)
        .args(["validate", "--request"])
        .arg(&request_path)
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "failed to run cardinal validate at {}: {err}",
                bin.display()
            )
        });

    assert!(
        output.status.success(),
        "cardinal validate exited with {}; stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("valid request:"),
        "stdout did not confirm validation: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        output.stderr.is_empty(),
        "validate should not warn on the checked-in example request; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_experiment_expand_smoke() {
    let dir = tempdir().unwrap();
    let request_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/multi-rerank-request.json");
    let variant_path = dir.path().join("variants.json");
    let out_path = dir.path().join("expanded.json");
    std::fs::write(
        &variant_path,
        serde_json::json!([
            {
                "source_attribute_id": "clarity",
                "id": "skim_resistance",
                "prompt": "resistance to superficial skimming",
                "polarity": "positive",
                "prompt_template_slug": "canonical_bucket_v1"
            }
        ])
        .to_string(),
    )
    .unwrap();

    let bin = cardinal_bin();
    let output = Command::new(&bin)
        .args(["experiment-expand", "--request"])
        .arg(&request_path)
        .arg("--out")
        .arg(&out_path)
        .args([
            "--prompt-template",
            "canonical_v2",
            "--prompt-template",
            "canonical_bucket_v1",
            "--include-negative",
            "--variant-json",
        ])
        .arg(&variant_path)
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "failed to run cardinal experiment-expand at {}: {err}",
                bin.display()
            )
        });

    assert!(
        output.status.success(),
        "cardinal experiment-expand exited with {}; stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("2 attributes -> 9 attributes"),
        "stdout did not summarize expansion: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let expanded: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    let attributes = expanded
        .pointer("/attributes")
        .and_then(|value| value.as_array())
        .unwrap();
    let ids: Vec<&str> = attributes
        .iter()
        .map(|attr| attr.pointer("/id").and_then(|id| id.as_str()).unwrap())
        .collect();
    assert_eq!(attributes.len(), 9);
    assert!(ids.contains(&"clarity__pos__canonical_v2"));
    assert!(ids.contains(&"clarity_negative__neg__canonical_bucket_v1"));
    assert!(ids.contains(&"evidence__pos__canonical_bucket_v1"));
    assert!(ids.contains(&"skim_resistance__pos__canonical_bucket_v1"));
}

#[test]
fn cli_rerank_validates_before_gateway_setup() {
    let dir = tempdir().unwrap();
    let request_path = dir.path().join("invalid-request.json");
    let out_path = dir.path().join("out.json");
    std::fs::write(
        &request_path,
        serde_json::json!({
            "entities": [
                {"id": "a", "text": "A"},
                {"id": "b", "text": "B"}
            ],
            "attributes": [
                {"id": "clarity", "prompt": "clarity", "weight": 1.0}
            ],
            "topk": {"k": 0}
        })
        .to_string(),
    )
    .unwrap();

    let bin = cardinal_bin();
    let output = Command::new(&bin)
        .args(["rerank", "--request"])
        .arg(&request_path)
        .arg("--out")
        .arg(&out_path)
        .env_remove("OPENROUTER_API_KEY")
        .output()
        .unwrap_or_else(|err| panic!("failed to run cardinal rerank at {}: {err}", bin.display()));

    assert!(
        !output.status.success(),
        "invalid request should fail before gateway setup"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("topk.k must be >= 1"),
        "expected validation error before gateway setup; stderr={stderr}"
    );
}

#[test]
fn cli_report_json_smoke() {
    let dir = tempdir().unwrap();

    let request_path = dir.path().join("request.json");
    let response_path = dir.path().join("response.json");
    let out_path = dir.path().join("report.json");
    let md_path = dir.path().join("report.md");

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
            min_explore_degree: 2,
        },
        gates: vec![],
        comparison_budget: Some(1),
        latency_budget_ms: None,
        model: Some("openai/gpt-5-mini".into()),
        rater_id: None,
        comparison_concurrency: Some(1),
        max_pair_repeats: Some(1),
        randomize_presentation_order: true,
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
            global_topk_error: 0.2,
            tolerated_error: req.topk.tolerated_error,
            k: req.topk.k,
            band_size: req.topk.band_size,
            comparisons_attempted: 3,
            comparisons_used: 2,
            comparisons_refused: 1,
            comparisons_cached: 1,
            comparison_budget: 3,
            latency_ms: 1,
            model_used: "openai/gpt-5-mini".into(),
            rater_id_used: "openai/gpt-5-mini".into(),
            provider_input_tokens: 123,
            provider_output_tokens: 45,
            provider_cost_nanodollars: 123_456_789,
            stop_reason: RerankStopReason::BudgetExhausted,
        },
    };

    std::fs::write(&request_path, serde_json::to_string_pretty(&req).unwrap()).unwrap();
    std::fs::write(&response_path, serde_json::to_string_pretty(&resp).unwrap()).unwrap();

    let bin = cardinal_bin();
    let status = Command::new(&bin)
        .args(["report", "--format", "json"])
        .arg("--request")
        .arg(&request_path)
        .arg("--response")
        .arg(&response_path)
        .arg("--out")
        .arg(&out_path)
        .status()
        .unwrap_or_else(|err| panic!("failed to run cardinal report at {}: {err}", bin.display()));
    assert!(status.success(), "cardinal report exited with {status}");

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
        v.pointer("/summary/provider_input_tokens")
            .and_then(|n| n.as_u64())
            .unwrap(),
        123
    );
    assert_eq!(
        v.pointer("/summary/provider_output_tokens")
            .and_then(|n| n.as_u64())
            .unwrap(),
        45
    );
    assert_eq!(
        v.pointer("/summary/provider_cost_nanodollars")
            .and_then(|n| n.as_i64())
            .unwrap(),
        123_456_789
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

    let status = Command::new(&bin)
        .args(["report"])
        .arg("--request")
        .arg(&request_path)
        .arg("--response")
        .arg(&response_path)
        .arg("--out")
        .arg(&md_path)
        .status()
        .unwrap_or_else(|err| panic!("failed to run cardinal report at {}: {err}", bin.display()));
    assert!(status.success(), "cardinal report exited with {status}");

    let markdown = std::fs::read_to_string(&md_path).unwrap();
    assert!(markdown.contains("## Run Status"));
    assert!(markdown.contains("Stop reason: `budget_exhausted`"));
    assert!(markdown.contains("Comparison budget: 3"));
    assert!(markdown.contains("Provider tokens input/output/total: 123/45/168"));
    assert!(markdown.contains("Provider cost: $0.123456789"));
    assert!(markdown.contains("## Warnings / Degraded State"));
    assert!(markdown.contains("non-converged stop reason `budget_exhausted`"));
    assert!(markdown.contains("Global top-k error 0.2000 exceeds tolerated error 0.1000"));
    assert!(markdown.contains("1 comparison(s) were refused"));
    assert!(markdown.contains("1 comparison(s) came from cache"));
    assert!(markdown.contains("budget before meeting the stopping tolerance"));
    assert!(markdown.contains("`clarity`: latent 1.000 ± 0.100"));
}
