use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const SUITE_REL: &str = "examples/live-method-suite.json";
const PACK_REL: &str = "artifacts/live/method-comparison-2026-06-30-suite-v1";
const SUITE_SHA256: &str = "2ad870ab5d8fab3fa0b5bc6c61ee20805872a88e83d18cbe034fc2c37d49d1ed";
const SUMMARY_SCHEMA: &str = "live_method_comparison_summary_v1";
const REFERENCE_METHOD: &str = "reference_pairwise_ratio";
const CANDIDATE_METHODS: [&str; 4] = [
    "scalar_matrix",
    "list_sort",
    "ordinal_pairwise",
    "cardinal_pairwise_ratio",
];

#[derive(Debug, Default, Clone)]
struct UsageTotals {
    calls: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    cost_nanodollars: u64,
    cost_usd: f64,
    latency_ms: u64,
    cost_is_estimate: bool,
}

#[derive(Debug, Default)]
struct BudgetBucket {
    case_count: usize,
    tau_total: f64,
    topk_total: f64,
    agreement_total: f64,
    usage: UsageTotals,
}

fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn read_json(path: &Path) -> Value {
    let text = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!("failed to read {}: {err}", path.display());
    });
    serde_json::from_str(&text).unwrap_or_else(|err| {
        panic!("failed to parse {} as JSON: {err}", path.display());
    })
}

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    value
        .get(name)
        .unwrap_or_else(|| panic!("missing JSON field `{name}` in {value}"))
}

fn array<'a>(value: &'a Value, name: &str) -> &'a Vec<Value> {
    field(value, name)
        .as_array()
        .unwrap_or_else(|| panic!("JSON field `{name}` is not an array in {value}"))
}

fn string<'a>(value: &'a Value, name: &str) -> &'a str {
    field(value, name)
        .as_str()
        .unwrap_or_else(|| panic!("JSON field `{name}` is not a string in {value}"))
}

fn number(value: &Value, name: &str) -> f64 {
    field(value, name)
        .as_f64()
        .unwrap_or_else(|| panic!("JSON field `{name}` is not a number in {value}"))
}

fn integer(value: &Value, name: &str) -> u64 {
    field(value, name)
        .as_u64()
        .unwrap_or_else(|| panic!("JSON field `{name}` is not a non-negative integer in {value}"))
}

fn boolean(value: &Value, name: &str) -> bool {
    field(value, name)
        .as_bool()
        .unwrap_or_else(|| panic!("JSON field `{name}` is not a boolean in {value}"))
}

fn collect_files(root: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).unwrap_or_else(|err| {
        panic!("failed to read directory {}: {err}", root.display());
    }) {
        let path = entry
            .unwrap_or_else(|err| panic!("failed to read entry under {}: {err}", root.display()))
            .path();
        if path.is_dir() {
            collect_files(&path, files);
        } else {
            files.push(path);
        }
    }
}

fn all_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_files(root, &mut files);
    files.sort();
    files
}

fn files_named(root: &Path, name: &str) -> Vec<PathBuf> {
    all_files(root)
        .into_iter()
        .filter(|path| path.file_name().is_some_and(|file_name| file_name == name))
        .collect()
}

fn usage_totals(value: &Value) -> UsageTotals {
    UsageTotals {
        calls: integer(value, "calls"),
        prompt_tokens: integer(value, "prompt_tokens"),
        completion_tokens: integer(value, "completion_tokens"),
        total_tokens: integer(value, "total_tokens"),
        cost_nanodollars: integer(value, "cost_nanodollars"),
        cost_usd: number(value, "cost_usd"),
        latency_ms: integer(value, "latency_ms"),
        cost_is_estimate: boolean(value, "cost_is_estimate"),
    }
}

fn add_usage(total: &mut UsageTotals, usage: &Value) {
    let usage = usage_totals(usage);
    total.calls += usage.calls;
    total.prompt_tokens += usage.prompt_tokens;
    total.completion_tokens += usage.completion_tokens;
    total.total_tokens += usage.total_tokens;
    total.cost_nanodollars += usage.cost_nanodollars;
    total.cost_usd += usage.cost_usd;
    total.latency_ms += usage.latency_ms;
    total.cost_is_estimate |= usage.cost_is_estimate;
}

fn assert_close(left: f64, right: f64, label: &str) {
    assert!(
        (left - right).abs() <= 1e-9,
        "{label}: expected {left} to equal {right}"
    );
}

fn assert_same_usage(expected: &UsageTotals, actual: &Value, label: &str) {
    assert_eq!(expected.calls, integer(actual, "calls"), "{label} calls");
    assert_eq!(
        expected.prompt_tokens,
        integer(actual, "prompt_tokens"),
        "{label} prompt tokens"
    );
    assert_eq!(
        expected.completion_tokens,
        integer(actual, "completion_tokens"),
        "{label} completion tokens"
    );
    assert_eq!(
        expected.total_tokens,
        integer(actual, "total_tokens"),
        "{label} total tokens"
    );
    assert_eq!(
        expected.cost_nanodollars,
        integer(actual, "cost_nanodollars"),
        "{label} cost nanodollars"
    );
    assert_close(
        expected.cost_usd,
        number(actual, "cost_usd"),
        &format!("{label} cost usd"),
    );
    assert_eq!(
        expected.latency_ms,
        integer(actual, "latency_ms"),
        "{label} latency"
    );
    assert_eq!(
        expected.cost_is_estimate,
        boolean(actual, "cost_is_estimate"),
        "{label} estimate flag"
    );
}

fn item_ids(case_json: &Value) -> BTreeSet<String> {
    array(case_json, "items")
        .iter()
        .map(|item| string(item, "id").to_owned())
        .collect()
}

fn ranked_ids(result: &Value) -> BTreeSet<String> {
    array(result, "ranked_ids")
        .iter()
        .map(|id| {
            id.as_str()
                .expect("ranked_ids entries must be strings")
                .to_owned()
        })
        .collect()
}

fn assert_metrics_are_bounded(result: &Value, label: &str) {
    let metrics = field(result, "metrics_vs_reference");
    let tau = number(metrics, "kendall_tau");
    let topk = number(metrics, "topk_jaccard");
    assert!(
        (-1.0..=1.0).contains(&tau),
        "{label} tau out of bounds: {tau}"
    );
    assert!(
        (0.0..=1.0).contains(&topk),
        "{label} top-k out of bounds: {topk}"
    );
}

#[test]
fn live_method_suite_fixture_has_frozen_public_shape() {
    let suite = read_json(&repo_path(SUITE_REL));
    assert_eq!(string(&suite, "name"), "live_method_suite_v1");
    assert!(
        string(&suite, "description").contains("Frozen public structured-judgment cases"),
        "suite description should state the fixture purpose"
    );

    let cases = array(&suite, "cases");
    assert_eq!(cases.len(), 6, "public fixture should not silently shrink");

    let mut case_ids = BTreeSet::new();
    for case in cases {
        let case_name = string(case, "name");
        assert!(
            case_ids.insert(case_name.to_owned()),
            "duplicate case name {case_name}"
        );
        assert!(
            string(case, "description").len() >= 40,
            "case {case_name} has a thin description"
        );

        let items = array(case, "items");
        assert_eq!(
            items.len(),
            5,
            "case {case_name} should preserve five candidate items"
        );
        let mut ids = BTreeSet::new();
        for item in items {
            let id = string(item, "id");
            let text = string(item, "text");
            assert!(
                ids.insert(id.to_owned()),
                "duplicate item id {id} in {case_name}"
            );
            assert!(text.len() >= 30, "item {case_name}/{id} text is too thin");
            let lower = text.to_ascii_lowercase();
            assert!(
                !lower.contains("todo"),
                "item {case_name}/{id} contains TODO language"
            );
            assert!(
                !lower.contains("placeholder"),
                "item {case_name}/{id} contains placeholder language"
            );
            assert!(
                !lower.contains("lorem"),
                "item {case_name}/{id} contains lorem ipsum language"
            );
        }

        let attributes = array(case, "attributes");
        assert_eq!(
            attributes.len(),
            3,
            "case {case_name} should preserve three orthogonal attributes"
        );
        let mut attr_ids = BTreeSet::new();
        let mut weight_sum = 0.0;
        for attr in attributes {
            let id = string(attr, "id");
            let prompt = string(attr, "prompt");
            assert!(
                attr_ids.insert(id.to_owned()),
                "duplicate attribute id {id} in {case_name}"
            );
            assert!(
                prompt.len() >= 40,
                "attribute {case_name}/{id} prompt is too thin"
            );
            let weight = number(attr, "weight");
            assert!(
                weight > 0.0,
                "attribute {case_name}/{id} has non-positive weight {weight}"
            );
            weight_sum += weight;
        }
        assert_close(weight_sum, 1.0, &format!("{case_name} attribute weights"));
    }
}

#[test]
fn live_method_evidence_pack_is_complete_portable_and_secret_free() {
    let pack_dir = repo_path(PACK_REL);
    let summary_path = pack_dir.join("summary.json");
    let summary = read_json(&summary_path);

    assert_eq!(string(&summary, "schema_version"), SUMMARY_SCHEMA);
    assert_eq!(string(field(&summary, "suite"), "path"), SUITE_REL);
    assert_eq!(
        string(field(&summary, "suite"), "sha256"),
        SUITE_SHA256,
        "summary should pin the exact frozen suite input"
    );

    assert!(
        pack_dir.join("README.md").is_file(),
        "evidence pack README is missing"
    );
    assert!(
        pack_dir.join("summary.md").is_file(),
        "evidence pack summary.md is missing"
    );

    let mut request_count = 0usize;
    let mut response_count = 0usize;
    let mut parsed_count = 0usize;
    let mut usage_count = 0usize;
    for file in all_files(&pack_dir) {
        let bytes = fs::read(&file)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", file.display()));
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            !text.contains("sk-or-v1-"),
            "{} contains an OpenRouter key prefix",
            file.display()
        );
        assert!(
            !text.contains("Authorization"),
            "{} contains an Authorization header",
            file.display()
        );
        assert!(
            !text.contains("Bearer "),
            "{} contains a bearer token",
            file.display()
        );
        assert!(
            !text.contains("/Users/"),
            "{} contains a local absolute user path",
            file.display()
        );
        assert!(
            !text.contains("/var/folders/"),
            "{} contains a local temp path",
            file.display()
        );

        match file.file_name().and_then(|name| name.to_str()) {
            Some("request.json") => {
                request_count += 1;
                assert!(
                    file.with_file_name("response.json").is_file(),
                    "{} lacks response.json",
                    file.display()
                );
                assert!(
                    file.with_file_name("parsed.json").is_file(),
                    "{} lacks parsed.json",
                    file.display()
                );
                assert!(
                    file.with_file_name("usage.json").is_file(),
                    "{} lacks usage.json",
                    file.display()
                );
            }
            Some("response.json") => response_count += 1,
            Some("parsed.json") => parsed_count += 1,
            Some("usage.json") => usage_count += 1,
            _ => {}
        }
    }
    assert!(
        request_count > 500,
        "live pack should contain every raw provider call record"
    );
    assert_eq!(
        request_count, response_count,
        "request/response count mismatch"
    );
    assert_eq!(request_count, parsed_count, "request/parsed count mismatch");
    assert_eq!(request_count, usage_count, "request/usage count mismatch");

    let cases = array(&summary, "cases");
    assert_eq!(integer(&summary, "case_count") as usize, cases.len());
    assert_eq!(cases.len(), 6);

    let mut total_usage = UsageTotals::default();
    for case_row in cases {
        let case_name = string(case_row, "case");
        let case_dir = pack_dir.join(case_name);
        assert!(case_dir.is_dir(), "missing case directory {case_name}");

        let case_json = read_json(&case_dir.join("case.json"));
        let expected_ids = item_ids(&case_json);
        let item_count = array(&case_json, "items").len();
        let attr_count = array(&case_json, "attributes").len();
        let pairwise_call_count = attr_count * item_count * (item_count - 1) / 2;

        let reference = field(case_row, "reference");
        assert_eq!(string(reference, "method"), REFERENCE_METHOD);
        assert_eq!(
            ranked_ids(reference),
            expected_ids,
            "reference ranks unknown items in {case_name}"
        );
        assert_metrics_are_bounded(reference, &format!("{case_name}/{REFERENCE_METHOD}"));
        assert_eq!(
            read_json(&case_dir.join("reference_pairwise_ratio.json")),
            *reference,
            "summary reference row should match the case-level JSON file"
        );
        add_usage(&mut total_usage, field(reference, "usage"));

        let reference_evidence =
            files_named(&case_dir.join("calls").join(REFERENCE_METHOD), "usage.json");
        assert_eq!(
            reference_evidence.len(),
            pairwise_call_count,
            "reference call count should equal attributes * nC2 for {case_name}"
        );

        let methods = array(case_row, "methods");
        assert_eq!(methods.len(), CANDIDATE_METHODS.len());
        let mut seen_methods = BTreeSet::new();
        for method in methods {
            let method_name = string(method, "method");
            assert!(
                CANDIDATE_METHODS.contains(&method_name),
                "unexpected method {method_name}"
            );
            assert!(
                seen_methods.insert(method_name.to_owned()),
                "duplicate method {method_name} in {case_name}"
            );
            assert_eq!(
                ranked_ids(method),
                expected_ids,
                "method {method_name} ranks unknown items in {case_name}"
            );
            assert_metrics_are_bounded(method, &format!("{case_name}/{method_name}"));
            assert_eq!(
                read_json(&case_dir.join(format!("{method_name}.json"))),
                *method,
                "summary method row should match the case-level JSON file"
            );
            add_usage(&mut total_usage, field(method, "usage"));

            let expected_evidence = match method_name {
                "scalar_matrix" | "list_sort" => 1,
                "ordinal_pairwise" | "cardinal_pairwise_ratio" => pairwise_call_count,
                other => panic!("unhandled method {other}"),
            };
            let actual_evidence =
                files_named(&case_dir.join("calls").join(method_name), "usage.json");
            assert_eq!(
                actual_evidence.len(),
                expected_evidence,
                "call count mismatch for {case_name}/{method_name}"
            );
        }
        for expected_method in CANDIDATE_METHODS {
            assert!(
                seen_methods.contains(expected_method),
                "missing {expected_method} in {case_name}"
            );
        }
    }

    assert_same_usage(&total_usage, field(&summary, "totals"), "summary totals");
}

#[test]
fn live_method_budget_normalized_rows_match_case_evidence() {
    let summary = read_json(&repo_path(PACK_REL).join("summary.json"));
    let mut buckets: BTreeMap<(String, String), BudgetBucket> = BTreeMap::new();

    for case in array(&summary, "cases") {
        let mut results = Vec::with_capacity(1 + CANDIDATE_METHODS.len());
        results.push(field(case, "reference"));
        results.extend(array(case, "methods"));

        for result in results {
            let method = string(result, "method").to_owned();
            let model = string(result, "model").to_owned();
            let metrics = field(result, "metrics_vs_reference");
            let tau = number(metrics, "kendall_tau");
            let topk = number(metrics, "topk_jaccard");
            let agreement = (((tau + 1.0) / 2.0) + topk) / 2.0;
            let bucket = buckets.entry((method, model)).or_default();
            bucket.case_count += 1;
            bucket.tau_total += tau;
            bucket.topk_total += topk;
            bucket.agreement_total += agreement;
            add_usage(&mut bucket.usage, field(result, "usage"));
        }
    }

    let rows = array(&summary, "budget_normalized_methods");
    assert_eq!(rows.len(), buckets.len());
    for row in rows {
        let key = (
            string(row, "method").to_owned(),
            string(row, "model").to_owned(),
        );
        let bucket = buckets
            .get(&key)
            .unwrap_or_else(|| panic!("budget row {:?} has no backing case results", key));
        assert_eq!(bucket.case_count as u64, integer(row, "case_count"));
        assert_close(
            bucket.tau_total / bucket.case_count as f64,
            number(row, "mean_kendall_tau"),
            "mean tau",
        );
        assert_close(
            bucket.topk_total / bucket.case_count as f64,
            number(row, "mean_topk_jaccard"),
            "mean top-k",
        );
        assert_close(
            bucket.agreement_total / bucket.case_count as f64,
            number(row, "mean_agreement_score"),
            "mean agreement",
        );
        assert_close(
            bucket.agreement_total,
            number(row, "agreement_score_total"),
            "agreement total",
        );
        assert_same_usage(
            &bucket.usage,
            field(row, "usage"),
            &format!("budget row {}", key.0),
        );

        let calls = bucket.usage.calls as f64;
        let total_tokens = bucket.usage.total_tokens as f64;
        if calls > 0.0 {
            assert_close(
                bucket.agreement_total / calls,
                number(row, "agreement_score_per_call"),
                "agreement per call",
            );
        }
        if total_tokens > 0.0 {
            assert_close(
                bucket.agreement_total / (total_tokens / 1000.0),
                number(row, "agreement_score_per_1k_tokens"),
                "agreement per 1k tokens",
            );
        }
        if bucket.usage.cost_usd > 0.0 {
            assert_close(
                bucket.agreement_total / bucket.usage.cost_usd,
                number(row, "agreement_score_per_usd"),
                "agreement per usd",
            );
        }
    }
}
