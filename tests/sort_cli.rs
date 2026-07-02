//! End-to-end tests for the `cardinal sort` CLI and the `sort_texts` /
//! `sort_documents` library surface, using a deterministic wiremock judge.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use cardinal_harness::gateway::openrouter::OpenRouterAdapter;
use cardinal_harness::gateway::{Attribution, GatewayConfig, NoopUsageSink, ProviderGateway};
use cardinal_harness::rerank::{sort_texts, RerankExecution, SortError, SortOptions};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Judges GOLD > SILVER > BRONZE > TIN deterministically.
#[derive(Clone, Copy)]
struct MetalJudge;

fn extract_between<'a>(s: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let start_idx = s.find(start)? + start.len();
    let rest = &s[start_idx..];
    let end_idx = rest.find(end)?;
    Some(&rest[..end_idx])
}

fn metal_score(ctx: &str) -> i32 {
    if ctx.contains("GOLD") {
        4
    } else if ctx.contains("SILVER") {
        3
    } else if ctx.contains("BRONZE") {
        2
    } else if ctx.contains("TIN") {
        1
    } else {
        0
    }
}

impl Respond for MetalJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let parsed: serde_json::Value = serde_json::from_slice(&request.body).unwrap_or_default();
        let user_content = parsed
            .get("messages")
            .and_then(|m| m.as_array())
            .and_then(|messages| {
                messages
                    .iter()
                    .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
                    .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                    .map(str::to_string)
            })
            .unwrap_or_default();

        let a_score = extract_between(&user_content, "<entity_A_context>", "</entity_A_context>")
            .map(metal_score)
            .unwrap_or(0);
        let b_score = extract_between(&user_content, "<entity_B_context>", "</entity_B_context>")
            .map(metal_score)
            .unwrap_or(0);

        let (higher, ratio) = if a_score >= b_score {
            (
                "A",
                if (a_score - b_score).abs() >= 2 {
                    3.9
                } else {
                    1.5
                },
            )
        } else {
            (
                "B",
                if (b_score - a_score).abs() >= 2 {
                    3.9
                } else {
                    1.5
                },
            )
        };
        let content = format!(r#"{{"higher_ranked":"{higher}","ratio":{ratio},"confidence":0.9}}"#);

        ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": { "content": content },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 10 }
        }))
    }
}

async fn start_judge() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(MetalJudge)
        .mount(&server)
        .await;
    server
}

fn cardinal_bin() -> PathBuf {
    let cargo_bin = option_env!("CARGO_BIN_EXE_cardinal").filter(|path| !path.is_empty());
    if let Some(path) = cargo_bin {
        let path = PathBuf::from(path);
        if path.exists() {
            return path;
        }
    }
    let test_exe = std::env::current_exe().expect("current test exe");
    let deps_dir = test_exe.parent().expect("deps dir");
    let target_dir = deps_dir.parent().expect("target dir");
    let fallback = target_dir.join(format!("cardinal{}", std::env::consts::EXE_SUFFIX));
    assert!(
        fallback.exists(),
        "failed to locate compiled cardinal binary at {}",
        fallback.display()
    );
    fallback
}

fn run_sort(server_uri: &str, args: &[&str], stdin: &str) -> std::process::Output {
    let mut child = Command::new(cardinal_bin())
        .arg("sort")
        .args(args)
        .env("OPENROUTER_API_KEY", "sk-test")
        .env("OPENROUTER_BASE_URL", server_uri)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cardinal sort");
    {
        use std::io::Write;
        child
            .stdin
            .as_mut()
            .expect("child stdin")
            .write_all(stdin.as_bytes())
            .expect("write stdin");
    }
    child.wait_with_output().expect("wait for cardinal sort")
}

const COMMON_ARGS: &[&str] = &[
    "--by",
    "shininess",
    "--model",
    "test/judge",
    "--no-cache",
    "--seed",
    "7",
    "--budget",
    "24",
];

#[tokio::test]
async fn library_sort_texts_orders_items_and_maps_texts() {
    let server = start_judge().await;
    let adapter =
        OpenRouterAdapter::with_config("sk-test", server.uri(), Duration::from_secs(5), None, None)
            .unwrap();
    let gateway = Arc::new(ProviderGateway::with_config(
        adapter,
        Arc::new(NoopUsageSink),
        GatewayConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(0),
        },
    ));
    let execution = RerankExecution::new(gateway, Attribution::new("test::sort"));

    let sorted = sort_texts(
        vec![
            "dull TIN spoon".into(),
            "shiny GOLD ring".into(),
            "old BRONZE coin".into(),
            "bright SILVER fork".into(),
        ],
        "shininess",
        execution,
        SortOptions {
            model: Some("test/judge".into()),
            comparison_budget: Some(24),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let texts: Vec<&str> = sorted.items.iter().map(|i| i.text.as_str()).collect();
    assert_eq!(
        texts,
        vec![
            "shiny GOLD ring",
            "bright SILVER fork",
            "old BRONZE coin",
            "dull TIN spoon",
        ]
    );
    let ranks: Vec<usize> = sorted.items.iter().map(|i| i.rank).collect();
    assert_eq!(ranks, vec![1, 2, 3, 4]);
    assert!(sorted.meta.comparisons_used > 0);
    for item in &sorted.items {
        assert!(item.latent_std.is_finite());
    }
}

#[tokio::test]
async fn library_sort_rejects_empty_input() {
    let server = start_judge().await;
    let adapter =
        OpenRouterAdapter::with_config("sk-test", server.uri(), Duration::from_secs(5), None, None)
            .unwrap();
    let gateway = Arc::new(ProviderGateway::with_config(
        adapter,
        Arc::new(NoopUsageSink),
        GatewayConfig::default(),
    ));
    let execution = RerankExecution::new(gateway, Attribution::new("test::sort"));
    let err = sort_texts(vec![], "shininess", execution, SortOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err, SortError::EmptyInput));
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_sort_plain_text_outputs_reordered_lines() {
    let server = start_judge().await;
    let input = "dull TIN spoon\n\nshiny GOLD ring\nold BRONZE coin\nbright SILVER fork\n";
    let output = run_sort(&server.uri(), COMMON_ARGS, input);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        stdout,
        "shiny GOLD ring\nbright SILVER fork\nold BRONZE coin\ndull TIN spoon\n",
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("sorted 4 items"), "stderr: {stderr}");
    assert!(stderr.contains("stop:"), "stderr: {stderr}");
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_sort_reverse_and_scores() {
    let server = start_judge().await;
    let input = "shiny GOLD ring\ndull TIN spoon\n";
    let mut args = COMMON_ARGS.to_vec();
    args.extend_from_slice(&["--reverse", "--scores", "--quiet"]);
    let output = run_sort(&server.uri(), &args, input);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].ends_with("\tdull TIN spoon"), "line: {}", lines[0]);
    assert!(
        lines[1].ends_with("\tshiny GOLD ring"),
        "line: {}",
        lines[1]
    );
    assert!(lines[0].contains('\u{b1}'), "line: {}", lines[0]);
    assert!(output.stderr.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_sort_json_array_input_and_json_output() {
    let server = start_judge().await;
    let input = r#"[
        {"id": "tin", "text": "dull TIN spoon"},
        {"id": "gold", "text": "shiny GOLD ring"},
        "old BRONZE coin"
    ]"#;
    let mut args = COMMON_ARGS.to_vec();
    args.extend_from_slice(&["--format", "json", "--quiet"]);
    let output = run_sort(&server.uri(), &args, input);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let items = parsed["items"].as_array().unwrap();
    assert_eq!(items.len(), 3);
    assert_eq!(items[0]["id"], "gold");
    assert_eq!(items[0]["rank"], 1);
    assert_eq!(items[1]["text"], "old BRONZE coin");
    assert_eq!(items[2]["id"], "tin");
    assert!(parsed["meta"]["comparisons_used"].as_u64().unwrap() > 0);
    assert!(parsed["meta"]["stop_reason"].is_string());
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_sort_csv_output_quotes_fields() {
    let server = start_judge().await;
    let input = "shiny GOLD ring, polished\ndull TIN spoon\n";
    let mut args = COMMON_ARGS.to_vec();
    args.extend_from_slice(&["--format", "csv", "--quiet"]);
    let output = run_sort(&server.uri(), &args, input);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines[0],
        "rank,id,latent_mean,latent_std,z_score,percentile,text"
    );
    assert!(
        lines[1].ends_with("\"shiny GOLD ring, polished\""),
        "line: {}",
        lines[1]
    );
    assert_eq!(lines.len(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_sort_empty_input_fails_cleanly() {
    let server = start_judge().await;
    let output = run_sort(&server.uri(), COMMON_ARGS, "\n\n");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("no items to sort"), "stderr: {stderr}");
}

#[tokio::test(flavor = "multi_thread")]
async fn cli_sort_fails_loudly_when_every_comparison_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server)
        .await;
    let args = [
        "--by",
        "shininess",
        "--model",
        "test/judge",
        "--no-cache",
        "--seed",
        "7",
        "--budget",
        "2",
    ];
    let output = run_sort(&server.uri(), &args, "shiny GOLD ring\ndull TIN spoon\n");
    assert!(!output.status.success());
    assert!(
        output.stdout.is_empty(),
        "stdout must stay empty on an uninformative sort"
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("uninformative"), "stderr: {stderr}");
}

#[test]
fn cli_sort_without_key_or_cache_only_fails_with_guidance() {
    let mut child = Command::new(cardinal_bin())
        .arg("sort")
        .args(["--by", "shininess", "--no-cache"])
        .env_remove("OPENROUTER_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cardinal sort");
    {
        use std::io::Write;
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(b"one\ntwo\n")
            .unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("OPENROUTER_API_KEY"), "stderr: {stderr}");
    assert!(stderr.contains("--cache-only"), "stderr: {stderr}");
}
