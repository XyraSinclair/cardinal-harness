//! CLI tests for the `judge`, `elaborate`, and `explain` commands.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

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
    assert!(fallback.exists(), "missing binary: {}", fallback.display());
    fallback
}

fn run_cardinal(server_uri: &str, args: &[&str], stdin: &str) -> std::process::Output {
    let mut child = Command::new(cardinal_bin())
        .args(args)
        .env("OPENROUTER_API_KEY", "sk-test")
        .env("OPENROUTER_BASE_URL", server_uri)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cardinal");
    {
        use std::io::Write;
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(stdin.as_bytes())
            .expect("write stdin");
    }
    child.wait_with_output().expect("wait")
}

/// Shininess judge that also serves elaboration and proposal chats.
#[derive(Clone, Copy)]
struct OmniJudge;

impl Respond for OmniJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let parsed: serde_json::Value = serde_json::from_slice(&request.body).unwrap_or_default();
        let system = parsed
            .get("messages")
            .and_then(|m| m.as_array())
            .and_then(|messages| {
                messages
                    .iter()
                    .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
                    .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                    .map(str::to_string)
            })
            .unwrap_or_default();
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

        let content = if system.contains("judging rubrics") {
            // Elaboration meta-prompt.
            "You are judging MOCK-RUBRIC shininess: reward light reflection only.".to_string()
        } else if system.contains("analyze rankings") {
            // Candidate proposal.
            r#"["shininess", "metallic lustre"]"#.to_string()
        } else {
            // Pairwise judgement (ratio or ordinal template).
            let a = extract_between(&user_content, "<entity_A_context>", "</entity_A_context>")
                .map(metal_score)
                .unwrap_or(0);
            let b = extract_between(&user_content, "<entity_B_context>", "</entity_B_context>")
                .map(metal_score)
                .unwrap_or(0);
            let higher = if a >= b { "A" } else { "B" };
            if system.contains("direction only") {
                format!(r#"{{"higher_ranked":"{higher}","confidence":0.8}}"#)
            } else {
                let ratio = if (a - b).abs() >= 2 { 3.9 } else { 1.5 };
                format!(r#"{{"higher_ranked":"{higher}","ratio":{ratio},"confidence":0.9}}"#)
            }
        };

        ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": { "content": content },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 10 }
        }))
    }
}

async fn start_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(OmniJudge)
        .mount(&server)
        .await;
    server
}

#[tokio::test(flavor = "multi_thread")]
async fn judge_reports_direction_ratio_and_cost() {
    let server = start_server().await;
    let output = run_cardinal(
        &server.uri(),
        &[
            "judge",
            "shiny GOLD ring",
            "dull TIN spoon",
            "--by",
            "shininess",
            "--model",
            "test/judge",
            "--no-cache",
        ],
        "",
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.starts_with("A wins"), "stdout: {stdout}");
    assert!(stdout.contains("ratio 3.9"), "stdout: {stdout}");
}

#[tokio::test(flavor = "multi_thread")]
async fn judge_json_and_show_prompt() {
    let server = start_server().await;
    let output = run_cardinal(
        &server.uri(),
        &[
            "judge",
            "dull TIN spoon",
            "shiny GOLD ring",
            "--by",
            "shininess",
            "--model",
            "test/judge",
            "--no-cache",
            "--json",
            "--show-prompt",
        ],
        "",
    );
    assert!(output.status.success());
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["higher_ranked"], "B");
    assert_eq!(parsed["refused"], false);
    assert!(parsed["cost_nanodollars"].is_number());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--- system ---"), "stderr: {stderr}");
    assert!(stderr.contains("shininess"), "stderr: {stderr}");
    assert!(stderr.contains("dull TIN spoon"), "stderr: {stderr}");
}

#[tokio::test(flavor = "multi_thread")]
async fn judge_supports_ordinal_template() {
    let server = start_server().await;
    let output = run_cardinal(
        &server.uri(),
        &[
            "judge",
            "shiny GOLD ring",
            "dull TIN spoon",
            "--by",
            "shininess",
            "--model",
            "test/judge",
            "--template",
            "ordinal_v1",
            "--no-cache",
            "--json",
        ],
        "",
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["higher_ranked"], "A");
}

#[tokio::test(flavor = "multi_thread")]
async fn elaborate_prints_only_the_rubric_on_stdout() {
    let server = start_server().await;
    let output = run_cardinal(
        &server.uri(),
        &["elaborate", "--by", "shininess", "--model", "test/judge"],
        "",
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        stdout.trim(),
        "You are judging MOCK-RUBRIC shininess: reward light reflection only."
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("elaborated \"shininess\""), "{stderr}");
}

#[tokio::test(flavor = "multi_thread")]
async fn sort_elaborate_uses_the_rubric_as_criterion() {
    let server = start_server().await;
    let output = run_cardinal(
        &server.uri(),
        &[
            "sort",
            "--by",
            "shininess",
            "--model",
            "test/judge",
            "--no-cache",
            "--budget",
            "12",
            "--elaborate",
        ],
        "shiny GOLD ring\ndull TIN spoon\n",
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("MOCK-RUBRIC"), "stderr: {stderr}");
    // The judge requests must carry the rubric, not the terse criterion.
    let requests = server.received_requests().await.unwrap();
    let pairwise: Vec<_> = requests
        .iter()
        .filter(|r| String::from_utf8_lossy(&r.body).contains("entity_A_context"))
        .collect();
    assert!(!pairwise.is_empty());
    for r in pairwise {
        assert!(
            String::from_utf8_lossy(&r.body).contains("MOCK-RUBRIC"),
            "pairwise request missing rubric"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn explain_scores_candidates_and_fits_weights() {
    let server = start_server().await;
    let output = run_cardinal(
        &server.uri(),
        &[
            "explain",
            "--candidate",
            "shininess",
            "--model",
            "test/judge",
            "--budget",
            "60",
            "--no-cache",
            "--seed",
            "7",
        ],
        "shiny GOLD ring\nbright SILVER fork\nold BRONZE coin\ndull TIN spoon\n",
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("shininess"), "stdout: {stdout}");
    assert!(
        stdout.contains("reconstructs your ranking"),
        "stdout: {stdout}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn explain_proposes_candidates_via_llm() {
    let server = start_server().await;
    let output = run_cardinal(
        &server.uri(),
        &[
            "explain",
            "--propose",
            "2",
            "--model",
            "test/judge",
            "--budget",
            "60",
            "--no-cache",
            "--seed",
            "7",
        ],
        "shiny GOLD ring\nbright SILVER fork\nold BRONZE coin\ndull TIN spoon\n",
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("proposed 2 candidate"), "stderr: {stderr}");
    assert!(stderr.contains("metallic lustre"), "stderr: {stderr}");
}
