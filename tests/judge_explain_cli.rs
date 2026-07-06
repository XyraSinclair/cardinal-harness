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
        } else if system.contains("decompose goals") {
            // Goal decomposition for automated AHP; the metal words make the
            // downstream pairwise importance judgements deterministic.
            r#"["shiny GOLD standard", "bright SILVER middle"]"#.to_string()
        } else if system.contains("distinguish one item") {
            // Distinguishing-attribute proposal.
            r#"["shininess", "metallic lustre"]"#.to_string()
        } else {
            // Pairwise judgement (ratio or ordinal template).
            let a = extract_between(&user_content, "<entity_A_context>", "</entity_A_context>")
                .map(metal_score)
                .unwrap_or(0);
            let b = extract_between(&user_content, "<entity_B_context>", "</entity_B_context>")
                .map(metal_score)
                .unwrap_or(0);
            // Polarity-aware: a negated criterion ("lack of X") reverses
            // the metal direction — the orbit-perfect behavior.
            let negated = user_content.contains("lack of");
            let (a, b) = if negated { (b, a) } else { (a, b) };
            let higher = if a >= b { "A" } else { "B" };
            let lower = if a >= b { "B" } else { "A" };
            let ratio = if (a - b).abs() >= 2 { 3.9 } else { 1.5 };
            if system.contains("direction only") {
                format!(r#"{{"higher_ranked":"{higher}","confidence":0.8}}"#)
            } else if system.contains("LESS of that attribute") {
                // Coherent group inverse: same magnitude, mirrored subject.
                format!(r#"{{"lower_ranked":"{lower}","ratio":{ratio},"confidence":0.9}}"#)
            } else if system.contains("what fraction") {
                // Coherent reciprocal.
                let fraction = 1.0 / ratio;
                format!(
                    r#"{{"higher_ranked":"{higher}","fraction":{fraction:.6},"confidence":0.9}}"#
                )
            } else {
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

/// A judge that follows the requester's leaning instead of the content:
/// when the criterion carries a "Note from the requester" quoting an item's
/// opening, it votes for whichever slot holds that item. Neutral calls fall
/// back to the metal scale. The spin probe must catch this.
#[derive(Clone, Copy)]
struct SycophantJudge;

impl Respond for SycophantJudge {
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

        let ctx_a = extract_between(&user_content, "<entity_A_context>", "</entity_A_context>")
            .unwrap_or_default();
        let ctx_b = extract_between(&user_content, "<entity_B_context>", "</entity_B_context>")
            .unwrap_or_default();

        let favored = extract_between(&user_content, "the one beginning «", "» is clearly stronger");
        let higher = match favored {
            Some(excerpt) if ctx_a.trim().starts_with(excerpt) => "A",
            Some(excerpt) if ctx_b.trim().starts_with(excerpt) => "B",
            _ => {
                if metal_score(ctx_a) >= metal_score(ctx_b) {
                    "A"
                } else {
                    "B"
                }
            }
        };
        let content = format!(r#"{{"higher_ranked":"{higher}","ratio":3.9,"confidence":0.9}}"#);
        ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": { "content": content },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 10 }
        }))
    }
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

#[tokio::test(flavor = "multi_thread")]
async fn weigh_produces_normalized_ahp_priority_vector() {
    // The star judge sees attribute descriptions as entities; more '*'
    // in the description = more important. Weights must come out ordered
    // and sum to 1.
    let server = start_server().await;
    let dir = std::env::temp_dir().join(format!("cardinal-weigh-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let output = run_cardinal(
        &server.uri(),
        &[
            "weigh",
            "--goal",
            "maximum shine",
            "--attribute",
            "gold=shiny GOLD standard",
            "--attribute",
            "tin=dull TIN afterthought",
            "--attribute",
            "silver=bright SILVER middle",
            "--model",
            "test/judge",
            "--budget",
            "24",
            "--cache",
            dir.join("weigh.sqlite").to_str().unwrap(),
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
    let weights = parsed["weights"].as_array().unwrap();
    assert_eq!(weights.len(), 3);
    let total: f64 = weights.iter().map(|w| w["weight"].as_f64().unwrap()).sum();
    assert!((total - 1.0).abs() < 1e-9, "weights sum to 1: {total}");
    // Ordered by the sort: gold > silver > tin, ratio-scale spread.
    assert_eq!(weights[0]["attribute"], "gold");
    assert_eq!(weights[2]["attribute"], "tin");
    assert!(
        weights[0]["weight"].as_f64().unwrap() > weights[2]["weight"].as_f64().unwrap() * 1.5,
        "priority vector must be ratio-scale, not flat: {weights:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn weigh_propose_runs_automated_ahp() {
    // No --attribute at all: the model decomposes the goal into
    // considerations, which are then weighed pairwise like any others.
    let server = start_server().await;
    let dir = std::env::temp_dir().join(format!("cardinal-weigh-ahp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let output = run_cardinal(
        &server.uri(),
        &[
            "weigh",
            "--goal",
            "maximum shine",
            "--propose",
            "2",
            "--model",
            "test/judge",
            "--budget",
            "24",
            "--cache",
            dir.join("weigh-ahp.sqlite").to_str().unwrap(),
            "--json",
        ],
        "",
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("proposed 2 considerations"),
        "stderr: {stderr}"
    );
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let weights = parsed["weights"].as_array().unwrap();
    assert_eq!(weights.len(), 2);
    let total: f64 = weights.iter().map(|w| w["weight"].as_f64().unwrap()).sum();
    assert!((total - 1.0).abs() < 1e-9, "weights sum to 1: {total}");
    // The mock judge ranks GOLD above SILVER for shine.
    assert!(
        weights[0]["attribute"].as_str().unwrap().contains("GOLD"),
        "gold-flavored consideration should dominate: {weights:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn distinguish_profiles_focal_item_measured() {
    // The propagation primitive: propose attributes for the focal item, then
    // MEASURE where it lands on each — the profile is the receipt, not the
    // proposer's say-so. Focal item is the shiniest metal, so it must come
    // out at the top percentile of both proposed attributes.
    let server = start_server().await;
    let dir = std::env::temp_dir().join(format!("cardinal-distinguish-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let output = run_cardinal(
        &server.uri(),
        &[
            "distinguish",
            "--focus",
            "1",
            "--model",
            "test/judge",
            "--budget",
            "64",
            "--cache",
            dir.join("distinguish.sqlite").to_str().unwrap(),
            "--json",
        ],
        "shiny GOLD ring\ndull TIN spoon\nplain BRONZE cup\nold SILVER coin\n",
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("proposed 2 distinguishing attributes"),
        "stderr: {stderr}"
    );
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["focal_id"], "item-0000");
    let attrs = parsed["attributes"].as_array().unwrap();
    assert_eq!(attrs.len(), 2);
    for attr in attrs {
        let pct = attr["percentile"].as_f64().unwrap();
        let z = attr["z_score"].as_f64().unwrap();
        assert!(
            pct >= 0.7,
            "gold ring must measure near the top: percentile {pct} in {attr}"
        );
        assert!(z > 0.0, "positive standout required: z {z}");
    }
    // Sorted best-direction-first.
    let z0 = attrs[0]["z_score"].as_f64().unwrap();
    let z1 = attrs[1]["z_score"].as_f64().unwrap();
    assert!(z0 >= z1, "profile must be sorted by z descending");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn judge_spin_confirms_framing_blind_judge() {
    // The metal judge only reads content: susceptibility must be ~0 and the
    // belief must survive both spins.
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
            "--spin",
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
    assert_eq!(parsed["belief_survives_spin"], true, "{parsed}");
    let chi = parsed["susceptibility_nats"].as_f64().unwrap();
    assert!(
        chi.abs() < 1e-9,
        "framing-blind judge must show zero susceptibility: {chi}"
    );
    assert_eq!(parsed["comparisons"], 6);
    let readings = parsed["readings"].as_array().unwrap();
    assert_eq!(readings.len(), 3);
    for reading in readings {
        assert!(
            reading["mean_log_ratio"].as_f64().unwrap() > 0.0,
            "gold beats tin under every framing: {reading}"
        );
    }
}

/// Follows the leaning proportionally to its intensity: a LINEAR responder.
/// Neutral falls back to the metal scale.
#[derive(Clone, Copy)]
struct LinearSycophantJudge;

fn framing_intensity(user: &str) -> u8 {
    if user.contains("slight hunch") {
        1
    } else if user.contains("fairly convinced") {
        2
    } else if user.contains("It is certain to me") {
        3
    } else {
        0
    }
}

impl Respond for LinearSycophantJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let parsed: serde_json::Value = serde_json::from_slice(&request.body).unwrap_or_default();
        let user = parsed["messages"]
            .as_array()
            .and_then(|m| {
                m.iter()
                    .find(|x| x["role"] == "user")
                    .and_then(|x| x["content"].as_str())
            })
            .unwrap_or("")
            .to_string();
        let ctx_a = extract_between(&user, "<entity_A_context>", "</entity_A_context>")
            .unwrap_or_default()
            .trim()
            .to_string();
        let favored = extract_between(&user, "the one beginning «", "»");
        let intensity = framing_intensity(&user);
        // Response in log-space: m_toward_favored = 0.3 * intensity, on a
        // pair with no intrinsic belief (equal metals): pure paramagnet.
        let m = 0.3 * f64::from(intensity);
        let toward_a = match favored {
            Some(excerpt) if ctx_a.starts_with(excerpt) => m,
            Some(_) => -m,
            None => 0.0,
        };
        let (higher, ratio) = if toward_a >= 0.0 {
            ("A", toward_a.exp())
        } else {
            ("B", (-toward_a).exp())
        };
        let content = format!(r#"{{"higher_ranked":"{higher}","ratio":{ratio:.6},"confidence":0.9}}"#);
        ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "message": { "content": content }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 10 }
        }))
    }
}

/// Ignores mild and moderate pressure, folds completely at maximum
/// intensity: a THRESHOLD sycophant. The two-point secant cannot tell this
/// judge from a mild linear responder; the sweep's linearity R² can.
#[derive(Clone, Copy)]
struct ThresholdSycophantJudge;

impl Respond for ThresholdSycophantJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let parsed: serde_json::Value = serde_json::from_slice(&request.body).unwrap_or_default();
        let user = parsed["messages"]
            .as_array()
            .and_then(|m| {
                m.iter()
                    .find(|x| x["role"] == "user")
                    .and_then(|x| x["content"].as_str())
            })
            .unwrap_or("")
            .to_string();
        if framing_intensity(&user) == 3 {
            let ctx_a = extract_between(&user, "<entity_A_context>", "</entity_A_context>")
                .unwrap_or_default()
                .trim()
                .to_string();
            if let Some(excerpt) = extract_between(&user, "the one beginning «", "»") {
                let higher = if ctx_a.starts_with(excerpt) { "A" } else { "B" };
                let content =
                    format!(r#"{{"higher_ranked":"{higher}","ratio":3.9,"confidence":0.9}}"#);
                return ResponseTemplate::new(200).set_body_json(json!({
                    "choices": [{ "message": { "content": content }, "finish_reason": "stop" }],
                    "usage": { "prompt_tokens": 10, "completion_tokens": 10 }
                }));
            }
        }
        OmniJudge.respond(request)
    }
}

/// Answers the "which has LESS" wording as if it had been asked "which has
/// MORE" — a judge that cannot invert its own scale. The wording probe must
/// catch the sign flip.
#[derive(Clone, Copy)]
struct InversionBlindJudge;

impl Respond for InversionBlindJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let parsed: serde_json::Value = serde_json::from_slice(&request.body).unwrap_or_default();
        let system = parsed["messages"]
            .as_array()
            .and_then(|m| {
                m.iter()
                    .find(|x| x["role"] == "system")
                    .and_then(|x| x["content"].as_str())
            })
            .unwrap_or("")
            .to_string();
        let user = parsed["messages"]
            .as_array()
            .and_then(|m| {
                m.iter()
                    .find(|x| x["role"] == "user")
                    .and_then(|x| x["content"].as_str())
            })
            .unwrap_or("")
            .to_string();
        let a = extract_between(&user, "<entity_A_context>", "</entity_A_context>")
            .map(metal_score)
            .unwrap_or(0);
        let b = extract_between(&user, "<entity_B_context>", "</entity_B_context>")
            .map(metal_score)
            .unwrap_or(0);
        let higher = if a >= b { "A" } else { "B" };
        let content = if system.contains("LESS of that attribute") {
            // BUG under test: names the HIGHER entity as "lower_ranked".
            format!(r#"{{"lower_ranked":"{higher}","ratio":3.9,"confidence":0.9}}"#)
        } else if system.contains("what fraction") {
            format!(r#"{{"higher_ranked":"{higher}","fraction":0.256410,"confidence":0.9}}"#)
        } else {
            format!(r#"{{"higher_ranked":"{higher}","ratio":3.9,"confidence":0.9}}"#)
        };
        ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "message": { "content": content }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 10 }
        }))
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn anp_produces_stochastic_limits_and_a_network_correction() {
    // Full network on the metal scale: three criteria, four alternatives,
    // every supermatrix edge measured through the mock judge. The checks
    // are structural/mathematical: stochasticity, convergence, ordering.
    let server = start_server().await;
    let dir = std::env::temp_dir().join(format!("cardinal-anp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let output = run_cardinal(
        &server.uri(),
        &[
            "anp",
            "--goal",
            "maximum shine",
            "--attribute",
            "gold=shiny GOLD standard",
            "--attribute",
            "silver=bright SILVER middle",
            "--attribute",
            "tin=dull TIN afterthought",
            "--model",
            "test/judge",
            "--budget",
            "48",
            "--cache",
            dir.join("anp.sqlite").to_str().unwrap(),
            "--json",
        ],
        "shiny GOLD ring\nold SILVER coin\nplain BRONZE cup\ndull TIN spoon\n",
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["converged"], true, "{parsed}");

    // Column-stochasticity of every column with outflow (goal + criteria +
    // alternatives).
    let s = parsed["supermatrix"].as_array().unwrap();
    let n = s.len();
    for col in 0..n {
        let sum: f64 = (0..n)
            .map(|row| s[row].as_array().unwrap()[col].as_f64().unwrap())
            .sum();
        assert!(
            (sum - 1.0).abs() < 1e-9 || sum.abs() < 1e-12,
            "column {col} must be stochastic or empty: {sum}"
        );
    }

    // Weights are distributions.
    let criteria = parsed["criteria"].as_array().unwrap();
    let direct: f64 = criteria.iter().map(|c| c["direct_weight"].as_f64().unwrap()).sum();
    let limiting: f64 = criteria
        .iter()
        .map(|c| c["limiting_weight"].as_f64().unwrap())
        .sum();
    assert!((direct - 1.0).abs() < 1e-9, "direct sums to 1: {direct}");
    assert!((limiting - 1.0).abs() < 1e-6, "limiting sums to 1: {limiting}");
    let alts = parsed["alternatives"].as_array().unwrap();
    let alt_sum: f64 = alts
        .iter()
        .map(|a| a["limiting_priority"].as_f64().unwrap())
        .sum();
    assert!((alt_sum - 1.0).abs() < 1e-6, "priorities sum to 1: {alt_sum}");

    // Dominance ordering on the metal scale, in both clusters.
    assert_eq!(criteria[0]["name"], "gold");
    assert!(
        criteria[0]["limiting_weight"].as_f64().unwrap()
            > criteria[2]["limiting_weight"].as_f64().unwrap(),
        "gold criterion must dominate tin in the limit: {parsed}"
    );
    assert_eq!(
        alts[0]["id"], "item-0000",
        "gold ring must lead the limiting priorities: {parsed}"
    );
    // The network correction is a measured quantity present per criterion.
    for c in criteria {
        assert!(
            c["network_delta"].as_f64().is_some(),
            "delta must be reported: {c}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two DISTINCT content-blind position pathologies that counterbalancing
/// cannot tell apart — and the orbit transform separates exactly:
///
/// - NAME-AFFINITY: always outputs the token "A", whatever the question.
///   Under the less-wording, naming A means A loses, so the pulled-back
///   orbit is m(g) = (−1)^{s+p+w}·ln 2 — the TRIPLE character.
/// - SLOT-FAVORITISM: always ranks the slot-A entity higher, answering
///   each wording coherently (less-wording: names B as lower). Orbit
///   m(g) = (−1)^{s+p}·ln 2 — the order·polarity character.
///
/// (The first draft of this test assumed both were the same pathology;
/// the transform refuted that — the derivation note is in orbit.rs.)
#[derive(Clone, Copy)]
struct NameAffinityJudge;

impl Respond for NameAffinityJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let parsed: serde_json::Value = serde_json::from_slice(&request.body).unwrap_or_default();
        let system = parsed["messages"]
            .as_array()
            .and_then(|m| {
                m.iter()
                    .find(|x| x["role"] == "system")
                    .and_then(|x| x["content"].as_str())
            })
            .unwrap_or("");
        let content = if system.contains("LESS of that attribute") {
            r#"{"lower_ranked":"A","ratio":2.0,"confidence":0.9}"#
        } else {
            r#"{"higher_ranked":"A","ratio":2.0,"confidence":0.9}"#
        };
        ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "message": { "content": content.to_string() }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 10 }
        }))
    }
}

#[derive(Clone, Copy)]
struct SlotFavoritismJudge;

impl Respond for SlotFavoritismJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let parsed: serde_json::Value = serde_json::from_slice(&request.body).unwrap_or_default();
        let system = parsed["messages"]
            .as_array()
            .and_then(|m| {
                m.iter()
                    .find(|x| x["role"] == "system")
                    .and_then(|x| x["content"].as_str())
            })
            .unwrap_or("");
        let content = if system.contains("LESS of that attribute") {
            // Favoring slot A coherently: B is the lower one.
            r#"{"lower_ranked":"B","ratio":2.0,"confidence":0.9}"#
        } else {
            r#"{"higher_ranked":"A","ratio":2.0,"confidence":0.9}"#
        };
        ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "message": { "content": content.to_string() }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 10 }
        }))
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn orbit_transform_puts_all_energy_in_belief_for_a_perfect_judge() {
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
            "--orbit",
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
    let coeffs = parsed["coefficients"].as_array().unwrap();
    let belief = coeffs[0].as_f64().unwrap();
    assert!(
        (belief - 3.9f64.ln()).abs() < 1e-9,
        "belief = orbit mean = ln 3.9: {belief}"
    );
    for (k, coeff) in coeffs.iter().enumerate().skip(1) {
        assert!(
            coeff.as_f64().unwrap().abs() < 1e-9,
            "perfect judge has zero bias spectrum: coeff {k} = {coeff}"
        );
    }
    assert!((parsed["coherence"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    assert!(parsed["parseval_residual"].as_f64().unwrap() < 1e-12);
    assert_eq!(parsed["comparisons"], 8);
}

async fn orbit_coefficients<R: Respond + 'static>(judge: R) -> Vec<f64> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(judge)
        .mount(&server)
        .await;
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
            "--orbit",
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
    assert!(parsed["parseval_residual"].as_f64().unwrap() < 1e-12);
    parsed["coefficients"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn orbit_transform_separates_the_two_position_pathologies() {
    // Both judges are pure position bias to a counterbalance probe. The
    // transform separates them: name-affinity excites exactly the triple
    // character (index 7 = s·p·w); slot-favoritism exactly order·polarity
    // (index 3 = s·p). Belief is zero for both — they believe nothing.
    let name = orbit_coefficients(NameAffinityJudge).await;
    let ln2 = 2.0f64.ln();
    for (k, c) in name.iter().enumerate() {
        if k == 7 {
            assert!(
                (c - ln2).abs() < 1e-9,
                "name-affinity is the triple character: coeff[7] = {c}"
            );
        } else {
            assert!(c.abs() < 1e-9, "coeff[{k}] must vanish: {c}");
        }
    }

    let slot = orbit_coefficients(SlotFavoritismJudge).await;
    for (k, c) in slot.iter().enumerate() {
        if k == 3 {
            assert!(
                (c - ln2).abs() < 1e-9,
                "slot-favoritism is order·polarity: coeff[3] = {c}"
            );
        } else {
            assert!(c.abs() < 1e-9, "coeff[{k}] must vanish: {c}");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn judge_wordings_agree_for_a_coherent_judge() {
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
            "--wordings",
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
    assert_eq!(parsed["sign_consistent"], true, "{parsed}");
    let d = parsed["max_disagreement_nats"].as_f64().unwrap();
    assert!(
        d < 1e-4,
        "times-more, fraction, and times-less must recover one log-ratio \
         (tolerance = the mock's 6-decimal fraction rounding): {d}"
    );
    assert_eq!(parsed["comparisons"], 6);
}

#[tokio::test(flavor = "multi_thread")]
async fn judge_wordings_catch_an_inversion_blind_judge() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(InversionBlindJudge)
        .mount(&server)
        .await;
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
            "--wordings",
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
    assert_eq!(
        parsed["sign_consistent"], false,
        "the less-wording sign flip must surface: {parsed}"
    );
    let d = parsed["max_disagreement_nats"].as_f64().unwrap();
    assert!(d > 2.0, "sign flip at ratio 3.9 = 2·ln(3.9) disagreement: {d}");
}

#[tokio::test(flavor = "multi_thread")]
async fn judge_sweep_fits_a_linear_responder() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(LinearSycophantJudge)
        .mount(&server)
        .await;
    let output = run_cardinal(
        &server.uri(),
        &[
            "judge",
            "plain IRON bar",
            "plain IRON rod",
            "--by",
            "shininess",
            "--model",
            "test/judge",
            "--no-cache",
            "--sweep",
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
    let chi = parsed["chi_slope"].as_f64().unwrap();
    let r2 = parsed["linearity_r2"].as_f64().unwrap();
    assert!((chi - 0.3).abs() < 0.02, "slope must recover 0.3: {chi}");
    assert!(r2 > 0.98, "pure linear responder must fit: {r2}");
    assert_eq!(parsed["comparisons"], 14);
}

#[tokio::test(flavor = "multi_thread")]
async fn judge_sweep_exposes_a_threshold_sycophant() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ThresholdSycophantJudge)
        .mount(&server)
        .await;
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
            "--sweep",
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
    let chi = parsed["chi_slope"].as_f64().unwrap();
    let r2 = parsed["linearity_r2"].as_f64().unwrap();
    // The judge holds firm at fields −2..+3 and folds only at −3: nonzero
    // slope with a BAD linear fit — the sweep's signature of a threshold.
    assert!(chi > 0.1, "folding at one end pulls the slope up: {chi}");
    assert!(
        r2 < 0.9,
        "a step is not a line — R² must expose it: {r2}"
    );
    assert_eq!(parsed["belief_survives_sweep"], false, "{parsed}");
}

#[tokio::test(flavor = "multi_thread")]
async fn judge_spin_catches_a_sycophant() {
    // The sycophant votes for whoever the requester leans toward: the probe
    // must report positive susceptibility and a belief that does NOT survive.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(SycophantJudge)
        .mount(&server)
        .await;
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
            "--spin",
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
    assert_eq!(parsed["belief_survives_spin"], false, "{parsed}");
    let chi = parsed["susceptibility_nats"].as_f64().unwrap();
    assert!(
        chi > 1.0,
        "sycophant must show large positive susceptibility: {chi}"
    );
}
