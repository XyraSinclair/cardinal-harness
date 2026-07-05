//! The benchmark validates ITSELF: scripted judges with known pathologies
//! must land where the dimensions say they should. A benchmark that cannot
//! separate an oracle from a constant, a position-biased judge, a
//! sycophant, and a cyclic judge is not measuring judgement.

use std::sync::Arc;
use std::time::Duration;

use cardinal_harness::gateway::openrouter::OpenRouterAdapter;
use cardinal_harness::gateway::{GatewayConfig, NoopUsageSink, ProviderGateway};
use cardinal_harness::rerank::{run_judge_bench, JudgeBenchOptions, JudgeBenchReport, CORPUS};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

fn extract_between<'a>(s: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let i = s.find(start)? + start.len();
    let rest = &s[i..];
    let j = rest.find(end)?;
    Some(&rest[..j])
}

/// Latent depths for the corpus, index-aligned. The mock judges' shared
/// ground truth — the benchmark itself never sees these.
const DEPTHS: [f64; 8] = [4.0, 3.6, 3.2, 2.6, 1.8, 1.2, 0.6, 0.1];

fn corpus_index(text: &str) -> Option<usize> {
    CORPUS.iter().position(|&t| t == text.trim())
}

fn user_content(request: &Request) -> String {
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap_or_default();
    body["messages"]
        .as_array()
        .and_then(|m| {
            m.iter()
                .find(|x| x["role"] == "user")
                .and_then(|x| x["content"].as_str())
        })
        .unwrap_or("")
        .to_string()
}

fn contexts(user: &str) -> (String, String) {
    let a = extract_between(user, "<entity_A_context>", "</entity_A_context>")
        .unwrap_or("")
        .trim()
        .to_string();
    let b = extract_between(user, "<entity_B_context>", "</entity_B_context>")
        .unwrap_or("")
        .trim()
        .to_string();
    (a, b)
}

fn answer(higher: &str, ratio: f64) -> ResponseTemplate {
    let content = format!(r#"{{"higher_ranked":"{higher}","ratio":{ratio},"confidence":0.9}}"#);
    ResponseTemplate::new(200).set_body_json(json!({
        "choices": [{
            "message": { "content": content },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 10 }
    }))
}

/// Ideal judge: reads content, respects polarity and paraphrase, ignores
/// framing, log-additive ratios, exact ties on identical items.
#[derive(Clone, Copy)]
struct OracleJudge;

impl Respond for OracleJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let user = user_content(request);
        let (a, b) = contexts(&user);
        if a == b {
            return answer("A", 1.0);
        }
        let (Some(ia), Some(ib)) = (corpus_index(&a), corpus_index(&b)) else {
            return answer("A", 1.0);
        };
        let reversed = user.contains("shallowness");
        let mut d = DEPTHS[ia] - DEPTHS[ib];
        if reversed {
            d = -d;
        }
        let higher = if d >= 0.0 { "A" } else { "B" };
        // exp(|d|/2): log-additive in the latent, so cycle sums vanish.
        answer(higher, (d.abs() / 2.0).exp())
    }
}

/// Follows the requester's leaning whenever one is present; oracle-like
/// otherwise.
#[derive(Clone, Copy)]
struct SycophantJudge;

impl Respond for SycophantJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let user = user_content(request);
        let (a, b) = contexts(&user);
        if let Some(excerpt) = extract_between(&user, "the one beginning «", "» is clearly stronger")
        {
            if a.starts_with(excerpt) {
                return answer("A", 3.9);
            }
            if b.starts_with(excerpt) {
                return answer("B", 3.9);
            }
        }
        OracleJudge.respond(request)
    }
}

/// Always ties: perfectly consistent, zero discrimination.
#[derive(Clone, Copy)]
struct ConstantJudge;

impl Respond for ConstantJudge {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        answer("A", 1.0)
    }
}

/// Always prefers whatever sits in slot A.
#[derive(Clone, Copy)]
struct PositionJudge;

impl Respond for PositionJudge {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        answer("A", 2.0)
    }
}

/// Rock-paper-scissors around the corpus ring: each item beats the next
/// three, ties at distance four. Order-consistent and antisymmetric, but
/// globally cyclic — pure structural frustration.
#[derive(Clone, Copy)]
struct CyclicJudge;

impl Respond for CyclicJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let user = user_content(request);
        let (a, b) = contexts(&user);
        if a == b {
            return answer("A", 1.0);
        }
        let (Some(ia), Some(ib)) = (corpus_index(&a), corpus_index(&b)) else {
            return answer("A", 1.0);
        };
        let gap = (ib + 8 - ia) % 8;
        match gap {
            1..=3 => answer("A", 3.9),
            4 => answer("A", 1.0),
            _ => answer("B", 3.9),
        }
    }
}

async fn bench_with<R: Respond + 'static>(judge: R) -> JudgeBenchReport {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(judge)
        .mount(&server)
        .await;
    let adapter =
        OpenRouterAdapter::with_config("sk-test", server.uri(), Duration::from_secs(5), None, None)
            .unwrap();
    let gateway = ProviderGateway::with_config(
        adapter,
        Arc::new(NoopUsageSink),
        GatewayConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(0),
        },
    );
    run_judge_bench(
        &gateway,
        None,
        JudgeBenchOptions {
            model: "test/judge".to_string(),
            ..Default::default()
        },
    )
    .await
    .expect("bench run")
}

#[tokio::test(flavor = "multi_thread")]
async fn oracle_judge_scores_high_on_every_dimension() {
    let r = bench_with(OracleJudge).await;
    assert_eq!(r.order_flip.value, Some(0.0), "{}", diag(&r));
    assert!(r.order_residual.value.unwrap() < 1e-9, "{}", diag(&r));
    assert!(
        r.frustration.value.unwrap() < 0.05,
        "log-additive oracle must have ~zero curl: {}",
        diag(&r)
    );
    assert_eq!(r.spin_survival.value, Some(1.0), "{}", diag(&r));
    assert!(r.polarity.value.unwrap() < -0.9, "{}", diag(&r));
    assert!(r.paraphrase.value.unwrap() > 0.9, "{}", diag(&r));
    assert!(r.null_bias.value.unwrap() < 1e-9, "{}", diag(&r));
    assert!(r.signal.value.unwrap() > 0.3, "{}", diag(&r));
    assert!(
        r.judge_score.unwrap() > 0.5,
        "oracle must lead the board: {}",
        diag(&r)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn constant_judge_is_zeroed_by_the_signal_axis() {
    let r = bench_with(ConstantJudge).await;
    assert!(r.signal.value.unwrap() < 1e-12, "{}", diag(&r));
    // Perfect consistency everywhere it can be measured...
    assert!(r.null_bias.value.unwrap() < 1e-12, "{}", diag(&r));
    // ...and still (near) zero headline: consistency cannot hide no signal.
    assert!(
        r.judge_score.unwrap() < 0.05,
        "constant judge must not score: {}",
        diag(&r)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn position_judge_is_caught_by_order_swap_and_null_pairs() {
    let r = bench_with(PositionJudge).await;
    assert_eq!(
        r.order_flip.value,
        Some(1.0),
        "every decisive pair must flip: {}",
        diag(&r)
    );
    // Fused signal cancels to zero: position preference is not discrimination.
    assert!(r.signal.value.unwrap() < 1e-12, "{}", diag(&r));
    // Null pairs expose the same bias directly, in nats.
    assert!(
        (r.null_bias.value.unwrap() - 2.0f64.ln()).abs() < 1e-9,
        "{}",
        diag(&r)
    );
    assert!(r.judge_score.unwrap() < 0.05, "{}", diag(&r));
}

#[tokio::test(flavor = "multi_thread")]
async fn sycophant_judge_is_caught_by_spin_and_only_spin() {
    let oracle = bench_with(OracleJudge).await;
    let r = bench_with(SycophantJudge).await;
    // Without a leaning asker it is the oracle: consistency axes stay clean.
    assert_eq!(r.order_flip.value, Some(0.0), "{}", diag(&r));
    assert!(r.polarity.value.unwrap() < -0.9, "{}", diag(&r));
    // With one, the belief follows the framing.
    assert_eq!(
        r.spin_survival.value,
        Some(0.0),
        "no clear belief survives: {}",
        diag(&r)
    );
    assert!(
        r.susceptibility.value.unwrap() > 1.0,
        "chi must be large: {}",
        diag(&r)
    );
    assert!(
        r.judge_score.unwrap() < oracle.judge_score.unwrap(),
        "sycophancy must cost rank: {} vs {}",
        diag(&r),
        diag(&oracle)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cyclic_judge_is_caught_by_frustration_and_only_frustration() {
    let oracle = bench_with(OracleJudge).await;
    let r = bench_with(CyclicJudge).await;
    // Locally impeccable: no order flips, no reciprocal residual.
    assert_eq!(r.order_flip.value, Some(0.0), "{}", diag(&r));
    assert!(r.order_residual.value.unwrap() < 1e-9, "{}", diag(&r));
    // Globally incoherent: the ring preference cannot be scored.
    assert!(
        r.frustration.value.unwrap() > 0.3,
        "planted cycles must show as curl: {}",
        diag(&r)
    );
    assert!(
        r.judge_score.unwrap() < oracle.judge_score.unwrap(),
        "cycles must cost rank: {} vs {}",
        diag(&r),
        diag(&oracle)
    );
}

fn diag(r: &JudgeBenchReport) -> String {
    format!(
        "judge_score={:?} signal={:?} flip={:?} residual={:?} curl={:?} spin={:?} chi={:?} pol={:?} para={:?} null={:?}",
        r.judge_score,
        r.signal.value,
        r.order_flip.value,
        r.order_residual.value,
        r.frustration.value,
        r.spin_survival.value,
        r.susceptibility.value,
        r.polarity.value,
        r.paraphrase.value,
        r.null_bias.value
    )
}
