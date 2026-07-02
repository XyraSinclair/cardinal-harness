//! End-to-end test for explain_ranking with a deterministic wiremock judge.

use std::sync::Arc;
use std::time::Duration;

use cardinal_harness::gateway::openrouter::OpenRouterAdapter;
use cardinal_harness::gateway::{Attribution, GatewayConfig, NoopUsageSink, ProviderGateway};
use cardinal_harness::rerank::{explain_ranking, ExplainOptions, RerankDocument, RerankExecution};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Knows shininess (GOLD > SILVER > BRONZE > TIN); knows nothing about any
/// other attribute (answers an uninformative near-tie).
#[derive(Clone, Copy)]
struct ShininessOnlyJudge;

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

impl Respond for ShininessOnlyJudge {
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

        let content = if user_content.contains("shininess") {
            let a = extract_between(&user_content, "<entity_A_context>", "</entity_A_context>")
                .map(metal_score)
                .unwrap_or(0);
            let b = extract_between(&user_content, "<entity_B_context>", "</entity_B_context>")
                .map(metal_score)
                .unwrap_or(0);
            let (higher, ratio) = if a >= b {
                ("A", if (a - b).abs() >= 2 { 3.9 } else { 1.5 })
            } else {
                ("B", if (b - a).abs() >= 2 { 3.9 } else { 1.5 })
            };
            format!(r#"{{"higher_ranked":"{higher}","ratio":{ratio},"confidence":0.9}}"#)
        } else {
            // No idea about this attribute: near-tie, low confidence.
            r#"{"higher_ranked":"A","ratio":1.05,"confidence":0.2}"#.to_string()
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

#[tokio::test]
async fn explain_identifies_the_attribute_behind_a_ranking() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ShininessOnlyJudge)
        .mount(&server)
        .await;

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
    let execution = RerankExecution::new(gateway, Attribution::new("test::explain"));

    // The believed ranking IS the shininess order, best first.
    let documents = vec![
        RerankDocument {
            id: "gold".into(),
            text: "shiny GOLD ring".into(),
        },
        RerankDocument {
            id: "silver".into(),
            text: "bright SILVER fork".into(),
        },
        RerankDocument {
            id: "bronze".into(),
            text: "old BRONZE coin".into(),
        },
        RerankDocument {
            id: "tin".into(),
            text: "dull TIN spoon".into(),
        },
    ];

    let explanation = explain_ranking(
        documents,
        vec!["shininess".into(), "age of the object".into()],
        execution,
        ExplainOptions {
            model: Some("test/judge".into()),
            comparison_budget: Some(80),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let shininess = &explanation.attributes[0];
    let age = &explanation.attributes[1];
    assert_eq!(shininess.prompt, "shininess");

    // The judge tracks shininess perfectly: it alone reconstructs the order.
    assert!(
        shininess.spearman_alone.unwrap() > 0.9,
        "shininess: {shininess:?}"
    );
    // The judge is uninformative about age; it must not dominate the fit.
    assert!(
        shininess.fitted_weight > age.fitted_weight,
        "weights: shininess {shininess:?} vs age {age:?}"
    );
    assert!(
        explanation.combined_spearman.unwrap() > 0.9,
        "combined: {:?}",
        explanation.combined_spearman
    );
    assert!(explanation.meta.comparisons_used > 0);
}
