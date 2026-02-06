use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cardinal_harness::gateway::openrouter::{ChatProvider, OpenRouterAdapter};
use cardinal_harness::gateway::{
    Attribution, ChatModel, ChatRequest, FinishReason, GatewayConfig, Message, NoopUsageSink,
    ProviderError, ProviderGateway, RateLimitSource,
};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

#[tokio::test]
async fn openrouter_parses_success_content_and_usage() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": { "content": "hello" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cost_details": { "upstream_inference_cost": 0.000001 }
            }
        })))
        .mount(&server)
        .await;

    let adapter =
        OpenRouterAdapter::with_config("sk-test", server.uri(), Duration::from_secs(5), None, None)
            .unwrap();

    let req = ChatRequest::new(
        ChatModel::openrouter("openai/gpt-5-mini"),
        vec![Message::user("hi")],
        Attribution::new("test"),
    );

    let resp = adapter.chat(&req).await.unwrap();
    assert_eq!(resp.content, "hello");
    assert_eq!(resp.finish_reason, FinishReason::Stop);
    assert_eq!(resp.input_tokens, 10);
    assert_eq!(resp.output_tokens, 20);
    assert_eq!(
        resp.cost_nanodollars,
        cardinal_harness::gateway::chat_cost("openai/gpt-5-mini", 10, 20)
    );
    assert_eq!(resp.upstream_cost_nanodollars, Some(1_000));
}

#[tokio::test]
async fn openrouter_falls_back_to_tool_call_arguments_when_content_empty() {
    let server = MockServer::start().await;
    let args = r#"{"foo": 1}"#;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"arguments": args}}]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let adapter =
        OpenRouterAdapter::with_config("sk-test", server.uri(), Duration::from_secs(5), None, None)
            .unwrap();

    let req = ChatRequest::new(
        ChatModel::openrouter("openai/gpt-5-mini"),
        vec![Message::user("hi")],
        Attribution::new("test"),
    )
    .json();

    let resp = adapter.chat(&req).await.unwrap();
    assert_eq!(resp.content, args);
    assert_eq!(resp.finish_reason, FinishReason::ToolCalls);
}

#[tokio::test]
async fn openrouter_detects_refusal_from_content() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": { "content": "I cannot comply with that request." },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let adapter =
        OpenRouterAdapter::with_config("sk-test", server.uri(), Duration::from_secs(5), None, None)
            .unwrap();

    let req = ChatRequest::new(
        ChatModel::openrouter("openai/gpt-5-mini"),
        vec![Message::user("hi")],
        Attribution::new("test"),
    );

    let err = adapter.chat(&req).await.unwrap_err();
    assert!(matches!(err, ProviderError::Refused { .. }));
}

#[tokio::test]
async fn openrouter_classifies_http_429_as_remote_rate_limit_and_keeps_context() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("x-request-id", "abc123")
                .set_body_json(json!({
                    "error": { "message": "rate limited", "code": "rate_limit_exceeded" }
                })),
        )
        .mount(&server)
        .await;

    let adapter =
        OpenRouterAdapter::with_config("sk-test", server.uri(), Duration::from_secs(5), None, None)
            .unwrap();

    let req = ChatRequest::new(
        ChatModel::openrouter("openai/gpt-5-mini"),
        vec![Message::user("hi")],
        Attribution::new("test"),
    );

    let err = adapter.chat(&req).await.unwrap_err();
    match err {
        ProviderError::RateLimited {
            retry_after,
            limit_source,
            context,
        } => {
            assert_eq!(limit_source, RateLimitSource::Remote);
            assert_eq!(retry_after, Duration::from_secs(60));
            let ctx = context.expect("expected error context");
            assert_eq!(ctx.http_status, Some(429));
            assert_eq!(ctx.provider_code.as_deref(), Some("rate_limit_exceeded"));
            assert_eq!(ctx.request_id.as_deref(), Some("abc123"));
        }
        other => panic!("expected RateLimited, got {other:?}"),
    }
}

#[derive(Clone)]
struct FlipResponder {
    calls: Arc<AtomicUsize>,
    first: ResponseTemplate,
    second: ResponseTemplate,
}

impl Respond for FlipResponder {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            self.first.clone()
        } else {
            self.second.clone()
        }
    }
}

#[tokio::test]
async fn provider_gateway_retries_on_retryable_errors_and_succeeds() {
    let server = MockServer::start().await;

    let calls = Arc::new(AtomicUsize::new(0));
    let first = ResponseTemplate::new(500).set_body_json(json!({
        "error": { "message": "transient error", "code": "internal" }
    }));
    let second = ResponseTemplate::new(200).set_body_json(json!({
        "choices": [{
            "message": { "content": "ok" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1 }
    }));

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(FlipResponder {
            calls,
            first,
            second,
        })
        .mount(&server)
        .await;

    let adapter =
        OpenRouterAdapter::with_config("sk-test", server.uri(), Duration::from_secs(5), None, None)
            .unwrap();
    let gateway = ProviderGateway::with_config(
        adapter,
        Arc::new(NoopUsageSink),
        GatewayConfig {
            max_retries: 1,
            retry_base_delay: Duration::from_millis(0),
        },
    );

    let req = ChatRequest::new(
        ChatModel::openrouter("openai/gpt-5-mini"),
        vec![Message::user("hi")],
        Attribution::new("test"),
    );

    let resp = gateway.chat(req).await.unwrap();
    assert_eq!(resp.content, "ok");

    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 2);
}
