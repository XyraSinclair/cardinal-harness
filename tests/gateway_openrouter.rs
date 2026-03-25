use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cardinal_harness::gateway::openrouter::OpenRouterAdapter;
use cardinal_harness::gateway::usage::{CallStatus, ProviderCallRecord, UsageSink};
use cardinal_harness::gateway::{
    Attribution, ChatModel, ChatRequest, FinishReason, GatewayConfig, Message, NoopUsageSink,
    ProviderError, ProviderGateway, RateLimitSource, ReasoningConfig,
};
use serde_json::json;
use uuid::Uuid;
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

#[tokio::test]
async fn openrouter_parses_success_content_and_usage() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": { "content": "hello", "reasoning": "think first" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "prompt_tokens_details": {
                    "cached_tokens": 4,
                    "cache_write_tokens": 6
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 7
                },
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
    assert_eq!(resp.reasoning.as_deref(), Some("think first"));
    assert_eq!(resp.reasoning_tokens, Some(7));
    assert_eq!(resp.finish_reason, FinishReason::Stop);
    assert_eq!(resp.input_tokens, 10);
    assert_eq!(resp.output_tokens, 20);
    assert_eq!(resp.cache_read_tokens, Some(4));
    assert_eq!(resp.cache_write_tokens, Some(6));
    assert_eq!(
        resp.cost_nanodollars,
        cardinal_harness::gateway::chat_cost("openai/gpt-5-mini", 10, 20)
    );
    assert_eq!(resp.upstream_cost_nanodollars, Some(1_000));
}

#[tokio::test]
async fn openrouter_serializes_reasoning_request_config() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains("\"reasoning\":{\"effort\":\"low\"}"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": { "content": "hello" },
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
    )
    .reasoning(ReasoningConfig::low());

    adapter.chat(&req).await.unwrap();
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
async fn openrouter_requests_and_parses_logprobs() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains("\"logprobs\":true"))
        .and(body_string_contains("\"top_logprobs\":5"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": { "content": "hello" },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [{
                        "token": "2.1",
                        "logprob": -0.2,
                        "top_logprobs": [
                            { "token": "2.1", "logprob": -0.2 },
                            { "token": "1.5", "logprob": -1.3 }
                        ]
                    }]
                }
            }],
            "usage": { "prompt_tokens": 3, "completion_tokens": 2 }
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
    .with_logprobs(5);

    let resp = adapter.chat(&req).await.unwrap();
    let logprobs = resp.output_logprobs.expect("expected output logprobs");
    assert_eq!(logprobs.len(), 1);
    assert_eq!(logprobs[0].token, "2.1");
    assert_eq!(logprobs[0].top_alternatives.len(), 2);
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

#[derive(Default)]
struct RecordingUsageSink {
    records: Mutex<Vec<ProviderCallRecord>>,
}

impl RecordingUsageSink {
    fn snapshot(&self) -> Vec<ProviderCallRecord> {
        self.records.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl UsageSink for RecordingUsageSink {
    async fn record(&self, record: ProviderCallRecord) {
        self.records.lock().unwrap().push(record);
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

#[tokio::test]
async fn provider_gateway_records_usage_for_retry_then_success() {
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
        "usage": { "prompt_tokens": 3, "completion_tokens": 5 }
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
    let sink = Arc::new(RecordingUsageSink::default());
    let gateway = ProviderGateway::with_config(
        adapter,
        sink.clone(),
        GatewayConfig {
            max_retries: 1,
            retry_base_delay: Duration::from_millis(0),
        },
    );

    let user_id = Uuid::new_v4();
    let job_id = Uuid::new_v4();
    let req = ChatRequest::new(
        ChatModel::openrouter("openai/gpt-5-mini"),
        vec![Message::user("hi")],
        Attribution::new("test::usage")
            .with_user(user_id)
            .with_job(job_id),
    );

    let resp = gateway.chat(req).await.unwrap();
    assert_eq!(resp.content, "ok");

    let records = sink.snapshot();
    assert_eq!(records.len(), 2);

    let first = &records[0];
    assert_eq!(first.status, CallStatus::Error);
    assert_eq!(first.error_code.as_deref(), Some("provider_error"));
    assert_eq!(first.provider, "openrouter");
    assert_eq!(first.endpoint, "chat/completions");
    assert_eq!(first.model, "openai/gpt-5-mini");
    assert_eq!(first.caller, "test::usage");
    assert_eq!(first.user_id, Some(user_id));
    assert_eq!(first.job_id, Some(job_id));
    assert_eq!(first.input_tokens, 0);
    assert_eq!(first.output_tokens, 0);

    let second = &records[1];
    assert_eq!(second.status, CallStatus::Success);
    assert_eq!(second.error_code, None);
    assert_eq!(second.provider, "openrouter");
    assert_eq!(second.endpoint, "chat/completions");
    assert_eq!(second.model, "openai/gpt-5-mini");
    assert_eq!(second.caller, "test::usage");
    assert_eq!(second.user_id, Some(user_id));
    assert_eq!(second.job_id, Some(job_id));
    assert_eq!(second.input_tokens, 3);
    assert_eq!(second.output_tokens, 5);
    assert_eq!(
        second.cost_nanodollars,
        cardinal_harness::gateway::chat_cost("openai/gpt-5-mini", 3, 5)
    );
}

#[tokio::test]
async fn provider_gateway_records_usage_for_retryable_failure() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({
            "error": { "message": "still broken", "code": "internal" }
        })))
        .mount(&server)
        .await;

    let adapter =
        OpenRouterAdapter::with_config("sk-test", server.uri(), Duration::from_secs(5), None, None)
            .unwrap();
    let sink = Arc::new(RecordingUsageSink::default());
    let gateway = ProviderGateway::with_config(
        adapter,
        sink.clone(),
        GatewayConfig {
            max_retries: 1,
            retry_base_delay: Duration::from_millis(0),
        },
    );

    let err = gateway
        .chat(ChatRequest::new(
            ChatModel::openrouter("openai/gpt-5-mini"),
            vec![Message::user("hi")],
            Attribution::new("test::failure"),
        ))
        .await
        .unwrap_err();

    match err {
        ProviderError::Provider { retryable, .. } => assert!(retryable),
        other => panic!("expected retryable provider error, got {other:?}"),
    }

    let records = sink.snapshot();
    assert_eq!(records.len(), 2);
    assert!(records
        .iter()
        .all(|record| record.status == CallStatus::Error));
    assert!(records
        .iter()
        .all(|record| record.error_code.as_deref() == Some("provider_error")));

    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 2);
}
