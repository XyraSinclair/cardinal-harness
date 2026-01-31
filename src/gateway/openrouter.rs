//! OpenRouter adapter for chat completions.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

use super::error::{ErrorContext, ProviderError};
use super::pricing::chat_cost;
use super::types::*;

// =============================================================================
// TRAIT
// =============================================================================

/// Trait for chat completion providers.
#[async_trait]
pub trait ChatProvider: Send + Sync {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError>;
}

// =============================================================================
// OPENROUTER ADAPTER
// =============================================================================

/// Maximum allowed response content length (1MB).
const MAX_RESPONSE_LEN: usize = 1_024 * 1_024;

/// Maximum allowed input characters (~125k tokens).
const MAX_INPUT_CHARS: usize = 500_000;

/// OpenRouter API adapter for chat completions.
#[derive(Debug, Clone)]
pub struct OpenRouterAdapter {
    client: reqwest::Client,
    base_url: String,
}

impl OpenRouterAdapter {
    /// Create from API key.
    pub fn new(api_key: impl Into<String>) -> Result<Self, ProviderError> {
        Self::with_config(
            api_key,
            "https://openrouter.ai/api/v1",
            Duration::from_secs(120),
            None,
            None,
        )
    }

    /// Create from environment variable.
    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .map_err(|_| ProviderError::config("OPENROUTER_API_KEY not set"))?;

        let base_url = std::env::var("OPENROUTER_BASE_URL")
            .unwrap_or_else(|_| "https://openrouter.ai/api/v1".into());

        let timeout = std::env::var("OPENROUTER_TIMEOUT_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(120));

        let referer = std::env::var("OPENROUTER_REFERER").ok();
        let app_title = std::env::var("OPENROUTER_APP_TITLE").ok();

        Self::with_config(api_key, base_url, timeout, referer, app_title)
    }

    /// Create with custom configuration.
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        timeout: Duration,
        referer: Option<String>,
        app_title: Option<String>,
    ) -> Result<Self, ProviderError> {
        let api_key = api_key.into();
        let base_url = base_url.into();

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let auth_value = HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|_| ProviderError::config("Invalid API key format"))?;
        headers.insert(AUTHORIZATION, auth_value);

        if let Some(ref r) = referer {
            if let Ok(v) = HeaderValue::from_str(r) {
                headers.insert("HTTP-Referer", v);
            }
        }

        if let Some(ref t) = app_title {
            if let Ok(v) = HeaderValue::from_str(t) {
                headers.insert("X-Title", v);
            }
        }

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .default_headers(headers)
            .gzip(true)
            .build()
            .map_err(|e| ProviderError::config(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self { client, base_url })
    }

    fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Extract request ID from response headers.
    fn extract_request_id(headers: &reqwest::header::HeaderMap) -> Option<String> {
        headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
    }

    /// Check if message indicates a refusal.
    fn is_refusal(msg: &str) -> bool {
        let l = msg.trim_start().to_lowercase();
        let first_line = l.lines().next().unwrap_or("");

        const PREFIXES: &[&str] = &[
            "refus",
            "i cannot",
            "i can't",
            "i won't",
            "i will not",
            "i am unable to",
            "i'm unable to",
            "unable to comply",
            "unable to assist",
            "unable to help",
            "unable to provide",
        ];

        PREFIXES.iter().any(|p| first_line.starts_with(p)) || l.contains("request was refused")
    }
}

// =============================================================================
// API TYPES
// =============================================================================

#[derive(Serialize)]
struct ChatApiRequest<'a> {
    model: &'a str,
    messages: &'a [ApiMessage],
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: String,
}

impl From<&Message> for ApiMessage {
    fn from(m: &Message) -> Self {
        Self {
            role: match m.role {
                Role::System => "system".to_string(),
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
            },
            content: m.content.clone(),
        }
    }
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: &'static str,
}

#[derive(Deserialize)]
struct ChatApiResponse {
    choices: Option<Vec<Choice>>,
    usage: Option<Usage>,
    error: Option<ApiError>,
}

#[derive(Deserialize)]
struct Choice {
    message: Option<ChoiceMessage>,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize)]
struct ToolCall {
    function: Option<ToolFunction>,
}

#[derive(Deserialize)]
struct ToolFunction {
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    #[serde(default)]
    cost_details: Option<CostDetails>,
}

#[derive(Deserialize)]
struct CostDetails {
    upstream_inference_cost: Option<f64>,
}

#[derive(Deserialize)]
struct ApiError {
    message: Option<String>,
    code: Option<String>,
}

// =============================================================================
// CHAT PROVIDER IMPL
// =============================================================================

#[async_trait]
impl ChatProvider for OpenRouterAdapter {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        // Validate input size
        let total_chars: usize = req.messages.iter().map(|m| m.content.len()).sum();

        if total_chars > MAX_INPUT_CHARS {
            return Err(ProviderError::invalid_request(format!(
                "Input too large: {total_chars} chars (max {MAX_INPUT_CHARS})"
            )));
        }

        let start = Instant::now();

        let messages: Vec<ApiMessage> = req.messages.iter().map(ApiMessage::from).collect();

        let api_req = ChatApiRequest {
            model: req.model.model_id(),
            messages: &messages,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            response_format: if req.json_mode {
                Some(ResponseFormat {
                    format_type: "json_object",
                })
            } else {
                None
            },
        };

        let mut response = self
            .client
            .post(self.chat_url())
            .json(&api_req)
            .send()
            .await?;

        let status = response.status();
        let request_id = Self::extract_request_id(response.headers());

        // Stream response to enforce size limit
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            let new_len = bytes.len() + chunk.len();
            if new_len > MAX_RESPONSE_LEN {
                return Err(ProviderError::provider(
                    "openrouter",
                    format!("Response too large: {new_len} bytes"),
                    false,
                ));
            }
            bytes.extend_from_slice(&chunk);
        }

        let body = String::from_utf8_lossy(&bytes).to_string();

        // Build error context
        let ctx = ErrorContext::new().with_status(status.as_u16());
        let ctx = if let Some(id) = &request_id {
            ctx.with_request_id(id)
        } else {
            ctx
        };

        if !status.is_success() {
            // Try to parse error
            if let Ok(parsed) = serde_json::from_str::<ChatApiResponse>(&body) {
                if let Some(error) = parsed.error {
                    let message = error.message.unwrap_or_default();
                    let ctx = if let Some(code) = error.code {
                        ctx.with_code(&code)
                    } else {
                        ctx
                    };

                    return Err(match status.as_u16() {
                        429 => ProviderError::rate_limited_remote(Duration::from_secs(60), ctx),
                        _ => ProviderError::provider_with_context(
                            "openrouter",
                            message,
                            status.as_u16() >= 500,
                            ctx,
                        ),
                    });
                }
            }

            return Err(ProviderError::provider_with_context(
                "openrouter",
                format!("HTTP {}", status.as_u16()),
                status.as_u16() >= 500,
                ctx,
            ));
        }

        let parsed: ChatApiResponse = serde_json::from_str(&body).map_err(|e| {
            ProviderError::provider("openrouter", format!("Invalid JSON: {e}"), false)
        })?;

        // Check for API-level error
        if let Some(error) = parsed.error {
            let message = error.message.unwrap_or_default();
            if Self::is_refusal(&message) {
                return Err(ProviderError::refused(message));
            }
            return Err(ProviderError::provider("openrouter", message, false));
        }

        // Extract content
        let choice = parsed
            .choices
            .and_then(|c| c.into_iter().next())
            .ok_or_else(|| {
                ProviderError::provider("openrouter", "No choices in response", false)
            })?;

        let mut content = choice
            .message
            .map(|m| {
                let content = m.content.unwrap_or_default();
                if !content.trim().is_empty() {
                    return content;
                }

                // Some providers/models emit structured output via tool calls even when
                // response_format=json_object is requested. Fall back to tool call args.
                let args = m
                    .tool_calls
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|tc| tc.function.and_then(|f| f.arguments))
                    .find(|s| !s.trim().is_empty())
                    .unwrap_or_default();

                args
            })
            .unwrap_or_default();

        // Normalize content for downstream parsers.
        if content.len() > MAX_RESPONSE_LEN {
            content.truncate(MAX_RESPONSE_LEN);
        }

        // Check for refusal in content
        if Self::is_refusal(&content) {
            return Err(ProviderError::refused(content));
        }

        // Extract usage
        let usage = parsed.usage.ok_or_else(|| {
            ProviderError::provider("openrouter", "Missing usage in response", false)
        })?;

        let input_tokens = usage.prompt_tokens.unwrap_or(0);
        let output_tokens = usage.completion_tokens.unwrap_or(0);

        let latency = start.elapsed();
        let cost = chat_cost(req.model.model_id(), input_tokens, output_tokens);
        let upstream_cost_nanodollars = usage
            .cost_details
            .and_then(|d| d.upstream_inference_cost)
            .map(|usd| ((usd * 1_000_000_000.0).round() as i64).max(0));

        Ok(ChatResponse {
            content,
            input_tokens,
            output_tokens,
            cost_nanodollars: cost,
            upstream_cost_nanodollars,
            latency,
            finish_reason: FinishReason::from(choice.finish_reason),
        })
    }
}
