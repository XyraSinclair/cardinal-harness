//! Provider gateway for OpenRouter chat completions.

pub mod error;
pub mod openrouter;
pub mod pricing;
pub mod types;
pub mod usage;

use std::sync::Arc;
use std::time::Duration;

use tokio::time::sleep;

use openrouter::{ChatProvider, OpenRouterAdapter};
use usage::{CallStatus, ProviderCallRecord, UsageSink as UsageSinkTrait};

pub use error::{ErrorContext, ProviderError, RateLimitSource};
pub use pricing::*;
pub use types::*;
pub use usage::{NoopUsageSink, StderrUsageSink, UsageSink};

#[async_trait::async_trait]
pub trait ChatGateway: Send + Sync {
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, ProviderError>;
}

#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub max_retries: u32,
    pub retry_base_delay: Duration,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            retry_base_delay: Duration::from_secs(1),
        }
    }
}

pub struct ProviderGateway<U: UsageSinkTrait> {
    openrouter: OpenRouterAdapter,
    usage_sink: Arc<U>,
    config: GatewayConfig,
}

#[async_trait::async_trait]
impl<U: UsageSinkTrait> ChatGateway for ProviderGateway<U> {
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, ProviderError> {
        ProviderGateway::chat(self, req).await
    }
}

impl<U: UsageSinkTrait> ProviderGateway<U> {
    pub fn from_env(usage_sink: Arc<U>) -> Result<Self, ProviderError> {
        let openrouter = OpenRouterAdapter::from_env()?;
        Ok(Self {
            openrouter,
            usage_sink,
            config: GatewayConfig::default(),
        })
    }

    pub fn with_config(
        openrouter: OpenRouterAdapter,
        usage_sink: Arc<U>,
        config: GatewayConfig,
    ) -> Self {
        Self {
            openrouter,
            usage_sink,
            config,
        }
    }

    pub async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, ProviderError> {
        let mut last_error: Option<ProviderError> = None;

        for attempt in 0..=self.config.max_retries {
            let result = self.openrouter.chat(&req).await;
            match result {
                Ok(resp) => {
                    self.record_usage(&req, &resp, CallStatus::Success, None)
                        .await;
                    return Ok(resp);
                }
                Err(err) => {
                    let code = err.code().to_string();
                    self.record_usage(&req, &ChatResponse::empty(), CallStatus::Error, Some(code))
                        .await;

                    if !err.is_retryable() || attempt == self.config.max_retries {
                        return Err(err);
                    }

                    let delay = backoff_delay(self.config.retry_base_delay, attempt);
                    last_error = Some(err);
                    sleep(delay).await;
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| ProviderError::provider("openrouter", "unknown error", false)))
    }

    async fn record_usage(
        &self,
        req: &ChatRequest,
        resp: &ChatResponse,
        status: CallStatus,
        error_code: Option<String>,
    ) {
        let record = ProviderCallRecord::new(
            req.model.provider(),
            "chat/completions",
            req.model.model_id(),
            req.attribution.caller,
        )
        .tokens(resp.input_tokens as i32, resp.output_tokens as i32)
        .cost(resp.cost_nanodollars)
        .upstream_cost(resp.upstream_cost_nanodollars)
        .user(req.attribution.user_id)
        .job(req.attribution.job_id)
        .latency(resp.latency.as_millis() as i32);

        let record = if status == CallStatus::Error {
            record.error(error_code.unwrap_or_else(|| "provider_error".to_string()))
        } else {
            record
        };

        self.usage_sink.record(record).await;
    }
}

fn backoff_delay(base: Duration, attempt: u32) -> Duration {
    let multiplier = 2u64.pow(attempt.min(5));
    base * multiplier as u32
}

impl ChatResponse {
    fn empty() -> Self {
        Self {
            content: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            cost_nanodollars: 0,
            upstream_cost_nanodollars: None,
            latency: Duration::from_millis(0),
            finish_reason: FinishReason::Unknown("error".to_string()),
            output_logprobs: None,
            cache_read_tokens: None,
            cache_write_tokens: None,
        }
    }
}
