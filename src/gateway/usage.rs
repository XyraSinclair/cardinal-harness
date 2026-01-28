//! Usage tracking via the UsageSink trait.
//!
//! The gateway logs all calls through a UsageSink. This decouples the gateway
//! from any specific storage backend:
//! - API server uses DbUsageSink (writes to provider_calls table)
//! - CLI tools use NoopUsageSink or JsonLinesSink
//! - Tests use NoopUsageSink or MockUsageSink

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Status of a provider call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallStatus {
    Success,
    Error,
}

impl CallStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            CallStatus::Success => "success",
            CallStatus::Error => "error",
        }
    }
}

/// Spend categories for daily budget enforcement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpendGroup {
    /// Chat/completions (LLM comparisons, rerank, etc).
    Chat,
    /// Embeddings and embedding batches.
    Embeddings,
}

/// Record of a provider API call for logging.
#[derive(Debug, Clone)]
pub struct ProviderCallRecord {
    /// Provider name: "openai", "openrouter", etc.
    pub provider: &'static str,
    /// Endpoint: "embeddings", "chat/completions", "batch", etc.
    pub endpoint: &'static str,
    /// Model used.
    pub model: String,
    /// Input tokens consumed.
    pub input_tokens: i32,
    /// Output tokens generated (0 for embeddings).
    pub output_tokens: i32,
    /// Cost in nanodollars (1e-9 USD).
    pub cost_nanodollars: i64,
    /// Provider-reported upstream inference cost in nanodollars, if available.
    pub upstream_cost_nanodollars: Option<i64>,
    /// User who made the request (if known).
    pub user_id: Option<Uuid>,
    /// Job this request is part of (if any).
    pub job_id: Option<Uuid>,
    /// Batch ID for batch operations.
    pub batch_id: Option<String>,
    /// Latency in milliseconds.
    pub latency_ms: i32,
    /// Call status.
    pub status: CallStatus,
    /// Error code if status is Error.
    pub error_code: Option<String>,
    /// Which code path made this call.
    pub caller: &'static str,
    /// Provider request ID (for debugging).
    pub request_id: Option<String>,
    /// When the call was made.
    pub timestamp: DateTime<Utc>,
}

impl ProviderCallRecord {
    /// Create a new record with required fields, defaulting others.
    pub fn new(
        provider: &'static str,
        endpoint: &'static str,
        model: impl Into<String>,
        caller: &'static str,
    ) -> Self {
        Self {
            provider,
            endpoint,
            model: model.into(),
            input_tokens: 0,
            output_tokens: 0,
            cost_nanodollars: 0,
            upstream_cost_nanodollars: None,
            user_id: None,
            job_id: None,
            batch_id: None,
            latency_ms: 0,
            status: CallStatus::Success,
            error_code: None,
            caller,
            request_id: None,
            timestamp: Utc::now(),
        }
    }

    pub fn tokens(mut self, input: i32, output: i32) -> Self {
        self.input_tokens = input;
        self.output_tokens = output;
        self
    }

    pub fn cost(mut self, nanodollars: i64) -> Self {
        self.cost_nanodollars = nanodollars;
        self
    }

    pub fn upstream_cost(mut self, nanodollars: Option<i64>) -> Self {
        self.upstream_cost_nanodollars = nanodollars;
        self
    }

    pub fn user(mut self, user_id: Option<Uuid>) -> Self {
        self.user_id = user_id;
        self
    }

    pub fn job(mut self, job_id: Option<Uuid>) -> Self {
        self.job_id = job_id;
        self
    }

    pub fn batch(mut self, batch_id: impl Into<String>) -> Self {
        self.batch_id = Some(batch_id.into());
        self
    }

    pub fn latency(mut self, ms: i32) -> Self {
        self.latency_ms = ms;
        self
    }

    pub fn error(mut self, code: impl Into<String>) -> Self {
        self.status = CallStatus::Error;
        self.error_code = Some(code.into());
        self
    }

    pub fn request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }
}

/// Trait for recording provider call usage.
///
/// Implement this trait to customize where usage data is stored.
#[async_trait]
pub trait UsageSink: Send + Sync {
    /// Record a provider call. This should be fire-and-forget:
    /// failures should be logged but not propagated.
    async fn record(&self, record: ProviderCallRecord);

    /// Return total provider spend for the current day (USD), if available.
    async fn daily_spend_usd(&self) -> Option<f64> {
        None
    }

    /// Return provider spend for the current day within a category, if available.
    async fn daily_spend_usd_for_group(&self, _group: SpendGroup) -> Option<f64> {
        None
    }
}

/// No-op usage sink that discards all records.
/// Useful for CLI tools and tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopUsageSink;

#[async_trait]
impl UsageSink for NoopUsageSink {
    async fn record(&self, _record: ProviderCallRecord) {
        // Discard
    }

    async fn daily_spend_usd(&self) -> Option<f64> {
        None
    }

    async fn daily_spend_usd_for_group(&self, _group: SpendGroup) -> Option<f64> {
        None
    }
}

/// Usage sink that writes to stderr as JSON lines.
/// Useful for CLI tools that want to capture usage.
#[derive(Debug, Clone, Copy, Default)]
pub struct StderrUsageSink;

#[async_trait]
impl UsageSink for StderrUsageSink {
    async fn record(&self, record: ProviderCallRecord) {
        // Simple JSON output to stderr
        eprintln!(
            r#"{{"provider":"{}","endpoint":"{}","model":"{}","tokens":{},"cost_nanos":{},"status":"{}","caller":"{}"}}"#,
            record.provider,
            record.endpoint,
            record.model,
            record.input_tokens + record.output_tokens,
            record.cost_nanodollars,
            record.status.as_str(),
            record.caller,
        );
    }

    async fn daily_spend_usd(&self) -> Option<f64> {
        None
    }

    async fn daily_spend_usd_for_group(&self, _group: SpendGroup) -> Option<f64> {
        None
    }
}
