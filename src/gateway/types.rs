//! Core types for the provider gateway.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

// =============================================================================
// ATTRIBUTION
// =============================================================================

/// Attribution for cost tracking and debugging.
///
/// Every request through the gateway carries attribution so we know:
/// - Who made the request (user_id)
/// - What job it's part of (job_id)
/// - Which code path triggered it (caller)
#[derive(Debug, Clone, Default)]
pub struct Attribution {
    /// User who initiated the request (if known).
    pub user_id: Option<Uuid>,
    /// Job this request is part of (for rating jobs, batch jobs, etc.).
    pub job_id: Option<Uuid>,
    /// Which code path made this call, for debugging.
    /// Use a static string like "scry::embed" or "job_executor::compare".
    pub caller: &'static str,
}

impl Attribution {
    pub fn new(caller: &'static str) -> Self {
        Self {
            caller,
            ..Default::default()
        }
    }

    pub fn with_user(mut self, user_id: Uuid) -> Self {
        self.user_id = Some(user_id);
        self
    }

    pub fn with_job(mut self, job_id: Uuid) -> Self {
        self.job_id = Some(job_id);
        self
    }
}

// =============================================================================
// EMBEDDING TYPES
// =============================================================================

/// Embedding model to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedModel {
    /// OpenAI text-embedding-3-large (3072 dimensions)
    OpenAI3Large,
    /// OpenAI text-embedding-3-small (1536 dimensions)
    OpenAI3Small,
}

impl EmbedModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbedModel::OpenAI3Large => "text-embedding-3-large",
            EmbedModel::OpenAI3Small => "text-embedding-3-small",
        }
    }

    pub fn dimensions(&self) -> usize {
        match self {
            EmbedModel::OpenAI3Large => 3072,
            EmbedModel::OpenAI3Small => 1536,
        }
    }

    pub fn provider(&self) -> &'static str {
        "openai"
    }
}

/// Request to embed texts.
#[derive(Debug, Clone)]
pub struct EmbedRequest {
    /// Model to use for embedding.
    pub model: EmbedModel,
    /// Texts to embed. Each text produces one embedding vector.
    pub texts: Vec<String>,
    /// Attribution for cost tracking.
    pub attribution: Attribution,
}

impl EmbedRequest {
    pub fn new(model: EmbedModel, texts: Vec<String>, attribution: Attribution) -> Self {
        Self {
            model,
            texts,
            attribution,
        }
    }

    /// Single text convenience constructor.
    pub fn single(model: EmbedModel, text: String, attribution: Attribution) -> Self {
        Self::new(model, vec![text], attribution)
    }
}

/// Response from embedding request.
#[derive(Debug, Clone)]
pub struct EmbedResponse {
    /// Embedding vectors, one per input text.
    pub embeddings: Vec<Vec<f32>>,
    /// Total tokens consumed.
    pub tokens: u32,
    /// Cost in nanodollars (1e-9 USD).
    pub cost_nanodollars: i64,
    /// Time taken for the request.
    pub latency: Duration,
}

// =============================================================================
// CHAT TYPES
// =============================================================================

/// Chat message role.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// A chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Chat model specification.
#[derive(Debug, Clone)]
pub enum ChatModel {
    /// OpenRouter model, e.g. "anthropic/claude-3-5-haiku"
    OpenRouter(String),
}

impl ChatModel {
    pub fn openrouter(model_id: impl Into<String>) -> Self {
        ChatModel::OpenRouter(model_id.into())
    }

    pub fn model_id(&self) -> &str {
        match self {
            ChatModel::OpenRouter(id) => id,
        }
    }

    pub fn provider(&self) -> &'static str {
        match self {
            ChatModel::OpenRouter(_) => "openrouter",
        }
    }

    /// Extract route for rate limiting (e.g. "anthropic" from "anthropic/claude-3-5-haiku").
    pub fn route(&self) -> &str {
        match self {
            ChatModel::OpenRouter(id) => id.split('/').next().unwrap_or(id),
        }
    }
}

/// Request for chat completion.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// Model to use.
    pub model: ChatModel,
    /// Messages in the conversation.
    pub messages: Vec<Message>,
    /// Sampling temperature (0.0 - 2.0).
    pub temperature: f32,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Whether to request JSON output.
    pub json_mode: bool,
    /// Attribution for cost tracking.
    pub attribution: Attribution,
}

impl ChatRequest {
    pub fn new(model: ChatModel, messages: Vec<Message>, attribution: Attribution) -> Self {
        Self {
            model,
            messages,
            temperature: 0.0,
            max_tokens: None,
            json_mode: false,
            attribution,
        }
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }

    pub fn json(mut self) -> Self {
        self.json_mode = true;
        self
    }
}

/// Reason the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Unknown(String),
}

impl From<Option<String>> for FinishReason {
    fn from(s: Option<String>) -> Self {
        match s.as_deref() {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::Length,
            Some("content_filter") => FinishReason::ContentFilter,
            Some("tool_calls") => FinishReason::ToolCalls,
            Some(other) => FinishReason::Unknown(other.to_string()),
            None => FinishReason::Unknown("none".to_string()),
        }
    }
}

/// Response from chat completion.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Generated content.
    pub content: String,
    /// Input tokens consumed.
    pub input_tokens: u32,
    /// Output tokens generated.
    pub output_tokens: u32,
    /// Cost in nanodollars.
    pub cost_nanodollars: i64,
    /// Provider-reported upstream inference cost (nanodollars), if available.
    ///
    /// For OpenRouter this is derived from `usage.cost_details.upstream_inference_cost`.
    /// Used for auditing pricing drift vs our internal token pricing registry.
    pub upstream_cost_nanodollars: Option<i64>,
    /// Time taken for the request.
    pub latency: Duration,
    /// Why the model stopped.
    pub finish_reason: FinishReason,
}

// =============================================================================
// BATCH TYPES
// =============================================================================

/// Opaque handle to an uploaded file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileId(pub String);

impl FileId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Opaque handle to a batch job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchId(pub String);

impl BatchId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Batch API endpoint type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchEndpoint {
    Embeddings,
    ChatCompletions,
}

impl BatchEndpoint {
    pub fn as_str(&self) -> &'static str {
        match self {
            BatchEndpoint::Embeddings => "/v1/embeddings",
            BatchEndpoint::ChatCompletions => "/v1/chat/completions",
        }
    }
}

/// State of a batch job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchState {
    Validating,
    InProgress,
    Finalizing,
    Completed,
    Failed,
    Expired,
    Cancelling,
    Cancelled,
}

impl From<&str> for BatchState {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "validating" => BatchState::Validating,
            "in_progress" => BatchState::InProgress,
            "finalizing" => BatchState::Finalizing,
            "completed" => BatchState::Completed,
            "failed" => BatchState::Failed,
            "expired" => BatchState::Expired,
            "cancelling" => BatchState::Cancelling,
            "cancelled" => BatchState::Cancelled,
            _ => BatchState::InProgress, // Default to in_progress for unknown states
        }
    }
}

impl BatchState {
    /// Whether this is a terminal state (no more changes expected).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            BatchState::Completed
                | BatchState::Failed
                | BatchState::Expired
                | BatchState::Cancelled
        )
    }

    /// Whether this state indicates success.
    pub fn is_success(&self) -> bool {
        matches!(self, BatchState::Completed)
    }
}

/// Status of a batch job.
#[derive(Debug, Clone)]
pub struct BatchStatus {
    /// Current state.
    pub state: BatchState,
    /// Number of completed requests.
    pub completed: usize,
    /// Number of failed requests.
    pub failed: usize,
    /// Total number of requests.
    pub total: usize,
    /// Output file ID (available when completed).
    pub output_file_id: Option<FileId>,
    /// Error file ID (available when there are failures).
    pub error_file_id: Option<FileId>,
}

impl BatchStatus {
    /// Progress as a fraction (0.0 - 1.0).
    pub fn progress(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.completed + self.failed) as f32 / self.total as f32
        }
    }
}
