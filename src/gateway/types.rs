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
    /// Whether to request token-level logprobs in the response.
    ///
    /// When true, the provider returns log-probabilities for output tokens.
    /// For structured outputs (ratio ladder selection), logprobs over ratio
    /// tokens provide a direct confidence signal that's cheaper and potentially
    /// more calibrated than self-reported confidence.
    pub logprobs: bool,
    /// Number of top alternative logprobs to return per token position.
    /// Only meaningful when `logprobs` is true. Typically 5-20.
    pub top_logprobs: Option<u32>,
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
            logprobs: false,
            top_logprobs: None,
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

    /// Request token-level logprobs with the specified number of alternatives.
    pub fn with_logprobs(mut self, top_n: u32) -> Self {
        self.logprobs = true;
        self.top_logprobs = Some(top_n);
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

/// A single token's logprob entry with alternatives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogprob {
    /// The token string.
    pub token: String,
    /// Log-probability of this token.
    pub logprob: f64,
    /// Top alternative tokens at this position (if requested).
    pub top_alternatives: Vec<TokenAlternative>,
}

/// An alternative token at a given position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAlternative {
    /// The alternative token string.
    pub token: String,
    /// Log-probability of this alternative.
    pub logprob: f64,
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
    /// Per-token logprobs for the output, if requested and available.
    ///
    /// When present, this enables logprob-based confidence extraction:
    /// the distribution over ratio ladder tokens directly measures the model's
    /// uncertainty about the comparison.
    pub output_logprobs: Option<Vec<TokenLogprob>>,
    /// Input tokens served from provider prompt cache (if reported).
    pub cache_read_tokens: Option<u32>,
    /// Input tokens written to provider prompt cache (if reported).
    pub cache_write_tokens: Option<u32>,
}

/// Source of confidence information for an observation.
///
/// Different confidence sources have different calibration properties.
/// The rating engine can apply per-source calibration curves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceSource {
    /// Model self-reported confidence (from JSON output field).
    SelfReported(f64),
    /// Derived from logprob distribution over ratio tokens.
    Logprob {
        /// Shannon entropy of the ratio token distribution (lower = more certain).
        entropy: f64,
        /// Probability mass on the selected ratio token.
        top_prob: f64,
        /// Probability mass within one ladder step of the selected ratio.
        neighborhood_prob: f64,
    },
    /// Future: provider-reported internal coherence metrics.
    LabsCoherence {
        /// Provider-computed internal consistency score.
        internal_consistency: f64,
        /// Provider-computed epistemic uncertainty estimate.
        epistemic_uncertainty: f64,
    },
    /// Weighted blend of multiple confidence sources.
    Blended {
        /// The blended scalar confidence value.
        value: f64,
        /// Contributing sources and their weights.
        components: Vec<(String, f64)>,
    },
}

impl ConfidenceSource {
    /// Extract a scalar confidence value for use in the rating engine.
    ///
    /// This collapses the rich confidence information into a single float
    /// suitable for the `g(c)` mapping function.
    pub fn as_scalar(&self) -> f64 {
        match self {
            ConfidenceSource::SelfReported(c) => *c,
            ConfidenceSource::Logprob {
                top_prob,
                neighborhood_prob,
                ..
            } => {
                // Blend top_prob and neighborhood_prob.
                // neighborhood_prob is more robust to long tails.
                0.4 * top_prob + 0.6 * neighborhood_prob
            }
            ConfidenceSource::LabsCoherence {
                internal_consistency,
                epistemic_uncertainty,
            } => {
                // High consistency + low uncertainty = high confidence.
                (internal_consistency * (1.0 - epistemic_uncertainty)).clamp(0.0, 1.0)
            }
            ConfidenceSource::Blended { value, .. } => *value,
        }
    }
}

/// Extract confidence from logprob distribution over a ratio ladder.
///
/// Given the output logprobs and the set of valid ratio tokens, compute
/// confidence metrics from the token probability distribution.
pub fn confidence_from_logprobs(
    logprobs: &[TokenLogprob],
    selected_ratio: f64,
    ratio_ladder: &[f64],
) -> Option<ConfidenceSource> {
    // Find the token position corresponding to the ratio output.
    // For JSON output, the ratio token might be at various positions.
    // We look for a token whose text matches or contains the ratio value.
    let selected_str = format!("{selected_ratio}");

    let ratio_position = logprobs.iter().find(|lp| {
        lp.token.trim() == selected_str
            || lp.token.contains(&selected_str)
    })?;

    let top_prob = ratio_position.logprob.exp();

    // Compute neighborhood probability: mass within one ladder step.
    let selected_idx = ratio_ladder
        .iter()
        .position(|&r| (r - selected_ratio).abs() < 1e-6)?;

    let neighbor_indices: Vec<usize> = (0..ratio_ladder.len())
        .filter(|&i| {
            let dist = if i > selected_idx {
                i - selected_idx
            } else {
                selected_idx - i
            };
            dist <= 1
        })
        .collect();

    let mut neighborhood_prob = top_prob;
    for alt in &ratio_position.top_alternatives {
        for &ni in &neighbor_indices {
            let neighbor_str = format!("{}", ratio_ladder[ni]);
            if alt.token.trim() == neighbor_str || alt.token.contains(&neighbor_str) {
                neighborhood_prob += alt.logprob.exp();
            }
        }
    }
    neighborhood_prob = neighborhood_prob.clamp(0.0, 1.0);

    // Compute entropy over the ratio distribution.
    let mut probs: Vec<f64> = vec![top_prob];
    for alt in &ratio_position.top_alternatives {
        probs.push(alt.logprob.exp());
    }
    // Normalize (top_logprobs may not cover all probability mass).
    let total: f64 = probs.iter().sum();
    let remaining = (1.0 - total).max(0.0);
    if remaining > 0.0 {
        probs.push(remaining);
    }

    let entropy: f64 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    Some(ConfidenceSource::Logprob {
        entropy,
        top_prob,
        neighborhood_prob,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_source_self_reported() {
        let cs = ConfidenceSource::SelfReported(0.85);
        assert!((cs.as_scalar() - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_confidence_source_logprob() {
        let cs = ConfidenceSource::Logprob {
            entropy: 0.5,
            top_prob: 0.8,
            neighborhood_prob: 0.95,
        };
        // 0.4 * 0.8 + 0.6 * 0.95 = 0.32 + 0.57 = 0.89
        let scalar = cs.as_scalar();
        assert!((scalar - 0.89).abs() < 1e-6, "got {scalar}");
    }

    #[test]
    fn test_confidence_source_labs_coherence() {
        let cs = ConfidenceSource::LabsCoherence {
            internal_consistency: 0.9,
            epistemic_uncertainty: 0.1,
        };
        // 0.9 * (1.0 - 0.1) = 0.81
        assert!((cs.as_scalar() - 0.81).abs() < 1e-6);
    }

    #[test]
    fn test_confidence_source_blended() {
        let cs = ConfidenceSource::Blended {
            value: 0.75,
            components: vec![("self_report".into(), 0.5), ("logprob".into(), 0.5)],
        };
        assert!((cs.as_scalar() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_confidence_from_logprobs_basic() {
        let ladder = vec![1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.1, 2.5];

        let logprobs = vec![TokenLogprob {
            token: "2.5".to_string(),
            logprob: (-0.22_f64).into(), // ~0.80 probability
            top_alternatives: vec![
                TokenAlternative {
                    token: "2.1".to_string(),
                    logprob: -1.61, // ~0.20
                },
                TokenAlternative {
                    token: "3.1".to_string(),
                    logprob: -4.61, // ~0.01
                },
            ],
        }];

        let result = confidence_from_logprobs(&logprobs, 2.5, &ladder);
        assert!(result.is_some(), "should extract confidence");

        let cs = result.unwrap();
        match cs {
            ConfidenceSource::Logprob {
                entropy,
                top_prob,
                neighborhood_prob,
            } => {
                assert!(top_prob > 0.7 && top_prob < 0.9, "top_prob={top_prob}");
                assert!(neighborhood_prob >= top_prob, "neighborhood should include top");
                assert!(entropy > 0.0, "entropy should be positive");
            }
            _ => panic!("expected Logprob variant"),
        }
    }

    #[test]
    fn test_confidence_from_logprobs_missing_ratio() {
        let ladder = vec![1.0, 1.5, 2.0];
        let logprobs = vec![TokenLogprob {
            token: "hello".to_string(),
            logprob: -0.1,
            top_alternatives: vec![],
        }];

        // Ratio 2.0 not in any logprob token.
        let result = confidence_from_logprobs(&logprobs, 2.0, &ladder);
        assert!(result.is_none());
    }

    #[test]
    fn test_chat_request_logprobs_builder() {
        let req = ChatRequest::new(
            ChatModel::openrouter("test/model"),
            vec![Message::user("hi")],
            Attribution::new("test"),
        )
        .with_logprobs(5);

        assert!(req.logprobs);
        assert_eq!(req.top_logprobs, Some(5));
    }

    #[test]
    fn test_chat_request_default_no_logprobs() {
        let req = ChatRequest::new(
            ChatModel::openrouter("test/model"),
            vec![Message::user("hi")],
            Attribution::new("test"),
        );

        assert!(!req.logprobs);
        assert!(req.top_logprobs.is_none());
    }
}
