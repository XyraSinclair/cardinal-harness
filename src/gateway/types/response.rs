use serde::{Deserialize, Serialize};
use std::time::Duration;

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
    /// Provider-native completion identifier, when returned in the response body.
    pub provider_call_id: Option<String>,
    /// Provider request identifier, when returned in response headers.
    pub provider_request_id: Option<String>,
    /// Model identifier the provider reports it actually served, when available.
    ///
    /// Measurement runs should assert this against the requested model; routed
    /// channels (OpenRouter variants, Claude Code aliases) can serve a
    /// different concrete model than the one named in the request.
    pub served_model: Option<String>,
    /// Generated content.
    pub content: String,
    /// Provider-returned reasoning text, if available.
    pub reasoning: Option<String>,
    /// Reasoning token count, when the provider reports it separately.
    pub reasoning_tokens: Option<u32>,
    /// Input tokens consumed.
    pub input_tokens: u32,
    /// Output tokens generated.
    pub output_tokens: u32,
    /// Cost in nanodollars.
    ///
    /// If `cost_is_estimate` is true, this came from the local fallback estimate rather than
    /// an exact pricing-table entry or provider-reported cost.
    pub cost_nanodollars: i64,
    /// True when `cost_nanodollars` used fallback pricing because no exact local pricing entry
    /// or provider-reported upstream cost was available.
    pub cost_is_estimate: bool,
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
    /// This is raw provider metadata. A valid ladder-level posterior may require
    /// continuation rescoring rather than naive inspection of one emitted token.
    pub output_logprobs: Option<Vec<TokenLogprob>>,
    /// Input tokens served from provider prompt cache (if reported).
    pub cache_read_tokens: Option<u32>,
    /// Input tokens written to provider prompt cache (if reported).
    pub cache_write_tokens: Option<u32>,
}

/// Provenance for confidence-like information retained with an observation.
///
/// These values are descriptive metadata. Solver precision must come from a
/// measured response distribution, not from collapsing this enum to a scalar.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConfidenceSource {
    /// Model self-reported confidence (from JSON output field).
    SelfReported(f64),
    /// Derived from a valid answer-level logprob scoring path.
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
