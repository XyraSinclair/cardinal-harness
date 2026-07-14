//! Core types for the provider gateway.

use crate::discrete::{DiscreteDistribution, WeightedValue};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Neg, Sub};
use std::time::Duration;
use uuid::Uuid;

// =============================================================================
// ATTRIBUTION
// =============================================================================

/// Attribution for cost tracking and debugging.
///
/// Every request through the gateway carries attribution so we know:
/// - Who made the request (user_id)
/// - Which API key initiated the request (api_key_id)
/// - What job it's part of (job_id)
/// - Which code path triggered it (caller)
#[derive(Debug, Clone, Default)]
pub struct Attribution {
    /// User who initiated the request (if known).
    pub user_id: Option<Uuid>,
    /// API key that initiated the request (if known).
    pub api_key_id: Option<Uuid>,
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

    pub fn with_api_key(mut self, api_key_id: Uuid) -> Self {
        self.api_key_id = Some(api_key_id);
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
    /// These are useful for diagnostics and future answer-level rescoring, but
    /// decimal ratio ladders do not admit a valid confidence estimate from a
    /// single token-position peek.
    pub logprobs: bool,
    /// Number of top alternative logprobs to return per token position.
    /// Only meaningful when `logprobs` is true. Typically 5-20.
    pub top_logprobs: Option<u32>,
    /// Optional normalized reasoning configuration for providers that support it.
    pub reasoning: Option<ReasoningConfig>,
    /// OpenAI-style cache-routing hint (`prompt_cache_key`): should be
    /// derived from the STABLE content (template + attribute + entities)
    /// and independent of any nonce or padding, so repeat draws route to
    /// the same provider cache slot. None = omit.
    pub prompt_cache_key: Option<String>,
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
            reasoning: None,
            prompt_cache_key: None,
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

    pub fn reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Xhigh,
    High,
    Medium,
    Low,
    Minimal,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
}

impl ReasoningConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: Some(false),
            effort: None,
            max_tokens: None,
            exclude: None,
        }
    }

    pub fn low() -> Self {
        Self {
            enabled: None,
            effort: Some(ReasoningEffort::Low),
            max_tokens: None,
            exclude: None,
        }
    }

    pub fn low_with_excluded_trace() -> Self {
        Self {
            exclude: Some(true),
            ..Self::low()
        }
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PairwisePreferredSide {
    A,
    B,
}

impl PairwisePreferredSide {
    fn index(self) -> usize {
        match self {
            PairwisePreferredSide::A => 0,
            PairwisePreferredSide::B => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RatioBucket {
    R00,
    R01,
    R02,
    R03,
    R04,
    R05,
    R06,
    R07,
    R08,
    R09,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    R16,
}

impl RatioBucket {
    pub const ALL: [Self; 17] = [
        Self::R00,
        Self::R01,
        Self::R02,
        Self::R03,
        Self::R04,
        Self::R05,
        Self::R06,
        Self::R07,
        Self::R08,
        Self::R09,
        Self::R10,
        Self::R11,
        Self::R12,
        Self::R13,
        Self::R14,
        Self::R15,
        Self::R16,
    ];

    pub fn all() -> &'static [Self] {
        &Self::ALL
    }

    pub fn index(self) -> usize {
        match self {
            Self::R00 => 0,
            Self::R01 => 1,
            Self::R02 => 2,
            Self::R03 => 3,
            Self::R04 => 4,
            Self::R05 => 5,
            Self::R06 => 6,
            Self::R07 => 7,
            Self::R08 => 8,
            Self::R09 => 9,
            Self::R10 => 10,
            Self::R11 => 11,
            Self::R12 => 12,
            Self::R13 => 13,
            Self::R14 => 14,
            Self::R15 => 15,
            Self::R16 => 16,
        }
    }

    pub fn ratio(self) -> f64 {
        match self {
            Self::R00 => 1.0,
            Self::R01 => 1.05,
            Self::R02 => 1.1,
            Self::R03 => 1.2,
            Self::R04 => 1.3,
            Self::R05 => 1.5,
            Self::R06 => 1.75,
            Self::R07 => 2.1,
            Self::R08 => 2.5,
            Self::R09 => 3.1,
            Self::R10 => 3.9,
            Self::R11 => 5.1,
            Self::R12 => 6.8,
            Self::R13 => 9.2,
            Self::R14 => 12.7,
            Self::R15 => 18.0,
            Self::R16 => 26.0,
        }
    }

    pub fn ln_ratio(self) -> f64 {
        self.ratio().ln()
    }

    pub fn from_ratio(ratio: f64) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|bucket| (bucket.ratio() - ratio).abs() < 1e-9)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PairwiseAnswer {
    A(RatioBucket),
    B(RatioBucket),
    Refuse,
}

impl PairwiseAnswer {
    pub fn observation(side: PairwisePreferredSide, ratio_bucket: RatioBucket) -> Self {
        match side {
            PairwisePreferredSide::A => Self::A(ratio_bucket),
            PairwisePreferredSide::B => Self::B(ratio_bucket),
        }
    }

    pub fn preferred_side(self) -> Option<PairwisePreferredSide> {
        match self {
            Self::A(_) => Some(PairwisePreferredSide::A),
            Self::B(_) => Some(PairwisePreferredSide::B),
            Self::Refuse => None,
        }
    }

    pub fn ratio_bucket(self) -> Option<RatioBucket> {
        match self {
            Self::A(bucket) | Self::B(bucket) => Some(bucket),
            Self::Refuse => None,
        }
    }

    pub fn ratio(self) -> Option<f64> {
        self.ratio_bucket().map(RatioBucket::ratio)
    }

    pub fn signed_ln_ratio(self) -> Option<f64> {
        match self {
            Self::A(bucket) => Some(bucket.ln_ratio()),
            Self::B(bucket) => Some(-bucket.ln_ratio()),
            Self::Refuse => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SignedLogRatioDistribution {
    pub distribution: DiscreteDistribution<f64>,
    pub abstain_probability: f64,
}

impl SignedLogRatioDistribution {
    const DEFAULT_MAX_SUPPORT: usize = 128;
    const MERGE_TOLERANCE: f64 = 1e-12;

    pub fn new(distribution: DiscreteDistribution<f64>, abstain_probability: f64) -> Self {
        Self {
            distribution,
            abstain_probability: abstain_probability.clamp(0.0, 1.0),
        }
    }

    pub fn from_answer_distribution(
        answer_distribution: &DiscreteDistribution<PairwiseAnswer>,
    ) -> Self {
        let mut support = Vec::with_capacity(answer_distribution.support.len());
        let mut abstain_probability = 0.0;

        for entry in &answer_distribution.support {
            if let Some(value) = entry.value.signed_ln_ratio() {
                push_merged_float_probability(
                    &mut support,
                    value,
                    entry.probability,
                    Self::MERGE_TOLERANCE,
                );
            } else {
                abstain_probability += entry.probability;
            }
        }

        Self::new(
            DiscreteDistribution::new(support, answer_distribution.residual_probability),
            abstain_probability,
        )
    }

    pub fn modeled_probability(&self) -> f64 {
        self.distribution.support_probability()
    }

    pub fn total_probability(&self) -> f64 {
        self.distribution.total_probability() + self.abstain_probability
    }

    pub fn mean(&self) -> Option<f64> {
        self.distribution.expectation_by(|value| *value)
    }

    pub fn variance(&self) -> Option<f64> {
        self.distribution.variance_by(|value| *value)
    }

    pub fn probability_positive(&self) -> f64 {
        self.distribution.probability_of(|value| *value > 0.0)
    }

    pub fn probability_negative(&self) -> f64 {
        self.distribution.probability_of(|value| *value < 0.0)
    }

    pub fn probability_within(&self, delta: f64) -> f64 {
        let radius = delta.abs();
        self.distribution
            .probability_of(|value| value.abs() <= radius)
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self::new(
            DiscreteDistribution::new(
                self.distribution
                    .support
                    .iter()
                    .map(|entry| WeightedValue {
                        value: entry.value * factor,
                        probability: entry.probability,
                    })
                    .collect(),
                self.distribution.residual_probability,
            ),
            self.abstain_probability,
        )
    }

    pub fn compress(&self, max_support: usize) -> Self {
        if self.distribution.support.len() <= max_support || max_support == 0 {
            return self.clone();
        }

        let mut sorted_support = self.distribution.support.clone();
        sorted_support.sort_by(|left, right| {
            left.value
                .partial_cmp(&right.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let chunk_size = sorted_support.len().div_ceil(max_support);
        let mut compressed = Vec::with_capacity(max_support);

        for chunk in sorted_support.chunks(chunk_size) {
            let probability: f64 = chunk.iter().map(|entry| entry.probability).sum();
            if probability <= 0.0 {
                continue;
            }

            let weighted_value = chunk
                .iter()
                .map(|entry| entry.value * entry.probability)
                .sum::<f64>()
                / probability;

            compressed.push(WeightedValue {
                value: weighted_value,
                probability,
            });
        }

        Self::new(
            DiscreteDistribution::new(compressed, self.distribution.residual_probability),
            self.abstain_probability,
        )
    }

    pub fn convolve(&self, other: &Self) -> Self {
        let mut support =
            Vec::with_capacity(self.distribution.support.len() * other.distribution.support.len());

        for left in &self.distribution.support {
            for right in &other.distribution.support {
                push_merged_float_probability(
                    &mut support,
                    left.value + right.value,
                    left.probability * right.probability,
                    Self::MERGE_TOLERANCE,
                );
            }
        }

        let abstain_probability = (self.abstain_probability + other.abstain_probability
            - self.abstain_probability * other.abstain_probability)
            .clamp(0.0, 1.0);
        let support_probability =
            self.distribution.support_probability() * other.distribution.support_probability();
        let residual_probability =
            (1.0 - support_probability - abstain_probability).clamp(0.0, 1.0);

        Self::new(
            DiscreteDistribution::new(support, residual_probability),
            abstain_probability,
        )
        .compress(Self::DEFAULT_MAX_SUPPORT)
    }
}

impl Neg for SignedLogRatioDistribution {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.scale(-1.0)
    }
}

impl Add for SignedLogRatioDistribution {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.convolve(&rhs)
    }
}

impl Sub for SignedLogRatioDistribution {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Mul<f64> for SignedLogRatioDistribution {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self.scale(rhs)
    }
}

fn push_merged_float_probability(
    support: &mut Vec<WeightedValue<f64>>,
    value: f64,
    probability: f64,
    tolerance: f64,
) {
    if !value.is_finite() || !probability.is_finite() || probability <= 0.0 {
        return;
    }

    if let Some(existing) = support
        .iter_mut()
        .find(|entry| (entry.value - value).abs() <= tolerance)
    {
        let combined_probability = existing.probability + probability;
        existing.value =
            (existing.value * existing.probability + value * probability) / combined_probability;
        existing.probability = combined_probability;
        return;
    }

    support.push(WeightedValue { value, probability });
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwiseLogprobPosterior {
    /// Selected semantic answer in the structured output.
    pub selected_answer: PairwiseAnswer,
    /// Preferred side chosen by the model in the parsed structured output.
    pub selected_higher_ranked: PairwisePreferredSide,
    /// Ratio chosen by the model in the parsed structured output.
    pub selected_ratio: f64,
    /// Ratio bucket chosen by the model in the parsed structured output.
    pub selected_ratio_bucket: RatioBucket,
    /// Discrete posterior over the preferred side token.
    pub higher_ranked_distribution: DiscreteDistribution<PairwisePreferredSide>,
    /// Discrete posterior over ratio ladder buckets at the selected token position.
    pub ratio_distribution: DiscreteDistribution<RatioBucket>,
    /// Approximate semantic posterior over pairwise answer states.
    pub answer_distribution: DiscreteDistribution<PairwiseAnswer>,
    /// Latent posterior over signed log-ratio values, suitable for solver-side algebra.
    pub signed_ln_ratio_distribution: SignedLogRatioDistribution,
    /// Derived confidence metrics from the ratio distribution.
    pub confidence: ConfidenceSource,
}

impl PairwiseLogprobPosterior {
    pub fn mean_signed_ln_ratio(&self) -> Option<f64> {
        self.signed_ln_ratio_distribution.mean()
    }

    pub fn variance_signed_ln_ratio(&self) -> Option<f64> {
        self.signed_ln_ratio_distribution.variance()
    }

    pub fn probability_positive(&self) -> f64 {
        self.signed_ln_ratio_distribution.probability_positive()
    }

    pub fn probability_negative(&self) -> f64 {
        self.signed_ln_ratio_distribution.probability_negative()
    }
}

/// Response from chat completion.
#[derive(Debug, Clone)]
pub struct ChatResponse {
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

fn token_numeric_value(token: &str) -> Option<f64> {
    let mut start = None;
    let mut end = 0usize;
    let mut dot_count = 0usize;

    for (idx, ch) in token.char_indices() {
        if start.is_none() {
            let next_is_digit = token[idx + ch.len_utf8()..]
                .chars()
                .next()
                .is_some_and(|next| next.is_ascii_digit());
            if ch.is_ascii_digit() || (ch == '.' && next_is_digit) {
                start = Some(idx);
                end = idx + ch.len_utf8();
                if ch == '.' {
                    dot_count = 1;
                }
            }
            continue;
        }

        if ch.is_ascii_digit() {
            end = idx + ch.len_utf8();
            continue;
        }

        if ch == '.' {
            dot_count += 1;
            end = idx + ch.len_utf8();
            continue;
        }

        break;
    }

    let start = start?;
    let numeric = &token[start..end];
    if numeric == "." || dot_count > 1 {
        return None;
    }

    numeric.parse::<f64>().ok()
}

fn ratio_ladder_index_for_token(token: &str, ratio_ladder: &[f64]) -> Option<usize> {
    let parsed = token_numeric_value(token)?;
    ratio_ladder
        .iter()
        .position(|ratio| (ratio - parsed).abs() < 1e-9)
}

fn pairwise_preferred_side_for_token(token: &str) -> Option<PairwisePreferredSide> {
    let letters: String = token
        .chars()
        .filter(|ch| ch.is_ascii_alphabetic())
        .collect();
    match letters.to_ascii_uppercase().as_str() {
        "A" => Some(PairwisePreferredSide::A),
        "B" => Some(PairwisePreferredSide::B),
        _ => None,
    }
}

fn collect_token_probabilities(
    position: &TokenLogprob,
    support_len: usize,
    mut token_index: impl FnMut(&str) -> Option<usize>,
) -> (Vec<f64>, f64) {
    let mut probabilities = vec![0.0; support_len];

    if let Some(idx) = token_index(&position.token) {
        probabilities[idx] = position.logprob.exp();
    }

    for alternative in &position.top_alternatives {
        if let Some(idx) = token_index(&alternative.token) {
            probabilities[idx] = probabilities[idx].max(alternative.logprob.exp());
        }
    }

    let covered_mass: f64 = probabilities.iter().sum();
    let residual_probability = (1.0 - covered_mass).max(0.0);
    (probabilities, residual_probability)
}

pub fn truncate_output_logprobs(
    logprobs: &[TokenLogprob],
    max_alternatives: usize,
) -> Vec<TokenLogprob> {
    logprobs
        .iter()
        .map(|entry| TokenLogprob {
            token: entry.token.clone(),
            logprob: entry.logprob,
            top_alternatives: entry
                .top_alternatives
                .iter()
                .take(max_alternatives)
                .cloned()
                .collect(),
        })
        .collect()
}

/// Build a pairwise posterior from winner and ratio token alternatives.
///
/// This helper assumes the preferred side and ratio each correspond to a single
/// token position whose alternatives enumerate the relevant support. That is a
/// useful synthetic model and can be valid for genuinely atomic vocabularies,
/// but it is not sufficient for decimal ratio ladders without continuation
/// rescoring.
pub fn pairwise_logprob_posterior(
    logprobs: &[TokenLogprob],
    selected_higher_ranked: PairwisePreferredSide,
    selected_ratio: f64,
    ratio_ladder: &[f64],
) -> Option<PairwiseLogprobPosterior> {
    let selected_ratio_bucket = RatioBucket::from_ratio(selected_ratio)?;
    let selected_answer =
        PairwiseAnswer::observation(selected_higher_ranked, selected_ratio_bucket);

    let higher_ranked_position = logprobs.iter().find(|lp| {
        pairwise_preferred_side_for_token(&lp.token)
            .is_some_and(|side| side == selected_higher_ranked)
    })?;

    // Find the token position corresponding to the ratio output.
    let ratio_position = logprobs.iter().find(|lp| {
        ratio_ladder_index_for_token(&lp.token, ratio_ladder)
            .is_some_and(|idx| (ratio_ladder[idx] - selected_ratio).abs() < 1e-9)
    })?;

    let selected_idx = ratio_ladder
        .iter()
        .position(|&r| (r - selected_ratio).abs() < 1e-9)?;
    let (winner_probs, winner_residual_probability) =
        collect_token_probabilities(higher_ranked_position, 2, |token| {
            pairwise_preferred_side_for_token(token).map(PairwisePreferredSide::index)
        });
    let higher_ranked_distribution = DiscreteDistribution::new(
        [PairwisePreferredSide::A, PairwisePreferredSide::B]
            .into_iter()
            .enumerate()
            .map(|(idx, side)| WeightedValue {
                value: side,
                probability: winner_probs[idx],
            })
            .collect(),
        winner_residual_probability,
    );

    let (ratio_probs, ratio_residual_probability) =
        collect_token_probabilities(ratio_position, ratio_ladder.len(), |token| {
            ratio_ladder_index_for_token(token, ratio_ladder)
        });

    let neighbor_indices: Vec<usize> = (0..ratio_ladder.len())
        .filter(|&i| i.abs_diff(selected_idx) <= 1)
        .collect();
    let ratio_distribution = DiscreteDistribution::new(
        RatioBucket::all()
            .iter()
            .copied()
            .zip(ratio_probs)
            .map(|(ratio_bucket, probability)| WeightedValue {
                value: ratio_bucket,
                probability,
            })
            .collect(),
        ratio_residual_probability,
    );
    let answer_distribution = higher_ranked_distribution
        .product(&ratio_distribution, |higher_ranked, ratio| {
            PairwiseAnswer::observation(*higher_ranked, *ratio)
        });
    let signed_ln_ratio_distribution =
        SignedLogRatioDistribution::from_answer_distribution(&answer_distribution);
    let top_prob = answer_distribution.probability_of(|answer| *answer == selected_answer);
    let neighborhood_prob = answer_distribution.probability_of(|answer| {
        answer.preferred_side() == Some(selected_higher_ranked)
            && answer
                .ratio_bucket()
                .map(|bucket| neighbor_indices.contains(&bucket.index()))
                .unwrap_or(false)
    });
    let neighborhood_prob = neighborhood_prob.clamp(0.0, 1.0);
    let confidence = ConfidenceSource::Logprob {
        entropy: answer_distribution.entropy(),
        top_prob,
        neighborhood_prob,
    };

    Some(PairwiseLogprobPosterior {
        selected_answer,
        selected_higher_ranked,
        selected_ratio,
        selected_ratio_bucket,
        higher_ranked_distribution,
        ratio_distribution,
        answer_distribution,
        signed_ln_ratio_distribution,
        confidence,
    })
}

/// Extract confidence from logprob distribution over a ratio ladder.
///
/// Given the output logprobs and the set of valid ratio tokens, compute
/// confidence metrics from the token probability distribution.
pub fn confidence_from_logprobs(
    logprobs: &[TokenLogprob],
    selected_higher_ranked: PairwisePreferredSide,
    selected_ratio: f64,
    ratio_ladder: &[f64],
) -> Option<ConfidenceSource> {
    pairwise_logprob_posterior(
        logprobs,
        selected_higher_ranked,
        selected_ratio,
        ratio_ladder,
    )
    .map(|posterior| posterior.confidence)
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
    fn test_confidence_from_logprobs_basic() {
        let ladder = vec![1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.1, 2.5];

        let logprobs = vec![
            TokenLogprob {
                token: "\"A\"".to_string(),
                logprob: -0.1,
                top_alternatives: vec![TokenAlternative {
                    token: "\"B\"".to_string(),
                    logprob: -2.4,
                }],
            },
            TokenLogprob {
                token: "2.5".to_string(),
                logprob: -0.22_f64, // ~0.80 probability
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
            },
        ];

        let result = confidence_from_logprobs(&logprobs, PairwisePreferredSide::A, 2.5, &ladder);
        assert!(result.is_some(), "should extract confidence");

        let cs = result.unwrap();
        match cs {
            ConfidenceSource::Logprob {
                entropy,
                top_prob,
                neighborhood_prob,
            } => {
                assert!(top_prob > 0.7 && top_prob < 0.9, "top_prob={top_prob}");
                assert!(
                    neighborhood_prob >= top_prob,
                    "neighborhood should include top"
                );
                assert!(entropy > 0.0, "entropy should be positive");
            }
            _ => panic!("expected Logprob variant"),
        }
    }

    #[test]
    fn test_confidence_from_logprobs_missing_ratio() {
        let ladder = vec![1.0, 1.5, 2.0];
        let logprobs = vec![
            TokenLogprob {
                token: "\"A\"".to_string(),
                logprob: -0.1,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "hello".to_string(),
                logprob: -0.1,
                top_alternatives: vec![],
            },
        ];

        // Ratio 2.0 not in any logprob token.
        let result = confidence_from_logprobs(&logprobs, PairwisePreferredSide::A, 2.0, &ladder);
        assert!(result.is_none());
    }

    #[test]
    fn test_confidence_from_logprobs_handles_ratio_one_without_prefix_collision() {
        let ladder = vec![1.0, 1.05, 12.7];
        let logprobs = vec![
            TokenLogprob {
                token: "\"A\"".to_string(),
                logprob: -0.1,
                top_alternatives: vec![TokenAlternative {
                    token: "\"B\"".to_string(),
                    logprob: -2.3,
                }],
            },
            TokenLogprob {
                token: "\"1.0\"".to_string(),
                logprob: -0.1,
                top_alternatives: vec![TokenAlternative {
                    token: "12.7".to_string(),
                    logprob: -2.0,
                }],
            },
        ];

        let result = confidence_from_logprobs(&logprobs, PairwisePreferredSide::A, 1.0, &ladder);
        assert!(result.is_some(), "should match the 1.0 token exactly");

        let cs = result.unwrap();
        match cs {
            ConfidenceSource::Logprob { top_prob, .. } => {
                assert!(top_prob > 0.8, "top_prob={top_prob}");
            }
            _ => panic!("expected Logprob variant"),
        }
    }

    #[test]
    fn test_pairwise_logprob_posterior_captures_winner_uncertainty() {
        let ladder = vec![1.0, 1.5, 2.5];
        let confident_ratio = TokenLogprob {
            token: "2.5".to_string(),
            logprob: -0.05,
            top_alternatives: vec![TokenAlternative {
                token: "1.5".to_string(),
                logprob: -3.5,
            }],
        };
        let confident_winner = TokenLogprob {
            token: "\"A\"".to_string(),
            logprob: -0.05,
            top_alternatives: vec![TokenAlternative {
                token: "\"B\"".to_string(),
                logprob: -3.5,
            }],
        };
        let ambiguous_winner = TokenLogprob {
            token: "\"A\"".to_string(),
            logprob: -0.7,
            top_alternatives: vec![TokenAlternative {
                token: "\"B\"".to_string(),
                logprob: -0.75,
            }],
        };

        let high_confidence = pairwise_logprob_posterior(
            &[confident_winner.clone(), confident_ratio.clone()],
            PairwisePreferredSide::A,
            2.5,
            &ladder,
        )
        .expect("posterior");
        let low_confidence = pairwise_logprob_posterior(
            &[ambiguous_winner, confident_ratio],
            PairwisePreferredSide::A,
            2.5,
            &ladder,
        )
        .expect("posterior");

        let high_b = high_confidence
            .higher_ranked_distribution
            .probability_of(|side| *side == PairwisePreferredSide::B);
        let low_b = low_confidence
            .higher_ranked_distribution
            .probability_of(|side| *side == PairwisePreferredSide::B);
        assert!(high_b < 0.1);
        assert!(low_b > 0.4);
    }

    #[test]
    fn test_ratio_bucket_roundtrips_and_pairwise_answer_maps_to_latent() {
        let bucket = RatioBucket::from_ratio(2.5).expect("bucket");
        assert_eq!(bucket, RatioBucket::R08);
        assert!((bucket.ratio() - 2.5).abs() < 1e-9);

        let answer = PairwiseAnswer::observation(PairwisePreferredSide::B, bucket);
        assert_eq!(answer.preferred_side(), Some(PairwisePreferredSide::B));
        assert_eq!(answer.ratio_bucket(), Some(RatioBucket::R08));
        assert_eq!(answer.ratio(), Some(2.5));
        assert!(answer.signed_ln_ratio().expect("latent") < 0.0);
    }

    #[test]
    fn test_signed_log_ratio_distribution_operator_overloads() {
        let left =
            SignedLogRatioDistribution::from_answer_distribution(&DiscreteDistribution::new(
                vec![
                    WeightedValue {
                        value: PairwiseAnswer::A(RatioBucket::R05),
                        probability: 0.5,
                    },
                    WeightedValue {
                        value: PairwiseAnswer::B(RatioBucket::R00),
                        probability: 0.5,
                    },
                ],
                0.0,
            ));
        let right =
            SignedLogRatioDistribution::from_answer_distribution(&DiscreteDistribution::new(
                vec![WeightedValue {
                    value: PairwiseAnswer::A(RatioBucket::R05),
                    probability: 1.0,
                }],
                0.0,
            ));

        let added = left.clone() + right.clone();
        let negated = -right.clone();
        let subtracted = added.clone() - right.clone();
        let scaled = right.clone() * 2.0;

        assert!(added.mean().expect("mean") > right.mean().expect("mean"));
        assert!(negated.mean().expect("mean") < 0.0);
        assert!(subtracted.mean().expect("mean") < added.mean().expect("mean"));
        assert!(scaled.mean().expect("mean") > right.mean().expect("mean"));
    }

    #[test]
    fn test_token_parsers_handle_common_structured_output_noise() {
        assert_eq!(token_numeric_value(".5"), Some(0.5));
        assert_eq!(token_numeric_value("Ratio: 1.05"), Some(1.05));
        assert_eq!(token_numeric_value("1.0.5"), None);

        assert_eq!(
            pairwise_preferred_side_for_token("**A**"),
            Some(PairwisePreferredSide::A)
        );
        assert_eq!(
            pairwise_preferred_side_for_token("B."),
            Some(PairwisePreferredSide::B)
        );
    }

    #[test]
    fn test_push_merged_float_probability_updates_centroid() {
        let mut support = Vec::new();
        push_merged_float_probability(&mut support, 1.0, 0.5, 1e-12);
        push_merged_float_probability(&mut support, 1.0 + 0.9e-12, 0.5, 1e-12);

        assert_eq!(support.len(), 1, "values within tolerance should merge");
        assert!((support[0].probability - 1.0).abs() < 1e-12);
        assert!((support[0].value - (1.0 + 0.45e-12)).abs() < 1e-13);
    }

    #[test]
    fn test_signed_log_ratio_distribution_bimodality_survives_latent_projection() {
        let bimodal =
            SignedLogRatioDistribution::from_answer_distribution(&DiscreteDistribution::new(
                vec![
                    WeightedValue {
                        value: PairwiseAnswer::A(RatioBucket::R00),
                        probability: 0.5,
                    },
                    WeightedValue {
                        value: PairwiseAnswer::A(RatioBucket::R16),
                        probability: 0.5,
                    },
                ],
                0.0,
            ));
        let local_blur =
            SignedLogRatioDistribution::from_answer_distribution(&DiscreteDistribution::new(
                vec![
                    WeightedValue {
                        value: PairwiseAnswer::A(RatioBucket::R07),
                        probability: 0.5,
                    },
                    WeightedValue {
                        value: PairwiseAnswer::A(RatioBucket::R08),
                        probability: 0.5,
                    },
                ],
                0.0,
            ));

        assert!((bimodal.probability_positive() - 0.5).abs() < 1e-12);
        assert!((local_blur.probability_positive() - 1.0).abs() < 1e-12);
        assert!(
            bimodal.variance().expect("variance") > local_blur.variance().expect("variance"),
            "far-apart magnitude uncertainty should inflate latent variance more than local blur"
        );
    }

    #[test]
    fn test_signed_log_ratio_distribution_long_run_convolution_preserves_mass_and_mean() {
        let base = SignedLogRatioDistribution::new(
            DiscreteDistribution::new(
                vec![
                    WeightedValue {
                        value: -0.4,
                        probability: 0.4,
                    },
                    WeightedValue {
                        value: 1.2,
                        probability: 0.6,
                    },
                ],
                0.0,
            ),
            0.0,
        );

        let analytical_mean = base.mean().expect("mean");
        let analytical_variance = base.variance().expect("variance");
        let mut current = base.clone();
        let steps = 24;

        for _ in 0..steps {
            current = (current + base.clone()).compress(32);
            assert!(
                (current.total_probability() - 1.0).abs() < 1e-9,
                "probability mass should stay normalized after repeated convolution/compression"
            );
        }

        let expected_mean = analytical_mean * (steps + 1) as f64;
        let expected_variance = analytical_variance * (steps + 1) as f64;
        assert!((current.mean().expect("mean") - expected_mean).abs() < 1e-9);
        assert!(
            current.variance().expect("variance") <= expected_variance + 1e-9,
            "compression should not hallucinate extra variance"
        );
    }

    #[test]
    fn test_pairwise_logprob_posterior_tracks_residual_mass_from_unmodeled_alternatives() {
        let ladder = vec![1.0, 1.5];
        let posterior = pairwise_logprob_posterior(
            &[
                TokenLogprob {
                    token: "\"A\"".to_string(),
                    logprob: 0.4f64.ln(),
                    top_alternatives: vec![TokenAlternative {
                        token: "\"garbage\"".to_string(),
                        logprob: 0.6f64.ln(),
                    }],
                },
                TokenLogprob {
                    token: "1.5".to_string(),
                    logprob: 0.5f64.ln(),
                    top_alternatives: vec![],
                },
            ],
            PairwisePreferredSide::A,
            1.5,
            &ladder,
        )
        .expect("posterior");

        assert!((posterior.higher_ranked_distribution.support_probability() - 0.4).abs() < 1e-9);
        assert!((posterior.higher_ranked_distribution.residual_probability - 0.6).abs() < 1e-9);
        assert!((posterior.ratio_distribution.support_probability() - 0.5).abs() < 1e-9);
        assert!((posterior.ratio_distribution.residual_probability - 0.5).abs() < 1e-9);
        assert!((posterior.answer_distribution.support_probability() - 0.2).abs() < 1e-9);
        assert!((posterior.answer_distribution.residual_probability - 0.8).abs() < 1e-9);
        assert!((posterior.answer_distribution.total_probability() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pairwise_logprob_posterior_exposes_answer_and_latent_distributions() {
        let ladder = vec![1.0, 1.5, 2.5];
        let posterior = pairwise_logprob_posterior(
            &[
                TokenLogprob {
                    token: "\"A\"".to_string(),
                    logprob: -0.1,
                    top_alternatives: vec![TokenAlternative {
                        token: "\"B\"".to_string(),
                        logprob: -2.3,
                    }],
                },
                TokenLogprob {
                    token: "2.5".to_string(),
                    logprob: -0.22,
                    top_alternatives: vec![TokenAlternative {
                        token: "1.5".to_string(),
                        logprob: -1.61,
                    }],
                },
            ],
            PairwisePreferredSide::A,
            2.5,
            &ladder,
        )
        .expect("posterior");

        assert_eq!(posterior.selected_ratio_bucket, RatioBucket::R08);
        assert_eq!(
            posterior.selected_answer,
            PairwiseAnswer::A(RatioBucket::R08)
        );
        assert!(posterior.answer_distribution.top_probability() > 0.6);
        assert!(posterior.mean_signed_ln_ratio().expect("mean") > 0.0);
        assert!(posterior.variance_signed_ln_ratio().expect("variance") >= 0.0);
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

    #[test]
    fn test_attribution_with_api_key_builder() {
        let api_key_id = Uuid::new_v4();
        let attribution = Attribution::new("test").with_api_key(api_key_id);

        assert_eq!(attribution.api_key_id, Some(api_key_id));
        assert_eq!(attribution.caller, "test");
    }
}
