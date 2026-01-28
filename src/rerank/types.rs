//! Request/response types for the reranking API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Tier 1: Simple Rerank (/v1/rerank)
// =============================================================================

/// Input document for reranking.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RerankDocument {
    /// Stable identifier for the document.
    pub id: String,
    /// Text content shown to the rater.
    pub text: String,
}

/// Request for single-attribute reranking.
#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    /// Optional query context (folded into attribute_prompt).
    #[serde(default)]
    pub query: Option<String>,

    /// Documents to rerank.
    pub documents: Vec<RerankDocument>,

    /// Attribute identifier (for caching).
    #[serde(default = "default_attribute_id")]
    pub attribute_id: String,

    /// Natural language description of the attribute.
    #[serde(default = "default_attribute_prompt")]
    pub attribute_prompt: String,

    /// Focus region: return/optimize for top k.
    #[serde(default)]
    pub top_k: Option<usize>,

    /// Maximum pairwise comparisons to make.
    #[serde(default)]
    pub comparison_budget: Option<usize>,

    /// Maximum time budget in milliseconds.
    #[serde(default)]
    pub latency_budget_ms: Option<u64>,

    /// Stop when top-k error falls below this threshold.
    #[serde(default = "default_tolerated_error")]
    pub tolerated_error: f64,

    /// Model to use for comparisons.
    #[serde(default)]
    pub model: Option<String>,

    /// Logical rater ID for planner.
    #[serde(default)]
    pub rater_id: Option<String>,

    /// Maximum number of pairwise comparisons to run concurrently.
    /// Defaults to a conservative internal value when omitted.
    #[serde(default)]
    pub comparison_concurrency: Option<usize>,

    /// Maximum total repeats per (attribute, pair) during this rerank run.
    ///
    /// Each successful pairwise comparison increments repeats by 1.
    #[serde(default)]
    pub max_pair_repeats: Option<usize>,
}

fn default_attribute_id() -> String {
    "relevance".to_string()
}

fn default_attribute_prompt() -> String {
    "relevance to the query".to_string()
}

fn default_tolerated_error() -> f64 {
    0.1
}

/// Per-document result in the rerank response.
#[derive(Debug, Serialize)]
pub struct RerankResult {
    /// Document identifier.
    pub id: String,
    /// 1-based rank among results.
    pub rank: usize,
    /// Posterior mean in latent space.
    pub latent_mean: f64,
    /// Posterior std in latent space.
    pub latent_std: f64,
    /// Robust z-score: (x - median) / (MAD * 1.4826).
    pub z_score: f64,
    /// Shifted so min = 1.0.
    pub min_normalized: f64,
    /// Percentile among documents (0..1).
    pub percentile: f64,
}

/// Metadata for a rerank response.
#[derive(Debug, Serialize)]
pub struct RerankMeta {
    /// Estimated top-k error (sum of p_flip in band).
    pub topk_error: f64,
    /// User-specified threshold.
    pub tolerated_error: f64,
    /// Total comparisons attempted (including refusals).
    pub comparisons_attempted: usize,
    /// Comparisons that produced observations.
    pub comparisons_used: usize,
    /// Comparisons where model refused.
    pub comparisons_refused: usize,
    /// Budget that was set.
    pub comparison_budget: usize,
    /// Elapsed time.
    pub latency_ms: u128,
    /// Model that was used.
    pub model_used: String,
    /// Rater ID that was used.
    pub rater_id_used: String,
    /// Provider input tokens consumed across all comparisons.
    pub provider_input_tokens: u32,
    /// Provider output tokens generated across all comparisons.
    pub provider_output_tokens: u32,
    /// Provider cost (nanodollars) across all comparisons.
    pub provider_cost_nanodollars: i64,

    /// Why the rerank loop stopped.
    pub stop_reason: RerankStopReason,
}

/// Response for single-attribute reranking.
#[derive(Debug, Serialize)]
pub struct RerankResponse {
    /// Ranked results, sorted by descending latent_mean.
    pub results: Vec<RerankResult>,
    /// Metadata about the reranking run.
    pub meta: RerankMeta,
}

// =============================================================================
// Tier 2: Multi-Attribute Rerank (/v1/rerank/multi)
// =============================================================================

/// Why the rerank loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RerankStopReason {
    /// Current top-k error is <= tolerated_error.
    ToleratedErrorMet,
    /// Certified separation bound implies stable top-k (consecutive checks).
    CertifiedStop,
    /// comparison_budget exhausted.
    BudgetExhausted,
    /// latency_budget_ms exceeded.
    LatencyBudgetExceeded,
    /// Cancellation requested (async worker).
    Cancelled,
    /// Planner produced no proposals.
    NoProposals,
    /// Proposals existed but none were eligible to run.
    NoNewPairs,
}

/// Input entity for multi-attribute reranking.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MultiRerankEntity {
    /// Stable identifier.
    pub id: String,
    /// Text content.
    pub text: String,
}

/// Attribute specification in multi-rerank request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MultiRerankAttributeSpec {
    /// Attribute identifier.
    pub id: String,
    /// Natural language description.
    pub prompt: String,
    /// Optional prompt template slug (e.g., canonical_v2).
    #[serde(default)]
    pub prompt_template_slug: Option<String>,
    /// Weight in global utility.
    pub weight: f64,
}

/// Top-k configuration for multi-rerank.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MultiRerankTopKSpec {
    /// Number of top items to focus on.
    pub k: usize,
    /// Exponent for weight emphasis in planning.
    #[serde(default = "default_weight_exponent")]
    pub weight_exponent: f64,
    /// Stop when global top-k error is below this.
    #[serde(default = "default_tolerated_error")]
    pub tolerated_error: f64,
    /// Frontier width for uncertainty tracking and candidate selection.
    #[serde(default = "default_band_size")]
    pub band_size: usize,
    /// Max active set size to enable effective-resistance variance for critical pair.
    #[serde(default = "default_effective_resistance_max_active")]
    pub effective_resistance_max_active: usize,
    /// Inflate sigma for certified stop to be conservative.
    #[serde(default = "default_stop_sigma_inflate")]
    pub stop_sigma_inflate: f64,
    /// Require this many consecutive certified checks to stop.
    #[serde(default = "default_stop_min_consecutive")]
    pub stop_min_consecutive: usize,
}

fn default_weight_exponent() -> f64 {
    1.3
}

fn default_band_size() -> usize {
    5
}

fn default_effective_resistance_max_active() -> usize {
    64
}

fn default_stop_sigma_inflate() -> f64 {
    1.25
}

fn default_stop_min_consecutive() -> usize {
    2
}

/// Gate specification for filtering entities.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MultiRerankGateSpec {
    /// Attribute to gate on.
    pub attribute_id: String,
    /// Unit for threshold: "latent", "z", "percentile", "min_norm".
    #[serde(default = "default_gate_unit")]
    pub unit: String,
    /// Comparison operator: ">=" or "<=".
    pub op: String,
    /// Threshold value.
    pub threshold: f64,
}

fn default_gate_unit() -> String {
    "latent".to_string()
}

/// Request for multi-attribute reranking.
#[derive(Debug, Deserialize, Serialize)]
pub struct MultiRerankRequest {
    /// Entities to rerank.
    pub entities: Vec<MultiRerankEntity>,

    /// Attributes with weights.
    pub attributes: Vec<MultiRerankAttributeSpec>,

    /// Top-k configuration.
    pub topk: MultiRerankTopKSpec,

    /// Optional gates for filtering.
    #[serde(default)]
    pub gates: Vec<MultiRerankGateSpec>,

    /// Maximum pairwise comparisons.
    #[serde(default)]
    pub comparison_budget: Option<usize>,

    /// Maximum time budget in milliseconds.
    #[serde(default)]
    pub latency_budget_ms: Option<u64>,

    /// Model to use.
    #[serde(default)]
    pub model: Option<String>,

    /// Logical rater ID.
    #[serde(default)]
    pub rater_id: Option<String>,

    /// Maximum number of pairwise comparisons to run concurrently.
    /// Defaults to a conservative internal value when omitted.
    #[serde(default)]
    pub comparison_concurrency: Option<usize>,

    /// Maximum total repeats per (attribute, pair) during this rerank run.
    ///
    /// Each successful pairwise comparison increments repeats by 1.
    #[serde(default)]
    pub max_pair_repeats: Option<usize>,
}

/// Per-attribute score summary.
#[derive(Debug, Serialize)]
pub struct AttributeScoreSummary {
    /// Posterior mean in latent space.
    pub latent_mean: f64,
    /// Posterior std.
    pub latent_std: f64,
    /// Robust z-score.
    pub z_score: f64,
    /// Min-normalized (min -> 1.0).
    pub min_normalized: f64,
    /// Percentile among feasible entities.
    pub percentile: f64,
}

/// Per-entity result in multi-rerank response.
#[derive(Debug, Serialize)]
pub struct MultiRerankEntityResult {
    /// Entity identifier.
    pub id: String,
    /// 1-based rank among feasible entities, None if infeasible.
    pub rank: Option<usize>,
    /// Whether entity passes all gates.
    pub feasible: bool,
    /// Combined utility mean.
    pub u_mean: f64,
    /// Combined utility std.
    pub u_std: f64,
    /// Probability of crossing the k-boundary (Gaussian approximation).
    pub p_flip: f64,
    /// Per-attribute scores.
    pub attribute_scores: HashMap<String, AttributeScoreSummary>,
}

/// Metadata for multi-rerank response.
#[derive(Debug, Serialize)]
pub struct MultiRerankMeta {
    /// Global top-k error (frontier inversion bound).
    pub global_topk_error: f64,
    /// User-specified threshold.
    pub tolerated_error: f64,
    /// k value used.
    pub k: usize,
    /// Frontier width used.
    pub band_size: usize,
    /// Total comparisons attempted.
    pub comparisons_attempted: usize,
    /// Comparisons that produced observations.
    pub comparisons_used: usize,
    /// Comparisons where model refused.
    pub comparisons_refused: usize,
    /// Budget that was set.
    pub comparison_budget: usize,
    /// Elapsed time.
    pub latency_ms: u128,
    /// Model that was used.
    pub model_used: String,
    /// Rater ID that was used.
    pub rater_id_used: String,
    /// Provider input tokens consumed across all comparisons.
    pub provider_input_tokens: u32,
    /// Provider output tokens generated across all comparisons.
    pub provider_output_tokens: u32,
    /// Provider cost (nanodollars) across all comparisons.
    pub provider_cost_nanodollars: i64,

    /// Why the rerank loop stopped.
    pub stop_reason: RerankStopReason,
}

/// Response for multi-attribute reranking.
#[derive(Debug, Serialize)]
pub struct MultiRerankResponse {
    /// Ranked entities.
    pub entities: Vec<MultiRerankEntityResult>,
    /// Metadata about the run.
    pub meta: MultiRerankMeta,
}

// =============================================================================
// Internal types
// =============================================================================

/// Direction of preference in a pairwise comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HigherRanked {
    A,
    B,
}

/// Result of a pairwise LLM comparison.
#[derive(Debug)]
pub enum PairwiseJudgement {
    /// Valid comparison result.
    Observation {
        higher_ranked: HigherRanked,
        ratio: f64,
        confidence: f64,
    },
    /// Model refused to judge.
    Refused,
}
