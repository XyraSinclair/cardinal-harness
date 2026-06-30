//! LLM pairwise comparison logic for reranking.
//!
//! Implements the contract between LLM JSON responses and solver observations.

use serde::Deserialize;
use tracing::warn;

use crate::cache::{
    CacheError, CachedJudgement, PairwiseCache, PairwiseCacheAttribute, PairwiseCacheEntity,
    PairwiseCacheKey, PairwiseCacheKeyParts, PairwiseCacheTemplate,
};
use crate::discrete::{DiscreteDistribution, WeightedValue};
use crate::gateway::{
    pairwise_logprob_posterior, truncate_output_logprobs, Attribution, ChatGateway, ChatModel,
    ChatRequest, ConfidenceSource, PairwiseAnswer, PairwiseLogprobPosterior, PairwisePreferredSide,
    ProviderError, RatioBucket, SignedLogRatioDistribution, TokenLogprob,
};
use crate::text_chunking::count_tokens;

use crate::prompts::{
    prompt_by_slug, EntityRef, PromptInstance, PromptTemplate, DEFAULT_PROMPT, RATIO_LADDER,
};

use super::types::{HigherRanked, PairwiseJudgement};

// =============================================================================
// Constants
// =============================================================================

/// Minimum variance (high confidence).
pub const SIGMA_MIN: f64 = 0.2;
/// Maximum variance (low confidence).
pub const SIGMA_MAX: f64 = 2.0;

/// Default max output tokens for pairwise judgements.
///
/// Reasoning-capable models can spend a large hidden budget before they emit the
/// visible JSON answer. A small cap suppresses judgement quality and can yield
/// empty visible output on OpenRouter.
pub const PAIRWISE_MAX_OUTPUT_TOKENS_DEFAULT: u32 = 8192;
pub const PAIRWISE_MAX_OUTPUT_TOKENS_GPT5: u32 = PAIRWISE_MAX_OUTPUT_TOKENS_DEFAULT;
pub const PAIRWISE_LOGPROBS_TOP_N_DEFAULT: u32 = 20;
pub const PAIRWISE_BUCKET_LOGPROB_MAX_ATTEMPTS: usize = 3;

pub fn pairwise_max_output_tokens(model: &str) -> u32 {
    if model.starts_with("openai/gpt-5") {
        PAIRWISE_MAX_OUTPUT_TOKENS_GPT5
    } else {
        PAIRWISE_MAX_OUTPUT_TOKENS_DEFAULT
    }
}

pub fn pairwise_logprobs_top_n() -> u32 {
    std::env::var("CARDINAL_PAIRWISE_LOGPROBS_TOP_N")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| (1..=50).contains(value))
        .unwrap_or(PAIRWISE_LOGPROBS_TOP_N_DEFAULT)
}

// =============================================================================
// JSON parsing
// =============================================================================

/// Raw JSON structure from LLM response.
#[derive(Debug, Deserialize)]
struct PairwiseEvalJson {
    #[serde(default)]
    higher_ranked: Option<String>,
    #[serde(default)]
    ratio: Option<f64>,
    #[serde(default)]
    ratio_bucket: Option<usize>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    refused: Option<bool>,
}

/// Error type for comparison operations.
#[derive(Debug, thiserror::Error)]
pub enum ComparisonError {
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Cache error: {0}")]
    Cache(#[from] CacheError),
    #[error("Cache miss: {0}")]
    CacheMiss(String),
}

/// Usage info for a single LLM comparison call.
#[derive(Debug, Clone)]
pub struct ComparisonUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub provider_cost_nanodollars: i64,
    pub provider_cost_is_estimate: bool,
    pub cached: bool,
    pub prompt_text: Option<String>,
    pub question_text: Option<String>,
    pub raw_output: Option<String>,
    pub output_logprobs: Option<Vec<TokenLogprob>>,
    pub pairwise_logprob_posterior: Option<PairwiseLogprobPosterior>,
}

#[derive(Debug, Clone, Copy)]
pub struct PairwiseComparisonEntity<'a> {
    pub id: &'a str,
    pub text: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub struct PairwiseComparisonAttribute<'a> {
    pub id: &'a str,
    pub prompt: &'a str,
    pub prompt_template_slug: Option<&'a str>,
}

#[derive(Debug, Clone, Copy)]
pub struct PairwiseComparisonSpec<'a> {
    pub model: &'a str,
    pub attribute: PairwiseComparisonAttribute<'a>,
    pub entity_a: PairwiseComparisonEntity<'a>,
    pub entity_b: PairwiseComparisonEntity<'a>,
}

impl PairwiseComparisonSpec<'_> {
    #[must_use]
    pub fn prompt_template(self) -> PromptTemplate {
        self.attribute
            .prompt_template_slug
            .and_then(prompt_by_slug)
            .unwrap_or(DEFAULT_PROMPT)
    }

    #[must_use]
    pub fn prompt_instance(self) -> PromptInstance {
        self.prompt_template().render(
            self.attribute.id,
            self.attribute.prompt,
            EntityRef::with_context("A", self.entity_a.text),
            EntityRef::with_context("B", self.entity_b.text),
        )
    }

    #[must_use]
    pub fn cache_key(self) -> PairwiseCacheKey {
        let template = self.prompt_template();
        let template_hash = template.template_hash();
        PairwiseCacheKey::from_parts(PairwiseCacheKeyParts {
            model: self.model,
            prompt_template: PairwiseCacheTemplate {
                slug: template.slug,
                template_hash: &template_hash,
            },
            attribute: PairwiseCacheAttribute {
                id: self.attribute.id,
                prompt: self.attribute.prompt,
            },
            entity_a: PairwiseCacheEntity {
                id: self.entity_a.id,
                text: self.entity_a.text,
            },
            entity_b: PairwiseCacheEntity {
                id: self.entity_b.id,
                text: self.entity_b.text,
            },
        })
    }
}

#[derive(Debug, Clone)]
pub struct PairwiseComparisonRequest<'a> {
    pub spec: PairwiseComparisonSpec<'a>,
    pub cache_only: bool,
    pub attribution: Attribution,
}

/// Parse LLM response JSON into a PairwiseJudgement.
pub fn parse_pairwise_response(
    raw: &str,
    _output_logprobs: Option<&[TokenLogprob]>,
) -> Result<PairwiseJudgement, ComparisonError> {
    // Try to extract JSON from the response (may have surrounding text)
    let json_str = extract_json(raw);

    let parsed: PairwiseEvalJson =
        serde_json::from_str(json_str).map_err(|e| ComparisonError::Parse(e.to_string()))?;

    if parsed.refused.unwrap_or(false) {
        return Ok(PairwiseJudgement::Refused);
    }

    let higher = parsed
        .higher_ranked
        .ok_or_else(|| ComparisonError::Parse("missing 'higher_ranked'".into()))?;
    let ratio = match parsed.ratio {
        Some(ratio) => ratio,
        None => {
            let bucket = parsed
                .ratio_bucket
                .ok_or_else(|| ComparisonError::Parse("missing 'ratio'".into()))?;
            *RATIO_LADDER.get(bucket).ok_or_else(|| {
                ComparisonError::Parse(format!(
                    "ratio_bucket out of allowed range [0,16]: {bucket}"
                ))
            })?
        }
    };

    if !(1.0..=26.0).contains(&ratio) {
        return Err(ComparisonError::Parse(format!(
            "ratio out of allowed range [1,26]: {ratio}"
        )));
    }

    let higher_ranked = match higher.to_uppercase().as_str() {
        "A" => HigherRanked::A,
        "B" => HigherRanked::B,
        other => {
            return Err(ComparisonError::Parse(format!(
                "invalid higher_ranked: {other}"
            )))
        }
    };
    let confidence = parsed
        .confidence
        .ok_or_else(|| ComparisonError::Parse("missing 'confidence'".into()))?;

    Ok(PairwiseJudgement::Observation {
        higher_ranked,
        ratio,
        confidence: confidence.clamp(0.0, 1.0),
    })
}

fn token_preferred_side(token: &str) -> Option<PairwisePreferredSide> {
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

fn token_bucket_index(token: &str) -> Option<usize> {
    let digits: String = token.chars().filter(|ch| ch.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let idx = digits.parse::<usize>().ok()?;
    (idx < RATIO_LADDER.len()).then_some(idx)
}

fn token_bucket_index_at(logprobs: &[TokenLogprob], position: usize) -> Option<usize> {
    let token = &logprobs[position].token;
    let previous_token = position
        .checked_sub(1)
        .and_then(|idx| logprobs.get(idx))
        .map(|entry| entry.token.as_str());
    token_bucket_index_with_previous(token, previous_token)
}

fn token_bucket_index_with_previous(token: &str, previous_token: Option<&str>) -> Option<usize> {
    let token_digits: String = token.chars().filter(|ch| ch.is_ascii_digit()).collect();
    if token_digits.len() == 1 {
        if let Some(previous) = previous_token {
            let previous_digits: String =
                previous.chars().filter(|ch| ch.is_ascii_digit()).collect();
            if previous_digits == "1" {
                let second_digit = token_digits.parse::<usize>().ok()?;
                let idx = 10 + second_digit;
                if idx < RATIO_LADDER.len() {
                    return Some(idx);
                }
            }
        }
    }
    token_bucket_index(token)
}

fn collect_distribution<T: Copy>(
    position: &TokenLogprob,
    support: &[T],
    mut token_index: impl FnMut(&str) -> Option<usize>,
) -> DiscreteDistribution<T> {
    let mut probabilities = vec![0.0; support.len()];

    if let Some(idx) = token_index(&position.token) {
        probabilities[idx] = position.logprob.exp();
    }

    for alternative in &position.top_alternatives {
        if let Some(idx) = token_index(&alternative.token) {
            probabilities[idx] = probabilities[idx].max(alternative.logprob.exp());
        }
    }

    let covered: f64 = probabilities.iter().sum();
    DiscreteDistribution::new(
        support
            .iter()
            .copied()
            .zip(probabilities)
            .map(|(value, probability)| WeightedValue { value, probability })
            .collect(),
        (1.0 - covered).max(0.0),
    )
}

fn previous_tokens_name_field(logprobs: &[TokenLogprob], position: usize, field: &str) -> bool {
    let start = position.saturating_sub(8);
    let context = logprobs[start..position]
        .iter()
        .map(|entry| entry.token.as_str())
        .collect::<String>()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
        .collect::<String>()
        .to_ascii_lowercase();
    context.contains(field)
}

fn pairwise_bucket_logprob_posterior(
    logprobs: &[TokenLogprob],
    selected_higher_ranked: PairwisePreferredSide,
    selected_ratio: f64,
) -> Option<PairwiseLogprobPosterior> {
    let selected_ratio_bucket = RatioBucket::from_ratio(selected_ratio)?;
    let selected_answer =
        PairwiseAnswer::observation(selected_higher_ranked, selected_ratio_bucket);

    let side_position = logprobs.iter().enumerate().find_map(|(idx, entry)| {
        (token_preferred_side(&entry.token) == Some(selected_higher_ranked)
            && previous_tokens_name_field(logprobs, idx, "higher_ranked"))
        .then_some(entry)
    })?;

    let selected_bucket_idx = selected_ratio_bucket.index();
    let bucket_position_idx = logprobs.iter().enumerate().find_map(|(idx, _entry)| {
        (token_bucket_index_at(logprobs, idx) == Some(selected_bucket_idx)
            && previous_tokens_name_field(logprobs, idx, "ratio_bucket"))
        .then_some(idx)
    })?;
    let bucket_position = &logprobs[bucket_position_idx];
    let bucket_previous_token = bucket_position_idx
        .checked_sub(1)
        .and_then(|idx| logprobs.get(idx))
        .map(|entry| entry.token.as_str());

    let higher_ranked_distribution = collect_distribution(
        side_position,
        &[PairwisePreferredSide::A, PairwisePreferredSide::B],
        |token| match token_preferred_side(token)? {
            PairwisePreferredSide::A => Some(0),
            PairwisePreferredSide::B => Some(1),
        },
    );
    let ratio_distribution = collect_distribution(bucket_position, RatioBucket::all(), |token| {
        token_bucket_index_with_previous(token, bucket_previous_token)
            .map(|idx| RatioBucket::ALL[idx].index())
    });

    let selected_idx = selected_ratio_bucket.index();
    let neighbor_indices: Vec<usize> = (0..RATIO_LADDER.len())
        .filter(|&idx| idx.abs_diff(selected_idx) <= 1)
        .collect();
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
    let confidence = ConfidenceSource::Logprob {
        entropy: answer_distribution.entropy(),
        top_prob,
        neighborhood_prob: neighborhood_prob.clamp(0.0, 1.0),
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

fn compact_bucket_output_logprobs(
    logprobs: &[TokenLogprob],
    selected_higher_ranked: PairwisePreferredSide,
    selected_ratio: f64,
) -> Option<Vec<TokenLogprob>> {
    let selected_ratio_bucket = RatioBucket::from_ratio(selected_ratio)?;
    let side_position = logprobs.iter().enumerate().find_map(|(idx, entry)| {
        (token_preferred_side(&entry.token) == Some(selected_higher_ranked)
            && previous_tokens_name_field(logprobs, idx, "higher_ranked"))
        .then_some(idx)
    })?;
    let bucket_position = logprobs.iter().enumerate().find_map(|(idx, _entry)| {
        (token_bucket_index_at(logprobs, idx) == Some(selected_ratio_bucket.index())
            && previous_tokens_name_field(logprobs, idx, "ratio_bucket"))
        .then_some(idx)
    })?;

    let mut compact = Vec::with_capacity(2);
    compact.push(logprobs[side_position].clone());
    if bucket_position != side_position {
        compact.push(logprobs[bucket_position].clone());
    }
    Some(compact)
}

fn fallback_stored_logprobs(
    response_logprobs: Option<&[TokenLogprob]>,
) -> Option<Vec<TokenLogprob>> {
    response_logprobs.map(|logprobs| truncate_output_logprobs(logprobs, 50))
}

/// Extract JSON object from response (handles models that add surrounding text).
fn extract_json(raw: &str) -> &str {
    let trimmed = raw.trim();

    // If it starts with {, assume it's already JSON
    if trimmed.starts_with('{') {
        // Find matching closing brace
        let mut depth = 0;
        let mut end_idx = 0;
        for (i, c) in trimmed.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end_idx = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if end_idx > 0 {
            return &trimmed[..end_idx];
        }
    }

    // Try to find JSON anywhere in the response
    if let Some(start) = trimmed.find('{') {
        let remainder = &trimmed[start..];
        let mut depth = 0;
        for (i, c) in remainder.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &remainder[..=i];
                    }
                }
                _ => {}
            }
        }
    }

    trimmed
}
// =============================================================================
// LLM comparison
// =============================================================================

/// Perform a pairwise comparison using the LLM.
pub async fn compare_pair(
    gateway: &dyn ChatGateway,
    cache: Option<&dyn PairwiseCache>,
    request: PairwiseComparisonRequest<'_>,
) -> Result<(PairwiseJudgement, ComparisonUsage), ComparisonError> {
    let prompt_instance = request.spec.prompt_instance();
    let cache_key = cache.map(|_| request.spec.cache_key());

    if let (Some(cache), Some(ref key)) = (cache, &cache_key) {
        match cache.get(key).await {
            Ok(Some(hit)) => {
                if let Some(judgement) = cached_to_judgement(&hit) {
                    let usage = ComparisonUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        provider_cost_nanodollars: 0,
                        provider_cost_is_estimate: false,
                        cached: true,
                        prompt_text: None,
                        question_text: None,
                        raw_output: None,
                        output_logprobs: None,
                        pairwise_logprob_posterior: None,
                    };
                    return Ok((judgement, usage));
                }
            }
            Ok(None) => {}
            Err(err) => {
                if request.cache_only {
                    return Err(ComparisonError::Cache(err));
                }
                warn!(error = %err, "Cache read failed; falling back to live comparison");
            }
        }
    }
    if request.cache_only {
        return Err(ComparisonError::CacheMiss(
            "cache_only is enabled and no cached judgement was found".to_string(),
        ));
    }

    let mut chat_request = ChatRequest::new(
        ChatModel::openrouter(request.spec.model),
        prompt_instance.to_messages(),
        request.attribution,
    )
    .max_tokens(PAIRWISE_MAX_OUTPUT_TOKENS_DEFAULT);
    if should_use_json_mode(request.spec.model) {
        chat_request = chat_request.json();
    }
    chat_request = chat_request.max_tokens(pairwise_max_output_tokens(request.spec.model));
    if model_supports_logprobs(request.spec.model) {
        chat_request = chat_request.with_logprobs(pairwise_logprobs_top_n());
    }

    let wants_bucket_pmf = request.spec.attribute.prompt_template_slug
        == Some("canonical_bucket_v1")
        && chat_request.logprobs;
    let max_live_attempts = if wants_bucket_pmf {
        PAIRWISE_BUCKET_LOGPROB_MAX_ATTEMPTS
    } else {
        1
    };
    let prompt_text = format!(
        "{}\n---\n{}",
        prompt_instance.system.as_str(),
        prompt_instance.user.as_str()
    );

    let mut input_tokens_total = 0u32;
    let mut output_tokens_total = 0u32;
    let mut provider_cost_total = 0i64;
    let mut provider_cost_is_estimate = false;

    for attempt_index in 0..max_live_attempts {
        let response = gateway.chat(chat_request.clone()).await?;
        input_tokens_total = input_tokens_total.saturating_add(response.input_tokens);
        output_tokens_total = output_tokens_total.saturating_add(response.output_tokens);
        provider_cost_total = provider_cost_total.saturating_add(response.cost_nanodollars);
        provider_cost_is_estimate |= response.cost_is_estimate;

        let mut usage = ComparisonUsage {
            input_tokens: input_tokens_total,
            output_tokens: output_tokens_total,
            provider_cost_nanodollars: provider_cost_total,
            provider_cost_is_estimate,
            cached: false,
            prompt_text: Some(prompt_text.clone()),
            question_text: Some(request.spec.attribute.prompt.to_string()),
            raw_output: Some(response.content.clone()),
            output_logprobs: None,
            pairwise_logprob_posterior: None,
        };

        match parse_pairwise_response(&response.content, response.output_logprobs.as_deref()) {
            Ok(judgement) => {
                if let PairwiseJudgement::Observation {
                    higher_ranked,
                    ratio,
                    ..
                } = &judgement
                {
                    let selected_side = match higher_ranked {
                        HigherRanked::A => PairwisePreferredSide::A,
                        HigherRanked::B => PairwisePreferredSide::B,
                    };
                    let raw_logprobs = response.output_logprobs.as_deref();
                    if request.spec.attribute.prompt_template_slug == Some("canonical_bucket_v1") {
                        usage.pairwise_logprob_posterior = raw_logprobs.and_then(|logprobs| {
                            pairwise_bucket_logprob_posterior(logprobs, selected_side, *ratio)
                        });
                        usage.output_logprobs = raw_logprobs
                            .and_then(|logprobs| {
                                compact_bucket_output_logprobs(logprobs, selected_side, *ratio)
                            })
                            .or_else(|| fallback_stored_logprobs(raw_logprobs));
                    } else {
                        usage.pairwise_logprob_posterior = raw_logprobs.and_then(|logprobs| {
                            pairwise_logprob_posterior(
                                logprobs,
                                selected_side,
                                *ratio,
                                RATIO_LADDER,
                            )
                        });
                        usage.output_logprobs = fallback_stored_logprobs(raw_logprobs);
                    }
                } else {
                    usage.output_logprobs =
                        fallback_stored_logprobs(response.output_logprobs.as_deref());
                }

                let should_retry_for_pmf = wants_bucket_pmf
                    && matches!(judgement, PairwiseJudgement::Observation { .. })
                    && usage.pairwise_logprob_posterior.is_none()
                    && attempt_index + 1 < max_live_attempts;
                if should_retry_for_pmf {
                    continue;
                }

                if let (Some(cache), Some(ref key)) = (cache, &cache_key) {
                    let entry = judgement_to_cached(&judgement, &usage);
                    let _ = cache.put(key, &entry).await;
                }
                return Ok((judgement, usage));
            }
            Err(ComparisonError::Parse(e)) => {
                usage.output_logprobs =
                    fallback_stored_logprobs(response.output_logprobs.as_deref());
                warn!(error = %e, "Failed to parse pairwise JSON response; treating as refusal");
                let judgement = PairwiseJudgement::Refused;
                if let (Some(cache), Some(ref key)) = (cache, &cache_key) {
                    let entry = judgement_to_cached(&judgement, &usage);
                    let _ = cache.put(key, &entry).await;
                }
                return Ok((judgement, usage));
            }
            Err(e) => return Err(e),
        }
    }

    unreachable!("live comparison loop always returns or errors")
}

/// Conservative estimate of input tokens for a single pairwise comparison prompt.
///
/// Used to reserve credits before executing the rerank. Overestimation is OK.
pub fn estimate_pairwise_input_tokens(
    attribute_name: &str,
    attribute_prompt: &str,
    prompt_template_slug: Option<&str>,
    entity_a_text: &str,
    entity_b_text: &str,
) -> u32 {
    let entity_a = EntityRef::with_context("A", entity_a_text);
    let entity_b = EntityRef::with_context("B", entity_b_text);
    let template = prompt_template_slug
        .and_then(prompt_by_slug)
        .unwrap_or(DEFAULT_PROMPT);
    let prompt_instance = template.render(attribute_name, attribute_prompt, entity_a, entity_b);
    let messages = prompt_instance.to_messages();

    // Count tokens in message content and add a small overhead per message
    // to account for role/formatting tokens.
    let content_tokens: usize = messages.iter().map(|m| count_tokens(&m.content)).sum();
    let overhead_tokens = 8usize.saturating_mul(messages.len());
    (content_tokens + overhead_tokens) as u32
}

// =============================================================================
// Mapping functions
// =============================================================================

/// Map confidence ∈ 0..=1 to variance in log-space.
///
/// High confidence → low variance (tight observation).
/// Low confidence → high variance (noisy observation).
pub fn confidence_to_variance(confidence: f64) -> f64 {
    let c = confidence.clamp(0.0, 1.0);
    let sigma = SIGMA_MAX - c * (SIGMA_MAX - SIGMA_MIN);
    sigma * sigma
}

/// Compute signed log-ratio from judgement.
///
/// Returns `ln_ratio` where:
/// - Positive means entity i scores higher than entity j
/// - Negative means entity j scores higher than entity i
pub fn compute_ln_ratio(higher_ranked: HigherRanked, ratio: f64) -> f64 {
    let ln_r = ratio.ln();
    match higher_ranked {
        HigherRanked::A => ln_r,  // A > B means i > j
        HigherRanked::B => -ln_r, // B > A means j > i
    }
}

fn cached_to_judgement(cached: &CachedJudgement) -> Option<PairwiseJudgement> {
    if cached.refused {
        return Some(PairwiseJudgement::Refused);
    }
    let higher = cached.higher_ranked.as_deref()?;
    let ratio = cached.ratio?;
    let confidence = cached.confidence?;
    if !(1.0..=26.0).contains(&ratio) {
        return None;
    }
    let higher_ranked = match higher.to_uppercase().as_str() {
        "A" => HigherRanked::A,
        "B" => HigherRanked::B,
        _ => return None,
    };
    Some(PairwiseJudgement::Observation {
        higher_ranked,
        ratio,
        confidence,
    })
}

/// Whether to request logprobs for this model.
///
/// Logprobs are only useful for non-reasoning models that support them.
/// - Anthropic: no logprobs via OpenRouter
/// - Reasoning models: output tokens are post-reasoning, so logprob
///   distribution doesn't reflect the actual deliberation
/// - `:thinking` suffix: OpenRouter convention for reasoning variants
fn model_supports_logprobs(model: &str) -> bool {
    // Anthropic never exposes logprobs via OpenRouter.
    if model.starts_with("anthropic/") {
        return false;
    }

    // `:thinking` suffix is OpenRouter's convention for reasoning variants.
    if model.contains(":thinking") {
        return false;
    }

    // Known reasoning model families by prefix/substring.
    let model_lower = model.to_lowercase();
    let is_reasoning = model_lower.starts_with("openai/o1")
        || model_lower.starts_with("openai/o3")
        || model_lower.starts_with("openai/o4")
        || model_lower.contains("deepseek-r1")
        || model_lower.contains("/qwq")
        || model_lower.contains("-thinking")
        || model_lower.contains("reasoning");

    if is_reasoning {
        return false;
    }

    // GPT-5.4 family: logprobs request causes OpenAI backend 502 via OpenRouter.
    // The upstream API crashes rather than returning logprobs for these models.
    if model_lower.starts_with("openai/gpt-5.4") {
        return false;
    }
    // Gemini 3.1 Pro Preview is reasoning-mandatory on OpenRouter and does not
    // advertise logprob/top_logprob support in live provider metadata. Treat
    // current and future Gemini Pro reasoning previews conservatively until a
    // non-reasoning endpoint explicitly exposes token logprobs.
    if model_lower.starts_with("google/gemini-3.1-pro")
        || model_lower.starts_with("google/gemini-3-pro")
    {
        return false;
    }

    true
}

fn should_use_json_mode(model: &str) -> bool {
    if std::env::var("CARDINAL_FORCE_JSON_MODE")
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    {
        return true;
    }

    model.starts_with("openai/") || !model.contains('/')
}

fn judgement_to_cached(judgement: &PairwiseJudgement, usage: &ComparisonUsage) -> CachedJudgement {
    match judgement {
        PairwiseJudgement::Refused => CachedJudgement {
            higher_ranked: None,
            ratio: None,
            confidence: None,
            refused: true,
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
            provider_cost_nanodollars: Some(usage.provider_cost_nanodollars),
        },
        PairwiseJudgement::Observation {
            higher_ranked,
            ratio,
            confidence,
        } => CachedJudgement {
            higher_ranked: Some(match higher_ranked {
                HigherRanked::A => "A".to_string(),
                HigherRanked::B => "B".to_string(),
            }),
            ratio: Some(*ratio),
            confidence: Some(*confidence),
            refused: false,
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
            provider_cost_nanodollars: Some(usage.provider_cost_nanodollars),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::{ChatResponse, FinishReason};
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::time::Duration;

    struct VecGateway {
        responses: Mutex<VecDeque<ChatResponse>>,
    }

    #[async_trait::async_trait]
    impl ChatGateway for VecGateway {
        async fn chat(
            &self,
            _req: ChatRequest,
        ) -> Result<ChatResponse, crate::gateway::ProviderError> {
            Ok(self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("test response"))
        }
    }

    fn response(content: &str, output_logprobs: Option<Vec<TokenLogprob>>) -> ChatResponse {
        ChatResponse {
            content: content.to_string(),
            reasoning: None,
            reasoning_tokens: None,
            input_tokens: 10,
            output_tokens: 5,
            cost_nanodollars: 100,
            cost_is_estimate: false,
            upstream_cost_nanodollars: None,
            latency: Duration::from_millis(1),
            finish_reason: FinishReason::Stop,
            output_logprobs,
            cache_read_tokens: None,
            cache_write_tokens: None,
        }
    }

    fn bucket_output_logprobs() -> Vec<TokenLogprob> {
        vec![
            TokenLogprob {
                token: "higher".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "_".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "ranked".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "\":\"".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "A".to_string(),
                logprob: -0.2,
                top_alternatives: vec![crate::gateway::TokenAlternative {
                    token: "B".to_string(),
                    logprob: -1.8,
                }],
            },
            TokenLogprob {
                token: "ratio".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "_".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "bucket".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "\":".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "5".to_string(),
                logprob: -0.3,
                top_alternatives: vec![crate::gateway::TokenAlternative {
                    token: "4".to_string(),
                    logprob: -1.4,
                }],
            },
        ]
    }

    #[test]
    fn test_parse_valid_json() {
        let raw = r#"{"higher_ranked": "A", "ratio": 1.3, "confidence": 0.74}"#;
        let result = parse_pairwise_response(raw, None).unwrap();
        match result {
            PairwiseJudgement::Observation {
                higher_ranked,
                ratio,
                confidence,
            } => {
                assert_eq!(higher_ranked, HigherRanked::A);
                assert!((ratio - 1.3).abs() < 0.001);
                assert!((confidence - 0.74).abs() < 0.001);
            }
            _ => panic!("Expected Observation"),
        }
    }

    #[test]
    fn test_parse_ratio_bucket_json() {
        let raw = r#"{"higher_ranked": "B", "ratio_bucket": 9, "confidence": 0.82}"#;
        let result = parse_pairwise_response(raw, None).unwrap();
        match result {
            PairwiseJudgement::Observation {
                higher_ranked,
                ratio,
                confidence,
            } => {
                assert_eq!(higher_ranked, HigherRanked::B);
                assert!((ratio - 3.1).abs() < 0.001);
                assert!((confidence - 0.82).abs() < 0.001);
            }
            _ => panic!("Expected Observation"),
        }
    }

    #[test]
    fn test_parse_refused() {
        let raw = r#"{"refused": true}"#;
        let result = parse_pairwise_response(raw, None).unwrap();
        assert!(matches!(result, PairwiseJudgement::Refused));
    }

    #[test]
    fn test_parse_with_surrounding_text() {
        let raw = r#"Here's my evaluation:
{"higher_ranked": "B", "ratio": 2.5, "confidence": 0.9}
That's my assessment."#;
        let result = parse_pairwise_response(raw, None).unwrap();
        match result {
            PairwiseJudgement::Observation {
                higher_ranked,
                ratio,
                ..
            } => {
                assert_eq!(higher_ranked, HigherRanked::B);
                assert!((ratio - 2.5).abs() < 0.001);
            }
            _ => panic!("Expected Observation"),
        }
    }

    #[test]
    fn test_parse_rejects_missing_confidence_even_with_logprobs() {
        let raw = r#"{"higher_ranked":"A","ratio":2.5}"#;
        let logprobs = vec![
            TokenLogprob {
                token: "\"A\"".to_string(),
                logprob: -0.1,
                top_alternatives: vec![crate::gateway::TokenAlternative {
                    token: "\"B\"".to_string(),
                    logprob: -2.3,
                }],
            },
            TokenLogprob {
                token: "2.5".to_string(),
                logprob: -0.22,
                top_alternatives: vec![crate::gateway::TokenAlternative {
                    token: "2.1".to_string(),
                    logprob: -1.61,
                }],
            },
        ];

        let err = parse_pairwise_response(raw, Some(&logprobs)).unwrap_err();
        assert!(
            matches!(err, ComparisonError::Parse(message) if message.contains("missing 'confidence'"))
        );
    }

    #[test]
    fn test_bucket_logprob_posterior_uses_ratio_bucket_field() {
        let logprobs = vec![
            TokenLogprob {
                token: "{\"".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "higher".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "_".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "ranked".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "\":\"".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "B".to_string(),
                logprob: -0.2,
                top_alternatives: vec![crate::gateway::TokenAlternative {
                    token: "A".to_string(),
                    logprob: -1.8,
                }],
            },
            TokenLogprob {
                token: "\",\"".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "ratio".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "_".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "bucket".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "\":".to_string(),
                logprob: -0.01,
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "9".to_string(),
                logprob: -0.3,
                top_alternatives: vec![
                    crate::gateway::TokenAlternative {
                        token: "8".to_string(),
                        logprob: -1.4,
                    },
                    crate::gateway::TokenAlternative {
                        token: "10".to_string(),
                        logprob: -2.0,
                    },
                ],
            },
        ];

        let posterior = pairwise_bucket_logprob_posterior(&logprobs, PairwisePreferredSide::B, 3.1)
            .expect("posterior");
        assert_eq!(posterior.selected_ratio_bucket, RatioBucket::R09);
        assert!(posterior.answer_distribution.support_probability() > 0.0);
        assert!(posterior.confidence.as_scalar() > 0.0);
        assert!(posterior.probability_negative() > posterior.probability_positive());
    }

    #[test]
    fn test_bucket_logprob_posterior_handles_split_two_digit_bucket() {
        let mut logprobs = bucket_output_logprobs();
        let bucket = logprobs.last_mut().expect("bucket token");
        bucket.token = "1".to_string();
        bucket.logprob = -0.02;
        bucket.top_alternatives = vec![];
        logprobs.push(TokenLogprob {
            token: "2".to_string(),
            logprob: -0.3,
            top_alternatives: vec![
                crate::gateway::TokenAlternative {
                    token: "3".to_string(),
                    logprob: -0.9,
                },
                crate::gateway::TokenAlternative {
                    token: "1".to_string(),
                    logprob: -1.4,
                },
                crate::gateway::TokenAlternative {
                    token: "6".to_string(),
                    logprob: -2.0,
                },
            ],
        });

        let posterior = pairwise_bucket_logprob_posterior(&logprobs, PairwisePreferredSide::A, 6.8)
            .expect("posterior");
        let compact = compact_bucket_output_logprobs(&logprobs, PairwisePreferredSide::A, 6.8)
            .expect("compact logprobs");

        assert_eq!(posterior.selected_ratio_bucket, RatioBucket::R12);
        assert!(
            posterior
                .ratio_distribution
                .probability_of(|bucket| *bucket == RatioBucket::R12)
                > 0.0
        );
        assert!(
            posterior
                .ratio_distribution
                .probability_of(|bucket| *bucket == RatioBucket::R13)
                > 0.0
        );
        assert_eq!(compact.len(), 2);
        assert_eq!(compact[1].token, "2");
    }

    #[test]
    fn test_compact_bucket_output_logprobs_keeps_decisive_positions_only() {
        let logprobs = bucket_output_logprobs();
        let compact = compact_bucket_output_logprobs(&logprobs, PairwisePreferredSide::A, 1.5)
            .expect("compact logprobs");

        assert_eq!(compact.len(), 2);
        assert_eq!(compact[0].token, "A");
        assert_eq!(compact[1].token, "5");
    }

    #[tokio::test]
    async fn compare_pair_retries_bucket_prompt_until_pmf_available() {
        let content = r#"{"higher_ranked":"A","ratio_bucket":5,"confidence":0.85}"#;
        let gateway = VecGateway {
            responses: Mutex::new(VecDeque::from([
                response(content, None),
                response(content, Some(bucket_output_logprobs())),
            ])),
        };
        let request = PairwiseComparisonRequest {
            spec: PairwiseComparisonSpec {
                model: "google/gemma-4-26b-a4b-it",
                attribute: PairwiseComparisonAttribute {
                    id: "pmf",
                    prompt: "PMF test",
                    prompt_template_slug: Some("canonical_bucket_v1"),
                },
                entity_a: PairwiseComparisonEntity { id: "a", text: "A" },
                entity_b: PairwiseComparisonEntity { id: "b", text: "B" },
            },
            cache_only: false,
            attribution: Attribution::new("test::bucket_retry"),
        };

        let (judgement, usage) = compare_pair(&gateway, None, request).await.unwrap();
        assert!(matches!(judgement, PairwiseJudgement::Observation { .. }));
        assert_eq!(usage.input_tokens, 20);
        assert_eq!(usage.output_tokens, 10);
        assert_eq!(usage.provider_cost_nanodollars, 200);
        assert!(usage.output_logprobs.is_some());
        assert_eq!(usage.output_logprobs.as_ref().unwrap().len(), 2);
        assert!(usage.pairwise_logprob_posterior.is_some());
        assert_eq!(gateway.responses.lock().unwrap().len(), 0);
    }

    #[test]
    fn test_confidence_to_variance() {
        // High confidence → low variance
        let var_high = confidence_to_variance(1.0);
        assert!((var_high - 0.04).abs() < 0.001); // SIGMA_MIN^2

        // Low confidence → high variance
        let var_low = confidence_to_variance(0.0);
        assert!((var_low - 4.0).abs() < 0.001); // SIGMA_MAX^2

        // Mid confidence
        let var_mid = confidence_to_variance(0.5);
        assert!(var_mid > var_high && var_mid < var_low);
    }

    #[test]
    fn test_compute_ln_ratio() {
        let ln_a = compute_ln_ratio(HigherRanked::A, 2.0);
        assert!(ln_a > 0.0);

        let ln_b = compute_ln_ratio(HigherRanked::B, 2.0);
        assert!(ln_b < 0.0);

        assert!((ln_a + ln_b).abs() < 0.001);
    }

    #[test]
    fn test_model_supports_logprobs() {
        // Anthropic: no logprobs via OpenRouter
        assert!(!model_supports_logprobs("anthropic/claude-opus-4-6"));
        assert!(!model_supports_logprobs("anthropic/claude-sonnet-4.6"));
        assert!(!model_supports_logprobs("anthropic/claude-sonnet-4"));
        assert!(!model_supports_logprobs("anthropic/claude-haiku-4.5"));

        // Reasoning models: logprobs don't reflect deliberation
        assert!(!model_supports_logprobs("openai/o3"));
        assert!(!model_supports_logprobs("openai/o3-pro"));
        assert!(!model_supports_logprobs("openai/o4-mini"));
        assert!(!model_supports_logprobs("openai/o4-mini-high"));
        assert!(!model_supports_logprobs("openai/o1"));
        assert!(!model_supports_logprobs("openai/o1-pro"));
        assert!(!model_supports_logprobs("deepseek/deepseek-r1"));
        assert!(!model_supports_logprobs("deepseek/deepseek-r1-0528"));
        assert!(!model_supports_logprobs("qwen/qwq-32b"));

        // :thinking variants
        assert!(!model_supports_logprobs(
            "anthropic/claude-3.7-sonnet:thinking"
        ));
        assert!(!model_supports_logprobs("moonshotai/kimi-k2-thinking"));
        assert!(!model_supports_logprobs(
            "qwen/qwen3-235b-a22b-thinking-2507"
        ));
        assert!(!model_supports_logprobs("baidu/ernie-4.5-21b-a3b-thinking"));

        // GPT-5.4 family: logprobs crash OpenAI backend via OpenRouter
        assert!(!model_supports_logprobs("openai/gpt-5.4-mini"));
        assert!(!model_supports_logprobs("openai/gpt-5.4"));
        assert!(!model_supports_logprobs("openai/gpt-5.4-nano"));

        // Non-reasoning models: YES logprobs
        assert!(model_supports_logprobs("openai/gpt-4.1"));
        assert!(model_supports_logprobs("openai/gpt-4.1-mini"));
        assert!(model_supports_logprobs("openai/gpt-5-mini"));
        assert!(model_supports_logprobs("openai/gpt-5.2-pro"));
        assert!(model_supports_logprobs("google/gemini-2.5-pro"));
        assert!(model_supports_logprobs("google/gemini-2.5-flash"));
        assert!(!model_supports_logprobs("google/gemini-3.1-pro-preview"));
        assert!(model_supports_logprobs("moonshotai/kimi-k2-0905"));
        assert!(model_supports_logprobs("deepseek/deepseek-chat"));
        assert!(model_supports_logprobs("deepseek/deepseek-v3.2"));
    }

    #[test]
    fn test_should_use_json_mode_for_openai_and_local_models() {
        assert!(should_use_json_mode("openai/gpt-4.1"));
        assert!(should_use_json_mode("gemma4:31b"));
        assert!(!should_use_json_mode("google/gemma-4-31b-it"));
    }
}
