//! LLM pairwise comparison logic for reranking.
//!
//! Implements the contract between LLM JSON responses and solver observations.

use serde::Deserialize;
use tracing::warn;

use crate::cache::{CacheError, CachedJudgement, PairwiseCache, PairwiseCacheKey};
use crate::gateway::{
    Attribution, ChatModel, ChatRequest, ProviderError, ProviderGateway, UsageSink,
};
use crate::text_chunking::count_tokens;

use crate::prompts::{prompt_by_slug, EntityRef, DEFAULT_PROMPT};

use super::types::{HigherRanked, PairwiseJudgement};

// =============================================================================
// Constants
// =============================================================================

/// Minimum variance (high confidence).
pub const SIGMA_MIN: f64 = 0.2;
/// Maximum variance (low confidence).
pub const SIGMA_MAX: f64 = 2.0;

/// Hard cap on generation for a pairwise judgement.
///
/// Keeps costs bounded and ensures responses stay in the small JSON schema.
pub const PAIRWISE_MAX_OUTPUT_TOKENS_DEFAULT: u32 = 128;
pub const PAIRWISE_MAX_OUTPUT_TOKENS_GPT5: u32 = 512;

pub fn pairwise_max_output_tokens(model: &str) -> u32 {
    // GPT-5 family tends to spend ~128 tokens on internal reasoning before emitting any
    // visible output; a 128-token cap can yield empty `content` on OpenRouter.
    if model.starts_with("openai/gpt-5") {
        PAIRWISE_MAX_OUTPUT_TOKENS_GPT5
    } else {
        PAIRWISE_MAX_OUTPUT_TOKENS_DEFAULT
    }
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
#[derive(Debug, Clone, Copy)]
pub struct ComparisonUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub provider_cost_nanodollars: i64,
    pub cached: bool,
}

/// Parse LLM response JSON into a PairwiseJudgement.
pub fn parse_pairwise_response(raw: &str) -> Result<PairwiseJudgement, ComparisonError> {
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
    let ratio = parsed
        .ratio
        .ok_or_else(|| ComparisonError::Parse("missing 'ratio'".into()))?;
    let confidence = parsed
        .confidence
        .ok_or_else(|| ComparisonError::Parse("missing 'confidence'".into()))?;

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

    Ok(PairwiseJudgement::Observation {
        higher_ranked,
        ratio,
        confidence: confidence.clamp(0.0, 1.0),
    })
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
#[allow(clippy::too_many_arguments)]
pub async fn compare_pair<U: UsageSink>(
    gateway: &ProviderGateway<U>,
    cache: Option<&dyn PairwiseCache>,
    cache_only: bool,
    model: &str,
    attribute_name: &str,
    attribute_prompt: &str,
    prompt_template_slug: Option<&str>,
    entity_a_id: &str,
    entity_a_text: &str,
    entity_b_id: &str,
    entity_b_text: &str,
    attribution: Attribution,
) -> Result<(PairwiseJudgement, ComparisonUsage), ComparisonError> {
    let entity_a = EntityRef::with_context("A", entity_a_text);
    let entity_b = EntityRef::with_context("B", entity_b_text);

    let template = prompt_template_slug
        .and_then(prompt_by_slug)
        .unwrap_or(DEFAULT_PROMPT);
    let prompt_slug = template.slug;
    let template_hash = blake3::hash(format!("{}\n{}", template.system, template.user).as_bytes())
        .to_hex()
        .to_string();
    let cache_key = cache.map(|_| {
        PairwiseCacheKey::new(
            model,
            prompt_slug,
            &template_hash,
            attribute_name,
            attribute_prompt,
            entity_a_id,
            entity_a_text,
            entity_b_id,
            entity_b_text,
        )
    });

    if let (Some(cache), Some(ref key)) = (cache, &cache_key) {
        match cache.get(key).await {
            Ok(Some(hit)) => {
                if let Some(judgement) = cached_to_judgement(&hit) {
                    let usage = ComparisonUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        provider_cost_nanodollars: 0,
                        cached: true,
                    };
                    return Ok((judgement, usage));
                }
            }
            Ok(None) => {}
            Err(err) => {
                if cache_only {
                    return Err(ComparisonError::Cache(err));
                }
                warn!(error = %err, "Cache read failed; falling back to live comparison");
            }
        }
    }
    if cache_only {
        return Err(ComparisonError::CacheMiss(
            "cache_only is enabled and no cached judgement was found".to_string(),
        ));
    }

    let prompt_instance = template.render(attribute_name, attribute_prompt, entity_a, entity_b);

    let mut request = ChatRequest::new(
        ChatModel::openrouter(model),
        prompt_instance.to_messages(),
        attribution,
    )
    .max_tokens(pairwise_max_output_tokens(model));
    // Only OpenAI models reliably support response_format=json_object via OpenRouter.
    if model.starts_with("openai/") {
        request = request.json();
    }

    let response = gateway.chat(request).await?;

    let usage = ComparisonUsage {
        input_tokens: response.input_tokens,
        output_tokens: response.output_tokens,
        provider_cost_nanodollars: response.cost_nanodollars,
        cached: false,
    };

    match parse_pairwise_response(&response.content) {
        Ok(judgement) => {
            if let (Some(cache), Some(ref key)) = (cache, &cache_key) {
                let entry = judgement_to_cached(&judgement, &usage);
                let _ = cache.put(key, &entry).await;
            }
            Ok((judgement, usage))
        }
        Err(ComparisonError::Parse(e)) => {
            warn!(error = %e, "Failed to parse pairwise JSON response; treating as refusal");
            let judgement = PairwiseJudgement::Refused;
            if let (Some(cache), Some(ref key)) = (cache, &cache_key) {
                let entry = judgement_to_cached(&judgement, &usage);
                let _ = cache.put(key, &entry).await;
            }
            Ok((judgement, usage))
        }
        Err(e) => Err(e),
    }
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

    #[test]
    fn test_parse_valid_json() {
        let raw = r#"{"higher_ranked": "A", "ratio": 1.3, "confidence": 0.74}"#;
        let result = parse_pairwise_response(raw).unwrap();
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
    fn test_parse_refused() {
        let raw = r#"{"refused": true}"#;
        let result = parse_pairwise_response(raw).unwrap();
        assert!(matches!(result, PairwiseJudgement::Refused));
    }

    #[test]
    fn test_parse_with_surrounding_text() {
        let raw = r#"Here's my evaluation:
{"higher_ranked": "B", "ratio": 2.5, "confidence": 0.9}
That's my assessment."#;
        let result = parse_pairwise_response(raw).unwrap();
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
}
