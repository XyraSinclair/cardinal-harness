//! Simple single-attribute reranking.
//!
//! This is syntactic sugar over multi-attribute reranking with a single attribute.
//! Guarantees identical semantics while providing a simpler API surface.

use std::sync::Arc;

use crate::cache::PairwiseCache;
use crate::gateway::{Attribution, ProviderGateway, UsageSink};

use super::model_policy::ModelPolicy;
use super::multi::{multi_rerank, MultiRerankError};
use super::types::{
    MultiRerankAttributeSpec, MultiRerankEntity, MultiRerankRequest, MultiRerankTopKSpec,
    RerankMeta, RerankRequest, RerankResponse, RerankResult,
};

// =============================================================================
// Simple rerank
// =============================================================================

/// Convert a simple rerank request into a multi-rerank request.
pub fn to_multi_request(req: &RerankRequest) -> MultiRerankRequest {
    // Convert documents to entities
    let entities: Vec<MultiRerankEntity> = req
        .documents
        .iter()
        .map(|doc| MultiRerankEntity {
            id: doc.id.clone(),
            text: doc.text.clone(),
        })
        .collect();

    // Build attribute prompt, optionally incorporating query
    let attribute_prompt = if let Some(ref query) = req.query {
        format!("{} for query: '{}'", req.attribute_prompt, query)
    } else {
        req.attribute_prompt.clone()
    };

    // Single attribute with weight 1.0
    let attributes = vec![MultiRerankAttributeSpec {
        id: req.attribute_id.clone(),
        prompt: attribute_prompt,
        prompt_template_slug: None,
        weight: 1.0,
    }];

    // Top-k config
    let n = entities.len();
    let k = req.top_k.unwrap_or(n);
    let topk = MultiRerankTopKSpec {
        k,
        weight_exponent: 1.0, // No exponent effect with single attribute
        tolerated_error: req.tolerated_error,
        band_size: 5,
        effective_resistance_max_active: 64,
        stop_sigma_inflate: 1.25,
        stop_min_consecutive: 2,
    };

    MultiRerankRequest {
        entities,
        attributes,
        topk,
        gates: Vec::new(), // No gates in simple mode
        comparison_budget: req.comparison_budget,
        latency_budget_ms: req.latency_budget_ms,
        model: req.model.clone(),
        rater_id: req.rater_id.clone(),
        comparison_concurrency: req.comparison_concurrency,
        max_pair_repeats: req.max_pair_repeats,
    }
}

/// Run a single-attribute reranking session.
///
/// Internally converts to a multi-attribute request with one attribute.
/// If a cache is provided, cached pairwise judgements are reused.
pub async fn rerank<U: UsageSink>(
    gateway: Arc<ProviderGateway<U>>,
    cache: Option<&dyn PairwiseCache>,
    model_policy: Option<Arc<dyn ModelPolicy>>,
    req: RerankRequest,
    attribution: Attribution,
) -> Result<RerankResponse, MultiRerankError> {
    let multi_req = to_multi_request(&req);

    // Call multi_rerank for the single-attribute wrapper.
    let multi_resp =
        multi_rerank(gateway, cache, model_policy, multi_req, attribution, None).await?;

    // Map response to simple format
    let results: Vec<RerankResult> = multi_resp
        .entities
        .iter()
        .filter(|e| e.feasible)
        .map(|e| {
            // Get scores for our single attribute
            let attr_scores = e.attribute_scores.get(&req.attribute_id);

            RerankResult {
                id: e.id.clone(),
                rank: e.rank.unwrap_or(0),
                latent_mean: attr_scores.map(|s| s.latent_mean).unwrap_or(0.0),
                latent_std: attr_scores.map(|s| s.latent_std).unwrap_or(0.0),
                z_score: attr_scores.map(|s| s.z_score).unwrap_or(0.0),
                min_normalized: attr_scores.map(|s| s.min_normalized).unwrap_or(1.0),
                percentile: attr_scores.map(|s| s.percentile).unwrap_or(0.0),
            }
        })
        .collect();

    let meta = RerankMeta {
        topk_error: multi_resp.meta.global_topk_error,
        tolerated_error: multi_resp.meta.tolerated_error,
        comparisons_attempted: multi_resp.meta.comparisons_attempted,
        comparisons_used: multi_resp.meta.comparisons_used,
        comparisons_refused: multi_resp.meta.comparisons_refused,
        comparison_budget: multi_resp.meta.comparison_budget,
        latency_ms: multi_resp.meta.latency_ms,
        model_used: multi_resp.meta.model_used,
        rater_id_used: multi_resp.meta.rater_id_used,
        provider_input_tokens: multi_resp.meta.provider_input_tokens,
        provider_output_tokens: multi_resp.meta.provider_output_tokens,
        provider_cost_nanodollars: multi_resp.meta.provider_cost_nanodollars,
        stop_reason: multi_resp.meta.stop_reason,
    };

    Ok(RerankResponse { results, meta })
}
