//! Multi-attribute reranking / trait search orchestrator.
//!
//! Wires together:
//! - TraitSearchManager (multi-attribute top-k uncertainty logic)
//! - RatingEngine (per-attribute IRLS solver)
//! - Pairwise LLM comparisons on a ratio ladder with confidence
//!
//! Core loop:
//! 1. Solve per-attribute rating engines and build global utility + uncertainty.
//! 2. Estimate top-k error via TraitSearchManager::estimate_topk_error().
//! 3. If error > tolerated_error and budgets remain, call propose_batch()
//!    to select highest-value comparisons.
//! 4. For each proposed (attribute_id, i, j):
//!    - Call LLM with evaluator prompt.
//!    - Parse JSON `{higher_ranked, ratio, confidence}` or `{refused:true}`.
//!    - Map to (ln_ratio, variance) and feed into the corresponding engine.
//! 5. Repeat until top-k error ≤ tolerated_error or budget/latency hit.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use blake3;
use futures::stream::{self, StreamExt};

use crate::cache::{PairwiseCache, PairwiseCacheKey};
use crate::gateway::pricing as provider_pricing;
use crate::gateway::{Attribution, ProviderGateway, UsageSink};
use crate::prompts::{prompt_by_slug, DEFAULT_PROMPT};
use crate::text_chunking::count_tokens;

use crate::rating_engine::{
    AttributeParams, Config as EngineConfig, Observation, PlannerMode, RaterParams, RatingEngine,
};
use crate::trait_search::{
    AttributeConfig, GateSpec, TopKConfig, TraitSearchConfig, TraitSearchManager,
};

use super::comparison::{
    compare_pair, estimate_pairwise_input_tokens, pairwise_max_output_tokens, ComparisonError,
    PAIRWISE_MAX_OUTPUT_TOKENS_DEFAULT,
};
use super::model_policy::{ModelPolicy, ModelPolicyContext};
use super::options::RerankRunOptions;
use super::trace::{now_epoch_ms, ComparisonTrace, TraceError, TraceSink};
use super::types::{
    AttributeScoreSummary, HigherRanked, MultiRerankEntityResult, MultiRerankMeta,
    MultiRerankRequest, MultiRerankResponse, PairwiseJudgement, RerankStopReason,
};

// =============================================================================
// Constants
// =============================================================================

/// Default batch size for proposed comparisons.
const DEFAULT_BATCH_SIZE: usize = 32;

/// Default model if not specified.
/// GPT-5-mini: $0.25/1M input, $2.00/1M output - good balance of quality/cost for pairwise comparisons.
const DEFAULT_MODEL: &str = "openai/gpt-5-mini";

/// Default maximum number of comparisons to run concurrently.
const DEFAULT_COMPARISON_CONCURRENCY: usize = 8;
const MAX_COMPARISON_CONCURRENCY: usize = 64;

/// Rerank billing multiplier: 20% markup on top of provider cost.
const RERANK_MARKUP_NUM: i64 = 6;
const RERANK_MARKUP_DEN: i64 = 5;

// =============================================================================
// Error type
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum MultiRerankError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Trait search error: {0}")]
    TraitSearch(#[from] crate::trait_search::TraitSearchError),
    #[error("Rating engine error: {0}")]
    RatingEngine(String),
    #[error("Comparison error: {0}")]
    Comparison(#[from] ComparisonError),
    #[error("Trace error: {0}")]
    Trace(#[from] TraceError),
}

// =============================================================================
// Billing helpers
// =============================================================================

/// Apply 20% markup to provider cost, rounding up.
pub fn apply_rerank_markup(provider_cost_nanodollars: i64) -> i64 {
    if provider_cost_nanodollars <= 0 {
        return 0;
    }
    // ceil(cost * 6/5)
    (provider_cost_nanodollars.saturating_mul(RERANK_MARKUP_NUM) + (RERANK_MARKUP_DEN - 1))
        / RERANK_MARKUP_DEN
}

/// Conservative reservation estimate for a rerank request.
///
/// Reserves enough credits to cover the worst case (comparison_budget comparisons),
/// then refunds unused credits at completion.
#[derive(Debug, Clone, Copy)]
pub struct RerankChargeEstimate {
    pub comparison_budget: usize,
    pub input_tokens_per_comparison: u32,
    pub output_tokens_per_comparison: u32,
    pub provider_cost_max_nanodollars: i64,
    pub user_charge_max_nanodollars: i64,
}

pub fn estimate_max_rerank_charge(req: &MultiRerankRequest) -> RerankChargeEstimate {
    let n_entities = req.entities.len();
    let n_attributes = req.attributes.len();
    let comparison_budget = req
        .comparison_budget
        .unwrap_or_else(|| default_comparison_budget(n_entities, n_attributes));

    if n_entities < 2 || n_attributes == 0 || comparison_budget == 0 {
        return RerankChargeEstimate {
            comparison_budget,
            input_tokens_per_comparison: 0,
            output_tokens_per_comparison: PAIRWISE_MAX_OUTPUT_TOKENS_DEFAULT,
            provider_cost_max_nanodollars: 0,
            user_charge_max_nanodollars: 0,
        };
    }

    let model = req.model.as_deref().unwrap_or(DEFAULT_MODEL);
    let output_tokens_per_comparison = pairwise_max_output_tokens(model);

    // Worst-case attribute prompt: choose the largest prompt by token count (bounded, cheap).
    let (attr_id, attr_prompt, attr_template_slug) = req
        .attributes
        .iter()
        .map(|a| {
            (
                a.id.as_str(),
                a.prompt.as_str(),
                a.prompt_template_slug.as_deref(),
            )
        })
        .max_by_key(|(_, p, _)| count_tokens(p))
        .unwrap();

    // Worst-case entity texts: choose 2 longest by byte length (fast upper bound).
    let mut idxs: Vec<usize> = (0..n_entities).collect();
    idxs.sort_by_key(|&i| req.entities[i].text.len());
    idxs.reverse();
    let a_text = &req.entities[idxs[0]].text;
    let b_text = &req.entities[idxs[1]].text;

    let input_tokens_per_comparison =
        estimate_pairwise_input_tokens(attr_id, attr_prompt, attr_template_slug, a_text, b_text);

    // Provider cost per comparison at the capped max output tokens.
    let provider_cost_per_comparison = provider_pricing::chat_cost(
        model,
        input_tokens_per_comparison,
        output_tokens_per_comparison,
    );

    let provider_cost_max_nanodollars =
        provider_cost_per_comparison.saturating_mul(comparison_budget as i64);
    let user_charge_max_nanodollars = apply_rerank_markup(provider_cost_max_nanodollars);

    RerankChargeEstimate {
        comparison_budget,
        input_tokens_per_comparison,
        output_tokens_per_comparison,
        provider_cost_max_nanodollars,
        user_charge_max_nanodollars,
    }
}

// =============================================================================
// Orchestrator
// =============================================================================

/// Default comparison budget: 4 * n * num_attributes.
fn default_comparison_budget(n_entities: usize, n_attributes: usize) -> usize {
    4usize
        .saturating_mul(n_entities.max(1))
        .saturating_mul(n_attributes.max(1))
}

pub fn validate_multi_rerank_request(req: &MultiRerankRequest) -> Result<(), MultiRerankError> {
    if req.entities.len() < 2 {
        return Err(MultiRerankError::InvalidRequest(
            "entities must contain at least 2 items".into(),
        ));
    }
    if req.attributes.is_empty() {
        return Err(MultiRerankError::InvalidRequest(
            "attributes must not be empty".into(),
        ));
    }

    if req.topk.k == 0 {
        return Err(MultiRerankError::InvalidRequest(
            "topk.k must be >= 1".into(),
        ));
    }
    if req.topk.k > req.entities.len() {
        return Err(MultiRerankError::InvalidRequest(format!(
            "topk.k must be <= number of entities (k={}, n={})",
            req.topk.k,
            req.entities.len()
        )));
    }

    if matches!(req.comparison_budget, Some(0)) {
        return Err(MultiRerankError::InvalidRequest(
            "comparison_budget must be >= 1".into(),
        ));
    }

    if let Some(concurrency) = req.comparison_concurrency {
        if concurrency == 0 {
            return Err(MultiRerankError::InvalidRequest(
                "comparison_concurrency must be >= 1".into(),
            ));
        }
        if concurrency > MAX_COMPARISON_CONCURRENCY {
            return Err(MultiRerankError::InvalidRequest(format!(
                "comparison_concurrency must be <= {MAX_COMPARISON_CONCURRENCY}"
            )));
        }
    }

    if let Some(max) = req.max_pair_repeats {
        if max == 0 {
            return Err(MultiRerankError::InvalidRequest(
                "max_pair_repeats must be >= 1".into(),
            ));
        }
    }

    let mut entity_ids: HashSet<&str> = HashSet::new();
    for e in &req.entities {
        if !entity_ids.insert(e.id.as_str()) {
            return Err(MultiRerankError::InvalidRequest(format!(
                "duplicate entity id: {}",
                e.id
            )));
        }
    }

    let mut attribute_ids: HashSet<&str> = HashSet::new();
    for a in &req.attributes {
        if !a.weight.is_finite() {
            return Err(MultiRerankError::InvalidRequest(format!(
                "attribute weight must be finite (attribute_id={})",
                a.id
            )));
        }
        if let Some(slug) = a.prompt_template_slug.as_deref() {
            if prompt_by_slug(slug).is_none() {
                return Err(MultiRerankError::InvalidRequest(format!(
                    "unknown prompt_template_slug: {slug}"
                )));
            }
        }
        if !attribute_ids.insert(a.id.as_str()) {
            return Err(MultiRerankError::InvalidRequest(format!(
                "duplicate attribute id: {}",
                a.id
            )));
        }
    }

    for gate in &req.gates {
        if !attribute_ids.contains(gate.attribute_id.as_str()) {
            return Err(MultiRerankError::InvalidRequest(format!(
                "gate references unknown attribute: {}",
                gate.attribute_id
            )));
        }

        let unit = gate.unit.to_ascii_lowercase();
        match unit.as_str() {
            "latent" | "z" | "percentile" | "min_norm" => {}
            _ => {
                return Err(MultiRerankError::InvalidRequest(format!(
                    "unsupported gate unit: {}",
                    gate.unit
                )))
            }
        }

        match gate.op.as_str() {
            ">=" | "<=" => {}
            _ => {
                return Err(MultiRerankError::InvalidRequest(format!(
                    "unsupported gate op (expected \">=\" or \"<=\"): {}",
                    gate.op
                )))
            }
        }

        if unit == "percentile" && !(0.0..=1.0).contains(&gate.threshold) {
            return Err(MultiRerankError::InvalidRequest(format!(
                "percentile gate threshold must be in [0,1]: {}",
                gate.threshold
            )));
        }
    }

    Ok(())
}

fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Run a multi-attribute reranking session.
///
/// If a cache is provided, cached pairwise judgements are reused and new
/// judgements are written back to the cache.
pub async fn multi_rerank<U: UsageSink>(
    gateway: Arc<ProviderGateway<U>>,
    cache: Option<&dyn PairwiseCache>,
    model_policy: Option<Arc<dyn ModelPolicy>>,
    run_options: Option<&RerankRunOptions>,
    req: MultiRerankRequest,
    attribution: Attribution,
    cancel_flag: Option<&AtomicBool>,
) -> Result<MultiRerankResponse, MultiRerankError> {
    multi_rerank_with_trace(
        gateway,
        cache,
        model_policy,
        run_options,
        req,
        attribution,
        None,
        cancel_flag,
    )
    .await
}

/// Run a multi-attribute reranking session with optional trace output.
///
/// If a cache is provided, cached pairwise judgements are reused and new
/// judgements are written back to the cache.
#[allow(clippy::too_many_arguments)]
pub async fn multi_rerank_with_trace<U: UsageSink>(
    gateway: Arc<ProviderGateway<U>>,
    cache: Option<&dyn PairwiseCache>,
    model_policy: Option<Arc<dyn ModelPolicy>>,
    run_options: Option<&RerankRunOptions>,
    req: MultiRerankRequest,
    attribution: Attribution,
    trace: Option<&dyn TraceSink>,
    cancel_flag: Option<&AtomicBool>,
) -> Result<MultiRerankResponse, MultiRerankError> {
    validate_multi_rerank_request(&req)?;

    let n_entities = req.entities.len();
    let n_attributes = req.attributes.len();

    let comparison_budget = req
        .comparison_budget
        .unwrap_or_else(|| default_comparison_budget(n_entities, n_attributes));
    let latency_budget = req.latency_budget_ms.map(Duration::from_millis);
    let cache_only = run_options.map(|o| o.cache_only).unwrap_or(false);
    if cache_only && cache.is_none() {
        return Err(MultiRerankError::InvalidRequest(
            "cache_only requires a cache instance".into(),
        ));
    }

    let base_model = req.model.as_deref().unwrap_or(DEFAULT_MODEL);
    let rater_id = req.rater_id.as_deref().unwrap_or(base_model);
    let comparison_concurrency = req
        .comparison_concurrency
        .unwrap_or(DEFAULT_COMPARISON_CONCURRENCY);
    let max_pair_repeats = req.max_pair_repeats;

    // Build TraitSearchConfig
    let attributes_cfg: Vec<AttributeConfig> = req
        .attributes
        .iter()
        .map(|a| AttributeConfig::new(&a.id, a.weight))
        .collect();

    let topk_cfg = TopKConfig {
        k: req.topk.k,
        weight_exponent: req.topk.weight_exponent,
        tolerated_error: req.topk.tolerated_error,
        band_size: req.topk.band_size,
        effective_resistance_max_active: req.topk.effective_resistance_max_active,
        stop_sigma_inflate: req.topk.stop_sigma_inflate,
        stop_min_consecutive: req.topk.stop_min_consecutive,
    };

    // Convert gates (unit is validated in validate_multi_rerank_request()).
    let gates_cfg: Vec<GateSpec> = req
        .gates
        .iter()
        .map(|g| {
            GateSpec::new(
                &g.attribute_id,
                g.unit.to_ascii_lowercase(),
                &g.op,
                g.threshold,
            )
        })
        .collect();

    let config = TraitSearchConfig::new(n_entities, attributes_cfg, topk_cfg.clone(), gates_cfg);

    // Create engines for each attribute
    let mut engines: HashMap<String, RatingEngine> = HashMap::new();
    let mut raters: HashMap<String, RaterParams> = HashMap::new();
    raters.insert(rater_id.to_string(), RaterParams::default());

    let mut engine_cfg = EngineConfig::default();
    if let Some(options) = run_options {
        if let Some(seed) = options.rng_seed {
            engine_cfg.rng_seed = seed;
        }
    }
    engine_cfg.rank_weight_exponent = topk_cfg.weight_exponent;
    engine_cfg.top_k = Some(topk_cfg.k);
    if topk_cfg.k > 0 {
        let tail_weight =
            (1.0 / (topk_cfg.k as f64).powf(topk_cfg.weight_exponent)).clamp(0.05, 1.0);
        engine_cfg.tail_weight = tail_weight;
    }

    for attr in &req.attributes {
        let engine = RatingEngine::new(
            n_entities,
            AttributeParams::default(),
            raters.clone(),
            Some(engine_cfg.clone()),
        )
        .map_err(|e| MultiRerankError::RatingEngine(e.to_string()))?;
        engines.insert(attr.id.clone(), engine);
    }

    let mut manager = TraitSearchManager::new(config, engines)?;

    let start_time = Instant::now();

    let mut pair_repeats: HashMap<(usize, usize, usize), f64> = HashMap::new();

    let mut comparisons_attempted: usize = 0;
    let mut comparisons_used: usize = 0;
    let mut comparisons_refused: usize = 0;
    let mut comparisons_cached: usize = 0;

    let mut attribute_attempted: Vec<usize> = vec![0; n_attributes];
    let mut attribute_used: Vec<usize> = vec![0; n_attributes];

    let mut provider_input_tokens: u32 = 0;
    let mut provider_output_tokens: u32 = 0;
    let mut provider_cost_nanodollars: i64 = 0;

    let attr_id_to_index: HashMap<&str, usize> = req
        .attributes
        .iter()
        .enumerate()
        .map(|(idx, a)| (a.id.as_str(), idx))
        .collect();

    #[derive(Clone, Copy)]
    struct CompareTask {
        key: (usize, usize, usize),
        attr_idx: usize,
        i: usize,
        j: usize,
    }

    #[derive(Clone)]
    struct TraceFields {
        attribute_prompt_hash: String,
        prompt_template_slug: String,
        template_hash: String,
        entity_a_hash: String,
        entity_b_hash: String,
        cache_key_hash: String,
    }

    let mut refused_pairs: HashSet<(usize, usize, usize)> = HashSet::new();
    let mut models_used: HashSet<String> = HashSet::new();

    let stop_reason = 'rerank: loop {
        if let Some(flag) = cancel_flag {
            if flag.load(AtomicOrdering::Relaxed) {
                break 'rerank RerankStopReason::Cancelled;
            }
        }

        manager.recompute_global_state()?;
        let current_error = manager.estimate_topk_error();

        if manager.certified_stop() {
            break 'rerank RerankStopReason::CertifiedStop;
        }
        if current_error <= topk_cfg.tolerated_error {
            break 'rerank RerankStopReason::ToleratedErrorMet;
        }
        if comparisons_attempted >= comparison_budget {
            break 'rerank RerankStopReason::BudgetExhausted;
        }
        if let Some(limit) = latency_budget {
            if start_time.elapsed() >= limit {
                break 'rerank RerankStopReason::LatencyBudgetExceeded;
            }
        }

        let remaining_budget = comparison_budget.saturating_sub(comparisons_attempted);
        if remaining_budget == 0 {
            break 'rerank RerankStopReason::BudgetExhausted;
        }

        let batch_size = DEFAULT_BATCH_SIZE.min(remaining_budget);
        let proposal_request_size = (batch_size.saturating_mul(3)).max(batch_size);
        let proposals =
            manager.propose_batch(rater_id, proposal_request_size, PlannerMode::Hybrid)?;

        if proposals.is_empty() {
            break 'rerank RerankStopReason::NoProposals;
        }

        let mut batch_seen: HashSet<(usize, usize, usize)> = HashSet::new();
        let mut tasks: Vec<CompareTask> = Vec::with_capacity(batch_size);

        for proposal in proposals {
            let attr_id = proposal.attribute_id.as_str();
            let Some(&attr_idx) = attr_id_to_index.get(attr_id) else {
                continue;
            };

            let i = proposal.i;
            let j = proposal.j;
            if i >= req.entities.len() || j >= req.entities.len() {
                continue;
            }

            let (a, b) = if i <= j { (i, j) } else { (j, i) };
            let key = (attr_idx, a, b);

            if refused_pairs.contains(&key) {
                continue;
            }
            if !batch_seen.insert(key) {
                continue;
            }
            if let Some(max) = max_pair_repeats {
                if pair_repeats.get(&key).copied().unwrap_or(0.0) >= max as f64 {
                    continue;
                }
            }

            tasks.push(CompareTask {
                key,
                attr_idx,
                i,
                j,
            });

            if tasks.len() >= batch_size {
                break;
            }
        }

        if tasks.is_empty() {
            break 'rerank RerankStopReason::NoNewPairs;
        }

        let mut score_cache: HashMap<String, Vec<f64>> = HashMap::new();
        let mut std_cache: HashMap<String, Vec<f64>> = HashMap::new();
        if model_policy.is_some() {
            let mut attrs_in_batch: HashSet<&str> = HashSet::new();
            for task in &tasks {
                attrs_in_batch.insert(req.attributes[task.attr_idx].id.as_str());
            }
            for attr_id in attrs_in_batch {
                if let Some(scores) = manager.attribute_scores(attr_id) {
                    score_cache.insert(attr_id.to_string(), scores.to_vec());
                }
                if let Some(stds) = manager.attribute_std(attr_id) {
                    std_cache.insert(attr_id.to_string(), stds);
                }
            }
        }
        let score_cache = Arc::new(score_cache);
        let std_cache = Arc::new(std_cache);
        let base_model = base_model.to_string();
        let comparisons_attempted_snapshot = comparisons_attempted;
        let comparisons_used_snapshot = comparisons_used;

        let batch_results = stream::iter(tasks.into_iter().map(|task| {
            let gateway = gateway.clone();
            let attribution = attribution.clone();
            let policy = model_policy.clone();
            let attr = &req.attributes[task.attr_idx];
            let entity_a = &req.entities[task.i];
            let entity_b = &req.entities[task.j];
            let score_cache = score_cache.clone();
            let std_cache = std_cache.clone();
            let context = ModelPolicyContext {
                global_topk_error: current_error,
                comparisons_attempted: comparisons_attempted_snapshot,
                comparisons_used: comparisons_used_snapshot,
                attribute_comparisons_attempted: attribute_attempted[task.attr_idx],
                attribute_comparisons_used: attribute_used[task.attr_idx],
                attribute_id: &attr.id,
                i: task.i,
                j: task.j,
                attribute_scores: score_cache.get(&attr.id).map(|v| v.as_slice()),
                attribute_stds: std_cache.get(&attr.id).map(|v| v.as_slice()),
            };
            let selected_model = if let Some(policy) = policy.as_ref() {
                policy.select_model(&context)
            } else {
                base_model.clone()
            };
            async move {
                let judgement = compare_pair(
                    gateway.as_ref(),
                    cache,
                    cache_only,
                    &selected_model,
                    &attr.id,
                    &attr.prompt,
                    attr.prompt_template_slug.as_deref(),
                    &entity_a.id,
                    &entity_a.text,
                    &entity_b.id,
                    &entity_b.text,
                    attribution,
                )
                .await;
                (task, judgement, selected_model)
            }
        }))
        .buffer_unordered(comparison_concurrency)
        .collect::<Vec<_>>()
        .await;

        for (task, judgement, selected_model) in batch_results {
            comparisons_attempted += 1;
            let comparison_index = comparisons_attempted;
            attribute_attempted[task.attr_idx] =
                attribute_attempted[task.attr_idx].saturating_add(1);

            let attr = &req.attributes[task.attr_idx];
            let entity_a = &req.entities[task.i];
            let entity_b = &req.entities[task.j];

            models_used.insert(selected_model.clone());
            let attr_id = attr.id.as_str();

            let trace_fields = if trace.is_some() {
                let template = attr
                    .prompt_template_slug
                    .as_deref()
                    .and_then(prompt_by_slug)
                    .unwrap_or(DEFAULT_PROMPT);
                let prompt_slug = template.slug.to_string();
                let template_hash =
                    blake3::hash(format!("{}\n{}", template.system, template.user).as_bytes())
                        .to_hex()
                        .to_string();
                let cache_key = PairwiseCacheKey::new(
                    &selected_model,
                    &prompt_slug,
                    &template_hash,
                    &attr.id,
                    &attr.prompt,
                    &entity_a.id,
                    &entity_a.text,
                    &entity_b.id,
                    &entity_b.text,
                );
                Some(TraceFields {
                    attribute_prompt_hash: cache_key.attribute_prompt_hash,
                    prompt_template_slug: prompt_slug,
                    template_hash,
                    entity_a_hash: cache_key.entity_a_hash,
                    entity_b_hash: cache_key.entity_b_hash,
                    cache_key_hash: cache_key.key_hash,
                })
            } else {
                None
            };

            let build_trace = |cached: bool,
                               input_tokens: u32,
                               output_tokens: u32,
                               provider_cost_nanodollars: i64,
                               error: Option<String>| {
                let fields = trace_fields
                    .as_ref()
                    .expect("trace_fields set when trace active");
                ComparisonTrace {
                    timestamp_ms: now_epoch_ms(),
                    comparison_index,
                    attribute_id: attr.id.clone(),
                    attribute_index: task.attr_idx,
                    attribute_prompt_hash: fields.attribute_prompt_hash.clone(),
                    prompt_template_slug: fields.prompt_template_slug.clone(),
                    template_hash: fields.template_hash.clone(),
                    entity_a_id: entity_a.id.clone(),
                    entity_b_id: entity_b.id.clone(),
                    entity_a_index: task.i,
                    entity_b_index: task.j,
                    entity_a_hash: fields.entity_a_hash.clone(),
                    entity_b_hash: fields.entity_b_hash.clone(),
                    cache_key_hash: fields.cache_key_hash.clone(),
                    model: selected_model.clone(),
                    higher_ranked: None,
                    ratio: None,
                    confidence: None,
                    refused: false,
                    cached,
                    input_tokens,
                    output_tokens,
                    provider_cost_nanodollars,
                    error,
                }
            };

            match judgement {
                Ok((PairwiseJudgement::Refused, usage)) => {
                    if usage.cached {
                        comparisons_cached += 1;
                    }
                    provider_input_tokens =
                        provider_input_tokens.saturating_add(usage.input_tokens);
                    provider_output_tokens =
                        provider_output_tokens.saturating_add(usage.output_tokens);
                    provider_cost_nanodollars =
                        provider_cost_nanodollars.saturating_add(usage.provider_cost_nanodollars);
                    comparisons_refused += 1;
                    refused_pairs.insert(task.key);

                    if let Some(trace) = trace {
                        let mut event = build_trace(
                            usage.cached,
                            usage.input_tokens,
                            usage.output_tokens,
                            usage.provider_cost_nanodollars,
                            None,
                        );
                        event.refused = true;
                        trace.record(event)?;
                    }
                }
                Ok((
                    PairwiseJudgement::Observation {
                        higher_ranked,
                        ratio,
                        confidence,
                    },
                    usage,
                )) => {
                    if usage.cached {
                        comparisons_cached += 1;
                    }
                    provider_input_tokens =
                        provider_input_tokens.saturating_add(usage.input_tokens);
                    provider_output_tokens =
                        provider_output_tokens.saturating_add(usage.output_tokens);
                    provider_cost_nanodollars =
                        provider_cost_nanodollars.saturating_add(usage.provider_cost_nanodollars);
                    let (obs_i, obs_j) = match higher_ranked {
                        HigherRanked::A => (task.i, task.j),
                        HigherRanked::B => (task.j, task.i),
                    };
                    let obs = Observation::new(obs_i, obs_j, ratio, confidence, rater_id, 1.0);
                    if let Err(e) = manager.add_observation(attr_id, obs) {
                        tracing::warn!(
                            attribute_id = %attr_id,
                            error = %e,
                            "Failed to add observation"
                        );
                    } else {
                        comparisons_used += 1;
                        attribute_used[task.attr_idx] =
                            attribute_used[task.attr_idx].saturating_add(1);
                    }
                    *pair_repeats.entry(task.key).or_insert(0.0) += 1.0;

                    if let Some(trace) = trace {
                        let mut event = build_trace(
                            usage.cached,
                            usage.input_tokens,
                            usage.output_tokens,
                            usage.provider_cost_nanodollars,
                            None,
                        );
                        event.higher_ranked = Some(match higher_ranked {
                            HigherRanked::A => "A".to_string(),
                            HigherRanked::B => "B".to_string(),
                        });
                        event.ratio = Some(ratio);
                        event.confidence = Some(confidence);
                        trace.record(event)?;
                    }
                }
                Err(e) => {
                    if let Some(trace) = trace {
                        let event = build_trace(false, 0, 0, 0, Some(e.to_string()));
                        trace.record(event)?;
                    }
                    if cache_only {
                        return Err(MultiRerankError::Comparison(e));
                    }
                    tracing::warn!(
                        attribute_id = %attr_id,
                        i = task.i,
                        j = task.j,
                        error = %e,
                        "Comparison failed"
                    );
                }
            }
        }
    };

    // Final recompute and response assembly
    manager.recompute_global_state()?;
    manager.ensure_all_attribute_units()?;
    let global_topk_error = manager.estimate_topk_error();
    let latency_ms = start_time.elapsed().as_millis();

    // Per-attribute scores and derived units
    let n = req.entities.len();
    let mut attr_scores: HashMap<String, Vec<f64>> = HashMap::new();
    let mut attr_stds: HashMap<String, Vec<f64>> = HashMap::new();
    let mut attr_z: HashMap<String, Vec<f64>> = HashMap::new();
    let mut attr_min_norm: HashMap<String, Vec<f64>> = HashMap::new();
    let mut attr_pct: HashMap<String, Vec<f64>> = HashMap::new();

    for attr in &req.attributes {
        let id = &attr.id;
        if let Some(scores) = manager.attribute_scores(id) {
            let scores_vec = scores.to_vec();
            let stds = manager.attribute_std(id).unwrap_or_else(|| vec![0.0; n]);
            let z = manager
                .attribute_z_scores(id)
                .map(|v| v.to_vec())
                .unwrap_or_else(|| vec![0.0; n]);
            let min_norm = manager
                .attribute_min_norm(id)
                .map(|v| v.to_vec())
                .unwrap_or_else(|| vec![0.0; n]);
            let pct = manager
                .attribute_percentiles(id)
                .map(|v| v.to_vec())
                .unwrap_or_else(|| vec![0.0; n]);

            attr_scores.insert(id.clone(), scores_vec);
            attr_stds.insert(id.clone(), stds);
            attr_z.insert(id.clone(), z);
            attr_min_norm.insert(id.clone(), min_norm);
            attr_pct.insert(id.clone(), pct);
        }
    }

    // Build entity results
    let sorted_indices = manager.ranked_indices();
    let mut seen = vec![false; n];
    let mut entities_out: Vec<MultiRerankEntityResult> = Vec::with_capacity(n);

    // Feasible entities in rank order
    for idx in sorted_indices.iter().copied() {
        let state = manager.entity_state(idx);
        let feasible = state.feasible;
        let rank = state.rank;

        let u_mean = if feasible && state.u_mean.is_finite() {
            state.u_mean
        } else {
            0.0
        };
        let u_std = if feasible && state.u_var.is_finite() && state.u_var >= 0.0 {
            state.u_var.sqrt()
        } else {
            0.0
        };
        let p_flip = if state.p_flip.is_finite() {
            state.p_flip.clamp(0.0, 1.0)
        } else {
            0.0
        };

        let mut attr_map = HashMap::with_capacity(req.attributes.len());
        for attr in &req.attributes {
            let id = &attr.id;
            if let (Some(scores), Some(stds), Some(zs), Some(mns), Some(pcts)) = (
                attr_scores.get(id),
                attr_stds.get(id),
                attr_z.get(id),
                attr_min_norm.get(id),
                attr_pct.get(id),
            ) {
                if idx < scores.len() {
                    attr_map.insert(
                        id.clone(),
                        AttributeScoreSummary {
                            latent_mean: finite_or_zero(scores[idx]),
                            latent_std: finite_or_zero(stds[idx]),
                            z_score: finite_or_zero(zs[idx]),
                            min_normalized: finite_or_zero(mns[idx]),
                            percentile: finite_or_zero(pcts[idx]).clamp(0.0, 1.0),
                        },
                    );
                }
            }
        }

        entities_out.push(MultiRerankEntityResult {
            id: req.entities[idx].id.clone(),
            rank,
            feasible,
            u_mean,
            u_std,
            p_flip,
            attribute_scores: attr_map,
        });

        seen[idx] = true;
    }

    // Add remaining (infeasible) entities
    for idx in 0..n {
        if seen[idx] {
            continue;
        }

        let state = manager.entity_state(idx);

        let mut attr_map = HashMap::with_capacity(req.attributes.len());
        for attr in &req.attributes {
            let id = &attr.id;
            if let (Some(scores), Some(stds), Some(zs), Some(mns), Some(pcts)) = (
                attr_scores.get(id),
                attr_stds.get(id),
                attr_z.get(id),
                attr_min_norm.get(id),
                attr_pct.get(id),
            ) {
                if idx < scores.len() {
                    attr_map.insert(
                        id.clone(),
                        AttributeScoreSummary {
                            latent_mean: finite_or_zero(scores[idx]),
                            latent_std: finite_or_zero(stds[idx]),
                            z_score: finite_or_zero(zs[idx]),
                            min_normalized: finite_or_zero(mns[idx]),
                            percentile: finite_or_zero(pcts[idx]).clamp(0.0, 1.0),
                        },
                    );
                }
            }
        }

        entities_out.push(MultiRerankEntityResult {
            id: req.entities[idx].id.clone(),
            rank: state.rank,
            feasible: state.feasible,
            u_mean: if state.feasible && state.u_mean.is_finite() {
                state.u_mean
            } else {
                0.0
            },
            u_std: if state.feasible && state.u_var.is_finite() && state.u_var >= 0.0 {
                state.u_var.sqrt()
            } else {
                0.0
            },
            p_flip: if state.p_flip.is_finite() {
                state.p_flip.clamp(0.0, 1.0)
            } else {
                0.0
            },
            attribute_scores: attr_map,
        });
    }

    let meta = MultiRerankMeta {
        global_topk_error,
        tolerated_error: topk_cfg.tolerated_error,
        k: topk_cfg.k,
        band_size: topk_cfg.band_size,
        comparisons_attempted,
        comparisons_used,
        comparisons_refused,
        comparisons_cached,
        comparison_budget,
        latency_ms,
        model_used: if models_used.len() <= 1 {
            models_used
                .iter()
                .next()
                .cloned()
                .unwrap_or_else(|| base_model.to_string())
        } else {
            let mut models: Vec<String> = models_used.into_iter().collect();
            models.sort();
            format!("mixed: {}", models.join(", "))
        },
        rater_id_used: rater_id.to_string(),
        provider_input_tokens,
        provider_output_tokens,
        provider_cost_nanodollars,
        stop_reason,
    };

    Ok(MultiRerankResponse {
        entities: entities_out,
        meta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rerank::types::{
        MultiRerankAttributeSpec, MultiRerankEntity, MultiRerankGateSpec, MultiRerankTopKSpec,
    };

    fn base_request() -> MultiRerankRequest {
        MultiRerankRequest {
            entities: vec![
                MultiRerankEntity {
                    id: "a".to_string(),
                    text: "A".to_string(),
                },
                MultiRerankEntity {
                    id: "b".to_string(),
                    text: "B".to_string(),
                },
            ],
            attributes: vec![MultiRerankAttributeSpec {
                id: "attr".to_string(),
                prompt: "prompt".to_string(),
                prompt_template_slug: None,
                weight: 1.0,
            }],
            topk: MultiRerankTopKSpec {
                k: 1,
                weight_exponent: 1.3,
                tolerated_error: 0.1,
                band_size: 5,
                effective_resistance_max_active: 64,
                stop_sigma_inflate: 1.25,
                stop_min_consecutive: 2,
            },
            gates: Vec::new(),
            comparison_budget: Some(1),
            latency_budget_ms: None,
            model: None,
            rater_id: None,
            comparison_concurrency: None,
            max_pair_repeats: None,
        }
    }

    #[test]
    fn validate_rejects_unknown_gate_attribute() {
        let mut req = base_request();
        req.gates.push(MultiRerankGateSpec {
            attribute_id: "missing".to_string(),
            unit: "latent".to_string(),
            op: ">=".to_string(),
            threshold: 0.0,
        });
        let err = validate_multi_rerank_request(&req).unwrap_err();
        assert!(matches!(err, MultiRerankError::InvalidRequest(_)));
    }

    #[test]
    fn validate_rejects_unknown_gate_unit() {
        let mut req = base_request();
        req.gates.push(MultiRerankGateSpec {
            attribute_id: "attr".to_string(),
            unit: "wat".to_string(),
            op: ">=".to_string(),
            threshold: 0.0,
        });
        let err = validate_multi_rerank_request(&req).unwrap_err();
        assert!(matches!(err, MultiRerankError::InvalidRequest(_)));
    }

    #[test]
    fn validate_accepts_case_insensitive_gate_unit() {
        let mut req = base_request();
        req.gates.push(MultiRerankGateSpec {
            attribute_id: "attr".to_string(),
            unit: "Percentile".to_string(),
            op: ">=".to_string(),
            threshold: 0.5,
        });
        validate_multi_rerank_request(&req).unwrap();
    }

    #[test]
    fn validate_rejects_percentile_threshold_out_of_range() {
        let mut req = base_request();
        req.gates.push(MultiRerankGateSpec {
            attribute_id: "attr".to_string(),
            unit: "percentile".to_string(),
            op: ">=".to_string(),
            threshold: 1.1,
        });
        let err = validate_multi_rerank_request(&req).unwrap_err();
        assert!(matches!(err, MultiRerankError::InvalidRequest(_)));
    }

    #[test]
    fn validate_rejects_duplicate_attribute_ids() {
        let mut req = base_request();
        req.attributes.push(MultiRerankAttributeSpec {
            id: "attr".to_string(),
            prompt: "prompt2".to_string(),
            prompt_template_slug: None,
            weight: 1.0,
        });
        let err = validate_multi_rerank_request(&req).unwrap_err();
        assert!(matches!(err, MultiRerankError::InvalidRequest(_)));
    }

    #[test]
    fn validate_rejects_topk_k_gt_n() {
        let mut req = base_request();
        req.topk.k = 3;
        let err = validate_multi_rerank_request(&req).unwrap_err();
        assert!(matches!(err, MultiRerankError::InvalidRequest(_)));
    }

    #[test]
    fn validate_rejects_concurrency_zero() {
        let mut req = base_request();
        req.comparison_concurrency = Some(0);
        let err = validate_multi_rerank_request(&req).unwrap_err();
        assert!(matches!(err, MultiRerankError::InvalidRequest(_)));
    }

    #[test]
    fn validate_rejects_concurrency_too_high() {
        let mut req = base_request();
        req.comparison_concurrency = Some(MAX_COMPARISON_CONCURRENCY + 1);
        let err = validate_multi_rerank_request(&req).unwrap_err();
        assert!(matches!(err, MultiRerankError::InvalidRequest(_)));
    }

    #[test]
    fn validate_rejects_max_pair_repeats_zero() {
        let mut req = base_request();
        req.max_pair_repeats = Some(0);
        let err = validate_multi_rerank_request(&req).unwrap_err();
        assert!(matches!(err, MultiRerankError::InvalidRequest(_)));
    }

    #[test]
    fn validate_rejects_duplicate_entity_ids() {
        let mut req = base_request();
        req.entities.push(MultiRerankEntity {
            id: "a".to_string(),
            text: "A2".to_string(),
        });
        let err = validate_multi_rerank_request(&req).unwrap_err();
        assert!(matches!(err, MultiRerankError::InvalidRequest(_)));
    }

    #[test]
    fn validate_rejects_unknown_prompt_template_slug() {
        let mut req = base_request();
        req.attributes[0].prompt_template_slug = Some("does_not_exist".to_string());
        let err = validate_multi_rerank_request(&req).unwrap_err();
        assert!(matches!(err, MultiRerankError::InvalidRequest(_)));
    }

    #[test]
    fn validate_rejects_unsupported_gate_op() {
        let mut req = base_request();
        req.gates.push(MultiRerankGateSpec {
            attribute_id: "attr".to_string(),
            unit: "latent".to_string(),
            op: "=".to_string(),
            threshold: 0.0,
        });
        let err = validate_multi_rerank_request(&req).unwrap_err();
        assert!(matches!(err, MultiRerankError::InvalidRequest(_)));
    }

    #[test]
    fn validate_rejects_entities_len_lt_2() {
        let mut req = base_request();
        req.entities.truncate(1);
        let err = validate_multi_rerank_request(&req).unwrap_err();
        assert!(matches!(err, MultiRerankError::InvalidRequest(_)));
    }

    #[test]
    fn validate_rejects_empty_attributes() {
        let mut req = base_request();
        req.attributes.clear();
        let err = validate_multi_rerank_request(&req).unwrap_err();
        assert!(matches!(err, MultiRerankError::InvalidRequest(_)));
    }
}
