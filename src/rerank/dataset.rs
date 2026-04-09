//! Training dataset export for pairwise ratio fine-tuning.
//!
//! Converts rerank request + comparison trace data into JSONL records suitable
//! for supervised fine-tuning or distillation workflows.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::cache::PairwiseCacheKey;
use crate::prompts::RATIO_LADDER;
use crate::prompts::{prompt_by_slug, EntityRef, DEFAULT_PROMPT};

use super::trace::ComparisonTrace;
use super::types::{
    MultiRerankAttributeSpec, MultiRerankEntity, MultiRerankEntityResult, MultiRerankRequest,
    MultiRerankResponse,
};

#[derive(Debug, Clone, Default)]
pub struct PairwiseDatasetExportOptions {
    pub drop_cached: bool,
    pub drop_refusals: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PairwiseDatasetExportStats {
    pub total_traces: usize,
    pub exported_records: usize,
    pub snapped_off_ladder: usize,
    pub skipped_cached: usize,
    pub skipped_refusals: usize,
    pub skipped_errors: usize,
    pub skipped_incomplete: usize,
    pub skipped_unknown_attribute: usize,
    pub skipped_unknown_entity: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwiseDatasetExportResult {
    pub records: Vec<PairwiseDatasetRecord>,
    pub stats: PairwiseDatasetExportStats,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PairwisePromptGridExportStats {
    pub total_attributes: usize,
    pub unordered_pairs_per_attribute: usize,
    pub presentations_per_pair: usize,
    pub exported_records: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwisePromptGridExportResult {
    pub records: Vec<PairwisePromptRecord>,
    pub stats: PairwisePromptGridExportStats,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwiseDatasetRecord {
    pub schema_version: String,
    pub record_kind: String,
    pub pair_id: String,
    pub comparison_index: usize,
    pub attribute: PairwiseDatasetAttribute,
    pub entity_a: PairwiseDatasetEntity,
    pub entity_b: PairwiseDatasetEntity,
    pub canonical_pair: PairwiseDatasetCanonicalPair,
    pub presentation: PairwiseDatasetPresentation,
    pub target: PairwiseDatasetTarget,
    pub messages: Vec<PairwiseDatasetMessage>,
    pub metadata: PairwiseDatasetMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_context: Option<PairwiseDatasetResponseContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwiseDatasetAttribute {
    pub id: String,
    pub prompt: String,
    pub weight: f64,
    pub prompt_template_slug: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwiseDatasetEntity {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwiseDatasetCanonicalPair {
    pub entity_left_id: String,
    pub entity_right_id: String,
    pub entity_left_index: usize,
    pub entity_right_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwiseDatasetPresentation {
    pub displayed_entity_a_id: String,
    pub displayed_entity_b_id: String,
    pub swapped: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwiseDatasetTarget {
    pub assistant_json: String,
    pub refused: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_higher_ranked: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub higher_ranked: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ratio_bucket: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_higher_ranked: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_ratio_bucket: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_signed_ln_ratio: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwiseDatasetMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwiseDatasetMetadata {
    pub model: String,
    pub cached: bool,
    pub snapped_to_ladder: bool,
    pub attribute_prompt_hash: String,
    pub template_hash: String,
    pub cache_key_hash: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub provider_cost_nanodollars: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwisePromptRecord {
    pub schema_version: String,
    pub record_kind: String,
    pub prompt_index: usize,
    pub pair_id: String,
    pub attribute: PairwiseDatasetAttribute,
    pub entity_a: PairwiseDatasetEntity,
    pub entity_b: PairwiseDatasetEntity,
    pub canonical_pair: PairwiseDatasetCanonicalPair,
    pub presentation: PairwiseDatasetPresentation,
    pub messages: Vec<PairwiseDatasetMessage>,
    pub cache: PairwisePromptCacheMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwisePromptCacheMetadata {
    pub model: String,
    pub prompt_template_slug: String,
    pub template_hash: String,
    pub attribute_prompt_hash: String,
    pub entity_a_hash: String,
    pub entity_b_hash: String,
    pub cache_key_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairwiseDatasetResponseContext {
    pub topk_k: usize,
    pub entity_a_rank: Option<usize>,
    pub entity_b_rank: Option<usize>,
    pub entity_a_feasible: bool,
    pub entity_b_feasible: bool,
    pub entity_a_p_flip: f64,
    pub entity_b_p_flip: f64,
    pub near_topk_boundary: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_topk_boundary_distance: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct ObservationAssistantJson<'a> {
    higher_ranked: &'a str,
    ratio: f64,
    confidence: f64,
}

#[derive(Debug, Clone, Serialize)]
struct RefusalAssistantJson {
    refused: bool,
}

pub fn export_pairwise_prompt_grid(
    req: &MultiRerankRequest,
    model: &str,
    include_swaps: bool,
) -> PairwisePromptGridExportResult {
    let unordered_pairs_per_attribute =
        req.entities.len().saturating_sub(1) * req.entities.len() / 2;
    let presentations_per_pair = if include_swaps { 2 } else { 1 };
    let mut prompt_index = 0usize;
    let mut records = Vec::new();

    for attr in &req.attributes {
        let template = prompt_by_slug(attr.prompt_template_slug.as_deref().unwrap_or(""))
            .unwrap_or(DEFAULT_PROMPT);
        let template_hash = template.template_hash();

        for left_idx in 0..req.entities.len() {
            for right_idx in (left_idx + 1)..req.entities.len() {
                let left = &req.entities[left_idx];
                let right = &req.entities[right_idx];
                let presentations: &[bool] = if include_swaps {
                    &[false, true]
                } else {
                    &[false]
                };
                for &swapped in presentations {
                    prompt_index += 1;
                    let (display_a, display_b) = if swapped {
                        (right, left)
                    } else {
                        (left, right)
                    };
                    let prompt_instance = template.render(
                        &attr.id,
                        &attr.prompt,
                        EntityRef::with_context("A", display_a.text.clone()),
                        EntityRef::with_context("B", display_b.text.clone()),
                    );
                    let cache_key = PairwiseCacheKey::new(
                        model,
                        template.slug,
                        &template_hash,
                        &attr.id,
                        &attr.prompt,
                        &display_a.id,
                        &display_a.text,
                        &display_b.id,
                        &display_b.text,
                    );
                    let messages = prompt_instance
                        .to_messages()
                        .into_iter()
                        .map(|message| PairwiseDatasetMessage {
                            role: match message.role {
                                crate::gateway::Role::System => "system".to_string(),
                                crate::gateway::Role::User => "user".to_string(),
                                crate::gateway::Role::Assistant => "assistant".to_string(),
                            },
                            content: message.content,
                        })
                        .collect::<Vec<_>>();

                    records.push(PairwisePromptRecord {
                        schema_version: "pairwise_ratio_prompt_grid_v1".to_string(),
                        record_kind: "pairwise_inference_prompt".to_string(),
                        prompt_index,
                        pair_id: format!("{}::{}::{}", attr.id, left.id, right.id),
                        attribute: PairwiseDatasetAttribute {
                            id: attr.id.clone(),
                            prompt: attr.prompt.clone(),
                            weight: attr.weight,
                            prompt_template_slug: template.slug.to_string(),
                        },
                        entity_a: PairwiseDatasetEntity {
                            id: display_a.id.clone(),
                            text: display_a.text.clone(),
                        },
                        entity_b: PairwiseDatasetEntity {
                            id: display_b.id.clone(),
                            text: display_b.text.clone(),
                        },
                        canonical_pair: PairwiseDatasetCanonicalPair {
                            entity_left_id: left.id.clone(),
                            entity_right_id: right.id.clone(),
                            entity_left_index: left_idx,
                            entity_right_index: right_idx,
                        },
                        presentation: PairwiseDatasetPresentation {
                            displayed_entity_a_id: display_a.id.clone(),
                            displayed_entity_b_id: display_b.id.clone(),
                            swapped,
                        },
                        messages,
                        cache: PairwisePromptCacheMetadata {
                            model: cache_key.model,
                            prompt_template_slug: cache_key.prompt_template_slug,
                            template_hash: cache_key.template_hash,
                            attribute_prompt_hash: cache_key.attribute_prompt_hash,
                            entity_a_hash: cache_key.entity_a_hash,
                            entity_b_hash: cache_key.entity_b_hash,
                            cache_key_hash: cache_key.key_hash,
                        },
                    });
                }
            }
        }
    }

    PairwisePromptGridExportResult {
        stats: PairwisePromptGridExportStats {
            total_attributes: req.attributes.len(),
            unordered_pairs_per_attribute,
            presentations_per_pair,
            exported_records: records.len(),
        },
        records,
    }
}

pub fn export_pairwise_dataset(
    req: &MultiRerankRequest,
    resp: Option<&MultiRerankResponse>,
    traces: &[ComparisonTrace],
    opts: &PairwiseDatasetExportOptions,
) -> PairwiseDatasetExportResult {
    let response_map: HashMap<&str, &MultiRerankEntityResult> = resp
        .map(|response| {
            response
                .entities
                .iter()
                .map(|entity| (entity.id.as_str(), entity))
                .collect()
        })
        .unwrap_or_default();

    let mut stats = PairwiseDatasetExportStats {
        total_traces: traces.len(),
        ..PairwiseDatasetExportStats::default()
    };
    let mut records = Vec::new();

    for trace in traces {
        if trace.cached && opts.drop_cached {
            stats.skipped_cached += 1;
            continue;
        }
        if trace.refused && opts.drop_refusals {
            stats.skipped_refusals += 1;
            continue;
        }
        if trace.error.is_some() {
            stats.skipped_errors += 1;
            continue;
        }

        let Some(attr) = resolve_attribute(req, trace) else {
            stats.skipped_unknown_attribute += 1;
            continue;
        };
        let Some(canonical_left) = resolve_entity(req, trace.entity_a_index, &trace.entity_a_id)
        else {
            stats.skipped_unknown_entity += 1;
            continue;
        };
        let Some(canonical_right) = resolve_entity(req, trace.entity_b_index, &trace.entity_b_id)
        else {
            stats.skipped_unknown_entity += 1;
            continue;
        };

        let (display_a, display_b) = if trace.swapped {
            (canonical_right, canonical_left)
        } else {
            (canonical_left, canonical_right)
        };

        let template = prompt_by_slug(attr.prompt_template_slug.as_deref().unwrap_or(""))
            .unwrap_or(DEFAULT_PROMPT);
        let prompt_instance = template.render(
            &attr.id,
            &attr.prompt,
            EntityRef::with_context("A", display_a.text.clone()),
            EntityRef::with_context("B", display_b.text.clone()),
        );

        let prompt_messages = prompt_instance
            .to_messages()
            .into_iter()
            .map(|message| PairwiseDatasetMessage {
                role: match message.role {
                    crate::gateway::Role::System => "system".to_string(),
                    crate::gateway::Role::User => "user".to_string(),
                    crate::gateway::Role::Assistant => "assistant".to_string(),
                },
                content: message.content,
            })
            .collect::<Vec<_>>();

        let Some((target, snapped_to_ladder)) = build_target(trace) else {
            stats.skipped_incomplete += 1;
            continue;
        };
        if snapped_to_ladder {
            stats.snapped_off_ladder += 1;
        }

        let mut messages = prompt_messages;
        messages.push(PairwiseDatasetMessage {
            role: "assistant".to_string(),
            content: target.assistant_json.clone(),
        });

        let response_context =
            build_response_context(req.topk.k, &response_map, display_a, display_b);

        records.push(PairwiseDatasetRecord {
            schema_version: "pairwise_ratio_sft_v1".to_string(),
            record_kind: if trace.refused {
                "rerank_trace_refusal".to_string()
            } else {
                "rerank_trace_observation".to_string()
            },
            pair_id: format!("{}::{}::{}", attr.id, canonical_left.id, canonical_right.id),
            comparison_index: trace.comparison_index,
            attribute: PairwiseDatasetAttribute {
                id: attr.id.clone(),
                prompt: attr.prompt.clone(),
                weight: attr.weight,
                prompt_template_slug: template.slug.to_string(),
            },
            entity_a: PairwiseDatasetEntity {
                id: display_a.id.clone(),
                text: display_a.text.clone(),
            },
            entity_b: PairwiseDatasetEntity {
                id: display_b.id.clone(),
                text: display_b.text.clone(),
            },
            canonical_pair: PairwiseDatasetCanonicalPair {
                entity_left_id: canonical_left.id.clone(),
                entity_right_id: canonical_right.id.clone(),
                entity_left_index: trace.entity_a_index,
                entity_right_index: trace.entity_b_index,
            },
            presentation: PairwiseDatasetPresentation {
                displayed_entity_a_id: display_a.id.clone(),
                displayed_entity_b_id: display_b.id.clone(),
                swapped: trace.swapped,
            },
            target,
            messages,
            metadata: PairwiseDatasetMetadata {
                model: trace.model.clone(),
                cached: trace.cached,
                snapped_to_ladder,
                attribute_prompt_hash: trace.attribute_prompt_hash.clone(),
                template_hash: trace.template_hash.clone(),
                cache_key_hash: trace.cache_key_hash.clone(),
                input_tokens: trace.input_tokens,
                output_tokens: trace.output_tokens,
                provider_cost_nanodollars: trace.provider_cost_nanodollars,
            },
            response_context,
        });
    }

    records.sort_by_key(|record| record.comparison_index);
    stats.exported_records = records.len();

    PairwiseDatasetExportResult { records, stats }
}

fn resolve_attribute<'a>(
    req: &'a MultiRerankRequest,
    trace: &ComparisonTrace,
) -> Option<&'a MultiRerankAttributeSpec> {
    req.attributes
        .get(trace.attribute_index)
        .filter(|attr| attr.id == trace.attribute_id)
        .or_else(|| {
            req.attributes
                .iter()
                .find(|attr| attr.id == trace.attribute_id)
        })
}

fn resolve_entity<'a>(
    req: &'a MultiRerankRequest,
    idx: usize,
    expected_id: &str,
) -> Option<&'a MultiRerankEntity> {
    req.entities
        .get(idx)
        .filter(|entity| entity.id == expected_id)
        .or_else(|| req.entities.iter().find(|entity| entity.id == expected_id))
}

fn build_target(trace: &ComparisonTrace) -> Option<(PairwiseDatasetTarget, bool)> {
    if trace.refused {
        let assistant_json = serde_json::to_string(&RefusalAssistantJson { refused: true }).ok()?;
        return Some((
            PairwiseDatasetTarget {
                assistant_json,
                refused: true,
                raw_higher_ranked: None,
                raw_ratio: None,
                higher_ranked: None,
                ratio: None,
                ratio_bucket: None,
                confidence: None,
                canonical_higher_ranked: None,
                canonical_ratio: None,
                canonical_ratio_bucket: None,
                canonical_signed_ln_ratio: None,
            },
            false,
        ));
    }

    let raw_higher_ranked = trace.higher_ranked.as_deref()?;
    let raw_ratio = trace.ratio?;
    let confidence = trace.confidence?;
    let ratio = snap_ratio_to_ladder(raw_ratio);
    let snapped_to_ladder = (ratio - raw_ratio).abs() > 1e-9;

    let higher_ranked = if (ratio - 1.0).abs() < 1e-9 {
        "A".to_string()
    } else {
        raw_higher_ranked.to_string()
    };
    let ratio_bucket = canonical_ratio_string(ratio);
    let canonical_higher_ranked = if (ratio - 1.0).abs() < 1e-9 {
        "A".to_string()
    } else if trace.swapped {
        invert_higher_ranked(&higher_ranked)?
    } else {
        higher_ranked.clone()
    };
    let canonical_signed_ln_ratio = match canonical_higher_ranked.as_str() {
        "A" => Some(ratio.ln()),
        "B" => Some(-ratio.ln()),
        _ => None,
    }?;
    let assistant_json = serde_json::to_string(&ObservationAssistantJson {
        higher_ranked: &higher_ranked,
        ratio,
        confidence,
    })
    .ok()?;

    Some((
        PairwiseDatasetTarget {
            assistant_json,
            refused: false,
            raw_higher_ranked: Some(raw_higher_ranked.to_string()),
            raw_ratio: Some(raw_ratio),
            higher_ranked: Some(higher_ranked),
            ratio: Some(ratio),
            ratio_bucket: Some(ratio_bucket.clone()),
            confidence: Some(confidence),
            canonical_higher_ranked: Some(canonical_higher_ranked),
            canonical_ratio: Some(ratio),
            canonical_ratio_bucket: Some(ratio_bucket),
            canonical_signed_ln_ratio: Some(canonical_signed_ln_ratio),
        },
        snapped_to_ladder,
    ))
}

fn build_response_context(
    topk_k: usize,
    response_map: &HashMap<&str, &MultiRerankEntityResult>,
    entity_a: &MultiRerankEntity,
    entity_b: &MultiRerankEntity,
) -> Option<PairwiseDatasetResponseContext> {
    let result_a = response_map.get(entity_a.id.as_str())?;
    let result_b = response_map.get(entity_b.id.as_str())?;

    let min_topk_boundary_distance = [result_a.rank, result_b.rank]
        .into_iter()
        .flatten()
        .map(|rank| rank.abs_diff(topk_k))
        .min();
    let near_topk_boundary = min_topk_boundary_distance.is_some_and(|distance| distance <= 1);

    Some(PairwiseDatasetResponseContext {
        topk_k,
        entity_a_rank: result_a.rank,
        entity_b_rank: result_b.rank,
        entity_a_feasible: result_a.feasible,
        entity_b_feasible: result_b.feasible,
        entity_a_p_flip: result_a.p_flip,
        entity_b_p_flip: result_b.p_flip,
        near_topk_boundary,
        min_topk_boundary_distance,
    })
}

fn invert_higher_ranked(higher_ranked: &str) -> Option<String> {
    match higher_ranked {
        "A" => Some("B".to_string()),
        "B" => Some("A".to_string()),
        _ => None,
    }
}

fn canonical_ratio_string(ratio: f64) -> String {
    let mut s = format!("{ratio:.2}");
    while s.ends_with('0') && s.contains('.') {
        s.pop();
    }
    if s.ends_with('.') {
        s.push('0');
    }
    s
}

fn snap_ratio_to_ladder(ratio: f64) -> f64 {
    let mut best = RATIO_LADDER[0];
    let mut best_dist = (ratio - best).abs();
    for &candidate in &RATIO_LADDER[1..] {
        let dist = (ratio - candidate).abs();
        if dist < best_dist {
            best = candidate;
            best_dist = dist;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rerank::types::{
        MultiRerankAttributeSpec, MultiRerankEntityResult, MultiRerankMeta, MultiRerankTopKSpec,
        RerankStopReason,
    };

    fn request() -> MultiRerankRequest {
        MultiRerankRequest {
            entities: vec![
                MultiRerankEntity {
                    id: "left".to_string(),
                    text: "Left entity text".to_string(),
                },
                MultiRerankEntity {
                    id: "right".to_string(),
                    text: "Right entity text".to_string(),
                },
            ],
            attributes: vec![MultiRerankAttributeSpec {
                id: "clarity".to_string(),
                prompt: "Which entity is clearer?".to_string(),
                prompt_template_slug: Some("canonical_v2".to_string()),
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
                min_explore_degree: 2,
            },
            gates: Vec::new(),
            comparison_budget: Some(4),
            latency_budget_ms: None,
            model: Some("openai/gpt-5-mini".to_string()),
            rater_id: Some("test".to_string()),
            comparison_concurrency: Some(1),
            max_pair_repeats: Some(1),
            randomize_presentation_order: true,
        }
    }

    fn response() -> MultiRerankResponse {
        MultiRerankResponse {
            entities: vec![
                MultiRerankEntityResult {
                    id: "left".to_string(),
                    rank: Some(1),
                    feasible: true,
                    u_mean: 1.0,
                    u_std: 0.1,
                    p_flip: 0.05,
                    attribute_scores: HashMap::new(),
                },
                MultiRerankEntityResult {
                    id: "right".to_string(),
                    rank: Some(2),
                    feasible: true,
                    u_mean: 0.5,
                    u_std: 0.1,
                    p_flip: 0.07,
                    attribute_scores: HashMap::new(),
                },
            ],
            meta: MultiRerankMeta {
                global_topk_error: 0.1,
                tolerated_error: 0.1,
                k: 1,
                band_size: 5,
                comparisons_attempted: 1,
                comparisons_used: 1,
                comparisons_refused: 0,
                comparisons_cached: 0,
                comparison_budget: 4,
                latency_ms: 1,
                model_used: "openai/gpt-5-mini".to_string(),
                rater_id_used: "test".to_string(),
                provider_input_tokens: 10,
                provider_output_tokens: 5,
                provider_cost_nanodollars: 100,
                stop_reason: RerankStopReason::BudgetExhausted,
            },
        }
    }

    #[test]
    fn export_builds_swap_aware_records() {
        let traces = vec![ComparisonTrace {
            timestamp_ms: 0,
            comparison_index: 1,
            attribute_id: "clarity".to_string(),
            attribute_index: 0,
            attribute_prompt_hash: "attr_hash".to_string(),
            prompt_template_slug: "canonical_v2".to_string(),
            template_hash: "tmpl_hash".to_string(),
            entity_a_id: "left".to_string(),
            entity_b_id: "right".to_string(),
            entity_a_index: 0,
            entity_b_index: 1,
            entity_a_hash: "left_hash".to_string(),
            entity_b_hash: "right_hash".to_string(),
            cache_key_hash: "cache_hash".to_string(),
            model: "openai/gpt-5-mini".to_string(),
            higher_ranked: Some("A".to_string()),
            ratio: Some(2.1),
            confidence: Some(0.8),
            refused: false,
            cached: false,
            swapped: true,
            input_tokens: 10,
            output_tokens: 5,
            provider_cost_nanodollars: 100,
            error: None,
        }];

        let exported =
            export_pairwise_dataset(&request(), Some(&response()), &traces, &Default::default());
        assert_eq!(exported.stats.exported_records, 1);
        assert_eq!(exported.records.len(), 1);

        let record = &exported.records[0];
        assert_eq!(record.entity_a.id, "right");
        assert_eq!(record.entity_b.id, "left");
        assert!(record.presentation.swapped);
        assert_eq!(record.target.higher_ranked.as_deref(), Some("A"));
        assert_eq!(record.target.canonical_higher_ranked.as_deref(), Some("B"));
        assert!(record
            .target
            .canonical_signed_ln_ratio
            .is_some_and(|value| value < 0.0));
        assert_eq!(record.target.raw_ratio, Some(2.1));
        assert_eq!(
            record.target.assistant_json,
            r#"{"higher_ranked":"A","ratio":2.1,"confidence":0.8}"#
        );
        assert_eq!(record.messages.len(), 3);
        assert_eq!(record.messages[2].role, "assistant");
        assert!(record
            .response_context
            .as_ref()
            .is_some_and(|ctx| ctx.near_topk_boundary));
    }

    #[test]
    fn export_skips_error_records() {
        let traces = vec![ComparisonTrace {
            timestamp_ms: 0,
            comparison_index: 1,
            attribute_id: "clarity".to_string(),
            attribute_index: 0,
            attribute_prompt_hash: "attr_hash".to_string(),
            prompt_template_slug: "canonical_v2".to_string(),
            template_hash: "tmpl_hash".to_string(),
            entity_a_id: "left".to_string(),
            entity_b_id: "right".to_string(),
            entity_a_index: 0,
            entity_b_index: 1,
            entity_a_hash: "left_hash".to_string(),
            entity_b_hash: "right_hash".to_string(),
            cache_key_hash: "cache_hash".to_string(),
            model: "openai/gpt-5-mini".to_string(),
            higher_ranked: None,
            ratio: None,
            confidence: None,
            refused: false,
            cached: false,
            swapped: false,
            input_tokens: 0,
            output_tokens: 0,
            provider_cost_nanodollars: 0,
            error: Some("parse failed".to_string()),
        }];

        let exported = export_pairwise_dataset(&request(), None, &traces, &Default::default());
        assert_eq!(exported.records.len(), 0);
        assert_eq!(exported.stats.skipped_errors, 1);
    }

    #[test]
    fn export_snaps_ratio_to_ladder() {
        let traces = vec![ComparisonTrace {
            timestamp_ms: 0,
            comparison_index: 1,
            attribute_id: "clarity".to_string(),
            attribute_index: 0,
            attribute_prompt_hash: "attr_hash".to_string(),
            prompt_template_slug: "canonical_v2".to_string(),
            template_hash: "tmpl_hash".to_string(),
            entity_a_id: "left".to_string(),
            entity_b_id: "right".to_string(),
            entity_a_index: 0,
            entity_b_index: 1,
            entity_a_hash: "left_hash".to_string(),
            entity_b_hash: "right_hash".to_string(),
            cache_key_hash: "cache_hash".to_string(),
            model: "openai/gpt-5-mini".to_string(),
            higher_ranked: Some("A".to_string()),
            ratio: Some(2.0),
            confidence: Some(0.8),
            refused: false,
            cached: false,
            swapped: false,
            input_tokens: 10,
            output_tokens: 5,
            provider_cost_nanodollars: 100,
            error: None,
        }];

        let exported = export_pairwise_dataset(&request(), None, &traces, &Default::default());
        let record = &exported.records[0];
        assert_eq!(record.target.raw_ratio, Some(2.0));
        assert_eq!(record.target.ratio, Some(2.1));
        assert!(record.metadata.snapped_to_ladder);
        assert_eq!(exported.stats.snapped_off_ladder, 1);
    }

    #[test]
    fn export_prompt_grid_emits_both_presentations_and_cache_hashes() {
        let exported = export_pairwise_prompt_grid(&request(), "local/gx10-sft", true);
        assert_eq!(exported.stats.total_attributes, 1);
        assert_eq!(exported.stats.unordered_pairs_per_attribute, 1);
        assert_eq!(exported.stats.presentations_per_pair, 2);
        assert_eq!(exported.records.len(), 2);

        let forward = &exported.records[0];
        let swapped = &exported.records[1];
        assert_eq!(forward.entity_a.id, "left");
        assert_eq!(forward.entity_b.id, "right");
        assert!(!forward.presentation.swapped);
        assert_eq!(swapped.entity_a.id, "right");
        assert_eq!(swapped.entity_b.id, "left");
        assert!(swapped.presentation.swapped);
        assert_eq!(forward.cache.model, "local/gx10-sft");
        assert_ne!(forward.cache.cache_key_hash, swapped.cache.cache_key_hash);
        assert_eq!(forward.messages.len(), 2);
    }
}
