//! Report generation for rerank runs.

use std::collections::HashMap;

use blake3;
use serde::Serialize;

use super::types::{
    AttributeScoreSummary, MultiRerankAttributeSpec, MultiRerankEntityResult,
    MultiRerankGateSpec, MultiRerankMeta, MultiRerankRequest, MultiRerankResponse,
    RerankStopReason,
};

#[derive(Debug, Clone, Serialize)]
pub struct RerankReportOptions {
    pub top_n: usize,
    pub include_infeasible: bool,
    pub include_attribute_scores: bool,
    pub rng_seed: Option<u64>,
    pub model_policy: Option<String>,
    pub cache_only: bool,
}

impl Default for RerankReportOptions {
    fn default() -> Self {
        Self {
            top_n: 10,
            include_infeasible: false,
            include_attribute_scores: true,
            rng_seed: None,
            model_policy: None,
            cache_only: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankReport {
    pub request_hash: String,
    pub summary: ReportSummary,
    pub attributes: Vec<ReportAttribute>,
    pub gates: Vec<MultiRerankGateSpec>,
    pub top_entities: Vec<ReportEntity>,
    pub run_stamp: ReportStamp,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReportSummary {
    pub stop_reason: RerankStopReason,
    pub k: usize,
    pub global_topk_error: f64,
    pub tolerated_error: f64,
    pub comparisons_attempted: usize,
    pub comparisons_used: usize,
    pub comparisons_refused: usize,
    pub comparisons_cached: usize,
    pub comparison_budget: usize,
    pub latency_ms: u128,
    pub model_used: String,
    pub rater_id_used: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReportAttribute {
    pub id: String,
    pub prompt: String,
    pub weight: f64,
    pub prompt_template_slug: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReportEntity {
    pub id: String,
    pub rank: Option<usize>,
    pub feasible: bool,
    pub u_mean: f64,
    pub u_std: f64,
    pub p_flip: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribute_scores: Option<HashMap<String, AttributeScoreSummary>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReportStamp {
    pub rng_seed: Option<u64>,
    pub model_policy: Option<String>,
    pub cache_only: bool,
}

pub fn build_report(
    req: &MultiRerankRequest,
    resp: &MultiRerankResponse,
    opts: &RerankReportOptions,
) -> RerankReport {
    let request_hash = hash_request(req);
    let summary = ReportSummary::from_meta(&resp.meta);
    let attributes = req
        .attributes
        .iter()
        .map(ReportAttribute::from_attr)
        .collect::<Vec<_>>();

    let mut top_entities: Vec<ReportEntity> = Vec::new();
    for entity in &resp.entities {
        if !opts.include_infeasible && !entity.feasible {
            continue;
        }
        if top_entities.len() >= opts.top_n {
            break;
        }
        top_entities.push(ReportEntity::from_entity(entity, opts.include_attribute_scores));
    }

    RerankReport {
        request_hash,
        summary,
        attributes,
        gates: req.gates.clone(),
        top_entities,
        run_stamp: ReportStamp {
            rng_seed: opts.rng_seed,
            model_policy: opts.model_policy.clone(),
            cache_only: opts.cache_only,
        },
    }
}

impl ReportSummary {
    fn from_meta(meta: &MultiRerankMeta) -> Self {
        Self {
            stop_reason: meta.stop_reason,
            k: meta.k,
            global_topk_error: meta.global_topk_error,
            tolerated_error: meta.tolerated_error,
            comparisons_attempted: meta.comparisons_attempted,
            comparisons_used: meta.comparisons_used,
            comparisons_refused: meta.comparisons_refused,
            comparisons_cached: meta.comparisons_cached,
            comparison_budget: meta.comparison_budget,
            latency_ms: meta.latency_ms,
            model_used: meta.model_used.clone(),
            rater_id_used: meta.rater_id_used.clone(),
        }
    }
}

impl ReportAttribute {
    fn from_attr(attr: &MultiRerankAttributeSpec) -> Self {
        Self {
            id: attr.id.clone(),
            prompt: attr.prompt.clone(),
            weight: attr.weight,
            prompt_template_slug: attr.prompt_template_slug.clone(),
        }
    }
}

impl ReportEntity {
    fn from_entity(entity: &MultiRerankEntityResult, include_scores: bool) -> Self {
        Self {
            id: entity.id.clone(),
            rank: entity.rank,
            feasible: entity.feasible,
            u_mean: entity.u_mean,
            u_std: entity.u_std,
            p_flip: entity.p_flip,
            attribute_scores: if include_scores {
                Some(entity.attribute_scores.clone())
            } else {
                None
            },
        }
    }
}

pub fn render_report_markdown(report: &RerankReport) -> String {
    let mut out = String::new();
    out.push_str("# Rerank Report\n\n");
    out.push_str(&format!("- Request hash: `{}`\n", report.request_hash));
    out.push_str(&format!("- Stop reason: {:?}\n", report.summary.stop_reason));
    out.push_str(&format!("- k: {}\n", report.summary.k));
    out.push_str(&format!(
        "- Global top-k error: {:.4}\n",
        report.summary.global_topk_error
    ));
    out.push_str(&format!(
        "- Tolerated error: {:.4}\n",
        report.summary.tolerated_error
    ));
    out.push_str(&format!(
        "- Comparisons used/attempted/refused/cached: {}/{}/{}/{}\n",
        report.summary.comparisons_used,
        report.summary.comparisons_attempted,
        report.summary.comparisons_refused,
        report.summary.comparisons_cached
    ));
    out.push_str(&format!("- Model used: {}\n", report.summary.model_used));
    out.push_str(&format!("- Rater ID: {}\n", report.summary.rater_id_used));
    out.push_str(&format!("- Latency: {} ms\n", report.summary.latency_ms));
    if let Some(seed) = report.run_stamp.rng_seed {
        out.push_str(&format!("- RNG seed: {}\n", seed));
    }
    if let Some(policy) = &report.run_stamp.model_policy {
        out.push_str(&format!("- Model policy: {}\n", policy));
    }
    if report.run_stamp.cache_only {
        out.push_str("- Cache-only mode: true\n");
    }

    out.push_str("\n## Attributes\n\n");
    for attr in &report.attributes {
        out.push_str(&format!(
            "- `{}` (weight {:.3}) — {}\n",
            attr.id, attr.weight, attr.prompt
        ));
    }

    if !report.gates.is_empty() {
        out.push_str("\n## Gates\n\n");
        for gate in &report.gates {
            out.push_str(&format!(
                "- {} {} {} {}\n",
                gate.attribute_id, gate.op, gate.threshold, gate.unit
            ));
        }
    }

    out.push_str("\n## Top Entities\n\n");
    for entity in &report.top_entities {
        out.push_str(&format!(
            "- {} (rank {:?}, feasible {}, u_mean {:.3}, u_std {:.3}, p_flip {:.3})\n",
            entity.id, entity.rank, entity.feasible, entity.u_mean, entity.u_std, entity.p_flip
        ));
    }

    out
}

fn hash_request(req: &MultiRerankRequest) -> String {
    let bytes = serde_json::to_vec(req).unwrap_or_default();
    blake3::hash(&bytes).to_hex().to_string()
}
