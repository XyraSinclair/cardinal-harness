//! Analytic Network Process helpers built on the cardinal pairwise primitive.
//!
//! The core judgment remains:
//! `rater::attribute_prompt::entity_A::entity_B::ratio::confidence`.
//!
//! This module adds typed ANP contexts around that primitive so callers can:
//! - mark which prompts are expected to compose globally (`composable_ratio`)
//! - keep non-transitive prompts local (`pairwise_only_ratio`)
//! - fit sparse local priorities from confidence-weighted log-ratio judgments
//! - propose high-value next queries (context + pair)
//! - assemble a weighted supermatrix and solve damped stationary priorities
//! - run a single offline ANP demo pipeline from JSON-friendly input/output

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// How a context's pairwise ratios should be interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgmentKind {
    /// Ratios are expected to embed in a stable latent ratio scale.
    ComposableRatio,
    /// Ratios are usable locally but should not be globally propagated by default.
    PairwiseOnlyRatio,
}

/// ANP relation semantics for a context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    /// Preference-style comparison ("better under criterion X").
    Preference,
    /// Influence-style comparison ("more influence on node X").
    Influence,
}

/// Group of nodes in the ANP network.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cluster {
    pub id: String,
    pub label: String,
}

/// Node in the ANP network.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub cluster_id: String,
    pub label: String,
}

/// Typed context for pairwise judgments.
///
/// Interpret this as:
/// "Compare entities in `source_cluster_id` with respect to `target_node_id`."
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgmentContext {
    pub id: String,
    pub relation_type: RelationType,
    pub target_node_id: String,
    pub source_cluster_id: String,
    pub prompt_text: String,
    pub semantics_version: u32,
    pub judgment_kind: JudgmentKind,
    /// Optional weight among incoming source clusters for this target node.
    /// If multiple contexts share the same target node, weights are normalized.
    pub incoming_cluster_weight: Option<f64>,
}

/// A single pairwise ratio judgment under a context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseJudgment {
    pub context_id: String,
    pub entity_a_id: String,
    pub entity_b_id: String,
    pub ratio: f64,
    pub confidence: f64,
    pub rater_id: String,
    #[serde(default)]
    pub notes: Option<String>,
}

/// Lightweight ANP network container.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AnpNetwork {
    pub clusters: Vec<Cluster>,
    pub nodes: Vec<Node>,
    pub contexts: Vec<JudgmentContext>,
}

/// Configuration for local context fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalFitConfig {
    /// Floor term in confidence -> weight mapping.
    pub confidence_floor: f64,
    /// Exponent in confidence -> weight mapping.
    pub confidence_gamma: f64,
    /// Tikhonov regularization on reduced normal equations.
    pub ridge_lambda: f64,
    /// If weighted RMSE is above this threshold, suggest pairwise_only.
    pub composable_weighted_rmse_threshold: f64,
    /// Require at least this many judgments before suggesting composable.
    pub min_judgments_for_composable: usize,
}

impl Default for LocalFitConfig {
    fn default() -> Self {
        Self {
            confidence_floor: 1e-3,
            confidence_gamma: 2.0,
            ridge_lambda: 1e-9,
            composable_weighted_rmse_threshold: 0.35,
            min_judgments_for_composable: 3,
        }
    }
}

/// Result of fitting one ANP context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalFitResult {
    pub context_id: String,
    pub node_ids: Vec<String>,
    pub ln_scores: Vec<f64>,
    pub priorities: Vec<f64>,
    /// Approximate posterior variance per node score in local gauge coordinates.
    /// The first node is gauge-pinned and therefore has near-zero variance.
    pub diag_cov: Vec<f64>,
    pub residuals: Vec<f64>,
    pub weighted_rmse: f64,
    pub mean_abs_residual: f64,
    pub total_weight: f64,
    pub judgment_count: usize,
    pub suggested_judgment_kind: JudgmentKind,
}

/// Weighted ANP supermatrix.
#[derive(Debug, Clone)]
pub struct Supermatrix {
    pub node_ids: Vec<String>,
    /// Column-stochastic matrix. Column `j` encodes incoming influence on target node `j`.
    pub values: DMatrix<f64>,
}

/// Power-iteration configuration for damped stationary priorities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationaryConfig {
    /// Damping factor in (0,1). Typical values: 0.8..0.95.
    pub damping: f64,
    /// L1 delta threshold for convergence.
    pub tolerance: f64,
    /// Iteration cap.
    pub max_iterations: usize,
    /// Optional teleport/base distribution. If omitted, uses uniform.
    pub teleport: Option<Vec<f64>>,
}

impl Default for StationaryConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            tolerance: 1e-10,
            max_iterations: 10_000,
            teleport: None,
        }
    }
}

/// Result of stationary solve on a supermatrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StationaryResult {
    pub distribution: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub l1_delta: f64,
}

#[derive(Debug, Error, PartialEq)]
pub enum AnpError {
    #[error("network must contain at least one node")]
    EmptyNetwork,
    #[error("source node set for local fit is empty")]
    EmptySourceNodes,
    #[error("context {context_id} has no valid judgments")]
    NoJudgments { context_id: String },
    #[error("duplicate node id: {node_id}")]
    DuplicateNodeId { node_id: String },
    #[error("unknown node id referenced by context or judgment: {node_id}")]
    UnknownNodeId { node_id: String },
    #[error("unknown cluster id: {cluster_id}")]
    UnknownClusterId { cluster_id: String },
    #[error("context id mismatch for local fit: expected {expected}, got {got}")]
    ContextIdMismatch { expected: String, got: String },
    #[error("invalid ratio in context {context_id}: {ratio}")]
    InvalidRatio { context_id: String, ratio: f64 },
    #[error("invalid confidence in context {context_id}: {confidence}")]
    InvalidConfidence { context_id: String, confidence: f64 },
    #[error("singular local system while fitting context {context_id}")]
    SingularLocalSystem { context_id: String },
    #[error("missing local fit for context {context_id}")]
    MissingLocalFit { context_id: String },
    #[error("local fit for context {context_id} references node {node_id} outside source cluster {source_cluster_id}")]
    FitNodeOutsideSourceCluster {
        context_id: String,
        node_id: String,
        source_cluster_id: String,
    },
    #[error("invalid incoming cluster weight in context {context_id}: {weight}")]
    InvalidIncomingClusterWeight { context_id: String, weight: f64 },
    #[error("supermatrix is not square")]
    NonSquareSupermatrix,
    #[error("invalid damping factor: {damping}")]
    InvalidDamping { damping: f64 },
    #[error("invalid teleport distribution")]
    InvalidTeleport,
    #[error("distribution length {got} does not match node count {expected}")]
    DistributionLengthMismatch { expected: usize, got: usize },
    #[error("context {context_id} has no candidate nodes in source cluster {source_cluster_id}")]
    EmptySourceClusterForContext {
        context_id: String,
        source_cluster_id: String,
    },
}

fn confidence_to_weight(confidence: f64, cfg: &LocalFitConfig) -> f64 {
    let c = confidence.clamp(0.0, 1.0);
    cfg.confidence_floor + (1.0 - cfg.confidence_floor) * c.powf(cfg.confidence_gamma)
}

fn classify_context_kind(
    context: &JudgmentContext,
    weighted_rmse: f64,
    judgment_count: usize,
    cfg: &LocalFitConfig,
) -> JudgmentKind {
    if matches!(context.judgment_kind, JudgmentKind::PairwiseOnlyRatio) {
        return JudgmentKind::PairwiseOnlyRatio;
    }
    if judgment_count < cfg.min_judgments_for_composable {
        return JudgmentKind::PairwiseOnlyRatio;
    }
    if weighted_rmse > cfg.composable_weighted_rmse_threshold {
        return JudgmentKind::PairwiseOnlyRatio;
    }
    JudgmentKind::ComposableRatio
}

fn covariance_diag(h: &DMatrix<f64>) -> Option<Vec<f64>> {
    let n = h.nrows();
    if n == 0 || n != h.ncols() {
        return None;
    }
    if let Some(chol) = h.clone().cholesky() {
        let mut out = vec![0.0; n];
        for idx in 0..n {
            let mut e = DVector::<f64>::zeros(n);
            e[idx] = 1.0;
            let x = chol.solve(&e);
            out[idx] = x[idx];
        }
        return Some(out);
    }
    let lu = h.clone().lu();
    let mut out = vec![0.0; n];
    for idx in 0..n {
        let mut e = DVector::<f64>::zeros(n);
        e[idx] = 1.0;
        let x = lu.solve(&e)?;
        out[idx] = x[idx];
    }
    Some(out)
}

/// Fit one context via confidence-weighted log-ratio least squares.
///
/// Gauge convention: first `source_nodes` entry is pinned to zero in log-space.
pub fn fit_context(
    context: &JudgmentContext,
    source_nodes: &[Node],
    judgments: &[PairwiseJudgment],
    cfg: &LocalFitConfig,
) -> Result<LocalFitResult, AnpError> {
    if source_nodes.is_empty() {
        return Err(AnpError::EmptySourceNodes);
    }

    let mut node_index = HashMap::with_capacity(source_nodes.len());
    for (idx, node) in source_nodes.iter().enumerate() {
        if let Some(_prev) = node_index.insert(node.id.clone(), idx) {
            return Err(AnpError::DuplicateNodeId {
                node_id: node.id.clone(),
            });
        }
    }

    if source_nodes.len() == 1 {
        return Ok(LocalFitResult {
            context_id: context.id.clone(),
            node_ids: vec![source_nodes[0].id.clone()],
            ln_scores: vec![0.0],
            priorities: vec![1.0],
            diag_cov: vec![0.0],
            residuals: Vec::new(),
            weighted_rmse: 0.0,
            mean_abs_residual: 0.0,
            total_weight: 0.0,
            judgment_count: 0,
            suggested_judgment_kind: context.judgment_kind,
        });
    }

    #[derive(Clone, Copy)]
    struct Row {
        i: usize,
        j: usize,
        y: f64,
        w: f64,
    }

    let mut rows: Vec<Row> = Vec::with_capacity(judgments.len());
    for j in judgments {
        if j.context_id != context.id {
            return Err(AnpError::ContextIdMismatch {
                expected: context.id.clone(),
                got: j.context_id.clone(),
            });
        }
        if !j.ratio.is_finite() || j.ratio <= 0.0 {
            return Err(AnpError::InvalidRatio {
                context_id: context.id.clone(),
                ratio: j.ratio,
            });
        }
        if !j.confidence.is_finite() {
            return Err(AnpError::InvalidConfidence {
                context_id: context.id.clone(),
                confidence: j.confidence,
            });
        }
        let i = *node_index
            .get(&j.entity_a_id)
            .ok_or_else(|| AnpError::UnknownNodeId {
                node_id: j.entity_a_id.clone(),
            })?;
        let k = *node_index
            .get(&j.entity_b_id)
            .ok_or_else(|| AnpError::UnknownNodeId {
                node_id: j.entity_b_id.clone(),
            })?;
        if i == k {
            continue;
        }
        let y = j.ratio.ln();
        let w = confidence_to_weight(j.confidence, cfg);
        rows.push(Row { i, j: k, y, w });
    }

    if rows.is_empty() {
        return Err(AnpError::NoJudgments {
            context_id: context.id.clone(),
        });
    }

    let n = source_nodes.len();
    let reduced_n = n - 1;
    let mut h = DMatrix::<f64>::zeros(reduced_n, reduced_n);
    let mut b = DVector::<f64>::zeros(reduced_n);

    for row in &rows {
        let mut add_coeff = |idx: usize, coeff: f64| {
            if idx == 0 {
                return;
            }
            let ridx = idx - 1;
            b[ridx] += row.w * row.y * coeff;
            h[(ridx, ridx)] += row.w * coeff * coeff;
        };

        add_coeff(row.i, 1.0);
        add_coeff(row.j, -1.0);

        if row.i != 0 && row.j != 0 {
            let ri = row.i - 1;
            let rj = row.j - 1;
            let cross = row.w * -1.0;
            h[(ri, rj)] += cross;
            h[(rj, ri)] += cross;
        }
    }

    for d in 0..reduced_n {
        h[(d, d)] += cfg.ridge_lambda;
    }

    let reduced_diag_cov = covariance_diag(&h).ok_or_else(|| AnpError::SingularLocalSystem {
        context_id: context.id.clone(),
    })?;

    let solved = h
        .clone()
        .cholesky()
        .map(|chol| chol.solve(&b))
        .or_else(|| h.clone().lu().solve(&b))
        .ok_or_else(|| AnpError::SingularLocalSystem {
            context_id: context.id.clone(),
        })?;

    let mut ln_scores = vec![0.0; n];
    for idx in 1..n {
        ln_scores[idx] = solved[idx - 1];
    }

    let mut diag_cov = vec![0.0; n];
    for idx in 1..n {
        diag_cov[idx] = reduced_diag_cov[idx - 1].max(0.0);
    }

    let mut residuals = Vec::with_capacity(rows.len());
    let mut weighted_sq = 0.0;
    let mut abs_sum = 0.0;
    let mut total_weight = 0.0;
    for row in &rows {
        let pred = ln_scores[row.i] - ln_scores[row.j];
        let res = row.y - pred;
        residuals.push(res);
        weighted_sq += row.w * res * res;
        abs_sum += res.abs();
        total_weight += row.w;
    }
    let weighted_rmse = if total_weight > 0.0 {
        (weighted_sq / total_weight).sqrt()
    } else {
        0.0
    };
    let mean_abs_residual = if residuals.is_empty() {
        0.0
    } else {
        abs_sum / residuals.len() as f64
    };

    let max_ln = ln_scores
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let mut priorities: Vec<f64> = ln_scores.iter().map(|s| (s - max_ln).exp()).collect();
    let z: f64 = priorities.iter().sum();
    if z > 0.0 {
        for p in &mut priorities {
            *p /= z;
        }
    }

    let suggested_judgment_kind = classify_context_kind(context, weighted_rmse, rows.len(), cfg);

    Ok(LocalFitResult {
        context_id: context.id.clone(),
        node_ids: source_nodes.iter().map(|n| n.id.clone()).collect(),
        ln_scores,
        priorities,
        diag_cov,
        residuals,
        weighted_rmse,
        mean_abs_residual,
        total_weight,
        judgment_count: rows.len(),
        suggested_judgment_kind,
    })
}

/// Query policy for selecting which context to ask next.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSelectionConfig {
    /// Default sensitivity used when no explicit context sensitivity is provided.
    pub default_sensitivity: f64,
    /// Multiplicative weight on inconsistency (weighted RMSE).
    pub inconsistency_weight: f64,
    /// Multiplicative weight on exploration term (low-sample contexts).
    pub exploration_weight: f64,
    /// If true, skip contexts that are explicitly typed pairwise-only.
    pub composable_only: bool,
}

impl Default for ContextSelectionConfig {
    fn default() -> Self {
        Self {
            default_sensitivity: 1.0,
            inconsistency_weight: 0.5,
            exploration_weight: 0.25,
            composable_only: false,
        }
    }
}

/// Scored context candidate for active query selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextPriority {
    pub context_id: String,
    pub score: f64,
    pub sensitivity: f64,
    pub uncertainty: f64,
    pub inconsistency: f64,
    pub exploration: f64,
    pub has_fit: bool,
}

/// Pair candidate inside one context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairPriority {
    pub entity_a_id: String,
    pub entity_b_id: String,
    /// Approximate variance of score difference.
    pub variance: f64,
}

/// Full next-query proposal for an agent loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NextQueryProposal {
    pub context_id: String,
    pub entity_a_id: String,
    pub entity_b_id: String,
    pub context_score: f64,
    pub pair_variance: f64,
}

/// Rank contexts by expected decision value using sensitivity, uncertainty, and inconsistency.
pub fn rank_contexts_for_query(
    network: &AnpNetwork,
    local_fits: &HashMap<String, LocalFitResult>,
    context_sensitivity: &HashMap<String, f64>,
    cfg: &ContextSelectionConfig,
) -> Vec<ContextPriority> {
    let mut out = Vec::new();
    for ctx in &network.contexts {
        if cfg.composable_only && matches!(ctx.judgment_kind, JudgmentKind::PairwiseOnlyRatio) {
            continue;
        }
        let sensitivity = *context_sensitivity
            .get(&ctx.id)
            .unwrap_or(&cfg.default_sensitivity);
        let fit_opt = local_fits.get(&ctx.id);
        let (uncertainty, inconsistency, exploration, has_fit) = if let Some(fit) = fit_opt {
            let uncertainty = if fit.diag_cov.is_empty() {
                1.0
            } else {
                fit.diag_cov.iter().sum::<f64>() / fit.diag_cov.len() as f64
            };
            let inconsistency = fit.weighted_rmse;
            let exploration = 1.0 / (fit.judgment_count as f64 + 1.0).sqrt();
            (uncertainty, inconsistency, exploration, true)
        } else {
            (1.0, 0.0, 1.0, false)
        };
        let score = sensitivity * uncertainty
            + cfg.inconsistency_weight * inconsistency
            + cfg.exploration_weight * exploration;
        out.push(ContextPriority {
            context_id: ctx.id.clone(),
            score,
            sensitivity,
            uncertainty,
            inconsistency,
            exploration,
            has_fit,
        });
    }
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.context_id.cmp(&b.context_id))
    });
    out
}

/// Pick the highest-variance pair from one local fit.
pub fn select_next_pair(fit: &LocalFitResult) -> Option<PairPriority> {
    if fit.node_ids.len() < 2 {
        return None;
    }
    let mut best: Option<PairPriority> = None;
    for i in 0..fit.node_ids.len() {
        for j in (i + 1)..fit.node_ids.len() {
            let var = fit.diag_cov.get(i).copied().unwrap_or(0.0)
                + fit.diag_cov.get(j).copied().unwrap_or(0.0);
            let candidate = PairPriority {
                entity_a_id: fit.node_ids[i].clone(),
                entity_b_id: fit.node_ids[j].clone(),
                variance: var,
            };
            match &best {
                Some(cur) if cur.variance >= candidate.variance => {}
                _ => best = Some(candidate),
            }
        }
    }
    best
}

/// Choose the next context and pair to query.
///
/// Falls back to the first two nodes of the context's source cluster if no fit exists yet.
pub fn propose_next_query(
    network: &AnpNetwork,
    local_fits: &HashMap<String, LocalFitResult>,
    context_sensitivity: &HashMap<String, f64>,
    cfg: &ContextSelectionConfig,
) -> Option<NextQueryProposal> {
    let ranked = rank_contexts_for_query(network, local_fits, context_sensitivity, cfg);
    for ctx_priority in ranked {
        let ctx = network
            .contexts
            .iter()
            .find(|c| c.id == ctx_priority.context_id)?;
        if let Some(fit) = local_fits.get(&ctx.id) {
            if let Some(pair) = select_next_pair(fit) {
                return Some(NextQueryProposal {
                    context_id: ctx.id.clone(),
                    entity_a_id: pair.entity_a_id,
                    entity_b_id: pair.entity_b_id,
                    context_score: ctx_priority.score,
                    pair_variance: pair.variance,
                });
            }
        }
        let candidates: Vec<&Node> = network
            .nodes
            .iter()
            .filter(|n| n.cluster_id == ctx.source_cluster_id)
            .collect();
        if candidates.len() >= 2 {
            return Some(NextQueryProposal {
                context_id: ctx.id.clone(),
                entity_a_id: candidates[0].id.clone(),
                entity_b_id: candidates[1].id.clone(),
                context_score: ctx_priority.score,
                pair_variance: 0.0,
            });
        }
    }
    None
}

fn normalize_column(values: &mut DMatrix<f64>, col: usize) -> f64 {
    let mut sum = 0.0;
    for row in 0..values.nrows() {
        sum += values[(row, col)];
    }
    if sum > 0.0 {
        for row in 0..values.nrows() {
            values[(row, col)] /= sum;
        }
    }
    sum
}

/// Build a weighted, column-stochastic supermatrix from composable contexts.
pub fn build_weighted_supermatrix(
    network: &AnpNetwork,
    local_fits: &HashMap<String, LocalFitResult>,
) -> Result<Supermatrix, AnpError> {
    if network.nodes.is_empty() {
        return Err(AnpError::EmptyNetwork);
    }

    let mut node_index = HashMap::with_capacity(network.nodes.len());
    for (idx, node) in network.nodes.iter().enumerate() {
        if let Some(_prev) = node_index.insert(node.id.clone(), idx) {
            return Err(AnpError::DuplicateNodeId {
                node_id: node.id.clone(),
            });
        }
    }

    let cluster_exists: HashMap<String, ()> = network
        .clusters
        .iter()
        .map(|c| (c.id.clone(), ()))
        .collect();

    for node in &network.nodes {
        if !cluster_exists.contains_key(&node.cluster_id) {
            return Err(AnpError::UnknownClusterId {
                cluster_id: node.cluster_id.clone(),
            });
        }
    }
    for ctx in &network.contexts {
        if !node_index.contains_key(&ctx.target_node_id) {
            return Err(AnpError::UnknownNodeId {
                node_id: ctx.target_node_id.clone(),
            });
        }
        if !cluster_exists.contains_key(&ctx.source_cluster_id) {
            return Err(AnpError::UnknownClusterId {
                cluster_id: ctx.source_cluster_id.clone(),
            });
        }
    }

    let n = network.nodes.len();
    let mut values = DMatrix::<f64>::zeros(n, n);

    for (col_idx, target_node) in network.nodes.iter().enumerate() {
        let mut contexts_for_target: Vec<&JudgmentContext> = network
            .contexts
            .iter()
            .filter(|ctx| {
                ctx.target_node_id == target_node.id
                    && matches!(ctx.judgment_kind, JudgmentKind::ComposableRatio)
            })
            .collect();

        contexts_for_target.retain(|ctx| local_fits.contains_key(&ctx.id));

        if contexts_for_target.is_empty() {
            values[(col_idx, col_idx)] = 1.0;
            continue;
        }

        let mut alpha_sum = 0.0;
        for ctx in &contexts_for_target {
            if !cluster_exists.contains_key(&ctx.source_cluster_id) {
                return Err(AnpError::UnknownClusterId {
                    cluster_id: ctx.source_cluster_id.clone(),
                });
            }
            let alpha = ctx.incoming_cluster_weight.unwrap_or(1.0);
            if !alpha.is_finite() || alpha <= 0.0 {
                return Err(AnpError::InvalidIncomingClusterWeight {
                    context_id: ctx.id.clone(),
                    weight: alpha,
                });
            }
            alpha_sum += alpha;
        }

        if alpha_sum <= 0.0 {
            values[(col_idx, col_idx)] = 1.0;
            continue;
        }

        for ctx in &contexts_for_target {
            let fit = local_fits
                .get(&ctx.id)
                .ok_or_else(|| AnpError::MissingLocalFit {
                    context_id: ctx.id.clone(),
                })?;

            let alpha = ctx.incoming_cluster_weight.unwrap_or(1.0);
            let alpha_norm = alpha / alpha_sum;

            for (k, node_id) in fit.node_ids.iter().enumerate() {
                let row_idx = *node_index
                    .get(node_id)
                    .ok_or_else(|| AnpError::UnknownNodeId {
                        node_id: node_id.clone(),
                    })?;
                let row_node = &network.nodes[row_idx];
                if row_node.cluster_id != ctx.source_cluster_id {
                    return Err(AnpError::FitNodeOutsideSourceCluster {
                        context_id: ctx.id.clone(),
                        node_id: node_id.clone(),
                        source_cluster_id: ctx.source_cluster_id.clone(),
                    });
                }
                let p = *fit
                    .priorities
                    .get(k)
                    .ok_or_else(|| AnpError::MissingLocalFit {
                        context_id: ctx.id.clone(),
                    })?;
                if !p.is_finite() || p < 0.0 {
                    continue;
                }
                values[(row_idx, col_idx)] += alpha_norm * p;
            }
        }

        let col_sum = normalize_column(&mut values, col_idx);
        if col_sum <= 0.0 {
            values[(col_idx, col_idx)] = 1.0;
        }
    }

    Ok(Supermatrix {
        node_ids: network.nodes.iter().map(|n| n.id.clone()).collect(),
        values,
    })
}

fn normalize_distribution(v: &[f64]) -> Result<Vec<f64>, AnpError> {
    if v.is_empty() {
        return Err(AnpError::InvalidTeleport);
    }
    if !v.iter().all(|x| x.is_finite() && *x >= 0.0) {
        return Err(AnpError::InvalidTeleport);
    }
    let sum: f64 = v.iter().sum();
    if sum <= 0.0 {
        return Err(AnpError::InvalidTeleport);
    }
    Ok(v.iter().map(|x| x / sum).collect())
}

/// Solve damped stationary priorities for a supermatrix.
pub fn solve_stationary(
    supermatrix: &Supermatrix,
    cfg: &StationaryConfig,
) -> Result<StationaryResult, AnpError> {
    let n = supermatrix.values.nrows();
    if n == 0 {
        return Err(AnpError::EmptyNetwork);
    }
    if n != supermatrix.values.ncols() {
        return Err(AnpError::NonSquareSupermatrix);
    }
    if !(0.0 < cfg.damping && cfg.damping < 1.0) {
        return Err(AnpError::InvalidDamping {
            damping: cfg.damping,
        });
    }

    let teleport = match &cfg.teleport {
        Some(v) => {
            if v.len() != n {
                return Err(AnpError::DistributionLengthMismatch {
                    expected: n,
                    got: v.len(),
                });
            }
            normalize_distribution(v)?
        }
        None => vec![1.0 / n as f64; n],
    };

    let mut v = DVector::from_vec(teleport.clone());
    let teleport_vec = DVector::from_vec(teleport);
    let one_minus = 1.0 - cfg.damping;

    let mut l1_delta = f64::INFINITY;
    for iter in 1..=cfg.max_iterations {
        let mut next = &supermatrix.values * &v;
        next *= cfg.damping;
        for i in 0..n {
            next[i] += one_minus * teleport_vec[i];
        }
        let next_sum: f64 = next.iter().sum();
        if next_sum <= 0.0 || !next_sum.is_finite() {
            return Err(AnpError::InvalidTeleport);
        }
        next /= next_sum;

        l1_delta = (0..n).map(|i| (next[i] - v[i]).abs()).sum();
        v = next;
        if l1_delta <= cfg.tolerance {
            return Ok(StationaryResult {
                distribution: v.iter().copied().collect(),
                iterations: iter,
                converged: true,
                l1_delta,
            });
        }
    }

    Ok(StationaryResult {
        distribution: v.iter().copied().collect(),
        iterations: cfg.max_iterations,
        converged: false,
        l1_delta,
    })
}

/// Extract and renormalize stationary mass for one cluster.
pub fn cluster_priorities(
    network: &AnpNetwork,
    stationary: &StationaryResult,
    cluster_id: &str,
) -> Result<Vec<(String, f64)>, AnpError> {
    if stationary.distribution.len() != network.nodes.len() {
        return Err(AnpError::DistributionLengthMismatch {
            expected: network.nodes.len(),
            got: stationary.distribution.len(),
        });
    }

    let mut out = Vec::new();
    let mut total = 0.0;
    for (idx, node) in network.nodes.iter().enumerate() {
        if node.cluster_id == cluster_id {
            let p = stationary.distribution[idx];
            total += p;
            out.push((node.id.clone(), p));
        }
    }

    if total > 0.0 {
        for (_, p) in &mut out {
            *p /= total;
        }
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(out)
}

/// Per-context local fit warning captured during demo runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextFitWarning {
    pub context_id: String,
    pub message: String,
}

/// JSON-friendly demo request for running one ANP pipeline offline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnpDemoRequest {
    pub network: AnpNetwork,
    pub judgments: Vec<PairwiseJudgment>,
    #[serde(default)]
    pub context_sensitivity: HashMap<String, f64>,
    #[serde(default)]
    pub local_fit_config: Option<LocalFitConfig>,
    #[serde(default)]
    pub context_selection_config: Option<ContextSelectionConfig>,
    #[serde(default)]
    pub stationary_config: Option<StationaryConfig>,
    /// Optional cluster id for cluster-normalized priority output.
    #[serde(default)]
    pub priority_cluster_id: Option<String>,
}

/// JSON-friendly demo output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnpDemoOutput {
    pub local_fits: Vec<LocalFitResult>,
    pub fit_warnings: Vec<ContextFitWarning>,
    pub context_priorities: Vec<ContextPriority>,
    pub next_query: Option<NextQueryProposal>,
    pub supermatrix: Vec<Vec<f64>>,
    pub stationary: StationaryResult,
    pub cluster_priorities: Vec<(String, f64)>,
}

/// Run one full ANP pass for iterative agent workflows.
pub fn run_demo(req: AnpDemoRequest) -> Result<AnpDemoOutput, AnpError> {
    let local_cfg = req.local_fit_config.unwrap_or_default();
    let context_cfg = req.context_selection_config.unwrap_or_default();
    let stationary_cfg = req.stationary_config.unwrap_or_default();

    let mut grouped: HashMap<String, Vec<PairwiseJudgment>> = HashMap::new();
    for j in req.judgments {
        grouped.entry(j.context_id.clone()).or_default().push(j);
    }

    let mut local_fits_map: HashMap<String, LocalFitResult> = HashMap::new();
    let mut fit_warnings = Vec::new();
    for ctx in &req.network.contexts {
        let source_nodes: Vec<Node> = req
            .network
            .nodes
            .iter()
            .filter(|n| n.cluster_id == ctx.source_cluster_id)
            .cloned()
            .collect();
        if source_nodes.is_empty() {
            return Err(AnpError::EmptySourceClusterForContext {
                context_id: ctx.id.clone(),
                source_cluster_id: ctx.source_cluster_id.clone(),
            });
        }
        let empty = Vec::new();
        let judgment_slice = grouped.get(&ctx.id).unwrap_or(&empty);
        match fit_context(ctx, &source_nodes, judgment_slice, &local_cfg) {
            Ok(fit) => {
                local_fits_map.insert(ctx.id.clone(), fit);
            }
            Err(AnpError::NoJudgments { .. }) => {
                fit_warnings.push(ContextFitWarning {
                    context_id: ctx.id.clone(),
                    message: "no judgments".to_string(),
                });
            }
            Err(e) => return Err(e),
        }
    }

    let context_priorities = rank_contexts_for_query(
        &req.network,
        &local_fits_map,
        &req.context_sensitivity,
        &context_cfg,
    );
    let next_query = propose_next_query(
        &req.network,
        &local_fits_map,
        &req.context_sensitivity,
        &context_cfg,
    );
    let sm = build_weighted_supermatrix(&req.network, &local_fits_map)?;
    let stationary = solve_stationary(&sm, &stationary_cfg)?;

    let priority_cluster_id = req
        .priority_cluster_id
        .or_else(|| {
            req.network
                .clusters
                .iter()
                .find(|c| c.id == "alts")
                .map(|c| c.id.clone())
        })
        .or_else(|| req.network.clusters.first().map(|c| c.id.clone()));
    let cluster_priorities = if let Some(cluster_id) = priority_cluster_id {
        cluster_priorities(&req.network, &stationary, &cluster_id)?
    } else {
        Vec::new()
    };

    let mut local_fits: Vec<LocalFitResult> = local_fits_map.into_values().collect();
    local_fits.sort_by(|a, b| a.context_id.cmp(&b.context_id));
    let mut supermatrix = Vec::with_capacity(sm.values.nrows());
    for r in 0..sm.values.nrows() {
        let mut row = Vec::with_capacity(sm.values.ncols());
        for c in 0..sm.values.ncols() {
            row.push(sm.values[(r, c)]);
        }
        supermatrix.push(row);
    }

    Ok(AnpDemoOutput {
        local_fits,
        fit_warnings,
        context_priorities,
        next_query,
        supermatrix,
        stationary,
        cluster_priorities,
    })
}

/// One synthetic benchmark result comparing typed vs forced composability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnpBenchmarkResult {
    pub case_name: String,
    pub mode: String,
    pub top_entity_id: String,
    pub top_entity_score: f64,
    pub top1_correct: bool,
    pub kendall_tau: f64,
    pub next_query: Option<NextQueryProposal>,
}

fn rank_ids_from_cluster_priorities(items: &[(String, f64)]) -> Vec<String> {
    items.iter().map(|(id, _)| id.clone()).collect()
}

fn kendall_tau_from_order(predicted: &[String], truth: &[String]) -> f64 {
    if predicted.len() < 2 || predicted.len() != truth.len() {
        return 0.0;
    }
    let mut truth_pos = HashMap::new();
    for (idx, id) in truth.iter().enumerate() {
        truth_pos.insert(id, idx);
    }
    let mut concordant = 0.0;
    let mut discordant = 0.0;
    for i in 0..predicted.len() {
        for j in (i + 1)..predicted.len() {
            let a = &predicted[i];
            let b = &predicted[j];
            let Some(&ta) = truth_pos.get(a) else {
                continue;
            };
            let Some(&tb) = truth_pos.get(b) else {
                continue;
            };
            if ta < tb {
                concordant += 1.0;
            } else if ta > tb {
                discordant += 1.0;
            }
        }
    }
    let denom = concordant + discordant;
    if denom <= 0.0 {
        0.0
    } else {
        (concordant - discordant) / denom
    }
}

fn synthetic_open_ended_case(distance_kind: JudgmentKind) -> AnpDemoRequest {
    let network = AnpNetwork {
        clusters: vec![
            Cluster {
                id: "goal".to_string(),
                label: "Goal".to_string(),
            },
            Cluster {
                id: "criteria".to_string(),
                label: "Criteria".to_string(),
            },
            Cluster {
                id: "alts".to_string(),
                label: "Alternatives".to_string(),
            },
        ],
        nodes: vec![
            Node {
                id: "goal".to_string(),
                cluster_id: "goal".to_string(),
                label: "Goal".to_string(),
            },
            Node {
                id: "impact".to_string(),
                cluster_id: "criteria".to_string(),
                label: "Impact".to_string(),
            },
            Node {
                id: "distance".to_string(),
                cluster_id: "criteria".to_string(),
                label: "Distance".to_string(),
            },
            Node {
                id: "a".to_string(),
                cluster_id: "alts".to_string(),
                label: "Alternative A".to_string(),
            },
            Node {
                id: "b".to_string(),
                cluster_id: "alts".to_string(),
                label: "Alternative B".to_string(),
            },
            Node {
                id: "c".to_string(),
                cluster_id: "alts".to_string(),
                label: "Alternative C".to_string(),
            },
        ],
        contexts: vec![
            JudgmentContext {
                id: "ctx_goal_from_criteria".to_string(),
                relation_type: RelationType::Preference,
                target_node_id: "goal".to_string(),
                source_cluster_id: "criteria".to_string(),
                prompt_text:
                    "Compared to impact, how much more salient is distance for this stakeholder?"
                        .to_string(),
                semantics_version: 1,
                judgment_kind: JudgmentKind::ComposableRatio,
                incoming_cluster_weight: None,
            },
            JudgmentContext {
                id: "ctx_impact_from_alts".to_string(),
                relation_type: RelationType::Preference,
                target_node_id: "impact".to_string(),
                source_cluster_id: "alts".to_string(),
                prompt_text: "Expected impact over two years.".to_string(),
                semantics_version: 1,
                judgment_kind: JudgmentKind::ComposableRatio,
                incoming_cluster_weight: None,
            },
            JudgmentContext {
                id: "ctx_distance_from_alts".to_string(),
                relation_type: RelationType::Preference,
                target_node_id: "distance".to_string(),
                source_cluster_id: "alts".to_string(),
                prompt_text: "Strategic distance from current baseline.".to_string(),
                semantics_version: 1,
                judgment_kind: distance_kind,
                incoming_cluster_weight: None,
            },
        ],
    };

    let judgments = vec![
        PairwiseJudgment {
            context_id: "ctx_goal_from_criteria".to_string(),
            entity_a_id: "distance".to_string(),
            entity_b_id: "impact".to_string(),
            ratio: 5.0,
            confidence: 0.9,
            rater_id: "sim".to_string(),
            notes: None,
        },
        PairwiseJudgment {
            context_id: "ctx_impact_from_alts".to_string(),
            entity_a_id: "a".to_string(),
            entity_b_id: "b".to_string(),
            ratio: 2.0,
            confidence: 0.95,
            rater_id: "sim".to_string(),
            notes: None,
        },
        PairwiseJudgment {
            context_id: "ctx_impact_from_alts".to_string(),
            entity_a_id: "a".to_string(),
            entity_b_id: "c".to_string(),
            ratio: 4.0,
            confidence: 0.95,
            rater_id: "sim".to_string(),
            notes: None,
        },
        PairwiseJudgment {
            context_id: "ctx_impact_from_alts".to_string(),
            entity_a_id: "b".to_string(),
            entity_b_id: "c".to_string(),
            ratio: 2.0,
            confidence: 0.95,
            rater_id: "sim".to_string(),
            notes: None,
        },
        PairwiseJudgment {
            context_id: "ctx_distance_from_alts".to_string(),
            entity_a_id: "c".to_string(),
            entity_b_id: "a".to_string(),
            ratio: 6.0,
            confidence: 0.9,
            rater_id: "sim".to_string(),
            notes: Some("Example of non-composable framing axis".to_string()),
        },
        PairwiseJudgment {
            context_id: "ctx_distance_from_alts".to_string(),
            entity_a_id: "c".to_string(),
            entity_b_id: "b".to_string(),
            ratio: 6.0,
            confidence: 0.9,
            rater_id: "sim".to_string(),
            notes: Some("Example of non-composable framing axis".to_string()),
        },
        PairwiseJudgment {
            context_id: "ctx_distance_from_alts".to_string(),
            entity_a_id: "a".to_string(),
            entity_b_id: "b".to_string(),
            ratio: 1.0,
            confidence: 0.9,
            rater_id: "sim".to_string(),
            notes: Some("Example of non-composable framing axis".to_string()),
        },
    ];

    AnpDemoRequest {
        network,
        judgments,
        context_sensitivity: HashMap::new(),
        local_fit_config: None,
        context_selection_config: Some(ContextSelectionConfig {
            default_sensitivity: 1.0,
            inconsistency_weight: 0.5,
            exploration_weight: 0.25,
            composable_only: false,
        }),
        stationary_config: Some(StationaryConfig {
            damping: 0.85,
            tolerance: 1e-10,
            max_iterations: 10_000,
            teleport: None,
        }),
        priority_cluster_id: Some("alts".to_string()),
    }
}

/// Run a synthetic benchmark that highlights typed handling of pairwise-only contexts.
pub fn run_synthetic_benchmark_suite() -> Result<Vec<AnpBenchmarkResult>, AnpError> {
    let truth = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let scenarios = vec![
        (
            "open_ended_priority_with_pairwise_only_axis",
            "typed_pairwise_only",
            synthetic_open_ended_case(JudgmentKind::PairwiseOnlyRatio),
        ),
        (
            "open_ended_priority_with_pairwise_only_axis",
            "forced_composable",
            synthetic_open_ended_case(JudgmentKind::ComposableRatio),
        ),
    ];

    let mut out = Vec::new();
    for (case_name, mode, req) in scenarios {
        let result = run_demo(req)?;
        let ranked = rank_ids_from_cluster_priorities(&result.cluster_priorities);
        let top_entity_id = ranked.first().cloned().unwrap_or_default();
        let top_entity_score = result
            .cluster_priorities
            .first()
            .map(|(_, p)| *p)
            .unwrap_or(0.0);
        let kendall_tau = kendall_tau_from_order(&ranked, &truth);
        out.push(AnpBenchmarkResult {
            case_name: case_name.to_string(),
            mode: mode.to_string(),
            top_entity_id: top_entity_id.clone(),
            top_entity_score,
            top1_correct: top_entity_id == truth[0],
            kendall_tau,
            next_query: result.next_query,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, cluster_id: &str) -> Node {
        Node {
            id: id.to_string(),
            cluster_id: cluster_id.to_string(),
            label: id.to_string(),
        }
    }

    fn context(
        id: &str,
        target_node_id: &str,
        source_cluster_id: &str,
        kind: JudgmentKind,
    ) -> JudgmentContext {
        JudgmentContext {
            id: id.to_string(),
            relation_type: RelationType::Preference,
            target_node_id: target_node_id.to_string(),
            source_cluster_id: source_cluster_id.to_string(),
            prompt_text: "test prompt".to_string(),
            semantics_version: 1,
            judgment_kind: kind,
            incoming_cluster_weight: None,
        }
    }

    fn judgment(
        context_id: &str,
        a: &str,
        b: &str,
        ratio: f64,
        confidence: f64,
    ) -> PairwiseJudgment {
        PairwiseJudgment {
            context_id: context_id.to_string(),
            entity_a_id: a.to_string(),
            entity_b_id: b.to_string(),
            ratio,
            confidence,
            rater_id: "rater_test".to_string(),
            notes: None,
        }
    }

    #[test]
    fn fit_context_recovers_consistent_signal() {
        let ctx = context("ctx", "criterion", "alts", JudgmentKind::ComposableRatio);
        let nodes = vec![node("a", "alts"), node("b", "alts"), node("c", "alts")];
        let judgments = vec![
            judgment("ctx", "a", "b", 2.0, 0.95),
            judgment("ctx", "b", "c", 3.0, 0.95),
            judgment("ctx", "a", "c", 6.0, 0.95),
        ];
        let fit = fit_context(&ctx, &nodes, &judgments, &LocalFitConfig::default()).unwrap();

        assert_eq!(fit.judgment_count, 3);
        assert!(fit.weighted_rmse < 1e-8);
        assert_eq!(fit.suggested_judgment_kind, JudgmentKind::ComposableRatio);
        assert!(fit.priorities[0] > fit.priorities[1]);
        assert!(fit.priorities[1] > fit.priorities[2]);
    }

    #[test]
    fn fit_context_downgrades_inconsistent_signal() {
        let ctx = context("ctx", "criterion", "alts", JudgmentKind::ComposableRatio);
        let nodes = vec![node("a", "alts"), node("b", "alts"), node("c", "alts")];
        let judgments = vec![
            judgment("ctx", "a", "b", 2.0, 0.9),
            judgment("ctx", "b", "c", 2.0, 0.9),
            judgment("ctx", "a", "c", 1.0, 0.9),
        ];
        let fit = fit_context(&ctx, &nodes, &judgments, &LocalFitConfig::default()).unwrap();

        assert!(fit.weighted_rmse > 0.35);
        assert_eq!(fit.suggested_judgment_kind, JudgmentKind::PairwiseOnlyRatio);
    }

    #[test]
    fn build_supermatrix_and_stationary_distribution_work() {
        let network = AnpNetwork {
            clusters: vec![
                Cluster {
                    id: "goal".to_string(),
                    label: "Goal".to_string(),
                },
                Cluster {
                    id: "criteria".to_string(),
                    label: "Criteria".to_string(),
                },
                Cluster {
                    id: "alts".to_string(),
                    label: "Alternatives".to_string(),
                },
            ],
            nodes: vec![
                node("g", "goal"),
                node("c1", "criteria"),
                node("a1", "alts"),
                node("a2", "alts"),
            ],
            contexts: vec![
                context(
                    "ctx_goal_from_criteria",
                    "g",
                    "criteria",
                    JudgmentKind::ComposableRatio,
                ),
                context(
                    "ctx_criterion_from_alts",
                    "c1",
                    "alts",
                    JudgmentKind::ComposableRatio,
                ),
            ],
        };

        let mut fits = HashMap::new();
        fits.insert(
            "ctx_goal_from_criteria".to_string(),
            LocalFitResult {
                context_id: "ctx_goal_from_criteria".to_string(),
                node_ids: vec!["c1".to_string()],
                ln_scores: vec![0.0],
                priorities: vec![1.0],
                diag_cov: vec![0.0],
                residuals: vec![],
                weighted_rmse: 0.0,
                mean_abs_residual: 0.0,
                total_weight: 0.0,
                judgment_count: 1,
                suggested_judgment_kind: JudgmentKind::ComposableRatio,
            },
        );
        fits.insert(
            "ctx_criterion_from_alts".to_string(),
            LocalFitResult {
                context_id: "ctx_criterion_from_alts".to_string(),
                node_ids: vec!["a1".to_string(), "a2".to_string()],
                ln_scores: vec![0.0, 0.0],
                priorities: vec![0.8, 0.2],
                diag_cov: vec![0.3, 0.1],
                residuals: vec![],
                weighted_rmse: 0.0,
                mean_abs_residual: 0.0,
                total_weight: 0.0,
                judgment_count: 1,
                suggested_judgment_kind: JudgmentKind::ComposableRatio,
            },
        );

        let sm = build_weighted_supermatrix(&network, &fits).unwrap();
        for c in 0..sm.values.ncols() {
            let sum: f64 = (0..sm.values.nrows()).map(|r| sm.values[(r, c)]).sum();
            assert!((sum - 1.0).abs() < 1e-9);
        }

        let stationary = solve_stationary(&sm, &StationaryConfig::default()).unwrap();
        assert!(stationary.converged);
        let total: f64 = stationary.distribution.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);

        let alts = cluster_priorities(&network, &stationary, "alts").unwrap();
        assert_eq!(alts.len(), 2);
        assert!(alts[0].1 > alts[1].1);
    }

    #[test]
    fn select_next_pair_prefers_high_variance_pair() {
        let fit = LocalFitResult {
            context_id: "ctx".to_string(),
            node_ids: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            ln_scores: vec![0.0, 0.1, 0.2],
            priorities: vec![0.4, 0.35, 0.25],
            diag_cov: vec![0.01, 0.8, 0.9],
            residuals: vec![],
            weighted_rmse: 0.0,
            mean_abs_residual: 0.0,
            total_weight: 0.0,
            judgment_count: 3,
            suggested_judgment_kind: JudgmentKind::ComposableRatio,
        };
        let next = select_next_pair(&fit).unwrap();
        assert_eq!(next.entity_a_id, "b");
        assert_eq!(next.entity_b_id, "c");
        assert!(next.variance > 1.6);
    }

    #[test]
    fn run_demo_emits_next_query_and_stationary_distribution() {
        let network = AnpNetwork {
            clusters: vec![
                Cluster {
                    id: "criteria".to_string(),
                    label: "Criteria".to_string(),
                },
                Cluster {
                    id: "alts".to_string(),
                    label: "Alternatives".to_string(),
                },
            ],
            nodes: vec![
                node("c1", "criteria"),
                node("a1", "alts"),
                node("a2", "alts"),
            ],
            contexts: vec![JudgmentContext {
                id: "ctx".to_string(),
                relation_type: RelationType::Preference,
                target_node_id: "c1".to_string(),
                source_cluster_id: "alts".to_string(),
                prompt_text: "tractability".to_string(),
                semantics_version: 1,
                judgment_kind: JudgmentKind::ComposableRatio,
                incoming_cluster_weight: None,
            }],
        };
        let req = AnpDemoRequest {
            network,
            judgments: vec![PairwiseJudgment {
                context_id: "ctx".to_string(),
                entity_a_id: "a1".to_string(),
                entity_b_id: "a2".to_string(),
                ratio: 2.0,
                confidence: 0.9,
                rater_id: "r1".to_string(),
                notes: None,
            }],
            context_sensitivity: HashMap::new(),
            local_fit_config: None,
            context_selection_config: None,
            stationary_config: None,
            priority_cluster_id: Some("alts".to_string()),
        };

        let out = run_demo(req).unwrap();
        assert_eq!(out.local_fits.len(), 1);
        assert!(out.stationary.converged);
        assert!((out.stationary.distribution.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(out.next_query.is_some());
        assert_eq!(out.cluster_priorities.len(), 2);
    }
}
