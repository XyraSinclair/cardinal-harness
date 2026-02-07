//! Multi-attribute trait search manager built on top of RatingEngine.
//!
//! Combines multiple attribute-specific rating engines into a unified
//! objective function with weighted combination and gate-based filtering.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::rating_engine::{self, plan_edges_for_rater, Observation, PlanProposal, RatingEngine};
const SCALE_FLOOR: f64 = 1e-6;
const MAD_TO_SIGMA: f64 = 1.4826;
const MIN_ATTR_UNCERTAINTY_WEIGHT: f64 = 0.1;

/// Maximum batch size for propose_batch to prevent resource exhaustion.
const MAX_BATCH_SIZE: usize = 10_000;
/// Max active set size for targeted marginal variance refinement.
const MAX_REFINED_ACTIVE: usize = 64;
/// Skip candidate pairs with negligible inversion probability.
const MIN_PAIR_PROB: f64 = 1e-4;
/// Floor for soft top-k membership weighting.
const MIN_MEMBERSHIP_WEIGHT: f64 = 0.05;
/// Cap planner candidates to avoid O(N^2) explosions.
const MAX_PLANNER_CANDIDATES: usize = 50_000;

// ------------------------------------------------------------------
// Math utilities
// ------------------------------------------------------------------

fn median(sorted: &[f64]) -> f64 {
    let len = sorted.len();
    let mid = len / 2;
    if len % 2 == 1 {
        sorted[mid]
    } else {
        0.5 * (sorted[mid - 1] + sorted[mid])
    }
}

fn stddev_population(scores: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let n = indices.len() as f64;
    let mean = indices.iter().map(|&i| scores[i]).sum::<f64>() / n;
    let var = indices
        .iter()
        .map(|&i| {
            let d = scores[i] - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    var.max(0.0).sqrt()
}

/// Compute robust MAD scale for scores (for weight normalization).
fn compute_attribute_scale(scores: &[f64]) -> f64 {
    let n = scores.len();
    let mut finite: Vec<usize> = (0..n).filter(|&i| scores[i].is_finite()).collect();

    if finite.is_empty() {
        return SCALE_FLOOR;
    }

    finite.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(Ordering::Equal));
    let m = finite.len();
    let mid = m / 2;
    let med = if m % 2 == 1 {
        scores[finite[mid]]
    } else {
        0.5 * (scores[finite[mid - 1]] + scores[finite[mid]])
    };

    let mut devs: Vec<f64> = finite.iter().map(|&i| (scores[i] - med).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mad_raw = median(&devs);
    if mad_raw >= SCALE_FLOOR {
        return mad_raw;
    }

    // Degenerate/tied distributions can yield MAD=0 even when there's meaningful spread (e.g. many ties + a few outliers).
    // Using SCALE_FLOOR directly makes normalized scores/z-scores explode; fall back to stddev-based scaling.
    let sigma = stddev_population(scores, &finite).max(SCALE_FLOOR);
    (sigma / MAD_TO_SIGMA).max(SCALE_FLOOR)
}

/// Compute robust derived units for a single attribute.
///
/// Derived units are computed over finite scores (non-finite scores get 0.0).
/// Returns (mad_scale, z_scores, min_normalized, percentiles).
pub(crate) fn compute_attribute_units(scores: &[f64]) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = scores.len();
    let mut finite: Vec<usize> = (0..n).filter(|&i| scores[i].is_finite()).collect();

    let mut z = vec![0.0; n];
    let mut min_norm = vec![0.0; n];
    let mut pct = vec![0.0; n];

    if finite.is_empty() {
        return (SCALE_FLOOR, z, min_norm, pct);
    }

    finite.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(Ordering::Equal));
    let m = finite.len();
    let mid = m / 2;
    let med = if m % 2 == 1 {
        scores[finite[mid]]
    } else {
        0.5 * (scores[finite[mid - 1]] + scores[finite[mid]])
    };

    let min_val = scores[finite[0]];
    for (rank, &i) in finite.iter().enumerate() {
        pct[i] = (rank as f64 + 0.5) / (m as f64);
    }

    let mut devs: Vec<f64> = finite.iter().map(|&i| (scores[i] - med).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mad_raw = median(&devs);
    let mad = if mad_raw >= SCALE_FLOOR {
        mad_raw
    } else {
        // See compute_attribute_scale() for rationale.
        let sigma = stddev_population(scores, &finite).max(SCALE_FLOOR);
        (sigma / MAD_TO_SIGMA).max(SCALE_FLOOR)
    };
    let mad_sigma = (mad * MAD_TO_SIGMA).max(SCALE_FLOOR);

    for i in 0..n {
        if scores[i].is_finite() {
            z[i] = (scores[i] - med) / mad_sigma;
            min_norm[i] = (scores[i] - min_val) + 1.0;
        }
    }

    (mad, z, min_norm, pct)
}

/// Map tolerated error to a conservative normal quantile.
fn beta_from_tolerated_error(tolerated_error: f64) -> f64 {
    let e = tolerated_error.clamp(1e-6, 0.5);
    if e <= 0.01 {
        2.58
    } else if e <= 0.05 {
        1.96
    } else if e <= 0.1 {
        1.64
    } else {
        1.28
    }
}

fn inversion_prob(delta: f64, var: f64) -> f64 {
    if var <= 0.0 {
        return if delta <= 0.0 { 1.0 } else { 0.0 };
    }
    let z = delta / var.sqrt();
    (1.0 - rating_engine::normal_cdf(z)).clamp(0.0, 1.0)
}

// ------------------------------------------------------------------
// Configuration & errors
// ------------------------------------------------------------------

#[derive(Debug)]
pub enum TraitSearchError {
    EmptyAttributes,
    NonPositiveEntities,
    MissingEngine {
        attribute_id: String,
    },
    EnginesSizeMismatch,
    EntityCountMismatch {
        config_n: usize,
        engine_n: usize,
    },
    GateUnknownAttribute {
        attribute_id: String,
    },
    UnsupportedGateOp {
        op: String,
    },
    UnsupportedGateUnit {
        unit: String,
    },
    PosteriorLengthMismatch {
        attribute_id: String,
        scores_len: usize,
        diag_cov_len: usize,
        expected_n: usize,
    },
    PlannerError {
        message: String,
    },
    InternalError {
        message: String,
    },
}

impl std::fmt::Display for TraitSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TraitSearchError::EmptyAttributes => {
                write!(f, "TraitSearchConfig.attributes must not be empty")
            }
            TraitSearchError::NonPositiveEntities => {
                write!(f, "TraitSearchConfig.n_entities must be positive")
            }
            TraitSearchError::MissingEngine { attribute_id } => {
                write!(f, "Missing RatingEngine for attribute '{attribute_id}'")
            }
            TraitSearchError::EnginesSizeMismatch => {
                write!(f, "All RatingEngine instances must share the same n")
            }
            TraitSearchError::EntityCountMismatch { config_n, engine_n } => {
                write!(
                    f,
                    "TraitSearchConfig.n_entities={config_n} does not match engine n={engine_n}"
                )
            }
            TraitSearchError::GateUnknownAttribute { attribute_id } => {
                write!(f, "Gate references unknown attribute '{attribute_id}'")
            }
            TraitSearchError::UnsupportedGateOp { op } => {
                write!(f, "Unsupported gate op '{op}' (expected \">=\" or \"<=\")")
            }
            TraitSearchError::UnsupportedGateUnit { unit } => {
                write!(
                    f,
                    "Unsupported gate unit '{unit}' (expected \"latent\", \"z\", \"percentile\", or \"min_norm\")"
                )
            }
            TraitSearchError::PosteriorLengthMismatch {
                attribute_id,
                scores_len,
                diag_cov_len,
                expected_n,
            } => {
                write!(
                    f,
                    "SolveSummary size mismatch for '{attribute_id}': \
                     scores={scores_len}, cov={diag_cov_len}, expected={expected_n}"
                )
            }
            TraitSearchError::PlannerError { message } => {
                write!(f, "Planner error: {message}")
            }
            TraitSearchError::InternalError { message } => {
                write!(f, "Internal error: {message}")
            }
        }
    }
}

impl std::error::Error for TraitSearchError {}

pub type Result<T> = std::result::Result<T, TraitSearchError>;

#[derive(Debug, Clone)]
pub struct AttributeConfig {
    pub id: String,
    pub weight: f64,
}

impl AttributeConfig {
    pub fn new(id: impl Into<String>, weight: f64) -> Self {
        Self {
            id: id.into(),
            weight,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GateSpec {
    pub attribute_id: String,
    pub unit: String,
    pub op: String,
    pub threshold: f64,
}

impl GateSpec {
    pub fn new(
        attribute_id: impl Into<String>,
        unit: impl Into<String>,
        op: impl Into<String>,
        threshold: f64,
    ) -> Self {
        Self {
            attribute_id: attribute_id.into(),
            unit: unit.into(),
            op: op.into(),
            threshold,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TopKConfig {
    pub k: usize,
    pub weight_exponent: f64,
    pub tolerated_error: f64,
    pub band_size: usize,
    pub effective_resistance_max_active: usize,
    pub stop_sigma_inflate: f64,
    pub stop_min_consecutive: usize,
}

impl TopKConfig {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            weight_exponent: 1.3,
            tolerated_error: 0.1,
            band_size: 5,
            effective_resistance_max_active: 64,
            stop_sigma_inflate: 1.25,
            stop_min_consecutive: 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TraitSearchConfig {
    pub n_entities: usize,
    pub attributes: Vec<AttributeConfig>,
    pub topk: TopKConfig,
    pub gates: Vec<GateSpec>,
}

impl TraitSearchConfig {
    pub fn new(
        n_entities: usize,
        attributes: Vec<AttributeConfig>,
        topk: TopKConfig,
        gates: Vec<GateSpec>,
    ) -> Self {
        Self {
            n_entities,
            attributes,
            topk,
            gates,
        }
    }
}

// ------------------------------------------------------------------
// State structures
// ------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GlobalEntityState {
    pub idx: usize,
    pub feasible: bool,
    pub u_mean: f64,
    pub u_var: f64,
    pub rank: Option<usize>,
    pub p_flip: f64,
}

impl GlobalEntityState {
    fn new(idx: usize) -> Self {
        Self {
            idx,
            feasible: true,
            u_mean: 0.0,
            u_var: 0.0,
            rank: None,
            p_flip: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlobalPlanProposal {
    pub attribute_id: String,
    pub i: usize,
    pub j: usize,
    pub global_score: f64,
    pub core_score: f64,
    pub delta_info: f64,
    pub delta_rank_risk: f64,
}

// ------------------------------------------------------------------
// Main manager
// ------------------------------------------------------------------

#[derive(Debug)]
pub struct TraitSearchManager {
    config: TraitSearchConfig,
    engines: HashMap<String, RatingEngine>,
    n: usize,

    scales: HashMap<String, f64>,
    z_scores: HashMap<String, Vec<f64>>,
    min_norm: HashMap<String, Vec<f64>>,
    percentiles: HashMap<String, Vec<f64>>,

    entities: Vec<GlobalEntityState>,
    sorted_indices: Vec<usize>,
    band_indices: Vec<usize>,
    boundary_index: Option<usize>,

    state_valid: bool,
    stop_streak: usize,
    has_degraded: bool,
}

impl TraitSearchManager {
    pub fn new(
        config: TraitSearchConfig,
        mut engines: HashMap<String, RatingEngine>,
    ) -> Result<Self> {
        if config.attributes.is_empty() {
            return Err(TraitSearchError::EmptyAttributes);
        }
        if config.n_entities == 0 {
            return Err(TraitSearchError::NonPositiveEntities);
        }

        let mut engine_map: HashMap<String, RatingEngine> = HashMap::new();
        let mut n_opt: Option<usize> = None;

        for attr in &config.attributes {
            let id = &attr.id;
            let engine = engines
                .remove(id)
                .ok_or_else(|| TraitSearchError::MissingEngine {
                    attribute_id: id.clone(),
                })?;

            let engine_n = engine.n;
            if let Some(n0) = n_opt {
                if engine_n != n0 {
                    return Err(TraitSearchError::EnginesSizeMismatch);
                }
            } else {
                n_opt = Some(engine_n);
            }
            engine_map.insert(id.clone(), engine);
        }

        let n = n_opt.ok_or(TraitSearchError::EnginesSizeMismatch)?;
        if n != config.n_entities {
            return Err(TraitSearchError::EntityCountMismatch {
                config_n: config.n_entities,
                engine_n: n,
            });
        }

        let entities = (0..n).map(GlobalEntityState::new).collect();

        Ok(Self {
            config,
            engines: engine_map,
            n,
            scales: HashMap::new(),
            z_scores: HashMap::new(),
            min_norm: HashMap::new(),
            percentiles: HashMap::new(),
            entities,
            sorted_indices: Vec::new(),
            band_indices: Vec::new(),
            boundary_index: None,
            state_valid: false,
            stop_streak: 0,
            has_degraded: false,
        })
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    pub fn recompute_global_state(&mut self) -> Result<()> {
        self.solve_attributes()?;
        self.combine_attributes()?;
        self.rank_entities();
        if !self.band_indices.is_empty() && self.band_indices.len() <= MAX_REFINED_ACTIVE {
            let active = self.band_indices.clone();
            self.refine_active_variances(&active);
            self.rank_entities();
        }
        self.state_valid = true;
        Ok(())
    }

    pub fn invalidate(&mut self) {
        self.state_valid = false;
    }

    pub fn estimate_topk_error(&self) -> f64 {
        if !self.state_valid {
            return f64::INFINITY;
        }
        if self.band_indices.is_empty() {
            return 0.0;
        }

        let beta = beta_from_tolerated_error(self.config.topk.tolerated_error);
        let (lcb, ucb, _feasible) = self.compute_bounds(beta);
        let (incumbents, challengers) = self.frontier_sets(&lcb, &ucb);

        if incumbents.is_empty() || challengers.is_empty() {
            return 0.0;
        }

        let mut err = 0.0;
        for &i in &incumbents {
            for &j in &challengers {
                let delta = self.entities[i].u_mean - self.entities[j].u_mean;
                let var = self.global_diff_var_safe(i, j);
                err += inversion_prob(delta, var);
            }
        }
        err
    }

    pub fn propose_batch(
        &mut self,
        rater_id: &str,
        batch_size: usize,
        planner_mode: rating_engine::PlannerMode,
    ) -> Result<Vec<GlobalPlanProposal>> {
        // Clamp batch_size to prevent resource exhaustion
        let batch_size = batch_size.min(MAX_BATCH_SIZE);
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        if !self.state_valid {
            self.recompute_global_state()?;
        }

        let band = &self.band_indices;
        if band.len() < 2 {
            return Ok(Vec::new());
        }

        let beta = beta_from_tolerated_error(self.config.topk.tolerated_error);
        let (lcb, ucb, _feasible) = self.compute_bounds(beta);
        let (incumbents, challengers) = self.frontier_sets(&lcb, &ucb);
        let critical_pair = self
            .critical_pair(&lcb, &ucb)
            .map(|(i, j)| if i < j { (i, j) } else { (j, i) });

        if incumbents.is_empty() || challengers.is_empty() {
            return Ok(Vec::new());
        }

        let mut band_candidates = self.build_frontier_candidates(&incumbents, &challengers);
        if let Some((i_star, j_star)) = critical_pair {
            // Connectivity guardrail: ensure boundary items have minimal degree.
            let min_degree = 2;
            let mut anchor = self.sorted_indices.first().copied();
            if anchor == Some(i_star) || anchor == Some(j_star) {
                anchor = self.sorted_indices.get(1).copied();
            }
            if let Some(anchor_idx) = anchor {
                for attr in &self.config.attributes {
                    if let Some(engine) = self.engines.get(&attr.id) {
                        if !engine.has_min_degree(i_star, min_degree) {
                            let (a, b) = if i_star < anchor_idx {
                                (i_star, anchor_idx)
                            } else {
                                (anchor_idx, i_star)
                            };
                            band_candidates.push((a, b));
                        }
                        if !engine.has_min_degree(j_star, min_degree) {
                            let (a, b) = if j_star < anchor_idx {
                                (j_star, anchor_idx)
                            } else {
                                (anchor_idx, j_star)
                            };
                            band_candidates.push((a, b));
                        }
                    }
                }
            }
        }
        if let Some((i, j)) = critical_pair {
            band_candidates.push((i, j));
        }
        band_candidates.sort_unstable();
        band_candidates.dedup();
        if band_candidates.is_empty() {
            return Ok(Vec::new());
        }

        let mut pair_stats: HashMap<(usize, usize), f64> = HashMap::new();
        let use_effective = self.config.topk.effective_resistance_max_active > 0
            && self.band_indices.len() <= self.config.topk.effective_resistance_max_active;

        for &(i, j) in &band_candidates {
            let delta_mu = self.entities[i].u_mean - self.entities[j].u_mean;
            let base_var = if use_effective && Some((i, j)) == critical_pair {
                self.global_diff_var_effective(i, j)
                    .unwrap_or_else(|| self.global_diff_var_diag(i, j))
            } else {
                self.global_diff_var_diag(i, j)
            };
            let p_before = inversion_prob(delta_mu, base_var);
            pair_stats.insert((i, j), p_before);
        }

        let mut proposals: Vec<GlobalPlanProposal> = Vec::new();
        let mut critical_best: Option<GlobalPlanProposal> = None;

        let mut candidates = band_candidates.clone();
        if let Some((i, j)) = critical_pair {
            candidates.retain(|&(a, b)| !(a == i && b == j));
            candidates.insert(0, (i, j));
        }
        if candidates.len() > MAX_PLANNER_CANDIDATES {
            candidates.truncate(MAX_PLANNER_CANDIDATES);
        }

        for attr in &self.config.attributes {
            let attr_id = &attr.id;
            let engine =
                self.engines
                    .get(attr_id)
                    .ok_or_else(|| TraitSearchError::InternalError {
                        message: "engine map invariant violated".to_string(),
                    })?;

            let scale = self
                .scales
                .get(attr_id)
                .copied()
                .unwrap_or(SCALE_FLOOR)
                .max(SCALE_FLOOR);
            let uncertainty_weight = match engine.diag_cov() {
                Some(diag) => {
                    let mut sum = 0.0;
                    let mut count = 0usize;
                    for &idx in band {
                        if idx < diag.len() {
                            sum += diag[idx].max(0.0);
                            count += 1;
                        }
                    }
                    if count == 0 {
                        1.0
                    } else {
                        let avg_var = sum / (count as f64);
                        let denom = avg_var + scale * scale;
                        if denom > 0.0 {
                            (avg_var / denom).clamp(MIN_ATTR_UNCERTAINTY_WEIGHT, 1.0)
                        } else {
                            1.0
                        }
                    }
                }
                None => 1.0,
            };
            let weight_factor = (attr.weight / scale).powi(2) * uncertainty_weight;

            let proposals_attr =
                plan_edges_for_rater(engine, &candidates, rater_id, planner_mode, use_effective)
                    .map_err(|e| TraitSearchError::PlannerError {
                        message: e.to_string(),
                    })?;

            for PlanProposal {
                i,
                j,
                score,
                delta_info,
                delta_rank_risk,
                cost: _,
            } in proposals_attr
            {
                let (a, b) = if i <= j { (i, j) } else { (j, i) };
                let p_before = match pair_stats.get(&(a, b)) {
                    Some(v) => *v,
                    None => continue,
                };
                if p_before < MIN_PAIR_PROB {
                    continue;
                }

                let membership_weight = 0.5 * (self.entities[a].p_flip + self.entities[b].p_flip);
                let membership_weight = membership_weight.clamp(MIN_MEMBERSHIP_WEIGHT, 1.0);

                let weighted_score = weight_factor * score * p_before * membership_weight;
                if weighted_score <= 0.0 {
                    continue;
                }

                let proposal = GlobalPlanProposal {
                    attribute_id: attr_id.clone(),
                    i,
                    j,
                    global_score: weighted_score,
                    core_score: weight_factor * delta_rank_risk,
                    delta_info: weight_factor * delta_info,
                    delta_rank_risk: weight_factor * delta_rank_risk,
                };

                if Some((a, b)) == critical_pair {
                    let replace = match &critical_best {
                        Some(best) => proposal.global_score > best.global_score,
                        None => true,
                    };
                    if replace {
                        critical_best = Some(proposal.clone());
                    }
                }

                proposals.push(proposal);
            }
        }

        if proposals.is_empty() {
            if band_candidates.is_empty() {
                return Ok(Vec::new());
            }
            let mut attr_iter = self.config.attributes.iter().cycle();
            for &(i, j) in band_candidates.iter().take(batch_size) {
                let attr = attr_iter.next().ok_or(TraitSearchError::EmptyAttributes)?;
                proposals.push(GlobalPlanProposal {
                    attribute_id: attr.id.clone(),
                    i,
                    j,
                    global_score: 0.0,
                    core_score: 0.0,
                    delta_info: 0.0,
                    delta_rank_risk: 0.0,
                });
            }
        }

        proposals.sort_by(|a, b| {
            b.global_score
                .partial_cmp(&a.global_score)
                .unwrap_or(Ordering::Equal)
        });

        let mut deduped: Vec<GlobalPlanProposal> = Vec::with_capacity(batch_size);
        let mut seen: HashSet<(String, usize, usize)> = HashSet::new();

        if let Some(best) = critical_best {
            let (a, b) = if best.i <= best.j {
                (best.i, best.j)
            } else {
                (best.j, best.i)
            };
            let key = (best.attribute_id.clone(), a, b);
            if seen.insert(key) {
                deduped.push(best);
            }
        }

        for proposal in proposals.into_iter() {
            let (a, b) = if proposal.i <= proposal.j {
                (proposal.i, proposal.j)
            } else {
                (proposal.j, proposal.i)
            };
            let key = (proposal.attribute_id.clone(), a, b);

            if seen.insert(key) {
                deduped.push(proposal);
                if deduped.len() >= batch_size {
                    break;
                }
            }
        }

        Ok(deduped)
    }

    pub fn ranked_indices(&self) -> Vec<usize> {
        self.sorted_indices.clone()
    }

    pub fn entity_state(&self, idx: usize) -> &GlobalEntityState {
        &self.entities[idx]
    }

    pub fn attribute_scores(&self, attr_id: &str) -> Option<&[f64]> {
        self.engines.get(attr_id).and_then(|engine| engine.scores())
    }

    pub fn attribute_std(&self, attr_id: &str) -> Option<Vec<f64>> {
        self.engines
            .get(attr_id)
            .and_then(|engine| engine.diag_cov())
            .map(|diag| {
                diag.iter()
                    .map(|&v| v.max(0.0).sqrt())
                    .collect::<Vec<f64>>()
            })
    }

    pub fn attribute_z_scores(&self, attr_id: &str) -> Option<&[f64]> {
        self.z_scores.get(attr_id).map(|v| v.as_slice())
    }

    pub fn attribute_min_norm(&self, attr_id: &str) -> Option<&[f64]> {
        self.min_norm.get(attr_id).map(|v| v.as_slice())
    }

    pub fn attribute_percentiles(&self, attr_id: &str) -> Option<&[f64]> {
        self.percentiles.get(attr_id).map(|v| v.as_slice())
    }

    /// Ensure derived units (z, min_norm, percentiles) are computed for an attribute.
    /// These are only needed for gate evaluation and response payloads, so compute lazily.
    pub fn ensure_attribute_units(&mut self, attr_id: &str) -> Result<()> {
        if self.z_scores.contains_key(attr_id) {
            return Ok(());
        }
        let scores = self
            .engines
            .get(attr_id)
            .and_then(|engine| engine.scores())
            .ok_or_else(|| TraitSearchError::InternalError {
                message: "scores not available; call solve() first".to_string(),
            })?;
        let (scale, z, min_norm, pct) = compute_attribute_units(scores);
        self.scales.insert(attr_id.to_string(), scale);
        self.z_scores.insert(attr_id.to_string(), z);
        self.min_norm.insert(attr_id.to_string(), min_norm);
        self.percentiles.insert(attr_id.to_string(), pct);
        Ok(())
    }

    /// Ensure derived units are computed for all attributes (used at response time).
    pub fn ensure_all_attribute_units(&mut self) -> Result<()> {
        let attr_ids: Vec<String> = self
            .config
            .attributes
            .iter()
            .map(|a| a.id.clone())
            .collect();
        for attr_id in attr_ids {
            self.ensure_attribute_units(&attr_id)?;
        }
        Ok(())
    }

    /// Add observations for a specific attribute and invalidate cached state.
    pub fn add_observations(&mut self, attr_id: &str, observations: &[Observation]) -> Result<()> {
        if observations.is_empty() {
            return Ok(());
        }
        let engine =
            self.engines
                .get_mut(attr_id)
                .ok_or_else(|| TraitSearchError::MissingEngine {
                    attribute_id: attr_id.to_string(),
                })?;
        engine.add_observations(observations);
        self.invalidate();
        Ok(())
    }

    /// Add a single observation (convenience wrapper).
    pub fn add_observation(&mut self, attr_id: &str, observation: Observation) -> Result<()> {
        let observations = [observation];
        self.add_observations(attr_id, &observations)
    }

    // ------------------------------------------------------------------
    // Internal logic
    // ------------------------------------------------------------------

    fn solve_attributes(&mut self) -> Result<()> {
        self.scales.clear();
        self.z_scores.clear();
        self.min_norm.clear();
        self.percentiles.clear();
        self.has_degraded = false;

        let n = self.n;
        let mut units_needed: HashSet<&str> = HashSet::new();
        for gate in &self.config.gates {
            if gate.unit != "latent" {
                units_needed.insert(gate.attribute_id.as_str());
            }
        }

        for attr in &self.config.attributes {
            let id = &attr.id;
            let engine =
                self.engines
                    .get_mut(id)
                    .ok_or_else(|| TraitSearchError::InternalError {
                        message: "engine map invariant violated".to_string(),
                    })?;
            let summary = engine.solve();
            if summary.degraded {
                self.has_degraded = true;
            }

            if summary.scores.len() != n || summary.diag_cov.len() != n {
                return Err(TraitSearchError::PosteriorLengthMismatch {
                    attribute_id: id.clone(),
                    scores_len: summary.scores.len(),
                    diag_cov_len: summary.diag_cov.len(),
                    expected_n: n,
                });
            }

            let needs_units = units_needed.contains(id.as_str());
            let (scale, z, min_norm, pct) = if needs_units {
                compute_attribute_units(&summary.scores)
            } else {
                (
                    compute_attribute_scale(&summary.scores),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            };

            self.scales.insert(id.clone(), scale);
            if needs_units {
                self.z_scores.insert(id.clone(), z);
                self.min_norm.insert(id.clone(), min_norm);
                self.percentiles.insert(id.clone(), pct);
            }
        }

        Ok(())
    }

    fn combine_attributes(&mut self) -> Result<()> {
        let n = self.n;
        let mut u_mean = vec![0.0; n];
        let mut u_var = vec![0.0; n];

        for attr in &self.config.attributes {
            let engine =
                self.engines
                    .get(&attr.id)
                    .ok_or_else(|| TraitSearchError::InternalError {
                        message: "engine map invariant violated".to_string(),
                    })?;
            let scores = engine
                .scores()
                .ok_or_else(|| TraitSearchError::InternalError {
                    message: "scores not available; call solve() first".to_string(),
                })?;
            let diag_cov = engine
                .diag_cov()
                .ok_or_else(|| TraitSearchError::InternalError {
                    message: "diag_cov not available; call solve() first".to_string(),
                })?;
            let scale =
                *self
                    .scales
                    .get(&attr.id)
                    .ok_or_else(|| TraitSearchError::InternalError {
                        message: "scales map invariant violated".to_string(),
                    })?;
            let w = attr.weight;

            let inv_scale = 1.0 / scale;
            let inv_scale2 = inv_scale * inv_scale;
            let w2 = w * w;

            for i in 0..n {
                u_mean[i] += w * (scores[i] * inv_scale);
                u_var[i] += w2 * (diag_cov[i].max(0.0) * inv_scale2);
            }
        }

        let mut feasible_mask = vec![true; n];

        let gates = self.config.gates.clone();
        for gate in gates {
            let scores = self
                .engines
                .get(&gate.attribute_id)
                .and_then(|engine| engine.scores())
                .ok_or_else(|| TraitSearchError::GateUnknownAttribute {
                    attribute_id: gate.attribute_id.clone(),
                })?;

            let gate_vals: &[f64] = match gate.unit.as_str() {
                "latent" => scores,
                "z" => {
                    self.ensure_attribute_units(&gate.attribute_id)?;
                    self.z_scores.get(&gate.attribute_id).ok_or_else(|| {
                        TraitSearchError::InternalError {
                            message: "z_scores map invariant violated".to_string(),
                        }
                    })?
                }
                "percentile" => {
                    self.ensure_attribute_units(&gate.attribute_id)?;
                    self.percentiles.get(&gate.attribute_id).ok_or_else(|| {
                        TraitSearchError::InternalError {
                            message: "percentiles map invariant violated".to_string(),
                        }
                    })?
                }
                "min_norm" => {
                    self.ensure_attribute_units(&gate.attribute_id)?;
                    self.min_norm.get(&gate.attribute_id).ok_or_else(|| {
                        TraitSearchError::InternalError {
                            message: "min_norm map invariant violated".to_string(),
                        }
                    })?
                }
                _ => {
                    return Err(TraitSearchError::UnsupportedGateUnit {
                        unit: gate.unit.clone(),
                    })
                }
            };

            match gate.op.as_str() {
                ">=" => {
                    for i in 0..n {
                        feasible_mask[i] &= gate_vals[i] >= gate.threshold;
                    }
                }
                "<=" => {
                    for i in 0..n {
                        feasible_mask[i] &= gate_vals[i] <= gate.threshold;
                    }
                }
                _ => {
                    return Err(TraitSearchError::UnsupportedGateOp {
                        op: gate.op.clone(),
                    })
                }
            }
        }

        for idx in 0..n {
            let feasible = feasible_mask[idx];
            let state = &mut self.entities[idx];
            state.feasible = feasible;
            if feasible {
                state.u_mean = u_mean[idx];
                state.u_var = u_var[idx].max(0.0);
            } else {
                state.u_mean = f64::NEG_INFINITY;
                state.u_var = f64::INFINITY;
            }
            state.rank = None;
            state.p_flip = 0.0;
        }

        Ok(())
    }

    fn compute_bounds(&self, beta: f64) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let mut lcb = vec![f64::NEG_INFINITY; self.n];
        let mut ucb = vec![f64::NEG_INFINITY; self.n];
        let mut feasible = Vec::new();

        for (idx, state) in self.entities.iter().enumerate() {
            if !state.feasible {
                continue;
            }
            feasible.push(idx);
            let var = state.u_var.max(0.0);
            let std = var.sqrt();
            lcb[idx] = state.u_mean - beta * std;
            ucb[idx] = state.u_mean + beta * std;
        }

        (lcb, ucb, feasible)
    }

    fn critical_pair(&self, lcb: &[f64], ucb: &[f64]) -> Option<(usize, usize)> {
        let k = self.config.topk.k.max(1);
        let topk: Vec<usize> = self.sorted_indices.iter().copied().take(k).collect();
        if topk.is_empty() {
            return None;
        }

        let mut i_star = topk[0];
        let mut l_min = lcb[i_star];
        for &idx in &topk {
            if lcb[idx] < l_min {
                l_min = lcb[idx];
                i_star = idx;
            }
        }

        let topk_set: HashSet<usize> = topk.iter().copied().collect();
        let mut j_star: Option<usize> = None;
        let mut u_max = f64::NEG_INFINITY;
        for &idx in &self.sorted_indices {
            if topk_set.contains(&idx) {
                continue;
            }
            let u = ucb[idx];
            if u > u_max {
                u_max = u;
                j_star = Some(idx);
            }
        }

        j_star.map(|j| (i_star, j))
    }

    fn frontier_sets(&self, lcb: &[f64], ucb: &[f64]) -> (Vec<usize>, Vec<usize>) {
        let k = self.config.topk.k.max(1);
        let frontier_width = self.config.topk.band_size.max(1);

        let topk: Vec<usize> = self.sorted_indices.iter().copied().take(k).collect();
        if topk.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let topk_set: HashSet<usize> = topk.iter().copied().collect();
        let band_set: HashSet<usize> = self.band_indices.iter().copied().collect();

        let mut incumbents: Vec<usize> = topk
            .iter()
            .copied()
            .filter(|idx| band_set.contains(idx))
            .collect();
        incumbents.sort_by(|&a, &b| lcb[a].partial_cmp(&lcb[b]).unwrap_or(Ordering::Equal));
        incumbents.truncate(frontier_width.min(k));

        let mut challengers: Vec<usize> = self
            .band_indices
            .iter()
            .copied()
            .filter(|idx| !topk_set.contains(idx))
            .collect();
        challengers.sort_by(|&a, &b| ucb[b].partial_cmp(&ucb[a]).unwrap_or(Ordering::Equal));
        challengers.truncate(frontier_width);

        (incumbents, challengers)
    }

    fn build_frontier_candidates(
        &self,
        incumbents: &[usize],
        challengers: &[usize],
    ) -> Vec<(usize, usize)> {
        if incumbents.is_empty() || challengers.is_empty() {
            return Vec::new();
        }

        let mut candidates = Vec::with_capacity(incumbents.len() * challengers.len());
        for &i in incumbents {
            for &j in challengers {
                if i == j {
                    continue;
                }
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                candidates.push((a, b));
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    fn global_diff_var_safe(&self, i: usize, j: usize) -> f64 {
        let mut var = 0.0;
        for attr in &self.config.attributes {
            let attr_id = &attr.id;
            let engine = match self.engines.get(attr_id) {
                Some(e) => e,
                None => continue,
            };
            let diag = match engine.diag_cov() {
                Some(v) => v,
                None => continue,
            };
            let scale = self
                .scales
                .get(attr_id)
                .copied()
                .unwrap_or(SCALE_FLOOR)
                .max(SCALE_FLOOR);
            let w = attr.weight;
            let sigma_i = diag[i].max(0.0).sqrt();
            let sigma_j = diag[j].max(0.0).sqrt();
            let diff_var = (sigma_i + sigma_j) * (sigma_i + sigma_j);
            var += (w / scale).powi(2) * diff_var;
        }
        var
    }

    fn global_diff_var_diag(&self, i: usize, j: usize) -> f64 {
        let mut var = 0.0;
        for attr in &self.config.attributes {
            let attr_id = &attr.id;
            let engine = match self.engines.get(attr_id) {
                Some(e) => e,
                None => continue,
            };
            let diag = match engine.diag_cov() {
                Some(v) => v,
                None => continue,
            };
            let scale = self
                .scales
                .get(attr_id)
                .copied()
                .unwrap_or(SCALE_FLOOR)
                .max(SCALE_FLOOR);
            let w = attr.weight;
            let diff_var = (diag[i].max(0.0) + diag[j].max(0.0)).max(0.0);
            var += (w / scale).powi(2) * diff_var;
        }
        var
    }

    fn global_diff_var_effective(&self, i: usize, j: usize) -> Option<f64> {
        let mut var = 0.0;
        let mut seen = false;
        for attr in &self.config.attributes {
            let attr_id = &attr.id;
            let engine = self.engines.get(attr_id)?;
            let scale = self
                .scales
                .get(attr_id)
                .copied()
                .unwrap_or(SCALE_FLOOR)
                .max(SCALE_FLOOR);
            let w = attr.weight;

            if let Some(diff_var) = engine.diff_var_for(i, j) {
                var += (w / scale).powi(2) * diff_var.max(0.0);
                seen = true;
            }
        }
        if seen {
            Some(var)
        } else {
            None
        }
    }

    fn refine_active_variances(&mut self, active: &[usize]) {
        if active.is_empty() {
            return;
        }

        let mut refined = vec![0.0; active.len()];

        for attr in &self.config.attributes {
            let attr_id = &attr.id;
            let engine = match self.engines.get(attr_id) {
                Some(e) => e,
                None => continue,
            };
            let diag = match engine.diag_cov() {
                Some(v) => v,
                None => continue,
            };
            let scale = self
                .scales
                .get(attr_id)
                .copied()
                .unwrap_or(SCALE_FLOOR)
                .max(SCALE_FLOOR);
            let weight_factor = (attr.weight / scale).powi(2);

            let vars = engine
                .marginal_vars_for(active)
                .unwrap_or_else(|| active.iter().map(|&idx| diag[idx].max(0.0)).collect());

            for (pos, v) in vars.iter().enumerate() {
                refined[pos] += weight_factor * v.max(0.0);
            }
        }

        for (pos, &idx) in active.iter().enumerate() {
            self.entities[idx].u_var = refined[pos].max(0.0);
        }
    }

    pub fn certified_stop(&mut self) -> bool {
        if !self.state_valid {
            self.stop_streak = 0;
            return false;
        }
        if self.has_degraded {
            self.stop_streak = 0;
            return false;
        }
        // Certification assumes critical items are well-anchored in the comparison graph
        // (non-trivial degree and shared connectivity), avoiding premature stops on isolated items.
        let beta = beta_from_tolerated_error(self.config.topk.tolerated_error)
            * self.config.topk.stop_sigma_inflate.max(1.0);
        let (lcb, ucb, _feasible) = self.compute_bounds(beta);
        let (i_star, j_star) = match self.critical_pair(&lcb, &ucb) {
            Some(pair) => pair,
            None => {
                self.stop_streak = 0;
                return false;
            }
        };

        let min_degree = 2;
        let anchor_idx = self.sorted_indices.first().copied();
        for attr in &self.config.attributes {
            let engine = match self.engines.get(&attr.id) {
                Some(e) => e,
                None => {
                    self.stop_streak = 0;
                    return false;
                }
            };
            if !engine.has_min_degree(i_star, min_degree) {
                self.stop_streak = 0;
                return false;
            }
            if !engine.has_min_degree(j_star, min_degree) {
                self.stop_streak = 0;
                return false;
            }
            if let Some(anchor) = anchor_idx {
                if !engine.same_component(i_star, anchor) || !engine.same_component(j_star, anchor)
                {
                    self.stop_streak = 0;
                    return false;
                }
            }
        }

        let mut certified = lcb[i_star] > ucb[j_star];
        if certified {
            // Pre-stop verification on the critical pair using stronger variance.
            if let Some(var_eff) = self.global_diff_var_effective(i_star, j_star) {
                let delta = self.entities[i_star].u_mean - self.entities[j_star].u_mean;
                let margin = delta - beta * var_eff.max(0.0).sqrt();
                certified = margin > 0.0;
            }
        }

        if certified {
            self.stop_streak = self.stop_streak.saturating_add(1);
        } else {
            self.stop_streak = 0;
        }

        self.stop_streak >= self.config.topk.stop_min_consecutive.max(1)
    }

    fn rank_entities(&mut self) {
        let mut feasible_indices: Vec<usize> = self
            .entities
            .iter()
            .filter(|s| s.feasible)
            .map(|s| s.idx)
            .collect();

        if feasible_indices.is_empty() {
            self.sorted_indices.clear();
            self.band_indices.clear();
            self.boundary_index = None;
            return;
        }

        feasible_indices.sort_by(|&a, &b| {
            let ua = self.entities[a].u_mean;
            let ub = self.entities[b].u_mean;
            ub.partial_cmp(&ua).unwrap_or(Ordering::Equal)
        });
        self.sorted_indices = feasible_indices.clone();

        for (rank, idx) in feasible_indices.iter().enumerate() {
            self.entities[*idx].rank = Some(rank + 1);
        }

        let k = self.config.topk.k.max(1);
        if feasible_indices.len() <= k {
            self.boundary_index = feasible_indices.last().copied();
            self.band_indices.clear();
            for &idx in &feasible_indices {
                self.entities[idx].p_flip = 0.0;
            }
            return;
        }

        let boundary_idx = feasible_indices[k - 1];
        self.boundary_index = Some(boundary_idx);

        let boundary_mean = self.entities[boundary_idx].u_mean;
        let beta = beta_from_tolerated_error(self.config.topk.tolerated_error);
        let (lcb, ucb, _feasible) = self.compute_bounds(beta);

        let mut lcb_vals: Vec<f64> = feasible_indices.iter().map(|&idx| lcb[idx]).collect();
        lcb_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let theta_l = lcb_vals[k - 1];

        let mut ucb_vals: Vec<f64> = feasible_indices.iter().map(|&idx| ucb[idx]).collect();
        ucb_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let theta_u = ucb_vals[k];

        self.band_indices = feasible_indices
            .iter()
            .copied()
            .filter(|&idx| ucb[idx] >= theta_l && lcb[idx] <= theta_u)
            .collect();

        for &idx in &feasible_indices {
            let delta_mu = self.entities[idx].u_mean - boundary_mean;
            let delta_var = self.global_diff_var_safe(idx, boundary_idx);
            self.entities[idx].p_flip = inversion_prob(delta_mu, delta_var);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_attribute_units_basic() {
        let scores = vec![10.0, 20.0, 30.0];
        let (scale, z, min_norm, pct) = compute_attribute_units(&scores);

        assert!((scale - 10.0).abs() < 1e-9);
        assert_eq!(min_norm.len(), 3);
        assert_eq!(pct.len(), 3);
        assert_eq!(z.len(), 3);

        assert!((min_norm[0] - 1.0).abs() < 1e-9);
        assert!((min_norm[1] - 11.0).abs() < 1e-9);
        assert!((min_norm[2] - 21.0).abs() < 1e-9);

        assert!((pct[0] - (0.5 / 3.0)).abs() < 1e-9);
        assert!((pct[1] - (1.5 / 3.0)).abs() < 1e-9);
        assert!((pct[2] - (2.5 / 3.0)).abs() < 1e-9);

        // Median is 20, MAD is 10 -> sigma = 14.826 -> z = +/- 0.67449...
        assert!((z[1] - 0.0).abs() < 1e-12);
        assert!((z[0] + (10.0 / (10.0 * MAD_TO_SIGMA))).abs() < 1e-9);
        assert!((z[2] - (10.0 / (10.0 * MAD_TO_SIGMA))).abs() < 1e-9);
    }

    #[test]
    fn test_compute_attribute_units_all_equal() {
        let scores = vec![5.0, 5.0, 5.0];
        let (scale, z, min_norm, pct) = compute_attribute_units(&scores);

        assert!((scale - SCALE_FLOOR).abs() < 1e-12);
        assert_eq!(z, vec![0.0, 0.0, 0.0]);
        assert_eq!(min_norm, vec![1.0, 1.0, 1.0]);
        let mut pct_sorted = pct.clone();
        pct_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        assert_eq!(pct_sorted, vec![1.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0]);
    }

    #[test]
    fn test_compute_attribute_units_degenerate_mad() {
        // When >= 50% of values tie at the median, MAD can be zero even with non-trivial spread.
        // We should avoid exploding z-scores/weight normalization in this case.
        let scores = vec![0.0, 0.0, 0.097];
        let (scale, z, _min_norm, _pct) = compute_attribute_units(&scores);
        assert!(scale > 1e-3);
        assert!(z.iter().all(|v| v.is_finite()));
        assert!(z[2].abs() < 100.0);
    }
}
