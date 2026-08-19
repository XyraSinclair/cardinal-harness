// ------------------------------------------------------------------
// Configuration & errors
// ------------------------------------------------------------------

use super::*;

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
    pub min_explore_degree: usize,
    /// When set, stop spending forced-exploration comparisons on entities
    /// that already have at least one observation, sit below the top-k
    /// boundary, and whose probability of crossing it (`p_flip`) is under
    /// this threshold. The exploitation planner is already band-focused;
    /// this trims the exploration tail for entities the posterior says can
    /// be let go of. Pruned entities keep their scores and can re-enter if
    /// later evidence moves them back into the band.
    pub prune_p_topk_below: Option<f64>,
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
            min_explore_degree: 2,
            prune_p_topk_below: None,
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
    pub(super) fn new(idx: usize) -> Self {
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
    pub(super) config: TraitSearchConfig,
    pub(super) engines: HashMap<String, RatingEngine>,
    pub(super) n: usize,

    pub(super) scales: HashMap<String, f64>,
    pub(super) z_scores: HashMap<String, Vec<f64>>,
    pub(super) min_norm: HashMap<String, Vec<f64>>,
    pub(super) percentiles: HashMap<String, Vec<f64>>,

    pub(super) entities: Vec<GlobalEntityState>,
    pub(super) sorted_indices: Vec<usize>,
    pub(super) band_indices: Vec<usize>,
    pub(super) boundary_index: Option<usize>,

    pub(super) state_valid: bool,
    pub(super) stop_streak: usize,
    pub(super) has_degraded: bool,
    /// Per-attribute curl fraction from the latest solve: the share of
    /// judgement energy that is cyclically inconsistent (see
    /// `rating_engine::compute_hcr`).
    pub(super) frustration: HashMap<String, f64>,
    /// Entities excluded from further forced exploration by the
    /// `prune_p_topk_below` rule.
    pub(super) explore_pruned: HashSet<usize>,
}
