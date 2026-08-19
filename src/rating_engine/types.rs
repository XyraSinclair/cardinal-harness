use super::*;

// ---------------------------------------------------------------------
//  Config
// ---------------------------------------------------------------------

/// Configuration for the rating engine (IRLS solver + planner).
///
/// See `docs/ALGORITHM.md` for rationale behind these defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    // -- Robust IRLS (Huber loss) --------------------------------------------
    // Huber loss downweights outlier comparisons where the LLM was inconsistent.
    /// Huber loss threshold: residuals beyond k standard deviations are downweighted.
    /// 1.5 is the standard choice — aggressive enough to suppress outliers,
    /// mild enough to not discard borderline observations.
    pub huber_k: f64,
    /// Maximum IRLS iterations. 12 is usually more than enough for convergence.
    pub irls_max_iters: usize,
    /// IRLS convergence tolerance (relative change in scores between iterations).
    pub irls_tol: f64,

    // -- Numerical stability -------------------------------------------------
    /// Tikhonov regularization for the Hessian. Prevents singular matrices when
    /// the comparison graph is sparse. Should be negligibly small (1e-9).
    pub ridge_lambda: f64,
    /// Epsilon to prevent division by zero in weight calculations.
    pub tiny: f64,
    /// Cap on log-ratio observations. ln(26) ≈ 3.26, so 10.0 is very permissive.
    pub max_log_ratio: f64,

    // -- Variance estimation -------------------------------------------------
    /// Number of Hutchinson random probes for estimating diag(H^{-1}) when the
    /// matrix is too large for exact Cholesky inversion (>256 items).
    pub hutch_probes: usize,

    // -- Rank-weighted planning ----------------------------------------------
    // Controls how the planner prioritizes comparisons near the top of the ranking.
    /// Rank weighting exponent: w(pos) = 1/(pos+1)^a. Higher values focus more
    /// comparisons on the very top of the ranking.
    pub rank_weight_exponent: f64,

    /// Window around each rank position to consider for gap-closing comparisons.
    pub rank_band_window: usize,
    /// Score gaps smaller than this are considered "small" and targeted for resolution.
    pub small_gap_threshold: f64,
    /// Safety cap on candidate pairs to prevent unbounded planner iteration.
    pub max_rank_pairs: Option<usize>,

    // -- Top-K focus ---------------------------------------------------------
    /// If set, focus planning on identifying the top-k items specifically.
    pub top_k: Option<usize>,
    /// Weight given to items outside the top-k band (0.0 = ignore tail entirely).
    pub tail_weight: f64,

    // -- Planner blending ----------------------------------------------------
    /// Blend factor between information-gain and rank-risk objectives.
    /// 1.0 = pure rank-risk, 0.0 = pure information gain.
    pub lambda_risk: f64,

    /// RNG seed for reproducible planner tie-breaking and Hutchinson probes.
    pub rng_seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            huber_k: 1.5,
            irls_max_iters: 12,
            irls_tol: 1e-8,
            ridge_lambda: 1e-9,
            tiny: 1e-18,
            max_log_ratio: 10.0,
            hutch_probes: 12,
            rank_weight_exponent: 1.0,
            rank_band_window: 5,
            small_gap_threshold: 0.5,
            max_rank_pairs: Some(200_000),
            top_k: None,
            tail_weight: 0.0,
            lambda_risk: 1.0,
            rng_seed: 1337,
        }
    }
}

// ---------------------------------------------------------------------
//  Data model
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributeParams {
    /// Global noise / difficulty parameter T.
    pub temperature: f64,
}

impl Default for AttributeParams {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RaterParams {
    /// Efficacy / effective sample size (β).
    pub beta: f64,
    /// Cost used by planner.
    pub cost_per_edge: f64,
}

impl Default for RaterParams {
    fn default() -> Self {
        Self {
            beta: 1.0,
            cost_per_edge: 1.0,
        }
    }
}

/// Complete, content-addressed constructor input for a [`RatingEngine`].
///
/// The spec contains configuration, not observations. A trace row binds its
/// exact [`Observation`] to this spec's identity; together they are sufficient
/// to replay what entered the solver without baking policy into the evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineSpec {
    pub n: usize,
    pub attribute: AttributeParams,
    pub raters: Vec<(String, RaterParams)>,
    pub config: Config,
}

impl EngineSpec {
    /// Canonical binary encoding for content identity.
    ///
    /// Strings are length-prefixed, integers are normalized to little-endian
    /// `u64`, floats use their exact IEEE-754 bit patterns, and raters are
    /// sorted by identifier. JSON never participates in the identity.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        fn put_usize(out: &mut Vec<u8>, value: usize) {
            out.extend_from_slice(&(value as u64).to_le_bytes());
        }
        fn put_f64(out: &mut Vec<u8>, value: f64) {
            out.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        fn put_str(out: &mut Vec<u8>, value: &str) {
            put_usize(out, value.len());
            out.extend_from_slice(value.as_bytes());
        }
        fn put_optional_usize(out: &mut Vec<u8>, value: Option<usize>) {
            match value {
                Some(value) => {
                    out.push(1);
                    put_usize(out, value);
                }
                None => out.push(0),
            }
        }

        let AttributeParams { temperature } = &self.attribute;
        let mut raters: Vec<_> = self.raters.iter().collect();
        raters.sort_by(|left, right| left.0.cmp(&right.0));
        let Config {
            huber_k,
            irls_max_iters,
            irls_tol,
            ridge_lambda,
            tiny,
            max_log_ratio,
            hutch_probes,
            rank_weight_exponent,
            rank_band_window,
            small_gap_threshold,
            max_rank_pairs,
            top_k,
            tail_weight,
            lambda_risk,
            rng_seed,
        } = &self.config;

        let mut out = Vec::new();
        put_usize(&mut out, self.n);
        put_f64(&mut out, *temperature);
        put_usize(&mut out, raters.len());
        for (id, params) in raters {
            let RaterParams {
                beta,
                cost_per_edge,
            } = params;
            put_str(&mut out, id);
            put_f64(&mut out, *beta);
            put_f64(&mut out, *cost_per_edge);
        }

        put_f64(&mut out, *huber_k);
        put_usize(&mut out, *irls_max_iters);
        put_f64(&mut out, *irls_tol);
        put_f64(&mut out, *ridge_lambda);
        put_f64(&mut out, *tiny);
        put_f64(&mut out, *max_log_ratio);
        put_usize(&mut out, *hutch_probes);
        put_f64(&mut out, *rank_weight_exponent);
        put_usize(&mut out, *rank_band_window);
        put_f64(&mut out, *small_gap_threshold);
        put_optional_usize(&mut out, *max_rank_pairs);
        put_optional_usize(&mut out, *top_k);
        put_f64(&mut out, *tail_weight);
        put_f64(&mut out, *lambda_risk);
        out.extend_from_slice(&rng_seed.to_le_bytes());
        out
    }

    /// Content identity of the complete engine constructor input.
    #[must_use]
    pub fn id(&self) -> ContentId {
        ContentId::derive(ENGINE_SPEC_DOMAIN, &self.canonical_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub i: usize,
    pub j: usize,
    pub ratio: f64,
    pub confidence: f64,
    pub rater_id: String,
    pub reps: f64,
    /// Explicit information weight (1/variance in log-ratio space) for this
    /// observation. Point observations without this field receive unit
    /// precision. This is the channel by which PMF-derived evidence (for
    /// example answer-token logprob distributions via seriate) enters the
    /// solver with measured variance rather than uncalibrated self-assessment.
    /// Rater reliability (beta) and attribute temperature still scale it;
    /// they are orthogonal to per-judgement precision.
    pub precision: Option<f64>,
}

impl Observation {
    pub fn new(
        i: usize,
        j: usize,
        ratio: f64,
        confidence: f64,
        rater_id: impl Into<String>,
        reps: f64,
    ) -> Self {
        Self {
            i,
            j,
            ratio,
            confidence,
            rater_id: rater_id.into(),
            reps,
            precision: None,
        }
    }

    /// Build an observation from log-ratio moments (mean, variance), as
    /// produced by a judgement PMF. `log_ratio_mean` is signed relative to
    /// `(i, j)`: positive means `i` has more of the attribute. `variance`
    /// is floored by the caller; precision = 1/variance.
    pub fn from_log_ratio_moments(
        i: usize,
        j: usize,
        log_ratio_mean: f64,
        variance: f64,
        rater_id: impl Into<String>,
        reps: f64,
    ) -> Self {
        Self {
            i,
            j,
            ratio: log_ratio_mean.exp(),
            // Unused when precision is set; kept sane for display paths.
            confidence: 1.0,
            rater_id: rater_id.into(),
            reps,
            precision: Some(1.0 / variance.max(f64::MIN_POSITIVE)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub i: usize,
    pub j: usize,
    pub mu: f64,
    pub lam: f64,
    pub contributors: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct CalibrationEvidence {
    pub global_variance_obs: f64,
    pub global_mad_obs: f64,
    pub inferred_temperature: f64,
    pub rater_efficacy_obs: HashMap<String, f64>,
    pub rater_bias_obs: HashMap<String, f64>,
    pub rater_scatter_obs: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct SolveSummary {
    pub scores: Vec<f64>,
    pub residuals: Vec<f64>,
    pub diag_cov: Vec<f64>,
    pub hcr: f64,
    /// The cyclic energy split into triangle-auditable curl and
    /// triad-invisible harmonic components (see [`HodgeSplit`]).
    /// Invariant: `hodge.local_curl_frac + hodge.harmonic_frac ≈ hcr`.
    pub hodge: HodgeSplit,
    /// Fiedler value + Foster's-theorem check (None above the dense-eigen
    /// size cap). See [`SpectralDiagnostics`].
    pub spectral: Option<SpectralDiagnostics>,
    /// Leave-one-out consistency: each judgement vs the rest of the
    /// graph, correctly studentized. See [`LooDiagnostics`].
    pub loo: Option<LooDiagnostics>,
    pub pcr: f64,
    pub total_info: f64,
    pub expected_rank_reversals: f64,
    pub max_pair_reversal_prob: f64,
    pub rank_risk: f64,
    pub components: usize,
    pub cycle_dim: usize,
    pub calibration_evidence: CalibrationEvidence,
    pub degraded: bool,
}

#[derive(Debug, Clone)]
pub struct PlanProposal {
    pub i: usize,
    pub j: usize,
    pub score: f64,
    pub delta_info: f64,
    pub delta_rank_risk: f64,
    pub cost: f64,
}
