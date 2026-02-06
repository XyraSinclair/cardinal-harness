//! Rating Performance Core — robust ranking, diagnostics, and planning.
//!
//! Rust port of the Python `rating_engine.py` with the same public API:
//! - Config, AttributeParams, RaterParams, Observation, Edge
//! - CalibrationEvidence, SolveSummary, PlanProposal
//! - RatingEngine, plan_edges_for_rater, solve_once
//!
//! Implementation notes:
//! - Uses dense `nalgebra::DMatrix` + Cholesky instead of SciPy sparse solvers.
//! - IRLS stopping rule is fixed to avoid the `inf`-always-converges quirk
//!   in the original Python (so you actually get multiple robust iterations).
//! - Gauge pinning and rank / planner logic match the Python semantics.

use std::cmp::Ordering;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::f64::consts::SQRT_2;
use std::hash::{Hash, Hasher};

use nalgebra::linalg::{Cholesky, SymmetricEigen};
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use statrs::function::erf::erf;

/// Maximum items to prevent O(n²) memory exhaustion in dense solver.
/// At 5,000 items, matrix requires ~200 MB; larger scales need sparse solver.
const MAX_ITEMS: usize = 5_000;

/// Maximum candidates in planner to prevent DoS via unbounded iteration.
const MAX_CANDIDATES: usize = 50_000;

/// Maximum reps per observation to prevent ranking manipulation via extreme weights.
const MAX_REPS: f64 = 1000.0;

/// When the reduced system is small, compute exact diag(L^-1) via Cholesky solves.
const EXACT_DIAG_MAX_DIM: usize = 256;

type IrlsHuberSolveResult = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Option<Cholesky<f64, nalgebra::Dyn>>,
    bool,
);

type FuseBucketKey = (usize, usize);
type FuseBucketEntry = (f64, f64, String);
type FuseBuckets = HashMap<FuseBucketKey, Vec<FuseBucketEntry>>;

// ---------------------------------------------------------------------
//  Config
// ---------------------------------------------------------------------

/// Configuration for the rating engine (IRLS solver + planner).
///
/// See `docs/ALGORITHM.md` for rationale behind these defaults.
#[derive(Debug, Clone)]
pub struct Config {
    // -- Confidence mapping --------------------------------------------------
    // Maps LLM confidence (0..1) to observation weight via g(c) = eps + (1-eps)*c^gamma.
    // Higher gamma = more aggressive discounting of low-confidence judgments.

    /// Floor for confidence weight — even confidence=0 observations get this much weight.
    pub eps_confidence: f64,
    /// Exponent for confidence curve. 2.0 means the LLM must be quite confident
    /// before an observation gets substantial weight in the solver.
    pub gamma_confidence: f64,

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
            eps_confidence: 1e-3,
            gamma_confidence: 2.0,
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

#[derive(Debug, Clone)]
pub struct AttributeParams {
    /// Global noise / difficulty parameter T.
    pub temperature: f64,
}

impl Default for AttributeParams {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

#[derive(Debug, Clone)]
pub struct RaterParams {
    /// Efficacy / effective sample size (β).
    pub beta: f64,
    /// Cost used by planner.
    pub cost_per_edge: f64,
    /// Default confidence used by planner.
    pub default_confidence: f64,
}

impl Default for RaterParams {
    fn default() -> Self {
        Self {
            beta: 1.0,
            cost_per_edge: 1.0,
            default_confidence: 0.75,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Observation {
    pub i: usize,
    pub j: usize,
    pub ratio: f64,
    pub confidence: f64,
    pub rater_id: String,
    pub reps: f64,
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

// ---------------------------------------------------------------------
//  Utilities
// ---------------------------------------------------------------------

fn g_of_c(c: f64, cfg: &Config) -> f64 {
    let c = c.clamp(0.0, 1.0);
    cfg.eps_confidence + (1.0 - cfg.eps_confidence) * c.powf(cfg.gamma_confidence)
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn mad(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let m = median(x.to_vec());
    let devs: Vec<f64> = x.iter().map(|v| (v - m).abs()).collect();
    1.4826 * median(devs)
}

fn weighted_median(x: &[f64], w: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 || w.len() != n {
        return 0.0;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(Ordering::Equal));

    let mut x_sorted = Vec::with_capacity(n);
    let mut w_sorted = Vec::with_capacity(n);
    for i in idx {
        x_sorted.push(x[i]);
        w_sorted.push(w[i].max(0.0));
    }

    let w_sum: f64 = w_sorted.iter().sum();
    if w_sum <= 0.0 {
        return median(x_sorted);
    }

    let cutoff = 0.5 * w_sum;
    let mut cum = 0.0;
    for (xi, wi) in x_sorted.iter().zip(w_sorted.iter()) {
        cum += *wi;
        if cum >= cutoff {
            return *xi;
        }
    }
    *x_sorted.last().unwrap()
}

// ---------------------------------------------------------------------
//  Graph topology and gauge
// ---------------------------------------------------------------------

fn compute_components(n: usize, edges: &[Edge]) -> Vec<usize> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for e in edges {
        let i = e.i;
        let j = e.j;
        if i == j || i >= n || j >= n {
            continue;
        }
        adj[i].push(j);
        adj[j].push(i);
    }

    let mut labels = vec![usize::MAX; n];
    let mut comp_id = 0;
    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        labels[start] = comp_id;
        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if labels[v] == usize::MAX {
                    labels[v] = comp_id;
                    stack.push(v);
                }
            }
        }
        comp_id += 1;
    }
    labels
}

/// Pin one node per connected component (min index) and return:
/// - keep_idx: non-pinned nodes
/// - labels: component labels for each node
fn pin_nodes(n: usize, edges: &[Edge]) -> (Vec<usize>, Vec<usize>) {
    if edges.is_empty() {
        // Match Python behavior: no edges → keep all nodes free, each its own component.
        let labels: Vec<usize> = (0..n).collect();
        let keep_idx: Vec<usize> = (0..n).collect();
        return (keep_idx, labels);
    }

    let labels = compute_components(n, edges);
    let mut keep_mask = vec![true; n];

    let max_label = labels.iter().copied().max().unwrap_or(0);
    for c in 0..=max_label {
        let mut min_node: Option<usize> = None;
        for (node, &lab) in labels.iter().enumerate() {
            if lab == c {
                min_node = Some(match min_node {
                    None => node,
                    Some(m) => m.min(node),
                });
            }
        }
        if let Some(pin) = min_node {
            keep_mask[pin] = false;
        }
    }

    let keep_idx: Vec<usize> = (0..n).filter(|&i| keep_mask[i]).collect();
    (keep_idx, labels)
}

/// Shift scores so that min(score) = 0 for each connected component.
fn normalize_per_component(scores: &[f64], labels: &[usize]) -> Vec<f64> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = scores.to_vec();
    let max_label = labels.iter().copied().max().unwrap_or(0);
    for c in 0..=max_label {
        let mut min_score: Option<f64> = None;
        for (i, &lab) in labels.iter().enumerate() {
            if lab == c {
                let s = out[i];
                min_score = Some(match min_score {
                    None => s,
                    Some(m) => m.min(s),
                });
            }
        }
        if let Some(m) = min_score {
            for (i, &lab) in labels.iter().enumerate() {
                if lab == c {
                    out[i] -= m;
                }
            }
        }
    }
    out
}

fn build_pos_map(n: usize, keep_idx: &[usize]) -> Vec<Option<usize>> {
    let mut pos = vec![None; n];
    for (p, &node) in keep_idx.iter().enumerate() {
        if node < n {
            pos[node] = Some(p);
        }
    }
    pos
}

struct LinearSolveResult {
    s_full: Vec<f64>,
    diag_fallback: Vec<f64>,
    chol: Option<Cholesky<f64, nalgebra::Dyn>>,
    degraded: bool,
}

// ---------------------------------------------------------------------
//  IRLS with Huber loss
// ---------------------------------------------------------------------

/// Solve (B^T W B) s = B^T W mu in free coordinates (keep_idx).
/// Returns (solution, diagonal of L, Cholesky decomposition) for reuse.
fn solve_weighted_least_squares(
    n: usize,
    edges: &[Edge],
    mu: &[f64],
    lam_eff: &[f64],
    keep_idx: &[usize],
    cfg: &Config,
) -> LinearSolveResult {
    let m = edges.len();
    let kdim = keep_idx.len();

    if kdim == 0 {
        return LinearSolveResult {
            s_full: vec![0.0; n],
            diag_fallback: Vec::new(),
            chol: None,
            degraded: false,
        };
    }
    if m == 0 {
        return LinearSolveResult {
            s_full: vec![0.0; n],
            diag_fallback: vec![0.0; kdim],
            chol: None,
            degraded: false,
        };
    }

    let pos = build_pos_map(n, keep_idx);

    let build_system = |ridge_lambda: f64| -> (DMatrix<f64>, DVector<f64>, Vec<f64>) {
        let mut l_red = DMatrix::<f64>::zeros(kdim, kdim);
        let mut rhs_red = DVector::<f64>::zeros(kdim);

        for (k, e) in edges.iter().enumerate() {
            let i = e.i;
            let j = e.j;
            let w = lam_eff[k];
            if w <= 0.0 {
                continue;
            }
            let mu_k = mu[k];

            let pi_opt = if i < n { pos[i] } else { None };
            let pj_opt = if j < n { pos[j] } else { None };

            if let Some(pi) = pi_opt {
                l_red[(pi, pi)] += w;
                rhs_red[pi] += w * mu_k;
            }
            if let Some(pj) = pj_opt {
                l_red[(pj, pj)] += w;
                rhs_red[pj] -= w * mu_k;
            }
            if let (Some(pi), Some(pj)) = (pi_opt, pj_opt) {
                l_red[(pi, pj)] -= w;
                l_red[(pj, pi)] -= w;
            }
        }

        if ridge_lambda > 0.0 {
            for d in 0..kdim {
                l_red[(d, d)] += ridge_lambda;
            }
        }

        let mut diag_fallback = Vec::with_capacity(kdim);
        for d in 0..kdim {
            diag_fallback.push(l_red[(d, d)]);
        }

        (l_red, rhs_red, diag_fallback)
    };

    let base_ridge = cfg.ridge_lambda.max(0.0);
    let mut ridge_candidates = Vec::new();
    ridge_candidates.push(base_ridge);

    let mut ridge = if base_ridge > 0.0 { base_ridge } else { 1e-9 };
    for _ in 0..4 {
        ridge *= 10.0;
        ridge_candidates.push(ridge);
    }

    let mut diag_fallback = Vec::new();
    let mut chol: Option<Cholesky<f64, nalgebra::Dyn>> = None;
    let mut x = DVector::<f64>::zeros(kdim);
    let mut used_ridge = base_ridge;
    let want_eig_fallback = kdim <= EXACT_DIAG_MAX_DIM;
    let mut last_l_red: Option<DMatrix<f64>> = None;
    let mut last_rhs_red: Option<DVector<f64>> = None;

    for ridge_lambda in ridge_candidates.iter() {
        let (l_red, rhs_red, diag) = build_system(*ridge_lambda);
        diag_fallback = diag;
        if want_eig_fallback {
            last_l_red = Some(l_red.clone());
            last_rhs_red = Some(rhs_red.clone());
        }
        let attempt = Cholesky::new(l_red);
        if let Some(c) = attempt {
            x = c.solve(&rhs_red);
            chol = Some(c);
            used_ridge = *ridge_lambda;
            break;
        }
        used_ridge = *ridge_lambda;
    }

    if chol.is_none() && want_eig_fallback {
        if let (Some(l_red), Some(rhs_red)) = (last_l_red, last_rhs_red) {
            let eig = SymmetricEigen::new(l_red);
            let mut inv = eig.eigenvalues.clone();
            for i in 0..inv.len() {
                let denom = if inv[i].abs() <= cfg.tiny {
                    cfg.tiny
                } else {
                    inv[i].abs()
                };
                inv[i] = 1.0 / denom;
            }
            let vt_rhs = eig.eigenvectors.transpose() * rhs_red;
            let scaled = inv.component_mul(&vt_rhs);
            x = &eig.eigenvectors * scaled;

            let mut diag_inv = Vec::with_capacity(kdim);
            for i in 0..kdim {
                let mut acc = 0.0;
                for k in 0..kdim {
                    let v = eig.eigenvectors[(i, k)];
                    acc += v * v * inv[k];
                }
                diag_inv.push(acc.max(0.0));
            }
            diag_fallback = diag_inv;
        }
    }

    let mut s_full = vec![0.0; n];
    for (p, &node) in keep_idx.iter().enumerate() {
        s_full[node] = x[p];
    }

    let degraded = chol.is_none() || used_ridge > base_ridge + cfg.tiny;

    LinearSolveResult {
        s_full,
        diag_fallback,
        chol,
        degraded,
    }
}

/// Hutchinson estimation of diag(L^-1) in reduced coordinates.
/// Accepts precomputed Cholesky to avoid redundant O(n³) decomposition.
fn hutchinson_diag(
    diag_fallback: &[f64],
    precomputed_chol: Option<&Cholesky<f64, nalgebra::Dyn>>,
    probes: usize,
    cfg: &Config,
    rng: &mut StdRng,
) -> Vec<f64> {
    let n = diag_fallback.len();
    if n == 0 {
        return Vec::new();
    }

    let chol = match precomputed_chol {
        Some(c) => c,
        None => {
            // Fallback: diagonal approximation when Cholesky fails (ill-conditioned or
            // weakly connected Laplacian). This is conservative but less accurate.
            let mut diag = Vec::with_capacity(n);
            for &d in diag_fallback {
                let denom = if d.abs() <= cfg.tiny {
                    cfg.tiny
                } else {
                    d.abs()
                };
                diag.push((1.0 / denom).max(0.0));
            }
            return diag;
        }
    };

    if n <= EXACT_DIAG_MAX_DIM {
        let mut diag = Vec::with_capacity(n);
        for i in 0..n {
            let mut e = DVector::<f64>::zeros(n);
            e[i] = 1.0;
            let x = chol.solve(&e);
            diag.push(x[i].max(0.0));
        }
        return diag;
    }

    let probes = probes.max(1);
    let mut acc = DVector::<f64>::zeros(n);

    for _ in 0..probes {
        let z = DVector::from_iterator(
            n,
            (0..n).map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 }),
        );
        let x = chol.solve(&z);
        acc += z.component_mul(&x);
    }

    let inv_probes = 1.0 / (probes as f64);
    (0..n).map(|i| (acc[i] * inv_probes).max(0.0)).collect()
}

/// Robust IRLS loop with Huber loss.
fn solve_irls_huber(
    n: usize,
    edges: &[Edge],
    cfg: &Config,
    keep_idx: &[usize],
) -> IrlsHuberSolveResult {
    let m = edges.len();
    if m == 0 {
        return (
            vec![0.0; n],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            false,
        );
    }

    let mu: Vec<f64> = edges.iter().map(|e| e.mu).collect();
    let lam_raw: Vec<f64> = edges.iter().map(|e| e.lam).collect();
    let mut lam_eff = lam_raw.clone();

    let mut residuals = vec![0.0; m];

    let mut last_obj: Option<f64> = None;

    for _ in 0..cfg.irls_max_iters {
        let solve = solve_weighted_least_squares(n, edges, &mu, &lam_eff, keep_idx, cfg);
        let s_candidate = solve.s_full;

        for (k, e) in edges.iter().enumerate() {
            residuals[k] = mu[k] - (s_candidate[e.i] - s_candidate[e.j]);
        }
        if residuals.is_empty() {
            break;
        }

        let scale_mad = mad(&residuals);
        let scale = if scale_mad <= cfg.tiny {
            residuals.iter().map(|r| r.abs()).fold(0.0, f64::max)
        } else {
            scale_mad
        };
        if scale <= cfg.tiny {
            // Perfect fit (no residual spread) — keep weights unchanged.
            break;
        }
        let delta = cfg.huber_k * scale;

        let mut z = vec![1.0; m];
        for k in 0..m {
            let abs_r = residuals[k].abs();
            if abs_r > delta {
                z[k] = delta / (abs_r + cfg.tiny);
            }
        }

        let lam_eff_new: Vec<f64> = lam_raw
            .iter()
            .zip(z.iter())
            .map(|(lr, zz)| lr * zz)
            .collect();

        let obj: f64 = lam_eff_new
            .iter()
            .zip(residuals.iter())
            .map(|(lam, r)| lam * r * r)
            .sum();

        if let Some(prev) = last_obj {
            if (prev - obj).abs() <= cfg.irls_tol * prev.max(1.0) {
                lam_eff = lam_eff_new;
                break;
            }
        }

        lam_eff = lam_eff_new;
        last_obj = Some(obj);
    }

    // Final solve with converged weights - reuse Cholesky for hutchinson_diag
    let final_solve = solve_weighted_least_squares(n, edges, &mu, &lam_eff, keep_idx, cfg);
    let s_full = final_solve.s_full;

    for (k, e) in edges.iter().enumerate() {
        residuals[k] = mu[k] - (s_full[e.i] - s_full[e.j]);
    }

    let seed = probe_seed(cfg.rng_seed, edges, &lam_eff);
    let mut probe_rng = StdRng::seed_from_u64(seed);
    let diag_red = hutchinson_diag(
        &final_solve.diag_fallback,
        final_solve.chol.as_ref(),
        cfg.hutch_probes,
        cfg,
        &mut probe_rng,
    );

    (
        s_full,
        residuals,
        lam_eff,
        diag_red,
        final_solve.chol,
        final_solve.degraded,
    )
}

// ---------------------------------------------------------------------
//  Diagnostics: HCR / PCR-lite
// ---------------------------------------------------------------------

fn compute_hcr(mu: &[f64], residuals: &[f64], lam_eff: &[f64], cfg: &Config) -> f64 {
    if mu.is_empty() || lam_eff.is_empty() {
        return 0.0;
    }
    let num: f64 = if residuals.is_empty() {
        0.0
    } else {
        lam_eff
            .iter()
            .zip(residuals.iter())
            .map(|(lam, r)| lam * r * r)
            .sum()
    };
    let den: f64 = lam_eff
        .iter()
        .zip(mu.iter())
        .map(|(lam, m)| lam * m * m)
        .sum::<f64>()
        + cfg.tiny;

    let hcr = num / den;
    hcr.clamp(0.0, 1.0)
}

fn probe_seed(seed: u64, edges: &[Edge], lam_eff: &[f64]) -> u64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    edges.len().hash(&mut hasher);
    for (k, e) in edges.iter().enumerate() {
        e.i.hash(&mut hasher);
        e.j.hash(&mut hasher);
        e.mu.to_bits().hash(&mut hasher);
        e.lam.to_bits().hash(&mut hasher);
        if k < lam_eff.len() {
            lam_eff[k].to_bits().hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn compute_pcr_lite(mu: &[f64], residuals: &[f64], lam_eff: &[f64], cfg: &Config) -> f64 {
    if mu.is_empty() || residuals.is_empty() || lam_eff.is_empty() {
        return 0.0;
    }
    let w_sum: f64 = lam_eff.iter().sum::<f64>() + cfg.tiny;
    let mse_resid: f64 = lam_eff
        .iter()
        .zip(residuals.iter())
        .map(|(lam, r)| lam * r * r)
        .sum::<f64>()
        / w_sum;
    let mean_mu = mu.iter().sum::<f64>() / (mu.len() as f64);
    let var_signal: f64 = mu
        .iter()
        .map(|v| {
            let d = v - mean_mu;
            d * d
        })
        .sum::<f64>()
        / (mu.len() as f64)
        + cfg.tiny;

    let pcr = 1.0 - mse_resid / var_signal;
    pcr.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------
//  Ranker: ranks, weights, pair selection, P_flip
// ---------------------------------------------------------------------

fn ranks_from_scores(scores: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let n = scores.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));
    let mut rank_of = vec![0usize; n];
    for (pos, &idx) in order.iter().enumerate() {
        rank_of[idx] = pos;
    }
    (order, rank_of)
}

fn rank_weights(n: usize, cfg: &Config) -> Vec<f64> {
    let a = cfg.rank_weight_exponent.max(0.0);
    if a == 0.0 {
        return vec![1.0; n];
    }
    (1..=n)
        .map(|k| 1.0 / (k as f64).powf(a))
        .collect::<Vec<f64>>()
}

#[derive(Default)]
struct RankCache {
    order: Vec<usize>,
    rank_of: Vec<usize>,
    w_rank: Vec<f64>,
}

fn pair_rank_weight(
    scores: &[f64],
    i: usize,
    j: usize,
    cfg: &Config,
    cache: &mut RankCache,
) -> f64 {
    if cache.order.is_empty() {
        let (order, rank_of) = ranks_from_scores(scores);
        let w_rank_vec = rank_weights(scores.len(), cfg);
        cache.order = order;
        cache.rank_of = rank_of;
        cache.w_rank = w_rank_vec;
    }

    let rank_of = &cache.rank_of;
    let w_rank_vec = &cache.w_rank;
    let ri = rank_of[i];
    let rj = rank_of[j];
    let pi = ri + 1;
    let pj = rj + 1;

    let mut base = 0.5 * (w_rank_vec[ri] + w_rank_vec[rj]);

    if let Some(top_k) = cfg.top_k {
        let tail_weight = cfg.tail_weight.clamp(0.0, 1.0);
        if pi > top_k && pj > top_k {
            base *= tail_weight;
        }
    }

    base
}

fn select_rank_pairs(scores: &[f64], cfg: &Config) -> Vec<(usize, usize)> {
    let n = scores.len();
    if n <= 1 {
        return Vec::new();
    }

    let (order, _) = ranks_from_scores(scores);
    let mut pairs: Vec<(usize, usize)> = Vec::new();

    let w = cfg.rank_band_window.max(1);

    // Adjacent neighbors + Rank band (combined: positions 0..w+1 from each)
    for (pos, &i) in order.iter().enumerate() {
        for &j in order[(pos + 1)..std::cmp::min(n, pos + w + 1)].iter() {
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            pairs.push((a, b));
        }
    }

    // Small-gap pairs
    let thr = cfg.small_gap_threshold.max(0.0);
    for (pos, &i) in order.iter().enumerate() {
        let s_i = scores[i];
        for &j in order.iter().skip(pos + 1) {
            let s_j = scores[j];
            if (s_i - s_j).abs() <= thr {
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                pairs.push((a, b));
            } else {
                break;
            }
        }
    }

    pairs.sort_unstable();
    pairs.dedup();
    if let Some(max_pairs) = cfg.max_rank_pairs {
        if pairs.len() > max_pairs {
            pairs.truncate(max_pairs);
        }
    }
    pairs
}

pub(crate) fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / SQRT_2))
}

fn pair_prob_and_flip(
    scores: &[f64],
    diag_cov: &[f64],
    i: usize,
    j: usize,
    cfg: &Config,
) -> (f64, f64) {
    let diff = scores[i] - scores[j];
    let var_diff = (diag_cov[i] + diag_cov[j]).max(0.0);

    let p_gt = if var_diff <= cfg.tiny {
        if diff > 0.0 {
            1.0
        } else if diff < 0.0 {
            0.0
        } else {
            0.5
        }
    } else {
        let z = diff / var_diff.sqrt();
        normal_cdf(z)
    };

    let mut p_flip = if diff < 0.0 { p_gt } else { 1.0 - p_gt };
    if diff == 0.0 {
        p_flip = 0.5;
    }
    p_flip = p_flip.clamp(0.0, 1.0);
    (p_gt, p_flip)
}

fn compute_rank_stability(scores: &[f64], diag_cov: &[f64], cfg: &Config) -> (f64, f64, f64) {
    let n = scores.len();
    if n <= 1 {
        return (0.0, 0.0, 0.0);
    }

    let pairs = select_rank_pairs(scores, cfg);
    if pairs.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut cache = RankCache::default();
    let mut total_flip = 0.0;
    let mut max_flip = 0.0;
    let mut rank_risk = 0.0;

    for (i, j) in pairs {
        let (_, p_flip) = pair_prob_and_flip(scores, diag_cov, i, j, cfg);
        total_flip += p_flip;
        if p_flip > max_flip {
            max_flip = p_flip;
        }
        let w_ij = pair_rank_weight(scores, i, j, cfg, &mut cache);
        rank_risk += w_ij * p_flip;
    }

    (total_flip, max_flip, rank_risk)
}

// ---------------------------------------------------------------------
//  Calibration evidence
// ---------------------------------------------------------------------

fn compute_calibration_evidence(
    residuals: &[f64],
    edges: &[Edge],
    lam_eff: &[f64],
    cfg: &Config,
) -> CalibrationEvidence {
    let m = residuals.len();
    if m == 0 || edges.is_empty() || lam_eff.is_empty() {
        return CalibrationEvidence {
            global_variance_obs: 0.0,
            global_mad_obs: 0.0,
            inferred_temperature: 0.0,
            rater_efficacy_obs: HashMap::new(),
            rater_bias_obs: HashMap::new(),
            rater_scatter_obs: HashMap::new(),
        };
    }

    let w_edge = lam_eff;
    let w_sum: f64 = w_edge.iter().sum::<f64>() + cfg.tiny;
    let mse_global: f64 = w_edge
        .iter()
        .zip(residuals.iter())
        .map(|(w, r)| w * r * r)
        .sum::<f64>()
        / w_sum;
    let global_var = mse_global;
    let global_mad = mad(residuals);

    let mut r_residuals: HashMap<String, Vec<f64>> = HashMap::new();
    let mut r_weights: HashMap<String, Vec<f64>> = HashMap::new();

    for (k, e) in edges.iter().enumerate() {
        let r_k = residuals[k];
        let lam_tot = e.lam.max(cfg.tiny);
        let lam_eff_k = w_edge[k];
        if lam_eff_k <= 0.0 {
            continue;
        }
        for (rid, lam_r) in e.contributors.iter() {
            let phi = lam_r.max(0.0) / lam_tot;
            let w = lam_eff_k * phi;
            if w <= 0.0 {
                continue;
            }
            r_residuals.entry(rid.clone()).or_default().push(r_k);
            r_weights.entry(rid.clone()).or_default().push(w);
        }
    }

    let mut rater_efficacy = HashMap::new();
    let mut rater_bias = HashMap::new();
    let mut rater_scatter = HashMap::new();

    let empty_weights: Vec<f64> = Vec::new();
    for (rid, vals) in r_residuals.iter() {
        let rs = vals;
        let ws = r_weights.get(rid).unwrap_or(&empty_weights);
        if rs.is_empty() || ws.is_empty() {
            continue;
        }
        let w_sum_r: f64 = ws.iter().sum::<f64>() + cfg.tiny;
        let mse_r: f64 = rs
            .iter()
            .zip(ws.iter())
            .map(|(r, w)| w * r * r)
            .sum::<f64>()
            / w_sum_r;
        let bias_r = weighted_median(rs, ws);

        let beta_hat = if mse_r <= 0.0 {
            0.0
        } else {
            global_var / mse_r
        };
        let scatter = if global_var > 0.0 {
            (mse_r / global_var.max(cfg.tiny)).sqrt()
        } else {
            0.0
        };

        rater_efficacy.insert(rid.clone(), beta_hat.max(0.0));
        rater_bias.insert(rid.clone(), bias_r);
        rater_scatter.insert(rid.clone(), scatter);
    }

    CalibrationEvidence {
        global_variance_obs: global_var,
        global_mad_obs: global_mad,
        inferred_temperature: global_var,
        rater_efficacy_obs: rater_efficacy,
        rater_bias_obs: rater_bias,
        rater_scatter_obs: rater_scatter,
    }
}

// ---------------------------------------------------------------------
//  Engine
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RatingEngine {
    pub n: usize,
    pub attr: AttributeParams,
    pub raters: HashMap<String, RaterParams>,
    pub cfg: Config,

    pub edges: Vec<Edge>,
    edge_index: HashMap<(usize, usize), usize>,

    // Cached topology
    keep_idx: Vec<usize>,
    labels: Vec<usize>,
    topology_dirty: bool,

    // Cached solve results
    last_scores: Option<Vec<f64>>,
    last_diag_cov: Option<Vec<f64>>,
    last_residuals: Option<Vec<f64>>,
    last_lam_eff: Option<Vec<f64>>,
    last_chol: Option<Cholesky<f64, nalgebra::Dyn>>,
}

impl RatingEngine {
    pub fn new(
        n: usize,
        attr: AttributeParams,
        raters: HashMap<String, RaterParams>,
        cfg: Option<Config>,
    ) -> Result<Self, &'static str> {
        if n == 0 {
            return Err("Item count must be positive");
        }
        if n > MAX_ITEMS {
            return Err("Item count exceeds maximum allowed (5,000)");
        }
        let cfg = cfg.unwrap_or_default();
        Ok(Self {
            n,
            attr,
            raters,
            cfg,
            edges: Vec::new(),
            edge_index: HashMap::new(),
            keep_idx: (0..n).collect(),
            labels: (0..n).collect(),
            topology_dirty: true,
            last_scores: None,
            last_diag_cov: None,
            last_residuals: None,
            last_lam_eff: None,
            last_chol: None,
        })
    }

    pub fn scores(&self) -> Option<&[f64]> {
        self.last_scores.as_deref()
    }

    pub fn diag_cov(&self) -> Option<&[f64]> {
        self.last_diag_cov.as_deref()
    }

    /// Targeted marginal variances for a subset of indices (same order as input).
    /// Uses the current reduced Laplacian solve state when available.
    pub fn marginal_vars_for(&self, indices: &[usize]) -> Option<Vec<f64>> {
        if indices.is_empty() {
            return Some(Vec::new());
        }

        let chol = self.last_chol.as_ref()?;
        let pos = build_pos_map(self.n, &self.keep_idx);

        let rdim = chol.l().nrows();
        let mut out = Vec::with_capacity(indices.len());
        for &idx in indices {
            let p = match pos.get(idx).copied().flatten() {
                Some(v) => v,
                None => {
                    out.push(0.0);
                    continue;
                }
            };
            if p >= rdim {
                out.push(0.0);
                continue;
            }
            let mut b = DVector::<f64>::zeros(rdim);
            b[p] = 1.0;
            let x = chol.solve(&b);
            out.push(x[p].max(0.0));
        }

        Some(out)
    }

    /// Variance of score difference s_i - s_j using the reduced Laplacian.
    pub fn diff_var_for(&self, i: usize, j: usize) -> Option<f64> {
        let chol = self.last_chol.as_ref()?;
        let diag_cov = self.last_diag_cov.as_ref()?;
        if i >= self.labels.len() || j >= self.labels.len() {
            return None;
        }
        if self.labels[i] != self.labels[j] {
            return Some((diag_cov[i] + diag_cov[j]).max(0.0));
        }
        let pos = build_pos_map(self.n, &self.keep_idx);
        let pi = pos.get(i).copied().flatten();
        let pj = pos.get(j).copied().flatten();

        let rdim = chol.l().nrows();
        if pi.is_none() && pj.is_none() {
            return Some((diag_cov[i] + diag_cov[j]).max(0.0));
        }
        if let Some(pi) = pi {
            if pi >= rdim {
                return Some((diag_cov[i] + diag_cov[j]).max(0.0));
            }
        }
        if let Some(pj) = pj {
            if pj >= rdim {
                return Some((diag_cov[i] + diag_cov[j]).max(0.0));
            }
        }
        let mut b = DVector::<f64>::zeros(rdim);
        if let Some(pi) = pi {
            b[pi] = 1.0;
        }
        if let Some(pj) = pj {
            b[pj] = -1.0;
        }
        let x = chol.solve(&b);
        Some(b.dot(&x).max(0.0))
    }

    /// Whether two nodes are in the same connected component (based on current edges).
    pub fn same_component(&self, i: usize, j: usize) -> bool {
        if self.topology_dirty {
            return false;
        }
        if i >= self.labels.len() || j >= self.labels.len() {
            return false;
        }
        self.labels[i] == self.labels[j]
    }

    pub fn has_min_degree(&self, idx: usize, min_degree: usize) -> bool {
        if min_degree == 0 {
            return true;
        }
        let mut deg = 0usize;
        for edge in &self.edges {
            if edge.i == idx || edge.j == idx {
                deg += 1;
                if deg >= min_degree {
                    return true;
                }
            }
        }
        false
    }

    fn mark_dirty_after_edges_change(&mut self, topology_changed: bool) {
        if topology_changed {
            self.topology_dirty = true;
        }
        self.last_scores = None;
        self.last_diag_cov = None;
        self.last_residuals = None;
        self.last_lam_eff = None;
        self.last_chol = None;
    }

    fn fuse_bulk(&mut self, observations: &[Observation]) {
        let t = self.attr.temperature.max(self.cfg.tiny);
        let mut buckets: FuseBuckets = HashMap::new();

        for ob in observations {
            let i = ob.i;
            let j = ob.j;
            if i == j {
                continue;
            }
            if i >= self.n || j >= self.n {
                continue;
            }

            let (u, v, sign) = if i < j { (i, j, 1.0) } else { (j, i, -1.0) };

            if !ob.ratio.is_finite() || ob.ratio <= 0.0 {
                continue;
            }
            let ratio = ob.ratio.max(self.cfg.tiny);
            let mut log_r = sign * ratio.ln();
            let max_log = self.cfg.max_log_ratio;
            log_r = log_r.clamp(-max_log, max_log);

            let beta_r = match self.raters.get(&ob.rater_id) {
                Some(r) => r.beta.max(self.cfg.tiny),
                None => continue, // skip unknown raters
            };
            let c = ob.confidence.clamp(0.0, 1.0);
            let reps = ob.reps.clamp(0.0, MAX_REPS);

            let lam = (beta_r * g_of_c(c, &self.cfg) * reps) / t;
            if !lam.is_finite() || lam <= 0.0 {
                continue;
            }

            buckets
                .entry((u, v))
                .or_default()
                .push((log_r, lam, ob.rater_id.clone()));
        }

        let mut edges = Vec::new();
        let mut edge_index = HashMap::new();

        for ((i, j), lst) in buckets.into_iter() {
            if lst.is_empty() {
                continue;
            }
            let mut num = 0.0;
            let mut lam_total = 0.0;
            let mut contribs: HashMap<String, f64> = HashMap::new();
            for (mu_obs, lam, rid) in lst.into_iter() {
                lam_total += lam;
                num += mu_obs * lam;
                *contribs.entry(rid).or_insert(0.0) += lam;
            }
            if lam_total <= 0.0 {
                continue;
            }
            let mu = num / lam_total;
            let idx = edges.len();
            edges.push(Edge {
                i,
                j,
                mu,
                lam: lam_total,
                contributors: contribs,
            });
            edge_index.insert((i, j), idx);
        }

        self.edges = edges;
        self.edge_index = edge_index;
    }

    /// Replace edge set by fusing all observations (bulk ingest/reset).
    /// Use `add_observations` for incremental updates.
    pub fn ingest(&mut self, observations: &[Observation]) {
        self.fuse_bulk(observations);
        self.mark_dirty_after_edges_change(true);
    }

    pub fn add_observations(&mut self, observations: &[Observation]) {
        let t = self.attr.temperature.max(self.cfg.tiny);
        let mut new_edge_added = false;

        for ob in observations {
            let i = ob.i;
            let j = ob.j;
            if i == j {
                continue;
            }
            if i >= self.n || j >= self.n {
                continue;
            }

            let (u, v, sign) = if i < j { (i, j, 1.0) } else { (j, i, -1.0) };

            if !ob.ratio.is_finite() || ob.ratio <= 0.0 {
                continue;
            }
            let ratio = ob.ratio.max(self.cfg.tiny);
            let mut log_r = sign * ratio.ln();
            let max_log = self.cfg.max_log_ratio;
            log_r = log_r.clamp(-max_log, max_log);

            let beta_r = match self.raters.get(&ob.rater_id) {
                Some(r) => r.beta.max(self.cfg.tiny),
                None => continue, // skip unknown raters
            };
            let c = ob.confidence.clamp(0.0, 1.0);
            let reps = ob.reps.clamp(0.0, MAX_REPS);
            let lam_new = (beta_r * g_of_c(c, &self.cfg) * reps) / t;
            if !lam_new.is_finite() || lam_new <= 0.0 {
                continue;
            }

            let key = (u, v);
            if let Some(idx) = self.edge_index.get(&key).copied() {
                let e = &mut self.edges[idx];
                let lam_prev = e.lam;
                let lam_tot = lam_prev + lam_new;
                if lam_tot <= 0.0 {
                    continue;
                }
                let mu_prev = e.mu;
                let mu_new = (mu_prev * lam_prev + log_r * lam_new) / lam_tot;
                e.mu = mu_new;
                e.lam = lam_tot;
                *e.contributors.entry(ob.rater_id.clone()).or_insert(0.0) += lam_new;
            } else {
                let mut contributors = HashMap::new();
                contributors.insert(ob.rater_id.clone(), lam_new);
                let e = Edge {
                    i: u,
                    j: v,
                    mu: log_r,
                    lam: lam_new,
                    contributors,
                };
                self.edges.push(e);
                let idx = self.edges.len() - 1;
                self.edge_index.insert(key, idx);
                new_edge_added = true;
            }
        }

        self.mark_dirty_after_edges_change(new_edge_added);
    }

    fn ensure_topology(&mut self) {
        if self.topology_dirty {
            let (keep_idx, labels) = pin_nodes(self.n, &self.edges);
            self.keep_idx = keep_idx;
            self.labels = labels;
            self.topology_dirty = false;
        }
    }

    pub fn solve(&mut self) -> SolveSummary {
        self.ensure_topology();
        let keep_idx = self.keep_idx.clone();

        let (s, residuals, lam_eff, diag_red, chol, degraded) =
            solve_irls_huber(self.n, &self.edges, &self.cfg, &keep_idx);

        let mut diag_cov = vec![0.0; self.n];
        if !diag_red.is_empty() && diag_red.len() == keep_idx.len() {
            for (pos, &node) in keep_idx.iter().enumerate() {
                diag_cov[node] = diag_red[pos].max(0.0);
            }
        }
        if self.edges.is_empty() {
            diag_cov.fill(1.0);
        }
        if !self.edges.is_empty() {
            let mut keep_mask = vec![false; self.n];
            for &node in &keep_idx {
                if node < self.n {
                    keep_mask[node] = true;
                }
            }
            let components = if self.labels.is_empty() {
                self.n
            } else {
                self.labels.iter().copied().max().unwrap_or(0) + 1
            };
            let mut comp_max: Vec<f64> = vec![0.0; components];
            for i in 0..self.n {
                if !keep_mask[i] {
                    continue;
                }
                if i >= self.labels.len() {
                    continue;
                }
                let c = self.labels[i];
                let v = diag_cov[i];
                if v.is_finite() {
                    comp_max[c] = comp_max[c].max(v);
                }
            }
            let mut global_max = comp_max.iter().copied().fold(0.0, f64::max);
            if global_max <= 0.0 {
                global_max = 1.0;
            }
            for i in 0..self.n {
                if keep_mask[i] {
                    continue;
                }
                if i >= self.labels.len() {
                    continue;
                }
                let c = self.labels[i];
                let fallback = if c < comp_max.len() && comp_max[c] > 0.0 {
                    comp_max[c]
                } else {
                    global_max
                };
                diag_cov[i] = fallback;
            }
        }

        let mu: Vec<f64> = self.edges.iter().map(|e| e.mu).collect();
        let scores_norm = normalize_per_component(&s, &self.labels);

        let hcr = compute_hcr(&mu, &residuals, &lam_eff, &self.cfg);
        let pcr = compute_pcr_lite(&mu, &residuals, &lam_eff, &self.cfg);

        let (expected_rev, max_flip, rank_risk) =
            compute_rank_stability(&scores_norm, &diag_cov, &self.cfg);

        let cal_evidence =
            compute_calibration_evidence(&residuals, &self.edges, &lam_eff, &self.cfg);

        let m = self.edges.len();
        let components = if self.labels.is_empty() {
            self.n
        } else {
            self.labels.iter().copied().max().unwrap_or(0) + 1
        };
        let cycle_dim = (m as isize - self.n as isize + components as isize).max(0) as usize;

        let total_info: f64 = lam_eff.iter().sum();

        self.last_scores = Some(scores_norm.clone());
        self.last_diag_cov = Some(diag_cov.clone());
        self.last_residuals = Some(residuals.clone());
        self.last_lam_eff = Some(lam_eff.clone());
        self.last_chol = chol;

        SolveSummary {
            scores: scores_norm,
            residuals,
            diag_cov,
            hcr,
            pcr,
            total_info,
            expected_rank_reversals: expected_rev,
            max_pair_reversal_prob: max_flip,
            rank_risk,
            components,
            cycle_dim,
            calibration_evidence: cal_evidence,
            degraded,
        }
    }

    pub fn pair_probability(&self, i: usize, j: usize) -> Result<(f64, f64), &'static str> {
        match (&self.last_scores, &self.last_diag_cov) {
            (Some(scores), Some(diag_cov)) => {
                Ok(pair_prob_and_flip(scores, diag_cov, i, j, &self.cfg))
            }
            _ => Err("No solve() results available"),
        }
    }

    pub fn rank_stability(&self) -> Result<(f64, f64, f64), &'static str> {
        match (&self.last_scores, &self.last_diag_cov) {
            (Some(scores), Some(diag_cov)) => {
                Ok(compute_rank_stability(scores, diag_cov, &self.cfg))
            }
            _ => Err("No solve() results available"),
        }
    }
}

// ---------------------------------------------------------------------
//  Planner
// ---------------------------------------------------------------------

/// Compute effective resistance using precomputed Cholesky and position map.
/// Avoids O(N³) Cholesky per pair when called in a loop.
fn effective_resistance_with_chol(
    diag_cov: &[f64],
    labels: &[usize],
    i: usize,
    j: usize,
    chol: &Cholesky<f64, nalgebra::Dyn>,
    pos: &[Option<usize>],
) -> f64 {
    if labels[i] != labels[j] {
        return (diag_cov[i] + diag_cov[j]).max(0.0);
    }

    let pi = pos.get(i).copied().flatten();
    let pj = pos.get(j).copied().flatten();

    if pi.is_none() && pj.is_none() {
        return (diag_cov[i] + diag_cov[j]).max(0.0);
    }

    let rdim = chol.l().nrows();
    let mut b = DVector::<f64>::zeros(rdim);
    if let Some(p) = pi {
        b[p] += 1.0;
    }
    if let Some(p) = pj {
        b[p] -= 1.0;
    }

    let x = chol.solve(&b);
    let r = b.dot(&x);
    r.max(0.0)
}

#[derive(Debug, Clone, Copy)]
pub enum PlannerMode {
    Cardinal,
    Ordinal,
    Hybrid,
}

/// Plans which entity pairs to compare next for the given rater.
///
/// PlannerMode controls optimization objective: Cardinal maximizes information gain,
/// Ordinal minimizes rank uncertainty, Hybrid blends both (weighted by `lambda_risk`).
/// When `use_effective_resistance` is true, uses full graph effective resistance (slower, more accurate);
/// otherwise uses diagonal covariance approximation (faster).
/// Returns proposals sorted by `score` (descending): utility-per-cost of observing each pair.
/// Candidates spanning disconnected components are handled via component-aware fallbacks
/// (effective resistance reduces to diagonal covariance across components).
pub fn plan_edges_for_rater(
    engine: &RatingEngine,
    candidates: &[(usize, usize)],
    rater_id: &str,
    mode: PlannerMode,
    use_effective_resistance: bool,
) -> Result<Vec<PlanProposal>, &'static str> {
    if candidates.len() > MAX_CANDIDATES {
        return Err("Candidate count exceeds maximum allowed (50,000)");
    }

    let scores = match &engine.last_scores {
        Some(s) => s,
        None => return Err("Engine has no solve() state; call solve() first."),
    };
    let diag_cov = match &engine.last_diag_cov {
        Some(c) => c,
        None => return Err("Engine has no solve() state; call solve() first."),
    };

    let cfg = &engine.cfg;
    let n = engine.n;

    let r = match engine.raters.get(rater_id) {
        Some(r) => r,
        None => return Err("Unknown rater_id"),
    };

    let beta_r = r.beta.max(cfg.tiny);
    let cost = r.cost_per_edge.max(cfg.tiny);
    let c_def = r.default_confidence.clamp(0.0, 1.0);
    let t = engine.attr.temperature.max(cfg.tiny);

    let lam_new = (beta_r * g_of_c(c_def, cfg)) / t;

    // Pre-compute Cholesky and position map once (O(N³)), not per-candidate.
    let mut er_pos: Option<Vec<Option<usize>>> = None;
    let mut er_chol: Option<&Cholesky<f64, nalgebra::Dyn>> = None;
    if use_effective_resistance
        && engine.last_chol.is_some()
        && engine.last_diag_cov.is_some()
        && !engine.keep_idx.is_empty()
    {
        er_pos = Some(build_pos_map(engine.n, &engine.keep_idx));
        er_chol = engine.last_chol.as_ref();
    }

    let mut proposals = Vec::new();
    let mut cache = RankCache::default();

    for &(i_raw, j_raw) in candidates {
        let i = i_raw;
        let j = j_raw;
        if i == j || i >= n || j >= n {
            continue;
        }

        let r_ij = if let (Some(chol), Some(pos)) = (er_chol, er_pos.as_ref()) {
            effective_resistance_with_chol(diag_cov, &engine.labels, i, j, chol, pos)
        } else {
            (diag_cov[i] + diag_cov[j]).max(0.0)
        };

        let delta_info = 0.5 * (1.0 + lam_new * r_ij).ln();

        let diff = scores[i] - scores[j];
        let var_before = r_ij.max(cfg.tiny);

        let z_before = diff / var_before.sqrt();
        let p_gt_before = normal_cdf(z_before);
        let mut p_flip_before = if diff < 0.0 {
            p_gt_before
        } else {
            1.0 - p_gt_before
        };
        if diff == 0.0 {
            p_flip_before = 0.5;
        }
        p_flip_before = p_flip_before.clamp(0.0, 1.0);

        let var_after = var_before / (1.0 + lam_new * var_before).max(cfg.tiny);
        let z_after = diff / var_after.max(cfg.tiny).sqrt();
        let p_gt_after = normal_cdf(z_after);
        let mut p_flip_after = if diff < 0.0 {
            p_gt_after
        } else {
            1.0 - p_gt_after
        };
        if diff == 0.0 {
            p_flip_after = 0.5;
        }
        p_flip_after = p_flip_after.clamp(0.0, 1.0);

        let w_ij = pair_rank_weight(scores, i, j, cfg, &mut cache);
        let delta_rank_risk = w_ij * (p_flip_before - p_flip_after);

        let score = match mode {
            PlannerMode::Cardinal => delta_info / cost,
            PlannerMode::Ordinal => delta_rank_risk / cost,
            PlannerMode::Hybrid => (delta_info + cfg.lambda_risk * delta_rank_risk) / cost,
        };

        proposals.push(PlanProposal {
            i,
            j,
            score,
            delta_info,
            delta_rank_risk,
            cost,
        });
    }

    proposals.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                let min_a = a.i.min(a.j);
                let min_b = b.i.min(b.j);
                min_a.cmp(&min_b)
            })
            .then_with(|| {
                let max_a = a.i.max(a.j);
                let max_b = b.i.max(b.j);
                max_a.cmp(&max_b)
            })
    });

    Ok(proposals)
}

// ---------------------------------------------------------------------
//  Convenience: one-shot solve
// ---------------------------------------------------------------------

pub fn solve_once(
    n: usize,
    attr: AttributeParams,
    raters: HashMap<String, RaterParams>,
    observations: &[Observation],
    cfg: Option<Config>,
) -> SolveSummary {
    let mut eng = RatingEngine::new(n, attr, raters, cfg).expect("Invalid item count");
    eng.ingest(observations);
    eng.solve()
}
