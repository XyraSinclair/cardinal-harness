//! Repeat-draw pooling with a heterogeneity floor (DerSimonian–Laird).
//!
//! Model (two-level, the de Finetti-compatible hierarchy for repeat
//! elicitation): draw t on pair p reads
//!
//!   m_{p,t} = Δ_p + b_p + ε_{p,t},   b_p ~ (0, σ_b²),  ε ~ (0, σ_w²)
//!
//! where Δ_p is the score difference the graph can explain and b_p is the
//! pair's STRUCTURAL offset — the per-pair component of frustration that
//! more sampling can never remove. Naive pooling weights a pair's mean by
//! k/σ_w², so a heavily-resampled frustrated pair acquires unbounded
//! precision and drags the solve toward its bias. The correct pooled
//! variance has a floor:
//!
//!   Var(m̄_p) = σ_b² + σ_w²/k_p   →   precision ≤ 1/σ_b², no matter k.
//!
//! Estimation is the DerSimonian–Laird moment method adapted to the
//! graph: (1) pool within-pair variances for σ_w²; (2) solve naively and
//! form Cochran's Q from the weighted graph residuals; (3)
//! σ_b² = max(0, (Q − df)/c) with df = the cycle dimension (the residual
//! degrees of freedom of a graph fit) and c = Σw − Σw²/Σw; (4) re-solve
//! with floored variances. σ_b² estimated here IS the per-pair reading of
//! the frustration energy the Hodge machinery measures at the field level
//! — the two views must agree in order of magnitude, and the tests plant
//! both regimes.

use std::collections::HashMap;

use serde::Serialize;

use crate::rating_engine::{AttributeParams, Config, Observation, RaterParams, RatingEngine};

/// Repeat draws for one pair: signed log-ratios toward `i`.
#[derive(Debug, Clone)]
pub struct RepeatDraws {
    pub i: usize,
    pub j: usize,
    pub draws: Vec<f64>,
}

/// Result of [`pool_repeats`].
#[derive(Debug, Serialize)]
pub struct PooledSolve {
    /// Latent scores from the heterogeneity-floored solve.
    pub scores: Vec<f64>,
    /// Latent scores a naive k/σ_w² pooling would have produced.
    pub scores_naive: Vec<f64>,
    /// Pooled within-pair (per-draw) variance.
    pub sigma_w2: f64,
    /// DL between-pair heterogeneity — the sampling-proof variance floor.
    /// Zero when the graph explains the pair means to within draw noise.
    pub sigma_b2: f64,
    /// Cochran's Q from the naive solve's weighted residuals.
    pub q_statistic: f64,
    /// Residual degrees of freedom used (cycle dimension of the graph).
    pub degrees_of_freedom: usize,
}

fn solve_with_vars(
    n: usize,
    means: &[(usize, usize, f64)],
    variances: &[f64],
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut raters = HashMap::new();
    raters.insert("pool".to_string(), RaterParams::default());
    let mut engine = RatingEngine::new(
        n,
        AttributeParams::default(),
        raters,
        Some(Config::default()),
    )
    .ok()?;
    let obs: Vec<Observation> = means
        .iter()
        .zip(variances.iter())
        .map(|(&(i, j, m), &v)| {
            Observation::from_log_ratio_moments(i, j, m, v.max(1e-9), "pool", 1.0)
        })
        .collect();
    engine.ingest(&obs);
    let summary = engine.solve();
    Some((summary.scores, summary.residuals, vec![summary.hcr]))
}

/// Pool repeat draws with the DerSimonian–Laird heterogeneity floor and
/// solve. Requires at least one pair with ≥ 2 draws (σ_w² is otherwise
/// unidentified).
pub fn pool_repeats(n: usize, pairs: &[RepeatDraws]) -> Option<PooledSolve> {
    if pairs.is_empty() {
        return None;
    }
    // σ_w²: pooled within-pair variance.
    let mut ss = 0.0f64;
    let mut dof = 0usize;
    for p in pairs {
        if p.draws.len() >= 2 {
            let k = p.draws.len() as f64;
            let mean = p.draws.iter().sum::<f64>() / k;
            ss += p.draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>();
            dof += p.draws.len() - 1;
        }
    }
    if dof == 0 {
        return None;
    }
    let sigma_w2 = (ss / dof as f64).max(1e-12);

    let means: Vec<(usize, usize, f64)> = pairs
        .iter()
        .map(|p| {
            (
                p.i,
                p.j,
                p.draws.iter().sum::<f64>() / p.draws.len().max(1) as f64,
            )
        })
        .collect();
    let naive_vars: Vec<f64> = pairs
        .iter()
        .map(|p| sigma_w2 / p.draws.len().max(1) as f64)
        .collect();

    let (scores_naive, residuals, _) = solve_with_vars(n, &means, &naive_vars)?;

    // Cochran's Q over the graph fit; residual df = cycle dimension.
    let weights: Vec<f64> = naive_vars.iter().map(|v| 1.0 / v).collect();
    let q_statistic: f64 = weights
        .iter()
        .zip(residuals.iter())
        .map(|(w, r)| w * r * r)
        .sum();
    let vertices: std::collections::HashSet<usize> =
        pairs.iter().flat_map(|p| [p.i, p.j]).collect();
    // Components via union-find over the touched vertices.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        root
    }
    for p in pairs {
        let (a, b) = (find(&mut parent, p.i), find(&mut parent, p.j));
        if a != b {
            parent[a] = b;
        }
    }
    let components = vertices
        .iter()
        .filter(|&&v| find(&mut parent, v) == v)
        .count()
        .max(1);
    let degrees_of_freedom = (pairs.len() + components).saturating_sub(vertices.len());

    let sum_w: f64 = weights.iter().sum();
    let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
    let c = sum_w - sum_w2 / sum_w.max(1e-12);
    let sigma_b2 = if c > 0.0 {
        ((q_statistic - degrees_of_freedom as f64) / c).max(0.0)
    } else {
        0.0
    };

    let floored_vars: Vec<f64> = pairs
        .iter()
        .map(|p| sigma_b2 + sigma_w2 / p.draws.len().max(1) as f64)
        .collect();
    let (scores, _, _) = solve_with_vars(n, &means, &floored_vars)?;

    Some(PooledSolve {
        scores,
        scores_naive,
        sigma_w2,
        sigma_b2,
        q_statistic,
        degrees_of_freedom,
    })
}
