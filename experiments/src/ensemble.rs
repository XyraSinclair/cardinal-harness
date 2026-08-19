//! Judge portfolio theory: the geometry of a model ensemble and the
//! minimum-variance weighting of judges under CORRELATED errors and cost.
//!
//! Setup. J judges each produce a latent vector over the same entities
//! under the same attribute. Z-score each vector (quotients out per-judge
//! gain and gauge); the correlation matrix R of the z-scored vectors is
//! the measured geometry: d(i,j) = √(2(1−ρ_ij)) embeds judges in a
//! metric space.
//!
//! One-factor model: judge_i = l_i·f + e_i, consensus f, loadings l,
//! error covariance Ψ = R − llᵀ (FULL matrix — clones share error, and
//! that correlation must be kept, not diagonalized away). Then:
//!
//! - **loadings by Spearman's triad identity** (1904):
//!   l_i² = ρ_ij·ρ_ik / ρ_jk, taken as a median over triads for
//!   robustness to a minority of correlated-error pairs. (Two failed
//!   designs preceded this: the raw correlation's participation ratio
//!   counts opinion dimensions, ≈ 1 for any consensual ensemble; and
//!   √λ₁·v₁ loadings are inconsistent across ensemble sizes. Both were
//!   caught by this module's own planted tests.)
//! - **total information** I = lᵀΨ⁻¹l — the GLS precision on the
//!   consensus. Clones collapse through Ψ's off-diagonal block; a
//!   noisier-but-independent judge strictly adds (Schur complement > 0):
//!   the diversification theorem. Information is a portfolio quantity.
//! - **optimal weights** w ∝ Ψ⁻¹l — minimum-variance unbiased under the
//!   measured error covariance. THE answer to "how should a fixed roster
//!   of models be weighted": by loading, discounted by shared error.
//! - **marginal information** ΔI_i = I − I₋ᵢ and ΔI_i/cost_i — the
//!   budgeted-portfolio ranking: which judges earn their place.
//! - **effective error sources**: participation ratio of Ψ normalized to
//!   unit diagonal — how many independent error channels the roster has.
//!
//! Small-J caveat carried in the diagnostics: triad loadings need J ≥ 3 and
//! are only clone-robust when honest triads are the majority (J ≥ 6 for
//! one clone pair); J = 2 falls back to the symmetric √ρ and is labeled
//! by `n_entities`/J as everything else is.

use nalgebra::{DMatrix, DVector};
use serde::Serialize;

/// One judge's portfolio diagnostics.
#[derive(Debug, Clone, Serialize)]
pub struct JudgePortfolioEntry {
    pub judge: String,
    /// Loading on the consensus factor (triad-median estimate).
    pub loading: f64,
    /// Idiosyncratic variance ψ_i = 1 − l_i² (diagonal of Ψ).
    pub idiosyncratic: f64,
    /// Minimum-variance weight ∝ (Ψ⁻¹l)_i, normalized to sum 1.
    pub weight: f64,
    /// Marginal information ΔI = I − I₋ᵢ (what this judge adds to the
    /// roster's precision on the consensus).
    pub marginal_information: f64,
    /// ΔI per dollar (None when no cost given).
    pub information_per_dollar: Option<f64>,
}

/// Result of [`judge_geometry`].
#[derive(Debug, Serialize)]
pub struct JudgeGeometry {
    pub judges: Vec<JudgePortfolioEntry>,
    /// Correlation matrix of z-scored latents (row-major, J×J).
    pub correlation: Vec<Vec<f64>>,
    /// λ₁/Σλ of R — how one-dimensional the ensemble's opinion is.
    pub consensus_share: f64,
    /// Participation ratio of the unit-normalized error covariance Ψ —
    /// the number of independent error channels. Clones share one.
    pub effective_error_sources: f64,
    /// I = lᵀΨ⁻¹l — total precision on the consensus.
    pub total_information: f64,
    /// Judges ranked by marginal information per dollar (indices into
    /// `judges`); by raw marginal information when no costs given.
    pub portfolio_order: Vec<usize>,
    pub n_entities: usize,
}

fn z_score(v: &[f64]) -> Option<Vec<f64>> {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    if var <= 1e-15 {
        return None;
    }
    let sd = var.sqrt();
    Some(v.iter().map(|x| (x - mean) / sd).collect())
}

fn triad_loadings(r: &DMatrix<f64>) -> Vec<f64> {
    let j = r.nrows();
    if j == 2 {
        let l = r[(0, 1)].max(0.0).sqrt();
        return vec![l, l];
    }
    (0..j)
        .map(|i| {
            let mut estimates = Vec::new();
            for a in 0..j {
                if a == i {
                    continue;
                }
                for b in (a + 1)..j {
                    if b == i {
                        continue;
                    }
                    let denom = r[(a, b)];
                    if denom.abs() < 0.05 {
                        continue;
                    }
                    let l2 = r[(i, a)] * r[(i, b)] / denom;
                    if l2.is_finite() {
                        estimates.push(l2.clamp(0.0, 0.998));
                    }
                }
            }
            if estimates.is_empty() {
                return 0.0;
            }
            estimates.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            estimates[estimates.len() / 2].sqrt()
        })
        .collect()
}

fn information(psi: &DMatrix<f64>, loadings: &[f64]) -> Option<f64> {
    let j = psi.nrows();
    let ridge = DMatrix::<f64>::identity(j, j) * 1e-6;
    let inv = (psi.clone() + ridge).try_inverse()?;
    let l = DVector::<f64>::from_column_slice(loadings);
    Some((l.transpose() * &inv * &l)[(0, 0)])
}

/// Compute the ensemble geometry from per-judge latent vectors over the
/// same entities. `costs` (dollars per equivalent run) enables the
/// portfolio ranking. Requires ≥ 2 judges with nonzero variance and
/// ≥ 3 entities.
pub fn judge_geometry(
    names: &[String],
    latents: &[Vec<f64>],
    costs: Option<&[f64]>,
) -> Option<JudgeGeometry> {
    let j = latents.len();
    if j < 2 || names.len() != j {
        return None;
    }
    let n = latents[0].len();
    if n < 3 || latents.iter().any(|l| l.len() != n) {
        return None;
    }
    let z: Vec<Vec<f64>> = latents.iter().map(|l| z_score(l)).collect::<Option<_>>()?;

    let mut r = DMatrix::<f64>::identity(j, j);
    for a in 0..j {
        for b in (a + 1)..j {
            let rho = z[a]
                .iter()
                .zip(z[b].iter())
                .map(|(x, y)| x * y)
                .sum::<f64>()
                / n as f64;
            r[(a, b)] = rho;
            r[(b, a)] = rho;
        }
    }

    let eig = nalgebra::SymmetricEigen::new(r.clone());
    let lambda_max = eig
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);
    let sum: f64 = eig.eigenvalues.iter().map(|l| l.max(0.0)).sum();
    let consensus_share = if sum > 0.0 { lambda_max / sum } else { 0.0 };

    let loadings = triad_loadings(&r);

    // Full error covariance Ψ = R − llᵀ, floored on the diagonal, then
    // PROJECTED to the PSD cone (eigenvalue clipping, Higham-style).
    // With real ensembles (ρ up to 0.98, small n) the raw Ψ leaves the
    // cone from sampling noise and lᵀΨ⁻¹l goes negative — a live run
    // caught total information −1.94 before this projection existed. A
    // precision is nonnegative by definition; the projection makes the
    // estimator honor that.
    let mut psi = r.clone();
    for a in 0..j {
        for b in 0..j {
            psi[(a, b)] -= loadings[a] * loadings[b];
        }
        psi[(a, a)] = psi[(a, a)].max(1e-3);
    }
    let psi = {
        let eig = nalgebra::SymmetricEigen::new(psi);
        let clipped = DVector::<f64>::from_iterator(j, eig.eigenvalues.iter().map(|l| l.max(1e-3)));
        &eig.eigenvectors * DMatrix::from_diagonal(&clipped) * eig.eigenvectors.transpose()
    };

    let total_information = information(&psi, &loadings)?;

    // Optimal weights w ∝ Ψ⁻¹ l.
    let ridge = DMatrix::<f64>::identity(j, j) * 1e-6;
    let psi_inv = (psi.clone() + ridge).try_inverse()?;
    let l_vec = DVector::<f64>::from_column_slice(&loadings);
    let raw_w = &psi_inv * &l_vec;
    let w_sum: f64 = raw_w.iter().sum();

    // Marginal information: drop each judge, recompute.
    let mut marginal = Vec::with_capacity(j);
    for drop in 0..j {
        let keep: Vec<usize> = (0..j).filter(|&k| k != drop).collect();
        let mut sub = DMatrix::<f64>::zeros(j - 1, j - 1);
        for (p, &a) in keep.iter().enumerate() {
            for (q, &b) in keep.iter().enumerate() {
                sub[(p, q)] = psi[(a, b)];
            }
        }
        let sub_l: Vec<f64> = keep.iter().map(|&k| loadings[k]).collect();
        let without = information(&sub, &sub_l).unwrap_or(0.0);
        marginal.push((total_information - without).max(0.0));
    }

    // Effective error sources: PR of Ψ normalized to unit diagonal.
    let mut psi_norm = DMatrix::<f64>::identity(j, j);
    for a in 0..j {
        for b in 0..j {
            let d = (psi[(a, a)] * psi[(b, b)]).sqrt();
            psi_norm[(a, b)] = (psi[(a, b)] / d).clamp(-1.0, 1.0);
        }
        psi_norm[(a, a)] = 1.0;
    }
    let psi_eig = nalgebra::SymmetricEigen::new(psi_norm);
    let ps: f64 = psi_eig.eigenvalues.iter().map(|l| l.max(0.0)).sum();
    let ps2: f64 = psi_eig.eigenvalues.iter().map(|l| l.max(0.0).powi(2)).sum();
    let effective_error_sources = if ps2 > 0.0 { ps * ps / ps2 } else { 0.0 };

    let mut judges = Vec::with_capacity(j);
    for i in 0..j {
        judges.push(JudgePortfolioEntry {
            judge: names[i].clone(),
            loading: loadings[i],
            idiosyncratic: (1.0 - loadings[i] * loadings[i]).max(0.0),
            weight: if w_sum.abs() > 1e-12 {
                raw_w[i] / w_sum
            } else {
                1.0 / j as f64
            },
            marginal_information: marginal[i],
            information_per_dollar: costs.map(|c| marginal[i] / c[i].max(1e-12)),
        });
    }
    let mut portfolio_order: Vec<usize> = (0..j).collect();
    portfolio_order.sort_by(|&x, &y| {
        let key = |k: usize| {
            judges[k]
                .information_per_dollar
                .unwrap_or(judges[k].marginal_information)
        };
        key(y)
            .partial_cmp(&key(x))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Some(JudgeGeometry {
        judges,
        correlation: (0..j)
            .map(|a| (0..j).map(|b| r[(a, b)]).collect())
            .collect(),
        consensus_share,
        effective_error_sources,
        total_information,
        portfolio_order,
        n_entities: n,
    })
}
