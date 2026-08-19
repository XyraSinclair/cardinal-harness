use super::*;

pub(super) const SCALE_FLOOR: f64 = 1e-6;
pub(super) const MAD_TO_SIGMA: f64 = 1.4826;
pub(super) const MIN_ATTR_UNCERTAINTY_WEIGHT: f64 = 0.1;

/// Maximum batch size for propose_batch to prevent resource exhaustion.
pub(super) const MAX_BATCH_SIZE: usize = 10_000;
/// Max active set size for targeted marginal variance refinement.
pub(super) const MAX_REFINED_ACTIVE: usize = 64;
/// Skip candidate pairs with negligible inversion probability.
pub(super) const MIN_PAIR_PROB: f64 = 1e-4;
/// Floor for soft top-k membership weighting.
pub(super) const MIN_MEMBERSHIP_WEIGHT: f64 = 0.05;
/// Cap planner candidates to avoid O(N^2) explosions.
pub(super) const MAX_PLANNER_CANDIDATES: usize = 50_000;

// ------------------------------------------------------------------
// Math utilities
// ------------------------------------------------------------------

pub(super) fn median(sorted: &[f64]) -> f64 {
    let len = sorted.len();
    let mid = len / 2;
    if len % 2 == 1 {
        sorted[mid]
    } else {
        0.5 * (sorted[mid - 1] + sorted[mid])
    }
}

/// Cap outlier variance contributions using a robust upper fence.
///
/// When combining per-attribute variances into global utility variance,
/// one sparse attribute can dominate if its posterior variance is enormous.
/// This caps each contribution at median + 3*IQR of the entity's contributions,
/// preventing a single under-observed attribute from nuking global confidence.
pub(super) fn robust_capped_sum(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n <= 2 {
        return values.iter().sum();
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1 = sorted[n / 4];
    let q3 = sorted[(3 * n) / 4];
    let iqr = (q3 - q1).max(0.0);
    let fence = q3 + 3.0 * iqr;

    // Also ensure fence is at least the median * 10 to avoid over-capping
    // when most contributions are small but legitimately varied.
    let med = sorted[n / 2];
    let min_fence = med * 10.0;
    let effective_fence = fence.max(min_fence).max(1e-12);

    let mut sum = 0.0;
    for v in values.iter() {
        sum += v.min(effective_fence);
    }
    sum
}

pub(super) fn stddev_population(scores: &[f64], indices: &[usize]) -> f64 {
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
pub(super) fn compute_attribute_scale(scores: &[f64]) -> f64 {
    let mut finite: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter_map(|(i, score)| score.is_finite().then_some(i))
        .collect();

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
#[doc(hidden)]
pub fn compute_attribute_units(scores: &[f64]) -> AttributeUnits {
    let n = scores.len();
    let mut finite: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter_map(|(i, score)| score.is_finite().then_some(i))
        .collect();

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

    for (i, score) in scores.iter().copied().enumerate() {
        if score.is_finite() {
            z[i] = (score - med) / mad_sigma;
            min_norm[i] = (score - min_val) + 1.0;
        }
    }

    (mad, z, min_norm, pct)
}

/// Map tolerated error to a conservative normal quantile.
pub(super) fn beta_from_tolerated_error(tolerated_error: f64) -> f64 {
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

pub(super) fn inversion_prob(delta: f64, var: f64) -> f64 {
    if var <= 0.0 {
        return if delta <= 0.0 { 1.0 } else { 0.0 };
    }
    let z = delta / var.sqrt();
    (1.0 - rating_engine::normal_cdf(z)).clamp(0.0, 1.0)
}
