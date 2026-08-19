use super::math::{mad, weighted_median};
use super::*;

// ---------------------------------------------------------------------
//  Ranker: ranks, weights, pair selection, P_flip
// ---------------------------------------------------------------------

pub(super) fn ranks_from_scores(scores: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let n = scores.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));
    let mut rank_of = vec![0usize; n];
    for (pos, &idx) in order.iter().enumerate() {
        rank_of[idx] = pos;
    }
    (order, rank_of)
}

pub(super) fn rank_weights(n: usize, cfg: &Config) -> Vec<f64> {
    let a = cfg.rank_weight_exponent.max(0.0);
    if a == 0.0 {
        return vec![1.0; n];
    }
    (1..=n)
        .map(|k| 1.0 / (k as f64).powf(a))
        .collect::<Vec<f64>>()
}

#[derive(Default)]
pub(super) struct RankCache {
    order: Vec<usize>,
    rank_of: Vec<usize>,
    w_rank: Vec<f64>,
}

pub(super) fn pair_rank_weight(
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

pub(super) fn select_rank_pairs(scores: &[f64], cfg: &Config) -> Vec<(usize, usize)> {
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

pub(super) fn pair_prob_and_flip(
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

pub(super) fn compute_rank_stability(
    scores: &[f64],
    diag_cov: &[f64],
    cfg: &Config,
) -> (f64, f64, f64) {
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

pub(super) fn compute_calibration_evidence(
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
