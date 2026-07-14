//! Template-gain calibration: mixed-wording evidence that calibrates itself.
//!
//! The wording-invariance measurements (2026-07-05) found that the SAME
//! judgement elicited through different wordings comes back with different
//! magnitudes — the fraction wording runs +0.35…+0.92 nats hotter than the
//! ratio wording on frontier models, directions intact. The
//! paradigm-respecting response is not "never mix wordings": it is to treat
//! each template as an instrument with an unknown gain, and fit the gains
//! JOINTLY with the scores — exactly how an experiment calibrates detector
//! channels against each other on shared events.
//!
//! Model: an observation from template t on pair (i, j) reads
//!     m_obs = g_t · (s_i − s_j) + ε
//! with one reference template pinned at g = 1 (the gauge choice — gains
//! are only identified up to a common scale, the same freedom the additive
//! score gauge has). The bilinear fit alternates two closed-form steps:
//! scores given gains (the ordinary least-squares solve on rescaled
//! evidence), then gains given scores (per-template regression through the
//! origin). Each step cannot increase the residual, so the alternation
//! converges; well-connected graphs settle in a handful of rounds.

use std::collections::HashMap;

use serde::Serialize;

use crate::rating_engine::{AttributeParams, Config, Observation, RaterParams, RatingEngine};

/// One mixed-template observation: signed log-ratio toward `i`.
#[derive(Debug, Clone)]
pub struct GainObservation {
    pub i: usize,
    pub j: usize,
    /// Signed log-ratio toward `i`, as elicited (uncorrected).
    pub log_ratio: f64,
    /// Which instrument produced it.
    pub template: String,
}

/// Result of [`solve_with_template_gains`].
#[derive(Debug, Serialize)]
pub struct GainCalibratedSolve {
    /// Latent scores in the calibrated (reference-template) scale.
    pub scores: Vec<f64>,
    /// Fitted multiplicative gain per template (reference pinned at 1.0).
    /// A gain of 1.5 means this wording elicits magnitudes 1.5× hotter
    /// than the reference for the same underlying difference.
    pub gains: Vec<(String, f64)>,
    /// Alternation rounds until the gains moved < 1e-6.
    pub iterations: usize,
    /// Root-mean-square residual of the final fit (nats).
    pub rms_residual: f64,
    /// RMS residual of a naive solve that ignores templates — the price of
    /// NOT calibrating, on the same data.
    pub rms_residual_uncalibrated: f64,
}

fn solve_scores(n: usize, obs: &[(usize, usize, f64)]) -> Option<Vec<f64>> {
    let mut raters = HashMap::new();
    raters.insert("gain".to_string(), RaterParams::default());
    let mut engine = RatingEngine::new(
        n,
        AttributeParams::default(),
        raters,
        Some(Config::default()),
    )
    .ok()?;
    let observations: Vec<Observation> = obs
        .iter()
        .map(|&(i, j, m)| Observation::from_log_ratio_moments(i, j, m, 1.0, "gain", 1.0))
        .collect();
    engine.ingest(&observations);
    Some(engine.solve().scores)
}

fn rms(obs: &[GainObservation], scores: &[f64], gains: &HashMap<String, f64>) -> f64 {
    let sum: f64 = obs
        .iter()
        .map(|o| {
            let g = gains.get(&o.template).copied().unwrap_or(1.0);
            let pred = g * (scores[o.i] - scores[o.j]);
            (o.log_ratio - pred).powi(2)
        })
        .sum();
    (sum / obs.len().max(1) as f64).sqrt()
}

/// Fit scores and per-template gains jointly. `reference` names the
/// template whose gain is pinned to 1 (the scale everything is reported
/// in); it must appear in the observations.
pub fn solve_with_template_gains(
    n: usize,
    obs: &[GainObservation],
    reference: &str,
) -> Option<GainCalibratedSolve> {
    if obs.is_empty() || !obs.iter().any(|o| o.template == reference) {
        return None;
    }
    let mut templates: Vec<String> = obs.iter().map(|o| o.template.clone()).collect();
    templates.sort();
    templates.dedup();

    let mut gains: HashMap<String, f64> = templates.iter().map(|t| (t.clone(), 1.0)).collect();

    // Uncalibrated baseline: one solve pretending every template has g = 1.
    let naive: Vec<(usize, usize, f64)> = obs.iter().map(|o| (o.i, o.j, o.log_ratio)).collect();
    let naive_scores = solve_scores(n, &naive)?;
    let unit_gains: HashMap<String, f64> = templates.iter().map(|t| (t.clone(), 1.0)).collect();
    let rms_residual_uncalibrated = rms(obs, &naive_scores, &unit_gains);

    let mut scores = naive_scores;
    let mut iterations = 0usize;
    for round in 0..50 {
        iterations = round + 1;
        // Scores given gains: rescale each observation into reference
        // units; precision scales with g² (a hotter channel is
        // proportionally noisier per reference-nat, absent better
        // information).
        let rescaled: Vec<(usize, usize, f64)> = obs
            .iter()
            .map(|o| {
                let g = gains[&o.template].max(1e-6);
                (o.i, o.j, o.log_ratio / g)
            })
            .collect();
        scores = solve_scores(n, &rescaled)?;

        // Gains given scores: per-template regression through the origin.
        let mut moved = 0.0f64;
        for t in &templates {
            if t == reference {
                continue;
            }
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for o in obs.iter().filter(|o| &o.template == t) {
                let d = scores[o.i] - scores[o.j];
                num += o.log_ratio * d;
                den += d * d;
            }
            if den > 1e-12 {
                let new_gain = (num / den).max(1e-3);
                moved = moved.max((new_gain - gains[t]).abs());
                gains.insert(t.clone(), new_gain);
            }
        }
        if moved < 1e-6 {
            break;
        }
    }

    let rms_residual = rms(obs, &scores, &gains);
    let mut gains_out: Vec<(String, f64)> =
        templates.iter().map(|t| (t.clone(), gains[t])).collect();
    gains_out.sort_by(|a, b| a.0.cmp(&b.0));

    Some(GainCalibratedSolve {
        scores,
        gains: gains_out,
        iterations,
        rms_residual,
        rms_residual_uncalibrated,
    })
}
