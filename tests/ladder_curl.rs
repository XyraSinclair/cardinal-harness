//! Is "quantization curl" a property of discretization in general, or of
//! THIS ladder's non-constant log step?
//!
//! The repo ladder `[1.0, 1.05, ..., 26.0]` has monotonically increasing
//! log-steps (0.049 → 0.368), so ln(r_a) + ln(r_b) ≠ ln(r_c) even for a
//! perfectly transitive judge — chained comparisons land between rungs and
//! the residual shows up as Hodge curl. A constant-log-step ladder
//! (r_k = 26^(k/16), step ≈ 0.204) makes rung arithmetic exact whenever
//! the true log-ratios sit near rungs. This test measures both floors on
//! the same planted-transitive judge. Receipt printed either way.

use std::collections::HashMap;

use cardinal_harness::rating_engine::{
    AttributeParams, Config, Observation, RaterParams, RatingEngine,
};

/// The repo's ladder (src/prompts.rs).
const REPO_LADDER: [f64; 17] = [
    1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.1, 2.5, 3.1, 3.9, 5.1, 6.8, 9.2, 12.7, 18.0, 26.0,
];

fn geometric_ladder() -> Vec<f64> {
    (0..17).map(|k| 26.0f64.powf(k as f64 / 16.0)).collect()
}

fn quantize(ratio: f64, ladder: &[f64]) -> f64 {
    *ladder
        .iter()
        .min_by(|a, b| {
            (a.ln() - ratio.ln())
                .abs()
                .partial_cmp(&(b.ln() - ratio.ln()).abs())
                .unwrap()
        })
        .unwrap()
}

/// Planted transitive judge over n items with latent gaps, all-pairs
/// comparisons, log-ratios quantized to the given ladder. Returns hcr.
fn hcr_with_ladder(ladder: &[f64]) -> f64 {
    // Latents chosen so pairwise true ratios span the ladder's range and do
    // NOT sit exactly on rungs of either ladder.
    let latents: [f64; 8] = [0.00, 0.31, 0.74, 1.13, 1.62, 2.05, 2.51, 3.10];
    let n = latents.len();
    let mut raters = HashMap::new();
    raters.insert("sim".to_string(), RaterParams::default());
    let mut engine =
        RatingEngine::new(n, AttributeParams::default(), raters, Some(Config::default())).unwrap();
    let mut obs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let true_ratio = (latents[i] - latents[j]).abs().exp();
            let q = quantize(true_ratio, ladder);
            let (hi, lo) = if latents[i] >= latents[j] {
                (i, j)
            } else {
                (j, i)
            };
            obs.push(Observation::new(hi, lo, q, 0.9, "sim", 1.0));
        }
    }
    engine.ingest(&obs);
    engine.solve().hcr
}

#[test]
fn geometric_ladder_reduces_quantization_curl() {
    let repo_hcr = hcr_with_ladder(&REPO_LADDER);
    let geo = geometric_ladder();
    let geo_hcr = hcr_with_ladder(&geo);
    // The receipt, either way:
    eprintln!("LADDER-CURL repo_ladder hcr = {repo_hcr:.5}");
    eprintln!("LADDER-CURL geometric     hcr = {geo_hcr:.5}");
    eprintln!(
        "LADDER-CURL ratio = {:.2}x (2026-07-05 measured: repo 0.00198, geometric \
         0.00155, 1.28x — the uneven log-step is real but THIRD-ORDER: both \
         full-ladder floors are ~0.002, two orders below the ~0.13 seen from a \
         judge that uses only two rungs. Quantization curl is dominated by how \
         few rungs the judge actually uses, not by rung spacing.)",
        repo_hcr / geo_hcr.max(1e-12)
    );
    assert!(
        geo_hcr < repo_hcr,
        "constant-log-step ladder must not increase quantization curl: \
         geo {geo_hcr:.5} vs repo {repo_hcr:.5}"
    );
    // Two-sided pin so silent regressions in either direction surface.
    assert!(repo_hcr > 1e-4, "repo ladder floor vanished: {repo_hcr:.6}");
    assert!(repo_hcr < 0.05, "repo ladder floor exploded: {repo_hcr:.6}");
}
