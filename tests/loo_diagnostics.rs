//! Leave-one-out consistency: every judgement audited by the rest of the
//! graph. Sum-over-histories with correct studentization — planted
//! corruption flagged at |z| > 3, clean data unflagged, the leverage
//! trace tied to Foster's theorem, and bridges honestly unaudited.

use std::collections::HashMap;

use cardinal_harness::rating_engine::{
    AttributeParams, Config, Observation, RaterParams, RatingEngine,
};

fn engine(n: usize) -> RatingEngine {
    let mut raters = HashMap::new();
    raters.insert("sim".to_string(), RaterParams::default());
    RatingEngine::new(
        n,
        AttributeParams::default(),
        raters,
        Some(Config::default()),
    )
    .unwrap()
}

fn all_pairs_obs(latents: &[f64], noise_seed: u64, noise: f64) -> Vec<Observation> {
    let mut state = noise_seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let n = latents.len();
    let mut obs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let m = latents[i] - latents[j] + noise * next();
            obs.push(Observation::new(i, j, m.exp(), 0.9, "sim", 1.0));
        }
    }
    obs
}

#[test]
fn planted_corruption_is_flagged_and_clean_edges_are_not() {
    let latents = [0.0, 0.4, 0.9, 1.3, 1.8, 2.4];
    let mut obs = all_pairs_obs(&latents, 7, 0.02);
    // Corrupt one judgement hard: pair (1,4) reversed and inflated.
    let corrupt_index = obs
        .iter()
        .position(|o| o.i == 1 && o.j == 4)
        .expect("pair present");
    obs[corrupt_index].ratio = ((latents[4] - latents[1]) + 1.5).exp();

    let mut e = engine(6);
    e.ingest(&obs);
    let summary = e.solve();
    let loo = summary.loo.expect("small graph gets LOO");
    assert!(
        loo.flagged.len() == 1,
        "exactly the corrupted judgement flagged: {:?} (max |z| {})",
        loo.flagged,
        loo.max_abs_z
    );
    assert!(loo.max_abs_z > 3.0, "corruption must exceed 3σ: {loo:?}");
    assert_eq!(loo.bridges, 0, "all-pairs graph has no bridges");
}

#[test]
fn clean_data_produces_no_flags() {
    let latents = [0.0, 0.5, 1.0, 1.6, 2.1];
    let mut e = engine(5);
    e.ingest(&all_pairs_obs(&latents, 11, 0.02));
    let summary = e.solve();
    let loo = summary.loo.expect("loo");
    assert!(
        loo.flagged.is_empty(),
        "no corruption planted, none may be found: {:?}",
        loo.flagged
    );
}

#[test]
fn leverage_trace_equals_model_degrees_of_freedom() {
    // Σ h_e = n − c exactly — the hat-matrix trace IS Foster's theorem.
    let latents = [0.0, 0.7, 1.2, 1.9];
    let mut e = engine(4);
    e.ingest(&all_pairs_obs(&latents, 13, 0.05));
    let summary = e.solve();
    let spectral = summary.spectral.expect("spectral");
    let trace: f64 = spectral.edge_leverage.iter().sum();
    assert!((trace - 3.0).abs() < 1e-6, "trace(H) = n − c = 3: {trace}");
}

#[test]
fn a_bridge_is_counted_as_unaudited_not_scored() {
    // Path graph: every edge is a bridge — no judgement has a second
    // opinion, and the diagnostic must say so rather than invent one.
    let mut e = engine(4);
    let obs = vec![
        Observation::new(0, 1, 1.5, 0.9, "sim", 1.0),
        Observation::new(1, 2, 1.5, 0.9, "sim", 1.0),
        Observation::new(2, 3, 1.5, 0.9, "sim", 1.0),
    ];
    e.ingest(&obs);
    let summary = e.solve();
    let loo = summary.loo.expect("loo");
    assert_eq!(loo.bridges, 3, "{loo:?}");
    assert!(loo.flagged.is_empty());
    assert!(loo.z.iter().all(Option::is_none), "{loo:?}");
}
