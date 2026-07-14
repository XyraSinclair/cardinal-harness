//! The gain-calibrated solver recovers planted per-template gains and
//! beats the naive mixed solve on its own residual — the estimator-level
//! answer to the measured wording bias (fraction wording runs hotter than
//! ratio wording for the same belief).

use cardinal_harness::gain_calibration::{solve_with_template_gains, GainObservation};

fn planted(
    n: usize,
    gains: &[(&str, f64)],
    noise: f64,
    seed: u64,
) -> (Vec<f64>, Vec<GainObservation>) {
    // Deterministic latents and a cheap LCG for reproducible noise.
    let latents: Vec<f64> = (0..n).map(|i| i as f64 * 0.45).collect();
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to (-1, 1).
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let mut obs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for (template, g) in gains {
                let m = g * (latents[i] - latents[j]) + noise * next();
                obs.push(GainObservation {
                    i,
                    j,
                    log_ratio: m,
                    template: (*template).to_string(),
                });
            }
        }
    }
    (latents, obs)
}

#[test]
fn recovers_planted_gains_and_beats_naive_solve() {
    let (_latents, obs) = planted(
        8,
        &[
            ("canonical_v2", 1.0),
            ("fraction_v1", 1.8),
            ("less_v1", 0.7),
        ],
        0.05,
        7,
    );
    let solve = solve_with_template_gains(8, &obs, "canonical_v2").expect("solve");
    let gain = |t: &str| {
        solve
            .gains
            .iter()
            .find(|(name, _)| name == t)
            .map(|(_, g)| *g)
            .unwrap()
    };
    assert!(
        (gain("canonical_v2") - 1.0).abs() < 1e-9,
        "reference pinned: {}",
        gain("canonical_v2")
    );
    assert!(
        (gain("fraction_v1") - 1.8).abs() < 0.1,
        "hot channel recovered: {}",
        gain("fraction_v1")
    );
    assert!(
        (gain("less_v1") - 0.7).abs() < 0.1,
        "cool channel recovered: {}",
        gain("less_v1")
    );
    assert!(
        solve.rms_residual < solve.rms_residual_uncalibrated / 2.0,
        "calibration must at least halve the residual on gain-mixed data: \
         {} vs naive {}",
        solve.rms_residual,
        solve.rms_residual_uncalibrated
    );
    eprintln!(
        "GAIN-CAL gains recovered: canonical 1.000, fraction {:.3}, less {:.3} \
         · rms {:.4} vs naive {:.4} · {} rounds",
        gain("fraction_v1"),
        gain("less_v1"),
        solve.rms_residual,
        solve.rms_residual_uncalibrated,
        solve.iterations
    );
}

#[test]
fn uniform_gains_change_nothing() {
    // When every template truly has gain 1, calibration must be a no-op:
    // gains stay ~1 and the residual matches the naive solve.
    let (_l, obs) = planted(6, &[("a", 1.0), ("b", 1.0)], 0.05, 11);
    let solve = solve_with_template_gains(6, &obs, "a").expect("solve");
    for (name, g) in &solve.gains {
        assert!(
            (g - 1.0).abs() < 0.05,
            "no phantom gain may appear: {name} = {g}"
        );
    }
    // k−1 free gain parameters legitimately soak a sliver of noise; the
    // honest no-op bound is "no more than a few percent", not exact
    // equality.
    assert!(
        solve.rms_residual_uncalibrated / solve.rms_residual < 1.1,
        "no-op case must not manufacture real improvement: {} vs {}",
        solve.rms_residual,
        solve.rms_residual_uncalibrated
    );
}

#[test]
fn direction_only_disagreement_is_not_absorbed_into_gain() {
    // A template whose DIRECTIONS disagree (sign-flipped half the time) is
    // broken, not miscalibrated: the fitted gain collapses toward zero
    // rather than pretending a scale factor explains it. The floor (1e-3)
    // and the residual make the breakage visible.
    let (_l, mut obs) = planted(6, &[("good", 1.0), ("broken", 1.0)], 0.0, 13);
    let mut k = 0usize;
    for o in obs.iter_mut() {
        if o.template == "broken" {
            if k.is_multiple_of(2) {
                o.log_ratio = -o.log_ratio;
            }
            k += 1;
        }
    }
    let solve = solve_with_template_gains(6, &obs, "good").expect("solve");
    let broken = solve
        .gains
        .iter()
        .find(|(n, _)| n == "broken")
        .map(|(_, g)| *g)
        .unwrap();
    assert!(
        broken < 0.35,
        "sign-incoherent channel must collapse, not calibrate: {broken}"
    );
}
