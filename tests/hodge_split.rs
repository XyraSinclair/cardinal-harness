//! The full Hodge split: cyclic residual = triangle-auditable curl ⊕
//! triad-invisible harmonic, w-orthogonally. Two planted extremes sit at
//! opposite ends of the same statistic while hcr (the un-split total)
//! looks identical — plus the Pythagoras invariant and the design pin
//! that the JCB graph cannot host harmonic disagreement at all.
//!
//! Test inputs are PURE CYCLE FLOWS: divergence-free μ makes the gradient
//! projection zero, so residual = μ exactly and every energy is
//! hand-checkable.

use cardinal_harness::rating_engine::{compute_hodge_split, Config};

fn cfg() -> Config {
    Config::default()
}

#[test]
fn filled_triangle_curl_is_entirely_local() {
    // Pure cycle flow around a FILLED triangle 0→1→2→0. Canonical edges
    // (0,1), (1,2), (0,2): the cycle traverses (0,2) backwards, so the
    // divergence-free flow is (x, x, −x).
    let endpoints = [(0, 1), (1, 2), (0, 2)];
    let flow = [0.4, 0.4, -0.4];
    let lam = [1.0, 1.0, 1.0];
    let split = compute_hodge_split(&endpoints, &flow, &flow, &lam, 3, &cfg());
    assert_eq!(split.filled_triangles, 1);
    assert_eq!(split.harmonic_dim, 0, "one cycle, one triangle: no room");
    assert!(
        split.harmonic_frac < 1e-12,
        "triangle curl must be fully local: {split:?}"
    );
    // Pure cycle flow: ALL energy is cyclic and all of it local.
    assert!(
        (split.local_curl_frac - 1.0).abs() < 1e-9,
        "hcr of a pure cycle flow is 1, all local: {split:?}"
    );
}

#[test]
fn chordless_four_cycle_is_entirely_harmonic() {
    // Pure cycle flow around 0→1→2→3→0 with NO chords judged. Canonical
    // edges (0,1), (1,2), (2,3), (0,3); the loop traverses (0,3) backwards.
    let endpoints = [(0, 1), (1, 2), (2, 3), (0, 3)];
    let flow = [0.4, 0.4, 0.4, -0.4];
    let lam = [1.0, 1.0, 1.0, 1.0];
    let split = compute_hodge_split(&endpoints, &flow, &flow, &lam, 4, &cfg());
    assert_eq!(split.filled_triangles, 0);
    assert_eq!(split.harmonic_dim, 1, "one unfillable cycle");
    assert!(
        split.local_curl_frac < 1e-12,
        "no triangles → nothing auditable: {split:?}"
    );
    assert!(
        (split.harmonic_frac - 1.0).abs() < 1e-9,
        "the same inconsistency is now invisible to every triad audit: {split:?}"
    );
}

#[test]
fn mixed_graph_obeys_pythagoras_and_separates_the_parts() {
    // Disjoint triangle {0,1,2} and chordless square {3,4,5,6}, each
    // carrying a pure cycle flow of different magnitude. The split must
    // put the triangle's energy in local, the square's in harmonic, and
    // the two fractions must sum to the total cyclic fraction (= 1 here).
    let endpoints = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (5, 6), (3, 6)];
    let flow = [0.3, 0.3, -0.3, 0.7, 0.7, 0.7, -0.7];
    let lam = [1.0; 7];
    let split = compute_hodge_split(&endpoints, &flow, &flow, &lam, 7, &cfg());
    assert_eq!(split.filled_triangles, 1);
    assert_eq!(split.harmonic_dim, 1);
    // Energies: triangle 3·0.09 = 0.27, square 4·0.49 = 1.96, total 2.23.
    let total = 3.0 * 0.09 + 4.0 * 0.49;
    assert!(
        (split.local_curl_frac - 3.0 * 0.09 / total).abs() < 1e-9,
        "{split:?}"
    );
    assert!(
        (split.harmonic_frac - 4.0 * 0.49 / total).abs() < 1e-9,
        "{split:?}"
    );
    assert!(
        (split.local_curl_frac + split.harmonic_frac - 1.0).abs() < 1e-9,
        "Pythagoras: parts must sum to the whole: {split:?}"
    );
}

#[test]
fn solver_summary_split_sums_to_hcr_on_a_planted_cyclic_judge() {
    // End-to-end through the real solver: rock-paper-scissors flows on a
    // triangle-rich graph. The SolveSummary invariant
    // local + harmonic ≈ hcr must hold on real solver residuals (which
    // are only approximately w-orthogonal to gradients after IRLS
    // reweighting — hence the honest tolerance).
    use cardinal_harness::rating_engine::{
        AttributeParams, Observation, RaterParams, RatingEngine,
    };
    use std::collections::HashMap;

    let n = 6;
    let mut raters = HashMap::new();
    raters.insert("sim".to_string(), RaterParams::default());
    let mut engine = RatingEngine::new(
        n,
        AttributeParams::default(),
        raters,
        Some(Config::default()),
    )
    .unwrap();
    // i beats i+1 and i+2 (mod 6) by 2.5×: strongly cyclic.
    let mut obs = Vec::new();
    for i in 0..n {
        for stride in [1usize, 2] {
            let j = (i + stride) % n;
            obs.push(Observation::new(i, j, 2.5, 0.9, "sim", 1.0));
        }
    }
    engine.ingest(&obs);
    let summary = engine.solve();
    let split = summary.hodge;
    assert!(summary.hcr > 0.3, "planted cycles must show: {summary:?}");
    assert!(
        (split.local_curl_frac + split.harmonic_frac - summary.hcr).abs() < 0.02,
        "split must decompose hcr: local {} + harmonic {} vs hcr {}",
        split.local_curl_frac,
        split.harmonic_frac,
        summary.hcr
    );
    // Stride-{1,2} on a 6-ring is triangle-rich (i, i+1, i+2 all joined):
    // the curl should be dominantly local/auditable.
    assert!(split.local_curl_frac > split.harmonic_frac, "{split:?}");
}

#[test]
fn atlas_winner_c8_134_hosts_both_diagnostics_at_the_same_budget() {
    // The design-atlas headline (docs/DESIGN_ATLAS.md): swapping stride 2
    // for stride 3 in the JCB graph keeps 20 edges, 16 triangles, and
    // Fiedler 4.0, while raising harmonic_dim from 0 to 1 — both curl
    // diagnostics alive on ONE connected graph. Pinned so the recommended
    // v2 core design's profile cannot silently drift.
    let n = 8usize;
    let mut edges = std::collections::BTreeSet::new();
    for s in [1usize, 3, 4] {
        for i in 0..n {
            let j = (i + s) % n;
            edges.insert((i.min(j), i.max(j)));
        }
    }
    let edges: Vec<(usize, usize)> = edges.into_iter().collect();
    assert_eq!(edges.len(), 20);
    let ones = vec![1.0; edges.len()];
    let split = compute_hodge_split(&edges, &ones, &ones, &ones, n, &cfg());
    assert_eq!(split.filled_triangles, 16);
    assert_eq!(
        split.harmonic_dim, 1,
        "the strictly-better design: {split:?}"
    );
    let spectral =
        cardinal_harness::rating_engine::spectral_diagnostics(&edges, &ones, n, 256).unwrap();
    assert!((spectral.fiedler_value - 4.0).abs() < 1e-9, "{spectral:?}");
}

#[test]
fn jcb_graph_design_cannot_host_harmonic_disagreement() {
    // The benchmark's stride-{1,2,4} graph: 20 edges, cycle dim 13,
    // 16 triangles whose curl space has rank exactly 13. Pinned here so
    // any future corpus/pair-design change that silently loses this
    // property (or gains it) surfaces. Values are irrelevant to the
    // dimension computation; use a generic flow.
    let pairs = cardinal_harness::rerank::core_pairs();
    let m = pairs.len();
    let flow: Vec<f64> = (0..m).map(|k| ((k * 7 % 11) as f64 - 5.0) / 10.0).collect();
    let lam = vec![1.0; m];
    let split = compute_hodge_split(&pairs, &flow, &flow, &lam, 8, &cfg());
    assert_eq!(split.filled_triangles, 16);
    assert_eq!(
        split.harmonic_dim, 0,
        "triangles span the whole cycle space on this design — harmonic \
         diagnostics on the JCB graph are zero BY CONSTRUCTION, not by judge \
         virtue: {split:?}"
    );
}
