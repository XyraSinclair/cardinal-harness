//! Elicitation-program equivalence: the free structure of evidence is a
//! commutative monoid of sufficient statistics — two programs that
//! accumulate the same multiset of observations must produce the same
//! posterior, regardless of arrival order or batching. Plus spectral
//! receipts: Foster's theorem as a free correctness invariant and the
//! Fiedler value against hand-computed spectra.
//!
//! The split/merge case tests a claimed BOUNDARY from the theory notes
//! (notes/math-frontier-2026-07-05/theory.md §3): that re-partitioning one
//! observation's weight stops being invariant once Huber clipping is
//! active. The architecture says otherwise — fusion is per-pair BEFORE
//! IRLS, and the λ-weighted mean is linear, so same-pair re-partition
//! should be exactly invariant. Whichever way the machine decides is the
//! pin; the theory note's claim is corrected by the outcome.

use std::collections::HashMap;

use cardinal_harness::rating_engine::{
    spectral_receipts, AttributeParams, Config, Observation, RaterParams, RatingEngine,
};

fn raters() -> HashMap<String, RaterParams> {
    let mut m = HashMap::new();
    m.insert("sim".to_string(), RaterParams::default());
    m
}

fn engine(n: usize) -> RatingEngine {
    RatingEngine::new(n, AttributeParams::default(), raters(), Some(Config::default())).unwrap()
}

/// Deterministic pseudo-random observation set over n items, including a
/// gross outlier so the Huber regime is active.
fn observations(n: usize) -> Vec<Observation> {
    let mut obs = Vec::new();
    let mut state = 99u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for i in 0..n {
        for j in (i + 1)..n {
            let ratio = 1.1 + 2.5 * next();
            obs.push(Observation::new(i, j, ratio, 0.8, "sim", 1.0));
        }
    }
    // One gross outlier: guarantees an active Huber clip at convergence.
    obs.push(Observation::new(0, 1, 25.0, 0.95, "sim", 1.0));
    obs
}

#[test]
fn posterior_is_invariant_to_arrival_order_and_batching() {
    let n = 7;
    let base = observations(n);

    let mut a = engine(n);
    a.ingest(&base);
    let sa = a.solve();

    // Reversed order through the BULK path: since the fuse buckets are
    // ordered (BTreeMap — a fix forced by the judgment packet's
    // byte-identity pin), same multiset → BIT-identical posterior.
    let mut shuffled = base.clone();
    shuffled.reverse();
    let mut b = engine(n);
    b.ingest(&shuffled);
    let sb = b.solve();
    for (x, y) in sa.scores.iter().zip(sb.scores.iter()) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "bulk path: same multiset must give byte-identical posterior"
        );
    }
    assert_eq!(sa.hcr.to_bits(), sb.hcr.to_bits());

    // The INCREMENTAL path makes the weaker, still-true promise: batched
    // arrival agrees to numerical tolerance (its edge order is arrival
    // order by design).
    let mut c = engine(n);
    for chunk in shuffled.chunks(5) {
        c.add_observations(chunk);
    }
    let sc = c.solve();
    for (x, y) in sa.scores.iter().zip(sc.scores.iter()) {
        assert!(
            (x - y).abs() < 1e-9,
            "incremental path: same multiset within tolerance: {x} vs {y}"
        );
    }
}

#[test]
fn same_pair_weight_repartition_is_invariant_even_under_active_huber() {
    // One reps-2 observation vs two reps-1 copies on the same pair, in a
    // configuration where that pair's fused edge is a clipped outlier.
    let n = 5;
    let mut merged = observations(n);
    merged.push(Observation::new(2, 3, 9.0, 0.9, "sim", 2.0));

    let mut split = observations(n);
    split.push(Observation::new(2, 3, 9.0, 0.9, "sim", 1.0));
    split.push(Observation::new(2, 3, 9.0, 0.9, "sim", 1.0));

    let mut a = engine(n);
    a.ingest(&merged);
    let sa = a.solve();
    let mut b = engine(n);
    b.ingest(&split);
    let sb = b.solve();

    for (x, y) in sa.scores.iter().zip(sb.scores.iter()) {
        assert!(
            (x - y).abs() < 1e-9,
            "per-pair fusion precedes IRLS, and the λ-weighted mean is \
             linear: same-pair re-partition must be exactly invariant, \
             Huber or no Huber (this corrects the theory note's claimed \
             boundary): {x} vs {y}"
        );
    }
}

#[test]
fn foster_theorem_holds_on_hand_computed_graphs() {
    // Unweighted path P3: R_eff = 1 per edge, sum 2 = n − 1; Fiedler = 1.
    let s = spectral_receipts(&[(0, 1), (1, 2)], &[1.0, 1.0], 3, 256).unwrap();
    assert!(s.foster_residual < 1e-9, "{s:?}");
    assert!((s.fiedler_value - 1.0).abs() < 1e-9, "{s:?}");

    // Weighted triangle, w = 2: Fiedler = 6; Foster sum = 3·2·(1/3·1/2·2)…
    // exactly n − 1 = 2 by the theorem regardless of weights.
    let s = spectral_receipts(&[(0, 1), (1, 2), (0, 2)], &[2.0, 2.0, 2.0], 3, 256).unwrap();
    assert!(s.foster_residual < 1e-9, "{s:?}");
    assert!((s.fiedler_value - 6.0).abs() < 1e-9, "{s:?}");

    // Two disconnected edges: components = 2, expected sum = 4 − 2 = 2,
    // Fiedler = smallest NONZERO eigenvalue = 2·w = 2.
    let s = spectral_receipts(&[(0, 1), (2, 3)], &[1.0, 1.0], 4, 256).unwrap();
    assert!(s.foster_residual < 1e-9, "{s:?}");
    assert!((s.expected_resistance_sum - 2.0).abs() < 1e-12, "{s:?}");
    assert!((s.fiedler_value - 2.0).abs() < 1e-9, "{s:?}");
}

#[test]
fn solver_populates_spectral_receipts_and_foster_holds_end_to_end() {
    let n = 7;
    let mut e = engine(n);
    e.ingest(&observations(n));
    let summary = e.solve();
    let spectral = summary.spectral.expect("small graph must get receipts");
    assert!(
        spectral.foster_residual < 1e-6,
        "Foster's theorem is exact; a residual means broken linear algebra: {spectral:?}"
    );
    assert!(
        spectral.fiedler_value > 0.0,
        "connected comparison graph must have positive algebraic connectivity: {spectral:?}"
    );
}
