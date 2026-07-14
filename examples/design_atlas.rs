//! The design atlas: exhaustive enumeration of circulant comparison-graph
//! designs, scored by their invariant profile. Runs offline in
//! milliseconds; the point is to CHOOSE designs from an enumerated space
//! instead of hand-picking strides and hoping.
//!
//! For each n ∈ {8, 10, 12} and each nonempty stride subset
//! S ⊆ {1..⌊n/2⌋}, the circulant graph C_n(S) gets:
//!
//! - m: edges (calls per order per attribute — the budget)
//! - triangles: filled 3-cliques (support for the LOCAL curl diagnostic)
//! - harmonic_dim: cycle_dim − rank(curl) (support for the GLOBAL,
//!   triad-invisible diagnostic — zero means that diagnostic is structurally
//!   dead on this design)
//! - fiedler: algebraic connectivity at unit weights (identifiability;
//!   posterior variance along the worst direction scales as 1/fiedler)
//!
//! Usage: cargo run --example design_atlas

use cardinal_harness::rating_engine::{compute_hodge_split, spectral_diagnostics, Config};

fn circulant(n: usize, strides: &[usize]) -> Vec<(usize, usize)> {
    let mut edges = std::collections::BTreeSet::new();
    for &s in strides {
        for i in 0..n {
            let j = (i + s) % n;
            edges.insert((i.min(j), i.max(j)));
        }
    }
    edges.into_iter().collect()
}

fn main() {
    let cfg = Config::default();
    println!("n strides            m tri harm  fiedler  fiedler/m");
    let mut best_dual: Vec<(f64, String)> = Vec::new();
    for n in [8usize, 10, 12] {
        let max_stride = n / 2;
        for mask in 1u32..(1 << max_stride) {
            let strides: Vec<usize> = (0..max_stride)
                .filter(|b| mask & (1 << b) != 0)
                .map(|b| b + 1)
                .collect();
            let edges = circulant(n, &strides);
            let m = edges.len();
            let ones = vec![1.0; m];
            let Some(spectral) = spectral_diagnostics(&edges, &ones, n, 256) else {
                continue;
            };
            if spectral.fiedler_value < 1e-9 {
                continue; // disconnected
            }
            let hodge = compute_hodge_split(&edges, &ones, &ones, &ones, n, &cfg);
            let line = format!(
                "{n} {:<16} {m:>3} {:>3} {:>4} {:>8.4} {:>9.4}",
                format!("{strides:?}"),
                hodge.filled_triangles,
                hodge.harmonic_dim,
                spectral.fiedler_value,
                spectral.fiedler_value / m as f64,
            );
            println!("{line}");
            // The dual designs: BOTH diagnostics alive on one connected graph.
            if hodge.filled_triangles >= 4 && hodge.harmonic_dim >= 1 {
                best_dual.push((spectral.fiedler_value / m as f64, line));
            }
        }
    }
    best_dual.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\nDUAL-DIAGNOSTIC designs (triangles ≥ 4 AND harmonic_dim ≥ 1), by fiedler/edge:");
    for (_, line) in best_dual.iter().take(12) {
        println!("  {line}");
    }
    if best_dual.is_empty() {
        println!("  NONE — no connected circulant on these sizes supports both diagnostics;");
        println!("  mixed/disjoint-block designs are forced, not a style choice.");
    }
}
