//! Stochastic transitivity: the consistency hierarchy on choice
//! PROBABILITIES, from repeat draws.
//!
//! The Hodge machinery audits mean log-ratios; a judge can telescope its
//! means exactly (zero curl) while its draw-to-draw DIRECTIONS are
//! incoherent — probabilistic intransitivity that no mean-level diagnostic
//! can see. The classical hierarchy, for a triad oriented so both
//! premises hold (p_ij ≥ ½, p_jk ≥ ½):
//!
//!   WST (weak):     p_ik ≥ ½
//!   MST (moderate): p_ik ≥ min(p_ij, p_jk)
//!   SST (strong):   p_ik ≥ max(p_ij, p_jk)
//!
//! WST ⊂ MST ⊂ SST: strong is what a coherent random-utility judge with
//! well-ordered latents satisfies; weak is the floor below which no
//! scoring model exists at all. Small theorem the first implementation
//! tripped over: every 3-tournament has a Hamiltonian path, so an
//! orientation with both premises ≥ ½ ALWAYS exists — and a WST violation
//! is then exactly equivalent to the majority relation being cyclic
//! (a beats b beats c beats a). The `cyclic` flag records that
//! equivalence rather than a separate case.
//!
//! Estimation honesty: p̂ from k draws carries binomial noise, so every
//! violation is reported with its DEPTH in combined-standard-error units;
//! `*_violations_2se` count only violations deeper than 2 SE — the ones
//! sampling noise can't explain.

use serde::Serialize;

use crate::repeat_pooling::RepeatDraws;

/// Diagnostics for one testable triad.
#[derive(Debug, Clone, Serialize)]
pub struct TriadTest {
    /// Entity indices, oriented so premises hold: p(a>b) ≥ ½, p(b>c) ≥ ½.
    /// For a cyclic tournament the orientation is the cycle itself.
    pub triad: (usize, usize, usize),
    pub p_ab: f64,
    pub p_bc: f64,
    pub p_ac: f64,
    pub cyclic: bool,
    pub wst_violated: bool,
    pub mst_violated: bool,
    pub sst_violated: bool,
    /// Depth of the strongest violated threshold below p_ac, in combined
    /// SE units (0 when nothing violated).
    pub violation_depth_se: f64,
}

/// Result of [`stochastic_transitivity`].
#[derive(Debug, Serialize)]
pub struct TransitivityReport {
    pub triads: Vec<TriadTest>,
    pub testable_triads: usize,
    pub wst_violations: usize,
    pub mst_violations: usize,
    pub sst_violations: usize,
    /// Violations deeper than 2 combined standard errors.
    pub wst_violations_2se: usize,
    pub mst_violations_2se: usize,
    pub sst_violations_2se: usize,
    /// Smallest draw count among used pairs (the noise floor's driver).
    pub min_draws: usize,
}

fn p_hat(draws: &[f64]) -> (f64, usize) {
    let k = draws.len();
    if k == 0 {
        return (0.5, 0);
    }
    let wins: f64 = draws
        .iter()
        .map(|&m| {
            if m > 0.0 {
                1.0
            } else if m < 0.0 {
                0.0
            } else {
                0.5
            }
        })
        .sum();
    (wins / k as f64, k)
}

fn se(p: f64, k: usize) -> f64 {
    if k == 0 {
        return f64::INFINITY;
    }
    ((p * (1.0 - p)).max(0.25 / k as f64) / k as f64).sqrt()
}

/// Compute stochastic-transitivity diagnostics from repeat draws.
/// A triad is testable when all three of its pairs carry draws.
pub fn stochastic_transitivity(pairs: &[RepeatDraws]) -> TransitivityReport {
    use std::collections::HashMap;
    // p(i beats j) with i < j canonical; and draw counts.
    let mut p: HashMap<(usize, usize), (f64, usize)> = HashMap::new();
    let mut nodes: Vec<usize> = Vec::new();
    for rd in pairs {
        let (a, b) = (rd.i.min(rd.j), rd.i.max(rd.j));
        let (mut ph, k) = p_hat(&rd.draws);
        if rd.i > rd.j {
            ph = 1.0 - ph;
        }
        p.insert((a, b), (ph, k));
        for v in [a, b] {
            if !nodes.contains(&v) {
                nodes.push(v);
            }
        }
    }
    nodes.sort_unstable();
    // p(x beats y) for any order.
    let prob = |x: usize, y: usize| -> Option<(f64, usize)> {
        let (a, b) = (x.min(y), x.max(y));
        let &(ph, k) = p.get(&(a, b))?;
        Some(if x < y { (ph, k) } else { (1.0 - ph, k) })
    };

    let mut triads = Vec::new();
    let mut min_draws = usize::MAX;
    for ii in 0..nodes.len() {
        for jj in (ii + 1)..nodes.len() {
            for kk in (jj + 1)..nodes.len() {
                let (x, y, z) = (nodes[ii], nodes[jj], nodes[kk]);
                let (Some(_), Some(_), Some(_)) = (prob(x, y), prob(y, z), prob(x, z)) else {
                    continue;
                };
                // Orientation with both premises ≥ ½ always exists
                // (Hamiltonian path of the 3-tournament).
                let (a, b, c) = [
                    (x, y, z),
                    (x, z, y),
                    (y, x, z),
                    (y, z, x),
                    (z, x, y),
                    (z, y, x),
                ]
                .into_iter()
                .find(|&(a, b, c)| prob(a, b).unwrap().0 >= 0.5 && prob(b, c).unwrap().0 >= 0.5)
                .expect("every 3-tournament has a Hamiltonian path");
                let (p_ab, k_ab) = prob(a, b).unwrap();
                let (p_bc, k_bc) = prob(b, c).unwrap();
                let (p_ac, k_ac) = prob(a, c).unwrap();
                min_draws = min_draws.min(k_ab).min(k_bc).min(k_ac);

                let wst = p_ac < 0.5;
                let mst = p_ac < p_ab.min(p_bc);
                let sst = p_ac < p_ab.max(p_bc);
                // WST violation ⟺ the majority relation is a 3-cycle.
                let cyclic = wst;
                let combined_se =
                    (se(p_ab, k_ab).powi(2) + se(p_bc, k_bc).powi(2) + se(p_ac, k_ac).powi(2))
                        .sqrt();
                let threshold = if cyclic || wst {
                    0.5
                } else if mst {
                    p_ab.min(p_bc)
                } else if sst {
                    p_ab.max(p_bc)
                } else {
                    p_ac // nothing violated
                };
                let violation_depth_se = if wst || mst || sst {
                    ((threshold - p_ac) / combined_se).max(0.0)
                } else {
                    0.0
                };
                triads.push(TriadTest {
                    triad: (a, b, c),
                    p_ab,
                    p_bc,
                    p_ac,
                    cyclic,
                    wst_violated: wst,
                    mst_violated: mst,
                    sst_violated: sst,
                    violation_depth_se,
                });
            }
        }
    }

    let count = |f: fn(&TriadTest) -> bool| triads.iter().filter(|t| f(t)).count();
    let count_2se = |f: fn(&TriadTest) -> bool| {
        triads
            .iter()
            .filter(|t| f(t) && t.violation_depth_se > 2.0)
            .count()
    };
    TransitivityReport {
        testable_triads: triads.len(),
        wst_violations: count(|t| t.wst_violated),
        mst_violations: count(|t| t.mst_violated),
        sst_violations: count(|t| t.sst_violated),
        wst_violations_2se: count_2se(|t| t.wst_violated),
        mst_violations_2se: count_2se(|t| t.mst_violated),
        sst_violations_2se: count_2se(|t| t.sst_violated),
        min_draws: if min_draws == usize::MAX {
            0
        } else {
            min_draws
        },
        triads,
    }
}
