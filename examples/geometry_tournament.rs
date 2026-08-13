//! Elicitation-geometry tournament: endpoint loss per dollar across
//! instrument geometries (Oracle roadmap 2026-08-12, item #1).
//!
//! The decimal-ledger battery (decimal_ledger_groundtruth.rs) settled
//! DECODER FIDELITY: the kernel extracts what the token distribution
//! holds. This tournament attacks the next layer — DECISION VALUE: which
//! elicitation geometry moves the end-to-end accuracy/cost frontier once
//! every arm feeds the same real solver over the same latent world?
//!
//! Shared latent core: n=8 items with HARD NEAR-TIES (adjacent gaps down
//! to 0.03 nats, far below single-judgement percept noise), a quantizer
//! judge whose percept is y ~ Normal(z, σ) in log10 space, and 1.5%
//! multiplicative jitter on every logprob read (the census-measured
//! provider floor). Every arm maps its channel output to (mean, var)
//! evidence → Observation::from_log_ratio_moments → the production IRLS
//! engine. Cost is counted in CALLS (input-token-dominated pricing: one
//! pair prompt costs the same whatever the answer shape; the reasoning
//! arm pays 10 units/call).
//!
//! Arms:
//!   mc          decimal instrument, sampled text only (frequency MC) —
//!               the no-logprob baseline; C=1 is the classic point judge
//!   peel        decimal instrument, K=C redraws through the REAL
//!               decimal_ledger::analyze exact-atom kernel
//!   grid16      single-token 16-way signed-log codebook: full PMF
//!               visible in ONE call (top-k ≤ 20); extra calls average
//!               jitter
//!   radix4      3-stage radix-4 signed-log code: full 4-way conditionals
//!               at every VISITED node per call; credal peel across calls
//!   bin-lp      adaptive offset-binary staircase, logprob-read Bernoulli
//!               ("is A > e^h × B?" — read p from the one answer token)
//!   bin-smp     same staircase, sampled bit only (provider-universal)
//!   reason      sharper judge (σ/2.5), point answer, 10 units/call —
//!               the no-logprob strong-reasoner opportunity-cost baseline
//!
//! Distortion appendix: snap-to-.0 attractor (peel), ±25% β
//! miscalibration (bin-lp), codebook boundary shift (grid16) — the
//! instrument-validity failures Oracle flagged as the dangerous layer.
//!
//! Run: cargo run --release --example geometry_tournament

use std::collections::{BTreeMap, HashMap};

use ratiometer::rating_engine::{AttributeParams, Observation, RaterParams, RatingEngine};
use ratiometer::rerank::decimal_ledger::{analyze, DrawTrajectory, NodeObs};

const LN10: f64 = std::f64::consts::LN_10;
const JITTER: f64 = 0.015;
const TOPK: usize = 5;
const REPS: usize = 12;
const BUDGETS: [usize; 6] = [1, 2, 4, 8, 16, 32];
const SIGMA_JUDGE: f64 = 0.25; // log10 units (harvest-measured scale)
const NEAR_TIE_NATS: f64 = 0.12;
const TARGET_RMSE: f64 = 0.15;
const EVIDENCE_VAR_FLOOR: f64 = 1e-3;

// Latent scores in nats. Near-tie block at the bottom (0.03, 0.07 gaps),
// a mid cluster (0.45/0.07/0.15), and clear separations above.
const N: usize = 8;
const S: [f64; N] = [0.00, 0.03, 0.10, 0.55, 0.62, 1.60, 1.75, 3.20];

struct XorShift(u64);
impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Standard normal CDF via Abramowitz–Stegun 7.1.26 erf (|err| < 1.5e-7).
fn phi(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * (x.abs() / std::f64::consts::SQRT_2));
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = 1.0 - poly * (-(x * x) / 2.0).exp();
    if x >= 0.0 {
        0.5 * (1.0 + erf)
    } else {
        0.5 * (1.0 - erf)
    }
}

fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let (mut c, mut d) = (0i64, 0i64);
    for i in 0..n {
        for j in i + 1..n {
            let s = (a[i] - a[j]) * (b[i] - b[j]);
            if s > 0.0 {
                c += 1;
            } else if s < 0.0 {
                d += 1;
            }
        }
    }
    (c - d) as f64 / (c + d).max(1) as f64
}

fn jit(rng: &mut XorShift, p: f64) -> f64 {
    (p * (1.0 + (rng.f64() * 2.0 - 1.0) * JITTER)).clamp(0.0, 1.0)
}

// ===================== decimal instrument truth =====================
// (identical construction to the battery's matrix gauntlet)

struct DecimalTruth {
    leaves: Vec<((char, String, String), f64)>,
    cum: Vec<f64>,
    p_dir: BTreeMap<char, f64>,
    p_int: BTreeMap<char, Vec<(String, f64)>>,
    p_frac: BTreeMap<(char, String), Vec<(String, f64)>>,
}

fn decimal_pmf(mu: f64, sigma: f64, snap_alpha: f64) -> Vec<((char, String, String), f64)> {
    let mut acc: BTreeMap<(char, String, String), f64> = BTreeMap::new();
    for deci in 10u32..=9999 {
        let r = deci as f64 / 10.0;
        let (int_s, frac_d) = (deci / 10, deci % 10);
        let lo = if deci == 10 { 0.0 } else { (r - 0.05).log10() };
        let hi = if deci == 9999 {
            f64::INFINITY
        } else {
            (r + 0.05).log10()
        };
        for (dir, mass) in [
            ('A', phi((hi - mu) / sigma) - phi((lo - mu) / sigma)),
            ('B', phi((-lo - mu) / sigma) - phi((-hi - mu) / sigma)),
        ] {
            if mass < 1e-14 {
                continue;
            }
            let (kept, snapped) = if frac_d == 0 {
                (mass, 0.0)
            } else {
                (mass * (1.0 - snap_alpha), mass * snap_alpha)
            };
            if kept > 0.0 {
                *acc.entry((dir, int_s.to_string(), frac_d.to_string()))
                    .or_default() += kept;
            }
            if snapped > 0.0 {
                *acc.entry((dir, int_s.to_string(), "0".to_string()))
                    .or_default() += snapped;
            }
        }
    }
    acc.into_iter().collect()
}

fn decimal_truth(leaves: Vec<((char, String, String), f64)>) -> DecimalTruth {
    let mut p_dir: BTreeMap<char, f64> = BTreeMap::new();
    let mut p_int_acc: BTreeMap<(char, String), f64> = BTreeMap::new();
    let mut p_frac_acc: BTreeMap<(char, String, String), f64> = BTreeMap::new();
    for ((dir, int_s, frac_s), p) in &leaves {
        *p_dir.entry(*dir).or_default() += p;
        *p_int_acc.entry((*dir, int_s.clone())).or_default() += p;
        *p_frac_acc
            .entry((*dir, int_s.clone(), frac_s.clone()))
            .or_default() += p;
    }
    let mut cum = Vec::with_capacity(leaves.len());
    let mut acc = 0.0;
    for (_, p) in &leaves {
        acc += p;
        cum.push(acc);
    }
    let mut p_int: BTreeMap<char, Vec<(String, f64)>> = BTreeMap::new();
    for ((dir, int_s), p) in &p_int_acc {
        p_int
            .entry(*dir)
            .or_default()
            .push((int_s.clone(), *p / p_dir[dir]));
    }
    for v in p_int.values_mut() {
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
    }
    let mut p_frac: BTreeMap<(char, String), Vec<(String, f64)>> = BTreeMap::new();
    for ((dir, int_s, frac_s), p) in &p_frac_acc {
        let denom = p_int_acc[&(*dir, int_s.clone())];
        p_frac
            .entry((*dir, int_s.clone()))
            .or_default()
            .push((frac_s.clone(), *p / denom));
    }
    for v in p_frac.values_mut() {
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
    }
    DecimalTruth {
        leaves,
        cum,
        p_dir,
        p_int,
        p_frac,
    }
}

impl DecimalTruth {
    fn sample_leaf(&self, rng: &mut XorShift) -> &(char, String, String) {
        let total = *self.cum.last().unwrap();
        let u = rng.f64() * total;
        let i = self.cum.partition_point(|&c| c < u);
        &self.leaves[i.min(self.leaves.len() - 1)].0
    }

    fn leaf_z(dir: char, int_s: &str, frac_s: &str) -> f64 {
        let r: f64 = format!("{int_s}.{frac_s}").parse().unwrap();
        let s = if dir == 'A' { 1.0 } else { -1.0 };
        s * r.clamp(1.0, 999.9).ln()
    }

    /// One API draw for the peel arm: exact conditionals + top-k sidebands,
    /// jittered, optionally mirrored.
    fn draw(&self, rng: &mut XorShift, mirror: bool) -> DrawTrajectory {
        let (dir, int_s, frac_s) = self.sample_leaf(rng).clone();
        let flip = |d: char| {
            if mirror {
                if d == 'A' {
                    'B'
                } else {
                    'A'
                }
            } else {
                d
            }
        };
        let dir_node = NodeObs {
            chosen: (flip(dir).to_string(), jit(rng, self.p_dir[&dir])),
            top: self
                .p_dir
                .iter()
                .map(|(d, p)| (flip(*d).to_string(), jit(rng, *p)))
                .collect(),
        };
        let ints = &self.p_int[&dir];
        let chosen_int_p = ints.iter().find(|(s, _)| *s == int_s).unwrap().1;
        let int_node = NodeObs {
            chosen: (int_s.clone(), jit(rng, chosen_int_p)),
            top: ints
                .iter()
                .take(TOPK)
                .map(|(s, p)| (s.clone(), jit(rng, *p)))
                .collect(),
        };
        let fracs = &self.p_frac[&(dir, int_s.clone())];
        let chosen_frac_p = fracs.iter().find(|(s, _)| *s == frac_s).unwrap().1;
        let frac_node = NodeObs {
            chosen: (frac_s.clone(), jit(rng, chosen_frac_p)),
            top: fracs
                .iter()
                .take(TOPK)
                .map(|(s, p)| (s.clone(), jit(rng, *p)))
                .collect(),
        };
        DrawTrajectory {
            dir: flip(dir),
            int_tok: int_s,
            frac_tok: frac_s,
            nodes: [dir_node, int_node, frac_node],
        }
    }

    /// One sampled-text draw (mc / reason arms): signed ln ratio only.
    fn draw_z(&self, rng: &mut XorShift) -> f64 {
        let (dir, int_s, frac_s) = self.sample_leaf(rng);
        Self::leaf_z(*dir, int_s, frac_s)
    }
}

// ===================== grid16: single-token log codebook =====================
// 8 log10-uniform bins per direction over [0, 3]; 16 leaves total, all
// inside top-20 — one call exposes the entire first-level PMF.

const GRID_BINS: usize = 8;
const GRID_SPAN: f64 = 3.0; // log10

struct Grid16 {
    /// (z_center_nats, mass), all 16 leaves.
    leaves: Vec<(f64, f64)>,
}

fn grid16_truth(mu: f64, sigma: f64, edge_shift: f64) -> Grid16 {
    let w = GRID_SPAN / GRID_BINS as f64;
    let mut leaves = Vec::with_capacity(2 * GRID_BINS);
    for b in 0..GRID_BINS {
        let lo = b as f64 * w + if b == 0 { 0.0 } else { edge_shift };
        let hi = if b + 1 == GRID_BINS {
            f64::INFINITY
        } else {
            (b + 1) as f64 * w + edge_shift
        };
        let center = (b as f64 + 0.5) * w; // estimator's nominal center
        let pa = phi((hi - mu) / sigma) - phi((lo - mu) / sigma);
        let pb = phi((-lo - mu) / sigma) - phi((-hi - mu) / sigma);
        leaves.push((center * LN10, pa));
        leaves.push((-center * LN10, pb));
    }
    Grid16 { leaves }
}

impl Grid16 {
    /// One call: the full jittered PMF read. Estimator: moments over
    /// nominal bin centers, averaged across calls.
    fn measure(&self, rng: &mut XorShift, calls: usize, mirror: bool) -> (f64, f64) {
        let mut mean_acc = 0.0;
        let mut var_acc = 0.0;
        for _ in 0..calls {
            let read: Vec<(f64, f64)> = self
                .leaves
                .iter()
                .map(|&(z, p)| (if mirror { -z } else { z }, jit(rng, p)))
                .collect();
            let total: f64 = read.iter().map(|(_, p)| p).sum();
            let mean: f64 = read.iter().map(|(z, p)| z * p).sum::<f64>() / total.max(1e-12);
            // Jitter propagation: var of the PMF-mean under independent
            // per-bin multiplicative noise, plus half-bin quantization.
            let w_nats = GRID_SPAN / GRID_BINS as f64 * LN10;
            let vjit: f64 = read
                .iter()
                .map(|(z, p)| ((z - mean) * JITTER * p / 3f64.sqrt()).powi(2))
                .sum();
            mean_acc += mean;
            var_acc += vjit + w_nats * w_nats / 12.0;
        }
        let c = calls as f64;
        (mean_acc / c, (var_acc / c) / c)
    }
}

// ===================== radix4: hierarchical signed-log code =====================
// dir node (A/B), then 3 quaternary digits over log10 [0,3]: 64 bins/side
// of width 3/64 ≈ 0.047 log10. Every visited node's full 4-way conditional
// is visible (≤ top-20). Credal peel: exact atoms for fully-known paths,
// per-frontier cells for the rest.

struct Radix4 {
    /// leaf index (dir, d1, d2, d3) -> mass; plus marginalized conditionals.
    p_dir: [f64; 2],
    cond1: [[f64; 4]; 2],
    cond2: BTreeMap<(usize, usize), [f64; 4]>,
    cond3: BTreeMap<(usize, usize, usize), [f64; 4]>,
    cum: Vec<((usize, usize, usize, usize), f64)>,
}

fn radix4_truth(mu: f64, sigma: f64) -> Radix4 {
    let nb = 64usize;
    let w = GRID_SPAN / nb as f64;
    let mut leaf = BTreeMap::new();
    for d in 0..2 {
        for b in 0..nb {
            let lo = b as f64 * w;
            let hi = if b + 1 == nb {
                f64::INFINITY
            } else {
                (b + 1) as f64 * w
            };
            let mass = if d == 0 {
                phi((hi - mu) / sigma) - phi((lo - mu) / sigma)
            } else {
                phi((-lo - mu) / sigma) - phi((-hi - mu) / sigma)
            };
            leaf.insert((d, b / 16, (b / 4) % 4, b % 4), mass);
        }
    }
    let mut p_dir = [0.0; 2];
    let mut m1: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    let mut m2: BTreeMap<(usize, usize, usize), f64> = BTreeMap::new();
    for (&(d, a, b, c), &p) in &leaf {
        p_dir[d] += p;
        *m1.entry((d, a)).or_default() += p;
        *m2.entry((d, a, b)).or_default() += p;
        let _ = c;
    }
    let mut cond1 = [[0.0; 4]; 2];
    for (&(d, a), &p) in &m1 {
        cond1[d][a] = p / p_dir[d].max(1e-300);
    }
    let mut cond2 = BTreeMap::new();
    for (&(d, a, b), &p) in &m2 {
        cond2.entry((d, a)).or_insert([0.0; 4]).as_mut_slice()[b] = p / m1[&(d, a)].max(1e-300);
    }
    let mut cond3 = BTreeMap::new();
    for (&(d, a, b, c), &p) in &leaf {
        cond3.entry((d, a, b)).or_insert([0.0; 4]).as_mut_slice()[c] =
            p / m2[&(d, a, b)].max(1e-300);
    }
    let mut cum = Vec::with_capacity(leaf.len());
    let mut acc = 0.0;
    for (&k, &p) in &leaf {
        acc += p;
        cum.push((k, acc));
    }
    Radix4 {
        p_dir,
        cond1,
        cond2,
        cond3,
        cum,
    }
}

impl Radix4 {
    fn leaf_z(d: usize, a: usize, b: usize, c: usize) -> f64 {
        let bin = a * 16 + b * 4 + c;
        let center = (bin as f64 + 0.5) * (GRID_SPAN / 64.0);
        let s = if d == 0 { 1.0 } else { -1.0 };
        s * center * LN10
    }

    fn sample_leaf(&self, rng: &mut XorShift) -> (usize, usize, usize, usize) {
        let total = self.cum.last().unwrap().1;
        let u = rng.f64() * total;
        let i = self.cum.partition_point(|&(_, c)| c < u);
        self.cum[i.min(self.cum.len() - 1)].0
    }

    /// C calls: each draws a path, exposing jittered 4-way conditionals at
    /// every visited node. Credal moments over the union of visited nodes.
    // Digit indices double as code values here; index loops are the clear form.
    #[allow(clippy::needless_range_loop)]
    fn measure(&self, rng: &mut XorShift, calls: usize, mirror: bool) -> (f64, f64) {
        // Jittered conditional reads keyed by node; first visit wins (a
        // repeat visit would re-jitter — averaging them is a refinement the
        // v1 estimator skips, matching the decimal kernel's trie behavior).
        let mut seen_dir: Option<[f64; 2]> = None;
        let mut seen1: BTreeMap<usize, [f64; 4]> = BTreeMap::new();
        let mut seen2: BTreeMap<(usize, usize), [f64; 4]> = BTreeMap::new();
        let mut seen3: BTreeMap<(usize, usize, usize), [f64; 4]> = BTreeMap::new();
        for _ in 0..calls {
            let (d, a, b, c) = self.sample_leaf(rng);
            let _ = c;
            seen_dir.get_or_insert_with(|| [jit(rng, self.p_dir[0]), jit(rng, self.p_dir[1])]);
            seen1
                .entry(d)
                .or_insert_with(|| self.cond1[d].map(|p| jit(rng, p)));
            seen2
                .entry((d, a))
                .or_insert_with(|| self.cond2[&(d, a)].map(|p| jit(rng, p)));
            seen3
                .entry((d, a, b))
                .or_insert_with(|| self.cond3[&(d, a, b)].map(|p| jit(rng, p)));
        }
        // Enumerate: exact atoms where the whole path is known; cells at
        // the deepest known frontier otherwise.
        let sdir = seen_dir.unwrap();
        let mut atoms: Vec<(f64, f64)> = Vec::new(); // (z, p)
        let mut cells: Vec<(f64, f64, f64)> = Vec::new(); // (lo, hi, p)
        let sgn = |d: usize| if (d == 0) != mirror { 1.0 } else { -1.0 };
        let span = |d: usize, lo10: f64, hi10: f64| -> (f64, f64) {
            let (a, b) = (sgn(d) * lo10 * LN10, sgn(d) * hi10 * LN10);
            (a.min(b), a.max(b))
        };
        for d in 0..2 {
            let pd = sdir[d];
            if pd <= 0.0 {
                continue;
            }
            match seen1.get(&d) {
                None => {
                    let (lo, hi) = span(d, 0.0, GRID_SPAN);
                    cells.push((lo, hi, pd));
                }
                Some(c1) => {
                    for a in 0..4 {
                        let pa = pd * c1[a];
                        if pa <= 0.0 {
                            continue;
                        }
                        match seen2.get(&(d, a)) {
                            None => {
                                let (lo, hi) = span(d, a as f64 * 0.75, (a as f64 + 1.0) * 0.75);
                                cells.push((lo, hi, pa));
                            }
                            Some(c2) => {
                                for b in 0..4 {
                                    let pb = pa * c2[b];
                                    if pb <= 0.0 {
                                        continue;
                                    }
                                    match seen3.get(&(d, a, b)) {
                                        None => {
                                            let base = a as f64 * 0.75 + b as f64 * 0.1875;
                                            let (lo, hi) = span(d, base, base + 0.1875);
                                            cells.push((lo, hi, pb));
                                        }
                                        Some(c3) => {
                                            for c in 0..4 {
                                                let pc = pb * c3[c];
                                                if pc > 0.0 {
                                                    let z = sgn(0) * 0.0 + {
                                                        let z0 = Self::leaf_z(d, a, b, c);
                                                        if mirror {
                                                            -z0
                                                        } else {
                                                            z0
                                                        }
                                                    };
                                                    atoms.push((z, pc));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let total: f64 = atoms.iter().map(|(_, p)| p).sum::<f64>()
            + cells.iter().map(|(_, _, p)| p).sum::<f64>();
        let t = total.max(1e-12);
        let mean = (atoms.iter().map(|(z, p)| z * p).sum::<f64>()
            + cells
                .iter()
                .map(|(lo, hi, p)| 0.5 * (lo + hi) * p)
                .sum::<f64>())
            / t;
        let var = (atoms
            .iter()
            .map(|(z, p)| p * (z - mean).powi(2))
            .sum::<f64>()
            + cells
                .iter()
                .map(|(lo, hi, p)| {
                    let mid = 0.5 * (lo + hi);
                    p * ((hi - lo).powi(2) / 12.0 + (mid - mean).powi(2))
                })
                .sum::<f64>())
            / t;
        // Per-leaf half-bin quantization + estimand var scaled by calls:
        // the PMF mean is the estimand; its read error shrinks with calls
        // only through added node coverage, already reflected in cells.
        let wq = GRID_SPAN / 64.0 * LN10;
        (mean, var.max(wq * wq / 12.0))
    }
}

// ===================== offset-binary staircase =====================

struct BinaryArm {
    /// True percept scatter in nats (judge answers A iff y > h, y~N(z,σn²)).
    sigma_n: f64,
    /// Estimator's assumed scatter (β misspecification when ≠ sigma_n).
    sigma_hat: f64,
    /// Read the Bernoulli p from logprobs (true) or sample a bit (false).
    logprob: bool,
}

impl BinaryArm {
    /// C probes, thresholds adapted to the running posterior mean.
    /// Grid Bayes over z ∈ [-4.5, 4.5]; returns posterior (mean, var).
    fn measure(&self, rng: &mut XorShift, z_true: f64, calls: usize, mirror: bool) -> (f64, f64) {
        let z_pres = if mirror { -z_true } else { z_true };
        const M: usize = 361;
        let zs: Vec<f64> = (0..M)
            .map(|i| -4.5 + 9.0 * i as f64 / (M - 1) as f64)
            .collect();
        let mut logw = vec![0.0f64; M]; // flat prior over the domain
        let mut h = 0.0f64;
        for _ in 0..calls {
            let p_true = phi((z_pres - h) / self.sigma_n);
            if self.logprob {
                let p_read = jit(rng, p_true);
                // Observation model: p_read ≈ Φ((z-h)/σ̂) + jitter noise.
                let sd = (JITTER / 3f64.sqrt()) * p_true.max(1.0 - p_true) + 1e-3;
                for (i, z) in zs.iter().enumerate() {
                    let pm = phi((z - h) / self.sigma_hat);
                    logw[i] += -0.5 * ((p_read - pm) / sd).powi(2);
                }
            } else {
                let bit = rng.f64() < p_true;
                for (i, z) in zs.iter().enumerate() {
                    let pm = phi((z - h) / self.sigma_hat).clamp(1e-9, 1.0 - 1e-9);
                    logw[i] += if bit { pm.ln() } else { (1.0 - pm).ln() };
                }
            }
            // Adaptive threshold: posterior mean (info-max near p=0.5).
            let maxw = logw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ws: Vec<f64> = logw.iter().map(|w| (w - maxw).exp()).collect();
            let tw: f64 = ws.iter().sum();
            h = zs.iter().zip(&ws).map(|(z, w)| z * w).sum::<f64>() / tw;
        }
        let maxw = logw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ws: Vec<f64> = logw.iter().map(|w| (w - maxw).exp()).collect();
        let tw: f64 = ws.iter().sum();
        let mean = zs.iter().zip(&ws).map(|(z, w)| z * w).sum::<f64>() / tw;
        let var = zs
            .iter()
            .zip(&ws)
            .map(|(z, w)| w * (z - mean).powi(2))
            .sum::<f64>()
            / tw;
        (mean, var)
    }
}

// ===================== tournament harness =====================

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Mc,
    Peel,
    Grid16,
    Radix4,
    BinLp,
    BinSmp,
    Reason,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Arm::Mc => "mc",
            Arm::Peel => "peel",
            Arm::Grid16 => "grid16",
            Arm::Radix4 => "radix4",
            Arm::BinLp => "bin-lp",
            Arm::BinSmp => "bin-smp",
            Arm::Reason => "reason",
        }
    }
    /// Cost units actually spent per presentation given budget C units.
    fn cost(self, c: usize) -> f64 {
        match self {
            Arm::Reason => 10.0 * ((c as f64 / 10.0).ceil()).max(1.0),
            _ => c as f64,
        }
    }
}

struct World {
    dec: BTreeMap<(usize, usize), DecimalTruth>,
    dec_sharp: BTreeMap<(usize, usize), DecimalTruth>, // reason arm judge
    grid: BTreeMap<(usize, usize), Grid16>,
    radix: BTreeMap<(usize, usize), Radix4>,
}

// Pair indices (i, j) are used as both keys and S[] lookups; index loops are the clear form.
#[allow(clippy::needless_range_loop)]
fn build_world(snap: f64, edge_shift: f64) -> World {
    let mut dec = BTreeMap::new();
    let mut dec_sharp = BTreeMap::new();
    let mut grid = BTreeMap::new();
    let mut radix = BTreeMap::new();
    for i in 0..N {
        for j in i + 1..N {
            let mu = (S[i] - S[j]) / LN10;
            dec.insert((i, j), decimal_truth(decimal_pmf(mu, SIGMA_JUDGE, snap)));
            dec_sharp.insert(
                (i, j),
                decimal_truth(decimal_pmf(mu, SIGMA_JUDGE / 2.5, 0.0)),
            );
            grid.insert((i, j), grid16_truth(mu, SIGMA_JUDGE, edge_shift));
            radix.insert((i, j), radix4_truth(mu, SIGMA_JUDGE));
        }
    }
    World {
        dec,
        dec_sharp,
        grid,
        radix,
    }
}

fn mc_moments(zs: &[f64]) -> (f64, f64) {
    let n = zs.len() as f64;
    let mean = zs.iter().sum::<f64>() / n;
    if zs.len() < 2 {
        return (mean, 1.0); // single point draw: honest wide variance
    }
    let var = zs.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / (n - 1.0) / n;
    (mean, var)
}

/// One presentation's evidence for (arm, pair, budget): (mean_presented, var).
#[allow(clippy::too_many_arguments)]
fn measure(
    arm: Arm,
    world: &World,
    ij: (usize, usize),
    c: usize,
    rng: &mut XorShift,
    mirror: bool,
    sigma_hat_scale: f64,
) -> (f64, f64) {
    let sigma_n = SIGMA_JUDGE * LN10;
    match arm {
        Arm::Mc => {
            let t = &world.dec[&ij];
            let zs: Vec<f64> = (0..c)
                .map(|_| {
                    let z = t.draw_z(rng);
                    if mirror {
                        -z
                    } else {
                        z
                    }
                })
                .collect();
            mc_moments(&zs)
        }
        Arm::Peel => {
            let t = &world.dec[&ij];
            if c < 2 {
                let z = t.draw_z(rng);
                return (if mirror { -z } else { z }, 1.0);
            }
            let draws: Vec<DrawTrajectory> = (0..c).map(|_| t.draw(rng, mirror)).collect();
            let out = analyze(&draws).expect("peel analyze");
            (out.mean, out.var)
        }
        Arm::Grid16 => world.grid[&ij].measure(rng, c, mirror),
        Arm::Radix4 => world.radix[&ij].measure(rng, c, mirror),
        Arm::BinLp => BinaryArm {
            sigma_n,
            sigma_hat: sigma_n * sigma_hat_scale,
            logprob: true,
        }
        .measure(rng, S[ij.0] - S[ij.1], c, mirror),
        Arm::BinSmp => BinaryArm {
            sigma_n,
            sigma_hat: sigma_n * sigma_hat_scale,
            logprob: false,
        }
        .measure(rng, S[ij.0] - S[ij.1], c, mirror),
        Arm::Reason => {
            let t = &world.dec_sharp[&ij];
            let calls = ((c as f64 / 10.0).ceil() as usize).max(1);
            let zs: Vec<f64> = (0..calls)
                .map(|_| {
                    let z = t.draw_z(rng);
                    if mirror {
                        -z
                    } else {
                        z
                    }
                })
                .collect();
            mc_moments(&zs)
        }
    }
}

struct Endpoint {
    tau: f64,
    rmse: f64,
    near_tie_acc: f64,
    cover2: f64,
}

#[allow(clippy::needless_range_loop)]
fn run_matrix(arm: Arm, world: &World, c: usize, seed: u64, sigma_hat_scale: f64) -> Endpoint {
    let mut observations = Vec::new();
    let mut raters = HashMap::new();
    raters.insert("judge".to_string(), RaterParams::default());
    for i in 0..N {
        for j in i + 1..N {
            for swapped in [false, true] {
                let mut rng = XorShift(
                    seed ^ ((i as u64) << 40) ^ ((j as u64) << 24) ^ ((swapped as u64) << 1) | 1,
                );
                let (mean_p, var) =
                    measure(arm, world, (i, j), c, &mut rng, swapped, sigma_hat_scale);
                let mean_ij = if swapped { -mean_p } else { mean_p };
                observations.push(Observation::from_log_ratio_moments(
                    i,
                    j,
                    mean_ij,
                    var.max(EVIDENCE_VAR_FLOOR),
                    "judge",
                    1.0,
                ));
            }
        }
    }
    let mut engine =
        RatingEngine::new(N, AttributeParams::default(), raters, None).expect("engine");
    engine.ingest(&observations);
    engine.solve();
    let scores = engine.scores().expect("scores").to_vec();
    let mut errs = Vec::new();
    let mut zsc = Vec::new();
    let (mut nt_ok, mut nt_all) = (0usize, 0usize);
    for i in 0..N {
        for j in i + 1..N {
            let latent = S[i] - S[j];
            let rec = scores[i] - scores[j];
            errs.push(rec - latent);
            if latent.abs() < NEAR_TIE_NATS {
                nt_all += 1;
                if rec.signum() == latent.signum() {
                    nt_ok += 1;
                }
            }
            if let Some(dv) = engine.diff_var_for(i, j) {
                if dv > 0.0 {
                    zsc.push((rec - latent) / dv.sqrt());
                }
            }
        }
    }
    Endpoint {
        tau: kendall_tau(&scores, &S),
        rmse: (errs.iter().map(|e| e * e).sum::<f64>() / errs.len() as f64).sqrt(),
        near_tie_acc: nt_ok as f64 / nt_all.max(1) as f64,
        cover2: zsc.iter().filter(|z| z.abs() <= 2.0).count() as f64 / zsc.len().max(1) as f64,
    }
}

struct Row {
    cost: f64,
    tau: f64,
    rmse: f64,
    near: f64,
    cover: f64,
}

fn sweep(arm: Arm, world: &World, sigma_hat_scale: f64, seed0: u64) -> Vec<Row> {
    BUDGETS
        .iter()
        .map(|&c| {
            let mut taus = Vec::new();
            let mut rmses = Vec::new();
            let mut nears = Vec::new();
            let mut covers = Vec::new();
            for rep in 0..REPS {
                let e = run_matrix(
                    arm,
                    world,
                    c,
                    seed0 ^ ((c as u64) << 32) ^ (rep as u64 * 0x9E37) | 1,
                    sigma_hat_scale,
                );
                taus.push(e.tau);
                rmses.push(e.rmse);
                nears.push(e.near_tie_acc);
                covers.push(e.cover2);
            }
            let m = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            Row {
                cost: arm.cost(c) * 2.0 * (N * (N - 1) / 2) as f64,
                tau: m(&taus),
                rmse: m(&rmses),
                near: m(&nears),
                cover: m(&covers),
            }
        })
        .collect()
}

/// Interpolated cost (in call units, whole matrix) to reach TARGET_RMSE.
fn cost_to_target(rows: &[Row]) -> Option<f64> {
    for w in rows.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if a.rmse <= TARGET_RMSE {
            return Some(a.cost);
        }
        if b.rmse <= TARGET_RMSE {
            // linear in log-cost between the bracketing rows
            let f = (a.rmse - TARGET_RMSE) / (a.rmse - b.rmse).max(1e-12);
            return Some((a.cost.ln() + f * (b.cost.ln() - a.cost.ln())).exp());
        }
    }
    rows.last()
        .filter(|r| r.rmse <= TARGET_RMSE)
        .map(|r| r.cost)
}

fn print_sweep(label: &str, rows: &[Row]) {
    println!("\n-- {label} --");
    println!(
        "{:>6} {:>8} {:>8} {:>8} {:>9} {:>8}",
        "budget", "cost", "tau", "rmse", "near-tie", "cover2"
    );
    for (i, r) in rows.iter().enumerate() {
        println!(
            "{:>6} {:>8.0} {:>8.3} {:>8.4} {:>9.3} {:>8.3}",
            BUDGETS[i], r.cost, r.tau, r.rmse, r.near, r.cover
        );
    }
}

fn main() {
    println!(
        "elicitation-geometry tournament: n={N} items, 28 pairs × 2 counterbalanced presentations"
    );
    println!(
        "latent (nats): {:?}  | near-tie pairs: |gap| < {NEAR_TIE_NATS}",
        S
    );
    println!(
        "judge percept σ = {SIGMA_JUDGE} log10 ({:.3} nats); logprob jitter ±{:.1}%; cost unit = 1 call (input-dominated); reason arm = 10 units/call, σ/2.5",
        SIGMA_JUDGE * LN10,
        JITTER * 100.0
    );

    let world = build_world(0.0, 0.0);
    let arms = [
        Arm::Mc,
        Arm::Peel,
        Arm::Grid16,
        Arm::Radix4,
        Arm::BinLp,
        Arm::BinSmp,
        Arm::Reason,
    ];
    let mut all: Vec<(Arm, Vec<Row>)> = Vec::new();
    for arm in arms {
        let rows = sweep(arm, &world, 1.0, 0x7017_0000);
        print_sweep(arm.label(), &rows);
        all.push((arm, rows));
    }

    println!("\n==================== frontier summary ====================");
    println!(
        "{:>8} {:>18} {:>14} {:>14}",
        "arm", "cost→rmse≤0.15", "best rmse", "best near-tie"
    );
    for (arm, rows) in &all {
        let ctt = cost_to_target(rows);
        let best_rmse = rows.iter().map(|r| r.rmse).fold(f64::INFINITY, f64::min);
        let best_near = rows.iter().map(|r| r.near).fold(0.0, f64::max);
        println!(
            "{:>8} {:>18} {:>14.4} {:>14.3}",
            arm.label(),
            ctt.map_or("—".to_string(), |c| format!("{c:.0}")),
            best_rmse,
            best_near
        );
    }

    // Sanity assertions (loose: this is a measurement, not a property battery).
    let get = |a: Arm| &all.iter().find(|(x, _)| *x == a).unwrap().1;
    let peel = get(Arm::Peel);
    let mc = get(Arm::Mc);
    // Logprob harvesting must beat frequency MC at the production budget.
    assert!(
        peel[3].rmse <= mc[3].rmse * 1.05,
        "peel must not lose to mc at C=8 ({} vs {})",
        peel[3].rmse,
        mc[3].rmse
    );
    // Every logprob arm must order the scale essentially perfectly at C=32.
    for arm in [Arm::Peel, Arm::Grid16, Arm::Radix4, Arm::BinLp] {
        let rows = get(arm);
        assert!(
            rows[5].tau >= 0.95,
            "{} tau {} at C=32",
            arm.label(),
            rows[5].tau
        );
    }

    // ---- Distortion appendix: instrument-validity failures. ----
    println!("\n==================== distortion appendix ====================");

    // (a) decimal snap-to-.0 attractor at 30%: peel endpoint bias.
    {
        let snapped = build_world(0.3, 0.0);
        let rows = sweep(Arm::Peel, &snapped, 1.0, 0x7017_A000);
        print_sweep("peel + 30% snap-to-.0 attractor", &rows);
    }
    // (b) grid16 codebook boundary shift (+0.05 log10 on every inner edge).
    {
        let shifted = build_world(0.0, 0.05);
        let rows = sweep(Arm::Grid16, &shifted, 1.0, 0x7017_B000);
        print_sweep("grid16 + codebook edge shift +0.05 log10", &rows);
    }
    // (c) offset-binary β miscalibration: estimator assumes σ̂ = 0.75σ / 1.25σ.
    for scale in [0.75, 1.25] {
        let rows = sweep(Arm::BinLp, &world, scale, 0x7017_C000);
        print_sweep(&format!("bin-lp with σ̂ = {scale}σ (β miscal)"), &rows);
    }

    println!("\nTournament complete. Loss-vs-cost table above is the deliverable; see notes/decimal-pmf-2026-08-10/TOURNAMENT.md for the reading.");
}
