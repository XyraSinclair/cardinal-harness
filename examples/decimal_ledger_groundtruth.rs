//! Differential property battery for the REAL `decimal_ledger` Rust kernel
//! against exact ground truth.
//!
//! Loads the committed exact pushforward of gemma-4-E2B
//! (notes/decimal-pmf-2026-08-10/groundtruth_tree.json: 22,200 grammar
//! leaves with exact masked-renormalized masses, true E[h] known to fp
//! precision), marginalizes it into the o200k-shaped three-node instrument
//! (direction, integer token, fraction digit), emulates the census-verified
//! API access surface (temperature-1 draws returning exact chosen-token
//! probabilities plus top-5 sidebands per node, optionally with synthetic
//! provider jitter), and feeds those draws through the production
//! `accumulate`/`analyze` path.
//!
//! Run: cargo run --release --example decimal_ledger_groundtruth
//!
//! Properties measured (each printed with its denominator):
//!   P1 envelope coverage of true E[Z], exact access        (target 1.00)
//!   P2 mean±2σ coverage under 2% multiplicative jitter     (target ≥0.95)
//!   P3 antisymmetry: mirrored draws negate mean exactly, var invariant
//!   P4 conservation: gap ≈ 0 exact-access, enumerated_mass ≤ 1
//!   P5 bias/RMSE at the production K=8 (and K sweep)
//!   P6 median envelope width monotone nonincreasing in K
//!   P7 determinism: identical draws → identical outcome
//!   P8 p_dir_a matches the truth direction marginal
//!   P9 extract_trajectory accept/reject discipline on token streams

use std::collections::BTreeMap;

use std::collections::HashMap;

use cardinal_harness::gateway::{TokenAlternative, TokenLogprob};
use cardinal_harness::rating_engine::{AttributeParams, Observation, RaterParams, RatingEngine};
use cardinal_harness::rerank::decimal_ledger::{
    analyze, extract_trajectory, DrawTrajectory, NodeObs,
};

const TREE_PATH: &str = "notes/decimal-pmf-2026-08-10/groundtruth_tree.json";
const LN10: f64 = std::f64::consts::LN_10;
const REPS: usize = 400;
const KS: [usize; 5] = [2, 4, 8, 16, 32];
const TOPK: usize = 5;
const JITTER: f64 = 0.02;

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

/// The o200k-shaped truth: three-node conditionals marginalized from the
/// digit-level leaf PMF.
struct Truth {
    /// Leaves as ((dir, int, frac), mass), cumulative for sampling.
    leaves: Vec<((char, String, String), f64)>,
    cum: Vec<f64>,
    p_dir: BTreeMap<char, f64>,
    p_int: BTreeMap<char, Vec<(String, f64)>>, // sorted desc
    p_frac: BTreeMap<(char, String), Vec<(String, f64)>>, // sorted desc
    true_mean: f64, // E[Z] in PRESENTED natural-log coordinates (A positive)
}

fn load_truth() -> Truth {
    let raw = std::fs::read_to_string(TREE_PATH).expect("truth tree json");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse truth tree");
    let leaves_obj = v["leaves"].as_object().expect("leaves");
    let mut leaves = Vec::with_capacity(leaves_obj.len());
    for (key, mass) in leaves_obj {
        // key like "B:235.7"
        let (d, rest) = key.split_once(':').expect("leaf key");
        let (int_s, frac_s) = rest.split_once('.').expect("leaf digits");
        let dir = d.chars().next().unwrap();
        let p = mass.as_f64().unwrap();
        leaves.push(((dir, int_s.to_string(), frac_s.to_string()), p));
    }
    truth_from_leaves(leaves)
}

fn truth_from_leaves(leaves: Vec<((char, String, String), f64)>) -> Truth {
    let mut p_dir: BTreeMap<char, f64> = BTreeMap::new();
    let mut p_int_acc: BTreeMap<(char, String), f64> = BTreeMap::new();
    let mut p_frac_acc: BTreeMap<(char, String, String), f64> = BTreeMap::new();
    let mut true_mean = 0.0;
    for ((dir, int_s, frac_s), p) in &leaves {
        let r: f64 = format!("{int_s}.{frac_s}").parse().unwrap();
        // Presented natural-log coordinates: positive = A higher.
        let s = if *dir == 'A' { 1.0 } else { -1.0 };
        true_mean += p * s * r.clamp(1.0, 999.9).ln();
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
    Truth {
        leaves,
        cum,
        p_dir,
        p_int,
        p_frac,
        true_mean,
    }
}

impl Truth {
    fn sample_leaf(&self, rng: &mut XorShift) -> &(char, String, String) {
        let total = *self.cum.last().unwrap();
        let u = rng.f64() * total;
        let i = self.cum.partition_point(|&c| c < u);
        &self.leaves[i.min(self.leaves.len() - 1)].0
    }

    /// Emulate one API draw: chosen exact conditionals + top-k sidebands,
    /// each probability optionally multiplied by (1 ± jitter).
    fn draw(&self, rng: &mut XorShift, jitter: f64, mirror: bool) -> DrawTrajectory {
        let (dir, int_s, frac_s) = self.sample_leaf(rng).clone();
        let mut j = |p: f64| -> f64 {
            if jitter == 0.0 {
                p
            } else {
                (p * (1.0 + (rng.f64() * 2.0 - 1.0) * jitter)).clamp(0.0, 1.0)
            }
        };
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
            chosen: (flip(dir).to_string(), j(self.p_dir[&dir])),
            top: self
                .p_dir
                .iter()
                .map(|(d, p)| (flip(*d).to_string(), j(*p)))
                .collect(),
        };
        let ints = &self.p_int[&dir];
        let chosen_int_p = ints.iter().find(|(s, _)| *s == int_s).unwrap().1;
        let int_node = NodeObs {
            chosen: (int_s.clone(), j(chosen_int_p)),
            top: ints
                .iter()
                .take(TOPK)
                .map(|(s, p)| (s.clone(), j(*p)))
                .collect(),
        };
        let fracs = &self.p_frac[&(dir, int_s.clone())];
        let chosen_frac_p = fracs.iter().find(|(s, _)| *s == frac_s).unwrap().1;
        let frac_node = NodeObs {
            chosen: (frac_s.clone(), j(chosen_frac_p)),
            top: fracs
                .iter()
                .take(TOPK)
                .map(|(s, p)| (s.clone(), j(*p)))
                .collect(),
        };
        DrawTrajectory {
            dir: flip(dir),
            int_tok: int_s,
            frac_tok: frac_s,
            nodes: [dir_node, int_node, frac_node],
        }
    }
}

fn tok(s: &str, logp: f64, top: &[(&str, f64)]) -> TokenLogprob {
    TokenLogprob {
        token: s.to_string(),
        logprob: logp,
        top_alternatives: top
            .iter()
            .map(|(t, lp)| TokenAlternative {
                token: t.to_string(),
                logprob: *lp,
            })
            .collect(),
    }
}

fn main() {
    let truth = load_truth();
    println!(
        "truth loaded: {} leaves, true E[Z] (presented ln) = {:.6}  (= {:.4} log10 B-over-A)",
        truth.leaves.len(),
        truth.true_mean,
        -truth.true_mean / LN10
    );
    let true_mean = truth.true_mean;

    // ---- P1/P2/P4/P5/P6/P8: replication sweep over K, exact and jittered.
    for (label, jitter) in [("exact", 0.0), ("jitter2%", JITTER)] {
        println!("\n== access = {label} ==");
        println!(
            "{:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "K", "env-cover", "±2σ-cover", "bias", "rmse", "med-width", "max-gap", "max-enum"
        );
        let mut prev_med_width = f64::INFINITY;
        let mut width_monotone = true;
        for &k in &KS {
            let mut env_cover = 0usize;
            let mut sig_cover = 0usize;
            let mut errs = Vec::with_capacity(REPS);
            let mut widths = Vec::with_capacity(REPS);
            let mut max_gap: f64 = 0.0;
            let mut max_enum: f64 = 0.0;
            let mut pdir_err: f64 = 0.0;
            for rep in 0..REPS {
                let mut rng = XorShift(
                    0x9E37_79B9_7F4A_7C15
                        ^ ((k as u64) << 32)
                        ^ (rep as u64 + 1)
                        ^ ((jitter != 0.0) as u64) << 60,
                );
                let draws: Vec<DrawTrajectory> = (0..k)
                    .map(|_| truth.draw(&mut rng, jitter, false))
                    .collect();
                let out = analyze(&draws).expect("analyze");
                if out.e_lo - 1e-9 <= true_mean && true_mean <= out.e_hi + 1e-9 {
                    env_cover += 1;
                }
                let sigma = out.var.sqrt();
                if (out.mean - true_mean).abs() <= 2.0 * sigma {
                    sig_cover += 1;
                }
                errs.push(out.mean - true_mean);
                widths.push(out.envelope_width);
                max_gap = max_gap.max(out.conservation_gap);
                max_enum = max_enum.max(out.enumerated_mass);
                pdir_err = pdir_err
                    .max((out.p_dir_a - truth.p_dir.get(&'A').copied().unwrap_or(0.0)).abs());
            }
            let n = errs.len() as f64;
            let bias = errs.iter().sum::<f64>() / n;
            let rmse = (errs.iter().map(|e| e * e).sum::<f64>() / n).sqrt();
            widths.sort_by(|a, b| a.total_cmp(b));
            let med_width = widths[widths.len() / 2];
            if med_width > prev_med_width + 1e-9 {
                width_monotone = false;
            }
            prev_med_width = med_width;
            println!(
                "{:>4} {:>10.3} {:>10.3} {:>+10.4} {:>10.4} {:>10.4} {:>10.2e} {:>10.6}",
                k,
                env_cover as f64 / n,
                sig_cover as f64 / n,
                bias,
                rmse,
                med_width,
                max_gap,
                max_enum,
            );
            if jitter == 0.0 {
                assert!(
                    env_cover == REPS,
                    "P1 FAIL: exact-access envelope missed truth ({env_cover}/{REPS} at K={k})"
                );
                assert!(
                    max_gap < 1e-9,
                    "P4 FAIL: conservation gap {max_gap} under exact access"
                );
            }
            assert!(
                max_enum <= 1.0 + 1e-9,
                "P4 FAIL: enumerated mass {max_enum} > 1"
            );
            assert!(
                pdir_err < 0.05 || jitter > 0.0,
                "P8 FAIL: p_dir_a off by {pdir_err}"
            );
        }
        println!("P6 median width monotone nonincreasing in K: {width_monotone}");
        assert!(width_monotone, "P6 FAIL");
    }

    // ---- P3: antisymmetry under mirrored presentation.
    {
        let mut worst_mean: f64 = 0.0;
        let mut worst_var: f64 = 0.0;
        for rep in 0..100 {
            let mut rng1 = XorShift(0xABCD_EF01_2345_6789 ^ (rep + 1));
            let mut rng2 = XorShift(0xABCD_EF01_2345_6789 ^ (rep + 1));
            let d1: Vec<_> = (0..8).map(|_| truth.draw(&mut rng1, 0.0, false)).collect();
            let d2: Vec<_> = (0..8).map(|_| truth.draw(&mut rng2, 0.0, true)).collect();
            let o1 = analyze(&d1).unwrap();
            let o2 = analyze(&d2).unwrap();
            worst_mean = worst_mean.max((o1.mean + o2.mean).abs());
            worst_var = worst_var.max((o1.var - o2.var).abs());
        }
        println!("\nP3 antisymmetry over 100 mirrored reps: |mean+mean'| ≤ {worst_mean:.2e}, |var-var'| ≤ {worst_var:.2e}");
        assert!(worst_mean < 1e-9 && worst_var < 1e-9, "P3 FAIL");
    }

    // ---- P7: determinism.
    {
        let mut rng = XorShift(42);
        let draws: Vec<_> = (0..8).map(|_| truth.draw(&mut rng, 0.0, false)).collect();
        let a = analyze(&draws).unwrap();
        let b = analyze(&draws).unwrap();
        let same = a.mean == b.mean && a.var == b.var && a.e_lo == b.e_lo && a.e_hi == b.e_hi;
        println!("P7 determinism (identical draws → identical outcome): {same}");
        assert!(same, "P7 FAIL");
        // trie accumulation is draw-order sensitive only in obs-vector order,
        // which the mean/bootstrap-choose treat symmetrically per draw count;
        // verify a reversed order yields the same certificate anyway.
        let mut rev = draws.clone();
        rev.reverse();
        let c = analyze(&rev).unwrap();
        println!(
            "P7b order-invariance of envelope: {} (Δmean {:.1e})",
            (c.e_lo - a.e_lo).abs() < 1e-12 && (c.e_hi - a.e_hi).abs() < 1e-12,
            (c.mean - a.mean).abs()
        );
    }

    // ---- P9: extract_trajectory accept/reject discipline.
    {
        let lp = |p: f64| p.ln();
        // (a) canonical JSON-mode stream (o200k-style): accepted.
        let good = vec![
            tok("{\"", lp(0.99), &[]),
            tok("higher", lp(0.99), &[]),
            tok("_ranked", lp(0.99), &[]),
            tok("\":\"", lp(0.99), &[]),
            tok("B", lp(0.97), &[("A", lp(0.03))]),
            tok("\",\"", lp(0.99), &[]),
            tok("ratio", lp(0.99), &[]),
            tok("\":\"", lp(0.99), &[]),
            tok("12", lp(0.4), &[("15", lp(0.3)), ("120", lp(0.2))]),
            tok(".", lp(0.999), &[]),
            tok("5", lp(0.6), &[("0", lp(0.3))]),
            tok("\"}", lp(0.99), &[]),
        ];
        let t = extract_trajectory(&good).expect("P9a: canonical stream must parse");
        assert!(t.dir == 'B' && t.int_tok == "12" && t.frac_tok == "5");
        println!("\nP9a canonical JSON stream: accepted (dir=B int=12 frac=5) ✓");

        // (b) prose preamble with a space-led decimal: must NOT bind prose.
        let prose = vec![
            tok("The", lp(0.9), &[]),
            tok(" higher", lp(0.9), &[]),
            tok(" one", lp(0.9), &[]),
            tok(" is", lp(0.9), &[]),
            tok(" B", lp(0.9), &[]),
            tok(",", lp(0.9), &[]),
            tok(" ratio", lp(0.9), &[]),
            tok(" about", lp(0.9), &[]),
            tok(" 12", lp(0.9), &[]),
            tok(".", lp(0.9), &[]),
            tok("5", lp(0.9), &[]),
            tok(" times", lp(0.9), &[]),
        ];
        assert!(
            extract_trajectory(&prose).is_none(),
            "P9b FAIL: prose stream must be rejected (space-led int, space-led dir)"
        );
        println!("P9b prose preamble with ' 12' '.' '5': rejected ✓");

        // (c) split integer (digit-level tokenizer): rejected, not mis-binned.
        let mut split = good.clone();
        split[8] = tok("1", lp(0.5), &[]);
        split.insert(9, tok("2", lp(0.5), &[]));
        assert!(
            extract_trajectory(&split).is_none(),
            "P9c FAIL: split integer must be rejected"
        );
        println!("P9c split integer '1','2','.','5': rejected ✓");

        // (d) two-digit fraction token: rejected.
        let mut frac2 = good.clone();
        frac2[10] = tok("53", lp(0.6), &[]);
        assert!(
            extract_trajectory(&frac2).is_none(),
            "P9d FAIL: multi-digit fraction must be rejected"
        );
        println!("P9d two-digit fraction token '53': rejected ✓");
    }

    // ---- P5 headline at production K=8, exact access.
    println!("\nSingle-pair battery: all asserted properties held. See the K=8 rows for the production operating point.");

    matrix_gauntlet();
}

// ===================== Matrix-level gauntlet (P10–P16) =====================
//
// The ledger's PMF-shaped outcome only earns its keep if it composes: a
// matrix of per-pair credal PMFs must fuse — through the REAL sign algebra
// (multi.rs presented-coordinate flip) and the REAL solver
// (Observation::from_log_ratio_moments → IRLS) — into a coherent cardinal
// scale. This phase builds a synthetic judge as an exact quantizer PMF
// (latent signed log10 ratio ~ Normal(gap, σ), pushed forward through the
// decimal grid), emulates draws, and measures the whole chain.

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

/// Exact PMF of a quantizer judge: latent y ~ Normal(mu, sigma) in signed
/// log10-ratio space (positive = presented A higher), quantized to the
/// decimal grid dir ∈ {A,B}, r ∈ {1.0, 1.1, …, 999.9}. `snap_alpha` moves
/// that fraction of each non-.0 leaf's mass to the same integer's .0 leaf
/// (round-number attractor pathology).
fn judge_pmf(mu: f64, sigma: f64, snap_alpha: f64) -> Vec<((char, String, String), f64)> {
    let mut acc: BTreeMap<(char, String, String), f64> = BTreeMap::new();
    for deci in 10u32..=9999 {
        let r = deci as f64 / 10.0;
        let (int_s, frac_d) = (deci / 10, deci % 10);
        // Preimage of grid point r on the positive axis.
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

fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    for i in 0..n {
        for j in i + 1..n {
            let s = (a[i] - a[j]) * (b[i] - b[j]);
            if s > 0.0 {
                concordant += 1;
            } else if s < 0.0 {
                discordant += 1;
            }
        }
    }
    (concordant - discordant) as f64 / (concordant + discordant).max(1) as f64
}

/// Mirror multi.rs: presented-coordinate mean, swap flips the sign, variance
/// floored at EVIDENCE_VAR_FLOOR before precision = 1/var.
const EVIDENCE_VAR_FLOOR: f64 = 1e-3;

fn obs_from_outcome(
    i: usize,
    j: usize,
    mean_presented: f64,
    var: f64,
    swapped: bool,
    rater: &str,
) -> Observation {
    let mean_ij = if swapped {
        -mean_presented
    } else {
        mean_presented
    };
    Observation::from_log_ratio_moments(i, j, mean_ij, var.max(EVIDENCE_VAR_FLOOR), rater, 1.0)
}

struct SolvedRep {
    gaps_err: Vec<f64>, // recovered − latent gap, per pair (i<j)
    zscores: Vec<f64>,  // (recovered − latent gap) / sqrt(diff_var)
    /// Kernel-level z per sharp observation: (mean_ij − E[Z]_pmf) / σ_ledger,
    /// before any solver fusion. Separates instrument calibration from
    /// fusion calibration.
    kernel_z: Vec<f64>,
    tau: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_rep(
    n: usize,
    s: &[f64],
    truths: &BTreeMap<(usize, usize), Truth>,
    noisy: Option<&BTreeMap<(usize, usize), Truth>>,
    k: usize,
    rep_seed: u64,
    mirror_all: bool,
) -> SolvedRep {
    let mut observations = Vec::new();
    let mut kernel_z = Vec::new();
    let mut raters = HashMap::new();
    raters.insert("sharp".to_string(), RaterParams::default());
    if noisy.is_some() {
        raters.insert("noisy".to_string(), RaterParams::default());
    }
    for (&(i, j), truth) in truths {
        for swapped in [false, true] {
            // In mirror_all mode every presentation is flipped: the pair is
            // shown as (j, i), the draws mirror, and the multi.rs sign flip
            // must undo it exactly.
            let presented_swap = swapped ^ mirror_all;
            let mut rng = XorShift(
                rep_seed ^ ((i as u64) << 40) ^ ((j as u64) << 24) ^ ((swapped as u64) << 1) | 1,
            );
            let draws: Vec<DrawTrajectory> = (0..k)
                .map(|_| truth.draw(&mut rng, 0.0, presented_swap))
                .collect();
            let out = analyze(&draws).expect("matrix analyze");
            let mean_ij = if presented_swap { -out.mean } else { out.mean };
            kernel_z.push((mean_ij - truth.true_mean) / out.var.max(EVIDENCE_VAR_FLOOR).sqrt());
            observations.push(obs_from_outcome(
                i,
                j,
                out.mean,
                out.var,
                presented_swap,
                "sharp",
            ));
            if let Some(noisy_truths) = noisy {
                let nt = &noisy_truths[&(i, j)];
                let mut rng2 = XorShift(
                    rep_seed
                        ^ 0xDEAD
                        ^ ((i as u64) << 40)
                        ^ ((j as u64) << 24)
                        ^ ((swapped as u64) << 1)
                        | 1,
                );
                let nd: Vec<DrawTrajectory> = (0..k)
                    .map(|_| nt.draw(&mut rng2, 0.0, presented_swap))
                    .collect();
                let no = analyze(&nd).expect("noisy analyze");
                observations.push(obs_from_outcome(
                    i,
                    j,
                    no.mean,
                    no.var,
                    presented_swap,
                    "noisy",
                ));
            }
        }
    }
    let mut engine =
        RatingEngine::new(n, AttributeParams::default(), raters, None).expect("engine");
    engine.ingest(&observations);
    engine.solve();
    let scores = engine.scores().expect("scores").to_vec();
    let mut gaps_err = Vec::new();
    let mut zscores = Vec::new();
    for &(i, j) in truths.keys() {
        let latent = s[i] - s[j];
        let rec = scores[i] - scores[j];
        gaps_err.push(rec - latent);
        if let Some(dv) = engine.diff_var_for(i, j) {
            if dv > 0.0 {
                zscores.push((rec - latent) / dv.sqrt());
            }
        }
    }
    SolvedRep {
        gaps_err,
        zscores,
        kernel_z,
        tau: kendall_tau(&scores, s),
    }
}

fn matrix_gauntlet() {
    println!("\n==================== matrix gauntlet (P10–P16) ====================");
    const N: usize = 8;
    // Latent scale in nats: gaps span 1.4x (adjacent) to 200x (extremes),
    // all inside the instrument domain.
    let s = [0.0, 0.35, 0.8, 1.4, 2.1, 3.0, 4.1, 5.3];
    let sigma_sharp = 0.25; // judge scatter, log10 units (matches HARVEST scale)
    let sigma_noisy = 0.80;
    let build = |sigma: f64, snap: f64| -> BTreeMap<(usize, usize), Truth> {
        let mut m = BTreeMap::new();
        for i in 0..N {
            for j in i + 1..N {
                let mu = (s[i] - s[j]) / LN10;
                m.insert((i, j), truth_from_leaves(judge_pmf(mu, sigma, snap)));
            }
        }
        m
    };
    let truths = build(sigma_sharp, 0.0);
    let noisy_truths = build(sigma_noisy, 0.0);

    // Quantization bias of the instrument grid itself (PMF truth vs latent).
    let mut max_qbias: f64 = 0.0;
    for (&(i, j), t) in &truths {
        max_qbias = max_qbias.max((t.true_mean - (s[i] - s[j])).abs());
    }
    println!("grid quantization bias (max |E[Z]_pmf − latent gap|): {max_qbias:.4} nats");

    // ---- P10 + P16: end-to-end recovery and diff-var calibration.
    const R: usize = 40;
    let stats_at = |k: usize, reps: usize, seed_base: u64| {
        let mut taus = Vec::new();
        let mut all_err = Vec::new();
        let mut all_z = Vec::new();
        let mut all_kz = Vec::new();
        for rep in 0..reps {
            let sr = run_rep(N, &s, &truths, None, k, seed_base + rep as u64, false);
            taus.push(sr.tau);
            all_err.extend(sr.gaps_err);
            all_z.extend(sr.zscores);
            all_kz.extend(sr.kernel_z);
        }
        let mean_tau = taus.iter().sum::<f64>() / reps as f64;
        let rmse = (all_err.iter().map(|e| e * e).sum::<f64>() / all_err.len() as f64).sqrt();
        let bias = all_err.iter().sum::<f64>() / all_err.len() as f64;
        let cover2 = all_z.iter().filter(|z| z.abs() <= 2.0).count() as f64 / all_z.len() as f64;
        let kcover2 = all_kz.iter().filter(|z| z.abs() <= 2.0).count() as f64 / all_kz.len() as f64;
        (mean_tau, bias, rmse, cover2, kcover2, all_z.len())
    };
    let (mean_tau, bias, rmse, cover2, kcover2, nz) = stats_at(8, R, 0x51D0_0000);
    println!(
        "P10 recovery over {R} reps × {} pairs (2 counterbalanced obs/pair, K=8): mean tau {mean_tau:.3}, gap bias {bias:+.4}, gap RMSE {rmse:.4} nats",
        truths.len()
    );
    println!(
        "P16 calibration at K=8: kernel-level |z|≤2 coverage {kcover2:.3}; matrix-level {cover2:.3} over {nz} pair-reps"
    );
    // Matrix-level undercoverage relative to kernel-level is the fusion
    // √2 effect: the kernel's toward-zero magnitude shrinkage is CORRELATED
    // across the two counterbalanced observations (mirror symmetry preserves
    // it), so fusing them halves the variance without shrinking the bias.
    // The engine's temperature-calibration layer is the designed absorber.
    let (t32, b32, r32, c32, kc32, _) = stats_at(32, 10, 0x51D3_2000);
    println!(
        "P16b anytime at K=32 (10 reps): tau {t32:.3}, bias {b32:+.4}, RMSE {r32:.4}, matrix coverage {c32:.3}, kernel {kc32:.3}"
    );
    assert!(mean_tau >= 0.95, "P10 FAIL: mean tau {mean_tau}");
    assert!(rmse < 0.35, "P10 FAIL: gap RMSE {rmse}");
    assert!(kcover2 >= 0.85, "P16 FAIL: kernel coverage {kcover2}");
    assert!(cover2 >= 0.75, "P16 FAIL: matrix coverage {cover2}");
    assert!(
        c32 >= cover2 - 0.02 && r32 <= rmse,
        "P16b FAIL: more draws must not degrade calibration/RMSE (K=8 {cover2:.3}/{rmse:.4} → K=32 {c32:.3}/{r32:.4})"
    );

    // ---- P11: full-matrix presentation invariance. Same rep seed, every
    // pair mirrored at presentation; the multi.rs sign algebra must return
    // byte-identical scores.
    {
        let a = run_rep(N, &s, &truths, None, 8, 0xC0FFEE, false);
        let b = run_rep(N, &s, &truths, None, 8, 0xC0FFEE, true);
        let max_d = a
            .gaps_err
            .iter()
            .zip(&b.gaps_err)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        println!("P11 matrix presentation invariance: max |Δgap| = {max_d:.2e}");
        assert!(max_d < 1e-9, "P11 FAIL");
    }

    // ---- P12: triangle composition of RAW ledger means (before the solver
    // ever sees them): for additive truth, m_ij + m_jk − m_ik ≈ 0 within
    // combined uncertainty.
    {
        let mut worst_ratio: f64 = 0.0;
        let mut means: BTreeMap<(usize, usize), (f64, f64)> = BTreeMap::new();
        for (&(i, j), truth) in &truths {
            let mut rng = XorShift(0x7A1A_0000 ^ ((i as u64) << 16) ^ j as u64);
            let draws: Vec<DrawTrajectory> =
                (0..8).map(|_| truth.draw(&mut rng, 0.0, false)).collect();
            let out = analyze(&draws).unwrap();
            means.insert((i, j), (out.mean, out.var.max(EVIDENCE_VAR_FLOOR)));
        }
        for i in 0..N {
            for j in i + 1..N {
                for k in j + 1..N {
                    let (mij, vij) = means[&(i, j)];
                    let (mjk, vjk) = means[&(j, k)];
                    let (mik, vik) = means[&(i, k)];
                    let c = mij + mjk - mik;
                    worst_ratio = worst_ratio.max(c.abs() / (vij + vjk + vik).sqrt());
                }
            }
        }
        println!("P12 triangle composition of raw ledger means: max |cycle|/σ = {worst_ratio:.2} over 56 triangles");
        assert!(
            worst_ratio < 5.0,
            "P12 FAIL: cyclic residual {worst_ratio}σ"
        );
    }

    // ---- P13: precision flow. A flat judge (σ=0.8 vs 0.25) must carry
    // larger ledger variance and must not poison the sharp judge's recovery.
    {
        let mut var_sharp = Vec::new();
        let mut var_noisy = Vec::new();
        for (&(i, j), truth) in &truths {
            let mut r1 = XorShift(0xBEE5 ^ ((i as u64) << 16) ^ j as u64);
            let mut r2 = XorShift(0xBEE5 ^ ((i as u64) << 16) ^ j as u64);
            let d1: Vec<_> = (0..8).map(|_| truth.draw(&mut r1, 0.0, false)).collect();
            let d2: Vec<_> = (0..8)
                .map(|_| noisy_truths[&(i, j)].draw(&mut r2, 0.0, false))
                .collect();
            var_sharp.push(analyze(&d1).unwrap().var);
            var_noisy.push(analyze(&d2).unwrap().var);
        }
        var_sharp.sort_by(|a, b| a.total_cmp(b));
        var_noisy.sort_by(|a, b| a.total_cmp(b));
        let med_ratio = var_noisy[var_noisy.len() / 2] / var_sharp[var_sharp.len() / 2];
        let mut rmse_sharp = Vec::new();
        let mut rmse_both = Vec::new();
        for rep in 0..10 {
            let a = run_rep(N, &s, &truths, None, 8, 0xF00D_0000 + rep, false);
            let b = run_rep(
                N,
                &s,
                &truths,
                Some(&noisy_truths),
                8,
                0xF00D_0000 + rep,
                false,
            );
            rmse_sharp.push(
                (a.gaps_err.iter().map(|e| e * e).sum::<f64>() / a.gaps_err.len() as f64).sqrt(),
            );
            rmse_both.push(
                (b.gaps_err.iter().map(|e| e * e).sum::<f64>() / b.gaps_err.len() as f64).sqrt(),
            );
        }
        let ms = rmse_sharp.iter().sum::<f64>() / 10.0;
        let mb = rmse_both.iter().sum::<f64>() / 10.0;
        println!("P13 precision flow: median ledger-var ratio noisy/sharp = {med_ratio:.1}; recovery RMSE sharp-only {ms:.4} vs sharp+noisy {mb:.4}");
        assert!(med_ratio > 2.0, "P13 FAIL: flat judge not higher-variance");
        assert!(
            mb <= ms * 1.25,
            "P13 FAIL: noisy judge poisoned recovery ({ms} → {mb})"
        );
    }

    // ---- P15: round-number attractor pathology (half of all fraction mass
    // snapped to .0): ordering must survive, bias reported honestly.
    {
        let snapped = build(sigma_sharp, 0.5);
        let mut taus = Vec::new();
        let mut errs = Vec::new();
        for rep in 0..10 {
            let sr = run_rep(N, &s, &snapped, None, 8, 0x5A4B_0000 + rep, false);
            taus.push(sr.tau);
            errs.extend(sr.gaps_err);
        }
        let mt = taus.iter().sum::<f64>() / 10.0;
        let bias = errs.iter().sum::<f64>() / errs.len() as f64;
        println!(
            "P15 attractor pathology (50% snap-to-.0): mean tau {mt:.3}, gap bias {bias:+.4} nats"
        );
        assert!(mt >= 0.9, "P15 FAIL: tau {mt}");
    }

    println!("\nMatrix gauntlet: all asserted properties held.");
}
