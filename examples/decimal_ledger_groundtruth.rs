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

use cardinal_harness::gateway::{TokenAlternative, TokenLogprob};
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
    let mut p_dir: BTreeMap<char, f64> = BTreeMap::new();
    let mut p_int_acc: BTreeMap<(char, String), f64> = BTreeMap::new();
    let mut p_frac_acc: BTreeMap<(char, String, String), f64> = BTreeMap::new();
    let mut true_mean = 0.0;
    for (key, mass) in leaves_obj {
        // key like "B:235.7"
        let (d, rest) = key.split_once(':').expect("leaf key");
        let (int_s, frac_s) = rest.split_once('.').expect("leaf digits");
        let dir = d.chars().next().unwrap();
        let p = mass.as_f64().unwrap();
        let r: f64 = rest.parse().unwrap();
        // Python truth: h = +log10 when dir == 'B'. Presented natural-log
        // coordinates: positive = A higher. Convert: z = s_A * ln(clamped r).
        let s = if dir == 'A' { 1.0 } else { -1.0 };
        true_mean += p * s * r.clamp(1.0, 999.9).ln();
        *p_dir.entry(dir).or_default() += p;
        *p_int_acc.entry((dir, int_s.to_string())).or_default() += p;
        *p_frac_acc
            .entry((dir, int_s.to_string(), frac_s.to_string()))
            .or_default() += p;
        leaves.push(((dir, int_s.to_string(), frac_s.to_string()), p));
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
    println!("\nAll asserted properties held. See the K=8 rows above for the production operating point.");
}
