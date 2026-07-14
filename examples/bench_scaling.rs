//! Lightweight scaling benchmark for the public rating-engine path.
//!
//! This intentionally avoids an external benchmark framework so it can run in
//! `cargo run --example bench_scaling` and emit JSONL measurements for docs/CI jobs.

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use cardinal_harness::rating_engine::{
    plan_edges_for_rater, AttributeParams, Config, Observation, PlannerMode, RaterParams,
    RatingEngine,
};
use serde_json::json;

const MAX_PLANNER_CANDIDATES: usize = 50_000;

const DEFAULT_MAX_N: usize = 250;
const DEFAULT_OUT: &str = "artifacts/bench/scaling.jsonl";
const REPORT_SCHEMA: &str = "bench_scaling_v2";
const RATER_ID: &str = "bench";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = Options::parse()?;
    if let Some(parent) = opts.out.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = File::create(&opts.out)?;
    let mut out = BufWriter::new(file);
    let build_metadata = build_metadata();

    for n in [10, 50, 100, 250, 500, 1_000, 5_000]
        .into_iter()
        .filter(|n| *n <= opts.max_n)
    {
        for density in [Density::SparseChain, Density::Banded3] {
            let row = run_case(n, density, opts.max_n, &build_metadata)?;
            writeln!(out, "{}", serde_json::to_string(&row)?)?;
        }
    }

    out.flush()?;
    eprintln!("wrote {}", opts.out.display());
    Ok(())
}

#[derive(Debug)]
struct Options {
    out: PathBuf,
    max_n: usize,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out = PathBuf::from(DEFAULT_OUT);
        let mut max_n = DEFAULT_MAX_N;
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out" => {
                    let value = args.next().ok_or("--out requires a path")?;
                    out = PathBuf::from(value);
                }
                "--max-n" => {
                    let value = args.next().ok_or("--max-n requires a value")?;
                    max_n = value.parse()?;
                }
                "--help" | "-h" => {
                    println!(
                        "Usage: cargo run --example bench_scaling -- [--out PATH] [--max-n N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument: {other}").into()),
            }
        }

        Ok(Self { out, max_n })
    }
}

#[derive(Debug, Clone, Copy)]
enum Density {
    SparseChain,
    Banded3,
}

impl Density {
    fn as_str(self) -> &'static str {
        match self {
            Self::SparseChain => "sparse_chain",
            Self::Banded3 => "banded_3",
        }
    }
}

fn run_case(
    n: usize,
    density: Density,
    max_n_arg: usize,
    build_metadata: &serde_json::Value,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let observations = observations_for(n, density);
    let candidates = planner_candidates(n);
    let mut engine = RatingEngine::new(
        n,
        AttributeParams::default(),
        rater_params(),
        Some(Config::default()),
    )?;

    let solve_started = Instant::now();
    engine.ingest(&observations);
    let summary = engine.solve();
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;

    let planner_started = Instant::now();
    let proposals =
        plan_edges_for_rater(&engine, &candidates, RATER_ID, PlannerMode::Hybrid, false)?;
    let planner_ms = planner_started.elapsed().as_secs_f64() * 1_000.0;

    Ok(json!({
        "report_schema": REPORT_SCHEMA,
        "build": build_metadata,
        "limits": {
            "max_n_arg": max_n_arg,
            "planner_candidate_cap": MAX_PLANNER_CANDIDATES,
        },
        "measurement": {
            "sample": "single_shot",
            "timer": "std::time::Instant",
            "solve_scope": "engine.ingest(observations) + engine.solve()",
            "planner_scope": "plan_edges_for_rater(engine, precomputed_candidates, hybrid)",
            "fixture_generation_timed": false,
        },
        "n": n,
        "density": density.as_str(),
        "observations": observations.len(),
        "planner_candidates": candidates.len(),
        "planner_proposals": proposals.len(),
        "solve_ms": solve_ms,
        "planner_ms": planner_ms,
        "components": summary.components,
        "cycle_dim": summary.cycle_dim,
        "top_score": summary.scores.first().copied(),
        "median_diag_cov": median(summary.diag_cov),
    }))
}

fn observations_for(n: usize, density: Density) -> Vec<Observation> {
    let mut observations = Vec::new();
    let window = match density {
        Density::SparseChain => 1,
        Density::Banded3 => 3,
    };

    for i in 0..n {
        let upper = (i + window + 1).min(n);
        for j in (i + 1)..upper {
            let gap = (j - i) as f64;
            let ratio = (1.08_f64.powf(gap)).clamp(1.0, 26.0);
            observations.push(Observation::new(i, j, ratio, 0.9, RATER_ID, 1.0));
        }
    }

    observations
}

fn planner_candidates(n: usize) -> Vec<(usize, usize)> {
    let mut candidates = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            candidates.push((i, j));
            if candidates.len() >= MAX_PLANNER_CANDIDATES {
                return candidates;
            }
        }
    }
    candidates
}

fn build_metadata() -> serde_json::Value {
    json!({
        "profile": if cfg!(debug_assertions) { "debug_assertions" } else { "optimized" },
        "debug_assertions": cfg!(debug_assertions),
        "target_os": env::consts::OS,
        "target_arch": env::consts::ARCH,
        "package_version": env!("CARGO_PKG_VERSION"),
    })
}

fn rater_params() -> HashMap<String, RaterParams> {
    let mut raters = HashMap::new();
    raters.insert(RATER_ID.to_string(), RaterParams::default());
    raters
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    Some(values[values.len() / 2])
}
