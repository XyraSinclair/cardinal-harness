//! The judge portfolio, computed from a committed benchmark pack at zero
//! marginal cost: per-model primary latent vectors + per-model run costs
//! → ensemble geometry, optimal weights, marginal information per dollar.
//!
//! Usage: cargo run --example judge_portfolio -- <reports.jsonl>
//!
//! Cost caveat: use a FRESH (no-cache) pack — a cached run's costs are
//! near zero and poison the per-dollar column.

use cardinal_harness::rerank::judge_geometry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: judge_portfolio <reports.jsonl>")?;
    let mut names = Vec::new();
    let mut latents = Vec::new();
    let mut costs = Vec::new();
    for line in std::fs::read_to_string(&path)?.lines() {
        let r: serde_json::Value = serde_json::from_str(line)?;
        let scores: Vec<f64> = r["primary_scores"]
            .as_array()
            .ok_or("primary_scores missing")?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();
        names.push(r["model"].as_str().unwrap_or("?").to_string());
        latents.push(scores);
        costs.push(r["cost_nanodollars"].as_f64().unwrap_or(0.0) / 1e9);
    }
    let g = judge_geometry(&names, &latents, Some(&costs)).ok_or("geometry failed")?;
    println!(
        "consensus share {:.3} · effective error sources {:.2} · total information {:.2} · n = {}",
        g.consensus_share, g.effective_error_sources, g.total_information, g.n_entities
    );
    println!("{:<30} {:>7} {:>7} {:>8} {:>8} {:>10}", "judge", "load", "weight", "dI", "cost $", "dI/$");
    for &i in &g.portfolio_order {
        let e = &g.judges[i];
        println!(
            "{:<30} {:>7.3} {:>7.3} {:>8.3} {:>8.4} {:>10.1}",
            e.judge.split('/').next_back().unwrap_or(&e.judge),
            e.loading,
            e.weight,
            e.marginal_information,
            costs[i],
            e.information_per_dollar.unwrap_or(0.0)
        );
    }
    println!("\ncorrelation:");
    for row in &g.correlation {
        println!("  {}", row.iter().map(|v| format!("{v:+.2}")).collect::<Vec<_>>().join(" "));
    }
    Ok(())
}
