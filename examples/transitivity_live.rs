//! Live stochastic-transitivity profile: nonce draws on every pair of a
//! small entity set → choice probabilities → the WST/MST/SST diagnostics.
//!
//! Usage: transitivity_live <model> [k]

use std::sync::Arc;

use cardinal_harness::gateway::{Attribution, NoopUsageSink, ProviderGateway};
use cardinal_harness::repeat_pooling::RepeatDraws;
use cardinal_harness::rerank::{nonce_draws, stochastic_transitivity, CORPUS, PRIMARY_ATTRIBUTE};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .ok_or("usage: transitivity_live <model> [k]")?;
    let k: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;
    // Four adjacent-depth corpus items: contested pairs, where
    // probabilistic intransitivity would live if it lives anywhere.
    let items = [1usize, 2, 3, 4];
    let mut all = Vec::new();
    let mut cost = 0i64;
    for a in 0..items.len() {
        for b in (a + 1)..items.len() {
            let report = nonce_draws(
                &gateway,
                &model,
                "canonical_v2",
                PRIMARY_ATTRIBUTE,
                ("A", CORPUS[items[a]]),
                ("B", CORPUS[items[b]]),
                k,
                0.0,
                7,
                Attribution::new("cardinal::example::transitivity"),
            )
            .await?;
            cost += report.cost_nanodollars;
            let draws: Vec<f64> = report.draws.iter().flatten().copied().collect();
            eprintln!(
                "pair ({a},{b}): n={} p̂={:.2}",
                draws.len(),
                draws.iter().filter(|&&m| m > 0.0).count() as f64 / draws.len().max(1) as f64
            );
            all.push(RepeatDraws { i: a, j: b, draws });
        }
    }
    let report = stochastic_transitivity(&all);
    println!(
        "triads {} · WST {} ({} beyond 2se) · MST {} ({}) · SST {} ({}) · min draws {} · ${:.4}",
        report.testable_triads,
        report.wst_violations,
        report.wst_violations_2se,
        report.mst_violations,
        report.mst_violations_2se,
        report.sst_violations,
        report.sst_violations_2se,
        report.min_draws,
        cost as f64 / 1e9,
    );
    for t in &report.triads {
        println!(
            "  {:?} p_ab {:.2} p_bc {:.2} p_ac {:.2} {}{}depth {:.1}se",
            t.triad,
            t.p_ab,
            t.p_bc,
            t.p_ac,
            if t.cyclic { "CYCLIC " } else { "" },
            if t.sst_violated { "SST-viol " } else { "" },
            t.violation_depth_se,
        );
    }
    Ok(())
}
