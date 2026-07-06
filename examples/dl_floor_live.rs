//! The full repeat-elicitation stack, live: nonce draws on several pairs
//! → DerSimonian–Laird pooling → the structural/contextual variance
//! split (σ_b vs σ_w) and the floored solve, from real judgements.
//!
//! Usage: OPENROUTER_API_KEY=... cargo run --example dl_floor_live -- <model> [k]

use std::sync::Arc;

use cardinal_harness::gateway::{Attribution, NoopUsageSink, ProviderGateway};
use cardinal_harness::repeat_pooling::{pool_repeats, RepeatDraws};
use cardinal_harness::rerank::{nonce_draws, CORPUS, PRIMARY_ATTRIBUTE};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args().nth(1).ok_or("usage: dl_floor_live <model> [k]")?;
    let k: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(6);
    let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;
    // A cycle-bearing pair set over 5 corpus items: enough graph for the
    // DL residual to have degrees of freedom.
    let pairs: [(usize, usize); 7] =
        [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (0, 4)];
    let mut repeat = Vec::new();
    let mut cost = 0i64;
    let mut cached = 0u64;
    let mut input = 0u64;
    for &(i, j) in &pairs {
        let report = nonce_draws(
            &gateway,
            &model,
            "canonical_v2",
            PRIMARY_ATTRIBUTE,
            ("A", CORPUS[i]),
            ("B", CORPUS[j]),
            k,
            0.0,
            7,
            Attribution::new("cardinal::example::dl_floor"),
        )
        .await?;
        cost += report.cost_nanodollars;
        cached += report.cache_read_tokens_total;
        input += report.input_tokens_total;
        let draws: Vec<f64> = report.draws.iter().flatten().copied().collect();
        eprintln!(
            "pair ({i},{j}): n={} mean={:+.3} sigma_w={:.3}",
            draws.len(),
            report.mean.unwrap_or(f64::NAN),
            report.sigma_w.unwrap_or(f64::NAN),
        );
        repeat.push(RepeatDraws { i, j, draws });
    }
    let pooled = pool_repeats(5, &repeat).ok_or("pooling failed")?;
    println!(
        "sigma_w = {:.3} nats (contextual, per draw) · sigma_b = {:.3} nats (structural floor) · Q = {:.2} on {} df",
        pooled.sigma_w2.sqrt(),
        pooled.sigma_b2.sqrt(),
        pooled.q_statistic,
        pooled.degrees_of_freedom,
    );
    println!("floored scores: {:?}", pooled.scores.iter().map(|s| (s * 100.0).round() / 100.0).collect::<Vec<_>>());
    println!("naive   scores: {:?}", pooled.scores_naive.iter().map(|s| (s * 100.0).round() / 100.0).collect::<Vec<_>>());
    println!(
        "{} draws total · cached {cached}/{input} input tokens · ${:.4}",
        pairs.len() * k,
        cost as f64 / 1e9
    );
    Ok(())
}
