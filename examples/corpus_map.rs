//! The first real map: canonical attributes × real corpus entities ×
//! a judge portfolio, with the full evidence stack.
//!
//! Usage: corpus_map <entities.jsonl> <out_dir> <judge1,judge2> <cost_cap_dollars> <budget_per_run> "attr1" "attr2" ...

use std::io::Write;
use std::sync::Arc;

use cardinal_harness::cache::{PairwiseCache, SqlitePairwiseCache};
use cardinal_harness::gateway::{Attribution, NoopUsageSink, ProviderGateway};
use cardinal_harness::rerank::{
    sort_documents, RerankDocument, RerankExecution, RerankRunOptions, SortOptions,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [entities_path, out_dir, judges, cap, budget, attrs @ ..] = args.as_slice() else {
        return Err(
            "usage: corpus_map <entities.jsonl> <out> <j1,j2> <cap$> <budget> attrs...".into(),
        );
    };
    let judges: Vec<String> = judges.split(',').map(str::to_string).collect();
    let cap_nanodollars = (cap.parse::<f64>()? * 1e9) as i64;
    let budget: usize = budget.parse()?;

    let documents: Vec<RerankDocument> = std::fs::read_to_string(entities_path)?
        .lines()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            RerankDocument {
                id: v["id"].as_str().unwrap().to_string(),
                text: v["text"].as_str().unwrap().to_string(),
            }
        })
        .collect();
    eprintln!(
        "{} entities · {} judges · {} attributes · cap ${cap}",
        documents.len(),
        judges.len(),
        attrs.len()
    );

    let gateway = Arc::new(ProviderGateway::from_env(Arc::new(NoopUsageSink))?);
    let cache = SqlitePairwiseCache::new(std::path::PathBuf::from(format!(
        "{out_dir}/map-cache.sqlite"
    )))?;
    let mut total_cost = 0i64;
    let mut out = std::fs::File::create(format!("{out_dir}/latents.jsonl"))?;

    for attribute in attrs {
        for judge in &judges {
            if total_cost >= cap_nanodollars {
                eprintln!(
                    "COST CAP REACHED at ${:.2} — stopping cleanly",
                    total_cost as f64 / 1e9
                );
                return Ok(());
            }
            eprintln!("run: {judge} × \"{attribute}\"");
            let execution =
                RerankExecution::new(gateway.clone(), Attribution::new("cardinal::corpus_map"))
                    .cache(&cache as &dyn PairwiseCache)
                    .run_options(RerankRunOptions {
                        rng_seed: Some(7),
                        cache_only: false,
                    });
            let sorted = sort_documents(
                documents.clone(),
                attribute,
                execution,
                SortOptions {
                    model: Some(judge.clone()),
                    comparison_budget: Some(budget),
                    comparison_concurrency: Some(16),
                    ..Default::default()
                },
            )
            .await?;
            total_cost += sorted.meta.provider_cost_nanodollars;
            let record = serde_json::json!({
                "attribute": attribute,
                "judge": judge,
                "latents": sorted.items.iter().map(|i| (i.id.clone(), i.latent_mean, i.latent_std)).collect::<Vec<_>>(),
                "comparisons": sorted.meta.comparisons_used,
                "flips": sorted.meta.position_flips,
                "counterbalanced": sorted.meta.pairs_counterbalanced,
                "frustration": sorted.meta.judgement_frustration_mean,
                "cost_nanodollars": sorted.meta.provider_cost_nanodollars,
            });
            writeln!(out, "{record}")?;
            eprintln!(
                "  done: {} comparisons · flips {}/{} · ${:.3} (cum ${:.3})",
                sorted.meta.comparisons_used,
                sorted.meta.position_flips,
                sorted.meta.pairs_counterbalanced,
                sorted.meta.provider_cost_nanodollars as f64 / 1e9,
                total_cost as f64 / 1e9,
            );
        }
    }
    eprintln!("MAP COMPLETE · total ${:.3}", total_cost as f64 / 1e9);
    Ok(())
}
