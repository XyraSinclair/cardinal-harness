//! Minimal end-to-end example for `cardinal-harness`.
//!
//! To run:
//! - Set `OPENROUTER_API_KEY`
//! - `cargo run --example quickstart`

use std::sync::Arc;

use cardinal_harness::gateway::NoopUsageSink;
use cardinal_harness::rerank::{
    ModelLadderPolicy, MultiRerankAttributeSpec, MultiRerankEntity, MultiRerankRequest,
    MultiRerankTopKSpec, RerankRunOptions,
};
use cardinal_harness::{Attribution, ProviderGateway, SqlitePairwiseCache};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = SqlitePairwiseCache::new(SqlitePairwiseCache::default_path())?;
    let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;
    let model_policy = Arc::new(ModelLadderPolicy::default());
    let run_options = RerankRunOptions {
        rng_seed: None,
        cache_only: false,
    };

    let req = MultiRerankRequest {
        entities: vec![
            MultiRerankEntity {
                id: "a".into(),
                text: "Entity A text".into(),
            },
            MultiRerankEntity {
                id: "b".into(),
                text: "Entity B text".into(),
            },
            MultiRerankEntity {
                id: "c".into(),
                text: "Entity C text".into(),
            },
        ],
        attributes: vec![MultiRerankAttributeSpec {
            id: "clarity".into(),
            prompt: "clarity of explanation".into(),
            prompt_template_slug: Some("canonical_v2".into()),
            weight: 1.0,
        }],
        topk: MultiRerankTopKSpec {
            k: 2,
            weight_exponent: 1.3,
            tolerated_error: 0.1,
            band_size: 5,
            effective_resistance_max_active: 64,
            stop_sigma_inflate: 1.25,
            stop_min_consecutive: 2,
        },
        gates: vec![],
        comparison_budget: None,
        latency_budget_ms: None,
        model: None,
        rater_id: None,
        comparison_concurrency: None,
        max_pair_repeats: None,
    };

    let resp = cardinal_harness::rerank::multi_rerank(
        Arc::new(gateway),
        Some(&cache),
        Some(model_policy),
        Some(&run_options),
        req,
        Attribution::new("example::quickstart"),
        None,
    )
    .await?;

    println!("stop_reason: {:?}", resp.meta.stop_reason);
    Ok(())
}
