use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use cardinal_harness::cache::{CachedJudgement, PairwiseCacheKey, SqlitePairwiseCache};
use cardinal_harness::gateway::openrouter::OpenRouterAdapter;
use cardinal_harness::gateway::{GatewayConfig, NoopUsageSink, ProviderGateway};
use cardinal_harness::prompts::prompt_by_slug;
use cardinal_harness::rerank::{
    multi_rerank_with_trace, MultiRerankAttributeSpec, MultiRerankEntity, MultiRerankRequest,
    MultiRerankTopKSpec, RerankRunOptions,
};
use cardinal_harness::{Attribution, ComparisonTrace, PairwiseCache, TraceError, TraceSink};
use tempfile::tempdir;

#[derive(Default)]
struct VecTraceSink {
    events: Mutex<Vec<ComparisonTrace>>,
}

impl VecTraceSink {
    fn take(&self) -> Vec<ComparisonTrace> {
        std::mem::take(&mut self.events.lock().unwrap())
    }
}

impl TraceSink for VecTraceSink {
    fn record(&self, event: ComparisonTrace) -> Result<(), TraceError> {
        self.events.lock().unwrap().push(event);
        Ok(())
    }
}

fn test_gateway() -> ProviderGateway<NoopUsageSink> {
    let openrouter = OpenRouterAdapter::new("test").unwrap();
    ProviderGateway::with_config(
        openrouter,
        std::sync::Arc::new(NoopUsageSink),
        GatewayConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(0),
        },
    )
}

fn canonical_v2_template_hash() -> String {
    let template = prompt_by_slug("canonical_v2").unwrap();
    blake3::hash(format!("{}\n{}", template.system, template.user).as_bytes())
        .to_hex()
        .to_string()
}

fn make_request(model: &str) -> MultiRerankRequest {
    MultiRerankRequest {
        entities: vec![
            MultiRerankEntity {
                id: "a".into(),
                text: "Entity A text".into(),
            },
            MultiRerankEntity {
                id: "b".into(),
                text: "Entity B text".into(),
            },
        ],
        attributes: vec![MultiRerankAttributeSpec {
            id: "clarity".into(),
            prompt: "clarity of explanation".into(),
            prompt_template_slug: Some("canonical_v2".into()),
            weight: 1.0,
        }],
        topk: MultiRerankTopKSpec {
            k: 1,
            weight_exponent: 1.0,
            tolerated_error: 0.0,
            band_size: 5,
            effective_resistance_max_active: 64,
            stop_sigma_inflate: 1.25,
            stop_min_consecutive: 1,
        },
        gates: vec![],
        comparison_budget: Some(1),
        latency_budget_ms: None,
        model: Some(model.into()),
        rater_id: None,
        comparison_concurrency: Some(1),
        max_pair_repeats: Some(1),
    }
}

#[tokio::test]
async fn rerank_records_trace_for_cached_comparison() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("cache.sqlite");
    let cache = SqlitePairwiseCache::new(&db_path).unwrap();

    let model = "openai/gpt-5-mini";
    let prompt_slug = "canonical_v2";
    let template_hash = canonical_v2_template_hash();

    // Pre-populate both A/B orders so we get a cache hit regardless of proposal ordering.
    let key_ab = PairwiseCacheKey::new(
        model,
        prompt_slug,
        &template_hash,
        "clarity",
        "clarity of explanation",
        "a",
        "Entity A text",
        "b",
        "Entity B text",
    );
    let key_ba = PairwiseCacheKey::new(
        model,
        prompt_slug,
        &template_hash,
        "clarity",
        "clarity of explanation",
        "b",
        "Entity B text",
        "a",
        "Entity A text",
    );

    // Underlying truth: entity "a" is higher-ranked than entity "b".
    let value_ab = CachedJudgement {
        higher_ranked: Some("A".to_string()),
        ratio: Some(2.0),
        confidence: Some(0.9),
        refused: false,
        input_tokens: None,
        output_tokens: None,
        provider_cost_nanodollars: None,
    };
    let value_ba = CachedJudgement {
        higher_ranked: Some("B".to_string()),
        ratio: Some(2.0),
        confidence: Some(0.9),
        refused: false,
        input_tokens: None,
        output_tokens: None,
        provider_cost_nanodollars: None,
    };

    cache.put(&key_ab, &value_ab).await.unwrap();
    cache.put(&key_ba, &value_ba).await.unwrap();

    let gateway = test_gateway();
    let run_options = RerankRunOptions {
        rng_seed: Some(0),
        cache_only: true,
    };
    let req = make_request(model);

    let trace_sink = VecTraceSink::default();
    let resp = multi_rerank_with_trace(
        std::sync::Arc::new(gateway),
        Some(&cache),
        None,
        Some(&run_options),
        req.clone(),
        Attribution::new("test::rerank_trace_cached"),
        None,
        None,
        Some(&trace_sink),
        None,
    )
    .await
    .unwrap();
    assert_eq!(resp.meta.comparisons_attempted, 1);

    let events = trace_sink.take();
    assert_eq!(events.len(), 1);
    let event = &events[0];

    assert!(event.cached);
    assert_eq!(event.input_tokens, 0);
    assert_eq!(event.output_tokens, 0);
    assert_eq!(event.provider_cost_nanodollars, 0);
    assert!(event.error.is_none());
    assert!(event.higher_ranked.is_some());
    assert_eq!(event.ratio, Some(2.0));
    assert_eq!(event.confidence, Some(0.9));

    let entities: HashMap<&str, &str> = req
        .entities
        .iter()
        .map(|e| (e.id.as_str(), e.text.as_str()))
        .collect();

    let expected_key = PairwiseCacheKey::new(
        &event.model,
        &event.prompt_template_slug,
        &event.template_hash,
        &event.attribute_id,
        "clarity of explanation",
        &event.entity_a_id,
        entities[event.entity_a_id.as_str()],
        &event.entity_b_id,
        entities[event.entity_b_id.as_str()],
    );
    assert_eq!(event.cache_key_hash, expected_key.key_hash);
    assert_eq!(
        event.attribute_prompt_hash,
        expected_key.attribute_prompt_hash
    );
}

#[tokio::test]
async fn rerank_records_trace_on_cache_miss_in_cache_only_mode() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("cache.sqlite");
    let cache = SqlitePairwiseCache::new(&db_path).unwrap();

    let model = "openai/gpt-5-mini";
    let gateway = test_gateway();
    let run_options = RerankRunOptions {
        rng_seed: Some(0),
        cache_only: true,
    };
    let req = make_request(model);

    let trace_sink = VecTraceSink::default();
    let err = multi_rerank_with_trace(
        std::sync::Arc::new(gateway),
        Some(&cache),
        None,
        Some(&run_options),
        req,
        Attribution::new("test::rerank_trace_cache_miss"),
        None,
        None,
        Some(&trace_sink),
        None,
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("cache_only") || err.to_string().contains("Cache miss"),
        "unexpected error: {err}"
    );

    let events = trace_sink.take();
    assert_eq!(events.len(), 1);
    let event = &events[0];
    assert!(!event.cached);
    assert!(event.error.as_deref().unwrap_or("").contains("cache_only"));
}
