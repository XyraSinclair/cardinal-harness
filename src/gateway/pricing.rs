//! Model pricing registry.
//!
//! Centralized pricing data for all supported models.
//! Costs are in nanodollars (1e-9 USD) per token.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Pricing information for a model.
#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    /// Provider name.
    pub provider: &'static str,
    /// Cost per input token in nanodollars.
    pub input_nanos_per_token: i64,
    /// Cost per output token in nanodollars.
    pub output_nanos_per_token: i64,
}

impl ModelPricing {
    const fn new(provider: &'static str, input: i64, output: i64) -> Self {
        Self {
            provider,
            input_nanos_per_token: input,
            output_nanos_per_token: output,
        }
    }

    /// Calculate cost for a request.
    pub fn calculate_cost(&self, input_tokens: u32, output_tokens: u32) -> i64 {
        (input_tokens as i64) * self.input_nanos_per_token
            + (output_tokens as i64) * self.output_nanos_per_token
    }
}

// =============================================================================
// PRICING DATA
// =============================================================================

// OpenAI Embeddings
// text-embedding-3-large: $0.13/1M tokens = 130 nanodollars/token
// text-embedding-3-small: $0.02/1M tokens = 20 nanodollars/token
// Batch API: 50% discount

const OPENAI_EMBED_3_LARGE: ModelPricing = ModelPricing::new("openai", 130, 0);
const OPENAI_EMBED_3_SMALL: ModelPricing = ModelPricing::new("openai", 20, 0);
const OPENAI_EMBED_3_LARGE_BATCH: ModelPricing = ModelPricing::new("openai", 65, 0);
const OPENAI_EMBED_3_SMALL_BATCH: ModelPricing = ModelPricing::new("openai", 10, 0);

// OpenRouter pricing (verify periodically against OpenRouter model pages)
// Claude 3.5 Haiku: $0.80/1M input, $4.00/1M output
// Claude 3.5 Sonnet: $3.00/1M input, $15.00/1M output
// GPT-4o-mini: $0.15/1M input, $0.60/1M output

const CLAUDE_35_HAIKU: ModelPricing = ModelPricing::new("openrouter", 800, 4_000);
const CLAUDE_35_SONNET: ModelPricing = ModelPricing::new("openrouter", 3_000, 15_000);
const GPT_4O_MINI: ModelPricing = ModelPricing::new("openrouter", 150, 600);
// GPT-5-mini: $0.25/1M input, $2.00/1M output (released Aug 2025)
const GPT_5_MINI: ModelPricing = ModelPricing::new("openrouter", 250, 2_000);
// GPT-5.2 Chat: $1.75/1M input, $14.00/1M output
const GPT_5_2_CHAT: ModelPricing = ModelPricing::new("openrouter", 1_750, 14_000);
// Kimi K2 0905: $0.39/1M input, $1.90/1M output
const KIMI_K2_0905: ModelPricing = ModelPricing::new("openrouter", 390, 1_900);
// Claude Opus 4.5: $5.00/1M input, $25.00/1M output
const CLAUDE_OPUS_4_5: ModelPricing = ModelPricing::new("openrouter", 5_000, 25_000);

static PRICING_MAP: OnceLock<HashMap<&'static str, ModelPricing>> = OnceLock::new();

fn init_pricing() -> HashMap<&'static str, ModelPricing> {
    let mut map = HashMap::new();

    // OpenAI Embeddings
    map.insert("text-embedding-3-large", OPENAI_EMBED_3_LARGE);
    map.insert("text-embedding-3-small", OPENAI_EMBED_3_SMALL);
    map.insert("text-embedding-3-large:batch", OPENAI_EMBED_3_LARGE_BATCH);
    map.insert("text-embedding-3-small:batch", OPENAI_EMBED_3_SMALL_BATCH);

    // OpenRouter models
    map.insert("anthropic/claude-3-5-haiku", CLAUDE_35_HAIKU);
    map.insert("anthropic/claude-3-5-haiku-20241022", CLAUDE_35_HAIKU);
    map.insert("anthropic/claude-3-5-sonnet", CLAUDE_35_SONNET);
    map.insert("anthropic/claude-3-5-sonnet-20241022", CLAUDE_35_SONNET);
    map.insert("openai/gpt-4o-mini", GPT_4O_MINI);
    map.insert("openai/gpt-4o-mini-2024-07-18", GPT_4O_MINI);
    map.insert("openai/gpt-5-mini", GPT_5_MINI);
    map.insert("openai/gpt-5-mini-2025-08-07", GPT_5_MINI);
    map.insert("openai/gpt-5.2-chat", GPT_5_2_CHAT);
    map.insert("moonshotai/kimi-k2-0905", KIMI_K2_0905);
    map.insert("anthropic/claude-opus-4.5", CLAUDE_OPUS_4_5);

    map
}

/// Get pricing for a model.
pub fn get_pricing(model_id: &str) -> Option<ModelPricing> {
    let map = PRICING_MAP.get_or_init(init_pricing);
    map.get(model_id).copied()
}

/// Get pricing for a model, falling back to a default.
pub fn get_pricing_or_default(model_id: &str, default: ModelPricing) -> ModelPricing {
    get_pricing(model_id).unwrap_or(default)
}

/// Calculate embedding cost (sync API).
pub fn embedding_cost(model: &str, tokens: u32) -> i64 {
    let pricing = get_pricing(model).unwrap_or(OPENAI_EMBED_3_LARGE);
    pricing.calculate_cost(tokens, 0)
}

/// Calculate embedding cost (batch API, 50% discount).
pub fn embedding_cost_batch(model: &str, tokens: u32) -> i64 {
    let batch_model = format!("{model}:batch");
    let pricing = get_pricing(&batch_model).unwrap_or(OPENAI_EMBED_3_LARGE_BATCH);
    pricing.calculate_cost(tokens, 0)
}

/// Calculate chat cost.
pub fn chat_cost(model: &str, input_tokens: u32, output_tokens: u32) -> i64 {
    // Default to a mid-range model if unknown
    let default = ModelPricing::new("unknown", 1_000, 5_000);
    let pricing = get_pricing(model).unwrap_or(default);
    pricing.calculate_cost(input_tokens, output_tokens)
}

// =============================================================================
// CACHE-AWARE COST MODEL
// =============================================================================

/// Pricing model that accounts for provider prompt caching.
///
/// Modern providers implement paged KV-cache where tokens sharing a common
/// prefix are served at a discount. This is structurally ideal for cardinal-harness
/// because all comparisons for an attribute share the same system prompt + template.
#[derive(Debug, Clone, Copy)]
pub struct CacheAwarePricing {
    /// Base per-token pricing.
    pub base: ModelPricing,
    /// Multiplier for cache-hit input tokens (e.g., 0.1 for 90% discount).
    /// Anthropic: ~0.1, OpenAI: ~0.5.
    pub cache_hit_multiplier: f64,
    /// Multiplier for initial cache-write tokens (typically 1.0 or 1.25).
    pub cache_write_multiplier: f64,
    /// Cache page size in tokens for alignment estimation.
    /// Anthropic: 128 tokens, OpenAI: implementation-dependent.
    pub cache_page_size: usize,
}

impl CacheAwarePricing {
    /// Estimate cost for a comparison given prefix sharing information.
    ///
    /// `shared_prefix_tokens` is the number of tokens common to all comparisons
    /// for this attribute (system prompt + template + attribute description).
    /// `unique_tokens` is the per-comparison entity text.
    /// `output_tokens` is the expected output length.
    /// `is_first_in_batch` indicates whether this is the first comparison
    /// (which pays cache-write cost instead of cache-hit cost).
    pub fn estimate_comparison_cost(
        &self,
        shared_prefix_tokens: u32,
        unique_tokens: u32,
        output_tokens: u32,
        is_first_in_batch: bool,
    ) -> ComparisonCostEstimate {
        // Compute cache-eligible tokens (aligned to page boundary).
        let cacheable = if self.cache_page_size > 0 {
            let pages = shared_prefix_tokens as usize / self.cache_page_size;
            (pages * self.cache_page_size) as u32
        } else {
            shared_prefix_tokens
        };
        let uncacheable_prefix = shared_prefix_tokens - cacheable;

        let (cached_input_tokens, cache_write_tokens) = if is_first_in_batch {
            // First request: pays full price for cache write.
            (0u32, cacheable)
        } else {
            // Subsequent requests: cache hit on prefix.
            (cacheable, 0u32)
        };

        let uncached_input_tokens = uncacheable_prefix + unique_tokens;

        // Cost computation in nanodollars.
        let input_nanos = self.base.input_nanos_per_token;
        let cost = (cache_write_tokens as i64)
            * (input_nanos as f64 * self.cache_write_multiplier) as i64
            + (cached_input_tokens as i64)
                * (input_nanos as f64 * self.cache_hit_multiplier) as i64
            + (uncached_input_tokens as i64) * input_nanos
            + (output_tokens as i64) * self.base.output_nanos_per_token;

        ComparisonCostEstimate {
            shared_prefix_tokens,
            cached_input_tokens,
            cache_write_tokens,
            uncached_input_tokens,
            output_tokens,
            estimated_cost_nanodollars: cost,
        }
    }
}

/// Detailed cost breakdown for a single comparison.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ComparisonCostEstimate {
    /// Total shared prefix tokens (before page alignment).
    pub shared_prefix_tokens: u32,
    /// Tokens served from prompt cache (page-aligned).
    pub cached_input_tokens: u32,
    /// Tokens written to cache (first request only).
    pub cache_write_tokens: u32,
    /// Tokens not eligible for caching (unique entity text + unaligned prefix).
    pub uncached_input_tokens: u32,
    /// Expected output tokens.
    pub output_tokens: u32,
    /// Total estimated cost in nanodollars.
    pub estimated_cost_nanodollars: i64,
}

/// Known cache-aware pricing for supported providers.
pub fn cache_aware_pricing(model_id: &str) -> Option<CacheAwarePricing> {
    let base = get_pricing(model_id)?;
    // Determine cache parameters by provider.
    let provider_prefix = model_id.split('/').next().unwrap_or("");
    match provider_prefix {
        "anthropic" => Some(CacheAwarePricing {
            base,
            cache_hit_multiplier: 0.1,
            cache_write_multiplier: 1.25,
            cache_page_size: 128,
        }),
        "openai" => Some(CacheAwarePricing {
            base,
            cache_hit_multiplier: 0.5,
            cache_write_multiplier: 1.0,
            cache_page_size: 128, // Approximate; OpenAI doesn't publish exact page size.
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_cost() {
        // 1M tokens at $0.13/1M = $0.13 = 130,000,000 nanodollars
        let cost = embedding_cost("text-embedding-3-large", 1_000_000);
        assert_eq!(cost, 130_000_000);
    }

    #[test]
    fn test_embedding_cost_batch() {
        // 1M tokens at $0.065/1M = $0.065 = 65,000,000 nanodollars
        let cost = embedding_cost_batch("text-embedding-3-large", 1_000_000);
        assert_eq!(cost, 65_000_000);
    }

    #[test]
    fn test_chat_cost() {
        // 1K input + 1K output for Claude 3.5 Haiku
        // Input: 1000 * 800 = 800,000 nanos
        // Output: 1000 * 4000 = 4,000,000 nanos
        // Total: 4,800,000 nanos = $0.0048
        let cost = chat_cost("anthropic/claude-3-5-haiku", 1_000, 1_000);
        assert_eq!(cost, 4_800_000);
    }

    // =========================================================================
    // Cache-aware cost model tests
    // =========================================================================

    #[test]
    fn test_cache_aware_first_request_pays_full() {
        let pricing = CacheAwarePricing {
            base: CLAUDE_35_HAIKU,
            cache_hit_multiplier: 0.1,
            cache_write_multiplier: 1.25,
            cache_page_size: 128,
        };

        let est = pricing.estimate_comparison_cost(1000, 500, 100, true);

        // First request: cache_write on page-aligned prefix, full on rest.
        // 1000 tokens, page_size 128 -> floor(1000/128)*128 = 896 cacheable
        // 104 uncacheable prefix + 500 unique = 604 uncached input
        assert_eq!(est.cache_write_tokens, 896);
        assert_eq!(est.cached_input_tokens, 0);
        assert_eq!(est.uncached_input_tokens, 104 + 500);
        assert!(est.estimated_cost_nanodollars > 0);
    }

    #[test]
    fn test_cache_aware_subsequent_request_gets_discount() {
        let pricing = CacheAwarePricing {
            base: CLAUDE_35_HAIKU,
            cache_hit_multiplier: 0.1,
            cache_write_multiplier: 1.25,
            cache_page_size: 128,
        };

        let first = pricing.estimate_comparison_cost(1000, 500, 100, true);
        let second = pricing.estimate_comparison_cost(1000, 500, 100, false);

        // Second request should be cheaper (cache hit on prefix).
        assert!(
            second.estimated_cost_nanodollars < first.estimated_cost_nanodollars,
            "cached request ({}) should cost less than first ({})",
            second.estimated_cost_nanodollars,
            first.estimated_cost_nanodollars
        );
        assert_eq!(second.cached_input_tokens, 896);
        assert_eq!(second.cache_write_tokens, 0);
    }

    #[test]
    fn test_cache_aware_page_alignment() {
        let pricing = CacheAwarePricing {
            base: CLAUDE_35_HAIKU,
            cache_hit_multiplier: 0.1,
            cache_write_multiplier: 1.0,
            cache_page_size: 128,
        };

        // Exactly page-aligned: 256 tokens = 2 pages.
        let est = pricing.estimate_comparison_cost(256, 100, 50, false);
        assert_eq!(est.cached_input_tokens, 256);
        assert_eq!(est.uncached_input_tokens, 100);

        // Not page-aligned: 300 tokens -> floor(300/128)*128 = 256 cached.
        let est2 = pricing.estimate_comparison_cost(300, 100, 50, false);
        assert_eq!(est2.cached_input_tokens, 256);
        assert_eq!(est2.uncached_input_tokens, 44 + 100);
    }

    #[test]
    fn test_cache_aware_zero_prefix() {
        let pricing = CacheAwarePricing {
            base: CLAUDE_35_HAIKU,
            cache_hit_multiplier: 0.1,
            cache_write_multiplier: 1.0,
            cache_page_size: 128,
        };

        let est = pricing.estimate_comparison_cost(0, 500, 100, false);
        assert_eq!(est.cached_input_tokens, 0);
        assert_eq!(est.uncached_input_tokens, 500);
    }

    #[test]
    fn test_cache_aware_pricing_lookup() {
        // Anthropic models should have cache-aware pricing.
        let cap = cache_aware_pricing("anthropic/claude-opus-4.5");
        assert!(cap.is_some());
        let cap = cap.unwrap();
        assert!((cap.cache_hit_multiplier - 0.1).abs() < 1e-6);
        assert_eq!(cap.cache_page_size, 128);

        // OpenAI models should also have cache-aware pricing.
        let cap = cache_aware_pricing("openai/gpt-5-mini");
        assert!(cap.is_some());
        let cap = cap.unwrap();
        assert!((cap.cache_hit_multiplier - 0.5).abs() < 1e-6);

        // Unknown provider should return None.
        let cap = cache_aware_pricing("unknown/model");
        assert!(cap.is_none());
    }

    #[test]
    fn test_cache_savings_for_typical_rerank_job() {
        // Simulate a typical 30-entity, 1-attribute rerank (30*29/2 = 435 max comparisons,
        // but active selection typically does ~60-100).
        let cap = cache_aware_pricing("anthropic/claude-opus-4.5").unwrap();

        let n_comparisons = 80;
        let prefix_tokens = 1500u32; // system + template + attribute
        let entity_tokens = 400u32; // avg entity pair text
        let output_tokens = 80u32;

        let first_cost = cap.estimate_comparison_cost(prefix_tokens, entity_tokens, output_tokens, true);
        let cached_cost = cap.estimate_comparison_cost(prefix_tokens, entity_tokens, output_tokens, false);

        let total_cached = first_cost.estimated_cost_nanodollars
            + (n_comparisons - 1) as i64 * cached_cost.estimated_cost_nanodollars;

        // Compare with naive (no cache) cost.
        let naive_per = cap.base.calculate_cost(prefix_tokens + entity_tokens, output_tokens);
        let total_naive = n_comparisons as i64 * naive_per;

        let savings_pct = 100.0 * (1.0 - total_cached as f64 / total_naive as f64);
        // Should save at least 40% with caching.
        assert!(
            savings_pct > 40.0,
            "expected >40% savings, got {savings_pct:.1}%"
        );
    }
}
