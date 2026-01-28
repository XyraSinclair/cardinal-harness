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
    let batch_model = format!("{}:batch", model);
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
}
