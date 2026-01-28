//! Cardinal-harness: pairwise ratio LLM judging + multi-objective reranking.

pub mod cache;
pub mod gateway;
pub mod prompts;
pub mod rating_engine;
pub mod rerank;
pub mod text_chunking;
pub mod trait_search;

pub use cache::{PairwiseCache, PairwiseCacheKey, SqlitePairwiseCache};
pub use gateway::{Attribution, ProviderGateway, UsageSink};
pub use rerank::{multi_rerank, rerank, ComparisonError, MultiRerankError};
