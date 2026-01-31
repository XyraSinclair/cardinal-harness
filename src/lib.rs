#![forbid(unsafe_code)]

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
pub use rerank::{
    multi_rerank, multi_rerank_with_trace, rerank, rerank_with_trace, ComparisonError,
    ComparisonTrace, JsonlTraceSink, MultiRerankError, TraceError, TraceSink, TraceWorker,
};
