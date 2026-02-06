#![forbid(unsafe_code)]

//! # cardinal-harness
//!
//! The highest quality way to have LLMs put numbers on things.
//!
//! Instead of asking an LLM to "rate this 1–10" (unreliable, miscalibrated),
//! cardinal-harness asks pairwise ratio questions: "how many times more [attribute]
//! does A have than B?" A robust statistical solver (IRLS with Huber loss) combines
//! these noisy observations into globally consistent scores with uncertainty
//! estimates. The system selects the most informative pairs to query and stops
//! when the top-K ranking is sufficiently certain.
//!
//! See `docs/ALGORITHM.md` for the full design rationale.

pub mod cache;
pub mod gateway;
pub mod prompts;
pub mod rating_engine;
pub mod rerank;
pub mod text_chunking;
pub mod trait_search;

pub use cache::{PairwiseCache, PairwiseCacheKey, SqlitePairwiseCache};
pub use gateway::{Attribution, ChatGateway, ProviderGateway, UsageSink};
pub use rerank::{
    ComparisonEvent, ComparisonObserver,
    multi_rerank, multi_rerank_with_trace, rerank, rerank_with_trace, ComparisonError,
    ComparisonTrace, JsonlTraceSink, MultiRerankError, TraceError, TraceSink, TraceWorker,
    ObserverError, WarmStartData, WarmStartError, WarmStartProvider,
};
