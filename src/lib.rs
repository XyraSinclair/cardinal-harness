#![forbid(unsafe_code)]

//! # cardinal-harness
//!
//! The highest quality way to have LLMs put numbers on things.
//!
//! Instead of asking an LLM to "rate this 1–10" (unreliable, miscalibrated),
//! cardinal-harness asks pairwise ratio questions: "how many times more attribute
//! does A have than B?" A robust statistical solver (IRLS with Huber loss) combines
//! these noisy observations into globally consistent scores with uncertainty
//! estimates. The system selects the most informative pairs to query and stops
//! when the top-K ranking is sufficiently certain.
//!
//! See `docs/ALGORITHM.md` for the design rationale.

pub mod cache;
pub mod discrete;
pub mod gateway;
pub mod prompts;
pub mod rating_engine;
pub mod rerank;
pub mod text_chunking;
pub mod trait_search;

#[cfg(feature = "sqlite-store")]
pub use cache::SqlitePairwiseCache;
pub use cache::{PairwiseCache, PairwiseCacheKey};
pub use discrete::{DiscreteDistribution, WeightedValue};
pub use gateway::{Attribution, ChatGateway, ProviderGateway, UsageSink};
pub use rerank::{
    multi_rerank, rerank, ComparisonError, ComparisonEvent, ComparisonObserver, ComparisonTrace,
    JsonlTraceSink, MultiRerankError, ObserverError, RerankExecution, TraceError, TraceSink,
    TraceWorker, WarmStartData, WarmStartError, WarmStartProvider,
};

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}
