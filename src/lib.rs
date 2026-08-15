#![forbid(unsafe_code)]

//! # llmsorting
//!
//! The instrument that turns an LLM's felt sense into calibrated measurement.
//!
//! Instead of asking an LLM to "rate this 1–10" (unreliable, miscalibrated),
//! llmsorting asks pairwise ratio questions: "how many times more attribute
//! does A have than B?" A robust statistical solver (IRLS with Huber loss) combines
//! these noisy observations into globally consistent scores with uncertainty
//! estimates. The system selects the most informative pairs to query and stops
//! when the top-K ranking is sufficiently certain.
//!
//! The ontology, in five nouns: an **attribute** (any nameable dimension) over
//! entities, each holding a latent **magnitude** (only ratios are observable);
//! **instruments** (elicitation modes) emit **evidence** in one currency —
//! (E\[log-ratio\], honest variance) — which the solver fuses into a
//! **scaling**: every entity placed on a shared log-ratio scale with a
//! *reading* (magnitude ± uncertainty). A ranking is a scaling with the
//! spacing deleted.
//!
//! Known as `cardinal-harness` before 2026-08-12. See `docs/ALGORITHM.md` for
//! the design rationale and `docs/WHAT_WHY_HOW.md` for the one-page version.

pub mod cache;
pub mod censored_likelihood;
pub mod discrete;
pub mod gain_calibration;
pub mod gateway;
pub mod judgement_run;
pub mod landing;
pub mod packet;
pub mod prompts;
pub mod rating_engine;
pub mod repeat_pooling;
pub mod rerank;
pub mod seriate;
pub mod text_chunking;
pub mod trait_search;

#[cfg(feature = "sqlite-store")]
pub use cache::SqlitePairwiseCache;
pub use cache::{PairwiseCache, PairwiseCacheKey};
pub use discrete::{DiscreteDistribution, WeightedValue};
pub use gateway::{Attribution, ChatGateway, ProviderGateway, UsageSink};
pub use rerank::{
    expand_prompt_experiment_request, multi_rerank, rerank, sort_documents, sort_texts,
    AttributePolarity, AttributeVariantSpec, ComparisonError, ComparisonEvent, ComparisonObserver,
    ComparisonTrace, JsonlTraceSink, MultiRerankError, ObserverError, PromptExperimentConfig,
    PromptExperimentError, RerankExecution, SortError, SortOptions, SortedItem, SortedTexts,
    TraceError, TraceSink, TraceWorker, WarmStartData, WarmStartError, WarmStartProvider,
};

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}
