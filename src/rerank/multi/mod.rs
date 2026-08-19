//! Multi-attribute reranking / trait search orchestrator.
//!
//! Wires together:
//! - TraitSearchManager (multi-attribute top-k uncertainty logic)
//! - RatingEngine (per-attribute IRLS solver)
//! - Pairwise LLM comparisons on a ratio ladder with confidence
//!
//! Core loop:
//! 1. Solve per-attribute rating engines and build global utility + uncertainty.
//! 2. Estimate top-k error via TraitSearchManager::estimate_topk_error().
//! 3. If error > tolerated_error and budgets remain, call propose_batch()
//!    to select highest-value comparisons.
//! 4. For each proposed (attribute_id, i, j):
//!    - Call LLM with evaluator prompt.
//!    - Parse JSON `{higher_ranked, ratio, confidence}` or `{refused:true}`.
//!    - Map to (ln_ratio, variance) and feed into the corresponding engine.
//! 5. Repeat until top-k error ≤ tolerated_error or budget/latency hit.

mod execution;
mod orchestrator;
mod request;
mod response;
mod task;

pub use execution::{
    build_engine_config, build_trait_search_config, JudgementRunInstrumentation, RerankExecution,
};
pub use orchestrator::multi_rerank;
pub use request::{
    apply_rerank_markup, estimate_max_rerank_charge, validate_multi_rerank_request,
    MultiRerankError, RerankChargeEstimate, EVIDENCE_VAR_FLOOR,
};

#[cfg(test)]
mod tests;
