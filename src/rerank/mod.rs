//! Reranking API module.
//!
//! Provides LLM-powered pairwise comparison-based reranking with:
//! - Calibrated uncertainty (top-k error semantics)
//! - Multi-attribute composition with weighted traits
//! - Adaptive stopping when error tolerance is met
//!
//! Two API tiers:
//! - Simple: Single-attribute, query-document relevance
//! - Multi: Full trait search with gates and weights

pub mod comparison;
pub mod evaluation;
pub mod model_policy;
pub mod options;
pub mod policy_registry;
pub mod report;
pub mod multi;
pub mod simple;
pub mod types;
// No async worker in the standalone harness.

// Re-export main entry points
pub use comparison::{compare_pair, ComparisonError};
pub use model_policy::{ModelLadderPolicy, ModelPolicy, ModelPolicyContext};
pub use options::RerankRunOptions;
pub use policy_registry::{load_policy_from_path, PolicyConfig, PolicyRegistry, PolicySpec};
pub use report::{build_report, render_report_markdown, RerankReport, RerankReportOptions};
pub use multi::{
    apply_rerank_markup, estimate_max_rerank_charge, multi_rerank, MultiRerankError,
    RerankChargeEstimate,
};
pub use simple::rerank;
pub use types::*;
