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

pub mod anp;
pub mod bench;
pub mod canonize;
pub mod comparison;
pub mod elaborate;
pub mod evaluation;
pub mod experiments;
pub mod explain;
mod gates;
pub mod hooks;
pub mod model_policy;
pub mod multi;
pub mod options;
pub mod orbit;
pub mod policy_registry;
pub mod report;
pub mod simple;
pub mod sort;
pub mod spin;
pub mod trace;
pub mod types;
pub mod wordings;
// No async worker in the standalone harness.

// Re-export main entry points
pub use anp::{anp, AnpAlternative, AnpCriterion, AnpError, AnpOptions, AnpReport};
pub use canonize::{canonize, CandidateCanonicality, CanonizeError, CanonizeOptions, CanonizeReport};
pub use bench::{
    core_pairs, orbit_pairs, render_report as render_bench_report, run_judge_bench, BenchCall,
    DimensionStat, JudgeBenchOptions, JudgeBenchReport, CALLS_PER_RUN, CORPUS, HARMONIC_BLOCK,
    HARMONIC_CYCLE, OPPOSITE_ATTRIBUTE, PARAPHRASE_ATTRIBUTE, PRIMARY_ATTRIBUTE,
};
pub use comparison::{
    compare_pair, ComparisonError, PairwiseComparisonAttribute, PairwiseComparisonEntity,
    PairwiseComparisonRequest, PairwiseComparisonSpec,
};
pub use elaborate::{elaborate_criterion, ElaborateError, ElaboratedCriterion};
pub use experiments::{
    expand_prompt_experiment_request, AttributePolarity, AttributeVariantSpec,
    PromptExperimentConfig, PromptExperimentError,
};
pub use explain::{
    differentiation_profile, explain_ranking, propose_candidates, propose_distinguishing,
    propose_for_goal, propose_rewordings, AttributeDifferentiation, AttributeExplanation,
    DifferentiationProfile, ExplainError, ExplainOptions, Explanation, ProposalUsage,
};
pub use hooks::{
    ComparisonEvent, ComparisonObserver, ObserverError, WarmStartData, WarmStartError,
    WarmStartProvider,
};
pub use model_policy::ModelLadderPolicy;
pub use multi::{
    apply_rerank_markup, estimate_max_rerank_charge, multi_rerank, validate_multi_rerank_request,
    MultiRerankError, RerankChargeEstimate, RerankExecution,
};
pub use options::RerankRunOptions;
pub use orbit::{orbit_transform, OrbitReport, CHARACTERS};
pub use policy_registry::{load_policy_from_path, PolicyConfig, PolicyRegistry, PolicySpec};
pub use report::{build_report, render_report_markdown, RerankReport, RerankReportOptions};
pub use simple::rerank;
pub use sort::{
    sort_documents, sort_texts, SortError, SortOptions, SortProbe, SortProbeKind, SortedItem,
    SortedTexts,
};
pub use spin::{spin_probe, spin_sweep, SpinFraming, SpinProbeReport, SpinReading, SpinSweepReport, SweepReading};
pub use trace::{ComparisonTrace, JsonlTraceSink, TraceError, TraceSink, TraceWorker};
pub use wordings::{wording_invariance, WordingInvarianceReport, WordingReading, WORDING_SLUGS};
pub use types::*;
