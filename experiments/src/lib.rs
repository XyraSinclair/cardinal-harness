//! # llmsort-experiments
//!
//! The research side of llmsort: experimental verbs, live batteries, the
//! `cardinald` judgement-run daemon, and instruments whose evidence is not
//! yet in. Nothing here is published or promised; an instrument graduates
//! into the `llmsort` crate only after its evidence pack earns it.
//!
//! Evidence packs, campaign definitions, dated notes, and structured
//! judgements live in the cold archive:
//! <https://github.com/XyraSinclair/llmsort-lab>.

pub mod anp;
pub mod bench;
pub mod canonize;
pub mod ensemble;
pub mod evaluation;
pub mod experiments;
pub mod judgement_run;
pub mod landing;
pub mod slate;
pub mod transitivity;

pub use anp::{anp, AnpAlternative, AnpCriterion, AnpError, AnpOptions, AnpReport};
pub use bench::{
    core_pairs, orbit_pairs, render_report as render_bench_report, run_judge_bench, BenchCall,
    DimensionStat, JudgeBenchOptions, JudgeBenchReport, CALLS_PER_RUN, CORPUS, HARMONIC_BLOCK,
    HARMONIC_CYCLE, OPPOSITE_ATTRIBUTE, PARAPHRASE_ATTRIBUTE, PRIMARY_ATTRIBUTE,
};
pub use canonize::{
    canonize, planned_sorts, CandidateCanonicality, CanonizeError, CanonizeOptions, CanonizeReport,
};
pub use ensemble::{judge_geometry, JudgeGeometry, JudgePortfolioEntry};
pub use experiments::{
    expand_prompt_experiment_request, AttributePolarity, AttributeVariantSpec,
    PromptExperimentConfig, PromptExperimentError,
};
pub use slate::{propose_slate, SlateEntry, SlateError, SlateReport};
pub use transitivity::{stochastic_transitivity, TransitivityReport, TriadTest};
