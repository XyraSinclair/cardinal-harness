//! Multi-model generation → cardinal ranking → expert synthesis pipeline.
//!
//! The organic→diamond cycle:
//! 1. **Generate** — dispatch the same prompt to N models in parallel (divergent)
//! 2. **Rank** — feed outputs through multi-attribute pairwise reranking (convergent)
//! 3. **Synthesize** — expert model merges top-ranked outputs into one plan (diamond)
//!
//! Usage:
//! ```bash
//! cardinal pipeline --request pipeline.json --out session.json
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::anp::{AnpNetwork, Cluster, JudgmentContext, JudgmentKind, Node, RelationType};
use crate::cache::PairwiseCache;
use crate::gateway::{Attribution, ChatGateway, ChatModel, ChatRequest, Message};
use crate::rerank::model_policy::ModelPolicy;
use crate::rerank::trace::TraceSink;
use crate::rerank::types::{
    MultiRerankAttributeSpec, MultiRerankEntity, MultiRerankGateSpec, MultiRerankRequest,
    MultiRerankResponse, MultiRerankTopKSpec,
};
use crate::rerank::{multi_rerank_with_trace, JsonlTraceSink, RerankRunOptions};

// =============================================================================
// Types
// =============================================================================

/// A source file injected into generation prompts for codebase context.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ContextFile {
    pub path: String,
    pub content: String,
}

/// Predefined model tier presets for quick configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelPreset {
    /// Top-tier models: claude-opus-4-6, gpt-5.2-pro, gemini-2.5-pro
    Frontier,
    /// Mid-tier models: claude-sonnet-4-6, gpt-5-mini, gemini-2.5-flash
    Balanced,
    /// Fast iteration: gpt-5-mini, gemini-2.5-flash
    Fast,
}

impl ModelPreset {
    /// Return the model IDs for this preset.
    ///
    /// All tiers use SOTA models only — tiers differ in breadth (more models =
    /// more comparisons = higher cost, but richer signal).
    pub fn models(self) -> Vec<String> {
        match self {
            Self::Frontier => vec![
                "anthropic/claude-opus-4-6".into(),
                "openai/gpt-5.2-pro".into(),
                "google/gemini-3.1-pro".into(),
                "moonshotai/kimi-k2.5".into(),
                "x-ai/grok-4.1-fast".into(),
                "z-ai/glm-5".into(),
            ],
            Self::Balanced => vec![
                "anthropic/claude-opus-4-6".into(),
                "google/gemini-3.1-pro".into(),
                "x-ai/grok-4.1-fast".into(),
            ],
            Self::Fast => vec!["x-ai/grok-4.1-fast".into(), "z-ai/glm-5".into()],
        }
    }
}

/// Attribute to rank generated outputs on.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PipelineAttribute {
    pub id: String,
    pub prompt: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
}

fn default_weight() -> f64 {
    1.0
}

/// Top-K and stopping parameters for the ranking phase.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PipelineRankConfig {
    /// How many top generations to identify (default: 1).
    #[serde(default = "default_k")]
    pub k: usize,
    /// Stop when top-k error falls below this (default: 0.05).
    #[serde(default = "default_tolerated_error")]
    pub tolerated_error: f64,
    /// Maximum pairwise comparisons (default: auto).
    #[serde(default)]
    pub comparison_budget: Option<usize>,
    /// Model for pairwise judgments (default: openai/gpt-5-mini).
    #[serde(default)]
    pub judge_model: Option<String>,
    /// Model policy name or path for dynamic model selection.
    #[serde(default)]
    pub model_policy: Option<String>,
}

fn default_k() -> usize {
    1
}

fn default_tolerated_error() -> f64 {
    0.05
}

impl Default for PipelineRankConfig {
    fn default() -> Self {
        Self {
            k: 1,
            tolerated_error: 0.05,
            comparison_budget: None,
            judge_model: None,
            model_policy: None,
        }
    }
}

/// Full pipeline request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PipelineRequest {
    /// The task or question to send to all models.
    pub prompt: String,
    /// Optional system prompt prepended to each generation.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Models to generate from (OpenRouter model IDs).
    /// If empty, `preset` is used to populate models.
    pub models: Vec<String>,
    /// Model preset — used when `models` is empty.
    #[serde(default)]
    pub preset: Option<ModelPreset>,
    /// Source files to inject into generation prompts for codebase context.
    #[serde(default)]
    pub context_files: Vec<ContextFile>,
    /// Maximum token budget for context files (truncates if exceeded).
    #[serde(default)]
    pub max_context_tokens: Option<usize>,
    /// Attributes to rank on.
    pub attributes: Vec<PipelineAttribute>,
    /// Model to use for final synthesis.
    pub synthesis_model: String,
    /// Optional system prompt for the synthesis step.
    #[serde(default)]
    pub synthesis_system_prompt: Option<String>,
    /// Temperature for generation (default: 0.7).
    #[serde(default = "default_gen_temperature")]
    pub generation_temperature: f32,
    /// Temperature for synthesis (default: 0.3).
    #[serde(default = "default_synth_temperature")]
    pub synthesis_temperature: f32,
    /// Max tokens per generation (default: 4096).
    #[serde(default = "default_max_gen_tokens")]
    pub max_generation_tokens: u32,
    /// Max tokens for synthesis (default: 8192).
    #[serde(default = "default_max_synth_tokens")]
    pub max_synthesis_tokens: u32,
    /// Ranking configuration.
    #[serde(default)]
    pub rank_config: PipelineRankConfig,
}

fn default_gen_temperature() -> f32 {
    0.7
}
fn default_synth_temperature() -> f32 {
    0.3
}
fn default_max_gen_tokens() -> u32 {
    4096
}
fn default_max_synth_tokens() -> u32 {
    8192
}

/// One model's generation output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOutput {
    pub model: String,
    pub content: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost_nanodollars: i64,
    pub latency_ms: u64,
}

/// Synthesis output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisOutput {
    pub model: String,
    pub content: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost_nanodollars: i64,
    pub latency_ms: u64,
}

/// Cost breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCost {
    pub generation_cost_nanodollars: i64,
    pub ranking_cost_nanodollars: i64,
    pub synthesis_cost_nanodollars: i64,
    pub total_cost_nanodollars: i64,
}

/// Full pipeline session output.
#[derive(Debug, Serialize, Deserialize)]
pub struct PipelineSession {
    pub id: Uuid,
    pub created_at: String,
    pub request: PipelineRequest,
    pub generations: Vec<GenerationOutput>,
    pub ranking: MultiRerankResponse,
    pub synthesis: SynthesisOutput,
    pub cost: PipelineCost,
}

// =============================================================================
// Canonical attribute prompts
// =============================================================================
//
// These are the default evaluation dimensions for the pipeline. Each prompt is
// designed to elicit deep subjective judgment from the comparison model — not
// surface pattern-matching. The philosophy follows Norvid's principle of
// precision through formalization: each attribute is a precise specification
// of what "more" means in that dimension, with concrete anchors that
// distinguish genuine quality from its imitation.
//
// Eight dimensions, chosen to be orthogonal (each captures something the
// others miss) and collectively sufficient (a response that scores high on
// all eight is genuinely excellent by any standard we care about).

/// Canonical assessment attributes with Norvid-grade prompts.
///
/// Each attribute prompt is crafted to:
/// - Define the dimension precisely, distinguishing it from adjacent concepts
/// - Give the judge concrete positive and negative indicators
/// - Set anchors so the ratio ladder has natural meaning
/// - Reward depth of engagement over surface polish
pub fn default_assessment_attributes() -> Vec<PipelineAttribute> {
    vec![
        PipelineAttribute {
            id: "truthfulness".into(),
            prompt: concat!(
                "Truthfulness measures whether a response makes claims that are grounded in ",
                "observable reality — the actual codebase, real engineering constraints, genuine ",
                "tradeoffs — versus fabricating plausible-sounding assertions that cannot be ",
                "verified. A truthful response earns the reader's trust: when it says 'this ",
                "function is O(n²)', you can check and find it's right; when it's uncertain, ",
                "it says so rather than projecting false confidence. The critical distinction: ",
                "truthfulness is not about saying correct things (a response can be truthful and ",
                "wrong if it honestly represents its uncertainty), it's about the correspondence ",
                "between the response's confidence and its actual evidence. A response that makes ",
                "five bold claims with three wrong is LESS truthful than one that makes two ",
                "cautious claims, both right, and flags three areas it's unsure about. Watch for: ",
                "hallucinated function names or APIs, invented statistics, claims about code ",
                "behavior without evidence, false precision about performance numbers, and the ",
                "subtle tell of describing implementation details of code the author clearly ",
                "hasn't read.",
            ).into(),
            weight: 1.0,
        },
        PipelineAttribute {
            id: "diagnosis_precision".into(),
            prompt: concat!(
                "Diagnosis precision measures how specifically a response locates problems — not ",
                "'your architecture might have issues' but 'the DashMap in shoal::registry is ",
                "read without a fence after the Spiel commit, creating a window where stale state ",
                "is visible to concurrent readers.' The unit of precision is the intervention ",
                "point: can an engineer read this diagnosis and know exactly which file to open, ",
                "which function to look at, which data flow to trace? A precisely diagnosed ",
                "problem is halfway to being solved. An imprecisely diagnosed problem requires ",
                "another round of investigation before any work can begin. The hierarchy from ",
                "worst to best: vague category ('you have scaling issues'), named subsystem ",
                "('Slate queries are slow'), specific mechanism ('the Datalog join in ",
                "slate::query::execute materializes intermediate results into a Vec instead of ",
                "streaming, causing O(n×m) allocation for the cross-product before any filtering'), ",
                "and finally root cause with fix location ('the Datalog join materializes because ",
                "the planner in query::optimize doesn't push selections below joins — adding a ",
                "PushSelectDown rule to the optimizer would eliminate the intermediate allocation'). ",
                "Discount responses that substitute jargon density for actual specificity.",
            ).into(),
            weight: 0.9,
        },
        PipelineAttribute {
            id: "causal_depth".into(),
            prompt: concat!(
                "Causal depth measures how far a response traces the chain from surface ",
                "observation to root cause — and beyond to the structural forces that created ",
                "the root cause. Shallow analysis names a symptom. Moderate analysis identifies ",
                "the proximate cause. Deep analysis follows the causal chain to the architectural ",
                "decision, constraint, or incentive that made the proximate cause inevitable, and ",
                "proposes interventions at the level where they'll have durable effect rather than ",
                "patching symptoms. Example of shallow: 'the build is slow.' Moderate: 'the ",
                "build is slow because steel depends on nalgebra which takes 40s to compile.' ",
                "Deep: 'the build is slow because steel depends on nalgebra, which was pulled in ",
                "for a single matrix multiply in the cost model — the actual operation could be ",
                "done with a 20-line inline implementation, eliminating 40s of compile time and ",
                "breaking a dependency that also forces the minimum Rust version.' Notice that ",
                "depth is not length — a single sentence that identifies the true root cause is ",
                "deeper than three paragraphs that elaborate on the symptoms. The deepest ",
                "responses identify forces: why did the codebase evolve this way? What structural ",
                "incentive or historical accident produced this state? What would prevent the ",
                "problem from recurring after a fix?",
            ).into(),
            weight: 0.85,
        },
        PipelineAttribute {
            id: "intervention_economy".into(),
            prompt: concat!(
                "Intervention economy measures whether a response proposes the smallest change ",
                "that achieves the largest effect — maximum leverage, minimum disruption. This ",
                "is the engineering equivalent of Occam's razor: prefer the three-line fix over ",
                "the three-file refactor when both solve the problem. An economical intervention ",
                "respects the existing codebase as a living system with history, test coverage, ",
                "and downstream dependents. It doesn't propose 'rewrite the module' when 'add a ",
                "check on line 47' suffices. It distinguishes between what MUST change to solve ",
                "the problem and what COULD change to satisfy aesthetic preferences. Crucially, ",
                "economy is not laziness — the minimal intervention might be architecturally ",
                "significant if that's what the problem demands. The question is whether every ",
                "proposed change earns its keep: does each edit directly contribute to solving ",
                "the stated problem, or is the response padding recommendations to appear ",
                "thorough? Watch for: unnecessary abstraction layers, premature generalization ",
                "('let's make this configurable'), cleanup of adjacent code that isn't broken, ",
                "and the endemic failure mode of proposing six recommendations when the first ",
                "two solve 90% of the problem and the remaining four add risk without ",
                "proportionate value.",
            ).into(),
            weight: 0.8,
        },
        PipelineAttribute {
            id: "failure_imagination".into(),
            prompt: concat!(
                "Failure imagination measures whether a response thinks concretely about what ",
                "goes wrong — not as an abstract acknowledgment ('there could be edge cases') ",
                "but as specific, visualizable scenarios with conditions, triggers, and ",
                "consequences. A response with strong failure imagination says: 'if two clients ",
                "submit conflicting writes within the Spiel batch window (default 10ms), and ",
                "the second write's Slate entity references an attribute created by the first, ",
                "the reference will dangle until the next compaction cycle, during which any ",
                "Datalog query touching that entity will return incomplete results.' It thinks ",
                "about timing (race conditions, ordering), scale (what happens at 10x, 100x, ",
                "1000x the current load), boundaries (what happens when a u32 counter wraps, ",
                "when a Vec hits memory limits, when a network partition lasts longer than the ",
                "timeout), and second-order effects (fixing problem A introduces problem B). ",
                "Failure imagination is not pessimism — it's the skill of a builder who has ",
                "been woken at 3am by production incidents enough times to think 'and then what?' ",
                "after every design decision. Discount vague warnings ('this might not scale') ",
                "that don't specify the mechanism of failure.",
            ).into(),
            weight: 0.7,
        },
        PipelineAttribute {
            id: "compositional_awareness".into(),
            prompt: concat!(
                "Compositional awareness measures whether a response understands how the parts ",
                "of the system interact — reasoning about the whole rather than analyzing ",
                "components in isolation. A compositionally aware response, when discussing a ",
                "change to Slate's query engine, traces the effect through Spool (which reads ",
                "Slate state to route messages), through Shore (which queries Slate for connection ",
                "metadata), and through any modules that depend on Slate's query latency for their ",
                "SLA guarantees. It understands data flows across subsystem boundaries, shared ",
                "resources that create implicit coupling, and the emergent behaviors that arise ",
                "from component interactions that no single component's documentation describes. ",
                "The hierarchy: isolated analysis (treats each component as independent), ",
                "interface-aware analysis (understands the APIs between components but not the ",
                "runtime dynamics), and genuinely compositional analysis (understands how state ",
                "flows through the system at runtime, where backpressure propagates, which ",
                "failure modes cascade). The gold standard: identifying interactions between ",
                "subsystems that the original designers may not have intended or documented — ",
                "the emergent properties of the composed system.",
            ).into(),
            weight: 0.75,
        },
        PipelineAttribute {
            id: "epistemic_integrity".into(),
            prompt: concat!(
                "Epistemic integrity measures whether a response honestly represents its own ",
                "state of knowledge — applying the Norvid assurance ladder to itself. A response ",
                "with high epistemic integrity distinguishes between: claims it can verify by ",
                "pointing to specific code (Implemented-level assurance), claims derived from ",
                "general engineering principles applied to the architecture (Designed-level), ",
                "claims based on pattern-matching from similar systems (Mentioned-level), and ",
                "pure speculation (Proposed-level). It does not present speculation with the ",
                "confidence of verified fact. When it's working from incomplete information — ",
                "which is often, because no response has read the entire codebase — it flags ",
                "this: 'I haven't traced the full call path but based on the types involved, ",
                "this likely...' or 'assuming the standard Basin pattern here.' The failure mode ",
                "is the authoritative-sounding response that treats every claim as equally ",
                "certain, making it impossible for the reader to calibrate which recommendations ",
                "to trust and which to verify independently. The deepest form of epistemic ",
                "integrity is knowing what you don't know and saying so — identifying the ",
                "specific questions that would need to be answered to raise confidence, rather ",
                "than papering over uncertainty with fluent prose.",
            ).into(),
            weight: 0.65,
        },
        PipelineAttribute {
            id: "taste".into(),
            prompt: concat!(
                "Taste is the irreducible aesthetic dimension of engineering judgment — the ",
                "quality that separates a technically correct response from one that a senior ",
                "engineer would actually want to act on. Taste manifests as: knowing which ",
                "problems are important and which are pedantic (not every lint warning deserves ",
                "a paragraph), respecting the organic history of a codebase rather than imposing ",
                "textbook ideals onto a living system, understanding that 'correct' and 'good' ",
                "are different things (a response can be correct about every fact and still ",
                "propose the wrong priorities), and having a sense of proportion (the amount of ",
                "attention given to each topic should reflect its actual impact, not its ",
                "intellectual interest). A response with taste feels like it was written by ",
                "someone who has built and maintained systems at scale — someone who knows that ",
                "the elegant abstraction you didn't add is often worth more than the elegant ",
                "abstraction you did, that naming is worth arguing about but indentation isn't, ",
                "and that the best code change is sometimes no code change. Discount responses ",
                "that demonstrate technical knowledge without engineering wisdom: technically ",
                "impressive analyses that would lead you to spend a week on something that ",
                "doesn't matter, or that optimize for local perfection at the expense of global ",
                "coherence.",
            ).into(),
            weight: 0.6,
        },
    ]
}

/// Extended attributes: the original 8 plus verifiability and feasibility.
///
/// Also narrows the `taste` prompt to focus specifically on prioritization
/// judgment (avoiding overlap with compositional_awareness).
pub fn default_extended_attributes() -> Vec<PipelineAttribute> {
    let mut attrs = default_assessment_attributes();

    // Narrow taste to focus on prioritization judgment
    if let Some(taste) = attrs.iter_mut().find(|a| a.id == "taste") {
        taste.prompt = concat!(
            "Taste here means specifically prioritization judgment — the ability to rank ",
            "problems by actual impact and allocate attention proportionally. This is NOT ",
            "about compositional awareness (understanding system interactions) or epistemic ",
            "integrity (calibrating confidence). It's about the meta-skill of deciding what ",
            "matters most: does the response identify the 20% of issues that cause 80% of ",
            "the pain? Does it resist the temptation to enumerate every possible improvement ",
            "and instead focus on the ones with the highest leverage? A response with good ",
            "prioritization judgment might identify ten problems but explicitly rank them, ",
            "explain why the top two deserve immediate attention, and argue that several ",
            "others should be deferred or ignored entirely. The failure mode: treating all ",
            "observations as equally important, producing a flat list where a critical ",
            "architectural risk sits next to a naming convention quibble.",
        )
        .into();
    }

    // Add verifiability
    attrs.push(PipelineAttribute {
        id: "verifiability".into(),
        prompt: concat!(
            "Verifiability measures whether the claims in this response can be checked by an ",
            "engineer in under 5 minutes. Does it point to specific files, line numbers, ",
            "function signatures, error messages, or metrics that someone could go look at? ",
            "A highly verifiable response says 'the DashMap in crates/shoal/src/registry.rs ",
            "line 47 is read without synchronization' — you can open that file and check. A ",
            "low-verifiability response says 'there may be concurrency issues in the registry ",
            "layer' — you'd need to investigate for an hour before you even know if the claim ",
            "is true. Verifiability is not the same as truthfulness (a response can be ",
            "verifiable and wrong — the file reference is real but the analysis is incorrect). ",
            "It's about providing enough concrete anchors that the reader can efficiently ",
            "confirm or refute each claim. The gold standard: every substantive claim includes ",
            "a pointer to the evidence (file path, function name, data structure, error output) ",
            "that would confirm it. Watch for: claims about behavior without citing the code ",
            "path, performance assertions without numbers, and architectural claims that can't ",
            "be traced to specific implementations.",
        )
        .into(),
        weight: 0.7,
    });

    // Add feasibility
    attrs.push(PipelineAttribute {
        id: "feasibility".into(),
        prompt: concat!(
            "Feasibility measures how realistic it is to implement this response's proposals ",
            "given the current codebase state, team size, and practical constraints. A feasible ",
            "response proposes changes that could actually be landed in a reasonable iteration ",
            "— it doesn't require rewriting three subsystems, breaking every downstream ",
            "consumer, or introducing dependencies that conflict with existing architecture ",
            "principles. It accounts for the cost of the change: migration effort, test ",
            "coverage that needs updating, documentation, and the risk of introducing new bugs. ",
            "A response that proposes 'replace HashMap with a B-tree for sorted iteration' is ",
            "feasible. A response that proposes 'rewrite the storage layer in a capability-safe ",
            "language' is not, regardless of how technically sound the reasoning is. Feasibility ",
            "also considers sequencing: does the response propose changes in an order that's ",
            "actually executable, or does it require everything to happen simultaneously? The ",
            "failure mode is the intellectually stimulating proposal that would take six months ",
            "and touch every crate in the workspace. Discount responses that ignore practical ",
            "constraints in favor of theoretical ideals.",
        )
        .into(),
        weight: 0.65,
    });

    attrs
}

/// Default gate specifications for quality filtering.
///
/// Gates act as pass/fail filters — entities that fail any gate are marked
/// infeasible regardless of their attribute scores.
pub fn default_gates() -> Vec<MultiRerankGateSpec> {
    vec![MultiRerankGateSpec {
        attribute_id: "requirement_alignment".into(),
        unit: "latent".into(),
        op: ">=".into(),
        // Threshold at 0.0 latent means the response must be at least average
        // on requirement alignment to pass. This filters out responses that
        // drift to adjacent interesting problems instead of addressing the task.
        threshold: 0.0,
    }]
}

/// The requirement_alignment gate attribute prompt.
///
/// This is evaluated as a gate (pass/fail) rather than a scored attribute.
/// It checks whether a response actually addresses the stated task.
pub fn requirement_alignment_attribute() -> PipelineAttribute {
    PipelineAttribute {
        id: "requirement_alignment".into(),
        prompt: concat!(
            "Requirement alignment is a binary gate: does this response actually address the ",
            "stated task, or has it drifted to adjacent interesting problems? A response fails ",
            "this gate if it: answers a different question than the one asked, spends most of ",
            "its length on tangential topics that weren't requested, or addresses the topic at ",
            "the wrong level of abstraction (e.g., proposing architecture changes when asked ",
            "for a bug fix, or suggesting a one-line fix when asked for an architecture review). ",
            "This is not about quality — a mediocre but on-topic response passes, while a ",
            "brilliant but off-topic response fails. The test is simple: if someone reads only ",
            "the original prompt and then this response, would they say 'yes, this is answering ",
            "my question' or 'this is interesting but not what I asked'?",
        )
        .into(),
        weight: 1.0,
    }
}

// =============================================================================
// ANP cluster network for code evaluation
// =============================================================================

/// Build the 4-cluster ANP network for code evaluation.
///
/// Clusters:
/// - **Credibility**: truthfulness, epistemic_integrity, verifiability
/// - **Understanding**: diagnosis_precision, causal_depth, compositional_awareness
/// - **Robustness**: failure_imagination, feasibility
/// - **Intervention**: intervention_economy, taste
///
/// Inter-cluster influences:
/// - Credibility → Understanding (ComposableRatio): can't diagnose precisely if claims are wrong
/// - Understanding → Robustness (ComposableRatio): deep understanding enables better failure imagination
/// - Understanding → Intervention (ComposableRatio): precise diagnosis enables economical intervention
/// - Robustness → Intervention (PairwiseOnlyRatio): feasibility constrains intervention quality (local)
/// - Credibility → Intervention (PairwiseOnlyRatio): truthfulness influences taste (local)
pub fn build_code_evaluation_anp_network() -> AnpNetwork {
    let clusters = vec![
        Cluster {
            id: "credibility".into(),
            label: "Credibility".into(),
        },
        Cluster {
            id: "understanding".into(),
            label: "Understanding".into(),
        },
        Cluster {
            id: "robustness".into(),
            label: "Robustness".into(),
        },
        Cluster {
            id: "intervention".into(),
            label: "Intervention".into(),
        },
    ];

    let nodes = vec![
        // Credibility cluster
        Node {
            id: "truthfulness".into(),
            cluster_id: "credibility".into(),
            label: "Truthfulness".into(),
        },
        Node {
            id: "epistemic_integrity".into(),
            cluster_id: "credibility".into(),
            label: "Epistemic Integrity".into(),
        },
        Node {
            id: "verifiability".into(),
            cluster_id: "credibility".into(),
            label: "Verifiability".into(),
        },
        // Understanding cluster
        Node {
            id: "diagnosis_precision".into(),
            cluster_id: "understanding".into(),
            label: "Diagnosis Precision".into(),
        },
        Node {
            id: "causal_depth".into(),
            cluster_id: "understanding".into(),
            label: "Causal Depth".into(),
        },
        Node {
            id: "compositional_awareness".into(),
            cluster_id: "understanding".into(),
            label: "Compositional Awareness".into(),
        },
        // Robustness cluster
        Node {
            id: "failure_imagination".into(),
            cluster_id: "robustness".into(),
            label: "Failure Imagination".into(),
        },
        Node {
            id: "feasibility".into(),
            cluster_id: "robustness".into(),
            label: "Feasibility".into(),
        },
        // Intervention cluster
        Node {
            id: "intervention_economy".into(),
            cluster_id: "intervention".into(),
            label: "Intervention Economy".into(),
        },
        Node {
            id: "taste".into(),
            cluster_id: "intervention".into(),
            label: "Taste (Prioritization)".into(),
        },
    ];

    let contexts = vec![
        // Credibility → Understanding (ComposableRatio)
        // "Can't diagnose precisely if claims are wrong"
        JudgmentContext {
            id: "cred_to_diag".into(),
            relation_type: RelationType::Influence,
            target_node_id: "diagnosis_precision".into(),
            source_cluster_id: "credibility".into(),
            prompt_text: "How much does each credibility attribute influence the quality \
                          of diagnosis precision? Truthful claims and epistemic honesty \
                          are prerequisites for precise diagnosis."
                .into(),
            semantics_version: 1,
            judgment_kind: JudgmentKind::ComposableRatio,
            incoming_cluster_weight: Some(1.0),
        },
        JudgmentContext {
            id: "cred_to_causal".into(),
            relation_type: RelationType::Influence,
            target_node_id: "causal_depth".into(),
            source_cluster_id: "credibility".into(),
            prompt_text: "How much does each credibility attribute influence causal depth? \
                          Tracing root causes requires truthful grounding."
                .into(),
            semantics_version: 1,
            judgment_kind: JudgmentKind::ComposableRatio,
            incoming_cluster_weight: Some(0.8),
        },
        JudgmentContext {
            id: "cred_to_comp".into(),
            relation_type: RelationType::Influence,
            target_node_id: "compositional_awareness".into(),
            source_cluster_id: "credibility".into(),
            prompt_text: "How much does each credibility attribute influence compositional \
                          awareness? System-level reasoning requires honest uncertainty."
                .into(),
            semantics_version: 1,
            judgment_kind: JudgmentKind::ComposableRatio,
            incoming_cluster_weight: Some(0.6),
        },
        // Understanding → Robustness (ComposableRatio)
        // "Deep understanding enables better failure imagination"
        JudgmentContext {
            id: "understand_to_failure".into(),
            relation_type: RelationType::Influence,
            target_node_id: "failure_imagination".into(),
            source_cluster_id: "understanding".into(),
            prompt_text: "How much does each understanding attribute influence failure \
                          imagination? Precise diagnosis and causal depth enable concrete \
                          failure scenarios."
                .into(),
            semantics_version: 1,
            judgment_kind: JudgmentKind::ComposableRatio,
            incoming_cluster_weight: Some(1.0),
        },
        JudgmentContext {
            id: "understand_to_feasibility".into(),
            relation_type: RelationType::Influence,
            target_node_id: "feasibility".into(),
            source_cluster_id: "understanding".into(),
            prompt_text: "How much does each understanding attribute influence feasibility \
                          assessment? Understanding the system deeply enables realistic \
                          proposals."
                .into(),
            semantics_version: 1,
            judgment_kind: JudgmentKind::ComposableRatio,
            incoming_cluster_weight: Some(0.8),
        },
        // Understanding → Intervention (ComposableRatio)
        // "Precise diagnosis enables economical intervention"
        JudgmentContext {
            id: "understand_to_economy".into(),
            relation_type: RelationType::Influence,
            target_node_id: "intervention_economy".into(),
            source_cluster_id: "understanding".into(),
            prompt_text: "How much does each understanding attribute influence intervention \
                          economy? Precise diagnosis enables targeted, minimal changes."
                .into(),
            semantics_version: 1,
            judgment_kind: JudgmentKind::ComposableRatio,
            incoming_cluster_weight: Some(1.0),
        },
        // Robustness → Intervention (PairwiseOnlyRatio, local)
        // "Feasibility constrains intervention quality"
        JudgmentContext {
            id: "robust_to_economy".into(),
            relation_type: RelationType::Influence,
            target_node_id: "intervention_economy".into(),
            source_cluster_id: "robustness".into(),
            prompt_text: "How much does each robustness attribute locally influence \
                          intervention economy? Feasibility constrains what interventions \
                          are practical."
                .into(),
            semantics_version: 1,
            judgment_kind: JudgmentKind::PairwiseOnlyRatio,
            incoming_cluster_weight: Some(0.6),
        },
        // Credibility → Intervention (PairwiseOnlyRatio, local)
        // "Truthfulness influences taste"
        JudgmentContext {
            id: "cred_to_taste".into(),
            relation_type: RelationType::Influence,
            target_node_id: "taste".into(),
            source_cluster_id: "credibility".into(),
            prompt_text: "How much does each credibility attribute locally influence \
                          prioritization taste? Good judgment requires honest grounding."
                .into(),
            semantics_version: 1,
            judgment_kind: JudgmentKind::PairwiseOnlyRatio,
            incoming_cluster_weight: Some(0.5),
        },
    ];

    AnpNetwork {
        clusters,
        nodes,
        contexts,
    }
}

// =============================================================================
// Pipeline errors
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Generation failed for model {model}: {source}")]
    Generation {
        model: String,
        source: crate::gateway::error::ProviderError,
    },
    #[error("All generations failed")]
    AllGenerationsFailed,
    #[error("Ranking failed: {0}")]
    Ranking(#[from] crate::rerank::MultiRerankError),
    #[error("Synthesis failed: {0}")]
    Synthesis(crate::gateway::error::ProviderError),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Trace error: {0}")]
    Trace(#[from] crate::rerank::TraceError),
}

// =============================================================================
// Generate phase
// =============================================================================

async fn generate_all(
    gateway: &dyn ChatGateway,
    req: &PipelineRequest,
) -> Vec<Result<GenerationOutput, PipelineError>> {
    let tasks = req.models.iter().map(|model_id| {
        let messages = build_generation_messages(req);
        let chat_req = ChatRequest::new(
            ChatModel::openrouter(model_id),
            messages,
            Attribution::new("pipeline::generate"),
        )
        .temperature(req.generation_temperature)
        .max_tokens(req.max_generation_tokens);

        let model_id = model_id.clone();
        async move {
            let start = Instant::now();
            match gateway.chat(chat_req).await {
                Ok(resp) => Ok(GenerationOutput {
                    model: model_id,
                    content: resp.content,
                    input_tokens: resp.input_tokens,
                    output_tokens: resp.output_tokens,
                    cost_nanodollars: resp.cost_nanodollars,
                    latency_ms: start.elapsed().as_millis() as u64,
                }),
                Err(e) => Err(PipelineError::Generation {
                    model: model_id,
                    source: e,
                }),
            }
        }
    });

    stream::iter(tasks)
        .buffer_unordered(req.models.len())
        .collect()
        .await
}

fn build_generation_messages(req: &PipelineRequest) -> Vec<Message> {
    let mut messages = Vec::new();

    // System prompt + context files
    let mut system_parts = Vec::new();
    if let Some(sys) = &req.system_prompt {
        system_parts.push(sys.clone());
    }

    if !req.context_files.is_empty() {
        let context_block = build_context_block(&req.context_files, req.max_context_tokens);
        if !context_block.is_empty() {
            system_parts.push(context_block);
        }
    }

    if !system_parts.is_empty() {
        messages.push(Message::system(&system_parts.join("\n\n")));
    }

    messages.push(Message::user(&req.prompt));
    messages
}

/// Build a context block from context files, respecting an optional token budget.
fn build_context_block(files: &[ContextFile], max_tokens: Option<usize>) -> String {
    let mut block = String::from("## Context Files\n");
    let mut total_chars = block.len();

    // Rough approximation: 1 token ≈ 4 chars
    let char_budget = max_tokens.map(|t| t * 4);

    for file in files {
        let header = format!("\n### {}\n```\n", file.path);
        let footer = "\n```\n";
        let entry_len = header.len() + file.content.len() + footer.len();

        if let Some(budget) = char_budget {
            if total_chars + entry_len > budget {
                // Try to fit a truncated version
                let remaining =
                    budget.saturating_sub(total_chars + header.len() + footer.len() + 20);
                if remaining < 100 {
                    block.push_str("\n\n*[remaining context files truncated due to token budget]*");
                    break;
                }
                block.push_str(&header);
                block.push_str(&file.content[..remaining.min(file.content.len())]);
                block.push_str("\n... [truncated]");
                block.push_str(footer);
                block.push_str("\n*[remaining context files truncated due to token budget]*");
                break;
            }
        }

        block.push_str(&header);
        block.push_str(&file.content);
        block.push_str(footer);
        total_chars += entry_len;
    }

    block
}

/// Load context files from disk by reading file paths.
pub fn load_context_files(paths: &[String]) -> Result<Vec<ContextFile>, std::io::Error> {
    let mut files = Vec::new();
    for path_str in paths {
        let path = Path::new(path_str);
        let content = std::fs::read_to_string(path)?;
        files.push(ContextFile {
            path: path_str.clone(),
            content,
        });
    }
    Ok(files)
}

/// Expand glob patterns into file paths.
pub fn expand_context_globs(patterns: &[String]) -> Result<Vec<String>, std::io::Error> {
    let mut paths = Vec::new();
    for pattern in patterns {
        // Try as glob first
        if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
            match glob::glob(pattern) {
                Ok(entries) => {
                    for entry in entries {
                        match entry {
                            Ok(path) => {
                                if path.is_file() {
                                    paths.push(path.display().to_string());
                                }
                            }
                            Err(e) => {
                                eprintln!("[pipeline] glob error for {}: {}", pattern, e);
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[pipeline] invalid glob pattern {}: {}", pattern, e);
                }
            }
        } else {
            // Plain file path
            paths.push(pattern.clone());
        }
    }
    Ok(paths)
}

// =============================================================================
// Rank phase
// =============================================================================

fn build_rerank_request(
    req: &PipelineRequest,
    generations: &[GenerationOutput],
    gates: &[MultiRerankGateSpec],
) -> MultiRerankRequest {
    let entities: Vec<MultiRerankEntity> = generations
        .iter()
        .map(|g| MultiRerankEntity {
            id: g.model.clone(),
            text: g.content.clone(),
        })
        .collect();

    let mut attributes: Vec<MultiRerankAttributeSpec> = req
        .attributes
        .iter()
        .map(|a| MultiRerankAttributeSpec {
            id: a.id.clone(),
            prompt: a.prompt.clone(),
            prompt_template_slug: None,
            weight: a.weight,
        })
        .collect();

    // Ensure gate attributes exist in the attribute list
    for gate in gates {
        if !attributes.iter().any(|a| a.id == gate.attribute_id) {
            // Add requirement_alignment as a zero-weight attribute for gating
            if gate.attribute_id == "requirement_alignment" {
                let ra = requirement_alignment_attribute();
                attributes.push(MultiRerankAttributeSpec {
                    id: ra.id,
                    prompt: ra.prompt,
                    prompt_template_slug: None,
                    weight: 0.0, // gate-only, doesn't affect scoring
                });
            }
        }
    }

    let k = req.rank_config.k.min(entities.len());

    MultiRerankRequest {
        entities,
        attributes,
        topk: MultiRerankTopKSpec {
            k,
            weight_exponent: 1.3,
            tolerated_error: req.rank_config.tolerated_error,
            band_size: k.max(2),
            effective_resistance_max_active: 64,
            stop_sigma_inflate: 1.25,
            stop_min_consecutive: 2,
        },
        gates: gates.to_vec(),
        comparison_budget: req.rank_config.comparison_budget,
        latency_budget_ms: None,
        model: req.rank_config.judge_model.clone(),
        rater_id: None,
        comparison_concurrency: None,
        max_pair_repeats: None,
    }
}

// =============================================================================
// Synthesize phase
// =============================================================================

fn build_synthesis_prompt(
    req: &PipelineRequest,
    generations: &[GenerationOutput],
    ranking: &MultiRerankResponse,
) -> Vec<Message> {
    let system = req.synthesis_system_prompt.clone().unwrap_or_else(|| {
        "You are an expert technical synthesis agent. You receive multiple responses \
         to the same task from different AI models, ranked on multiple quality dimensions \
         using rigorous pairwise comparison. Your job: produce one response that is \
         strictly better than any individual input by taking the strongest elements from \
         each, resolving contradictions (favoring higher-ranked sources), and adding \
         any insights that emerge from seeing all responses together. Be concrete, \
         specific, and actionable."
            .to_string()
    });

    // Build the ranked generations section
    let mut ranked_section = String::new();

    // Map entity id → generation content
    let gen_by_model: HashMap<&str, &str> = generations
        .iter()
        .map(|g| (g.model.as_str(), g.content.as_str()))
        .collect();

    for entity in &ranking.entities {
        let rank = entity.rank.unwrap_or(0);
        let content = gen_by_model.get(entity.id.as_str()).unwrap_or(&"");

        ranked_section.push_str(&format!("\n### Rank {rank}: {}\n", entity.id));
        ranked_section.push_str(&format!(
            "Utility: {:.3} (std: {:.3}, p_flip: {:.3})\n",
            entity.u_mean, entity.u_std, entity.p_flip
        ));

        // Per-attribute scores
        if !entity.attribute_scores.is_empty() {
            ranked_section.push_str("Scores: ");
            let mut scores: Vec<_> = entity.attribute_scores.iter().collect();
            scores.sort_by_key(|(k, _)| (*k).clone());
            for (attr_id, summary) in &scores {
                ranked_section.push_str(&format!("{}={:.2} ", attr_id, summary.latent_mean));
            }
            ranked_section.push('\n');
        }

        ranked_section.push_str("\n```\n");
        ranked_section.push_str(content);
        ranked_section.push_str("\n```\n");
    }

    let meta = &ranking.meta;
    let user_prompt = format!(
        "## Original Task\n\n{}\n\n\
         ## Ranked Responses ({} models, {} comparisons, top-k error: {:.4})\n\
         {}\n\n\
         ## Instructions\n\n\
         Synthesize the best possible response to the original task. \
         Take the strongest elements from each ranked response, resolve any contradictions \
         (favoring higher-ranked sources), and produce a single, coherent, actionable output. \
         If you see insights that none of the individual responses captured, include those too.\n\n\
         Produce ONLY the synthesized response — no meta-commentary about the ranking process.",
        req.prompt, meta.model_used, meta.comparisons_used, meta.global_topk_error, ranked_section,
    );

    vec![Message::system(system), Message::user(user_prompt)]
}

// =============================================================================
// Full pipeline
// =============================================================================

/// Run the full generate → rank → synthesize pipeline.
pub async fn run_pipeline(
    gateway: Arc<dyn ChatGateway>,
    cache: Option<&dyn PairwiseCache>,
    model_policy: Option<Arc<dyn ModelPolicy>>,
    trace: Option<&dyn TraceSink>,
    mut req: PipelineRequest,
    gates: Vec<MultiRerankGateSpec>,
) -> Result<PipelineSession, PipelineError> {
    let session_id = Uuid::new_v4();

    // Resolve preset if models is empty
    if req.models.is_empty() {
        if let Some(preset) = req.preset {
            req.models = preset.models();
        }
    }

    // Validate
    if req.models.is_empty() {
        return Err(PipelineError::InvalidRequest(
            "models must not be empty (provide --models or --preset)".into(),
        ));
    }
    if req.models.len() < 2 {
        return Err(PipelineError::InvalidRequest(
            "pipeline requires at least 2 models for meaningful comparison".into(),
        ));
    }
    if req.attributes.is_empty() {
        return Err(PipelineError::InvalidRequest(
            "attributes must not be empty".into(),
        ));
    }

    // --- Phase 1: Generate ---
    eprintln!("[pipeline] generating from {} models...", req.models.len());
    let gen_results = generate_all(gateway.as_ref(), &req).await;

    let mut generations: Vec<GenerationOutput> = Vec::new();
    let mut gen_errors: Vec<String> = Vec::new();

    for result in gen_results {
        match result {
            Ok(gen) => {
                eprintln!(
                    "[pipeline]   {} — {} tokens, {}ms",
                    gen.model, gen.output_tokens, gen.latency_ms
                );
                generations.push(gen);
            }
            Err(PipelineError::Generation { model, source }) => {
                eprintln!("[pipeline]   {} — FAILED: {}", model, source);
                gen_errors.push(format!("{}: {}", model, source));
            }
            Err(e) => {
                gen_errors.push(e.to_string());
            }
        }
    }

    if generations.len() < 2 {
        return Err(PipelineError::AllGenerationsFailed);
    }

    let generation_cost: i64 = generations.iter().map(|g| g.cost_nanodollars).sum();

    // --- Phase 2: Rank ---
    eprintln!(
        "[pipeline] ranking {} generations on {} attributes...",
        generations.len(),
        req.attributes.len()
    );

    let rerank_req = build_rerank_request(&req, &generations, &gates);
    let ranking = multi_rerank_with_trace(
        gateway.clone(),
        cache,
        model_policy,
        Some(&RerankRunOptions {
            rng_seed: None,
            cache_only: false,
        }),
        rerank_req,
        Attribution::new("pipeline::rank"),
        None,
        None,
        trace,
        None,
    )
    .await?;

    let ranking_cost = ranking.meta.provider_cost_nanodollars;
    eprintln!(
        "[pipeline]   {} comparisons, top-k error: {:.4}, stop: {:?}",
        ranking.meta.comparisons_used, ranking.meta.global_topk_error, ranking.meta.stop_reason
    );

    // Print ranking summary
    for entity in &ranking.entities {
        let rank = entity.rank.unwrap_or(0);
        eprintln!(
            "[pipeline]   #{} {} (u={:.3}, p_flip={:.3})",
            rank, entity.id, entity.u_mean, entity.p_flip
        );
    }

    // --- Phase 3: Synthesize ---
    eprintln!("[pipeline] synthesizing with {}...", req.synthesis_model);
    let synth_messages = build_synthesis_prompt(&req, &generations, &ranking);
    let synth_req = ChatRequest::new(
        ChatModel::openrouter(&req.synthesis_model),
        synth_messages,
        Attribution::new("pipeline::synthesize"),
    )
    .temperature(req.synthesis_temperature)
    .max_tokens(req.max_synthesis_tokens);

    let synth_start = Instant::now();
    let synth_resp = gateway
        .chat(synth_req)
        .await
        .map_err(PipelineError::Synthesis)?;
    let synth_latency = synth_start.elapsed().as_millis() as u64;

    eprintln!(
        "[pipeline]   synthesis: {} tokens, {}ms",
        synth_resp.output_tokens, synth_latency
    );

    let synthesis = SynthesisOutput {
        model: req.synthesis_model.clone(),
        content: synth_resp.content,
        input_tokens: synth_resp.input_tokens,
        output_tokens: synth_resp.output_tokens,
        cost_nanodollars: synth_resp.cost_nanodollars,
        latency_ms: synth_latency,
    };

    let synthesis_cost = synthesis.cost_nanodollars;

    let cost = PipelineCost {
        generation_cost_nanodollars: generation_cost,
        ranking_cost_nanodollars: ranking_cost,
        synthesis_cost_nanodollars: synthesis_cost,
        total_cost_nanodollars: generation_cost
            .saturating_add(ranking_cost)
            .saturating_add(synthesis_cost),
    };

    let total_dollars = cost.total_cost_nanodollars as f64 / 1_000_000_000.0;
    eprintln!("[pipeline] done — total cost: ${:.4}", total_dollars);

    Ok(PipelineSession {
        id: session_id,
        created_at: Utc::now().to_rfc3339(),
        request: req,
        generations,
        ranking,
        synthesis,
        cost,
    })
}

/// Convenience: run pipeline and write trace to a file.
pub async fn run_pipeline_with_trace_file(
    gateway: Arc<dyn ChatGateway>,
    cache: Option<&dyn PairwiseCache>,
    model_policy: Option<Arc<dyn ModelPolicy>>,
    trace_path: Option<std::path::PathBuf>,
    req: PipelineRequest,
    gates: Vec<MultiRerankGateSpec>,
) -> Result<PipelineSession, PipelineError> {
    let (trace_sink, trace_worker) = if let Some(path) = trace_path {
        let (sink, worker) = JsonlTraceSink::new(path)?;
        (Some(sink), Some(worker))
    } else {
        (None, None)
    };
    let trace_ref = trace_sink.as_ref().map(|s| s as &dyn TraceSink);

    let result = run_pipeline(gateway, cache, model_policy, trace_ref, req, gates).await;

    drop(trace_sink);
    if let Some(worker) = trace_worker {
        let _ = worker.join();
    }

    result
}

// =============================================================================
// Flywheel: batch pipeline execution from a task manifest
// =============================================================================

/// A task manifest for batch pipeline execution.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FlywheelManifest {
    /// Tasks to run through the pipeline.
    pub tasks: Vec<FlywheelTask>,
    /// Default model preset for all tasks.
    #[serde(default)]
    pub preset: Option<ModelPreset>,
    /// Context files shared across all tasks.
    #[serde(default)]
    pub context_files: Vec<ContextFile>,
    /// Attributes to rank on (defaults to extended attributes if omitted).
    #[serde(default)]
    pub attributes: Option<Vec<PipelineAttribute>>,
    /// Model for final synthesis (per-task override possible).
    #[serde(default)]
    pub synthesis_model: Option<String>,
    /// Ranking configuration shared across tasks.
    #[serde(default)]
    pub rank_config: Option<PipelineRankConfig>,
    /// Maximum token budget for context files.
    #[serde(default)]
    pub max_context_tokens: Option<usize>,
}

/// A single task in a flywheel manifest.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FlywheelTask {
    /// Unique identifier for this task.
    pub id: String,
    /// The prompt for this task.
    pub prompt: String,
    /// Optional system prompt override.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Task-specific context files (appended to shared context).
    #[serde(default)]
    pub extra_context_files: Vec<ContextFile>,
    /// Optional per-task model list override.
    #[serde(default)]
    pub models: Option<Vec<String>>,
    /// Optional per-task synthesis model override.
    #[serde(default)]
    pub synthesis_model: Option<String>,
}

/// Result of a single flywheel task.
#[derive(Debug, Serialize, Deserialize)]
pub struct FlywheelResult {
    pub task_id: String,
    pub session: PipelineSession,
}

/// Summary of a flywheel run.
#[derive(Debug, Serialize, Deserialize)]
pub struct FlywheelSummary {
    pub tasks_completed: usize,
    pub tasks_failed: usize,
    pub total_cost_nanodollars: i64,
    pub task_summaries: Vec<FlywheelTaskSummary>,
}

/// Per-task summary in flywheel output.
#[derive(Debug, Serialize, Deserialize)]
pub struct FlywheelTaskSummary {
    pub task_id: String,
    pub success: bool,
    pub cost_nanodollars: i64,
    pub top_model: Option<String>,
    pub error: Option<String>,
}

/// Run a flywheel: iterate tasks from a manifest and run the pipeline for each.
pub async fn run_flywheel(
    gateway: Arc<dyn ChatGateway>,
    cache: Option<&dyn PairwiseCache>,
    model_policy: Option<Arc<dyn ModelPolicy>>,
    manifest: FlywheelManifest,
    out_dir: &Path,
    synthesis_out_dir: Option<&Path>,
    trace_dir: Option<&Path>,
    preset_override: Option<ModelPreset>,
    parallel: usize,
    gates: Vec<MultiRerankGateSpec>,
) -> FlywheelSummary {
    let attributes = manifest
        .attributes
        .clone()
        .unwrap_or_else(default_extended_attributes);

    let default_synth = manifest
        .synthesis_model
        .clone()
        .unwrap_or_else(|| "anthropic/claude-opus-4-6".into());

    let rank_config = manifest.rank_config.clone().unwrap_or_default();

    let preset = preset_override.or(manifest.preset);

    let task_count = manifest.tasks.len();
    eprintln!(
        "[flywheel] running {} tasks (parallel={})",
        task_count, parallel
    );

    let tasks = manifest.tasks;
    let mut task_summaries = Vec::with_capacity(task_count);
    let mut tasks_completed = 0usize;
    let mut tasks_failed = 0usize;
    let mut total_cost = 0i64;

    // Process tasks, potentially in parallel via stream
    let task_futures = tasks.into_iter().enumerate().map(|(idx, task)| {
        let gateway = gateway.clone();
        let model_policy = model_policy.clone();
        let attributes = attributes.clone();
        let default_synth = default_synth.clone();
        let rank_config = rank_config.clone();
        let shared_context = manifest.context_files.clone();
        let max_context_tokens = manifest.max_context_tokens;
        let out_dir = out_dir.to_path_buf();
        let synthesis_out_dir = synthesis_out_dir.map(|p| p.to_path_buf());
        let trace_dir = trace_dir.map(|p| p.to_path_buf());
        let gates = gates.clone();

        async move {
            let task_id = task.id.clone();
            eprintln!(
                "[flywheel] [{}/{}] starting task: {}",
                idx + 1,
                task_count,
                task_id
            );

            // Merge context files: shared + task-specific
            let mut context_files = shared_context;
            context_files.extend(task.extra_context_files.clone());

            // Build models list
            let models = task
                .models
                .clone()
                .unwrap_or_else(|| preset.map(|p| p.models()).unwrap_or_default());

            let synthesis_model = task
                .synthesis_model
                .clone()
                .unwrap_or_else(|| default_synth.clone());

            let req = PipelineRequest {
                prompt: task.prompt.clone(),
                system_prompt: task.system_prompt.clone(),
                models,
                preset,
                context_files,
                max_context_tokens,
                attributes: attributes.clone(),
                synthesis_model,
                synthesis_system_prompt: None,
                generation_temperature: 0.7,
                synthesis_temperature: 0.3,
                max_generation_tokens: 4096,
                max_synthesis_tokens: 8192,
                rank_config: rank_config.clone(),
            };

            let trace_path = trace_dir
                .as_ref()
                .map(|d| d.join(format!("{}.trace.jsonl", task_id)));

            let result =
                run_pipeline_with_trace_file(gateway, cache, model_policy, trace_path, req, gates)
                    .await;

            match result {
                Ok(session) => {
                    // Write session JSON
                    let session_path = out_dir.join(format!("{}.json", task_id));
                    if let Ok(json) = serde_json::to_string_pretty(&session) {
                        let _ = std::fs::write(&session_path, json);
                    }

                    // Write synthesis text
                    if let Some(ref synth_dir) = synthesis_out_dir {
                        let synth_path = synth_dir.join(format!("{}.md", task_id));
                        let _ = std::fs::write(&synth_path, &session.synthesis.content);
                    }

                    let top_model = session
                        .ranking
                        .entities
                        .iter()
                        .find(|e| e.rank == Some(1))
                        .map(|e| e.id.clone());

                    let cost = session.cost.total_cost_nanodollars;

                    eprintln!(
                        "[flywheel] [{}/{}] done: {} (${:.4})",
                        idx + 1,
                        task_count,
                        task_id,
                        cost as f64 / 1_000_000_000.0
                    );

                    (task_id, true, cost, top_model, None)
                }
                Err(e) => {
                    let err_msg = e.to_string();
                    eprintln!(
                        "[flywheel] [{}/{}] FAILED: {} — {}",
                        idx + 1,
                        task_count,
                        task_id,
                        err_msg
                    );
                    (task_id, false, 0, None, Some(err_msg))
                }
            }
        }
    });

    // Execute with concurrency control
    let results: Vec<_> = stream::iter(task_futures)
        .buffer_unordered(parallel.max(1))
        .collect()
        .await;

    for (task_id, success, cost, top_model, error) in results {
        if success {
            tasks_completed += 1;
        } else {
            tasks_failed += 1;
        }
        total_cost = total_cost.saturating_add(cost);
        task_summaries.push(FlywheelTaskSummary {
            task_id,
            success,
            cost_nanodollars: cost,
            top_model,
            error,
        });
    }

    let summary = FlywheelSummary {
        tasks_completed,
        tasks_failed,
        total_cost_nanodollars: total_cost,
        task_summaries,
    };

    let total_dollars = total_cost as f64 / 1_000_000_000.0;
    eprintln!(
        "[flywheel] complete — {} succeeded, {} failed, total cost: ${:.4}",
        tasks_completed, tasks_failed, total_dollars
    );

    summary
}
