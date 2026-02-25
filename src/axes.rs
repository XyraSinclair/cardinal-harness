//! Orthogonal evaluation axes for code and idea assessment.
//!
//! Three clusters, twelve axes, designed for genuine orthogonality and
//! exhaustive coverage of the evaluation space. Each axis is ratio-scalable
//! ("A has 3x more groundedness than B") and operationally grounded.
//!
//! ## Cluster structure
//!
//! - **Epistemic** (quality of understanding): groundedness, calibration,
//!   resolution, causal_depth, compositional_reach
//! - **Instrumental** (quality of proposed action): leverage, robustness,
//!   option_value, economy
//! - **Strategic** (meta-judgment about context): information_value,
//!   temporal_shape, prioritization
//!
//! ## ANP influence edges
//!
//! - Epistemic → Instrumental (ComposableRatio): better understanding enables
//!   better intervention design
//! - Epistemic → Strategic (ComposableRatio): calibration enables information
//!   value assessment
//! - Strategic → Instrumental (PairwiseOnlyRatio): temporal shape and
//!   prioritization constrain which interventions make sense (local, non-propagatable)
//!
//! ## Relationship to prior axes
//!
//! These supersede `default_assessment_attributes()` and
//! `default_extended_attributes()` in `pipeline.rs`. The old axes are retained
//! for backward compatibility with existing cached pairwise judgments.
//!
//! Migration mapping:
//! - truthfulness → groundedness (narrowed: evidence-backed claims only)
//! - epistemic_integrity → calibration (narrowed: confidence/uncertainty match)
//! - diagnosis_precision → resolution (renamed for generality beyond diagnostics)
//! - causal_depth → causal_depth (retained, sharpened)
//! - compositional_awareness → compositional_reach (renamed, boundary-crossing focus)
//! - intervention_economy → economy (broadened: all resource costs)
//! - failure_imagination → robustness (reframed: predictive, not just imaginative)
//! - taste → prioritization (operationalized: proportion of attention to impact)
//! - verifiability → groundedness (subsumed: checkable evidence is groundedness)
//! - feasibility → economy (subsumed: resource cost includes feasibility)
//! - (new) leverage, option_value, information_value, temporal_shape

use crate::anp::{AnpNetwork, Cluster, JudgmentContext, JudgmentKind, Node, RelationType};
use crate::pipeline::PipelineAttribute;

// =============================================================================
// Axis definitions
// =============================================================================

/// The twelve orthogonal evaluation axes across three clusters.
///
/// Designed for both code evaluation and idea evaluation. The context
/// (codebase analysis vs. strategic idea assessment) determines weighting,
/// not which axes apply — all twelve are always meaningful.
pub fn orthogonal_axes() -> Vec<PipelineAttribute> {
    let mut axes = Vec::with_capacity(12);
    axes.extend(epistemic_axes());
    axes.extend(instrumental_axes());
    axes.extend(strategic_axes());
    axes
}

// =============================================================================
// Cluster 1: Epistemic (quality of understanding)
// =============================================================================

/// Epistemic axes: how well does this artifact improve our understanding?
pub fn epistemic_axes() -> Vec<PipelineAttribute> {
    vec![
        // -----------------------------------------------------------------
        // 1. Groundedness
        // -----------------------------------------------------------------
        PipelineAttribute {
            id: "groundedness".into(),
            prompt: concat!(
                "Groundedness measures whether claims are anchored to checkable ",
                "evidence — specific file paths, function signatures, data structures, ",
                "error messages, benchmark numbers, or other artifacts that an engineer ",
                "could locate and verify. A grounded response says 'the DashMap in ",
                "crates/shoal/src/registry.rs line 47 uses get() without holding the ",
                "read guard across the subsequent insert, creating a TOCTOU window.' ",
                "An ungrounded response says 'there may be concurrency issues in the ",
                "registry layer.' Groundedness is NOT the same as correctness — a ",
                "response can be grounded and wrong (the file reference is real but ",
                "the analysis is incorrect), or correct and ungrounded (the claim is ",
                "true but provides no way to check it). The test is operational: for ",
                "each substantive claim, could you walk to a terminal and verify it ",
                "in under five minutes? Count the claims that pass this test versus ",
                "those that require an open-ended investigation before you even know ",
                "if they're true. A response where 8 out of 10 claims have concrete ",
                "evidence pointers is more grounded than one where 2 out of 3 do, ",
                "even though the ratio is lower — total grounded surface area matters. ",
                "Watch for the failure mode of grounding only the easy claims (file ",
                "exists, function takes these arguments) while leaving the hard claims ",
                "(performance characteristics, race conditions, emergent behavior) ",
                "floating without evidence.",
            )
            .into(),
            weight: 1.0,
        },
        // -----------------------------------------------------------------
        // 2. Calibration
        // -----------------------------------------------------------------
        PipelineAttribute {
            id: "calibration".into(),
            prompt: concat!(
                "Calibration measures whether a response's expressed confidence ",
                "matches its actual epistemic state — whether the uncertainty ",
                "signaling is honest and useful. A well-calibrated response applies ",
                "a clear assurance ladder to its own claims: 'I can verify this by ",
                "reading the code' versus 'this follows from general Rust ownership ",
                "semantics' versus 'I'm pattern-matching from similar systems, this ",
                "needs checking' versus 'this is speculation.' The critical test: if ",
                "you took every claim the response presents at high confidence and ",
                "checked them, what fraction would hold up? And conversely: does the ",
                "response flag genuine uncertainties, or does it paper over gaps with ",
                "fluent prose? A response that makes two careful claims — both correct, ",
                "both flagged with their evidence basis — is better calibrated than ",
                "one that makes ten authoritative-sounding claims where seven are right ",
                "and three are confabulated, even though the latter 'knows more.' The ",
                "deepest form of calibration is identifying the specific questions ",
                "whose answers would change the analysis: 'if the Spiel batch window ",
                "is configurable (I haven't checked), this race condition may not ",
                "exist in production.' Calibration is NOT timidity — a well-calibrated ",
                "response can be bold when its evidence is strong. The failure mode ",
                "is uniform confidence: treating every claim as equally certain, which ",
                "destroys the reader's ability to triage what to trust versus what to ",
                "verify. Another failure mode is performative hedging — adding 'might' ",
                "and 'possibly' to every sentence without actually distinguishing ",
                "high-confidence claims from low-confidence ones.",
            )
            .into(),
            weight: 0.95,
        },
        // -----------------------------------------------------------------
        // 3. Resolution
        // -----------------------------------------------------------------
        PipelineAttribute {
            id: "resolution".into(),
            prompt: concat!(
                "Resolution measures how specifically a response locates the thing ",
                "it's talking about — the granularity at which it carves reality. ",
                "This applies to problems, opportunities, mechanisms, and proposals ",
                "alike. The hierarchy from lowest to highest resolution: vague domain ",
                "('there are performance issues'), named subsystem ('Slate queries ",
                "are slow'), specific component ('the Datalog join executor in ",
                "slate::query::execute'), precise mechanism ('the join executor ",
                "materializes the full cross-product into a Vec before applying ",
                "filter predicates, causing O(n*m) allocation'), and finally ",
                "intervention-ready specificity ('the planner in query::optimize ",
                "lacks a PushSelectDown rule — adding one to the optimizer's rule ",
                "set would let filters execute during join enumeration instead of ",
                "after'). Resolution is distinct from groundedness: a response can ",
                "be highly grounded (every claim has a file reference) but low ",
                "resolution (the claims are all about the right file but never ",
                "narrow down which function, which data flow, which state ",
                "transition). Resolution is also distinct from causal depth: a ",
                "response can precisely locate a symptom (high resolution) without ",
                "tracing the cause (low depth). The unit of resolution is the ",
                "engineer's next action: does reading this tell you exactly where ",
                "to look, or does it tell you roughly where to start a search? ",
                "Discount responses that substitute jargon density for genuine ",
                "specificity — naming twelve architectural patterns is not the same ",
                "as locating one concrete mechanism.",
            )
            .into(),
            weight: 0.9,
        },
        // -----------------------------------------------------------------
        // 4. Causal depth
        // -----------------------------------------------------------------
        PipelineAttribute {
            id: "causal_depth".into(),
            prompt: concat!(
                "Causal depth measures how far a response traces the chain from ",
                "observation to origin — and how well it distinguishes symptoms, ",
                "proximate causes, root causes, and generative forces. Shallow: ",
                "'the build is slow.' One level deeper: 'the build is slow because ",
                "steel depends on nalgebra, which takes 40 seconds to compile.' ",
                "Deeper: 'steel depends on nalgebra because it was pulled in for a ",
                "single matrix multiply in the cost model — the actual operation is ",
                "a 3x3 determinant that could be a 20-line inline function.' Deepest: ",
                "'nalgebra was added during a sprint where the cost model needed ",
                "linear algebra and the team convention was to reach for crates.io ",
                "rather than inline — the same pattern explains the regex dependency ",
                "in config parsing and the chrono dependency for timestamp formatting. ",
                "The structural fix is a workspace policy against heavyweight deps ",
                "for single-function usage.' Notice the levels: symptom → proximate ",
                "cause → root technical cause → structural/organizational force that ",
                "generated the root cause. Depth is NOT length — a single sentence ",
                "that names the generative force is deeper than three paragraphs ",
                "elaborating on symptoms. Depth is also NOT causal chain length for ",
                "its own sake — what matters is reaching the level where intervention ",
                "has durable effect. A response that traces to the proximate cause ",
                "and proposes a fix there is less deep than one that identifies the ",
                "pattern generating similar problems elsewhere and proposes a ",
                "structural remedy, but deeper than one that only describes symptoms ",
                "in great detail. The gold standard: explaining why the codebase ",
                "evolved to this state, and what forces would need to change to ",
                "prevent recurrence.",
            )
            .into(),
            weight: 0.85,
        },
        // -----------------------------------------------------------------
        // 5. Compositional reach
        // -----------------------------------------------------------------
        PipelineAttribute {
            id: "compositional_reach".into(),
            prompt: concat!(
                "Compositional reach measures how well a response traces effects ",
                "across subsystem boundaries — understanding the system as a composed ",
                "whole rather than analyzing components in isolation. A response with ",
                "high compositional reach, when discussing a change to Slate's query ",
                "planner, traces the consequences through: Spool (which reads Slate ",
                "state to route messages), Shore (which queries Slate for connection ",
                "metadata), magma's effect dispatch (which uses Slate for capability ",
                "resolution), and any dynamic modules whose latency contracts depend ",
                "on Slate query performance. It understands shared resources that ",
                "create implicit coupling (the UnifiedMemoryPool budgets memory ",
                "across all engines; a memory-hungry Slate query starves Scree's ",
                "hot cache), data flows that cross crate boundaries (a Spiel oplog ",
                "entry triggers a Slate write which triggers a Spool event which ",
                "triggers a module handler), and emergent behaviors that no single ",
                "component's documentation describes. The hierarchy: isolated analysis ",
                "(one component at a time), interface-aware (understands API contracts ",
                "between components), runtime-aware (understands how state and ",
                "backpressure flow at runtime), and genuinely compositional (identifies ",
                "emergent properties of the composed system that the designers may ",
                "not have intended). Compositional reach is distinct from causal ",
                "depth: depth follows one causal chain to its origin; reach follows ",
                "one change to its many consequences across the system. A response ",
                "can be deep (traces one problem to its root) but narrow (doesn't ",
                "consider what else that root affects). The failure mode is mentioning ",
                "many subsystem names without tracing actual data flows or state ",
                "dependencies between them.",
            )
            .into(),
            weight: 0.8,
        },
    ]
}

// =============================================================================
// Cluster 2: Instrumental (quality of proposed action)
// =============================================================================

/// Instrumental axes: how effective is the proposed action?
pub fn instrumental_axes() -> Vec<PipelineAttribute> {
    vec![
        // -----------------------------------------------------------------
        // 6. Leverage
        // -----------------------------------------------------------------
        PipelineAttribute {
            id: "leverage".into(),
            prompt: concat!(
                "Leverage measures the ratio of impact to effort — how much ",
                "output a proposed action produces per unit of input. A high-leverage ",
                "change is one where a small, precisely targeted intervention creates ",
                "a disproportionately large improvement. Adding a PushSelectDown rule ",
                "to Slate's query optimizer (50 lines) that eliminates O(n*m) ",
                "materialization across all Datalog queries is extremely high leverage. ",
                "Rewriting the entire query engine for the same benefit is low leverage. ",
                "Leverage is NOT the same as impact — a change can have enormous impact ",
                "but also require enormous effort (low leverage), or modest impact ",
                "achieved with trivial effort (high leverage). It is also NOT the same ",
                "as economy — economy measures total cost, leverage measures the ratio. ",
                "A $10,000 project that saves $1,000,000/year has high leverage. A $100 ",
                "fix that saves $200/year has higher economy but lower leverage. The ",
                "hierarchy: negative leverage (the change creates more work than it ",
                "saves — common with premature abstractions), neutral (effort roughly ",
                "equals benefit), moderate (clear net positive, 2-5x return), high ",
                "(10x+ return, the change is a force multiplier), and structural ",
                "leverage (the change improves the rate at which future changes can ",
                "be made — like adding a test harness that makes all subsequent work ",
                "faster). Discount proposals that claim high leverage without ",
                "quantifying either the effort or the expected impact. Watch for the ",
                "failure mode of confusing activity with leverage: long lists of ",
                "changes that each individually make sense but collectively consume ",
                "more attention than they save.",
            )
            .into(),
            weight: 0.9,
        },
        // -----------------------------------------------------------------
        // 7. Robustness
        // -----------------------------------------------------------------
        PipelineAttribute {
            id: "robustness".into(),
            prompt: concat!(
                "Robustness measures how well a proposed solution performs across ",
                "the range of conditions it will actually encounter — not just the ",
                "happy path, but the realistic envelope of perturbations, scale ",
                "changes, environmental shifts, and adversarial inputs. A robust ",
                "proposal thinks in concrete scenarios: 'this lock-free queue works ",
                "at 10K msgs/s but what happens at 1M? The CAS loop will spin-wait ",
                "and burn CPU — we need a backoff strategy above the inflection ",
                "point.' It addresses timing (race conditions, ordering dependencies), ",
                "scale (what happens at 10x, 100x, 1000x current load), boundaries ",
                "(counter wraps, memory limits, timeout expiry), environmental change ",
                "(what if the dependency upgrades, the OS changes, the hardware ",
                "differs), and adversarial input (malformed data, hostile payloads, ",
                "resource exhaustion attacks). Robustness is NOT pessimism — it's ",
                "not about listing everything that could go wrong, it's about the ",
                "solution being designed so that realistic failure modes are handled ",
                "gracefully. A robust design degrades predictably rather than ",
                "failing catastrophically. The hierarchy: brittle (works only under ",
                "ideal conditions), tested (handles known edge cases), resilient ",
                "(degrades gracefully under stress), and antifragile (actually ",
                "improves from perturbation — like a retry with backoff that ",
                "discovers a faster path). Discount vague warnings ('this might not ",
                "scale') that don't specify the mechanism of failure or the ",
                "conditions under which robustness breaks down.",
            )
            .into(),
            weight: 0.85,
        },
        // -----------------------------------------------------------------
        // 8. Option value
        // -----------------------------------------------------------------
        PipelineAttribute {
            id: "option_value".into(),
            prompt: concat!(
                "Option value measures whether a proposed action preserves, creates, ",
                "or destroys future choices. Every decision narrows the space of ",
                "possible futures — option value tracks whether that narrowing is ",
                "deliberate and worthwhile or accidental and costly. A high option ",
                "value change opens doors: adding an abstraction boundary that lets ",
                "you swap storage engines later without rewriting consumers. Defining ",
                "a trait instead of hardcoding a concrete type. Choosing a data format ",
                "that's extensible (EAV with schema evolution) over one that's rigid ",
                "(fixed structs with no versioning). A low option value change closes ",
                "doors: committing to a specific wire format that every consumer ",
                "depends on, making it prohibitively expensive to change later. ",
                "Merging state that should stay separate, so future unbundling ",
                "requires archaeology. Adding a dependency that constrains your ",
                "minimum Rust version, platform support, or licensing. Option value ",
                "is distinct from robustness: a robust solution handles perturbations ",
                "within the current design; high option value means the design itself ",
                "can evolve. It's also distinct from leverage: a high-leverage change ",
                "produces big results now; high option value may produce no immediate ",
                "results but enables big results later. The hardest case: correctly ",
                "identifying one-way doors (irreversible commitments) versus two-way ",
                "doors (easily reversible experiments). Many proposals treat every ",
                "decision as equally consequential. The gold standard: explicitly ",
                "classifying which parts of a proposal are reversible, which aren't, ",
                "and designing the irreversible parts to be as narrow as possible.",
            )
            .into(),
            weight: 0.75,
        },
        // -----------------------------------------------------------------
        // 9. Economy
        // -----------------------------------------------------------------
        PipelineAttribute {
            id: "economy".into(),
            prompt: concat!(
                "Economy measures the fully-loaded cost of a proposed action — not ",
                "just the obvious implementation effort, but all the resources it ",
                "consumes: engineering time, cognitive load, review burden, test ",
                "surface expansion, operational complexity, migration risk, and ",
                "opportunity cost (what else could that time be spent on). An ",
                "economical proposal respects the existing codebase as a living ",
                "system with history, test coverage, and downstream dependents. It ",
                "proposes the minimum viable change rather than the maximum ",
                "aesthetically satisfying refactor. It distinguishes between what ",
                "MUST change to solve the problem and what COULD change to satisfy ",
                "preferences. Economy is NOT cheapness — the most economical path ",
                "for a genuinely complex problem may be an architecturally significant ",
                "change, because the cheap fix would accrue technical debt that costs ",
                "more over time. The question is whether every proposed change earns ",
                "its keep: does each edit directly contribute to solving the stated ",
                "problem, or is the response padding recommendations to appear ",
                "thorough? The hierarchy: wasteful (changes that create more work ",
                "than they save), adequate (solves the problem but at higher cost ",
                "than necessary), economical (minimum necessary changes, no padding), ",
                "and thrifty (identifies non-obvious ways to reduce cost — reusing ",
                "existing infrastructure, piggybacking on planned work, finding ",
                "changes that solve multiple problems simultaneously). Watch for: ",
                "unnecessary abstraction layers, premature generalization, cleanup ",
                "of adjacent code that isn't broken, and the endemic failure mode of ",
                "proposing six recommendations when the first two solve 90 percent ",
                "of the problem.",
            )
            .into(),
            weight: 0.8,
        },
    ]
}

// =============================================================================
// Cluster 3: Strategic (meta-judgment about context)
// =============================================================================

/// Strategic axes: how well is attention and effort allocated?
pub fn strategic_axes() -> Vec<PipelineAttribute> {
    vec![
        // -----------------------------------------------------------------
        // 10. Information value
        // -----------------------------------------------------------------
        PipelineAttribute {
            id: "information_value".into(),
            prompt: concat!(
                "Information value measures whether pursuing an idea or action ",
                "would reduce uncertainty about other important decisions — not just ",
                "solving the immediate problem but illuminating the landscape around ",
                "it. A high information value proposal is one where even partial ",
                "progress reveals something you couldn't learn any other way. Example: ",
                "'benchmarking Slate's Datalog joins under concurrent load would tell ",
                "us whether the performance ceiling is in the planner or the executor, ",
                "which determines whether the next six months of optimization work ",
                "targets the right component.' The benchmark itself might not ship, ",
                "but the information it produces has outsized value because it ",
                "redirects a large effort. Low information value means the outcome ",
                "is predictable regardless: 'adding this feature will definitely work, ",
                "the only question is how long it takes.' Information value is ",
                "distinct from leverage (a high-information experiment might have zero ",
                "direct impact) and distinct from economy (the experiment might be ",
                "expensive but worth it for what it reveals). The hierarchy: zero ",
                "information value (outcome is already known, doing this teaches ",
                "nothing), confirming (validates a belief already held — useful but ",
                "low surprise), discriminating (distinguishes between competing ",
                "hypotheses — tells you which fork to take), and restructuring (the ",
                "result would change the problem framing itself — revealing that the ",
                "question being asked is wrong). The gold standard: identifying ",
                "actions that are cheap to try and whose results would change what ",
                "you do next — probe experiments, minimum viable tests, reversible ",
                "spikes that generate maximum signal per dollar spent. Discount ",
                "proposals framed as 'we should investigate X' without specifying ",
                "what specific question the investigation answers and what you'd ",
                "do differently based on the result.",
            )
            .into(),
            weight: 0.7,
        },
        // -----------------------------------------------------------------
        // 11. Temporal shape
        // -----------------------------------------------------------------
        PipelineAttribute {
            id: "temporal_shape".into(),
            prompt: concat!(
                "Temporal shape measures how well a response understands the time ",
                "structure of value — whether an idea's worth is flat, decaying, ",
                "compounding, or step-function, and whether the response accounts ",
                "for this correctly. Different temporal shapes demand different ",
                "strategies. Compounding value (a test harness that makes every ",
                "subsequent change faster, a design pattern that reduces the cost ",
                "of future features) should be prioritized early even if the ",
                "immediate payoff is small. Decaying value (a market window closing, ",
                "a dependency about to be deprecated, a security vulnerability being ",
                "actively exploited) demands urgency proportional to the decay rate. ",
                "Flat value (a refactoring that's equally useful today or in six ",
                "months) can be scheduled for convenience. Step-function value (a ",
                "deadline, a conference, a release date) requires hitting the step ",
                "or getting zero value. A response with good temporal shape awareness ",
                "doesn't just say 'this is important' — it says 'this compounds: ",
                "every week we delay costs us not just that week's benefit but all ",
                "the derivative benefits that would have accumulated' or 'this is a ",
                "decaying opportunity: the API we'd integrate with is sunsetting in ",
                "Q3, after which this becomes 10x harder.' The failure mode is ",
                "treating all value as flat — scheduling compounding work alongside ",
                "flat work as if they're equivalent, or treating a decaying ",
                "opportunity with the same urgency (or lack thereof) as a stable one. ",
                "Another failure mode is false urgency — claiming decay or step-function ",
                "shape when the value is actually flat, in order to force prioritization. ",
                "The test: does the response explicitly identify the temporal shape of ",
                "each proposal, and does the prioritization respect those shapes?",
            )
            .into(),
            weight: 0.65,
        },
        // -----------------------------------------------------------------
        // 12. Prioritization
        // -----------------------------------------------------------------
        PipelineAttribute {
            id: "prioritization".into(),
            prompt: concat!(
                "Prioritization measures whether attention is allocated in proportion ",
                "to actual impact — the meta-skill of deciding what matters most and ",
                "spending words, analysis, and recommendations accordingly. This is ",
                "the axis that judges all other axes: a response can score well on ",
                "every other dimension and still fail on prioritization if it gives ",
                "equal weight to critical architectural risks and cosmetic style ",
                "preferences. A well-prioritized response identifies the 20 percent ",
                "of observations that account for 80 percent of the value, leads with ",
                "them, and explicitly deprioritizes or omits the rest. It might say: ",
                "'there are a dozen things I could flag here but only two matter: the ",
                "Spiel durability gap that risks data loss, and the Slate query ",
                "regression that blocks the release. Everything else can wait.' A ",
                "poorly prioritized response is a flat list where a critical race ",
                "condition sits next to a naming convention suggestion with no signal ",
                "about which one deserves action first. Prioritization is distinct from ",
                "temporal shape (which is about WHEN; prioritization is about WHAT), ",
                "from leverage (which is about efficiency; prioritization is about ",
                "selection), and from economy (which is about cost; prioritization is ",
                "about importance). The hierarchy: undifferentiated (everything is ",
                "listed equally), categorized (important vs. nice-to-have), ranked ",
                "(explicit ordering with reasoning), and triaged (ranked with explicit ",
                "cut lines: 'do these two now, defer these three to next sprint, ",
                "and actively decide not to do these four'). The deepest form of ",
                "prioritization argues for what should be deliberately left undone — ",
                "demonstrating that the author understands opportunity cost and ",
                "isn't just listing every observation to appear thorough.",
            )
            .into(),
            weight: 0.7,
        },
    ]
}

// =============================================================================
// ANP network for the orthogonal axes
// =============================================================================

/// Build the 3-cluster ANP network for the orthogonal evaluation axes.
///
/// Clusters:
/// - **Epistemic**: groundedness, calibration, resolution, causal_depth,
///   compositional_reach
/// - **Instrumental**: leverage, robustness, option_value, economy
/// - **Strategic**: information_value, temporal_shape, prioritization
///
/// Inter-cluster influences:
/// - Epistemic → Instrumental (ComposableRatio): understanding enables action
///   design. Groundedness and resolution directly constrain what interventions
///   can be precisely targeted. Causal depth determines whether fixes address
///   symptoms or causes.
/// - Epistemic → Strategic (ComposableRatio): calibration enables information
///   value assessment. You can only identify discriminating experiments if you
///   know what you don't know.
/// - Strategic → Instrumental (PairwiseOnlyRatio): temporal shape and
///   prioritization constrain which interventions make sense, but this influence
///   is contextual and should not propagate globally through the supermatrix.
pub fn build_orthogonal_anp_network() -> AnpNetwork {
    let clusters = vec![
        Cluster {
            id: "epistemic".into(),
            label: "Epistemic Quality".into(),
        },
        Cluster {
            id: "instrumental".into(),
            label: "Instrumental Quality".into(),
        },
        Cluster {
            id: "strategic".into(),
            label: "Strategic Quality".into(),
        },
    ];

    let nodes = vec![
        // Epistemic
        Node {
            id: "groundedness".into(),
            cluster_id: "epistemic".into(),
            label: "Groundedness".into(),
        },
        Node {
            id: "calibration".into(),
            cluster_id: "epistemic".into(),
            label: "Calibration".into(),
        },
        Node {
            id: "resolution".into(),
            cluster_id: "epistemic".into(),
            label: "Resolution".into(),
        },
        Node {
            id: "causal_depth".into(),
            cluster_id: "epistemic".into(),
            label: "Causal Depth".into(),
        },
        Node {
            id: "compositional_reach".into(),
            cluster_id: "epistemic".into(),
            label: "Compositional Reach".into(),
        },
        // Instrumental
        Node {
            id: "leverage".into(),
            cluster_id: "instrumental".into(),
            label: "Leverage".into(),
        },
        Node {
            id: "robustness".into(),
            cluster_id: "instrumental".into(),
            label: "Robustness".into(),
        },
        Node {
            id: "option_value".into(),
            cluster_id: "instrumental".into(),
            label: "Option Value".into(),
        },
        Node {
            id: "economy".into(),
            cluster_id: "instrumental".into(),
            label: "Economy".into(),
        },
        // Strategic
        Node {
            id: "information_value".into(),
            cluster_id: "strategic".into(),
            label: "Information Value".into(),
        },
        Node {
            id: "temporal_shape".into(),
            cluster_id: "strategic".into(),
            label: "Temporal Shape".into(),
        },
        Node {
            id: "prioritization".into(),
            cluster_id: "strategic".into(),
            label: "Prioritization".into(),
        },
    ];

    let contexts = vec![
        // =============================================================
        // Epistemic → Instrumental: understanding enables action design
        // =============================================================

        // Groundedness + Resolution → Leverage
        // Precisely located, evidence-backed understanding is what makes
        // high-leverage interventions possible. Without it you're guessing.
        JudgmentContext {
            id: "epistemic_to_leverage".into(),
            relation_type: RelationType::Influence,
            target_node_id: "leverage".into(),
            source_cluster_id: "epistemic".into(),
            prompt_text: concat!(
                "How much does each epistemic quality contribute to identifying ",
                "high-leverage interventions? Grounded, high-resolution understanding ",
                "lets you target the precise mechanism where a small change has large ",
                "effect. Without it, leverage estimates are guesswork.",
            )
            .into(),
            semantics_version: 2,
            judgment_kind: JudgmentKind::ComposableRatio,
            incoming_cluster_weight: Some(1.0),
        },
        // Causal depth + Compositional reach → Robustness
        // Tracing root causes and cross-boundary effects is how you
        // design solutions that don't break under perturbation.
        JudgmentContext {
            id: "epistemic_to_robustness".into(),
            relation_type: RelationType::Influence,
            target_node_id: "robustness".into(),
            source_cluster_id: "epistemic".into(),
            prompt_text: concat!(
                "How much does each epistemic quality contribute to robust solution ",
                "design? Deep causal understanding prevents symptom-patching. ",
                "Compositional reach prevents fixes that break other subsystems. ",
                "Calibration prevents overconfidence in untested assumptions.",
            )
            .into(),
            semantics_version: 2,
            judgment_kind: JudgmentKind::ComposableRatio,
            incoming_cluster_weight: Some(0.9),
        },
        // Causal depth → Option value
        // Understanding structural forces (not just symptoms) is what lets
        // you design changes that preserve future flexibility.
        JudgmentContext {
            id: "epistemic_to_option_value".into(),
            relation_type: RelationType::Influence,
            target_node_id: "option_value".into(),
            source_cluster_id: "epistemic".into(),
            prompt_text: concat!(
                "How much does each epistemic quality influence awareness of option ",
                "value? Causal depth reveals which decisions are reversible vs ",
                "irreversible. Compositional reach shows which abstractions are load-",
                "bearing vs superficial. Without these, you can't tell one-way doors ",
                "from two-way doors.",
            )
            .into(),
            semantics_version: 2,
            judgment_kind: JudgmentKind::ComposableRatio,
            incoming_cluster_weight: Some(0.7),
        },
        // Resolution + Groundedness → Economy
        // Precise, evidence-backed understanding is what lets you scope
        // work tightly and avoid wasted effort.
        JudgmentContext {
            id: "epistemic_to_economy".into(),
            relation_type: RelationType::Influence,
            target_node_id: "economy".into(),
            source_cluster_id: "epistemic".into(),
            prompt_text: concat!(
                "How much does each epistemic quality influence economical scoping? ",
                "High-resolution diagnosis prevents working on the wrong component. ",
                "Groundedness prevents investigating hallucinated issues. Calibration ",
                "prevents over-engineering uncertain areas.",
            )
            .into(),
            semantics_version: 2,
            judgment_kind: JudgmentKind::ComposableRatio,
            incoming_cluster_weight: Some(0.8),
        },
        // =============================================================
        // Epistemic → Strategic: knowing what you don't know enables
        // strategic judgment
        // =============================================================

        // Calibration → Information value
        // You can only design discriminating experiments if you know
        // where your uncertainty is.
        JudgmentContext {
            id: "epistemic_to_information_value".into(),
            relation_type: RelationType::Influence,
            target_node_id: "information_value".into(),
            source_cluster_id: "epistemic".into(),
            prompt_text: concat!(
                "How much does each epistemic quality contribute to identifying ",
                "high-information-value actions? Calibration (knowing what you don't ",
                "know) is the prerequisite for designing experiments that resolve ",
                "real uncertainty. Causal depth identifies which unknowns are ",
                "load-bearing. Compositional reach reveals where learning about one ",
                "component illuminates others.",
            )
            .into(),
            semantics_version: 2,
            judgment_kind: JudgmentKind::ComposableRatio,
            incoming_cluster_weight: Some(0.8),
        },
        // =============================================================
        // Strategic → Instrumental: context constrains action (local)
        // =============================================================

        // Temporal shape + Prioritization → Leverage assessment
        // What counts as high leverage depends on context: a compounding
        // change has different leverage than a decaying-window fix.
        // This is PairwiseOnlyRatio because it's contextual and should
        // not propagate globally.
        JudgmentContext {
            id: "strategic_to_leverage".into(),
            relation_type: RelationType::Influence,
            target_node_id: "leverage".into(),
            source_cluster_id: "strategic".into(),
            prompt_text: concat!(
                "How much does each strategic quality locally influence the ",
                "assessment of leverage? Temporal shape determines whether leverage ",
                "is immediate or compounding. Prioritization determines whether high-",
                "leverage work is actually the most important work right now. These ",
                "are contextual judgments that vary by situation.",
            )
            .into(),
            semantics_version: 2,
            judgment_kind: JudgmentKind::PairwiseOnlyRatio,
            incoming_cluster_weight: Some(0.5),
        },
        // Prioritization → Economy
        // What counts as economical depends on what else you could be doing.
        JudgmentContext {
            id: "strategic_to_economy".into(),
            relation_type: RelationType::Influence,
            target_node_id: "economy".into(),
            source_cluster_id: "strategic".into(),
            prompt_text: concat!(
                "How much does each strategic quality locally influence economy ",
                "assessment? Prioritization determines opportunity cost — the economy ",
                "of a change depends on what else that time could be spent on. ",
                "Temporal shape determines whether the cost should be paid now or ",
                "deferred. These are contextual, situation-dependent influences.",
            )
            .into(),
            semantics_version: 2,
            judgment_kind: JudgmentKind::PairwiseOnlyRatio,
            incoming_cluster_weight: Some(0.4),
        },
    ];

    AnpNetwork {
        clusters,
        nodes,
        contexts,
    }
}

// =============================================================================
// Weight profiles for different evaluation contexts
// =============================================================================

/// Weight overrides for code review / PR evaluation.
///
/// Epistemic axes weighted heavily — in code review, understanding what's
/// actually happening matters most. Strategic axes matter less because the
/// strategic decision to do this work was already made.
pub fn code_review_weights() -> Vec<(&'static str, f64)> {
    vec![
        // Epistemic — high
        ("groundedness", 1.0),
        ("calibration", 0.9),
        ("resolution", 0.95),
        ("causal_depth", 0.85),
        ("compositional_reach", 0.8),
        // Instrumental — medium-high
        ("leverage", 0.75),
        ("robustness", 0.85),
        ("option_value", 0.6),
        ("economy", 0.8),
        // Strategic — lower (strategic framing already decided)
        ("information_value", 0.4),
        ("temporal_shape", 0.3),
        ("prioritization", 0.65),
    ]
}

/// Weight overrides for idea / proposal evaluation.
///
/// Strategic axes weighted heavily — for ideas, the meta-question of
/// what's worth doing matters most. Epistemic axes still matter but
/// resolution is less critical (ideas are allowed to be less precise
/// than code interventions).
pub fn idea_evaluation_weights() -> Vec<(&'static str, f64)> {
    vec![
        // Epistemic — moderate
        ("groundedness", 0.7),
        ("calibration", 0.85),
        ("resolution", 0.6),
        ("causal_depth", 0.8),
        ("compositional_reach", 0.75),
        // Instrumental — high
        ("leverage", 1.0),
        ("robustness", 0.7),
        ("option_value", 0.85),
        ("economy", 0.75),
        // Strategic — high
        ("information_value", 0.9),
        ("temporal_shape", 0.8),
        ("prioritization", 0.95),
    ]
}

/// Weight overrides for architecture review / design doc evaluation.
///
/// Balanced across all clusters — architecture requires deep understanding,
/// sound proposals, AND strategic judgment.
pub fn architecture_review_weights() -> Vec<(&'static str, f64)> {
    vec![
        // Epistemic — high
        ("groundedness", 0.85),
        ("calibration", 0.9),
        ("resolution", 0.8),
        ("causal_depth", 0.95),
        ("compositional_reach", 1.0),
        // Instrumental — high
        ("leverage", 0.85),
        ("robustness", 0.9),
        ("option_value", 0.95),
        ("economy", 0.7),
        // Strategic — high
        ("information_value", 0.8),
        ("temporal_shape", 0.75),
        ("prioritization", 0.85),
    ]
}

/// Apply weight overrides to the orthogonal axes.
pub fn weighted_axes(weights: &[(&str, f64)]) -> Vec<PipelineAttribute> {
    let mut axes = orthogonal_axes();
    for (id, weight) in weights {
        if let Some(axis) = axes.iter_mut().find(|a| a.id == *id) {
            axis.weight = *weight;
        }
    }
    axes
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn axes_have_unique_ids() {
        let axes = orthogonal_axes();
        let ids: HashSet<&str> = axes.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids.len(), axes.len(), "duplicate axis IDs detected");
    }

    #[test]
    fn axes_count_is_twelve() {
        assert_eq!(orthogonal_axes().len(), 12);
        assert_eq!(epistemic_axes().len(), 5);
        assert_eq!(instrumental_axes().len(), 4);
        assert_eq!(strategic_axes().len(), 3);
    }

    #[test]
    fn anp_network_has_correct_structure() {
        let network = build_orthogonal_anp_network();
        assert_eq!(network.clusters.len(), 3);
        assert_eq!(network.nodes.len(), 12);

        // All nodes reference valid clusters
        let cluster_ids: HashSet<&str> = network.clusters.iter().map(|c| c.id.as_str()).collect();
        for node in &network.nodes {
            assert!(
                cluster_ids.contains(node.cluster_id.as_str()),
                "node {} references unknown cluster {}",
                node.id,
                node.cluster_id
            );
        }

        // All contexts reference valid nodes and clusters
        let node_ids: HashSet<&str> = network.nodes.iter().map(|n| n.id.as_str()).collect();
        for ctx in &network.contexts {
            assert!(
                node_ids.contains(ctx.target_node_id.as_str()),
                "context {} targets unknown node {}",
                ctx.id,
                ctx.target_node_id
            );
            assert!(
                cluster_ids.contains(ctx.source_cluster_id.as_str()),
                "context {} sources from unknown cluster {}",
                ctx.id,
                ctx.source_cluster_id
            );
        }
    }

    #[test]
    fn weight_profiles_cover_all_axes() {
        let axes = orthogonal_axes();
        let all_ids: HashSet<&str> = axes.iter().map(|a| a.id.as_str()).collect();

        for (name, weights) in [
            ("code_review", code_review_weights()),
            ("idea_evaluation", idea_evaluation_weights()),
            ("architecture_review", architecture_review_weights()),
        ] {
            let weight_ids: HashSet<&str> = weights.iter().map(|(id, _)| *id).collect();
            assert_eq!(all_ids, weight_ids, "{} weights don't cover all axes", name);
        }
    }

    #[test]
    fn all_weights_in_valid_range() {
        for weights in [
            code_review_weights(),
            idea_evaluation_weights(),
            architecture_review_weights(),
        ] {
            for (id, w) in &weights {
                assert!(
                    *w > 0.0 && *w <= 1.0,
                    "weight for {} is {} (expected 0 < w <= 1)",
                    id,
                    w
                );
            }
        }
    }

    #[test]
    fn anp_influence_directions_are_correct() {
        let network = build_orthogonal_anp_network();

        // Epistemic → Instrumental edges should be ComposableRatio
        let epistemic_to_instrumental: Vec<_> = network
            .contexts
            .iter()
            .filter(|c| c.source_cluster_id == "epistemic")
            .filter(|c| {
                network
                    .nodes
                    .iter()
                    .any(|n| n.id == c.target_node_id && n.cluster_id == "instrumental")
            })
            .collect();
        assert!(
            !epistemic_to_instrumental.is_empty(),
            "no epistemic→instrumental edges"
        );
        for ctx in &epistemic_to_instrumental {
            assert_eq!(
                ctx.judgment_kind,
                JudgmentKind::ComposableRatio,
                "epistemic→instrumental edge {} should be ComposableRatio",
                ctx.id
            );
        }

        // Strategic → Instrumental edges should be PairwiseOnlyRatio
        let strategic_to_instrumental: Vec<_> = network
            .contexts
            .iter()
            .filter(|c| c.source_cluster_id == "strategic")
            .filter(|c| {
                network
                    .nodes
                    .iter()
                    .any(|n| n.id == c.target_node_id && n.cluster_id == "instrumental")
            })
            .collect();
        assert!(
            !strategic_to_instrumental.is_empty(),
            "no strategic→instrumental edges"
        );
        for ctx in &strategic_to_instrumental {
            assert_eq!(
                ctx.judgment_kind,
                JudgmentKind::PairwiseOnlyRatio,
                "strategic→instrumental edge {} should be PairwiseOnlyRatio",
                ctx.id
            );
        }
    }
}
