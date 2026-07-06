//! The Judge Coherence Benchmark (JCB): score a model's *judgement* quality
//! with no ground-truth labels, purely from internal consistency under
//! meaning-preserving transformations — plus a signal axis so that a
//! constant judge cannot hide in perfect consistency.
//!
//! The claim being tested: a judgement deserves the name *belief* only if it
//! is (anti)symmetric under the transformations that shouldn't matter. The
//! benchmark measures, per model, on a fixed public corpus:
//!
//! | Dimension | Transformation | Perfect judge |
//! |---|---|---|
//! | signal | — | separates items (mean \|log-ratio\| > 0) |
//! | order invariance | swap presentation slots | same direction both orders |
//! | reciprocal residual | swap presentation slots | m(A,B) = −m(B,A) exactly |
//! | frustration | compose around cycles | zero Hodge curl |
//! | spin robustness | leaning-asker preamble | direction unmoved |
//! | polarity reversal | negate the attribute | scores anti-correlate (ρ → −1) |
//! | paraphrase stability | reword the attribute | scores correlate (ρ → +1) |
//! | null calibration | identical items | ratio 1.0 (zero directional mass) |
//!
//! Why this shape is hard to game: the dimensions cross-check. A judge that
//! answers by content-blind hash aces order invariance but cannot know that
//! the negated attribute must reverse (polarity) or that a reworded one must
//! not (paraphrase). A judge that always ties aces every consistency axis
//! and scores zero signal. A sycophant keeps its correlations and loses
//! spin. The composite multiplies signal by mean consistency, so zeroing
//! any side zeroes the headline.
//!
//! Cost: `CALLS_PER_RUN` comparisons per model (~$0.05 on mini-class
//! models). Deterministic corpus and pair design; temperature per template
//! default; every raw judgement is returned for receipt storage.

use std::collections::HashMap;

use futures::stream::{self, StreamExt};
use serde::Serialize;

use super::comparison::{
    compare_pair, ComparisonError, PairwiseComparisonAttribute, PairwiseComparisonEntity,
    PairwiseComparisonRequest, PairwiseComparisonSpec,
};
use super::sort::spearman;
use super::spin::{spin_probe, SpinProbeReport};
use super::types::PairwiseJudgement;
use crate::cache::PairwiseCache;
use crate::gateway::{Attribution, ChatGateway};
use crate::rating_engine::{AttributeParams, Config, Observation, RaterParams, RatingEngine};

/// The public corpus: eight short texts spanning depth on the primary
/// attribute. Fixed — the benchmark is a standardized instrument, and the
/// consistency dimensions are unfakeable by memorizing the corpus (they
/// constrain *relations between answers*, not answers).
pub const CORPUS: [&str; 8] = [
    "The obstacle is the way.",
    "We suffer more often in imagination than in reality.",
    "No man ever steps in the same river twice.",
    "A journey of a thousand miles begins with a single step.",
    "What gets measured gets managed.",
    "Early to bed and early to rise makes a man healthy, wealthy and wise.",
    "Live, laugh, love.",
    "Monday is the first day of the work week.",
];

/// Primary attribute: what the corpus is judged by.
pub const PRIMARY_ATTRIBUTE: &str = "depth of insight about living well";
/// The negation: a coherent judge's scores under it must anti-correlate.
pub const OPPOSITE_ATTRIBUTE: &str =
    "shallowness: the absence of any real insight about living well";
/// A rewording: a coherent judge's scores under it must correlate.
pub const PARAPHRASE_ATTRIBUTE: &str = "how much genuine wisdom about how to live it carries";

/// Pair design over the 8 corpus items: strides 1, 2, and 4 around the ring
/// — 20 pairs, connected and cycle-rich (triangles everywhere), so the curl
/// estimate has support.
#[must_use]
pub fn core_pairs() -> Vec<(usize, usize)> {
    let n = CORPUS.len();
    let mut pairs = Vec::new();
    for stride in [1usize, 2, 4] {
        for i in 0..n {
            let j = (i + stride) % n;
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            if !pairs.contains(&(a, b)) {
                pairs.push((a, b));
            }
        }
    }
    pairs
}

/// Spin pairs: three with a clear expected direction gap (survival is
/// scoreable) and two genuinely contested (χ is the measurement).
pub const SPIN_CLEAR_PAIRS: [(usize, usize); 3] = [(0, 7), (1, 6), (2, 7)];
pub const SPIN_CONTESTED_PAIRS: [(usize, usize); 2] = [(0, 1), (3, 4)];

/// Null texts: judged against themselves.
pub const NULL_INDICES: [usize; 4] = [0, 3, 5, 7];

/// Nuisance perturbations: semantically-null text edits a genuine judge
/// must see through. Three format edits apply to BOTH entities; the halo
/// suffix applies to entity j only (an asymmetric prestige cue). This is
/// the axis that kills content-blind hash judges: any bytes-keyed shortcut
/// changes its answer under a null edit (hash avalanche), while a reader
/// does not.
pub const PERTURBATIONS: [&str; 4] = ["whitespace", "markdown", "bullet", "halo"];

/// Core pairs that get the perturbation battery (every 3rd pair: 6 of 20).
#[must_use]
pub fn perturb_pairs() -> Vec<(usize, usize)> {
    core_pairs().into_iter().step_by(3).take(6).collect()
}

fn perturb_text(kind: &str, text: &str, is_target: bool) -> String {
    match kind {
        "whitespace" => format!("  {text}   "),
        "markdown" => format!("**{text}**"),
        "bullet" => format!("- {text}"),
        // Halo applies only to the target entity (j).
        "halo" if is_target => format!("{text} \u{2014} from a widely cited essay"),
        _ => text.to_string(),
    }
}

/// The harmonic block: four texts judged ONLY around a chordless 4-cycle
/// (both orders), disjoint from the main corpus graph. The stride graph's
/// triangles span its whole cycle space (harmonic_dim = 0, pinned in
/// tests/hodge_split.rs), so triad-invisible frustration is unmeasurable
/// there BY CONSTRUCTION; this block has cycle_dim = 1, zero triangles,
/// harmonic_dim = 1 — any non-closure of the loop is pure harmonic
/// energy, the kind no triad audit can ever see.
pub const HARMONIC_BLOCK: [&str; 4] = [
    "Fortune favors the bold.",
    "Look before you leap.",
    "He who hesitates is lost.",
    "Slow and steady wins the race.",
];

/// The chordless cycle over the harmonic block (block-local indices).
pub const HARMONIC_CYCLE: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (0, 3)];

/// Core pairs that get the full Z₂³ orbit transform (6 of 20).
#[must_use]
pub fn orbit_pairs() -> Vec<(usize, usize)> {
    core_pairs().into_iter().skip(1).step_by(3).take(6).collect()
}

/// Total provider calls in one benchmark run.
pub const CALLS_PER_RUN: usize = 20 * 2 + 20 + 20 + 4 + 6 * 4 + 5 * 6 + 6 * 8 + 4 * 2;

/// Options for [`run_judge_bench`].
#[derive(Debug, Clone)]
pub struct JudgeBenchOptions {
    /// Model slug (OpenRouter).
    pub model: String,
    /// Prompt template slug (default `canonical_v2`).
    pub template: String,
    /// Concurrent comparisons.
    pub concurrency: usize,
}

impl Default for JudgeBenchOptions {
    fn default() -> Self {
        Self {
            model: String::new(),
            template: "canonical_v2".to_string(),
            concurrency: 6,
        }
    }
}

/// One raw pairwise call receipt.
#[derive(Debug, Clone, Serialize)]
pub struct BenchCall {
    /// Which battery block this call belongs to.
    pub block: String,
    /// Canonical entity indices (i, j).
    pub i: usize,
    pub j: usize,
    /// Entity i presented in slot A.
    pub i_in_slot_a: bool,
    /// Signed log-ratio toward entity i; `None` = refused.
    pub log_ratio_toward_i: Option<f64>,
    /// Stated confidence when present.
    pub confidence: Option<f64>,
}

/// One dimension's stats.
#[derive(Debug, Clone, Serialize)]
pub struct DimensionStat {
    /// Raw measurement (units in `unit`).
    pub value: Option<f64>,
    /// 95% interval when computable (Wilson for rates, ±2se for means).
    pub ci95: Option<(f64, f64)>,
    /// Denominator behind the value — no rate without its base.
    pub n: usize,
    /// Unit of `value`.
    pub unit: &'static str,
    /// Normalized subscore in [0,1] (formula documented per dimension).
    pub subscore: Option<f64>,
}

/// Full benchmark report for one model.
#[derive(Debug, Serialize)]
pub struct JudgeBenchReport {
    pub model: String,
    pub template: String,

    /// Mean |fused log-ratio| across core pairs. Subscore 1 − e^(−value).
    pub signal: DimensionStat,
    /// Fraction of decisive core pairs whose direction flips under order
    /// swap. Subscore 1 − rate; Wilson CI.
    pub order_flip: DimensionStat,
    /// Mean |m_fwd − m_rev| / 2 (reciprocal antisymmetry residual, nats).
    /// Subscore e^(−value).
    pub order_residual: DimensionStat,
    /// Hodge curl fraction of the fused core graph. Subscore 1 − value.
    pub frustration: DimensionStat,
    /// Fraction of clear spin pairs whose belief survives both framings.
    /// Subscore = rate.
    pub spin_survival: DimensionStat,
    /// Mean |χ| across all spin pairs (nats per unit spin). Reported, not
    /// scored (contested pairs legitimately move).
    pub susceptibility: DimensionStat,
    /// Spearman ρ between primary and opposite-attribute scores.
    /// Subscore (1 − ρ)/2.
    pub polarity: DimensionStat,
    /// Spearman ρ between primary and paraphrase-attribute scores.
    /// Subscore (1 + ρ)/2.
    pub paraphrase: DimensionStat,
    /// Mean |log-ratio| on identical-item pairs (nats). Subscore e^(−value).
    pub null_bias: DimensionStat,
    /// Mean |drift| of the judgement under semantically-null text edits
    /// (whitespace, markdown, bullet, prestige-halo), vs the same pair's
    /// unperturbed same-order call. Nats. Subscore e^(−value). The axis
    /// that kills content-blind hash shortcuts.
    pub nuisance: DimensionStat,
    /// Per-perturbation mean |drift| breakdown (kind, nats, n).
    pub nuisance_breakdown: Vec<(String, f64, usize)>,
    /// Mean orbit coherence over the orbit pairs: the fraction of each
    /// judgment's energy in the G-invariant (belief) component of the
    /// Z₂³ character decomposition. Subscore = value — the spectral
    /// quantity the marginal axes approximate.
    pub orbit_coherence: DimensionStat,
    /// Mean share of judgment energy in the interaction characters
    /// (|S| ≥ 2) — bias structure invisible to every marginal probe.
    /// Reported, unscored (already inside 1 − coherence).
    pub interaction_share: DimensionStat,
    /// Harmonic fraction of the chordless-cycle block: triad-invisible
    /// frustration, measurable because the block's harmonic_dim = 1 by
    /// design. Subscore 1 − value.
    pub harmonic: DimensionStat,

    /// Mean of available consistency subscores (reciprocity = merged order
    /// flip + residual, frustration, spin survival, polarity, paraphrase,
    /// null).
    pub coherence: Option<f64>,
    /// Harmonic mean of the same subscores: one dead axis tanks it. The
    /// game-resistant aggregate, reported alongside the arithmetic headline.
    pub coherence_harmonic: Option<f64>,
    /// Headline: signal subscore × coherence. Zero signal or zero
    /// coherence zeroes it.
    pub judge_score: Option<f64>,

    /// Primary-attribute latent scores for the corpus (fused solve).
    pub primary_scores: Vec<f64>,
    /// Refusal count across all calls.
    pub refusals: usize,
    /// Calls attempted / answered from cache.
    pub comparisons: usize,
    pub comparisons_cached: usize,
    pub cost_nanodollars: i64,

    /// Raw per-call receipts (order-swap and attribute blocks).
    pub calls: Vec<BenchCall>,
    /// Raw spin reports per spin pair, keyed "i-j".
    pub spin_reports: Vec<(String, SpinProbeReport)>,
}

fn wilson_ci95(successes: usize, n: usize) -> Option<(f64, f64)> {
    if n == 0 {
        return None;
    }
    let z = 1.96f64;
    let n_f = n as f64;
    let p = successes as f64 / n_f;
    let denom = 1.0 + z * z / n_f;
    let center = (p + z * z / (2.0 * n_f)) / denom;
    let half = (z / denom) * (p * (1.0 - p) / n_f + z * z / (4.0 * n_f * n_f)).sqrt();
    Some(((center - half).max(0.0), (center + half).min(1.0)))
}

fn mean_ci95(samples: &[f64]) -> (Option<f64>, Option<(f64, f64)>) {
    if samples.is_empty() {
        return (None, None);
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    if samples.len() < 3 {
        return (Some(mean), None);
    }
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (var / n).sqrt();
    (Some(mean), Some((mean - 1.96 * se, mean + 1.96 * se)))
}

struct CallOutcome {
    log_ratio_toward_i: Option<f64>,
    confidence: Option<f64>,
    cached: bool,
    cost_nanodollars: i64,
}

#[expect(clippy::too_many_arguments)]
async fn one_call(
    gateway: &dyn ChatGateway,
    cache: Option<&dyn PairwiseCache>,
    model: &str,
    template: &str,
    attribute_prompt: &str,
    i: usize,
    j: usize,
    texts: (&str, &str),
    i_in_slot_a: bool,
    attribution: &Attribution,
) -> Result<CallOutcome, ComparisonError> {
    let (slot_a, slot_b, text_a, text_b) = if i_in_slot_a {
        (i, j, texts.0, texts.1)
    } else {
        (j, i, texts.1, texts.0)
    };
    let ids = ["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7"];
    let spec = PairwiseComparisonSpec {
        model,
        attribute: PairwiseComparisonAttribute {
            id: "bench",
            prompt: attribute_prompt,
            prompt_template_slug: Some(template),
        },
        entity_a: PairwiseComparisonEntity {
            id: ids[slot_a],
            text: text_a,
        },
        entity_b: PairwiseComparisonEntity {
            id: ids[slot_b],
            text: text_b,
        },
    };
    let (judgement, usage) = compare_pair(
        gateway,
        cache,
        PairwiseComparisonRequest {
            spec,
            cache_only: false,
            attribution: attribution.clone(),
        },
    )
    .await?;
    let confidence = match &judgement {
        PairwiseJudgement::Observation { confidence, .. } => Some(*confidence),
        PairwiseJudgement::Refused => None,
    };
    let log_ratio_toward_i =
        super::types::signed_log_ratio_toward_first(&judgement, i_in_slot_a);
    Ok(CallOutcome {
        log_ratio_toward_i,
        confidence,
        cached: usage.cached,
        cost_nanodollars: usage.provider_cost_nanodollars,
    })
}

/// Solve the harmonic block and return its harmonic energy fraction.
fn harmonic_split(observations: &[(usize, usize, f64)]) -> Option<f64> {
    let mut raters = HashMap::new();
    raters.insert("bench".to_string(), RaterParams::default());
    let mut engine = RatingEngine::new(
        HARMONIC_BLOCK.len(),
        AttributeParams::default(),
        raters,
        Some(Config::default()),
    )
    .ok()?;
    let obs: Vec<Observation> = observations
        .iter()
        .map(|&(i, j, m)| Observation::from_log_ratio_moments(i, j, m, 1.0, "bench", 1.0))
        .collect();
    engine.ingest(&obs);
    let summary = engine.solve();
    Some(summary.hodge.harmonic_frac)
}

fn solve_scores(
    n: usize,
    observations: &[(usize, usize, f64)],
    rater: &str,
) -> Option<(Vec<f64>, f64, usize)> {
    if observations.is_empty() {
        return None;
    }
    let mut raters = HashMap::new();
    raters.insert(rater.to_string(), RaterParams::default());
    let mut engine =
        RatingEngine::new(n, AttributeParams::default(), raters, Some(Config::default())).ok()?;
    let obs: Vec<Observation> = observations
        .iter()
        .map(|&(i, j, m)| Observation::from_log_ratio_moments(i, j, m, 1.0, rater, 1.0))
        .collect();
    engine.ingest(&obs);
    let summary = engine.solve();
    Some((summary.scores, summary.hcr, summary.cycle_dim))
}

/// Run the Judge Coherence Benchmark for one model.
pub async fn run_judge_bench(
    gateway: &dyn ChatGateway,
    cache: Option<&dyn PairwiseCache>,
    opts: JudgeBenchOptions,
) -> Result<JudgeBenchReport, ComparisonError> {
    let attribution = Attribution::new("cardinal::bench");
    let model = opts.model.clone();
    let template = opts.template.clone();
    let pairs = core_pairs();

    // ---- Build the call plan ----
    type PlanEntry = (
        &'static str,
        &'static str,
        usize,
        usize,
        bool,
        Option<&'static str>,
    );
    let mut plan: Vec<PlanEntry> = Vec::new();
    for &(i, j) in &pairs {
        plan.push(("core", PRIMARY_ATTRIBUTE, i, j, true, None));
        plan.push(("core", PRIMARY_ATTRIBUTE, i, j, false, None));
    }
    for &(i, j) in &pairs {
        plan.push(("opposite", OPPOSITE_ATTRIBUTE, i, j, true, None));
        plan.push(("paraphrase", PARAPHRASE_ATTRIBUTE, i, j, true, None));
    }
    for &i in &NULL_INDICES {
        plan.push(("null", PRIMARY_ATTRIBUTE, i, i, true, None));
    }
    for &(i, j) in &perturb_pairs() {
        for kind in PERTURBATIONS {
            plan.push(("nuisance", PRIMARY_ATTRIBUTE, i, j, true, Some(kind)));
        }
    }

    let concurrency = opts.concurrency.max(1);
    let results: Vec<(usize, Result<CallOutcome, ComparisonError>)> = stream::iter(
        plan.iter()
            .enumerate()
            .map(|(idx, &(_, attr, i, j, fwd, perturb))| {
                let attribution = &attribution;
                let model = model.as_str();
                let template = template.as_str();
                async move {
                    let (text_i, text_j) = match perturb {
                        Some(kind) => (
                            perturb_text(kind, CORPUS[i], false),
                            perturb_text(kind, CORPUS[j], true),
                        ),
                        None => (CORPUS[i].to_string(), CORPUS[j].to_string()),
                    };
                    let out = one_call(
                        gateway,
                        cache,
                        model,
                        template,
                        attr,
                        i,
                        j,
                        (&text_i, &text_j),
                        fwd,
                        attribution,
                    )
                    .await;
                    (idx, out)
                }
            }),
    )
    .buffer_unordered(concurrency)
    .collect()
    .await;

    let mut outcomes: Vec<Option<CallOutcome>> = (0..plan.len()).map(|_| None).collect();
    let mut cost = 0i64;
    let mut cached = 0usize;
    let mut refusals = 0usize;
    for (idx, out) in results {
        let out = out?;
        cost += out.cost_nanodollars;
        if out.cached {
            cached += 1;
        }
        if out.log_ratio_toward_i.is_none() {
            refusals += 1;
        }
        outcomes[idx] = Some(out);
    }

    let mut calls = Vec::with_capacity(plan.len());
    for (idx, &(block, _, i, j, fwd, perturb)) in plan.iter().enumerate() {
        let out = outcomes[idx].as_ref().expect("all outcomes filled");
        calls.push(BenchCall {
            block: match perturb {
                Some(kind) => format!("nuisance:{kind}"),
                None => block.to_string(),
            },
            i,
            j,
            i_in_slot_a: fwd,
            log_ratio_toward_i: out.log_ratio_toward_i,
            confidence: out.confidence,
        });
    }

    // ---- Core block: signal, order flip, residual, fused graph ----
    let core: Vec<&BenchCall> = calls.iter().filter(|c| c.block == "core").collect();
    let mut fused_ms = Vec::new();
    let mut residuals = Vec::new();
    let mut decisive_pairs = 0usize;
    let mut flips = 0usize;
    let mut fused_obs: Vec<(usize, usize, f64)> = Vec::new();
    for &(i, j) in &pairs {
        let fwd = core
            .iter()
            .find(|c| c.i == i && c.j == j && c.i_in_slot_a)
            .and_then(|c| c.log_ratio_toward_i);
        let rev = core
            .iter()
            .find(|c| c.i == i && c.j == j && !c.i_in_slot_a)
            .and_then(|c| c.log_ratio_toward_i);
        if let (Some(f), Some(r)) = (fwd, rev) {
            let fused = (f + r) / 2.0;
            fused_ms.push(fused.abs());
            residuals.push((f - r).abs() / 2.0);
            fused_obs.push((i, j, fused));
            if f != 0.0 && r != 0.0 {
                decisive_pairs += 1;
                if f.signum() != r.signum() {
                    flips += 1;
                }
            }
        }
    }
    let (signal_mean, signal_ci) = mean_ci95(&fused_ms);
    let (residual_mean, residual_ci) = mean_ci95(&residuals);
    let flip_rate = (decisive_pairs > 0).then(|| flips as f64 / decisive_pairs as f64);

    let primary = solve_scores(CORPUS.len(), &fused_obs, &model);
    let (primary_scores, hcr, cycle_dim) = match &primary {
        Some((scores, hcr, cycle_dim)) => (scores.clone(), Some(*hcr), *cycle_dim),
        None => (Vec::new(), None, 0),
    };

    // ---- Attribute blocks: polarity + paraphrase correlations ----
    let block_scores = |name: &str| -> Option<Vec<f64>> {
        let obs: Vec<(usize, usize, f64)> = calls
            .iter()
            .filter(|c| c.block == name)
            .filter_map(|c| c.log_ratio_toward_i.map(|m| (c.i, c.j, m)))
            .collect();
        solve_scores(CORPUS.len(), &obs, &model).map(|(s, _, _)| s)
    };
    let polarity_rho = block_scores("opposite")
        .filter(|_| !primary_scores.is_empty())
        .and_then(|s| spearman(&primary_scores, &s));
    let paraphrase_rho = block_scores("paraphrase")
        .filter(|_| !primary_scores.is_empty())
        .and_then(|s| spearman(&primary_scores, &s));

    // ---- Null block ----
    let null_ms: Vec<f64> = calls
        .iter()
        .filter(|c| c.block == "null")
        .filter_map(|c| c.log_ratio_toward_i.map(f64::abs))
        .collect();
    let (null_mean, null_ci) = mean_ci95(&null_ms);

    // ---- Nuisance block: drift vs the same pair's unperturbed call ----
    let baseline_m = |i: usize, j: usize| -> Option<f64> {
        calls
            .iter()
            .find(|c| c.block == "core" && c.i == i && c.j == j && c.i_in_slot_a)
            .and_then(|c| c.log_ratio_toward_i)
    };
    let mut nuisance_drifts: Vec<f64> = Vec::new();
    let mut breakdown: Vec<(String, f64, usize)> = Vec::new();
    for kind in PERTURBATIONS {
        let block = format!("nuisance:{kind}");
        let drifts: Vec<f64> = calls
            .iter()
            .filter(|c| c.block == block)
            .filter_map(|c| {
                let base = baseline_m(c.i, c.j)?;
                let m = c.log_ratio_toward_i?;
                Some((m - base).abs())
            })
            .collect();
        if !drifts.is_empty() {
            let mean = drifts.iter().sum::<f64>() / drifts.len() as f64;
            breakdown.push((kind.to_string(), mean, drifts.len()));
        }
        nuisance_drifts.extend(drifts);
    }
    let (nuisance_mean, nuisance_ci) = mean_ci95(&nuisance_drifts);

    // ---- Orbit block: full Z₂³ transform on selected core pairs ----
    let mut orbit_coherences: Vec<f64> = Vec::new();
    let mut interaction_shares: Vec<f64> = Vec::new();
    let mut extra_comparisons = 0usize;
    for &(i, j) in &orbit_pairs() {
        let report = super::orbit::orbit_transform(
            gateway,
            cache,
            &model,
            PRIMARY_ATTRIBUTE,
            ("o0", CORPUS[i]),
            ("o1", CORPUS[j]),
            &template,
            attribution.clone(),
        )
        .await?;
        cost += report.cost_nanodollars;
        cached += report.comparisons_cached;
        refusals += report.refusals;
        extra_comparisons += report.comparisons;
        if let Some(c) = report.coherence {
            orbit_coherences.push(c);
            let total: f64 = report.energies.iter().sum();
            if total > 0.0 {
                let interactions: f64 = [3usize, 5, 6, 7]
                    .iter()
                    .map(|&k| report.energies[k])
                    .sum();
                interaction_shares.push(interactions / total);
            }
        }
    }
    let (orbit_mean, orbit_ci) = mean_ci95(&orbit_coherences);
    let (interaction_mean, interaction_ci) = mean_ci95(&interaction_shares);

    // ---- Harmonic block: chordless 4-cycle, both orders, own solve ----
    let mut harmonic_outcomes: Vec<(usize, usize, [Option<f64>; 2])> = HARMONIC_CYCLE
        .iter()
        .map(|&(i, j)| (i, j, [None, None]))
        .collect();
    for (i, j, samples) in harmonic_outcomes.iter_mut() {
        for (slot, fwd) in [(0usize, true), (1usize, false)] {
            let out = one_call(
                gateway,
                cache,
                &model,
                &template,
                PRIMARY_ATTRIBUTE,
                *i,
                *j,
                (HARMONIC_BLOCK[*i], HARMONIC_BLOCK[*j]),
                fwd,
                &attribution,
            )
            .await?;
            cost += out.cost_nanodollars;
            extra_comparisons += 1;
            if out.cached {
                cached += 1;
            }
            match out.log_ratio_toward_i {
                Some(m) => samples[slot] = Some(m),
                None => refusals += 1,
            }
        }
    }
    let harmonic_obs: Vec<(usize, usize, f64)> = harmonic_outcomes
        .iter()
        .filter_map(|&(i, j, samples)| match samples {
            [Some(a), Some(b)] => Some((i, j, (a + b) / 2.0)),
            _ => None,
        })
        .collect();
    let harmonic_value = if harmonic_obs.len() == HARMONIC_CYCLE.len() {
        harmonic_split(&harmonic_obs)
    } else {
        None
    };

    // ---- Spin block ----
    let mut spin_reports = Vec::new();
    let mut clear_survivals = 0usize;
    let mut clear_assessed = 0usize;
    let mut chis = Vec::new();
    for (clear, &(i, j)) in SPIN_CLEAR_PAIRS
        .iter()
        .map(|p| (true, p))
        .chain(SPIN_CONTESTED_PAIRS.iter().map(|p| (false, p)))
    {
        let report = spin_probe(
            gateway,
            cache,
            &model,
            &template,
            PRIMARY_ATTRIBUTE,
            ("s0", CORPUS[i]),
            ("s1", CORPUS[j]),
            attribution.clone(),
        )
        .await?;
        cost += report.cost_nanodollars;
        cached += report.comparisons_cached;
        for reading in &report.readings {
            refusals += reading.refusals;
        }
        if let Some(chi) = report.susceptibility_nats {
            chis.push(chi.abs());
        }
        if clear {
            if let Some(survives) = report.belief_survives_spin {
                clear_assessed += 1;
                if survives {
                    clear_survivals += 1;
                }
            }
        }
        spin_reports.push((format!("{i}-{j}"), report));
    }
    let survival_rate = (clear_assessed > 0).then(|| clear_survivals as f64 / clear_assessed as f64);
    let (chi_mean, chi_ci) = mean_ci95(&chis);

    // ---- Subscores and composite ----
    let signal = DimensionStat {
        value: signal_mean,
        ci95: signal_ci,
        n: fused_ms.len(),
        unit: "nats",
        subscore: signal_mean.map(|m| 1.0 - (-m).exp()),
    };
    let order_flip = DimensionStat {
        value: flip_rate,
        ci95: wilson_ci95(flips, decisive_pairs),
        n: decisive_pairs,
        unit: "rate",
        subscore: flip_rate.map(|r| 1.0 - r),
    };
    let order_residual = DimensionStat {
        value: residual_mean,
        ci95: residual_ci,
        n: residuals.len(),
        unit: "nats",
        subscore: residual_mean.map(|r| (-r).exp()),
    };
    let hcr = hcr.filter(|h| h.is_finite());
    // Curl is a graph statistic: its support is the number of independent
    // cycles (|E| − |V| + components), NOT the edge count. A sparse graph
    // cannot estimate frustration; the denominator says so.
    let frustration = DimensionStat {
        value: hcr,
        ci95: None,
        n: cycle_dim,
        unit: "curl fraction",
        subscore: hcr.map(|h| (1.0 - h).clamp(0.0, 1.0)),
    };
    let spin_survival = DimensionStat {
        value: survival_rate,
        ci95: wilson_ci95(clear_survivals, clear_assessed),
        n: clear_assessed,
        unit: "rate",
        subscore: survival_rate,
    };
    let susceptibility = DimensionStat {
        value: chi_mean,
        ci95: chi_ci,
        n: chis.len(),
        unit: "nats/spin",
        subscore: None,
    };
    let polarity = DimensionStat {
        value: polarity_rho,
        ci95: None,
        n: CORPUS.len(),
        unit: "spearman",
        subscore: polarity_rho.map(|rho| ((1.0 - rho) / 2.0).clamp(0.0, 1.0)),
    };
    let paraphrase = DimensionStat {
        value: paraphrase_rho,
        ci95: None,
        n: CORPUS.len(),
        unit: "spearman",
        subscore: paraphrase_rho.map(|rho| ((1.0 + rho) / 2.0).clamp(0.0, 1.0)),
    };
    let null_bias = DimensionStat {
        value: null_mean,
        ci95: null_ci,
        n: null_ms.len(),
        unit: "nats",
        subscore: null_mean.map(|b| (-b).exp()),
    };
    let nuisance = DimensionStat {
        value: nuisance_mean,
        ci95: nuisance_ci,
        n: nuisance_drifts.len(),
        unit: "nats",
        subscore: nuisance_mean.map(|d| (-d).exp()),
    };
    let orbit_coherence = DimensionStat {
        value: orbit_mean,
        ci95: orbit_ci,
        n: orbit_coherences.len(),
        unit: "energy fraction",
        subscore: orbit_mean,
    };
    let interaction_share = DimensionStat {
        value: interaction_mean,
        ci95: interaction_ci,
        n: interaction_shares.len(),
        unit: "energy fraction",
        subscore: None,
    };
    let harmonic = DimensionStat {
        value: harmonic_value,
        ci95: None,
        n: usize::from(harmonic_obs.len() == HARMONIC_CYCLE.len()),
        unit: "energy fraction",
        subscore: harmonic_value.map(|h| (1.0 - h).clamp(0.0, 1.0)),
    };

    // Order-flip and order-residual are measured from the SAME two calls per
    // pair — two views of one transformation. They enter the composite as a
    // single reciprocity axis (their mean) so position-bias fixes are not
    // double-credited. Both raw stats stay reported.
    let reciprocity = match (order_flip.subscore, order_residual.subscore) {
        (Some(a), Some(b)) => Some((a + b) / 2.0),
        (a, b) => a.or(b),
    };
    // Coverage gate: refusing on hard pairs shrinks the curl numerator
    // (refused edges vanish from the graph), so a refusal-heavy run must
    // not be credited with transitivity. Original design DROPPED gated
    // axes — which the cyclic-judge test exposed as a reward (deleting a
    // bad judge's worst axis raises its mean). Gated axes now score ZERO:
    // refusal laundering strictly costs.
    let core_coverage = 1.0 - refusals as f64 / CALLS_PER_RUN.max(1) as f64;
    let gate = |subscore: Option<f64>| -> Option<f64> {
        if core_coverage >= 0.95 {
            subscore
        } else {
            Some(0.0)
        }
    };
    let frustration_gated = gate(frustration.subscore);
    let harmonic_gated = gate(harmonic.subscore);
    let consistency: Vec<f64> = [
        reciprocity,
        frustration_gated,
        harmonic_gated,
        spin_survival.subscore,
        polarity.subscore,
        paraphrase.subscore,
        null_bias.subscore,
        nuisance.subscore,
        orbit_coherence.subscore,
    ]
    .into_iter()
    .flatten()
    .collect();
    let coherence = (!consistency.is_empty())
        .then(|| consistency.iter().sum::<f64>() / consistency.len() as f64);
    // Harmonic mean of the same axes: the game-resistant aggregate (one dead
    // axis tanks it; pathologies cannot hide inside an average). Reported
    // alongside the arithmetic headline — at v1 corpus size a single unlucky
    // clear-pair zeroes an axis, so the harsher aggregate is a receipt, not
    // yet the ranking.
    let coherence_harmonic = (!consistency.is_empty()).then(|| {
        if consistency.iter().any(|&s| s <= 0.0) {
            0.0
        } else {
            consistency.len() as f64 / consistency.iter().map(|s| 1.0 / s).sum::<f64>()
        }
    });
    let judge_score = match (signal.subscore, coherence) {
        (Some(s), Some(c)) => Some(s * c),
        _ => None,
    };

    Ok(JudgeBenchReport {
        model,
        template,
        signal,
        order_flip,
        order_residual,
        frustration,
        spin_survival,
        susceptibility,
        polarity,
        paraphrase,
        null_bias,
        nuisance,
        nuisance_breakdown: breakdown,
        orbit_coherence,
        interaction_share,
        harmonic,
        coherence,
        coherence_harmonic,
        judge_score,
        primary_scores,
        refusals,
        comparisons: plan.len()
            + extra_comparisons
            + spin_reports.iter().map(|(_, r)| r.comparisons).sum::<usize>(),
        comparisons_cached: cached,
        cost_nanodollars: cost,
        calls,
        spin_reports,
    })
}

/// Render one report as a human stats block.
#[must_use]
pub fn render_report(report: &JudgeBenchReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let fmt_dim = |name: &str, d: &DimensionStat| -> String {
        let val = match d.value {
            Some(v) => format!("{v:+.3}"),
            None => "n/a".to_string(),
        };
        let ci = match d.ci95 {
            Some((lo, hi)) => format!(" [{lo:+.3}, {hi:+.3}]"),
            None => String::new(),
        };
        let sub = match d.subscore {
            Some(s) => format!("  → {s:.3}"),
            None => String::new(),
        };
        format!(
            "  {name:<16} {val}{ci} {unit} (n={n}){sub}\n",
            unit = d.unit,
            n = d.n
        )
    };
    let _ = writeln!(out, "model: {} · template: {}", report.model, report.template);
    out.push_str(&fmt_dim("signal", &report.signal));
    out.push_str(&fmt_dim("order-flip", &report.order_flip));
    out.push_str(&fmt_dim("order-residual", &report.order_residual));
    out.push_str(&fmt_dim("frustration", &report.frustration));
    out.push_str(&fmt_dim("spin-survival", &report.spin_survival));
    out.push_str(&fmt_dim("susceptibility", &report.susceptibility));
    out.push_str(&fmt_dim("polarity", &report.polarity));
    out.push_str(&fmt_dim("paraphrase", &report.paraphrase));
    out.push_str(&fmt_dim("null-bias", &report.null_bias));
    out.push_str(&fmt_dim("nuisance", &report.nuisance));
    out.push_str(&fmt_dim("orbit-coherence", &report.orbit_coherence));
    out.push_str(&fmt_dim("interaction", &report.interaction_share));
    out.push_str(&fmt_dim("harmonic", &report.harmonic));
    for (kind, mean, n) in &report.nuisance_breakdown {
        let _ = writeln!(out, "    nuisance:{kind:<12} {mean:+.3} nats (n={n})");
    }
    let _ = writeln!(
        out,
        "  coherence {:.3} (harmonic {:.3}) · JUDGE SCORE {:.3} · {} comparisons ({} cached) · {} refusals · ${:.4}",
        report.coherence.unwrap_or(f64::NAN),
        report.coherence_harmonic.unwrap_or(f64::NAN),
        report.judge_score.unwrap_or(f64::NAN),
        report.comparisons,
        report.comparisons_cached,
        report.refusals,
        report.cost_nanodollars as f64 / 1e9,
    );
    out
}
