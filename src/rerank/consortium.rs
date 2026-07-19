//! Consortium verdict: one pair, many minds, one belief with provenance.
//!
//! The robust atomic operation the rest of the stack composes from:
//! "A vs B, under this criterion, judged by a consortium of models" →
//! one fused belief with an explicit error budget, and one judgment
//! packet per judge so the evidence outlives the run.
//!
//! Composition, not new surface:
//!
//! 1. **Per judge, the orbit transform** (8 comparisons over Z₂³ =
//!    order × polarity × wording): the judge's belief is m̂(∅), the
//!    G-invariant projection, with every bias a named orthogonal
//!    coefficient. Equivariant-by-construction elicitation — the top of
//!    the probe→estimator ladder (PRINCIPLES §5).
//! 2. **Packets as the medium**: each complete orbit becomes a
//!    [`JudgmentPacket`] of 8 pulled-back observations (canonical
//!    coordinates, unit point precision), judge = the model slug. The
//!    consortium belief is computed BY FUSING THE PACKETS — the same
//!    path any two parties who exchanged packets would take — so the
//!    printed number and the portable evidence cannot drift apart.
//!    A judge whose orbit is incomplete (any refusal) contributes no
//!    packet: partial orbits carry un-averaged bias.
//! 3. **Error budget, experimentalist-style**: the cross-judge spread
//!    (population std of per-judge beliefs) is the systematic term the
//!    consortium adds; per-judge coherence and top bias name where each
//!    mind leaks; the correlation of orbit residuals across judges is
//!    the shared-error diagnostic (two judges whose biases co-move are
//!    partial clones — portfolio theory's error covariance at n = one
//!    pair). Diagnostic, not yet an estimator: with 8 orbit cells the
//!    correlation is quoted with its denominator, never used to weight.

use serde::Serialize;

use super::orbit::{orbit_transform, OrbitReport, CHARACTERS};
use crate::cache::PairwiseCache;
use crate::gateway::{Attribution, ChatGateway};
use crate::packet::{entity_text_hash, fuse, JudgmentPacket, PacketObservation};

/// One judge's contribution to the consortium.
#[derive(Debug, Serialize)]
pub struct ConsortiumJudge {
    pub model: String,
    /// m̂(∅) in nats toward the first item; `None` when the orbit was
    /// incomplete (refusals) and the judge contributed no evidence.
    pub belief: Option<f64>,
    /// Fraction of judgment energy that is invariant under the group.
    pub coherence: Option<f64>,
    /// Largest non-belief character by |coefficient|: (name, nats).
    pub top_bias: Option<(String, f64)>,
    pub refusals: usize,
    /// The full character decomposition (evidence, not summary).
    pub orbit: OrbitReport,
}

/// Result of [`consortium_verdict`].
#[derive(Debug, Serialize)]
pub struct ConsortiumReport {
    pub criterion: String,
    pub template: String,
    /// (id, blake3 of text) for the two items, as pinned in the packets.
    pub entity_a: (String, String),
    pub entity_b: (String, String),
    pub judges: Vec<ConsortiumJudge>,
    /// The fused belief in nats toward the first item, computed by
    /// fusing the per-judge packets (rater-aware robust solve). `None`
    /// when no judge completed its orbit.
    pub belief: Option<f64>,
    /// Mean over usable judges of the rms non-belief coefficient,
    /// √(Σ_{S≠∅} m̂(S)²) — the within-judge systematic. There is no
    /// statistical term at temperature 0: a point instrument has no
    /// measured sampling noise, and this report never invents one.
    pub orbit_bias_rms: Option<f64>,
    /// exp(belief): the ratio itself.
    pub ratio: Option<f64>,
    /// Plain mean of per-judge beliefs — the arithmetic cross-check on
    /// the fused value (they agree up to robust-weighting).
    pub judge_mean: Option<f64>,
    /// Population std of per-judge beliefs: the cross-judge systematic.
    pub judge_spread_nats: Option<f64>,
    /// Did every usable judge agree on the sign?
    pub direction_unanimous: Option<bool>,
    /// Correlation of orbit residuals (m(g) − belief) between usable
    /// judges over the 8 cells — the shared-bias diagnostic. Row order
    /// matches the usable judges' order in `judges`.
    pub residual_correlation: Option<Vec<Vec<f64>>>,
    /// One packet per usable judge: the portable evidence.
    pub packets: Vec<JudgmentPacket>,
    pub usable_judges: usize,
    pub comparisons: usize,
    pub comparisons_cached: usize,
    pub cost_nanodollars: i64,
}

/// Errors from [`consortium_verdict`].
#[derive(Debug, thiserror::Error)]
pub enum ConsortiumError {
    #[error("need at least 2 judge models, got {0}")]
    TooFewJudges(usize),
    #[error(transparent)]
    Comparison(#[from] super::comparison::ComparisonError),
    #[error("packet fusion failed: {0}")]
    Packet(#[from] crate::packet::PacketError),
}

/// Judge one pair with a consortium of models and fuse the verdict.
///
/// `created` stamps the packets (libraries own no clocks — the caller
/// supplies it). Entity ids should be stable labels: packets accrete
/// across runs by id + content hash, and fusion refuses when one id
/// carries two different texts.
#[expect(clippy::too_many_arguments)]
pub async fn consortium_verdict(
    gateway: &dyn ChatGateway,
    cache: Option<&dyn PairwiseCache>,
    models: &[String],
    criterion: &str,
    first: (&str, &str),
    second: (&str, &str),
    ratio_template: &str,
    created: &str,
    attribution: Attribution,
) -> Result<ConsortiumReport, ConsortiumError> {
    if models.len() < 2 {
        return Err(ConsortiumError::TooFewJudges(models.len()));
    }
    let entity_a = (first.0.to_string(), entity_text_hash(first.1));
    let entity_b = (second.0.to_string(), entity_text_hash(second.1));

    let mut judges = Vec::with_capacity(models.len());
    let mut packets = Vec::new();
    let mut comparisons = 0usize;
    let mut comparisons_cached = 0usize;
    let mut cost = 0i64;
    for model in models {
        let orbit = orbit_transform(
            gateway,
            cache,
            model,
            criterion,
            first,
            second,
            ratio_template,
            attribution.clone(),
        )
        .await?;
        comparisons += orbit.comparisons;
        comparisons_cached += orbit.comparisons_cached;
        cost += orbit.cost_nanodollars;
        let complete = orbit.refusals == 0;
        let belief = complete.then_some(orbit.coefficients[0]);
        let top_bias = complete.then(|| {
            let (idx, c) = orbit.coefficients[1..]
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.abs()
                        .partial_cmp(&b.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, c)| (i + 1, *c))
                .unwrap_or((1, 0.0));
            (CHARACTERS[idx].to_string(), c)
        });
        if complete {
            let mut packet = JudgmentPacket {
                version: 1,
                attribute: criterion.to_string(),
                template: ratio_template.to_string(),
                judge: model.clone(),
                entities: vec![entity_a.clone(), entity_b.clone()],
                observations: orbit
                    .orbit
                    .iter()
                    .map(|m| PacketObservation {
                        i: 0,
                        j: 1,
                        log_ratio: m.expect("complete orbit has all cells"),
                        precision: 1.0,
                    })
                    .collect(),
                created: created.to_string(),
            };
            packet.canonicalize()?;
            packets.push(packet);
        }
        judges.push(ConsortiumJudge {
            model: model.clone(),
            belief,
            coherence: orbit.coherence,
            top_bias,
            refusals: orbit.refusals,
            orbit,
        });
    }

    let usable: Vec<&ConsortiumJudge> = judges.iter().filter(|j| j.belief.is_some()).collect();
    let beliefs: Vec<f64> = usable.iter().filter_map(|j| j.belief).collect();
    let judge_mean = (!beliefs.is_empty())
        .then(|| beliefs.iter().sum::<f64>() / beliefs.len() as f64);
    let judge_spread_nats = (beliefs.len() >= 2).then(|| {
        let mean = judge_mean.expect("nonempty");
        (beliefs.iter().map(|b| (b - mean).powi(2)).sum::<f64>() / beliefs.len() as f64).sqrt()
    });
    let direction_unanimous = (!beliefs.is_empty())
        .then(|| beliefs.iter().all(|b| *b >= 0.0) || beliefs.iter().all(|b| *b <= 0.0));

    // Shared-bias diagnostic: correlation of orbit residuals over the 8
    // cells, per usable judge pair.
    let residual_correlation = (usable.len() >= 2).then(|| {
        let residuals: Vec<Vec<f64>> = usable
            .iter()
            .map(|j| {
                let b = j.belief.expect("usable");
                j.orbit
                    .orbit
                    .iter()
                    .map(|m| m.expect("complete orbit") - b)
                    .collect()
            })
            .collect();
        let corr = |x: &[f64], y: &[f64]| -> f64 {
            let n = x.len() as f64;
            let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
            let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
            for (a, b) in x.iter().zip(y) {
                sxy += (a - mx) * (b - my);
                sxx += (a - mx).powi(2);
                syy += (b - my).powi(2);
            }
            if sxx <= 1e-15 || syy <= 1e-15 {
                0.0
            } else {
                sxy / (sxx * syy).sqrt()
            }
        };
        residuals
            .iter()
            .map(|x| residuals.iter().map(|y| corr(x, y)).collect())
            .collect()
    });

    let orbit_bias_rms = (!usable.is_empty()).then(|| {
        usable
            .iter()
            .map(|j| j.orbit.energies[1..].iter().sum::<f64>().sqrt())
            .sum::<f64>()
            / usable.len() as f64
    });

    // The consortium belief travels through the packet medium: fuse the
    // per-judge packets exactly as an exchange partner would.
    let belief = if packets.is_empty() {
        None
    } else {
        let fused = fuse(&packets)?;
        let idx_a = fused
            .entities
            .iter()
            .position(|(id, _)| *id == entity_a.0)
            .expect("entity A survives fusion");
        let idx_b = fused
            .entities
            .iter()
            .position(|(id, _)| *id == entity_b.0)
            .expect("entity B survives fusion");
        Some(fused.scores[idx_a] - fused.scores[idx_b])
    };

    Ok(ConsortiumReport {
        criterion: criterion.to_string(),
        template: ratio_template.to_string(),
        entity_a,
        entity_b,
        usable_judges: usable.len(),
        judges,
        belief,
        orbit_bias_rms,
        ratio: belief.map(f64::exp),
        judge_mean,
        judge_spread_nats,
        direction_unanimous,
        residual_correlation,
        packets,
        comparisons,
        comparisons_cached,
        cost_nanodollars: cost,
    })
}
