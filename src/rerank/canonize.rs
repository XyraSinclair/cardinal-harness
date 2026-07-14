//! Canonize: converge on attributes that work as communication primitives.
//!
//! An attribute prompt is CANONICAL exactly when it induces the same
//! cardinal latent in different minds — when two judges, given only the
//! wording and the entities, recover each other's ordering. That is a
//! measurable property, not a matter of taste:
//!
//! - **transmissibility**: mean pairwise Spearman between the latent
//!   vectors different judge models produce for the same wording — the
//!   inter-mind agreement that makes the attribute a viable medium;
//! - **signal**: mean latent spread per judge — a wording every judge
//!   agrees is a tie transmits nothing;
//! - **independence**: max |ρ| against already-accepted attributes — a
//!   canonical SET spans independent dimensions, so near-duplicates of
//!   what is already accepted score low even when transmissible.
//!
//! The protocol: seed wording → propose refinements (hypotheses) →
//! measure every candidate across every judge → rank by measured
//! canonicality. This is steps 4–5 of the canonical-attribute loop
//! (FIRST_PRINCIPLES §8) as an operation instead of a paragraph.

use std::sync::Arc;

use serde::Serialize;

use super::multi::RerankExecution;
use super::options::RerankRunOptions;
use super::sort::{sort_documents, spearman, SortError, SortOptions};
use super::types::RerankDocument;
use crate::cache::PairwiseCache;
use crate::gateway::{Attribution, ChatGateway};

/// Options for [`canonize`].
#[derive(Debug, Clone)]
pub struct CanonizeOptions {
    /// Judge models (≥ 2 for transmissibility to be defined).
    pub judges: Vec<String>,
    /// Comparison budget per (candidate, judge) sort.
    pub comparison_budget: Option<usize>,
    /// RNG seed.
    pub seed: u64,
}

/// One candidate wording's measured canonicality.
#[derive(Debug, Clone, Serialize)]
pub struct CandidateCanonicality {
    pub prompt: String,
    /// Mean pairwise Spearman between judges' latent vectors. The
    /// inter-mind agreement — `None` with fewer than 2 judges' worth of
    /// usable scores.
    pub transmissibility: Option<f64>,
    /// Mean (over judges) standard deviation of latents: how much the
    /// wording separates the entities at all.
    pub signal_nats: f64,
    /// Max |Spearman| against the accepted attributes' latents (first
    /// judge), when accepted attributes were given: high = redundant
    /// dimension. `None` when no accepted set.
    pub redundancy: Option<f64>,
    /// Per-judge latent vectors, entity input order (supporting evidence).
    pub latents_per_judge: Vec<Vec<f64>>,
}

/// Result of [`canonize`].
#[derive(Debug, Serialize)]
pub struct CanonizeReport {
    /// Candidates sorted by transmissibility descending (unmeasured last).
    pub candidates: Vec<CandidateCanonicality>,
    pub judges: Vec<String>,
    pub comparisons_used: usize,
    pub cost_nanodollars: i64,
}

/// Errors from [`canonize`].
#[derive(Debug, thiserror::Error)]
pub enum CanonizeError {
    #[error("need at least 2 judges for transmissibility, got {0}")]
    TooFewJudges(usize),
    #[error("need at least 3 entities, got {0}")]
    TooFewEntities(usize),
    #[error("no candidate wordings")]
    NoCandidates,
    #[error(transparent)]
    Sort(#[from] SortError),
}

fn execution<'a>(
    gateway: Arc<dyn ChatGateway>,
    cache: Option<&'a dyn PairwiseCache>,
    seed: u64,
) -> RerankExecution<'a> {
    let mut execution = RerankExecution::new(gateway, Attribution::new("cardinal::canonize"))
        .run_options(RerankRunOptions {
            rng_seed: Some(seed),
            cache_only: false,
        });
    if let Some(cache) = cache {
        execution = execution.cache(cache);
    }
    execution
}

/// Measure candidate wordings across judges and rank by canonicality.
/// `accepted` are already-canonical attributes; candidates are additionally
/// scored for redundancy against them (measured with the first judge).
pub async fn canonize(
    gateway: Arc<dyn ChatGateway>,
    cache: Option<&dyn PairwiseCache>,
    entities: Vec<RerankDocument>,
    candidates: Vec<String>,
    accepted: Vec<String>,
    opts: CanonizeOptions,
) -> Result<CanonizeReport, CanonizeError> {
    if opts.judges.len() < 2 {
        return Err(CanonizeError::TooFewJudges(opts.judges.len()));
    }
    if entities.len() < 3 {
        return Err(CanonizeError::TooFewEntities(entities.len()));
    }
    if candidates.is_empty() {
        return Err(CanonizeError::NoCandidates);
    }
    let mut comparisons = 0usize;
    let mut cost = 0i64;

    // Latents for one (wording, judge), aligned to entity input order.
    let mut measure = async |prompt: &str, judge: &str| -> Result<Vec<f64>, CanonizeError> {
        let sorted = sort_documents(
            entities.clone(),
            prompt,
            execution(gateway.clone(), cache, opts.seed),
            SortOptions {
                model: Some(judge.to_string()),
                comparison_budget: opts.comparison_budget,
                ..Default::default()
            },
        )
        .await?;
        comparisons += sorted.meta.comparisons_used;
        cost += sorted.meta.provider_cost_nanodollars;
        let mut latents = vec![0.0f64; entities.len()];
        for item in &sorted.items {
            if let Some(idx) = entities.iter().position(|e| e.id == item.id) {
                latents[idx] = item.latent_mean;
            }
        }
        Ok(latents)
    };

    // Accepted attributes: measured once, first judge, for redundancy.
    let mut accepted_latents: Vec<Vec<f64>> = Vec::new();
    for attribute in &accepted {
        accepted_latents.push(measure(attribute, &opts.judges[0]).await?);
    }

    let mut results = Vec::with_capacity(candidates.len());
    for prompt in &candidates {
        let mut latents_per_judge = Vec::with_capacity(opts.judges.len());
        for judge in &opts.judges {
            latents_per_judge.push(measure(prompt, judge).await?);
        }
        // Transmissibility: mean pairwise Spearman across judges.
        let mut rhos = Vec::new();
        for a in 0..latents_per_judge.len() {
            for b in (a + 1)..latents_per_judge.len() {
                if let Some(rho) = spearman(&latents_per_judge[a], &latents_per_judge[b]) {
                    rhos.push(rho);
                }
            }
        }
        let transmissibility =
            (!rhos.is_empty()).then(|| rhos.iter().sum::<f64>() / rhos.len() as f64);
        // Signal: mean per-judge latent std.
        let signal_nats = latents_per_judge
            .iter()
            .map(|l| {
                let mean = l.iter().sum::<f64>() / l.len() as f64;
                (l.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / l.len() as f64).sqrt()
            })
            .sum::<f64>()
            / latents_per_judge.len().max(1) as f64;
        // Redundancy vs the accepted set (first judge's view).
        let redundancy = if accepted_latents.is_empty() {
            None
        } else {
            accepted_latents
                .iter()
                .filter_map(|acc| spearman(&latents_per_judge[0], acc))
                .map(f64::abs)
                .fold(None, |max: Option<f64>, rho| {
                    Some(max.map_or(rho, |m| m.max(rho)))
                })
        };
        results.push(CandidateCanonicality {
            prompt: prompt.clone(),
            transmissibility,
            signal_nats,
            redundancy,
            latents_per_judge,
        });
    }
    results.sort_by(|x, y| {
        y.transmissibility
            .unwrap_or(f64::NEG_INFINITY)
            .partial_cmp(&x.transmissibility.unwrap_or(f64::NEG_INFINITY))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(CanonizeReport {
        candidates: results,
        judges: opts.judges,
        comparisons_used: comparisons,
        cost_nanodollars: cost,
    })
}
