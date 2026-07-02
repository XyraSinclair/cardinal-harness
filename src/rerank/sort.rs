//! Sort plain text items by a natural-language criterion.
//!
//! This is the list-in, list-out convenience surface over the single-attribute
//! rerank path in [`super::simple`]. Same planner, same solver, same stopping
//! semantics, same receipts — with none of the request-shape ceremony.
//!
//! ```no_run
//! use std::sync::Arc;
//! use cardinal_harness::gateway::NoopUsageSink;
//! use cardinal_harness::rerank::sort::{sort_texts, SortOptions};
//! use cardinal_harness::rerank::RerankExecution;
//! use cardinal_harness::{Attribution, ProviderGateway};
//!
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;
//! let execution = RerankExecution::new(Arc::new(gateway), Attribution::new("docs::sort"));
//!
//! let sorted = sort_texts(
//!     vec![
//!         "a bird in the hand is worth two in the bush".into(),
//!         "premature optimization is the root of all evil".into(),
//!         "measure twice, cut once".into(),
//!     ],
//!     "usefulness as advice for a software engineer",
//!     execution,
//!     SortOptions::default(),
//! )
//! .await?;
//!
//! for item in &sorted.items {
//!     println!("{:>2}. {:.3} ± {:.3}  {}", item.rank, item.latent_mean, item.latent_std, item.text);
//! }
//! # Ok(())
//! # }
//! ```

use super::multi::{MultiRerankError, RerankExecution};
use super::simple;
use super::types::{RerankDocument, RerankMeta, RerankRequest};
use serde::{Deserialize, Serialize};

/// Attribute id used for cache keys by [`sort_texts`].
///
/// The cache key also hashes the criterion text, so different criteria never
/// collide even though they share this id.
pub const SORT_ATTRIBUTE_ID: &str = "sort";

/// Options for [`sort_texts`]. All fields optional; defaults match the
/// single-attribute rerank defaults.
#[derive(Debug, Clone, Default)]
pub struct SortOptions {
    /// Model slug (OpenRouter), e.g. `anthropic/claude-sonnet-4.6`.
    pub model: Option<String>,
    /// Maximum pairwise comparisons to spend.
    pub comparison_budget: Option<usize>,
    /// Certify only the top k of the list; the tail is still returned,
    /// ordered by posterior mean, but the stopping rule targets the top-k
    /// boundary. Default: the whole list.
    pub top_k: Option<usize>,
    /// Stop when the estimated top-k error falls below this (default 0.1).
    pub tolerated_error: Option<f64>,
    /// Maximum concurrent comparisons.
    pub comparison_concurrency: Option<usize>,
    /// Maximum repeats per pair.
    pub max_pair_repeats: Option<usize>,
}

/// One sorted item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortedItem {
    /// Stable id (synthesized `item-<n>` for plain text input).
    pub id: String,
    /// The original item text, byte-identical to the input.
    pub text: String,
    /// 1-based rank, best first.
    pub rank: usize,
    /// Posterior mean in latent (log-ratio) space.
    pub latent_mean: f64,
    /// Posterior standard deviation.
    pub latent_std: f64,
    /// Robust z-score across the list.
    pub z_score: f64,
    /// Percentile among items (0..1).
    pub percentile: f64,
}

/// Result of [`sort_texts`]: items sorted best-first, plus the full run
/// receipts (comparisons, tokens, cost, stop reason).
#[derive(Debug, Serialize)]
pub struct SortedTexts {
    /// Items in rank order (best first).
    pub items: Vec<SortedItem>,
    /// Run metadata: comparisons attempted/used/cached/refused, provider
    /// tokens and cost, stop reason.
    pub meta: RerankMeta,
}

/// Errors from [`sort_texts`] / [`sort_documents`].
#[derive(Debug, thiserror::Error)]
pub enum SortError {
    /// The input list was empty.
    #[error("cannot sort an empty list")]
    EmptyInput,
    /// Two documents shared the same id.
    #[error("duplicate document id: {0}")]
    DuplicateId(String),
    /// The underlying rerank failed.
    #[error(transparent)]
    Rerank(#[from] MultiRerankError),
}

/// Sort `texts` by `criterion`, best first.
///
/// Thin wrapper over the single-attribute rerank path: each pairwise ratio
/// judgement ("how many times more of the criterion does A have than B?")
/// becomes a noisy log-ratio observation; a robust solver fits globally
/// consistent scores with uncertainty; the planner spends each comparison
/// where it buys the most information about the (top-k) order.
pub async fn sort_texts(
    texts: Vec<String>,
    criterion: &str,
    execution: RerankExecution<'_>,
    opts: SortOptions,
) -> Result<SortedTexts, SortError> {
    let documents: Vec<RerankDocument> = texts
        .into_iter()
        .enumerate()
        .map(|(idx, text)| RerankDocument {
            id: format!("item-{idx:04}"),
            text,
        })
        .collect();
    sort_documents(documents, criterion, execution, opts).await
}

/// Sort caller-identified documents by `criterion`, best first.
///
/// Like [`sort_texts`], but the caller owns the document ids (which appear in
/// results, traces, and cache keys as entity ids).
pub async fn sort_documents(
    documents: Vec<RerankDocument>,
    criterion: &str,
    execution: RerankExecution<'_>,
    opts: SortOptions,
) -> Result<SortedTexts, SortError> {
    if documents.is_empty() {
        return Err(SortError::EmptyInput);
    }
    if documents.len() == 1 {
        // A single item is already sorted; the rerank path requires >= 2
        // entities, so return the trivial answer with an all-zero receipt.
        let doc = documents.into_iter().next().expect("len checked above");
        return Ok(SortedTexts {
            items: vec![SortedItem {
                id: doc.id,
                text: doc.text,
                rank: 1,
                latent_mean: 0.0,
                latent_std: 0.0,
                z_score: 0.0,
                percentile: 1.0,
            }],
            meta: RerankMeta {
                topk_error: 0.0,
                tolerated_error: opts.tolerated_error.unwrap_or(0.1),
                comparisons_attempted: 0,
                comparisons_used: 0,
                comparisons_refused: 0,
                comparisons_cached: 0,
                comparison_budget: 0,
                latency_ms: 0,
                model_used: String::new(),
                rater_id_used: String::new(),
                provider_input_tokens: 0,
                provider_output_tokens: 0,
                provider_cost_nanodollars: 0,
                provider_cost_is_estimate: false,
                stop_reason: super::types::RerankStopReason::ToleratedErrorMet,
            },
        });
    }
    {
        let mut seen = std::collections::HashSet::with_capacity(documents.len());
        for doc in &documents {
            if !seen.insert(doc.id.as_str()) {
                return Err(SortError::DuplicateId(doc.id.clone()));
            }
        }
    }
    let texts: std::collections::HashMap<String, String> = documents
        .iter()
        .map(|doc| (doc.id.clone(), doc.text.clone()))
        .collect();
    let request = sort_request(documents, criterion, &opts);
    let response = simple::rerank(request, execution).await?;

    let mut items: Vec<SortedItem> = response
        .results
        .iter()
        .map(|result| SortedItem {
            id: result.id.clone(),
            text: texts
                .get(&result.id)
                .expect("rerank results only contain input document ids")
                .clone(),
            rank: result.rank,
            latent_mean: result.latent_mean,
            latent_std: result.latent_std,
            z_score: result.z_score,
            percentile: result.percentile,
        })
        .collect();
    items.sort_by_key(|item| item.rank);

    Ok(SortedTexts {
        items,
        meta: response.meta,
    })
}

/// Build the underlying single-attribute request used by [`sort_texts`].
///
/// Exposed so callers (and the CLI) can construct the exact same request
/// while owning document ids themselves.
pub fn sort_request(
    documents: Vec<RerankDocument>,
    criterion: &str,
    opts: &SortOptions,
) -> RerankRequest {
    // A whole-list sort (top_k = None) must not degenerate to k = n: with
    // k = n there is no k/k+1 boundary, the top-k error is trivially zero,
    // and the loop would stop before the first comparison. Target the middle
    // boundary instead — the band around it covers the most crowded region,
    // and forced exploration (min_explore_degree) still touches every item.
    // Callers who care about a specific boundary should set top_k explicitly.
    let n = documents.len();
    let top_k = opts.top_k.filter(|&k| k < n).or(Some(n.div_ceil(2)));
    RerankRequest {
        query: None,
        documents,
        attribute_id: SORT_ATTRIBUTE_ID.to_string(),
        attribute_prompt: criterion.to_string(),
        top_k,
        comparison_budget: opts.comparison_budget,
        latency_budget_ms: None,
        tolerated_error: opts.tolerated_error.unwrap_or(0.1),
        model: opts.model.clone(),
        rater_id: None,
        comparison_concurrency: opts.comparison_concurrency,
        max_pair_repeats: opts.max_pair_repeats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_request_maps_options() {
        let docs = vec![
            RerankDocument {
                id: "item-0000".into(),
                text: "a".into(),
            },
            RerankDocument {
                id: "item-0001".into(),
                text: "b".into(),
            },
        ];
        let opts = SortOptions {
            model: Some("test/model".into()),
            comparison_budget: Some(7),
            top_k: Some(1),
            tolerated_error: Some(0.05),
            comparison_concurrency: Some(2),
            max_pair_repeats: Some(1),
        };
        let req = sort_request(docs, "clarity", &opts);
        assert_eq!(req.attribute_id, SORT_ATTRIBUTE_ID);
        assert_eq!(req.attribute_prompt, "clarity");
        assert_eq!(req.top_k, Some(1));
        assert_eq!(req.comparison_budget, Some(7));
        assert_eq!(req.tolerated_error, 0.05);
        assert_eq!(req.model.as_deref(), Some("test/model"));
        assert_eq!(req.comparison_concurrency, Some(2));
        assert_eq!(req.max_pair_repeats, Some(1));
        assert_eq!(req.documents.len(), 2);
    }

    #[test]
    fn sort_request_defaults_top_k_to_middle_boundary() {
        let docs = |n: usize| {
            (0..n)
                .map(|idx| RerankDocument {
                    id: format!("item-{idx:04}"),
                    text: format!("t{idx}"),
                })
                .collect::<Vec<_>>()
        };
        // Whole-list sort: no top_k => middle boundary, never k = n.
        let req = sort_request(docs(4), "clarity", &SortOptions::default());
        assert_eq!(req.top_k, Some(2));
        let req = sort_request(docs(9), "clarity", &SortOptions::default());
        assert_eq!(req.top_k, Some(5));
        // Degenerate k >= n is remapped the same way.
        let opts = SortOptions {
            top_k: Some(4),
            ..Default::default()
        };
        let req = sort_request(docs(4), "clarity", &opts);
        assert_eq!(req.top_k, Some(2));
        // Real boundaries pass through untouched.
        let opts = SortOptions {
            top_k: Some(3),
            ..Default::default()
        };
        let req = sort_request(docs(4), "clarity", &opts);
        assert_eq!(req.top_k, Some(3));
    }

    #[tokio::test]
    async fn sort_texts_rejects_empty_input() {
        // Construct a gateway that will never be reached: empty input fails first.
        let adapter = crate::gateway::openrouter::OpenRouterAdapter::with_config(
            "sk-unused",
            "http://127.0.0.1:9",
            std::time::Duration::from_secs(1),
            None,
            None,
        )
        .unwrap();
        let gateway = std::sync::Arc::new(crate::gateway::ProviderGateway::with_config(
            adapter,
            std::sync::Arc::new(crate::gateway::NoopUsageSink),
            crate::gateway::GatewayConfig::default(),
        ));
        let execution = RerankExecution::new(gateway, crate::gateway::Attribution::new("test"));
        let err = sort_texts(vec![], "clarity", execution, SortOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, SortError::EmptyInput));
    }
}
