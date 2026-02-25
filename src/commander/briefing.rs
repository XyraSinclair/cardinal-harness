//! Briefing phase — compile cross-run intelligence before spending compute.
//!
//! The officer reads the situation: prior runs, acceptance rates, model performance,
//! recent proposals. Then it generates a situational assessment via LLM to guide
//! decomposition with awareness of history.

use serde::{Deserialize, Serialize};

use crate::gateway::{Attribution, ChatGateway, ChatModel, ChatRequest, Message};

use super::extract::extract_json;
use super::store::{CommanderStore, ModelPerformanceSummary, ProposalStatus, RunBrief, StoreError};

// =============================================================================
// Types
// =============================================================================

/// Compiled context from prior runs, fed to the briefing LLM.
#[derive(Debug, Clone, Serialize)]
pub struct BriefingContext {
    pub recent_runs: Vec<RunBrief>,
    pub acceptance_rate: f64,
    pub cost_per_accepted: Option<f64>,
    pub recent_proposal_titles: Vec<(String, String)>, // (title, status)
    pub model_performance: Vec<ModelPerformanceSummary>,
}

/// LLM-generated briefing output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BriefingResponse {
    #[serde(default)]
    pub situation_summary: String,
    #[serde(default)]
    pub overlap_warnings: Vec<String>,
    #[serde(default = "default_task_range")]
    pub recommended_task_count: (usize, usize),
    #[serde(default)]
    pub cost_per_accepted: Option<f64>,
    #[serde(default)]
    pub acceptance_rate: Option<f64>,
    #[serde(default)]
    pub rejection_patterns: Vec<String>,
    #[serde(default)]
    pub directive_refinement: Option<String>,
}

fn default_task_range() -> (usize, usize) {
    (3, 6)
}

#[derive(Debug, thiserror::Error)]
pub enum BriefingError {
    #[error("Store error: {0}")]
    Store(#[from] StoreError),
    #[error("LLM call failed: {0}")]
    LlmFailed(#[from] crate::gateway::error::ProviderError),
    #[error("JSON parse failed: {0}")]
    JsonParse(String),
}

// =============================================================================
// System prompt
// =============================================================================

const BRIEFING_SYSTEM_PROMPT: &str = "\
You are an intelligence officer for an autonomous code analysis system. Before the system \
spends compute on a new analysis run, you review the situation: prior runs, acceptance rates, \
model performance, and recent proposals.

Your job is to produce a concise situational assessment that will guide the decomposition \
phase. Be honest and direct.

Analyze the provided context and produce:
- situation_summary: 2-3 sentences summarizing the current state (what's been explored, \
  what's working, what's not)
- overlap_warnings: array of strings flagging any overlap between the new directive and \
  recent proposals (avoid paying twice for the same insight)
- recommended_task_count: [min, max] range based on historical yield and the directive scope
- cost_per_accepted: the current cost per accepted proposal in nanodollars (from context)
- acceptance_rate: the current acceptance rate as a fraction (from context)
- rejection_patterns: patterns in rejected proposals (what the reviewer doesn't want)
- directive_refinement: optional suggestion to narrow/broaden the directive, or null if it's fine

Respond with JSON only:
{
  \"situation_summary\": \"...\",
  \"overlap_warnings\": [\"...\"],
  \"recommended_task_count\": [3, 6],
  \"cost_per_accepted\": null,
  \"acceptance_rate\": null,
  \"rejection_patterns\": [\"...\"],
  \"directive_refinement\": null
}";

// =============================================================================
// Briefing
// =============================================================================

/// Compile cross-run intelligence from the store.
pub async fn compile_briefing_context(
    store: &CommanderStore,
) -> Result<BriefingContext, BriefingError> {
    let recent_runs = store.recent_run_summaries(5).await?;
    let acceptance_rate = store.acceptance_rate(Some(5)).await?;
    let cost_per_accepted = store.cost_per_accepted().await?;
    let recent_titles = store.recent_proposal_titles(20).await?;
    let model_performance = store.model_performance_summary().await?;

    let titles_with_status: Vec<(String, String)> = recent_titles
        .into_iter()
        .map(|(title, status)| {
            let status_str = match status {
                ProposalStatus::Pending => "pending",
                ProposalStatus::Accepted => "accepted",
                ProposalStatus::Rejected => "rejected",
                ProposalStatus::Deferred => "deferred",
                ProposalStatus::Implemented => "implemented",
            };
            (title, status_str.to_string())
        })
        .collect();

    Ok(BriefingContext {
        recent_runs,
        acceptance_rate,
        cost_per_accepted,
        recent_proposal_titles: titles_with_status,
        model_performance,
    })
}

/// Run the briefing phase: compile context, call LLM, return assessment.
///
/// Returns the briefing response and cost in nanodollars.
pub async fn run_briefing(
    gateway: &dyn ChatGateway,
    model: &str,
    directive: &str,
    context: &BriefingContext,
) -> Result<(BriefingResponse, i64), BriefingError> {
    let context_json = serde_json::to_string_pretty(context).unwrap_or_default();

    let user_prompt =
        format!("## New Directive\n\n{directive}\n\n## Cross-Run Intelligence\n\n{context_json}");

    let messages = vec![
        Message::system(BRIEFING_SYSTEM_PROMPT),
        Message::user(user_prompt),
    ];

    let req = ChatRequest::new(
        ChatModel::openrouter(model),
        messages,
        Attribution::new("commander::briefing"),
    )
    .temperature(0.2)
    .max_tokens(2048)
    .json();

    let resp = gateway.chat(req).await?;
    let cost = resp.cost_nanodollars;

    let json_str = extract_json(&resp.content);
    let parsed: BriefingResponse = serde_json::from_str(json_str).map_err(|e| {
        let preview: String = resp.content.chars().take(500).collect();
        BriefingError::JsonParse(format!(
            "failed to parse briefing response: {} — raw: {}",
            e, preview
        ))
    })?;

    Ok((parsed, cost))
}

/// Format a briefing response as a markdown section for injection into decomposition.
pub fn format_briefing_for_prompt(briefing: &BriefingResponse) -> String {
    let mut parts = Vec::new();

    parts.push(format!("**Situation:** {}", briefing.situation_summary));

    if let Some(rate) = briefing.acceptance_rate {
        parts.push(format!("**Acceptance rate:** {:.0}%", rate * 100.0));
    }

    if let Some(cpa) = briefing.cost_per_accepted {
        parts.push(format!(
            "**Cost per accepted proposal:** ${:.2}",
            cpa / 1_000_000_000.0
        ));
    }

    if !briefing.overlap_warnings.is_empty() {
        parts.push("**Overlap warnings:**".to_string());
        for w in &briefing.overlap_warnings {
            parts.push(format!("- {w}"));
        }
    }

    if !briefing.rejection_patterns.is_empty() {
        parts.push("**Rejection patterns:**".to_string());
        for p in &briefing.rejection_patterns {
            parts.push(format!("- {p}"));
        }
    }

    if let Some(ref refinement) = briefing.directive_refinement {
        parts.push(format!("**Directive refinement suggestion:** {refinement}"));
    }

    parts.push(format!(
        "**Recommended task count:** {}-{}",
        briefing.recommended_task_count.0, briefing.recommended_task_count.1
    ));

    parts.join("\n")
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_briefing_response_parse() {
        let json = r#"{
            "situation_summary": "Two prior runs focused on durability.",
            "overlap_warnings": ["fsync proposals already exist"],
            "recommended_task_count": [3, 5],
            "cost_per_accepted": 2000000000.0,
            "acceptance_rate": 0.67,
            "rejection_patterns": ["documentation-only proposals"],
            "directive_refinement": null
        }"#;
        let resp: BriefingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.overlap_warnings.len(), 1);
        assert_eq!(resp.recommended_task_count, (3, 5));
        assert!((resp.acceptance_rate.unwrap() - 0.67).abs() < 1e-6);
    }

    #[test]
    fn test_briefing_response_defaults() {
        let json = r#"{"situation_summary": "Fresh start"}"#;
        let resp: BriefingResponse = serde_json::from_str(json).unwrap();
        assert!(resp.overlap_warnings.is_empty());
        assert_eq!(resp.recommended_task_count, (3, 6));
        assert!(resp.directive_refinement.is_none());
    }

    #[test]
    fn test_format_briefing_for_prompt() {
        let briefing = BriefingResponse {
            situation_summary: "Two runs complete.".to_string(),
            overlap_warnings: vec!["fsync already covered".to_string()],
            recommended_task_count: (3, 5),
            cost_per_accepted: Some(2_000_000_000.0),
            acceptance_rate: Some(0.67),
            rejection_patterns: vec!["docs-only".to_string()],
            directive_refinement: Some("Focus on concurrency".to_string()),
        };
        let formatted = format_briefing_for_prompt(&briefing);
        assert!(formatted.contains("Situation:"));
        assert!(formatted.contains("67%"));
        assert!(formatted.contains("$2.00"));
        assert!(formatted.contains("fsync already covered"));
        assert!(formatted.contains("Focus on concurrency"));
    }

    #[tokio::test]
    async fn test_compile_briefing_context_empty_store() {
        let dir = tempfile::tempdir().expect("failed to create tempdir");
        let path = dir.path().join("briefing_test.sqlite");
        std::mem::forget(dir);
        let store = CommanderStore::new(path).expect("create store");

        let ctx = compile_briefing_context(&store).await.unwrap();
        assert!(ctx.recent_runs.is_empty());
        assert_eq!(ctx.acceptance_rate, 0.0);
        assert_eq!(ctx.cost_per_accepted, None);
    }
}
