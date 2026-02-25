//! Reflection phase — after-action review of a commander run.
//!
//! The officer honestly assesses what worked, what was wasted, and what should
//! change for next time. Stores the reflection persistently for cross-run learning.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::gateway::{Attribution, ChatGateway, ChatModel, ChatRequest, Message};

use super::extract::extract_json;
use super::store::{CommanderStore, StoreError};

// =============================================================================
// Types
// =============================================================================

/// Data compiled from the run for reflection input.
#[derive(Debug, Clone, Serialize)]
pub struct RunReflectionData {
    pub run_id: i64,
    pub directive: String,
    pub tasks_completed: i64,
    pub tasks_failed: i64,
    pub proposals_generated: i64,
    pub decompose_cost: i64,
    pub flywheel_cost: i64,
    pub extract_cost: i64,
    pub briefing_cost: i64,
    pub total_cost: i64,
    pub cost_per_proposal: Option<f64>,
    pub prior_acceptance_rate: f64,
    pub prior_cost_per_accepted: Option<f64>,
    pub task_details: Vec<TaskReflectionDetail>,
    pub model_rankings_summary: Vec<ModelRankEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskReflectionDetail {
    pub task_id: String,
    pub success: bool,
    pub cost_nanodollars: i64,
    pub proposals_count: i64,
    pub top_model: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelRankEntry {
    pub model: String,
    pub task_id: String,
    pub rank: i64,
}

/// LLM-generated reflection output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionResponse {
    #[serde(default = "default_quality")]
    pub quality_score: f64,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub efficiency_analysis: String,
    #[serde(default)]
    pub task_quality_notes: Vec<String>,
    #[serde(default)]
    pub model_insights: HashMap<String, String>,
    #[serde(default)]
    pub recommendations: Vec<String>,
}

fn default_quality() -> f64 {
    0.5
}

#[derive(Debug, thiserror::Error)]
pub enum ReflectionError {
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

const REFLECTION_SYSTEM_PROMPT: &str = "\
You are a post-mission analyst for an autonomous code analysis system. A run has just \
completed. You receive detailed data about what happened: tasks, costs, proposals, model \
performance. Your job is to produce an honest after-action review.

Analyze the provided run data and produce:
- quality_score: 0.0-1.0 overall quality (0=wasted money, 1=every dollar produced value)
- summary: 2-3 sentences on the run's overall performance
- efficiency_analysis: analysis of cost efficiency — cost per proposal, comparison to prior runs
- task_quality_notes: array of per-task observations (which were productive, which were wasted)
- model_insights: object mapping model names to performance notes (which to trust, which to drop)
- recommendations: array of specific, actionable changes for next run

Be ruthlessly honest. If tasks produced zero proposals, say so. If a model consistently \
loses, recommend removing it. If the directive was too broad, say that.

Respond with JSON only:
{
  \"quality_score\": 0.7,
  \"summary\": \"...\",
  \"efficiency_analysis\": \"...\",
  \"task_quality_notes\": [\"...\"],
  \"model_insights\": {\"model-name\": \"...\"},
  \"recommendations\": [\"...\"]
}";

// =============================================================================
// Reflection
// =============================================================================

/// Compile run data for reflection from the store.
pub async fn compile_reflection_data(
    store: &CommanderStore,
    run_id: i64,
    briefing_cost: i64,
) -> Result<RunReflectionData, ReflectionError> {
    let run = store.get_run(run_id).await?;
    let tasks = store.get_tasks_for_run(run_id).await?;
    let proposals = store.list_proposals(None).await?;
    let rankings = store.get_model_rankings().await?;
    let prior_rate = store.acceptance_rate(Some(5)).await?;
    let prior_cpa = store.cost_per_accepted().await?;

    let run_proposals: Vec<_> = proposals.iter().filter(|p| p.run_id == run_id).collect();
    let proposals_generated = run_proposals.len() as i64;

    let cost_per_proposal = if proposals_generated > 0 {
        Some(run.total_cost_nanodollars as f64 / proposals_generated as f64)
    } else {
        None
    };

    let task_details: Vec<TaskReflectionDetail> = tasks
        .iter()
        .map(|t| {
            let task_proposals = run_proposals
                .iter()
                .filter(|p| p.task_id == t.task_id)
                .count();
            TaskReflectionDetail {
                task_id: t.task_id.clone(),
                success: t.success.unwrap_or(false),
                cost_nanodollars: t.cost_nanodollars,
                proposals_count: task_proposals as i64,
                top_model: t.top_model.clone(),
            }
        })
        .collect();

    let model_rankings_summary: Vec<ModelRankEntry> = rankings
        .iter()
        .filter(|r| r.run_id == run_id)
        .map(|r| ModelRankEntry {
            model: r.model.clone(),
            task_id: r.task_id.clone(),
            rank: r.rank,
        })
        .collect();

    Ok(RunReflectionData {
        run_id,
        directive: run.directive,
        tasks_completed: run.tasks_completed,
        tasks_failed: run.tasks_failed,
        proposals_generated,
        decompose_cost: run.decompose_cost_nanodollars,
        flywheel_cost: run.flywheel_cost_nanodollars,
        extract_cost: run.extract_cost_nanodollars,
        briefing_cost,
        total_cost: run.total_cost_nanodollars,
        cost_per_proposal,
        prior_acceptance_rate: prior_rate,
        prior_cost_per_accepted: prior_cpa,
        task_details,
        model_rankings_summary,
    })
}

/// Run the reflection LLM call.
///
/// Returns the reflection response and cost in nanodollars.
pub async fn run_reflection(
    gateway: &dyn ChatGateway,
    model: &str,
    data: &RunReflectionData,
) -> Result<(ReflectionResponse, i64), ReflectionError> {
    let data_json = serde_json::to_string_pretty(data).unwrap_or_default();

    let user_prompt = format!("## Run Data\n\n{data_json}");

    let messages = vec![
        Message::system(REFLECTION_SYSTEM_PROMPT),
        Message::user(user_prompt),
    ];

    let req = ChatRequest::new(
        ChatModel::openrouter(model),
        messages,
        Attribution::new("commander::reflection"),
    )
    .temperature(0.2)
    .max_tokens(4096)
    .json();

    let resp = gateway.chat(req).await?;
    let cost = resp.cost_nanodollars;

    let json_str = extract_json(&resp.content);
    let parsed: ReflectionResponse = serde_json::from_str(json_str).map_err(|e| {
        let preview: String = resp.content.chars().take(500).collect();
        ReflectionError::JsonParse(format!(
            "failed to parse reflection response: {} — raw: {}",
            e, preview
        ))
    })?;

    Ok((parsed, cost))
}

/// Print a reflection summary to stderr.
pub fn print_reflection_summary(reflection: &ReflectionResponse) {
    eprintln!();
    eprintln!("=== Officer Reflection ===");
    eprintln!("Quality: {:.0}/100", reflection.quality_score * 100.0);
    eprintln!("{}", reflection.summary);
    eprintln!();
    eprintln!("Efficiency: {}", reflection.efficiency_analysis);

    if !reflection.task_quality_notes.is_empty() {
        eprintln!();
        eprintln!("Task notes:");
        for note in &reflection.task_quality_notes {
            eprintln!("  - {note}");
        }
    }

    if !reflection.model_insights.is_empty() {
        eprintln!();
        eprintln!("Model insights:");
        for (model, insight) in &reflection.model_insights {
            eprintln!("  {model}: {insight}");
        }
    }

    if !reflection.recommendations.is_empty() {
        eprintln!();
        eprintln!("Recommendations:");
        for rec in &reflection.recommendations {
            eprintln!("  - {rec}");
        }
    }
    eprintln!();
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflection_response_parse() {
        let json = r#"{
            "quality_score": 0.8,
            "summary": "Productive run with good signal-to-noise.",
            "efficiency_analysis": "Cost per proposal: $0.50, below average.",
            "task_quality_notes": ["task-1 produced 3 proposals", "task-2 produced nothing"],
            "model_insights": {"opus": "consistently strong", "grok": "fast but shallow"},
            "recommendations": ["drop grok from frontier preset", "narrow the directive"]
        }"#;
        let resp: ReflectionResponse = serde_json::from_str(json).unwrap();
        assert!((resp.quality_score - 0.8).abs() < 1e-6);
        assert_eq!(resp.recommendations.len(), 2);
        assert_eq!(resp.model_insights.len(), 2);
    }

    #[test]
    fn test_reflection_response_defaults() {
        let json = r#"{"summary": "ok run"}"#;
        let resp: ReflectionResponse = serde_json::from_str(json).unwrap();
        assert!((resp.quality_score - 0.5).abs() < 1e-6);
        assert!(resp.recommendations.is_empty());
        assert!(resp.model_insights.is_empty());
    }

    #[test]
    fn test_print_reflection_summary_no_panic() {
        let resp = ReflectionResponse {
            quality_score: 0.7,
            summary: "Good run.".to_string(),
            efficiency_analysis: "Reasonable cost.".to_string(),
            task_quality_notes: vec!["task-1 was good".to_string()],
            model_insights: HashMap::from([("opus".to_string(), "strong".to_string())]),
            recommendations: vec!["do more".to_string()],
        };
        // Should not panic
        print_reflection_summary(&resp);
    }
}
