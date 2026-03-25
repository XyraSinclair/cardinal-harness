use std::path::Path;
use std::sync::Arc;

use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};

use crate::cache::PairwiseCache;
use crate::gateway::ChatGateway;
use crate::rerank::model_policy::ModelPolicy;
use crate::rerank::types::MultiRerankGateSpec;

use super::{
    default_extended_attributes, run_pipeline_with_trace_file, ContextFile, ModelPreset,
    PipelineAttribute, PipelineRankConfig, PipelineRequest, PipelineSession,
};

fn write_task_artifacts(
    task_id: &str,
    session: &PipelineSession,
    out_dir: &Path,
    synthesis_out_dir: Option<&Path>,
) -> Result<(), String> {
    let session_path = out_dir.join(format!("{}.json", task_id));
    let session_json = serde_json::to_string_pretty(session).map_err(|err| {
        format!(
            "failed to serialize session artifact {}: {}",
            session_path.display(),
            err
        )
    })?;
    std::fs::write(&session_path, session_json).map_err(|err| {
        format!(
            "failed to write session artifact {}: {}",
            session_path.display(),
            err
        )
    })?;

    if let Some(synth_dir) = synthesis_out_dir {
        let synth_path = synth_dir.join(format!("{}.md", task_id));
        std::fs::write(&synth_path, &session.synthesis.content).map_err(|err| {
            format!(
                "failed to write synthesis artifact {}: {}",
                synth_path.display(),
                err
            )
        })?;
    }

    Ok(())
}

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

            let mut context_files = shared_context;
            context_files.extend(task.extra_context_files.clone());

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
                .map(|dir| dir.join(format!("{}.trace.jsonl", task_id)));

            let result =
                run_pipeline_with_trace_file(gateway, cache, model_policy, trace_path, req, gates)
                    .await;

            match result {
                Ok(session) => {
                    let top_model = session
                        .ranking
                        .entities
                        .iter()
                        .find(|entity| entity.rank == Some(1))
                        .map(|entity| entity.id.clone());
                    let cost = session.cost.total_cost_nanodollars;
                    if let Err(err_msg) = write_task_artifacts(
                        &task_id,
                        &session,
                        &out_dir,
                        synthesis_out_dir.as_deref(),
                    ) {
                        eprintln!(
                            "[flywheel] [{}/{}] FAILED: {} — {}",
                            idx + 1,
                            task_count,
                            task_id,
                            err_msg
                        );
                        return (task_id, false, cost, top_model, Some(err_msg));
                    }

                    eprintln!(
                        "[flywheel] [{}/{}] done: {} (${:.4})",
                        idx + 1,
                        task_count,
                        task_id,
                        cost as f64 / 1_000_000_000.0
                    );

                    (task_id, true, cost, top_model, None)
                }
                Err(err) => {
                    let err_msg = err.to_string();
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
