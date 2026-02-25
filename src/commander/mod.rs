//! Code Commander — strategic agent that sits above the flywheel.
//!
//! The officer flow: brief → decompose_enhanced → flywheel → extract_enhanced → reflect → persist_everything
//!
//! Takes a high-level directive, briefs itself on prior run history, decomposes into
//! investigation tasks via LLM, runs the flywheel (multi-model generate → rank → synthesize),
//! extracts actionable proposals with dedup context, reflects on the run, and stores
//! everything persistently including full LLM audit trail.

pub mod briefing;
pub mod dashboard;
pub mod decompose;
pub mod extract;
pub mod reflection;
pub mod scan;
pub mod store;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::cache::SqlitePairwiseCache;
use crate::gateway::{ChatGateway, NoopUsageSink, ProviderGateway};
use crate::pipeline::{self, ModelPreset};
use crate::rerank::model_policy::ModelPolicy;

use self::briefing::{compile_briefing_context, run_briefing};
use self::decompose::{
    build_manifest, decompose_directive, estimate_flywheel_cost, gather_codebase_context,
};
use self::extract::extract_proposals;
use self::reflection::{compile_reflection_data, print_reflection_summary, run_reflection};
use self::store::{CommanderStore, ProposalStatus, RunStatus};

// =============================================================================
// Config
// =============================================================================

/// Configuration for a commander run.
#[derive(Debug, Clone)]
pub struct CommanderConfig {
    /// The high-level improvement directive.
    pub directive: String,
    /// Path to the SQLite store.
    pub store_path: PathBuf,
    /// Model preset for the flywheel.
    pub preset: ModelPreset,
    /// Hard spend cap in nanodollars.
    pub budget_nanodollars: i64,
    /// Path to pairwise cache for flywheel.
    pub cache_path: Option<PathBuf>,
    /// Number of concurrent flywheel tasks.
    pub parallel: usize,
    /// Model for decompose/extract LLM calls.
    pub commander_model: String,
    /// Enable requirement_alignment gate.
    pub use_gates: bool,
    /// Skip the reflection phase.
    pub no_reflection: bool,
    /// Fresh start: skip briefing and dedup context from prior runs.
    pub fresh: bool,
    /// Override the output directory (default: .cardinal_sessions/run_N/).
    pub output_dir: Option<PathBuf>,
}

impl Default for CommanderConfig {
    fn default() -> Self {
        Self {
            directive: String::new(),
            store_path: CommanderStore::default_path(),
            preset: ModelPreset::Balanced,
            budget_nanodollars: 5_000_000_000, // $5.00
            cache_path: None,
            parallel: 1,
            commander_model: "anthropic/claude-opus-4-6".into(),
            use_gates: false,
            no_reflection: false,
            fresh: false,
            output_dir: None,
        }
    }
}

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum CommanderError {
    #[error("Store error: {0}")]
    Store(#[from] store::StoreError),
    #[error("Decomposition error: {0}")]
    Decompose(#[from] decompose::DecomposeError),
    #[error("Extraction error: {0}")]
    Extract(#[from] extract::ExtractError),
    #[error("Briefing error: {0}")]
    Briefing(#[from] briefing::BriefingError),
    #[error("Reflection error: {0}")]
    Reflection(#[from] reflection::ReflectionError),
    #[error("Gateway error: {0}")]
    Gateway(String),
    #[error("Budget exceeded: estimated ${estimated:.2} exceeds budget ${budget:.2}")]
    BudgetExceeded { estimated: f64, budget: f64 },
    #[error("Cache error: {0}")]
    Cache(#[from] crate::cache::CacheError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// =============================================================================
// Persistent output directory
// =============================================================================

/// Compute the persistent output directory for a run.
/// Sessions, traces, and synthesis all live here.
fn run_output_dir(store_path: &Path, run_id: i64) -> PathBuf {
    let parent = store_path.parent().unwrap_or(Path::new("."));
    parent
        .join(".cardinal_sessions")
        .join(format!("run_{run_id}"))
}

// =============================================================================
// Orchestration
// =============================================================================

/// Run a commander session: brief → decompose → flywheel → extract → reflect → persist.
pub async fn run_command(config: CommanderConfig) -> Result<CommanderRunResult, CommanderError> {
    let store = CommanderStore::new(&config.store_path)?;

    eprintln!("[commander] directive: {}", config.directive);
    eprintln!(
        "[commander] preset: {:?}, budget: ${:.2}, model: {}",
        config.preset,
        config.budget_nanodollars as f64 / 1_000_000_000.0,
        config.commander_model
    );

    // Set up gateway
    let gateway: Arc<dyn ChatGateway> = Arc::new(
        ProviderGateway::from_env(Arc::new(NoopUsageSink))
            .map_err(|e| CommanderError::Gateway(e.to_string()))?,
    );

    // Create run record
    let preset_str = match config.preset {
        ModelPreset::Frontier => "frontier",
        ModelPreset::Balanced => "balanced",
        ModelPreset::Fast => "fast",
    };
    let run_id = store
        .create_run(
            &config.directive,
            &config.commander_model,
            preset_str,
            config.budget_nanodollars,
        )
        .await?;

    // --- Persistent output directory ---
    let out_dir = config
        .output_dir
        .clone()
        .unwrap_or_else(|| run_output_dir(&config.store_path, run_id));
    std::fs::create_dir_all(&out_dir)?;
    let synth_dir = out_dir.join("synthesis");
    std::fs::create_dir_all(&synth_dir)?;
    let trace_dir = out_dir.join("traces");
    std::fs::create_dir_all(&trace_dir)?;

    // --- Step 1: Briefing phase ---
    let (briefing, briefing_cost) = if config.fresh {
        eprintln!("[commander] --fresh: skipping briefing (clean slate)");
        (None, 0)
    } else {
        eprintln!("[commander] compiling operational briefing...");
        let briefing_context = compile_briefing_context(&store).await?;
        match run_briefing(
            gateway.as_ref(),
            &config.commander_model,
            &config.directive,
            &briefing_context,
        )
        .await
        {
            Ok(result) => {
                let (briefing, cost) = result;
                // Store LLM trace for briefing
                let ctx_json = serde_json::to_string(&briefing_context).unwrap_or_default();
                let parsed_json = serde_json::to_string(&briefing).unwrap_or_default();
                if let Err(e) = store
                    .insert_llm_trace(
                        run_id,
                        "briefing",
                        None,
                        &config.commander_model,
                        &ctx_json,
                        &parsed_json,
                        Some(&parsed_json),
                        cost,
                        0,
                        0,
                        0,
                    )
                    .await
                {
                    eprintln!("[commander] warning: failed to store briefing trace: {e}");
                }

                eprintln!("[commander] briefing: {}", briefing.situation_summary);
                if !briefing.overlap_warnings.is_empty() {
                    for w in &briefing.overlap_warnings {
                        eprintln!("[commander]   overlap: {w}");
                    }
                }
                (Some(briefing), cost)
            }
            Err(e) => {
                eprintln!("[commander] briefing failed (continuing without): {e}");
                (None, 0)
            }
        }
    };

    // --- Step 2: Gather codebase context ---
    eprintln!("[commander] gathering codebase context...");
    let codebase_context = gather_codebase_context();

    // --- Step 3: Decompose directive (enhanced with briefing) ---
    eprintln!("[commander] decomposing directive...");
    let (decomposition, decompose_cost, _decompose_raw) = match decompose_directive(
        gateway.as_ref(),
        &config.commander_model,
        &config.directive,
        &codebase_context,
        briefing.as_ref(),
    )
    .await
    {
        Ok(result) => {
            // Store LLM trace for decomposition
            let parsed_json = serde_json::to_string(&result.0).unwrap_or_default();
            if let Err(e) = store
                .insert_llm_trace(
                    run_id,
                    "decompose",
                    None,
                    &config.commander_model,
                    &config.directive,
                    &result.2,
                    Some(&parsed_json),
                    result.1,
                    0,
                    0,
                    0,
                )
                .await
            {
                eprintln!("[commander] warning: failed to store decompose trace: {e}");
            }
            result
        }
        Err(e) => {
            if let Err(e2) = store.update_run_status(run_id, RunStatus::Failed).await {
                eprintln!("[commander] warning: failed to update run status: {e2}");
            }
            return Err(e.into());
        }
    };

    let n_tasks = decomposition.tasks.len();
    eprintln!(
        "[commander] decomposed into {} tasks (difficulty: {})",
        n_tasks, decomposition.estimated_difficulty
    );

    // --- Step 4: Budget check ---
    let flywheel_estimate = estimate_flywheel_cost(n_tasks, config.preset);
    // Rough extraction cost: ~$0.02 per task
    let extract_estimate = (n_tasks as i64) * 20_000_000;
    // Include briefing cost in total
    let total_estimate = briefing_cost
        .saturating_add(decompose_cost)
        .saturating_add(flywheel_estimate)
        .saturating_add(extract_estimate);

    let est_dollars = total_estimate as f64 / 1_000_000_000.0;
    let budget_dollars = config.budget_nanodollars as f64 / 1_000_000_000.0;

    eprintln!(
        "[commander] estimated cost: ${:.2} (budget: ${:.2})",
        est_dollars, budget_dollars
    );

    if total_estimate > config.budget_nanodollars {
        if let Err(e) = store
            .update_run_status(run_id, RunStatus::BudgetExceeded)
            .await
        {
            eprintln!("[commander] warning: failed to update run status: {e}");
        }
        return Err(CommanderError::BudgetExceeded {
            estimated: est_dollars,
            budget: budget_dollars,
        });
    }

    // --- Step 5: Store decomposed tasks ---
    for (idx, task) in decomposition.tasks.iter().enumerate() {
        let globs_json = serde_json::to_string(&task.context_globs).unwrap_or_else(|_| "[]".into());
        store
            .insert_task(
                run_id,
                idx as i64,
                &task.id,
                &task.prompt,
                task.system_prompt.as_deref(),
                &globs_json,
                &task.rationale,
            )
            .await?;
    }

    // --- Step 6: Build manifest and run flywheel ---
    eprintln!("[commander] building manifest and running flywheel...");
    let manifest = build_manifest(&decomposition.tasks, config.preset, 10)?;

    let cache_inst = {
        let path = config
            .cache_path
            .unwrap_or_else(SqlitePairwiseCache::default_path);
        SqlitePairwiseCache::new(path)?
    };

    let gates = if config.use_gates {
        pipeline::default_gates()
    } else {
        Vec::new()
    };

    let summary = pipeline::run_flywheel(
        gateway.clone(),
        Some(&cache_inst),
        None::<Arc<dyn ModelPolicy>>,
        manifest,
        &out_dir,
        Some(&synth_dir),
        Some(&trace_dir),
        Some(config.preset),
        config.parallel,
        gates,
    )
    .await;

    let flywheel_cost = summary.total_cost_nanodollars;
    let mut tasks_completed: i64 = 0;
    let mut tasks_failed: i64 = 0;

    // --- Step 7: Process each task result (enhanced extraction with dedup) ---
    eprintln!("[commander] extracting proposals from {} tasks...", n_tasks);
    let mut total_extract_cost: i64 = 0;
    let mut total_proposals: usize = 0;

    // Fetch existing proposal titles for dedup (skip in fresh mode)
    let existing_titles = if config.fresh {
        Vec::new()
    } else {
        store.recent_proposal_titles(20).await.unwrap_or_default()
    };

    for ts in &summary.task_summaries {
        if !ts.success {
            tasks_failed += 1;
            if let Err(e) = store
                .update_task_result(run_id, &ts.task_id, false, 0, None, None)
                .await
            {
                eprintln!("[commander] warning: failed to update task result: {e}");
            }
            continue;
        }

        tasks_completed += 1;

        // Load session JSON (needed for rankings + fallback synthesis content)
        let session_path = out_dir.join(format!("{}.json", ts.task_id));
        let session_raw = std::fs::read_to_string(&session_path).ok();
        let session: Option<pipeline::PipelineSession> = session_raw.as_deref().and_then(|raw| {
            serde_json::from_str(raw)
                .map_err(|e| {
                    eprintln!(
                        "[commander] warning: failed to parse session JSON for {}: {e}",
                        ts.task_id
                    );
                    e
                })
                .ok()
        });

        // Resolve synthesis content: prefer .md file, fall back to session JSON
        let synth_path = synth_dir.join(format!("{}.md", ts.task_id));
        let synthesis_content = std::fs::read_to_string(&synth_path)
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| session.as_ref().map(|s| s.synthesis.content.clone()));

        let synthesis_content = match synthesis_content {
            Some(c) if !c.is_empty() => c,
            _ => {
                // No synthesis content at all — mark success but skip extraction
                store
                    .update_task_result(
                        run_id,
                        &ts.task_id,
                        true,
                        ts.cost_nanodollars,
                        ts.top_model.as_deref(),
                        None,
                    )
                    .await?;
                continue;
            }
        };

        if let Some(ref session) = session {
            process_task_result(
                &store,
                &gateway,
                &config.commander_model,
                run_id,
                &ts.task_id,
                &decomposition.tasks,
                ts.cost_nanodollars,
                ts.top_model.as_deref(),
                &synthesis_content,
                session,
                &existing_titles,
                &mut total_extract_cost,
                &mut total_proposals,
            )
            .await;
        } else {
            // No session JSON — store synthesis but skip rankings/extraction
            store
                .update_task_result(
                    run_id,
                    &ts.task_id,
                    true,
                    ts.cost_nanodollars,
                    ts.top_model.as_deref(),
                    Some(&synthesis_content),
                )
                .await?;
        }
    }

    // --- Step 8: Update run with final costs ---
    store
        .update_run_costs(
            run_id,
            decompose_cost,
            flywheel_cost,
            total_extract_cost,
            tasks_completed,
            tasks_failed,
        )
        .await?;

    // Store officer-specific costs
    let out_dir_str = out_dir.to_string_lossy().to_string();
    store
        .update_run_officer_costs(run_id, briefing_cost, 0, Some(&out_dir_str))
        .await?;

    // --- Step 9: Reflection phase ---
    let mut reflection_cost: i64 = 0;
    if !config.no_reflection {
        eprintln!("[commander] running reflection...");
        match run_reflection_phase(
            &store,
            gateway.as_ref(),
            &config.commander_model,
            run_id,
            briefing_cost,
        )
        .await
        {
            Ok(cost) => {
                reflection_cost = cost;
                // Update officer costs with reflection
                if let Err(e) = store
                    .update_run_officer_costs(
                        run_id,
                        briefing_cost,
                        reflection_cost,
                        Some(&out_dir_str),
                    )
                    .await
                {
                    eprintln!("[commander] warning: failed to update officer costs: {e}");
                }
            }
            Err(e) => {
                eprintln!("[commander] reflection failed (non-fatal): {e}");
            }
        }
    }

    let final_status = if tasks_failed > 0 && tasks_completed == 0 {
        RunStatus::Failed
    } else {
        RunStatus::Completed
    };
    store.update_run_status(run_id, final_status).await?;

    let total_cost = briefing_cost
        .saturating_add(decompose_cost)
        .saturating_add(flywheel_cost)
        .saturating_add(total_extract_cost)
        .saturating_add(reflection_cost);

    // Print summary
    eprintln!();
    eprintln!("=== Commander Run Complete ===");
    eprintln!(
        "Run #{}: {} tasks completed, {} failed",
        run_id, tasks_completed, tasks_failed
    );
    eprintln!("{} proposals generated", total_proposals);
    eprintln!(
        "Cost: ${:.4} (briefing: ${:.4}, decompose: ${:.4}, flywheel: ${:.4}, extract: ${:.4}, reflect: ${:.4})",
        total_cost as f64 / 1e9,
        briefing_cost as f64 / 1e9,
        decompose_cost as f64 / 1e9,
        flywheel_cost as f64 / 1e9,
        total_extract_cost as f64 / 1e9,
        reflection_cost as f64 / 1e9,
    );
    eprintln!("Output: {}", out_dir.display());
    eprintln!();
    eprintln!("Review proposals:");
    eprintln!(
        "  cardinal review --list --store {}",
        config.store_path.display()
    );
    eprintln!(
        "  cardinal dashboard --store {}",
        config.store_path.display()
    );
    eprintln!(
        "  cardinal traces --store {} --run {}",
        config.store_path.display(),
        run_id
    );

    Ok(CommanderRunResult {
        run_id,
        tasks_completed,
        tasks_failed,
        proposals_generated: total_proposals as i64,
        total_cost_nanodollars: total_cost,
    })
}

/// Run the reflection phase as a sub-step. Returns the reflection cost.
async fn run_reflection_phase(
    store: &CommanderStore,
    gateway: &dyn ChatGateway,
    commander_model: &str,
    run_id: i64,
    briefing_cost: i64,
) -> Result<i64, CommanderError> {
    let data = compile_reflection_data(store, run_id, briefing_cost).await?;
    let (response, cost) = run_reflection(gateway, commander_model, &data).await?;

    // Store LLM trace
    let data_json = serde_json::to_string(&data).unwrap_or_default();
    let parsed_json = serde_json::to_string(&response).unwrap_or_default();
    if let Err(e) = store
        .insert_llm_trace(
            run_id,
            "reflect",
            None,
            commander_model,
            &data_json,
            &parsed_json,
            Some(&parsed_json),
            cost,
            0,
            0,
            0,
        )
        .await
    {
        eprintln!("[commander] warning: failed to store reflection trace: {e}");
    }

    // Store reflection record
    let recommendations_json = serde_json::to_string(&response.recommendations).unwrap_or_default();
    let model_insights_json = serde_json::to_string(&response.model_insights).unwrap_or_default();
    store
        .insert_reflection(
            run_id,
            Some(response.quality_score),
            &response.summary,
            &response.efficiency_analysis,
            &recommendations_json,
            Some(&model_insights_json),
            cost,
        )
        .await?;

    // Print to stderr
    print_reflection_summary(&response);

    Ok(cost)
}

/// Process a single successful task: store result, extract proposals, store rankings.
async fn process_task_result(
    store: &CommanderStore,
    gateway: &Arc<dyn ChatGateway>,
    commander_model: &str,
    run_id: i64,
    task_id: &str,
    decomposed_tasks: &[decompose::DecomposedTask],
    cost: i64,
    top_model: Option<&str>,
    synthesis_content: &str,
    session: &pipeline::PipelineSession,
    existing_titles: &[(String, ProposalStatus)],
    total_extract_cost: &mut i64,
    total_proposals: &mut usize,
) {
    // Update task result
    if let Err(e) = store
        .update_task_result(
            run_id,
            task_id,
            true,
            cost,
            top_model,
            Some(synthesis_content),
        )
        .await
    {
        eprintln!("[commander] warning: failed to update task result for {task_id}: {e}");
    }

    // Store model rankings from session
    for entity in &session.ranking.entities {
        if let Some(rank) = entity.rank {
            if let Err(e) = store
                .insert_model_ranking(run_id, task_id, &entity.id, rank as i64, entity.u_mean)
                .await
            {
                eprintln!(
                    "[commander] warning: failed to store ranking for {task_id}/{}: {e}",
                    entity.id
                );
            }
        }
    }

    // Find the original task prompt for extraction context
    let task_prompt = decomposed_tasks
        .iter()
        .find(|t| t.id == task_id)
        .map(|t| t.prompt.as_str())
        .unwrap_or("");

    // Extract proposals (with dedup context)
    match extract_proposals(
        gateway.as_ref(),
        commander_model,
        task_id,
        task_prompt,
        synthesis_content,
        existing_titles,
    )
    .await
    {
        Ok((proposals, extract_cost, raw_output)) => {
            *total_extract_cost += extract_cost;

            // Store extraction LLM trace
            let parsed_json = serde_json::to_string(&proposals).unwrap_or_default();
            if let Err(e) = store
                .insert_llm_trace(
                    run_id,
                    "extract",
                    Some(task_id),
                    commander_model,
                    task_prompt,
                    &raw_output,
                    Some(&parsed_json),
                    extract_cost,
                    0,
                    0,
                    0,
                )
                .await
            {
                eprintln!("[commander] warning: failed to store extract trace: {e}");
            }

            for p in &proposals {
                if let Err(e) = store
                    .insert_proposal(
                        run_id,
                        task_id,
                        &p.title,
                        &p.description,
                        p.category_enum(),
                        p.priority_enum(),
                        &p.affected_files_json(),
                        p.effort_enum(),
                    )
                    .await
                {
                    eprintln!(
                        "[commander] warning: failed to store proposal '{}': {e}",
                        p.title
                    );
                }
            }
            *total_proposals += proposals.len();
            eprintln!(
                "[commander]   {} — {} proposals extracted",
                task_id,
                proposals.len()
            );
        }
        Err(e) => {
            eprintln!("[commander]   {} — extraction failed: {}", task_id, e);
        }
    }
}

/// Result of a commander run.
#[derive(Debug, Clone)]
pub struct CommanderRunResult {
    pub run_id: i64,
    pub tasks_completed: i64,
    pub tasks_failed: i64,
    pub proposals_generated: i64,
    pub total_cost_nanodollars: i64,
}
