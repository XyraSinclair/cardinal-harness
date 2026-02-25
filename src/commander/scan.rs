//! Continuous codebase health scanning via cheap LLM inference.
//!
//! Unlike `command` (creative decomposition via expensive LLM), scanning is mechanical:
//! enumerate workspace crates, fan out cheap models per profile, collect findings.
//! Reuses `extract_proposals()` so findings feed into the same store, dashboard, and review.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::gateway::{
    Attribution, ChatGateway, ChatModel, ChatRequest, Message, NoopUsageSink, ProviderGateway,
};

use super::extract::{extract_proposals, ExtractError};
use super::store::{CommanderStore, ProposalStatus, RunStatus};

// =============================================================================
// Scan profiles
// =============================================================================

/// A scan profile defines what kind of issue to look for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScanProfile {
    Compilation,
    DeadCode,
    TestGaps,
    Patterns,
    Deps,
    Security,
}

impl ScanProfile {
    pub fn all() -> &'static [ScanProfile] {
        &[
            ScanProfile::Compilation,
            ScanProfile::DeadCode,
            ScanProfile::TestGaps,
            ScanProfile::Patterns,
            ScanProfile::Deps,
            ScanProfile::Security,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            ScanProfile::Compilation => "compilation",
            ScanProfile::DeadCode => "dead-code",
            ScanProfile::TestGaps => "test-gaps",
            ScanProfile::Patterns => "patterns",
            ScanProfile::Deps => "deps",
            ScanProfile::Security => "security",
        }
    }

    pub fn from_str(s: &str) -> Option<ScanProfile> {
        match s {
            "compilation" => Some(ScanProfile::Compilation),
            "dead-code" | "deadcode" => Some(ScanProfile::DeadCode),
            "test-gaps" | "testgaps" => Some(ScanProfile::TestGaps),
            "patterns" => Some(ScanProfile::Patterns),
            "deps" => Some(ScanProfile::Deps),
            "security" => Some(ScanProfile::Security),
            _ => None,
        }
    }

    pub fn system_prompt(&self) -> &'static str {
        match self {
            ScanProfile::Compilation => SYSTEM_COMPILATION,
            ScanProfile::DeadCode => SYSTEM_DEAD_CODE,
            ScanProfile::TestGaps => SYSTEM_TEST_GAPS,
            ScanProfile::Patterns => SYSTEM_PATTERNS,
            ScanProfile::Deps => SYSTEM_DEPS,
            ScanProfile::Security => SYSTEM_SECURITY,
        }
    }

    pub fn task_prompt_template(&self) -> &'static str {
        match self {
            ScanProfile::Compilation => TASK_COMPILATION,
            ScanProfile::DeadCode => TASK_DEAD_CODE,
            ScanProfile::TestGaps => TASK_TEST_GAPS,
            ScanProfile::Patterns => TASK_PATTERNS,
            ScanProfile::Deps => TASK_DEPS,
            ScanProfile::Security => TASK_SECURITY,
        }
    }
}

// --- System prompts ---

const SYSTEM_COMPILATION: &str = "\
You are a Rust compilation analyst. You identify code that will fail to compile, \
has type errors, missing imports, incorrect trait implementations, or uses deprecated APIs. \
Focus on concrete errors that a compiler would catch. Output your findings as actionable proposals.";

const SYSTEM_DEAD_CODE: &str = "\
You are a dead code analyst for Rust codebases. You identify unused functions, structs, \
enum variants, modules, imports, and feature flags that are never referenced. \
Only flag items that are truly unreachable — not items that are pub API surface. \
Output your findings as actionable proposals.";

const SYSTEM_TEST_GAPS: &str = "\
You are a test coverage analyst for Rust codebases. You identify functions, modules, \
and code paths that lack test coverage. Focus on critical logic, error handling paths, \
and edge cases that should be tested but aren't. \
Output your findings as actionable proposals.";

const SYSTEM_PATTERNS: &str = "\
You are a Rust code quality analyst. You identify anti-patterns, inconsistent conventions, \
unnecessary complexity, and opportunities for idiomatic improvements. \
Focus on real issues that affect maintainability, not style nitpicks. \
Output your findings as actionable proposals.";

const SYSTEM_DEPS: &str = "\
You are a dependency analyst for Rust workspaces. You identify duplicate dependencies, \
outdated versions, unnecessary dependencies, and dependency conflicts. \
Focus on Cargo.toml structure and dependency health. \
Output your findings as actionable proposals.";

const SYSTEM_SECURITY: &str = "\
You are a security analyst for Rust codebases. You identify potential vulnerabilities: \
unsafe blocks without safety comments, unchecked arithmetic, path traversal risks, \
injection vectors, improper error handling that leaks information, and missing input validation. \
Output your findings as actionable proposals.";

// --- Task prompt templates (use {crate_names} and {context} placeholders) ---

const TASK_COMPILATION: &str = "\
Analyze the following Rust crates for compilation issues, type errors, missing imports, \
and incorrect API usage.

## Crates
{crate_names}

## Source Code
{context}

List any concrete compilation issues you find. For each issue, specify the file, \
the problem, and how to fix it.";

const TASK_DEAD_CODE: &str = "\
Analyze the following Rust crates for dead code: unused functions, structs, enum variants, \
imports, and modules that are never referenced.

## Crates
{crate_names}

## Source Code
{context}

List any dead code you find. For each item, explain why you believe it's unused and \
whether it's safe to remove.";

const TASK_TEST_GAPS: &str = "\
Analyze the following Rust crates for test coverage gaps. Identify critical logic, \
error paths, and edge cases that lack tests.

## Crates
{crate_names}

## Source Code
{context}

List the most important missing tests. For each gap, describe what should be tested \
and why it matters.";

const TASK_PATTERNS: &str = "\
Analyze the following Rust crates for anti-patterns, inconsistent conventions, and \
code quality issues that affect maintainability.

## Crates
{crate_names}

## Source Code
{context}

List concrete issues. For each, explain the problem and suggest the idiomatic fix. \
Skip style-only nitpicks.";

const TASK_DEPS: &str = "\
Analyze the following Cargo.toml files for dependency issues: duplicates, outdated \
versions, unnecessary deps, and potential conflicts.

## Crates
{crate_names}

## Cargo.toml Files
{context}

List concrete dependency issues. For each, explain the problem and the recommended fix.";

const TASK_SECURITY: &str = "\
Analyze the following Rust crates for security issues: unsafe blocks, unchecked \
arithmetic, path traversal, injection vectors, and information leaks.

## Crates
{crate_names}

## Source Code
{context}

List concrete security concerns. For each, explain the risk level and the recommended fix.";

// =============================================================================
// Config & types
// =============================================================================

/// Configuration for a scan run.
#[derive(Debug, Clone)]
pub struct ScanConfig {
    /// Which profiles to run.
    pub profiles: Vec<ScanProfile>,
    /// Filter crates by name pattern (exact or glob).
    pub crate_filter: Vec<String>,
    /// Number of crates per LLM task group.
    pub group_size: usize,
    /// OpenRouter model ID for scan analysis.
    pub model: String,
    /// OpenRouter model ID for proposal extraction (can differ from scan model).
    pub extract_model: String,
    /// Path to SQLite store.
    pub store_path: PathBuf,
    /// Path to workspace root (for Cargo.toml discovery).
    pub workspace_root: PathBuf,
    /// Hard budget cap in nanodollars.
    pub budget_nanodollars: i64,
    /// Number of concurrent tasks.
    pub parallel: usize,
    /// Diff against previous scan run.
    pub diff: bool,
    /// Output directory for traces.
    pub output_dir: Option<PathBuf>,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            profiles: vec![ScanProfile::Patterns],
            crate_filter: Vec::new(),
            group_size: 5,
            model: "google/gemini-3.1-pro-preview".into(),
            extract_model: "google/gemini-3.1-pro-preview".into(),
            store_path: CommanderStore::default_path(),
            workspace_root: PathBuf::from("."),
            budget_nanodollars: 2_000_000_000, // $2.00
            parallel: 4,
            diff: false,
            output_dir: None,
        }
    }
}

/// Lightweight crate info — just name and path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanCrateInfo {
    pub name: String,
    pub path: PathBuf,
}

/// A single scan task to send to the LLM.
#[derive(Debug, Clone)]
pub struct ScanTask {
    pub id: String,
    pub profile: ScanProfile,
    pub crate_names: Vec<String>,
    pub system_prompt: String,
    pub user_prompt: String,
}

/// Result of a single scan task.
#[derive(Debug)]
struct ScanTaskResult {
    task_id: String,
    profile: ScanProfile,
    scan_cost: i64,
    extract_cost: i64,
    proposals_count: usize,
    latency_ms: u64,
}

/// Result of the full scan run.
#[derive(Debug, Clone)]
pub struct ScanRunResult {
    pub run_id: i64,
    pub tasks_completed: i64,
    pub tasks_failed: i64,
    pub proposals_generated: i64,
    pub total_cost_nanodollars: i64,
    pub new_findings: Option<i64>,
}

#[derive(Debug, thiserror::Error)]
pub enum ScanError {
    #[error("Store error: {0}")]
    Store(#[from] super::store::StoreError),
    #[error("Gateway error: {0}")]
    Gateway(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parse error: {0}")]
    TomlParse(String),
    #[error("Budget exceeded: estimated ${estimated:.2} exceeds budget ${budget:.2}")]
    BudgetExceeded { estimated: f64, budget: f64 },
    #[error("No crates found in workspace")]
    NoCrates,
    #[error("Extract error: {0}")]
    Extract(#[from] ExtractError),
    #[error("LLM call failed: {0}")]
    LlmFailed(#[from] crate::gateway::error::ProviderError),
}

// =============================================================================
// Workspace enumeration
// =============================================================================

/// Parse the workspace Cargo.toml and enumerate all member crates.
/// Returns (name, path) pairs. Only needs name + path, no dependency graph.
pub fn enumerate_crates(workspace_root: &Path) -> Result<Vec<ScanCrateInfo>, ScanError> {
    let cargo_path = workspace_root.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_path).map_err(|e| ScanError::Io(e))?;

    let doc: toml::Value = content
        .parse()
        .map_err(|e: toml::de::Error| ScanError::TomlParse(e.to_string()))?;

    let members = doc
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
        .ok_or_else(|| ScanError::TomlParse("no workspace.members found".into()))?;

    let mut crates = Vec::new();

    for member in members {
        let pattern = match member.as_str() {
            Some(s) => s,
            None => continue,
        };

        // Expand glob patterns
        let full_pattern = workspace_root.join(pattern);
        let pattern_str = full_pattern.to_string_lossy().to_string();

        let paths = glob::glob(&pattern_str)
            .map_err(|e| ScanError::TomlParse(format!("bad glob pattern '{pattern}': {e}")))?;

        for entry in paths {
            let path = match entry {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Must have a Cargo.toml to be a crate
            let crate_cargo = path.join("Cargo.toml");
            if !crate_cargo.exists() {
                continue;
            }

            // Read crate name from its Cargo.toml
            let name = read_crate_name(&crate_cargo).unwrap_or_else(|| {
                path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".into())
            });

            crates.push(ScanCrateInfo { name, path });
        }
    }

    crates.sort_by(|a, b| a.name.cmp(&b.name));
    crates.dedup_by(|a, b| a.name == b.name);

    Ok(crates)
}

/// Read [package].name from a crate's Cargo.toml.
fn read_crate_name(cargo_path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(cargo_path).ok()?;
    let doc: toml::Value = content.parse().ok()?;
    doc.get("package")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .map(String::from)
}

/// Filter crates by name patterns. Supports exact match and glob patterns.
pub fn filter_crates(crates: &[ScanCrateInfo], patterns: &[String]) -> Vec<ScanCrateInfo> {
    if patterns.is_empty() {
        return crates.to_vec();
    }

    crates
        .iter()
        .filter(|c| {
            patterns.iter().any(|pat| {
                if pat.contains('*') || pat.contains('?') {
                    glob::Pattern::new(pat)
                        .map(|p| p.matches(&c.name))
                        .unwrap_or(false)
                } else {
                    c.name == *pat
                }
            })
        })
        .cloned()
        .collect()
}

// =============================================================================
// Task generation
// =============================================================================

/// Group crates into scan tasks with loaded context files.
pub fn generate_scan_tasks(
    crates: &[ScanCrateInfo],
    profile: ScanProfile,
    group_size: usize,
    workspace_root: &Path,
) -> Vec<ScanTask> {
    let group_size = group_size.max(1);
    let mut tasks = Vec::new();

    for (chunk_idx, chunk) in crates.chunks(group_size).enumerate() {
        let crate_names: Vec<String> = chunk.iter().map(|c| c.name.clone()).collect();
        let task_id = format!("scan-{}-{}", profile.name(), chunk_idx);

        // Load context for this chunk
        let context = load_chunk_context(chunk, profile, workspace_root);

        let user_prompt = profile
            .task_prompt_template()
            .replace("{crate_names}", &crate_names.join(", "))
            .replace("{context}", &context);

        tasks.push(ScanTask {
            id: task_id,
            profile,
            crate_names,
            system_prompt: profile.system_prompt().to_string(),
            user_prompt,
        });
    }

    tasks
}

/// Load context files for a chunk of crates.
/// - Deps profile: only Cargo.toml
/// - Other profiles: Cargo.toml + src/lib.rs + src/main.rs (if they exist)
/// Cap total context at ~128KB.
fn load_chunk_context(
    chunk: &[ScanCrateInfo],
    profile: ScanProfile,
    _workspace_root: &Path,
) -> String {
    const MAX_CONTEXT_BYTES: usize = 128 * 1024;
    let mut context = String::new();
    let mut total_bytes: usize = 0;

    for crate_info in chunk {
        // Always include Cargo.toml
        if let Some(content) = read_file_capped(
            &crate_info.path.join("Cargo.toml"),
            MAX_CONTEXT_BYTES - total_bytes,
        ) {
            context.push_str(&format!(
                "### {}/Cargo.toml\n```toml\n{}\n```\n\n",
                crate_info.name, content
            ));
            total_bytes = context.len();
        }

        if total_bytes >= MAX_CONTEXT_BYTES {
            break;
        }

        // Deps profile only needs Cargo.toml
        if profile == ScanProfile::Deps {
            continue;
        }

        // Try lib.rs
        let lib_path = crate_info.path.join("src/lib.rs");
        if let Some(content) = read_file_capped(&lib_path, MAX_CONTEXT_BYTES - total_bytes) {
            context.push_str(&format!(
                "### {}/src/lib.rs\n```rust\n{}\n```\n\n",
                crate_info.name, content
            ));
            total_bytes = context.len();
        }

        if total_bytes >= MAX_CONTEXT_BYTES {
            break;
        }

        // Try main.rs
        let main_path = crate_info.path.join("src/main.rs");
        if let Some(content) = read_file_capped(&main_path, MAX_CONTEXT_BYTES - total_bytes) {
            context.push_str(&format!(
                "### {}/src/main.rs\n```rust\n{}\n```\n\n",
                crate_info.name, content
            ));
            total_bytes = context.len();
        }

        if total_bytes >= MAX_CONTEXT_BYTES {
            break;
        }
    }

    context
}

/// Read a file up to max_bytes. Returns None if file doesn't exist.
fn read_file_capped(path: &Path, max_bytes: usize) -> Option<String> {
    if !path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(path).ok()?;
    if content.len() > max_bytes {
        Some(content[..max_bytes].to_string())
    } else {
        Some(content)
    }
}

// =============================================================================
// LLM execution
// =============================================================================

/// Run a single scan task: LLM call + proposal extraction.
async fn run_single_scan_task(
    gateway: &dyn ChatGateway,
    scan_model: &str,
    extract_model: &str,
    task: &ScanTask,
    store: &CommanderStore,
    run_id: i64,
    existing_titles: &[(String, ProposalStatus)],
) -> Result<ScanTaskResult, ScanError> {
    let start = Instant::now();

    // 1. Direct LLM call (no pipeline/ranking)
    let req = ChatRequest::new(
        ChatModel::openrouter(scan_model),
        vec![
            Message::system(&task.system_prompt),
            Message::user(&task.user_prompt),
        ],
        Attribution::new("commander::scan"),
    )
    .temperature(0.2)
    .max_tokens(4096);

    let resp = gateway.chat(req).await?;
    let scan_cost = resp.cost_nanodollars;
    let raw_output = resp.content.clone();

    // Store LLM trace
    if let Err(e) = store
        .insert_llm_trace(
            run_id,
            "scan",
            Some(&task.id),
            scan_model,
            &task.user_prompt,
            &raw_output,
            None,
            scan_cost,
            resp.input_tokens.into(),
            resp.output_tokens.into(),
            start.elapsed().as_millis() as i64,
        )
        .await
    {
        eprintln!(
            "[scan] warning: failed to store scan trace for {}: {e}",
            task.id
        );
    }

    // 2. Extract proposals from the raw output
    let task_prompt_summary = format!(
        "Scan profile: {} | Crates: {}",
        task.profile.name(),
        task.crate_names.join(", ")
    );

    let (proposals, extract_cost, _extract_raw) = extract_proposals(
        gateway,
        extract_model,
        &task.id,
        &task_prompt_summary,
        &raw_output,
        existing_titles,
    )
    .await?;

    // Store extraction trace
    let parsed_json = serde_json::to_string(&proposals).unwrap_or_default();
    if let Err(e) = store
        .insert_llm_trace(
            run_id,
            "extract",
            Some(&task.id),
            extract_model,
            &task_prompt_summary,
            &_extract_raw,
            Some(&parsed_json),
            extract_cost,
            0,
            0,
            0,
        )
        .await
    {
        eprintln!(
            "[scan] warning: failed to store extract trace for {}: {e}",
            task.id
        );
    }

    // 3. Store proposals
    let mut proposals_count = 0;
    for p in &proposals {
        if let Err(e) = store
            .insert_proposal(
                run_id,
                &task.id,
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
                "[scan] warning: failed to store proposal '{}': {e}",
                p.title
            );
        } else {
            proposals_count += 1;
        }
    }

    // 4. Store task record
    if let Err(e) = store
        .insert_task(
            run_id,
            0, // task_index not meaningful for scan
            &task.id,
            &task_prompt_summary,
            Some(&task.system_prompt),
            "[]", // no context globs
            &format!("scan:{}", task.profile.name()),
        )
        .await
    {
        eprintln!("[scan] warning: failed to store task {}: {e}", task.id);
    }

    if let Err(e) = store
        .update_task_result(
            run_id,
            &task.id,
            true,
            scan_cost + extract_cost,
            Some(scan_model),
            Some(&raw_output),
        )
        .await
    {
        eprintln!(
            "[scan] warning: failed to update task result {}: {e}",
            task.id
        );
    }

    let latency_ms = start.elapsed().as_millis() as u64;

    Ok(ScanTaskResult {
        task_id: task.id.clone(),
        profile: task.profile,
        scan_cost,
        extract_cost,
        proposals_count,
        latency_ms,
    })
}

// =============================================================================
// Orchestrator
// =============================================================================

/// Run a full scan: enumerate → filter → generate tasks → execute → extract → diff.
pub async fn run_scan(config: ScanConfig) -> Result<ScanRunResult, ScanError> {
    eprintln!(
        "[scan] profiles: {:?}",
        config.profiles.iter().map(|p| p.name()).collect::<Vec<_>>()
    );
    eprintln!(
        "[scan] model: {}, budget: ${:.2}, group_size: {}, parallel: {}",
        config.model,
        config.budget_nanodollars as f64 / 1e9,
        config.group_size,
        config.parallel,
    );

    // 1. Enumerate workspace crates
    let all_crates = enumerate_crates(&config.workspace_root)?;
    if all_crates.is_empty() {
        return Err(ScanError::NoCrates);
    }
    eprintln!("[scan] found {} crates in workspace", all_crates.len());

    // 2. Filter
    let crates = filter_crates(&all_crates, &config.crate_filter);
    if crates.is_empty() {
        return Err(ScanError::NoCrates);
    }
    if !config.crate_filter.is_empty() {
        eprintln!("[scan] filtered to {} crates", crates.len());
    }

    // 3. Generate tasks across all profiles
    let mut all_tasks: Vec<ScanTask> = Vec::new();
    for profile in &config.profiles {
        let tasks =
            generate_scan_tasks(&crates, *profile, config.group_size, &config.workspace_root);
        all_tasks.extend(tasks);
    }
    let n_tasks = all_tasks.len();
    eprintln!("[scan] generated {} tasks", n_tasks);

    // 4. Budget check (~$0.03/task scan + $0.02/task extraction)
    let est_per_task: i64 = 50_000_000; // $0.05
    let total_estimate = (n_tasks as i64) * est_per_task;
    let est_dollars = total_estimate as f64 / 1e9;
    let budget_dollars = config.budget_nanodollars as f64 / 1e9;

    eprintln!(
        "[scan] estimated cost: ${:.2} (budget: ${:.2})",
        est_dollars, budget_dollars
    );

    if total_estimate > config.budget_nanodollars {
        return Err(ScanError::BudgetExceeded {
            estimated: est_dollars,
            budget: budget_dollars,
        });
    }

    // 5. Set up store and gateway
    let store = CommanderStore::new(&config.store_path)?;
    let gateway: Arc<dyn ChatGateway> = Arc::new(
        ProviderGateway::from_env(Arc::new(NoopUsageSink))
            .map_err(|e| ScanError::Gateway(e.to_string()))?,
    );

    // Build directive string for the run
    let profile_names: Vec<&str> = config.profiles.iter().map(|p| p.name()).collect();
    let directive = format!("scan:{}", profile_names.join(","));

    let run_id = store
        .create_run(&directive, &config.model, "scan", config.budget_nanodollars)
        .await?;

    eprintln!("[scan] run #{}", run_id);

    // 6. Fetch existing titles for dedup
    let existing_titles = store.recent_proposal_titles(50).await.unwrap_or_default();

    // 7. Execute tasks with concurrency
    let mut tasks_completed: i64 = 0;
    let mut tasks_failed: i64 = 0;
    let mut total_proposals: i64 = 0;
    let mut total_scan_cost: i64 = 0;
    let mut total_extract_cost: i64 = 0;

    let semaphore = Arc::new(tokio::sync::Semaphore::new(config.parallel));

    // Spawn all tasks
    let mut handles = Vec::new();
    for task in all_tasks {
        let sem = semaphore.clone();
        let gw = gateway.clone();
        let st = store.clone();
        let scan_model = config.model.clone();
        let extract_model = config.extract_model.clone();
        let titles = existing_titles.clone();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore closed");
            run_single_scan_task(
                gw.as_ref(),
                &scan_model,
                &extract_model,
                &task,
                &st,
                run_id,
                &titles,
            )
            .await
        });
        handles.push(handle);
    }

    // Collect results
    for handle in handles {
        match handle.await {
            Ok(Ok(result)) => {
                tasks_completed += 1;
                total_scan_cost += result.scan_cost;
                total_extract_cost += result.extract_cost;
                total_proposals += result.proposals_count as i64;
                eprintln!(
                    "[scan]   {} ({}) — {} proposals, ${:.4}, {}ms",
                    result.task_id,
                    result.profile.name(),
                    result.proposals_count,
                    (result.scan_cost + result.extract_cost) as f64 / 1e9,
                    result.latency_ms,
                );
            }
            Ok(Err(e)) => {
                tasks_failed += 1;
                eprintln!("[scan]   task failed: {e}");
            }
            Err(e) => {
                tasks_failed += 1;
                eprintln!("[scan]   task panicked: {e}");
            }
        }
    }

    // 8. Update run costs
    let total_cost = total_scan_cost + total_extract_cost;
    store
        .update_run_costs(
            run_id,
            0,
            total_scan_cost,
            total_extract_cost,
            tasks_completed,
            tasks_failed,
        )
        .await?;

    let final_status = if tasks_failed > 0 && tasks_completed == 0 {
        RunStatus::Failed
    } else {
        RunStatus::Completed
    };
    store.update_run_status(run_id, final_status).await?;

    // 9. Diff against previous run (if requested)
    let new_findings = if config.diff {
        match diff_with_previous(&store, run_id, &profile_names).await {
            Ok(count) => {
                eprintln!("[scan] diff: {} new findings vs previous scan", count);
                Some(count)
            }
            Err(e) => {
                eprintln!("[scan] diff failed (non-fatal): {e}");
                None
            }
        }
    } else {
        None
    };

    // 10. Print summary
    eprintln!();
    eprintln!("=== Scan Complete ===");
    eprintln!(
        "Run #{}: {} tasks completed, {} failed",
        run_id, tasks_completed, tasks_failed
    );
    eprintln!("{} proposals generated", total_proposals);
    if let Some(new) = new_findings {
        eprintln!("{} new findings (vs previous scan)", new);
    }
    eprintln!(
        "Cost: ${:.4} (scan: ${:.4}, extract: ${:.4})",
        total_cost as f64 / 1e9,
        total_scan_cost as f64 / 1e9,
        total_extract_cost as f64 / 1e9,
    );
    eprintln!();
    eprintln!("Review findings:");
    eprintln!(
        "  cardinal review --list --store {}",
        config.store_path.display()
    );
    eprintln!(
        "  cardinal dashboard --store {}",
        config.store_path.display()
    );

    Ok(ScanRunResult {
        run_id,
        tasks_completed,
        tasks_failed,
        proposals_generated: total_proposals,
        total_cost_nanodollars: total_cost,
        new_findings,
    })
}

// =============================================================================
// Diff
// =============================================================================

/// Find the most recent completed scan run matching the same profiles.
async fn find_previous_scan_run(
    store: &CommanderStore,
    current_run_id: i64,
    profile_names: &[&str],
) -> Result<Option<i64>, ScanError> {
    let runs = store.list_runs().await?;
    let target_directive = format!("scan:{}", profile_names.join(","));

    for run in &runs {
        if run.id >= current_run_id {
            continue;
        }
        if run.status != RunStatus::Completed {
            continue;
        }
        if run.directive == target_directive {
            return Ok(Some(run.id));
        }
    }

    Ok(None)
}

/// Compare proposals from current run against previous run.
/// Returns count of genuinely new findings.
async fn diff_with_previous(
    store: &CommanderStore,
    current_run_id: i64,
    profile_names: &[&str],
) -> Result<i64, ScanError> {
    let prev_id = match find_previous_scan_run(store, current_run_id, profile_names).await? {
        Some(id) => id,
        None => {
            eprintln!("[scan] no previous scan run found for diff");
            return Ok(-1);
        }
    };

    eprintln!("[scan] diffing against run #{}", prev_id);

    // Get all proposals for both runs
    let all_proposals = store.list_proposals(None).await?;

    let current_titles: Vec<&str> = all_proposals
        .iter()
        .filter(|p| p.run_id == current_run_id)
        .map(|p| p.title.as_str())
        .collect();

    let prev_titles: Vec<String> = all_proposals
        .iter()
        .filter(|p| p.run_id == prev_id)
        .map(|p| normalize_title(&p.title))
        .collect();

    let mut new_count: i64 = 0;
    for title in &current_titles {
        let normalized = normalize_title(title);
        if !prev_titles
            .iter()
            .any(|pt| title_similarity(&normalized, pt) > 0.8)
        {
            new_count += 1;
        }
    }

    Ok(new_count)
}

/// Normalize a title for comparison: lowercase, strip punctuation, collapse whitespace.
fn normalize_title(title: &str) -> String {
    title
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Simple word-overlap similarity (Jaccard).
fn title_similarity(a: &str, b: &str) -> f64 {
    let a_words: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let b_words: std::collections::HashSet<&str> = b.split_whitespace().collect();

    if a_words.is_empty() && b_words.is_empty() {
        return 1.0;
    }

    let intersection = a_words.intersection(&b_words).count();
    let union = a_words.union(&b_words).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f64 / union as f64
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_prompts_non_empty() {
        for profile in ScanProfile::all() {
            assert!(
                !profile.system_prompt().is_empty(),
                "system prompt empty for {:?}",
                profile
            );
            assert!(
                !profile.task_prompt_template().is_empty(),
                "task prompt empty for {:?}",
                profile
            );
            assert!(
                profile.task_prompt_template().contains("{crate_names}"),
                "task prompt missing {{crate_names}} for {:?}",
                profile
            );
            assert!(
                profile.task_prompt_template().contains("{context}"),
                "task prompt missing {{context}} for {:?}",
                profile
            );
        }
    }

    #[test]
    fn test_profile_roundtrip() {
        for profile in ScanProfile::all() {
            let name = profile.name();
            let parsed = ScanProfile::from_str(name);
            assert_eq!(parsed, Some(*profile), "roundtrip failed for {name}");
        }
    }

    #[test]
    fn test_filter_crates_exact() {
        let crates = vec![
            ScanCrateInfo {
                name: "slate".into(),
                path: PathBuf::from("crates/slate"),
            },
            ScanCrateInfo {
                name: "spiel".into(),
                path: PathBuf::from("crates/spiel"),
            },
            ScanCrateInfo {
                name: "shore".into(),
                path: PathBuf::from("crates/shore"),
            },
        ];

        let filtered = filter_crates(&crates, &["slate".into()]);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "slate");
    }

    #[test]
    fn test_filter_crates_glob() {
        let crates = vec![
            ScanCrateInfo {
                name: "slate".into(),
                path: PathBuf::from("crates/slate"),
            },
            ScanCrateInfo {
                name: "slate-map".into(),
                path: PathBuf::from("crates/slate-map"),
            },
            ScanCrateInfo {
                name: "spiel".into(),
                path: PathBuf::from("crates/spiel"),
            },
        ];

        let filtered = filter_crates(&crates, &["slate*".into()]);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|c| c.name.starts_with("slate")));
    }

    #[test]
    fn test_filter_crates_empty_returns_all() {
        let crates = vec![
            ScanCrateInfo {
                name: "a".into(),
                path: PathBuf::from("a"),
            },
            ScanCrateInfo {
                name: "b".into(),
                path: PathBuf::from("b"),
            },
        ];

        let filtered = filter_crates(&crates, &[]);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_generate_scan_tasks_grouping() {
        let crates: Vec<ScanCrateInfo> = (0..12)
            .map(|i| ScanCrateInfo {
                name: format!("crate-{i}"),
                path: PathBuf::from(format!("/tmp/crate-{i}")),
            })
            .collect();

        let tasks = generate_scan_tasks(&crates, ScanProfile::Patterns, 5, Path::new("/tmp"));
        // 12 crates / group_size=5 → ceil(12/5) = 3 tasks
        assert_eq!(tasks.len(), 3);
        assert_eq!(tasks[0].crate_names.len(), 5);
        assert_eq!(tasks[1].crate_names.len(), 5);
        assert_eq!(tasks[2].crate_names.len(), 2);
    }

    #[test]
    fn test_generate_scan_tasks_single_group() {
        let crates = vec![
            ScanCrateInfo {
                name: "a".into(),
                path: PathBuf::from("/tmp/a"),
            },
            ScanCrateInfo {
                name: "b".into(),
                path: PathBuf::from("/tmp/b"),
            },
        ];

        let tasks = generate_scan_tasks(&crates, ScanProfile::Security, 10, Path::new("/tmp"));
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].crate_names, vec!["a", "b"]);
    }

    #[test]
    fn test_normalize_title() {
        assert_eq!(
            normalize_title("Fix: unused imports!"),
            "fix unused imports"
        );
        assert_eq!(normalize_title("  Hello   World  "), "hello world");
    }

    #[test]
    fn test_title_similarity() {
        assert!(title_similarity("fix unused imports", "fix unused imports") > 0.99);
        assert!(
            title_similarity(
                "fix unused imports in slate",
                "remove unused imports from slate"
            ) > 0.3
        );
        assert!(title_similarity("completely different", "no overlap whatsoever") < 0.2);
    }

    #[test]
    fn test_context_loading_deps_only_cargo() {
        // Deps profile should only include Cargo.toml, not source files
        // We test this by verifying the template doesn't include source
        let dir = tempfile::tempdir().unwrap();
        let crate_dir = dir.path().join("my-crate");
        std::fs::create_dir_all(crate_dir.join("src")).unwrap();
        std::fs::write(
            crate_dir.join("Cargo.toml"),
            "[package]\nname = \"my-crate\"\n",
        )
        .unwrap();
        std::fs::write(crate_dir.join("src/lib.rs"), "pub fn hello() {}").unwrap();

        let crate_info = ScanCrateInfo {
            name: "my-crate".into(),
            path: crate_dir,
        };

        let context = load_chunk_context(&[crate_info.clone()], ScanProfile::Deps, dir.path());
        assert!(context.contains("Cargo.toml"));
        assert!(!context.contains("lib.rs"));

        // Non-deps profile should include both
        let context2 = load_chunk_context(&[crate_info], ScanProfile::Patterns, dir.path());
        assert!(context2.contains("Cargo.toml"));
        assert!(context2.contains("lib.rs"));
    }
}
