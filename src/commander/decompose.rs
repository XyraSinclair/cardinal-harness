//! Decompose a high-level directive into flywheel tasks via LLM.

use serde::{Deserialize, Serialize};

use crate::gateway::{Attribution, ChatGateway, ChatModel, ChatRequest, Message};
use crate::pipeline::{
    expand_context_globs, load_context_files, FlywheelManifest, FlywheelTask, ModelPreset,
};

use super::briefing::BriefingResponse;
use super::extract::extract_json;

// =============================================================================
// Types
// =============================================================================

/// A decomposed task from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposedTask {
    pub id: String,
    pub prompt: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub context_globs: Vec<String>,
    #[serde(default)]
    pub rationale: String,
    #[serde(default)]
    pub success_criterion: Option<String>,
}

/// Full decomposition response from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionResponse {
    pub tasks: Vec<DecomposedTask>,
    #[serde(default)]
    pub context_summary: String,
    #[serde(default)]
    pub estimated_difficulty: String,
}

#[derive(Debug, thiserror::Error)]
pub enum DecomposeError {
    #[error("LLM call failed: {0}")]
    LlmFailed(#[from] crate::gateway::error::ProviderError),
    #[error("JSON parse failed: {0}")]
    JsonParse(String),
    #[error("Decomposition produced no tasks")]
    NoTasks,
    #[error("Context loading failed: {0}")]
    ContextLoad(#[from] std::io::Error),
}

// =============================================================================
// System prompt
// =============================================================================

const DECOMPOSE_SYSTEM_PROMPT: &str = "\
You are a strategic code analysis planner. You receive a high-level improvement directive \
for a codebase, along with context about the project structure and recent changes. Your job \
is to decompose this directive into 3-8 specific, self-contained investigation tasks that \
can each be sent to a multi-model analysis pipeline (generate → rank → synthesize).

Requirements:
- Produce 3-8 tasks. Each must be self-contained and specific enough that a model can \
  produce a useful analysis without prior context from other tasks.
- Use descriptive kebab-case IDs (e.g. \"audit-slate-concurrency\", \"trace-spiel-durability-gaps\").
- Each prompt should be 100-300 words, giving the model enough context to produce deep analysis.
- Include context_globs pointing to relevant source files (glob patterns like \"crates/slate/src/**/*.rs\").
- Include a rationale explaining why this task matters for the directive.
- At least one task should focus on failure modes / what could go wrong.
- At least one task should focus on concrete interventions / what to change.
- Favor depth over breadth — fewer, deeper tasks beat many shallow ones.

PARSIMONY PRESSURE: Each task costs real money. Only create a task if the expected insight \
value exceeds its cost. Fewer, deeper tasks always beat more, shallow ones. If the directive \
can be covered in 3 tasks, do not create 6.

SCOPE DEFENSE: Every task MUST trace back to the directive. If you cannot explain in one \
sentence how a task serves the directive, do not include it. Do not add tangential \
\"nice to have\" tasks.

If an Operational Briefing section is provided below, acknowledge any overlap warnings and \
adjust your task selection accordingly. Do not duplicate work that has already been done.

Respond with JSON only:
{
  \"tasks\": [
    {
      \"id\": \"kebab-case-id\",
      \"prompt\": \"Detailed investigation prompt...\",
      \"system_prompt\": null,
      \"context_globs\": [\"crates/relevant/src/**/*.rs\"],
      \"rationale\": \"Why this task matters...\",
      \"success_criterion\": \"What specific finding would make this task worth its cost\"
    }
  ],
  \"context_summary\": \"Brief summary of what the codebase context told you\",
  \"estimated_difficulty\": \"easy|moderate|hard\"
}";

// =============================================================================
// Decomposition
// =============================================================================

/// Decompose a directive into tasks using an LLM.
///
/// Returns the decomposition response, the LLM cost in nanodollars, and the raw LLM output.
pub async fn decompose_directive(
    gateway: &dyn ChatGateway,
    model: &str,
    directive: &str,
    codebase_context: &str,
    briefing: Option<&BriefingResponse>,
) -> Result<(DecompositionResponse, i64, String), DecomposeError> {
    let mut user_prompt = format!(
        "## Directive\n\n{}\n\n## Codebase Context\n\n{}",
        directive, codebase_context
    );

    // Inject briefing context if available
    if let Some(briefing) = briefing {
        let briefing_section = super::briefing::format_briefing_for_prompt(briefing);
        user_prompt.push_str(&format!(
            "\n\n## Operational Briefing\n\n{briefing_section}"
        ));
    }

    let messages = vec![
        Message::system(DECOMPOSE_SYSTEM_PROMPT),
        Message::user(user_prompt),
    ];

    let req = ChatRequest::new(
        ChatModel::openrouter(model),
        messages,
        Attribution::new("commander::decompose"),
    )
    .temperature(0.3)
    .max_tokens(8192)
    .json();

    let resp = gateway.chat(req).await?;
    let cost = resp.cost_nanodollars;
    let raw_output = resp.content.clone();

    let json_str = extract_json(&resp.content);
    let mut parsed: DecompositionResponse = serde_json::from_str(json_str).map_err(|e| {
        let preview: String = resp.content.chars().take(500).collect();
        DecomposeError::JsonParse(format!(
            "failed to parse decomposition: {} — raw: {}",
            e, preview
        ))
    })?;

    if parsed.tasks.is_empty() {
        return Err(DecomposeError::NoTasks);
    }

    // Clamp to 8 tasks max
    parsed.tasks.truncate(8);

    Ok((parsed, cost, raw_output))
}

/// Gather codebase context for decomposition.
///
/// Reads CLAUDE.md, git log, and BASIN_PRIMER.md (first 200 lines) if available.
pub fn gather_codebase_context() -> String {
    let mut parts = Vec::new();

    // CLAUDE.md
    if let Ok(content) = std::fs::read_to_string("CLAUDE.md") {
        let truncated: String = content.lines().take(150).collect::<Vec<_>>().join("\n");
        parts.push(format!(
            "### CLAUDE.md (first 150 lines)\n```\n{truncated}\n```"
        ));
    }

    // git log
    if let Ok(output) = std::process::Command::new("git")
        .args(["log", "--oneline", "-20"])
        .output()
    {
        if output.status.success() {
            let log = String::from_utf8_lossy(&output.stdout);
            parts.push(format!("### Recent commits\n```\n{log}\n```"));
        }
    }

    // BASIN_PRIMER.md
    if let Ok(content) = std::fs::read_to_string("docs/design/BASIN_PRIMER.md") {
        let truncated: String = content.lines().take(200).collect::<Vec<_>>().join("\n");
        parts.push(format!(
            "### BASIN_PRIMER.md (first 200 lines)\n```\n{truncated}\n```"
        ));
    }

    if parts.is_empty() {
        "No codebase context files found.".to_string()
    } else {
        parts.join("\n\n")
    }
}

/// Sensitive path patterns that must never be sent to remote models.
const DENIED_PATTERNS: &[&str] = &[
    ".env",
    ".git/",
    ".ssh/",
    ".gnupg/",
    ".aws/",
    "target/",
    "node_modules/",
    "credentials",
    "secret",
    ".key",
    ".pem",
    ".p12",
    ".pfx",
    "id_rsa",
    "id_ed25519",
];

/// Sanitize LLM-generated globs: reject absolute paths, `..` traversal,
/// and anything matching the denylist.
fn sanitize_globs(globs: &[String]) -> Vec<String> {
    globs
        .iter()
        .filter(|g| {
            let g = g.trim();
            // Reject absolute paths
            if g.starts_with('/') || g.starts_with('~') {
                eprintln!("[commander] rejected absolute glob: {g}");
                return false;
            }
            // Reject path traversal
            if g.contains("..") {
                eprintln!("[commander] rejected traversal glob: {g}");
                return false;
            }
            // Reject sensitive patterns
            let lower = g.to_lowercase();
            for denied in DENIED_PATTERNS {
                if lower.contains(denied) {
                    eprintln!("[commander] rejected sensitive glob: {g}");
                    return false;
                }
            }
            true
        })
        .cloned()
        .collect()
}

/// Verify that expanded file paths stay within the project root (after symlink resolution).
fn filter_safe_paths(paths: Vec<String>) -> Vec<String> {
    let root = std::env::current_dir().ok();
    paths
        .into_iter()
        .filter(|p| {
            let path = std::path::Path::new(p);
            // Canonicalize to resolve symlinks
            if let Ok(canonical) = path.canonicalize() {
                if let Some(ref root) = root {
                    if let Ok(canonical_root) = root.canonicalize() {
                        if !canonical.starts_with(&canonical_root) {
                            eprintln!(
                                "[commander] rejected out-of-root path: {}",
                                canonical.display()
                            );
                            return false;
                        }
                    }
                }
            }
            // Also reject sensitive files in expanded results
            let lower = p.to_lowercase();
            for denied in DENIED_PATTERNS {
                if lower.contains(denied) {
                    return false;
                }
            }
            true
        })
        .collect()
}

/// Build a FlywheelManifest from decomposed tasks, resolving context globs.
///
/// Each task's context_globs are sanitized (no absolute paths, no `..`,
/// no sensitive files), expanded, and loaded, capped at `max_files_per_task`.
pub fn build_manifest(
    tasks: &[DecomposedTask],
    preset: ModelPreset,
    max_files_per_task: usize,
) -> Result<FlywheelManifest, DecomposeError> {
    let mut flywheel_tasks = Vec::new();

    for task in tasks {
        // C1: Sanitize LLM-generated globs before expansion
        let safe_globs = sanitize_globs(&task.context_globs);

        // Expand and load context files
        let expanded = expand_context_globs(&safe_globs)?;

        // C1: Verify expanded paths stay within project root
        let safe_paths = filter_safe_paths(expanded);
        let capped: Vec<String> = safe_paths.into_iter().take(max_files_per_task).collect();

        // H5: Propagate context loading errors instead of swallowing
        let context_files = load_context_files(&capped)?;

        flywheel_tasks.push(FlywheelTask {
            id: task.id.clone(),
            prompt: task.prompt.clone(),
            system_prompt: task.system_prompt.clone(),
            extra_context_files: context_files,
            models: None,
            synthesis_model: None,
        });
    }

    Ok(FlywheelManifest {
        tasks: flywheel_tasks,
        preset: Some(preset),
        context_files: Vec::new(),
        attributes: None, // use defaults (extended)
        synthesis_model: None,
        rank_config: None,
        max_context_tokens: Some(32_000),
    })
}

/// Estimate the cost of running a flywheel with a given preset.
///
/// Returns estimated cost in nanodollars.
pub fn estimate_flywheel_cost(n_tasks: usize, preset: ModelPreset) -> i64 {
    let per_task_dollars = match preset {
        ModelPreset::Frontier => 2.0,
        ModelPreset::Balanced => 0.80,
        ModelPreset::Fast => 0.20,
    };
    let total = per_task_dollars * n_tasks as f64;
    (total * 1_000_000_000.0) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposition_response_parse() {
        let json = r#"{
            "tasks": [
                {
                    "id": "audit-concurrency",
                    "prompt": "Analyze concurrency patterns in Slate",
                    "context_globs": ["crates/slate/src/**/*.rs"],
                    "rationale": "Known DashMap issues"
                }
            ],
            "context_summary": "Rust storage platform",
            "estimated_difficulty": "moderate"
        }"#;
        let resp: DecompositionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.tasks.len(), 1);
        assert_eq!(resp.tasks[0].id, "audit-concurrency");
    }

    #[test]
    fn test_estimate_flywheel_cost() {
        let cost = estimate_flywheel_cost(5, ModelPreset::Fast);
        assert_eq!(cost, 1_000_000_000); // $1.00

        let cost = estimate_flywheel_cost(5, ModelPreset::Frontier);
        assert_eq!(cost, 10_000_000_000); // $10.00
    }

    #[test]
    fn test_gather_codebase_context_runs() {
        // Just verify it doesn't panic — actual content depends on CWD
        let _ = gather_codebase_context();
    }

    #[test]
    fn test_sanitize_globs_rejects_absolute() {
        let globs = vec!["/etc/passwd".into(), "crates/ok/**/*.rs".into()];
        let safe = sanitize_globs(&globs);
        assert_eq!(safe, vec!["crates/ok/**/*.rs"]);
    }

    #[test]
    fn test_sanitize_globs_rejects_traversal() {
        let globs = vec!["../../.ssh/id_rsa".into(), "src/**/*.rs".into()];
        let safe = sanitize_globs(&globs);
        assert_eq!(safe, vec!["src/**/*.rs"]);
    }

    #[test]
    fn test_sanitize_globs_rejects_sensitive() {
        let globs = vec![
            ".env".into(),
            ".git/config".into(),
            "target/debug/**".into(),
            "src/main.rs".into(),
        ];
        let safe = sanitize_globs(&globs);
        assert_eq!(safe, vec!["src/main.rs"]);
    }

    #[test]
    fn test_sanitize_globs_rejects_home_tilde() {
        let globs = vec!["~/.ssh/id_rsa".into()];
        let safe = sanitize_globs(&globs);
        assert!(safe.is_empty());
    }
}
