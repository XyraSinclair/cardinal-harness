//! Extract concrete proposals from flywheel synthesis outputs via LLM.

use serde::{Deserialize, Serialize};

use crate::gateway::{Attribution, ChatGateway, ChatModel, ChatRequest, Message};

use super::store::{EstimatedEffort, ProposalCategory, ProposalPriority, ProposalStatus};

// =============================================================================
// Types
// =============================================================================

/// Raw proposal as extracted from LLM JSON output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedProposal {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_category")]
    pub category: String,
    #[serde(default = "default_priority")]
    pub priority: String,
    #[serde(default)]
    pub affected_files: Vec<String>,
    #[serde(default = "default_effort")]
    pub estimated_effort: String,
}

fn default_category() -> String {
    "improvement".into()
}
fn default_priority() -> String {
    "medium".into()
}
fn default_effort() -> String {
    "medium".into()
}

impl ExtractedProposal {
    pub fn category_enum(&self) -> ProposalCategory {
        ProposalCategory::from_str(&self.category)
    }

    pub fn priority_enum(&self) -> ProposalPriority {
        ProposalPriority::from_str(&self.priority)
    }

    pub fn effort_enum(&self) -> EstimatedEffort {
        EstimatedEffort::from_str(&self.estimated_effort)
    }

    pub fn affected_files_json(&self) -> String {
        serde_json::to_string(&self.affected_files).unwrap_or_else(|_| "[]".into())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExtractionResponse {
    #[serde(default)]
    proposals: Vec<ExtractedProposal>,
}

#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    #[error("LLM call failed: {0}")]
    LlmFailed(#[from] crate::gateway::error::ProviderError),
    #[error("JSON extraction failed: {0}")]
    JsonParse(String),
}

// =============================================================================
// Extraction
// =============================================================================

const EXTRACTION_SYSTEM_PROMPT: &str = "\
You are a code improvement proposal extractor. You receive the synthesis output from a \
multi-model analysis of a codebase. Your job is to extract concrete, actionable proposals \
that could each be implemented as a single pull request.

For each proposal, provide:
- title: imperative mood, concise (e.g. \"Add fsync fence after Spiel batch commit\")
- description: 2-4 sentences explaining what to change and why
- category: one of bug_fix, refactor, performance, safety, architecture, improvement
- priority: one of critical, high, medium, low
- affected_files: array of file paths that would need to change
- estimated_effort: one of trivial, small, medium, large

Rules:
- Extract 0-5 proposals per synthesis. Quality over quantity.
- If the synthesis contains no actionable improvements, return an empty array.
- Do NOT fabricate proposals — only extract what is supported by the synthesis content.
- Each proposal must be specific enough that an engineer knows exactly what to do.
- Proposals should be independent — each one is a standalone PR.
- If an Existing Proposals section is provided, check for overlap. Only include a proposal \
  if it adds genuinely new information beyond what already exists.

Respond with JSON only:
{
  \"proposals\": [
    {
      \"title\": \"...\",
      \"description\": \"...\",
      \"category\": \"...\",
      \"priority\": \"...\",
      \"affected_files\": [\"...\"],
      \"estimated_effort\": \"...\"
    }
  ]
}";

/// Extract proposals from a synthesis output using an LLM.
///
/// Returns (proposals, cost_nanodollars, raw_llm_output).
pub async fn extract_proposals(
    gateway: &dyn ChatGateway,
    model: &str,
    task_id: &str,
    task_prompt: &str,
    synthesis_content: &str,
    existing_titles: &[(String, ProposalStatus)],
) -> Result<(Vec<ExtractedProposal>, i64, String), ExtractError> {
    let mut user_prompt = format!(
        "## Task\n{}\n\n## Task ID\n{}\n\n## Synthesis Output\n\n{}",
        task_prompt, task_id, synthesis_content
    );

    // Add dedup context if there are existing proposals
    if !existing_titles.is_empty() {
        user_prompt.push_str("\n\n## Existing Proposals\n\n");
        for (title, status) in existing_titles {
            let status_str = match status {
                ProposalStatus::Pending => "pending",
                ProposalStatus::Accepted => "accepted",
                ProposalStatus::Rejected => "rejected",
                ProposalStatus::Deferred => "deferred",
                ProposalStatus::Implemented => "implemented",
            };
            user_prompt.push_str(&format!("- [{status_str}] {title}\n"));
        }
    }

    let messages = vec![
        Message::system(EXTRACTION_SYSTEM_PROMPT),
        Message::user(user_prompt),
    ];

    let req = ChatRequest::new(
        ChatModel::openrouter(model),
        messages,
        Attribution::new("commander::extract"),
    )
    .temperature(0.1)
    .max_tokens(4096)
    .json();

    let resp = gateway.chat(req).await?;
    let cost = resp.cost_nanodollars;
    let raw_output = resp.content.clone();

    let json_str = extract_json(&resp.content);
    let parsed: ExtractionResponse = serde_json::from_str(json_str).map_err(|e| {
        let preview: String = resp.content.chars().take(500).collect();
        ExtractError::JsonParse(format!(
            "failed to parse extraction response: {} — raw: {}",
            e, preview
        ))
    })?;

    // Filter out proposals with empty titles
    let proposals: Vec<ExtractedProposal> = parsed
        .proposals
        .into_iter()
        .filter(|p| !p.title.trim().is_empty())
        .take(5)
        .collect();

    Ok((proposals, cost, raw_output))
}

// =============================================================================
// JSON extraction (copied from rerank/comparison.rs:126)
// =============================================================================

/// Extract a JSON object from potentially noisy LLM output.
///
/// Handles:
/// - Pure JSON responses
/// - JSON wrapped in markdown code fences
/// - JSON embedded in prose
pub fn extract_json(raw: &str) -> &str {
    let trimmed = raw.trim();

    // If it starts with {, find matching closing brace
    if trimmed.starts_with('{') {
        if let Some(end) = find_matching_brace(trimmed) {
            return &trimmed[..end];
        }
    }

    // Try to find JSON anywhere in the response
    if let Some(start) = trimmed.find('{') {
        let remainder = &trimmed[start..];
        if let Some(end) = find_matching_brace(remainder) {
            return &remainder[..end];
        }
    }

    trimmed
}

/// Find the byte offset of the matching closing brace, respecting JSON strings.
/// Tracks "inside string" state so braces within `"..."` are not counted.
fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for (i, c) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' && in_string {
            escape = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_pure() {
        let input = r#"{"proposals": []}"#;
        assert_eq!(extract_json(input), input);
    }

    #[test]
    fn test_extract_json_with_prose() {
        let input =
            "Here is my analysis:\n```json\n{\"proposals\": [{\"title\": \"Fix bug\"}]}\n```";
        let result = extract_json(input);
        assert!(result.starts_with('{'));
        assert!(result.ends_with('}'));
    }

    #[test]
    fn test_extract_json_nested() {
        let input = r#"{"proposals": [{"title": "test", "affected_files": ["a.rs"]}]}"#;
        assert_eq!(extract_json(input), input);
    }

    #[test]
    fn test_extracted_proposal_defaults() {
        let json = r#"{"title": "Do thing", "description": "desc"}"#;
        let p: ExtractedProposal = serde_json::from_str(json).unwrap();
        assert_eq!(p.category, "improvement");
        assert_eq!(p.priority, "medium");
        assert_eq!(p.estimated_effort, "medium");
        assert!(p.affected_files.is_empty());
    }

    #[test]
    fn test_extract_json_braces_in_strings() {
        // M3: braces inside JSON string values must not confuse extraction
        let input = r#"{"description": "Use {braces} literally", "count": 1}"#;
        assert_eq!(extract_json(input), input);

        let wrapped = r#"Here is the result: {"desc": "a {b} c", "x": 2} done"#;
        let result = extract_json(wrapped);
        assert_eq!(result, r#"{"desc": "a {b} c", "x": 2}"#);
    }

    #[test]
    fn test_extract_json_escaped_quotes_in_strings() {
        let input = r#"{"title": "Fix \"broken\" thing"}"#;
        assert_eq!(extract_json(input), input);
    }

    #[test]
    fn test_category_enum_conversion() {
        let p = ExtractedProposal {
            title: "t".into(),
            description: "d".into(),
            category: "bug_fix".into(),
            priority: "critical".into(),
            affected_files: vec!["a.rs".into()],
            estimated_effort: "trivial".into(),
        };
        assert_eq!(p.category_enum(), ProposalCategory::BugFix);
        assert_eq!(p.priority_enum(), ProposalPriority::Critical);
        assert_eq!(p.effort_enum(), EstimatedEffort::Trivial);
    }
}
