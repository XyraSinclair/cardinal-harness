//! Claude Code print-mode adapter for subscription-billed chat completions.
//!
//! Calls use the local `claude` subscription session and have zero marginal API cost.
//! [`ClaudeCodeConfig::config_dir`] sets `CLAUDE_CONFIG_DIR` and the child working directory.
//! Prepare that scratch directory with `scripts/claude_code_judge.py --pure`; this module does not
//! copy or refresh credentials.
//!
//! Without an isolated config dir, the operator's own CLAUDE.md, memory, and hooks run inside
//! every call — measured on a live smoke: ~78k input tokens for a one-line prompt, and a stop
//! hook rewrote the final result text. Measurement traffic should always set `config_dir`.
//!
//! A scratch config directory has no model preference. Callers must always set an explicit model
//! with `ChatModel::claude_code`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use serde::{de::IgnoredAny, Deserialize};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use super::error::{ErrorContext, ProviderError};
use super::types::{ChatRequest, ChatResponse, FinishReason, Message, Role};

const PROVIDER: &str = "claude-code";
const QUOTA_MARKERS: &[&str] = &["session limit", "resets ", "rate limit"];
const SAFEGUARD_MARKERS: &[&str] = &[
    "automated abuse",
    "content policy",
    "flagged",
    "safeguard",
    "safety policy",
];

/// Process configuration for [`ClaudeCodeAdapter`].
#[derive(Debug, Clone)]
pub struct ClaudeCodeConfig {
    /// The Claude Code executable name or path.
    pub binary: PathBuf,
    /// An optional Claude Code config directory for isolated judging context.
    pub config_dir: Option<PathBuf>,
    /// An optional Claude Code effort level.
    pub effort: Option<String>,
}

impl Default for ClaudeCodeConfig {
    fn default() -> Self {
        Self {
            binary: PathBuf::from("claude"),
            config_dir: None,
            effort: None,
        }
    }
}

/// Adapter for Claude Code print-mode chat completions.
#[derive(Debug, Clone, Default)]
pub struct ClaudeCodeAdapter {
    config: ClaudeCodeConfig,
}

impl ClaudeCodeAdapter {
    /// Create an adapter with explicit process configuration.
    pub fn new(config: ClaudeCodeConfig) -> Self {
        Self { config }
    }

    /// Execute a chat completion through Claude Code print mode.
    pub async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        if req.model.model_id().is_empty() {
            return Err(ProviderError::invalid_request(
                "Claude Code model must not be empty",
            ));
        }

        let (system_prompt, prompt) = map_messages(&req.messages);
        let mut command = Command::new(&self.config.binary);
        command
            .arg("-p")
            .arg("--output-format")
            .arg("json")
            .arg("--no-session-persistence")
            .arg("--model")
            .arg(req.model.model_id())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(system_prompt) = system_prompt {
            command.arg("--system-prompt").arg(system_prompt);
        }
        if let Some(effort) = self.config.effort.as_deref() {
            command.arg("--effort").arg(effort);
        }
        if let Some(config_dir) = self.config.config_dir.as_deref() {
            command.env("CLAUDE_CONFIG_DIR", config_dir);
            command.current_dir(config_dir);
        }

        let mut child = command.spawn().map_err(|error| {
            ProviderError::config(format!(
                "failed to spawn {}: {error}",
                self.config.binary.display()
            ))
        })?;

        let stdin_error = if let Some(mut stdin) = child.stdin.take() {
            match stdin.write_all(prompt.as_bytes()).await {
                Ok(()) => stdin.shutdown().await.err(),
                Err(error) => Some(error),
            }
        } else {
            None
        };

        let output = child.wait_with_output().await.map_err(|error| {
            ProviderError::provider(
                PROVIDER,
                format!("failed to wait for claude: {error}"),
                true,
            )
        })?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let envelope = parse_envelope(&stdout).map_err(|error| {
            let stdin_detail = stdin_error
                .as_ref()
                .map(|error| format!("; stdin: {error}"))
                .unwrap_or_default();
            ProviderError::provider(
                PROVIDER,
                format!(
                    "invalid JSON envelope: {error}{stdin_detail}; stdout: {:?}; stderr: {:?}",
                    tail(&stdout, 300),
                    tail(&stderr, 300)
                ),
                false,
            )
        })?;

        if envelope.is_error {
            return Err(classify_cli_error(envelope.result, envelope.uuid));
        }
        if !output.status.success() {
            return Err(ProviderError::provider(
                PROVIDER,
                format!(
                    "claude exited with {}; result: {}; stderr: {}",
                    output.status,
                    envelope.result,
                    tail(&stderr, 300)
                ),
                true,
            ));
        }
        if let Some(error) = stdin_error {
            return Err(ProviderError::provider(
                PROVIDER,
                format!("failed to write claude prompt: {error}"),
                true,
            ));
        }

        let usage = envelope.usage.ok_or_else(|| {
            ProviderError::provider(PROVIDER, "missing usage in JSON envelope", false)
        })?;
        let latency_ms = envelope.duration_api_ms.ok_or_else(|| {
            ProviderError::provider(PROVIDER, "missing duration_api_ms in JSON envelope", false)
        })?;
        let served_model = served_model(envelope.model_usage)?;
        let input_tokens = usage
            .input_tokens
            .checked_add(usage.cache_read_input_tokens.unwrap_or(0))
            .and_then(|tokens| tokens.checked_add(usage.cache_creation_input_tokens.unwrap_or(0)))
            .ok_or_else(|| {
                ProviderError::provider(PROVIDER, "input token count overflow", false)
            })?;

        Ok(ChatResponse {
            // Claude Code exposes no completion ID.
            provider_call_id: None,
            provider_request_id: envelope.uuid,
            served_model: Some(served_model),
            content: envelope.result,
            reasoning: None,
            reasoning_tokens: None,
            input_tokens,
            output_tokens: usage.output_tokens,
            cost_nanodollars: 0,
            cost_is_estimate: false,
            upstream_cost_nanodollars: None,
            latency: Duration::from_millis(latency_ms),
            finish_reason: map_finish_reason(envelope.stop_reason),
            output_logprobs: None,
            cache_read_tokens: usage.cache_read_input_tokens,
            cache_write_tokens: usage.cache_creation_input_tokens,
        })
    }
}

#[derive(Deserialize)]
struct ClaudeCodeEnvelope {
    result: String,
    #[serde(default)]
    is_error: bool,
    #[serde(default)]
    usage: Option<ClaudeCodeUsage>,
    #[serde(default, rename = "modelUsage")]
    model_usage: BTreeMap<String, IgnoredAny>,
    #[serde(default)]
    duration_api_ms: Option<u64>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    uuid: Option<String>,
}

#[derive(Deserialize)]
struct ClaudeCodeUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
}

fn parse_envelope(stdout: &str) -> Result<ClaudeCodeEnvelope, serde_json::Error> {
    let line = stdout
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or_default();
    serde_json::from_str(line)
}

fn map_messages(messages: &[Message]) -> (Option<String>, String) {
    let mut system_messages = Vec::new();
    let mut prompt = String::new();

    for message in messages {
        let label = match message.role {
            Role::System => {
                system_messages.push(message.content.as_str());
                continue;
            }
            Role::User => "User:\n",
            Role::Assistant => "Assistant:\n",
        };
        if !prompt.is_empty() {
            prompt.push_str("\n\n");
        }
        prompt.push_str(label);
        prompt.push_str(&message.content);
    }

    let system_prompt = (!system_messages.is_empty()).then(|| system_messages.join("\n\n"));
    (system_prompt, prompt)
}

fn classify_cli_error(message: String, request_id: Option<String>) -> ProviderError {
    let lowercase = message.to_ascii_lowercase();
    if QUOTA_MARKERS
        .iter()
        .any(|marker| lowercase.contains(marker))
    {
        let mut context = ErrorContext::new().with_code(message);
        if let Some(request_id) = request_id {
            context = context.with_request_id(request_id);
        }
        return ProviderError::rate_limited_subscription(context);
    }
    if SAFEGUARD_MARKERS
        .iter()
        .any(|marker| lowercase.contains(marker))
    {
        return ProviderError::refused(message);
    }

    let context = request_id.map(|request_id| ErrorContext::new().with_request_id(request_id));
    match context {
        Some(context) => ProviderError::provider_with_context(PROVIDER, message, true, context),
        None => ProviderError::provider(PROVIDER, message, true),
    }
}

fn served_model(model_usage: BTreeMap<String, IgnoredAny>) -> Result<String, ProviderError> {
    if model_usage.is_empty() {
        return Err(ProviderError::provider(
            PROVIDER,
            "missing modelUsage in JSON envelope",
            false,
        ));
    }
    Ok(model_usage.into_keys().collect::<Vec<_>>().join(","))
}

fn map_finish_reason(reason: Option<String>) -> FinishReason {
    match reason.as_deref() {
        Some("end_turn" | "stop" | "stop_sequence") => FinishReason::Stop,
        Some("length" | "max_tokens") => FinishReason::Length,
        Some("content_filter" | "refusal") => FinishReason::ContentFilter,
        Some("tool_calls" | "tool_use") => FinishReason::ToolCalls,
        Some(other) => FinishReason::Unknown(other.to_string()),
        None => FinishReason::Unknown("none".to_string()),
    }
}

fn tail(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars().rev().take(max_chars).collect::<Vec<_>>();
    chars.reverse();
    chars.into_iter().collect()
}
