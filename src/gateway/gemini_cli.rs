//! Gemini CLI adapter for subscription-billed chat completions.
//!
//! Calls shell out to the official `gemini` CLI (Google's gemini-cli) running
//! under a Google AI Pro / Google One "Login with Google" OAuth session, so
//! they draw the subscription's daily model-request allowance and have zero
//! marginal API cost.
//!
//! Auth lives in a persistent config directory (`HOME/.gemini/oauth_creds.json`)
//! that the operator provisions once; the CLI refreshes the OAuth token in place,
//! so this adapter carries no token-management code. Each child runs from a fresh
//! scratch directory (with `--skip-trust`) so no repository `GEMINI.md` context
//! can bleed into judgments — mirroring the codex adapter's isolation.
//!
//! The CLI resolves the requested `-m` slug against its own model table. On the
//! build verified 2026-08-15 (gemini-cli 0.54.4) the AI Pro session serves
//! `gemini-2.5-pro` natively and maps the flash family (`gemini-2.5-flash`,
//! `gemini-3.7-flash`) onto `gemini-3.5-flash`; the served name is read back
//! from the CLI's own stats and reported in `served_model`.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Instant;

use serde::Deserialize;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use super::error::{ErrorContext, ProviderError};
use super::types::{ChatRequest, ChatResponse, FinishReason, Message, Role};

const PROVIDER: &str = "gemini-cli";
const QUOTA_MARKERS: &[&str] = &[
    "resource_exhausted",
    "quota exceeded",
    "rate limit",
    "rate-limit",
    "429",
    "too many requests",
];
const SAFEGUARD_MARKERS: &[&str] = &[
    "blocked",
    "safety",
    "prohibited_content",
    "content policy",
    "recitation",
];

/// Process configuration for [`GeminiCliAdapter`].
#[derive(Debug, Clone)]
pub struct GeminiCliConfig {
    /// The `gemini` CLI binary (looked up on `PATH` when a bare name).
    pub binary: PathBuf,
    /// Config directory used as `HOME` for the child. Must contain
    /// `.gemini/oauth_creds.json` for the subscription account. When `None`,
    /// a call fails fast rather than silently falling back to an ambient login.
    pub home: Option<PathBuf>,
}

impl Default for GeminiCliConfig {
    fn default() -> Self {
        Self {
            binary: PathBuf::from("gemini"),
            home: None,
        }
    }
}

/// Adapter for Gemini CLI non-interactive chat completions.
#[derive(Debug, Clone, Default)]
pub struct GeminiCliAdapter {
    config: GeminiCliConfig,
}

impl GeminiCliAdapter {
    /// Create an adapter with explicit process configuration.
    pub fn new(config: GeminiCliConfig) -> Self {
        Self { config }
    }

    /// Execute a chat completion through `gemini -o json`.
    pub async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        if req.model.model_id().is_empty() {
            return Err(ProviderError::invalid_request(
                "Gemini CLI model must not be empty",
            ));
        }
        let home = self.config.home.as_ref().ok_or_else(|| {
            ProviderError::config(
                "Gemini CLI home is not configured (set CARDINAL_GEMINI_CLI_HOME to a dir \
                 containing .gemini/oauth_creds.json)",
            )
        })?;

        let prompt = map_messages(&req.messages);
        let scratch = tempfile::Builder::new()
            .prefix("cardinal-gemini-")
            .tempdir_in(std::env::temp_dir())
            .map_err(|error| {
                ProviderError::config(format!(
                    "failed to create Gemini CLI scratch directory: {error}"
                ))
            })?;

        let mut command = Command::new(&self.config.binary);
        command
            .arg("--skip-trust")
            .arg("-o")
            .arg("json")
            .arg("-m")
            .arg(req.model.model_id())
            // HOME carries the persistent OAuth session; CWD is a throwaway so no
            // repository GEMINI.md / settings can affect the judgment.
            .env("HOME", home)
            .env("GEMINI_CLI_TRUST_WORKSPACE", "true")
            .current_dir(scratch.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let start = Instant::now();
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
                format!("failed to wait for Gemini CLI: {error}"),
                true,
            )
        })?;
        let latency = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if !output.status.success() {
            // The CLI prints its failure on stderr and emits no JSON envelope.
            let detail = format!(
                "Gemini CLI exited with {}; stderr: {}",
                output.status,
                tail(&stderr, 600)
            );
            return Err(classify_cli_error(detail));
        }
        if let Some(error) = stdin_error {
            return Err(ProviderError::provider(
                PROVIDER,
                format!("failed to write Gemini CLI prompt: {error}"),
                true,
            ));
        }

        let envelope: GeminiEnvelope = serde_json::from_str(stdout.trim()).map_err(|error| {
            ProviderError::provider(
                PROVIDER,
                format!(
                    "invalid JSON envelope: {error}; stdout: {:?}; stderr: {:?}",
                    tail(&stdout, 500),
                    tail(&stderr, 300)
                ),
                false,
            )
        })?;

        // Token counts feed run denominators; a missing stats block must not
        // silently report zero (the codex adapter errors identically). Read the
        // stats borrow before moving `response` out of the envelope below.
        let (served_model, usage) = envelope.aggregate_usage().ok_or_else(|| {
            ProviderError::provider(
                PROVIDER,
                format!(
                    "missing per-model token stats in Gemini CLI envelope; stderr: {:?}",
                    tail(&stderr, 300)
                ),
                false,
            )
        })?;

        let content = envelope.response.ok_or_else(|| {
            ProviderError::provider(
                PROVIDER,
                format!(
                    "missing response field in Gemini CLI envelope; stderr: {:?}",
                    tail(&stderr, 300)
                ),
                false,
            )
        })?;

        Ok(ChatResponse {
            provider_call_id: envelope.session_id.clone(),
            provider_request_id: envelope.session_id,
            served_model: Some(served_model),
            content,
            reasoning: None,
            reasoning_tokens: usage.thoughts,
            input_tokens: usage.prompt,
            output_tokens: usage.candidates,
            cost_nanodollars: 0,
            cost_is_estimate: false,
            upstream_cost_nanodollars: None,
            latency,
            finish_reason: FinishReason::Stop,
            output_logprobs: None,
            cache_read_tokens: usage.cached,
            cache_write_tokens: None,
        })
    }
}

#[derive(Deserialize)]
struct GeminiEnvelope {
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    response: Option<String>,
    #[serde(default)]
    stats: Option<GeminiStats>,
}

#[derive(Deserialize)]
struct GeminiStats {
    #[serde(default)]
    models: std::collections::BTreeMap<String, GeminiModelStats>,
}

#[derive(Deserialize)]
struct GeminiModelStats {
    #[serde(default)]
    tokens: GeminiTokens,
}

#[derive(Debug, Default, Clone, Deserialize)]
struct GeminiTokens {
    #[serde(default)]
    prompt: u32,
    #[serde(default)]
    candidates: u32,
    #[serde(default)]
    thoughts: Option<u32>,
    #[serde(default)]
    cached: Option<u32>,
}

impl GeminiEnvelope {
    /// Sum token counts across served models and name the served model(s).
    /// Returns `None` when the CLI emitted no per-model stats.
    fn aggregate_usage(&self) -> Option<(String, GeminiTokens)> {
        let models = &self.stats.as_ref()?.models;
        if models.is_empty() {
            return None;
        }
        let served_model = models.keys().cloned().collect::<Vec<_>>().join(",");
        let mut total = GeminiTokens::default();
        for stats in models.values() {
            let t = &stats.tokens;
            total.prompt += t.prompt;
            total.candidates += t.candidates;
            total.thoughts = add_opt(total.thoughts, t.thoughts);
            total.cached = add_opt(total.cached, t.cached);
        }
        Some((served_model, total))
    }
}

fn add_opt(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (None, None) => None,
        (x, y) => Some(x.unwrap_or(0) + y.unwrap_or(0)),
    }
}

/// Fold messages into a single prompt. The Gemini CLI has no separate system
/// channel in non-interactive mode, so system content is prepended as a labeled
/// block ahead of the conversation.
fn map_messages(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for message in messages {
        let label = match message.role {
            Role::System => "System:\n",
            Role::User => "User:\n",
            Role::Assistant => "Assistant:\n",
        };
        if !prompt.is_empty() {
            prompt.push_str("\n\n");
        }
        prompt.push_str(label);
        prompt.push_str(&message.content);
    }
    prompt
}

fn classify_cli_error(message: String) -> ProviderError {
    let lowercase = message.to_ascii_lowercase();
    if QUOTA_MARKERS.iter().any(|marker| lowercase.contains(marker)) {
        let context = ErrorContext::new().with_code(message);
        return ProviderError::rate_limited_subscription(context);
    }
    if SAFEGUARD_MARKERS
        .iter()
        .any(|marker| lowercase.contains(marker))
    {
        return ProviderError::refused(message);
    }
    ProviderError::provider(PROVIDER, message, true)
}

fn tail(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars().rev().take(max_chars).collect::<Vec<_>>();
    chars.reverse();
    chars.into_iter().collect()
}
