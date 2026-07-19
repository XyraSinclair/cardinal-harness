//! Portable, durable records for finite-candidate, single-axis judgement runs.
//!
//! The record is the boundary between provider-backed execution and replay: once
//! persisted, callers can reload and project the response without a gateway.

pub mod edge;

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::gateway::{
    ChatGateway, ChatRequest, ChatResponse, ProviderError, ReasoningEffort, Role,
};
use crate::rating_engine::EngineSpec;
use crate::rerank::{
    multi_rerank, validate_multi_rerank_request, AttributeScoreSummary, ComparisonTrace,
    MultiRerankAttributeSpec, MultiRerankEntity, MultiRerankRequest, MultiRerankResponse,
    MultiRerankTopKSpec, RerankExecution, RerankStopReason, TraceError, TraceSink,
};

pub const JUDGEMENT_RUN_SCHEMA: &str = "cardinal.judgement-run.v1";
const RUN_REF_PREFIX: &str = "jrun_";
const PROVIDER_CALL_REF_PREFIX: &str = "pcall_";
const PROMPT_TEMPLATE_SLUG: &str = "canonical_v2";
const COMPARISON_CONCURRENCY: usize = 8;
const REQUEST_DIGEST_DOMAIN: &[u8] = b"cardinal.gateway-request.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JudgementPrivacy {
    Public,
    Private,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgementCandidate {
    pub id: String,
    pub text: String,
}

/// Finite-candidate, one-axis request accepted at the portable-run boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgementRunRequest {
    pub entities: Vec<JudgementCandidate>,
    pub axis_key: String,
    pub axis_prompt: String,
    pub requested_k: usize,
    pub model: String,
    pub privacy: JudgementPrivacy,
}

/// Validated request bytes that were used to construct the instrument.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedJudgementRunRequest {
    pub entities: Vec<JudgementCandidate>,
    pub axis_key: String,
    pub axis_prompt: String,
    pub requested_k: usize,
    pub model: String,
    pub privacy: JudgementPrivacy,
}

impl JudgementRunRequest {
    fn normalize(mut self) -> Result<NormalizedJudgementRunRequest, JudgementRunError> {
        for entity in &mut self.entities {
            entity.id = entity.id.trim().to_string();
        }
        self.axis_key = self.axis_key.trim().to_string();
        self.axis_prompt = self.axis_prompt.trim().to_string();
        self.model = self.model.trim().to_string();

        let normalized = NormalizedJudgementRunRequest {
            entities: self.entities,
            axis_key: self.axis_key,
            axis_prompt: self.axis_prompt,
            requested_k: self.requested_k,
            model: self.model,
            privacy: self.privacy,
        };
        normalized
            .validate()
            .map_err(JudgementRunError::InvalidRequest)?;
        Ok(normalized)
    }
}

impl NormalizedJudgementRunRequest {
    fn validate(&self) -> Result<(), String> {
        if self.entities.len() < 2 {
            return Err("entities must contain at least 2 items".to_string());
        }
        if self.axis_key.is_empty() {
            return Err("axis_key must not be blank".to_string());
        }
        if self.axis_prompt.is_empty() {
            return Err("axis_prompt must not be blank".to_string());
        }
        if self.model.is_empty() {
            return Err("model must not be blank".to_string());
        }
        if self.requested_k == 0 || self.requested_k > self.entities.len() {
            return Err(format!(
                "requested_k must be between 1 and the entity count ({})",
                self.entities.len()
            ));
        }

        let mut ids = HashSet::with_capacity(self.entities.len());
        for entity in &self.entities {
            if entity.id.is_empty() {
                return Err("entity ids must not be blank".to_string());
            }
            if entity.id.trim() != entity.id {
                return Err(format!("entity id is not normalized: {}", entity.id));
            }
            if entity.text.trim().is_empty() {
                return Err(format!("entity text must not be blank: {}", entity.id));
            }
            if !ids.insert(entity.id.as_str()) {
                return Err(format!("duplicate entity id: {}", entity.id));
            }
        }

        if self.axis_key.trim() != self.axis_key
            || self.axis_prompt.trim() != self.axis_prompt
            || self.model.trim() != self.model
        {
            return Err("request strings are not normalized".to_string());
        }
        Ok(())
    }
}

/// Exact resolved rerank invocation plus the solver constructor spec it produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgementInstrumentSpec {
    pub rerank_request: MultiRerankRequest,
    pub cache_enabled: bool,
    pub cache_only: bool,
    pub rng_seed: Option<u64>,
    pub engine_spec: Option<EngineSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgementRunUsage {
    pub provider_input_tokens: u32,
    pub provider_output_tokens: u32,
    pub provider_cost_nanodollars: i64,
    pub provider_cost_is_estimate: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgementAttributeScore {
    pub latent_mean: f64,
    pub latent_std: f64,
    pub percentile: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgementEntityScore {
    pub id: String,
    pub feasible: bool,
    pub p_flip: f64,
    pub attribute_score: JudgementAttributeScore,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgementRunResponse {
    pub entities: Vec<JudgementEntityScore>,
    pub global_topk_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum JudgementRunTerminal {
    Completed {
        stop_reason: RerankStopReason,
        response: JudgementRunResponse,
    },
    Cancelled {
        response: JudgementRunResponse,
    },
    Failed {
        error: String,
    },
}

impl JudgementRunTerminal {
    #[must_use]
    pub fn status(&self) -> &'static str {
        match self {
            Self::Completed { .. } => "completed",
            Self::Cancelled { .. } => "cancelled",
            Self::Failed { .. } => "failed",
        }
    }

    #[must_use]
    pub fn completed_response(&self) -> Option<&JudgementRunResponse> {
        match self {
            Self::Completed { response, .. } => Some(response),
            Self::Cancelled { .. } | Self::Failed { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgementProviderCall {
    pub call_ref: String,
    pub sequence: usize,
    pub provider: String,
    pub model: String,
    pub gateway_request_digest: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub outcome: JudgementProviderCallOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum JudgementProviderCallOutcome {
    Succeeded {
        provider_call_id: Option<String>,
        provider_request_id: Option<String>,
        input_tokens: u32,
        output_tokens: u32,
        cost_nanodollars: i64,
        cost_is_estimate: bool,
    },
    Failed {
        provider_request_id: Option<String>,
        error_code: String,
        error: String,
    },
}

/// Self-contained terminal atom. No provider capability is needed to read it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgementRunRecord {
    pub schema: String,
    pub run_ref: String,
    pub request: NormalizedJudgementRunRequest,
    pub instrument: JudgementInstrumentSpec,
    pub comparison_trace: Vec<ComparisonTrace>,
    pub provider_calls: Vec<JudgementProviderCall>,
    pub usage: JudgementRunUsage,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub terminal: JudgementRunTerminal,
}

impl JudgementRunRecord {
    #[must_use]
    pub fn completed_response(&self) -> Option<&JudgementRunResponse> {
        self.terminal.completed_response()
    }
}

/// Atomic, one-record-per-file JSON store keyed by opaque `run_ref`.
#[derive(Debug, Clone)]
pub struct JudgementRunStore {
    root: PathBuf,
}

impl JudgementRunStore {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn persist(&self, record: &JudgementRunRecord) -> Result<(), JudgementRunError> {
        validate_record(record)?;
        fs::create_dir_all(&self.root)?;
        let destination = self.record_path(&record.run_ref)?;
        let mut temporary = tempfile::NamedTempFile::new_in(&self.root)?;
        {
            let mut writer = BufWriter::new(temporary.as_file_mut());
            serde_json::to_writer(&mut writer, record)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        temporary.as_file().sync_all()?;
        temporary
            .persist_noclobber(&destination)
            .map_err(|error| JudgementRunError::Io(error.error))?;
        File::open(&self.root)?.sync_all()?;
        Ok(())
    }

    pub fn load(&self, run_ref: &str) -> Result<JudgementRunRecord, JudgementRunError> {
        let path = self.record_path(run_ref)?;
        let reader = BufReader::new(File::open(path)?);
        let record: JudgementRunRecord = serde_json::from_reader(reader)?;
        validate_record(&record)?;
        if record.run_ref != run_ref {
            return Err(JudgementRunError::InvalidRecord(format!(
                "requested run_ref {run_ref} does not match stored {}",
                record.run_ref
            )));
        }
        Ok(record)
    }

    fn record_path(&self, run_ref: &str) -> Result<PathBuf, JudgementRunError> {
        validate_opaque_ref(run_ref, RUN_REF_PREFIX).map_err(JudgementRunError::InvalidRunRef)?;
        Ok(self.root.join(format!("{run_ref}.json")))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum JudgementRunError {
    #[error("invalid judgement request: {0}")]
    InvalidRequest(String),
    #[error("execution cannot produce a portable v1 record: {0}")]
    UnsupportedExecution(String),
    #[error("invalid run_ref: {0}")]
    InvalidRunRef(String),
    #[error("invalid judgement-run record: {0}")]
    InvalidRecord(String),
    #[error("judgement run {run_ref} failed: {source}")]
    Execution {
        run_ref: String,
        #[source]
        source: crate::rerank::MultiRerankError,
    },
    #[error("judgement run {run_ref} violated its execution contract: {error}")]
    ExecutionInvariant { run_ref: String, error: String },
    #[error(
        "judgement run {run_ref} failed and its terminal record could not be persisted: execution={execution}; persistence={persistence}"
    )]
    FailedRecordPersistence {
        run_ref: String,
        execution: String,
        persistence: String,
    },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Execute, capture, and atomically persist one portable judgement run.
pub async fn execute_judgement_run(
    request: JudgementRunRequest,
    execution: RerankExecution<'_>,
    store: &JudgementRunStore,
) -> Result<JudgementRunRecord, JudgementRunError> {
    let request = request.normalize()?;
    let rerank_request = build_rerank_request(&request);
    validate_multi_rerank_request(&rerank_request)
        .map_err(|error| JudgementRunError::InvalidRequest(error.to_string()))?;

    let (gateway, upstream_trace, run_options, cache_enabled) = execution
        .judgement_run_instrumentation()
        .map_err(|reason| JudgementRunError::UnsupportedExecution(reason.to_string()))?;

    let run_ref = new_opaque_ref(RUN_REF_PREFIX);
    let started_at = Utc::now();
    let trace = CapturingTraceSink::new(upstream_trace);
    let provider_calls = Arc::new(Mutex::new(Vec::new()));
    let recording_gateway = Arc::new(RecordingGateway {
        inner: gateway,
        calls: Arc::clone(&provider_calls),
        next_sequence: AtomicUsize::new(0),
    });
    let instrumented = execution.with_judgement_run_instrumentation(recording_gateway, &trace);

    let mut instrument = JudgementInstrumentSpec {
        rerank_request: rerank_request.clone(),
        cache_enabled,
        cache_only: run_options.cache_only,
        rng_seed: run_options.rng_seed,
        engine_spec: None,
    };
    let result = multi_rerank(rerank_request, instrumented).await;
    let finished_at = Utc::now();
    let mut comparison_trace = trace.events();
    comparison_trace.sort_by_key(|event| (event.comparison_index, event.timestamp_ms));
    let mut provider_calls = lock_unpoisoned(&provider_calls).clone();
    provider_calls.sort_by_key(|call| call.sequence);

    match result {
        Ok(response) => {
            let projection = match project_response(&request.axis_key, response) {
                Ok(projection) => projection,
                Err(error) => {
                    let record = failed_record(
                        run_ref.clone(),
                        request,
                        instrument,
                        comparison_trace,
                        provider_calls,
                        started_at,
                        finished_at,
                        error.clone(),
                    );
                    persist_failed(store, &record, &error)?;
                    return Err(JudgementRunError::ExecutionInvariant { run_ref, error });
                }
            };
            instrument.engine_spec = Some(projection.engine_spec);
            let terminal = if projection.stop_reason == RerankStopReason::Cancelled {
                JudgementRunTerminal::Cancelled {
                    response: projection.response,
                }
            } else {
                JudgementRunTerminal::Completed {
                    stop_reason: projection.stop_reason,
                    response: projection.response,
                }
            };
            let record = JudgementRunRecord {
                schema: JUDGEMENT_RUN_SCHEMA.to_string(),
                run_ref,
                request,
                instrument,
                comparison_trace,
                provider_calls,
                usage: projection.usage,
                started_at,
                finished_at,
                terminal,
            };
            store.persist(&record)?;
            Ok(record)
        }
        Err(source) => {
            let execution_error = source.to_string();
            let record = failed_record(
                run_ref.clone(),
                request,
                instrument,
                comparison_trace,
                provider_calls,
                started_at,
                finished_at,
                execution_error.clone(),
            );
            persist_failed(store, &record, &execution_error)?;
            Err(JudgementRunError::Execution { run_ref, source })
        }
    }
}

fn build_rerank_request(request: &NormalizedJudgementRunRequest) -> MultiRerankRequest {
    MultiRerankRequest {
        entities: request
            .entities
            .iter()
            .map(|entity| MultiRerankEntity {
                id: entity.id.clone(),
                text: entity.text.clone(),
            })
            .collect(),
        attributes: vec![MultiRerankAttributeSpec {
            id: request.axis_key.clone(),
            prompt: request.axis_prompt.clone(),
            prompt_template_slug: Some(PROMPT_TEMPLATE_SLUG.to_string()),
            weight: 1.0,
        }],
        topk: MultiRerankTopKSpec {
            k: request.requested_k,
            weight_exponent: 1.3,
            tolerated_error: 0.1,
            band_size: 5,
            effective_resistance_max_active: 64,
            stop_sigma_inflate: 1.25,
            stop_min_consecutive: 2,
            min_explore_degree: 2,
            prune_p_topk_below: None,
        },
        gates: Vec::new(),
        comparison_budget: Some(4usize.saturating_mul(request.entities.len())),
        latency_budget_ms: None,
        model: Some(request.model.clone()),
        rater_id: Some(request.model.clone()),
        comparison_concurrency: Some(COMPARISON_CONCURRENCY),
        max_pair_repeats: None,
        randomize_presentation_order: true,
        counterbalance_pairs: false,
    }
}

struct Projection {
    response: JudgementRunResponse,
    usage: JudgementRunUsage,
    engine_spec: EngineSpec,
    stop_reason: RerankStopReason,
}

fn project_response(axis_key: &str, response: MultiRerankResponse) -> Result<Projection, String> {
    let MultiRerankResponse {
        entities,
        meta,
        pareto_front: _,
        attribute_correlations: _,
    } = response;
    if meta.warm_start_observations != 0 {
        return Err("warm-start observations entered a portable run".to_string());
    }
    let engine_spec = meta
        .engine_spec
        .ok_or_else(|| "rerank response omitted its engine spec".to_string())?;
    let mut projected = Vec::with_capacity(entities.len());
    for mut entity in entities {
        let AttributeScoreSummary {
            latent_mean,
            latent_std,
            percentile,
            ..
        } = entity
            .attribute_scores
            .remove(axis_key)
            .ok_or_else(|| format!("entity {} omitted axis {axis_key}", entity.id))?;
        projected.push(JudgementEntityScore {
            id: entity.id,
            feasible: entity.feasible,
            p_flip: entity.p_flip,
            attribute_score: JudgementAttributeScore {
                latent_mean,
                latent_std,
                percentile,
            },
        });
    }
    Ok(Projection {
        response: JudgementRunResponse {
            entities: projected,
            global_topk_error: meta.global_topk_error,
        },
        usage: JudgementRunUsage {
            provider_input_tokens: meta.provider_input_tokens,
            provider_output_tokens: meta.provider_output_tokens,
            provider_cost_nanodollars: meta.provider_cost_nanodollars,
            provider_cost_is_estimate: meta.provider_cost_is_estimate,
        },
        engine_spec,
        stop_reason: meta.stop_reason,
    })
}

fn failed_record(
    run_ref: String,
    request: NormalizedJudgementRunRequest,
    instrument: JudgementInstrumentSpec,
    comparison_trace: Vec<ComparisonTrace>,
    provider_calls: Vec<JudgementProviderCall>,
    started_at: DateTime<Utc>,
    finished_at: DateTime<Utc>,
    error: String,
) -> JudgementRunRecord {
    let usage = usage_from_calls(&provider_calls);
    JudgementRunRecord {
        schema: JUDGEMENT_RUN_SCHEMA.to_string(),
        run_ref,
        request,
        instrument,
        comparison_trace,
        provider_calls,
        usage,
        started_at,
        finished_at,
        terminal: JudgementRunTerminal::Failed { error },
    }
}

fn persist_failed(
    store: &JudgementRunStore,
    record: &JudgementRunRecord,
    execution: &str,
) -> Result<(), JudgementRunError> {
    store
        .persist(record)
        .map_err(|persistence| JudgementRunError::FailedRecordPersistence {
            run_ref: record.run_ref.clone(),
            execution: execution.to_string(),
            persistence: persistence.to_string(),
        })
}

fn usage_from_calls(calls: &[JudgementProviderCall]) -> JudgementRunUsage {
    let mut usage = JudgementRunUsage {
        provider_input_tokens: 0,
        provider_output_tokens: 0,
        provider_cost_nanodollars: 0,
        provider_cost_is_estimate: false,
    };
    for call in calls {
        if let JudgementProviderCallOutcome::Succeeded {
            input_tokens,
            output_tokens,
            cost_nanodollars,
            cost_is_estimate,
            ..
        } = call.outcome
        {
            usage.provider_input_tokens = usage.provider_input_tokens.saturating_add(input_tokens);
            usage.provider_output_tokens =
                usage.provider_output_tokens.saturating_add(output_tokens);
            usage.provider_cost_nanodollars = usage
                .provider_cost_nanodollars
                .saturating_add(cost_nanodollars);
            usage.provider_cost_is_estimate |= cost_is_estimate;
        }
    }
    usage
}

fn validate_record(record: &JudgementRunRecord) -> Result<(), JudgementRunError> {
    if record.schema != JUDGEMENT_RUN_SCHEMA {
        return Err(JudgementRunError::InvalidRecord(format!(
            "unsupported schema {}",
            record.schema
        )));
    }
    validate_opaque_ref(&record.run_ref, RUN_REF_PREFIX)
        .map_err(JudgementRunError::InvalidRunRef)?;
    record
        .request
        .validate()
        .map_err(JudgementRunError::InvalidRecord)?;
    if record.finished_at < record.started_at {
        return Err(JudgementRunError::InvalidRecord(
            "finished_at precedes started_at".to_string(),
        ));
    }

    let expected_request = build_rerank_request(&record.request);
    if serde_json::to_value(&expected_request)?
        != serde_json::to_value(&record.instrument.rerank_request)?
    {
        return Err(JudgementRunError::InvalidRecord(
            "instrument request does not match normalized request".to_string(),
        ));
    }

    for (expected_sequence, call) in record.provider_calls.iter().enumerate() {
        if call.sequence != expected_sequence {
            return Err(JudgementRunError::InvalidRecord(
                "provider calls are not in invocation order".to_string(),
            ));
        }
        validate_opaque_ref(&call.call_ref, PROVIDER_CALL_REF_PREFIX)
            .map_err(JudgementRunError::InvalidRecord)?;
        if call.finished_at < call.started_at {
            return Err(JudgementRunError::InvalidRecord(format!(
                "provider call {} finishes before it starts",
                call.call_ref
            )));
        }
    }

    match &record.terminal {
        JudgementRunTerminal::Completed { response, .. }
        | JudgementRunTerminal::Cancelled { response } => {
            let engine_spec = record.instrument.engine_spec.as_ref().ok_or_else(|| {
                JudgementRunError::InvalidRecord(
                    "successful terminal record omitted engine_spec".to_string(),
                )
            })?;
            validate_response_ids(&record.request, response)?;
            let engine_spec_id = engine_spec.id().0;
            if record.comparison_trace.iter().any(|event| {
                !event.engine_spec_id.is_empty() && event.engine_spec_id != engine_spec_id
            }) {
                return Err(JudgementRunError::InvalidRecord(
                    "comparison trace references a different engine spec".to_string(),
                ));
            }
            if usage_from_calls(&record.provider_calls) != record.usage {
                return Err(JudgementRunError::InvalidRecord(
                    "provider-call totals do not match run totals".to_string(),
                ));
            }
        }
        JudgementRunTerminal::Failed { .. } => {}
    }
    Ok(())
}

fn validate_response_ids(
    request: &NormalizedJudgementRunRequest,
    response: &JudgementRunResponse,
) -> Result<(), JudgementRunError> {
    if response.entities.len() != request.entities.len() {
        return Err(JudgementRunError::InvalidRecord(
            "response entity count does not match request".to_string(),
        ));
    }
    let expected: HashSet<_> = request
        .entities
        .iter()
        .map(|entity| entity.id.as_str())
        .collect();
    let actual: HashSet<_> = response
        .entities
        .iter()
        .map(|entity| entity.id.as_str())
        .collect();
    if actual.len() != response.entities.len() || actual != expected {
        return Err(JudgementRunError::InvalidRecord(
            "response entity ids do not match request".to_string(),
        ));
    }
    Ok(())
}

fn new_opaque_ref(prefix: &str) -> String {
    format!("{prefix}{}", Uuid::new_v4().simple())
}

fn validate_opaque_ref(value: &str, prefix: &str) -> Result<(), String> {
    let suffix = value
        .strip_prefix(prefix)
        .ok_or_else(|| format!("expected {prefix} prefix"))?;
    if suffix.len() != 32 {
        return Err("opaque reference must contain a simple UUID".to_string());
    }
    let uuid =
        Uuid::parse_str(suffix).map_err(|_| "opaque reference UUID is invalid".to_string())?;
    if uuid.get_version_num() != 4 {
        return Err("opaque reference UUID must be version 4".to_string());
    }
    Ok(())
}

struct CapturingTraceSink<'a> {
    upstream: Option<&'a dyn TraceSink>,
    events: Mutex<Vec<ComparisonTrace>>,
}

impl<'a> CapturingTraceSink<'a> {
    fn new(upstream: Option<&'a dyn TraceSink>) -> Self {
        Self {
            upstream,
            events: Mutex::new(Vec::new()),
        }
    }

    fn events(&self) -> Vec<ComparisonTrace> {
        lock_unpoisoned(&self.events).clone()
    }
}

impl TraceSink for CapturingTraceSink<'_> {
    fn record(&self, event: ComparisonTrace) -> Result<(), TraceError> {
        lock_unpoisoned(&self.events).push(event.clone());
        if let Some(upstream) = self.upstream {
            upstream.record(event)?;
        }
        Ok(())
    }
}

struct RecordingGateway {
    inner: Arc<dyn ChatGateway>,
    calls: Arc<Mutex<Vec<JudgementProviderCall>>>,
    next_sequence: AtomicUsize,
}

#[async_trait::async_trait]
impl ChatGateway for RecordingGateway {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let call_ref = new_opaque_ref(PROVIDER_CALL_REF_PREFIX);
        let provider = request.model.provider().to_string();
        let model = request.model.model_id().to_string();
        let gateway_request_digest = gateway_request_digest(&request);
        let started_at = Utc::now();
        let result = self.inner.chat(request).await;
        let finished_at = Utc::now();
        let outcome = match &result {
            Ok(response) => JudgementProviderCallOutcome::Succeeded {
                provider_call_id: response.provider_call_id.clone(),
                provider_request_id: response.provider_request_id.clone(),
                input_tokens: response.input_tokens,
                output_tokens: response.output_tokens,
                cost_nanodollars: response.cost_nanodollars,
                cost_is_estimate: response.cost_is_estimate,
            },
            Err(error) => JudgementProviderCallOutcome::Failed {
                provider_request_id: error
                    .context()
                    .and_then(|context| context.request_id.clone()),
                error_code: error.code().to_string(),
                error: error.to_string(),
            },
        };
        lock_unpoisoned(&self.calls).push(JudgementProviderCall {
            call_ref,
            sequence,
            provider,
            model,
            gateway_request_digest,
            started_at,
            finished_at,
            outcome,
        });
        result
    }
}

fn gateway_request_digest(request: &ChatRequest) -> String {
    fn put(hasher: &mut blake3::Hasher, bytes: &[u8]) {
        hasher.update(&(bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
    }
    fn put_optional_u32(hasher: &mut blake3::Hasher, value: Option<u32>) {
        match value {
            Some(value) => {
                hasher.update(&[1]);
                hasher.update(&value.to_be_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
    }
    fn put_optional_bool(hasher: &mut blake3::Hasher, value: Option<bool>) {
        match value {
            Some(value) => hasher.update(&[1, u8::from(value)]),
            None => hasher.update(&[0]),
        };
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(REQUEST_DIGEST_DOMAIN);
    put(&mut hasher, request.model.provider().as_bytes());
    put(&mut hasher, request.model.model_id().as_bytes());
    hasher.update(&(request.messages.len() as u64).to_be_bytes());
    for message in &request.messages {
        let role = match message.role {
            Role::System => b"system".as_slice(),
            Role::User => b"user".as_slice(),
            Role::Assistant => b"assistant".as_slice(),
        };
        put(&mut hasher, role);
        put(&mut hasher, message.content.as_bytes());
    }
    hasher.update(&request.temperature.to_bits().to_be_bytes());
    put_optional_u32(&mut hasher, request.max_tokens);
    hasher.update(&[u8::from(request.json_mode), u8::from(request.logprobs)]);
    put_optional_u32(&mut hasher, request.top_logprobs);
    match &request.reasoning {
        Some(reasoning) => {
            hasher.update(&[1]);
            put_optional_bool(&mut hasher, reasoning.enabled);
            let effort = reasoning.effort.as_ref().map(|effort| match effort {
                ReasoningEffort::Xhigh => b"xhigh".as_slice(),
                ReasoningEffort::High => b"high".as_slice(),
                ReasoningEffort::Medium => b"medium".as_slice(),
                ReasoningEffort::Low => b"low".as_slice(),
                ReasoningEffort::Minimal => b"minimal".as_slice(),
                ReasoningEffort::None => b"none".as_slice(),
            });
            match effort {
                Some(effort) => put(&mut hasher, effort),
                None => put(&mut hasher, b""),
            }
            put_optional_u32(&mut hasher, reasoning.max_tokens);
            put_optional_bool(&mut hasher, reasoning.exclude);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match request.prompt_cache_key.as_deref() {
        Some(key) => put(&mut hasher, key.as_bytes()),
        None => put(&mut hasher, b""),
    }
    hasher.finalize().to_hex().to_string()
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
