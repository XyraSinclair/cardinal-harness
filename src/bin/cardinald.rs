use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::rejection::JsonRejection;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use cardinal_harness::gateway::openrouter::OpenRouterAdapter;
use cardinal_harness::gateway::{
    Attribution, ChatGateway, ChatRequest, ChatResponse, ErrorContext, GatewayConfig,
    NoopUsageSink, ProviderError, ProviderGateway,
};
use cardinal_harness::judgement_run::{
    execute_judgement_run_with_ref, JudgementCandidate, JudgementPrivacy, JudgementRunRecord,
    JudgementRunRequest, JudgementRunStore, JudgementRunTerminal, NormalizedJudgementRunRequest,
};
use cardinal_harness::landing::{land_completed_run, ClickHouseLanding};
use cardinal_harness::rerank::{RerankExecution, RerankRunOptions, RerankStopReason};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use uuid::Uuid;

const DEFAULT_ADDR: &str = "127.0.0.1:8093";
const DEFAULT_MAX_CONCURRENT_RUNS: usize = 4;
const DEFAULT_RUN_DIR: &str = ".cardinald/runs";
const MAX_ENTITIES: usize = 200;
const MAX_ENTITY_TEXT_BYTES: usize = 8192;
const MAX_AXIS_PROMPT_BYTES: usize = 4096;

#[derive(Clone)]
struct AppState {
    store: JudgementRunStore,
    semaphore: Arc<Semaphore>,
    clickhouse: Option<Arc<ClickHouseLanding>>,
}

#[derive(Debug, Deserialize)]
struct CreateRunRequest {
    entities: Vec<JudgementCandidate>,
    axis_key: String,
    axis_prompt: String,
    requested_k: usize,
    model: String,
    privacy: JudgementPrivacy,
    #[serde(default)]
    owner_scope: Option<String>,
    #[serde(default)]
    lens: Option<String>,
}

#[derive(Debug, Serialize)]
struct AcceptedRun {
    run_ref: String,
    status: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DaemonRunMetadata {
    run_ref: String,
    request: NormalizedJudgementRunRequest,
    owner_scope: String,
    lens: String,
    created_at: DateTime<Utc>,
    status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct GetRunResponse {
    run_ref: String,
    status: String,
    privacy: JudgementPrivacy,
    owner_scope: String,
    lens: String,
    axis_key: String,
    axis_prompt: String,
    model: String,
    created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response: Option<CompletedResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct CompletedResponse {
    scores: Vec<ApiScore>,
    stop_reason: &'static str,
    comparisons_used: usize,
    provider_input_tokens: u32,
    provider_output_tokens: u32,
    cost_nanodollars: i64,
    cost_is_estimate: bool,
}

#[derive(Debug, Serialize)]
struct ApiScore {
    entity_id: String,
    rank: usize,
    latent_mean: f64,
    latent_std: f64,
    z_score: f64,
    percentile: f64,
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: message.into(),
        }
    }

    fn not_found() -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: "run not found".to_string(),
        }
    }

    fn internal() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "internal server error".to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({ "error": self.message })),
        )
            .into_response()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let address: SocketAddr = env_or("CARDINALD_ADDR", DEFAULT_ADDR)
        .parse()
        .map_err(|error| format!("invalid CARDINALD_ADDR: {error}"))?;
    let max_concurrent =
        parse_positive_usize("CARDINALD_MAX_CONCURRENT_RUNS", DEFAULT_MAX_CONCURRENT_RUNS)?;
    let run_dir = PathBuf::from(env_or("CARDINALD_RUN_DIR", DEFAULT_RUN_DIR));
    fs::create_dir_all(&run_dir)?;
    let store = JudgementRunStore::new(run_dir);

    let clickhouse = std::env::var("CARDINALD_CLICKHOUSE_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| ClickHouseLanding::from_url(&value).map(Arc::new))
        .transpose()
        .map_err(|error| format!("invalid CARDINALD_CLICKHOUSE_URL: {error}"))?;
    if let Some(client) = clickhouse.as_deref() {
        client.replay_pending(&store).await;
    }

    let state = AppState {
        store,
        semaphore: Arc::new(Semaphore::new(max_concurrent)),
        clickhouse,
    };
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/runs", post(create_run))
        .route("/v1/runs/{run_ref}", get(get_run))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(address).await?;
    eprintln!("cardinald: listening on {address}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn healthz() -> &'static str {
    "ok"
}

async fn create_run(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<CreateRunRequest>, JsonRejection>,
) -> Result<(StatusCode, Json<AcceptedRun>), ApiError> {
    let Json(payload) = payload.map_err(|_| ApiError::bad_request("invalid JSON request body"))?;
    validate_caps(&payload)?;

    let owner_scope = payload.owner_scope.unwrap_or_default();
    match payload.privacy {
        JudgementPrivacy::Public if !owner_scope.is_empty() => {
            return Err(ApiError::bad_request(
                "owner_scope must be empty for public runs",
            ));
        }
        JudgementPrivacy::Private if owner_scope.trim().is_empty() => {
            return Err(ApiError::bad_request(
                "owner_scope must be nonblank for private runs",
            ));
        }
        JudgementPrivacy::Public | JudgementPrivacy::Private => {}
    }
    let lens = payload.lens.unwrap_or_else(|| "api".to_string());
    if lens.trim().is_empty() {
        return Err(ApiError::bad_request("lens must not be blank"));
    }

    let request = JudgementRunRequest {
        entities: payload.entities,
        axis_key: payload.axis_key,
        axis_prompt: payload.axis_prompt,
        requested_k: payload.requested_k,
        model: payload.model,
        privacy: payload.privacy,
    };
    let normalized = request
        .clone()
        .normalize()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;

    let provider_key = provider_key(&headers)?;
    let gateway = build_gateway(provider_key)
        .map_err(|_| ApiError::unauthorized("provider key is invalid"))?;
    let run_ref = state.store.allocate_run_ref();
    let metadata = DaemonRunMetadata {
        run_ref: run_ref.clone(),
        request: normalized,
        owner_scope,
        lens,
        created_at: Utc::now(),
        status: "running".to_string(),
        error: None,
    };
    persist_new_metadata(state.store.root(), &metadata).map_err(|error| {
        eprintln!("cardinald: could not allocate run metadata: {error}");
        ApiError::internal()
    })?;

    let task_state = state.clone();
    let task_metadata = metadata.clone();
    let task_run_ref = run_ref.clone();
    tokio::spawn(async move {
        let permit = match Arc::clone(&task_state.semaphore).acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                mark_metadata_failed(
                    &task_state.store,
                    task_metadata,
                    "run queue shut down before execution",
                );
                return;
            }
        };
        let failure_store = task_state.store.clone();
        let failure_metadata = task_metadata.clone();
        let failure_run_ref = task_run_ref.clone();
        let joined = tokio::task::spawn_blocking(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    eprintln!("cardinald: could not start run executor: {error}");
                    mark_metadata_failed(
                        &task_state.store,
                        task_metadata,
                        "judgement run executor could not start",
                    );
                    return;
                }
            };
            runtime.block_on(execute_queued_run(
                task_state,
                task_metadata,
                task_run_ref,
                request,
                gateway,
                permit,
            ));
        })
        .await;
        if joined.is_err() {
            eprintln!("cardinald: run {failure_run_ref} executor stopped unexpectedly");
            mark_metadata_failed(
                &failure_store,
                failure_metadata,
                "judgement run executor stopped unexpectedly",
            );
        }
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(AcceptedRun {
            run_ref,
            status: "running",
        }),
    ))
}

async fn execute_queued_run(
    state: AppState,
    metadata: DaemonRunMetadata,
    run_ref: String,
    request: JudgementRunRequest,
    gateway: Arc<dyn ChatGateway>,
    permit: tokio::sync::OwnedSemaphorePermit,
) {
    let seed = rand::random::<u64>();
    let execution = RerankExecution::new(gateway, Attribution::new("cardinald::run")).run_options(
        RerankRunOptions {
            rng_seed: Some(seed),
            cache_only: false,
        },
    );
    let result =
        execute_judgement_run_with_ref(request, run_ref.clone(), execution, &state.store).await;
    drop(permit);

    match result {
        Ok(record) => {
            update_metadata_from_record(&state.store, metadata.clone(), &record);
            if matches!(record.terminal, JudgementRunTerminal::Completed { .. }) {
                land_completed_run(
                    state.clickhouse.as_deref(),
                    &state.store,
                    &record,
                    &metadata.lens,
                    &metadata.owner_scope,
                )
                .await;
            }
        }
        Err(_) => match state.store.load(&run_ref) {
            Ok(record) => {
                update_metadata_from_record(&state.store, metadata, &record);
            }
            Err(_) => {
                eprintln!("cardinald: run {run_ref} failed without a terminal record");
                mark_metadata_failed(
                    &state.store,
                    metadata,
                    "judgement run failed before its terminal record was persisted",
                );
            }
        },
    }
}

async fn get_run(
    State(state): State<AppState>,
    AxumPath(run_ref): AxumPath<String>,
) -> Result<Json<GetRunResponse>, ApiError> {
    if !valid_run_ref(&run_ref) {
        return Err(ApiError::not_found());
    }
    let metadata = load_metadata(state.store.root(), &run_ref).map_err(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            ApiError::not_found()
        } else {
            eprintln!("cardinald: could not read run metadata for {run_ref}: {error}");
            ApiError::internal()
        }
    })?;

    let record_path = state.store.root().join(format!("{run_ref}.json"));
    if record_path.is_file() {
        let record = state.store.load(&run_ref).map_err(|error| {
            eprintln!("cardinald: could not load terminal run {run_ref}: {error}");
            ApiError::internal()
        })?;
        Ok(Json(project_terminal(metadata, record)))
    } else {
        Ok(Json(GetRunResponse {
            run_ref: metadata.run_ref,
            status: metadata.status,
            privacy: metadata.request.privacy,
            owner_scope: metadata.owner_scope,
            lens: metadata.lens,
            axis_key: metadata.request.axis_key,
            axis_prompt: metadata.request.axis_prompt,
            model: metadata.request.model,
            created_at: metadata.created_at,
            response: None,
            error: metadata.error,
        }))
    }
}

fn project_terminal(metadata: DaemonRunMetadata, record: JudgementRunRecord) -> GetRunResponse {
    let (status, response, error) = match &record.terminal {
        JudgementRunTerminal::Completed {
            stop_reason,
            response,
        } => (
            "completed".to_string(),
            Some(project_completed(&record, *stop_reason, response)),
            None,
        ),
        JudgementRunTerminal::Cancelled { response } => (
            "cancelled".to_string(),
            Some(project_completed(
                &record,
                RerankStopReason::Cancelled,
                response,
            )),
            None,
        ),
        JudgementRunTerminal::Failed { error } => ("failed".to_string(), None, Some(error.clone())),
    };
    GetRunResponse {
        run_ref: record.run_ref,
        status,
        privacy: record.request.privacy,
        owner_scope: metadata.owner_scope,
        lens: metadata.lens,
        axis_key: record.request.axis_key,
        axis_prompt: record.request.axis_prompt,
        model: record.request.model,
        created_at: metadata.created_at,
        response,
        error,
    }
}

fn project_completed(
    record: &JudgementRunRecord,
    stop_reason: RerankStopReason,
    response: &cardinal_harness::judgement_run::JudgementRunResponse,
) -> CompletedResponse {
    let scores = response
        .entities
        .iter()
        .enumerate()
        .map(|(position, entity)| ApiScore {
            entity_id: entity.id.clone(),
            rank: entity.rank.unwrap_or(position + 1),
            latent_mean: entity.attribute_score.latent_mean,
            latent_std: entity.attribute_score.latent_std,
            z_score: entity.attribute_score.z_score,
            percentile: entity.attribute_score.percentile,
        })
        .collect();
    CompletedResponse {
        scores,
        stop_reason: stop_reason_name(stop_reason),
        comparisons_used: record
            .comparison_trace
            .iter()
            .filter(|trace| trace.solver_observation.is_some())
            .count(),
        provider_input_tokens: record.usage.provider_input_tokens,
        provider_output_tokens: record.usage.provider_output_tokens,
        cost_nanodollars: record.usage.provider_cost_nanodollars,
        cost_is_estimate: record.usage.provider_cost_is_estimate,
    }
}

fn validate_caps(payload: &CreateRunRequest) -> Result<(), ApiError> {
    if payload.entities.len() > MAX_ENTITIES {
        return Err(ApiError::bad_request(format!(
            "entities must contain at most {MAX_ENTITIES} items"
        )));
    }
    if let Some(entity) = payload
        .entities
        .iter()
        .find(|entity| entity.text.len() > MAX_ENTITY_TEXT_BYTES)
    {
        return Err(ApiError::bad_request(format!(
            "entity text exceeds {MAX_ENTITY_TEXT_BYTES} bytes: {}",
            entity.id
        )));
    }
    if payload.axis_prompt.len() > MAX_AXIS_PROMPT_BYTES {
        return Err(ApiError::bad_request(format!(
            "axis_prompt exceeds {MAX_AXIS_PROMPT_BYTES} bytes"
        )));
    }
    Ok(())
}

fn provider_key(headers: &HeaderMap) -> Result<String, ApiError> {
    let header_key = match headers.get("x-provider-key") {
        Some(value) => Some(
            value
                .to_str()
                .map_err(|_| ApiError::unauthorized("x-provider-key is invalid"))?
                .to_string(),
        ),
        None => None,
    };
    header_key
        .filter(|value| !value.is_empty())
        .or_else(|| std::env::var("CARDINALD_OPENROUTER_KEY").ok())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ApiError::unauthorized("OpenRouter provider key is required"))
}

fn build_gateway(provider_key: String) -> Result<Arc<dyn ChatGateway>, ProviderError> {
    let base_url = env_or("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1");
    let timeout = std::env::var("OPENROUTER_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(120));
    let referer = std::env::var("OPENROUTER_REFERER").ok();
    let app_title = std::env::var("OPENROUTER_APP_TITLE").ok();
    let adapter = OpenRouterAdapter::with_config(
        provider_key.clone(),
        base_url,
        timeout,
        referer,
        app_title,
    )?;
    let gateway =
        ProviderGateway::with_config(adapter, Arc::new(NoopUsageSink), GatewayConfig::default());
    Ok(Arc::new(SecretScrubbingGateway {
        inner: gateway,
        secret: provider_key,
    }))
}

struct SecretScrubbingGateway<G> {
    inner: G,
    secret: String,
}

#[async_trait::async_trait]
impl<G: ChatGateway> ChatGateway for SecretScrubbingGateway<G> {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
        self.inner
            .chat(request)
            .await
            .map_err(|error| scrub_provider_error(error, &self.secret))
    }
}

fn scrub_provider_error(error: ProviderError, secret: &str) -> ProviderError {
    let scrub = |value: String| value.replace(secret, "[REDACTED]");
    let scrub_context = |context: Option<ErrorContext>| {
        context.map(|mut context| {
            context.provider_code = context.provider_code.map(&scrub);
            context.request_id = context.request_id.map(&scrub);
            context
        })
    };
    match error {
        ProviderError::BudgetExceeded {
            limit_usd,
            spend_usd,
            retry_after,
        } => ProviderError::BudgetExceeded {
            limit_usd,
            spend_usd,
            retry_after,
        },
        ProviderError::RateLimited {
            retry_after,
            limit_source,
            context,
        } => ProviderError::RateLimited {
            retry_after,
            limit_source,
            context: scrub_context(context),
        },
        ProviderError::InvalidRequest { message, context } => ProviderError::InvalidRequest {
            message: scrub(message),
            context: scrub_context(context),
        },
        ProviderError::Refused { message, context } => ProviderError::Refused {
            message: scrub(message),
            context: scrub_context(context),
        },
        ProviderError::Provider {
            provider,
            message,
            retryable,
            context,
        } => ProviderError::Provider {
            provider,
            message: scrub(message),
            retryable,
            context: scrub_context(context),
        },
        ProviderError::Timeout(duration, context) => {
            ProviderError::Timeout(duration, scrub_context(context))
        }
        ProviderError::Http(error) => ProviderError::Http(error),
        ProviderError::Config(message) => ProviderError::Config(scrub(message)),
    }
}

fn persist_new_metadata(root: &Path, metadata: &DaemonRunMetadata) -> io::Result<()> {
    fs::create_dir_all(root)?;
    let path = metadata_path(root, &metadata.run_ref);
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, metadata).map_err(io::Error::other)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    File::open(root)?.sync_all()
}

fn persist_metadata(root: &Path, metadata: &DaemonRunMetadata) -> io::Result<()> {
    fs::create_dir_all(root)?;
    let path = metadata_path(root, &metadata.run_ref);
    let mut temporary = tempfile::NamedTempFile::new_in(root)?;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        serde_json::to_writer(&mut writer, metadata).map_err(io::Error::other)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    File::open(root)?.sync_all()
}

fn load_metadata(root: &Path, run_ref: &str) -> io::Result<DaemonRunMetadata> {
    let reader = BufReader::new(File::open(metadata_path(root, run_ref))?);
    serde_json::from_reader(reader).map_err(io::Error::other)
}

fn metadata_path(root: &Path, run_ref: &str) -> PathBuf {
    root.join(format!("{run_ref}.cardinald.json"))
}

fn update_metadata_from_record(
    store: &JudgementRunStore,
    mut metadata: DaemonRunMetadata,
    record: &JudgementRunRecord,
) {
    metadata.status = record.terminal.status().to_string();
    metadata.error = match &record.terminal {
        JudgementRunTerminal::Failed { error } => Some(error.clone()),
        JudgementRunTerminal::Completed { .. } | JudgementRunTerminal::Cancelled { .. } => None,
    };
    if let Err(error) = persist_metadata(store.root(), &metadata) {
        eprintln!(
            "cardinald: could not update run metadata for {}: {error}",
            record.run_ref
        );
    }
}

fn mark_metadata_failed(store: &JudgementRunStore, mut metadata: DaemonRunMetadata, error: &str) {
    metadata.status = "failed".to_string();
    metadata.error = Some(error.to_string());
    if let Err(persistence_error) = persist_metadata(store.root(), &metadata) {
        eprintln!(
            "cardinald: could not persist failure metadata for {}: {persistence_error}",
            metadata.run_ref
        );
    }
}

fn valid_run_ref(value: &str) -> bool {
    let Some(suffix) = value.strip_prefix("jrun_") else {
        return false;
    };
    suffix.len() == 32
        && suffix.bytes().all(|byte| byte.is_ascii_hexdigit())
        && Uuid::parse_str(suffix).is_ok_and(|uuid| uuid.get_version_num() == 4)
}

fn stop_reason_name(reason: RerankStopReason) -> &'static str {
    match reason {
        RerankStopReason::ToleratedErrorMet => "tolerated_error_met",
        RerankStopReason::CertifiedStop => "certified_stop",
        RerankStopReason::BudgetExhausted => "budget_exhausted",
        RerankStopReason::LatencyBudgetExceeded => "latency_budget_exceeded",
        RerankStopReason::Cancelled => "cancelled",
        RerankStopReason::NoProposals => "no_proposals",
        RerankStopReason::NoNewPairs => "no_new_pairs",
    }
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn parse_positive_usize(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("{name} must be a positive integer")),
        Err(_) => Ok(default),
    }
}
