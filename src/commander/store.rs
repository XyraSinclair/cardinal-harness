//! SQLite-backed persistent store for Commander runs, tasks, proposals, and model rankings.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Semaphore;

// =============================================================================
// Types
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
    BudgetExceeded,
}

impl RunStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::BudgetExceeded => "budget_exceeded",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "running" => Self::Running,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "budget_exceeded" => Self::BudgetExceeded,
            _ => Self::Failed,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub id: i64,
    pub directive: String,
    pub commander_model: String,
    pub preset: String,
    pub budget_nanodollars: i64,
    pub status: RunStatus,
    pub decompose_cost_nanodollars: i64,
    pub flywheel_cost_nanodollars: i64,
    pub extract_cost_nanodollars: i64,
    pub total_cost_nanodollars: i64,
    pub tasks_completed: i64,
    pub tasks_failed: i64,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: i64,
    pub run_id: i64,
    pub task_index: i64,
    pub task_id: String,
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub context_globs: String, // JSON array
    pub rationale: String,
    pub success: Option<bool>,
    pub cost_nanodollars: i64,
    pub top_model: Option<String>,
    pub synthesis_content: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalCategory {
    BugFix,
    Refactor,
    Performance,
    Safety,
    Architecture,
    Improvement,
}

impl ProposalCategory {
    fn as_str(self) -> &'static str {
        match self {
            Self::BugFix => "bug_fix",
            Self::Refactor => "refactor",
            Self::Performance => "performance",
            Self::Safety => "safety",
            Self::Architecture => "architecture",
            Self::Improvement => "improvement",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "bug_fix" => Self::BugFix,
            "refactor" => Self::Refactor,
            "performance" => Self::Performance,
            "safety" => Self::Safety,
            "architecture" => Self::Architecture,
            "improvement" => Self::Improvement,
            _ => Self::Improvement,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalPriority {
    Critical,
    High,
    Medium,
    Low,
}

impl ProposalPriority {
    fn as_str(self) -> &'static str {
        match self {
            Self::Critical => "critical",
            Self::High => "high",
            Self::Medium => "medium",
            Self::Low => "low",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "critical" => Self::Critical,
            "high" => Self::High,
            "medium" => Self::Medium,
            "low" => Self::Low,
            _ => Self::Medium,
        }
    }

    pub fn sort_key(self) -> u8 {
        match self {
            Self::Critical => 0,
            Self::High => 1,
            Self::Medium => 2,
            Self::Low => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Pending,
    Accepted,
    Rejected,
    Deferred,
    Implemented,
}

impl ProposalStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
            Self::Deferred => "deferred",
            Self::Implemented => "implemented",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "pending" => Self::Pending,
            "accepted" => Self::Accepted,
            "rejected" => Self::Rejected,
            "deferred" => Self::Deferred,
            "implemented" => Self::Implemented,
            _ => Self::Pending,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimatedEffort {
    Trivial,
    Small,
    Medium,
    Large,
}

impl EstimatedEffort {
    fn as_str(self) -> &'static str {
        match self {
            Self::Trivial => "trivial",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "trivial" => Self::Trivial,
            "small" => Self::Small,
            "medium" => Self::Medium,
            "large" => Self::Large,
            _ => Self::Medium,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub id: i64,
    pub short_id: String,
    pub run_id: i64,
    pub task_id: String,
    pub title: String,
    pub description: String,
    pub category: ProposalCategory,
    pub priority: ProposalPriority,
    pub affected_files: String, // JSON array
    pub estimated_effort: EstimatedEffort,
    pub status: ProposalStatus,
    pub reviewer_notes: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRanking {
    pub id: i64,
    pub run_id: i64,
    pub task_id: String,
    pub model: String,
    pub rank: i64,
    pub utility: f64,
    pub created_at: i64,
}

/// An LLM trace record — every LLM input/output for full audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmTrace {
    pub id: i64,
    pub run_id: i64,
    pub phase: String,
    pub task_id: Option<String>,
    pub model: String,
    pub input_messages: String,
    pub raw_output: String,
    pub parsed_output: Option<String>,
    pub cost_nanodollars: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub latency_ms: i64,
    pub created_at: i64,
}

/// A reflection record — after-action review for a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reflection {
    pub id: i64,
    pub run_id: i64,
    pub quality_score: Option<f64>,
    pub summary: String,
    pub efficiency_analysis: String,
    pub recommendations: String,
    pub model_insights: Option<String>,
    pub cost_nanodollars: i64,
    pub created_at: i64,
}

/// Brief summary of a past run for briefing context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBrief {
    pub id: i64,
    pub directive: String,
    pub status: RunStatus,
    pub tasks_completed: i64,
    pub tasks_failed: i64,
    pub total_cost_nanodollars: i64,
    pub proposals_count: i64,
    pub created_at: i64,
}

/// Model performance summary across all runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceSummary {
    pub model: String,
    pub avg_rank: f64,
    pub win_count: i64,
    pub total_appearances: i64,
}

// =============================================================================
// Error
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("store lock poisoned")]
    Poisoned,
    #[error("task join error: {0}")]
    Join(String),
    #[error("not found: {0}")]
    NotFound(String),
}

// =============================================================================
// Store
// =============================================================================

/// Maximum retry attempts for short_id collision on insert.
const SHORT_ID_RETRIES: usize = 3;
/// Length of the short_id prefix (12 hex chars = 48 bits of entropy).
const SHORT_ID_LEN: usize = 12;

#[derive(Clone)]
pub struct CommanderStore {
    conn: Arc<Mutex<Connection>>,
    /// Gate concurrent spawn_blocking calls to prevent Tokio blocking pool starvation.
    /// Only one blocking thread waits on the mutex at a time.
    sem: Arc<Semaphore>,
}

impl CommanderStore {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?; // L1: propagate error
            }
        }
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;\
             PRAGMA synchronous=NORMAL;\
             PRAGMA foreign_keys=ON;\
             PRAGMA busy_timeout=5000;",
        )?;
        Self::create_tables(&conn)?;
        Self::migrate_schema(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            sem: Arc::new(Semaphore::new(1)),
        })
    }

    pub fn default_path() -> PathBuf {
        if let Ok(path) = std::env::var("CARDINAL_COMMANDER_STORE") {
            return PathBuf::from(path);
        }
        PathBuf::from(".cardinal_commander.sqlite")
    }

    /// Acquire the semaphore, then lock the connection.
    /// L2: Recover from mutex poisoning — the SQLite connection is still usable.
    fn with_conn<F, R>(&self, f: F) -> Result<R, StoreError>
    where
        F: FnOnce(&Connection) -> Result<R, StoreError>,
    {
        let guard = self
            .conn
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&guard)
    }

    fn create_tables(conn: &Connection) -> Result<(), StoreError> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS runs (\
               id INTEGER PRIMARY KEY AUTOINCREMENT,\
               directive TEXT NOT NULL,\
               commander_model TEXT NOT NULL,\
               preset TEXT NOT NULL,\
               budget_nanodollars INTEGER NOT NULL,\
               status TEXT NOT NULL DEFAULT 'running',\
               decompose_cost_nanodollars INTEGER NOT NULL DEFAULT 0,\
               flywheel_cost_nanodollars INTEGER NOT NULL DEFAULT 0,\
               extract_cost_nanodollars INTEGER NOT NULL DEFAULT 0,\
               total_cost_nanodollars INTEGER NOT NULL DEFAULT 0,\
               tasks_completed INTEGER NOT NULL DEFAULT 0,\
               tasks_failed INTEGER NOT NULL DEFAULT 0,\
               created_at INTEGER NOT NULL,\
               updated_at INTEGER NOT NULL\
             );\
             CREATE TABLE IF NOT EXISTS tasks (\
               id INTEGER PRIMARY KEY AUTOINCREMENT,\
               run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,\
               task_index INTEGER NOT NULL,\
               task_id TEXT NOT NULL,\
               prompt TEXT NOT NULL,\
               system_prompt TEXT,\
               context_globs TEXT NOT NULL DEFAULT '[]',\
               rationale TEXT NOT NULL DEFAULT '',\
               success INTEGER,\
               cost_nanodollars INTEGER NOT NULL DEFAULT 0,\
               top_model TEXT,\
               synthesis_content TEXT,\
               created_at INTEGER NOT NULL,\
               updated_at INTEGER NOT NULL,\
               UNIQUE(run_id, task_id)\
             );\
             CREATE TABLE IF NOT EXISTS proposals (\
               id INTEGER PRIMARY KEY AUTOINCREMENT,\
               short_id TEXT NOT NULL UNIQUE,\
               run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,\
               task_id TEXT NOT NULL,\
               title TEXT NOT NULL,\
               description TEXT NOT NULL,\
               category TEXT NOT NULL DEFAULT 'improvement',\
               priority TEXT NOT NULL DEFAULT 'medium',\
               affected_files TEXT NOT NULL DEFAULT '[]',\
               estimated_effort TEXT NOT NULL DEFAULT 'medium',\
               status TEXT NOT NULL DEFAULT 'pending',\
               reviewer_notes TEXT,\
               created_at INTEGER NOT NULL,\
               updated_at INTEGER NOT NULL\
             );\
             CREATE TABLE IF NOT EXISTS model_rankings (\
               id INTEGER PRIMARY KEY AUTOINCREMENT,\
               run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,\
               task_id TEXT NOT NULL,\
               model TEXT NOT NULL,\
               rank INTEGER NOT NULL,\
               utility REAL NOT NULL,\
               created_at INTEGER NOT NULL\
             );\
             CREATE TABLE IF NOT EXISTS llm_traces (\
               id INTEGER PRIMARY KEY AUTOINCREMENT,\
               run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,\
               phase TEXT NOT NULL,\
               task_id TEXT,\
               model TEXT NOT NULL,\
               input_messages TEXT NOT NULL,\
               raw_output TEXT NOT NULL,\
               parsed_output TEXT,\
               cost_nanodollars INTEGER NOT NULL DEFAULT 0,\
               input_tokens INTEGER NOT NULL DEFAULT 0,\
               output_tokens INTEGER NOT NULL DEFAULT 0,\
               latency_ms INTEGER NOT NULL DEFAULT 0,\
               created_at INTEGER NOT NULL\
             );\
             CREATE TABLE IF NOT EXISTS reflections (\
               id INTEGER PRIMARY KEY AUTOINCREMENT,\
               run_id INTEGER NOT NULL UNIQUE REFERENCES runs(id) ON DELETE CASCADE,\
               quality_score REAL,\
               summary TEXT NOT NULL,\
               efficiency_analysis TEXT NOT NULL,\
               recommendations TEXT NOT NULL,\
               model_insights TEXT,\
               cost_nanodollars INTEGER NOT NULL DEFAULT 0,\
               created_at INTEGER NOT NULL\
             );\
             CREATE INDEX IF NOT EXISTS idx_tasks_run_id ON tasks(run_id, task_index);\
             CREATE INDEX IF NOT EXISTS idx_proposals_run_id ON proposals(run_id);\
             CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status);\
             CREATE INDEX IF NOT EXISTS idx_rankings_run_task ON model_rankings(run_id, task_id, rank);\
             CREATE INDEX IF NOT EXISTS idx_llm_traces_run ON llm_traces(run_id, phase);\
             CREATE INDEX IF NOT EXISTS idx_reflections_run ON reflections(run_id);",
        )?;
        Ok(())
    }

    /// Backward-compatible schema migration — adds new columns to existing tables.
    /// Uses ignore-if-exists pattern so it's safe to call on every open.
    fn migrate_schema(conn: &Connection) -> Result<(), StoreError> {
        // Each ALTER TABLE will fail silently if the column already exists
        let migrations = [
            "ALTER TABLE runs ADD COLUMN briefing_cost_nanodollars INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE runs ADD COLUMN reflection_cost_nanodollars INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE runs ADD COLUMN codebase_context_hash TEXT",
            "ALTER TABLE runs ADD COLUMN output_dir TEXT",
        ];
        for sql in &migrations {
            // SQLite returns "duplicate column name" error if column exists — ignore it
            match conn.execute(sql, []) {
                Ok(_) => {}
                Err(rusqlite::Error::SqliteFailure(err, _))
                    if err.code == rusqlite::ErrorCode::Unknown || err.extended_code == 1 =>
                {
                    // Column already exists — this is expected
                }
                Err(e) => {
                    // Check error message as fallback for "duplicate column name"
                    let msg = e.to_string();
                    if !msg.contains("duplicate column name") {
                        return Err(StoreError::Sqlite(e));
                    }
                }
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Runs
    // -------------------------------------------------------------------------

    pub async fn create_run(
        &self,
        directive: &str,
        commander_model: &str,
        preset: &str,
        budget_nanodollars: i64,
    ) -> Result<i64, StoreError> {
        let store = self.clone();
        let directive = directive.to_string();
        let commander_model = commander_model.to_string();
        let preset = preset.to_string();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let now = now_epoch();
                conn.execute(
                    "INSERT INTO runs (directive, commander_model, preset, budget_nanodollars, \
                     status, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        directive,
                        commander_model,
                        preset,
                        budget_nanodollars,
                        RunStatus::Running.as_str(),
                        now,
                        now,
                    ],
                )?;
                Ok(conn.last_insert_rowid())
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    pub async fn update_run_status(
        &self,
        run_id: i64,
        status: RunStatus,
    ) -> Result<(), StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let rows = conn.execute(
                    "UPDATE runs SET status = ?1, updated_at = ?2 WHERE id = ?3",
                    params![status.as_str(), now_epoch(), run_id],
                )?;
                if rows == 0 {
                    return Err(StoreError::NotFound(format!("run {run_id}")));
                }
                Ok(())
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    pub async fn update_run_costs(
        &self,
        run_id: i64,
        decompose_cost: i64,
        flywheel_cost: i64,
        extract_cost: i64,
        tasks_completed: i64,
        tasks_failed: i64,
    ) -> Result<(), StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let total = decompose_cost
                    .saturating_add(flywheel_cost)
                    .saturating_add(extract_cost);
                let rows = conn.execute(
                    "UPDATE runs SET decompose_cost_nanodollars = ?1, \
                     flywheel_cost_nanodollars = ?2, extract_cost_nanodollars = ?3, \
                     total_cost_nanodollars = ?4, tasks_completed = ?5, tasks_failed = ?6, \
                     updated_at = ?7 WHERE id = ?8",
                    params![
                        decompose_cost,
                        flywheel_cost,
                        extract_cost,
                        total,
                        tasks_completed,
                        tasks_failed,
                        now_epoch(),
                        run_id,
                    ],
                )?;
                if rows == 0 {
                    return Err(StoreError::NotFound(format!("run {run_id}")));
                }
                Ok(())
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    pub async fn get_run(&self, run_id: i64) -> Result<Run, StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                conn.query_row(
                    "SELECT id, directive, commander_model, preset, budget_nanodollars, status, \
                     decompose_cost_nanodollars, flywheel_cost_nanodollars, extract_cost_nanodollars, \
                     total_cost_nanodollars, tasks_completed, tasks_failed, created_at, updated_at \
                     FROM runs WHERE id = ?1",
                    params![run_id],
                    |row| row_to_run(row),
                )
                .map_err(|e| match e {
                    rusqlite::Error::QueryReturnedNoRows => {
                        StoreError::NotFound(format!("run {run_id}"))
                    }
                    other => StoreError::Sqlite(other),
                })
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    pub async fn list_runs(&self) -> Result<Vec<Run>, StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, directive, commander_model, preset, budget_nanodollars, status, \
                     decompose_cost_nanodollars, flywheel_cost_nanodollars, extract_cost_nanodollars, \
                     total_cost_nanodollars, tasks_completed, tasks_failed, created_at, updated_at \
                     FROM runs ORDER BY id DESC",
                )?;
                let mut rows = stmt.query([])?;
                let mut runs = Vec::new();
                while let Some(row) = rows.next()? {
                    runs.push(row_to_run(row)?);
                }
                Ok(runs)
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    // -------------------------------------------------------------------------
    // Tasks
    // -------------------------------------------------------------------------

    pub async fn insert_task(
        &self,
        run_id: i64,
        task_index: i64,
        task_id: &str,
        prompt: &str,
        system_prompt: Option<&str>,
        context_globs: &str,
        rationale: &str,
    ) -> Result<i64, StoreError> {
        let store = self.clone();
        let task_id = task_id.to_string();
        let prompt = prompt.to_string();
        let system_prompt = system_prompt.map(String::from);
        let context_globs = context_globs.to_string();
        let rationale = rationale.to_string();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let now = now_epoch();
                conn.execute(
                    "INSERT INTO tasks (run_id, task_index, task_id, prompt, system_prompt, \
                     context_globs, rationale, created_at, updated_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    params![
                        run_id,
                        task_index,
                        task_id,
                        prompt,
                        system_prompt,
                        context_globs,
                        rationale,
                        now,
                        now,
                    ],
                )?;
                Ok(conn.last_insert_rowid())
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    pub async fn update_task_result(
        &self,
        run_id: i64,
        task_id: &str,
        success: bool,
        cost_nanodollars: i64,
        top_model: Option<&str>,
        synthesis_content: Option<&str>,
    ) -> Result<(), StoreError> {
        let store = self.clone();
        let task_id = task_id.to_string();
        let top_model = top_model.map(String::from);
        let synthesis_content = synthesis_content.map(String::from);
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let rows = conn.execute(
                    "UPDATE tasks SET success = ?1, cost_nanodollars = ?2, top_model = ?3, \
                     synthesis_content = ?4, updated_at = ?5 \
                     WHERE run_id = ?6 AND task_id = ?7",
                    params![
                        success as i64,
                        cost_nanodollars,
                        top_model,
                        synthesis_content,
                        now_epoch(),
                        run_id,
                        task_id,
                    ],
                )?;
                if rows == 0 {
                    return Err(StoreError::NotFound(format!(
                        "task {task_id} in run {run_id}"
                    )));
                }
                Ok(())
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    pub async fn get_tasks_for_run(&self, run_id: i64) -> Result<Vec<Task>, StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, run_id, task_index, task_id, prompt, system_prompt, context_globs, \
                     rationale, success, cost_nanodollars, top_model, synthesis_content, \
                     created_at, updated_at \
                     FROM tasks WHERE run_id = ?1 ORDER BY task_index",
                )?;
                let mut rows = stmt.query(params![run_id])?;
                let mut tasks = Vec::new();
                while let Some(row) = rows.next()? {
                    tasks.push(row_to_task(row)?);
                }
                Ok(tasks)
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    // -------------------------------------------------------------------------
    // Proposals
    // -------------------------------------------------------------------------

    pub async fn insert_proposal(
        &self,
        run_id: i64,
        task_id: &str,
        title: &str,
        description: &str,
        category: ProposalCategory,
        priority: ProposalPriority,
        affected_files: &str,
        estimated_effort: EstimatedEffort,
    ) -> Result<i64, StoreError> {
        let store = self.clone();
        let task_id = task_id.to_string();
        let title = title.to_string();
        let description = description.to_string();
        let affected_files = affected_files.to_string();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let now = now_epoch();
                // H6: Retry on short_id collision (12-char hex = 48 bits)
                for attempt in 0..SHORT_ID_RETRIES {
                    let short_id = uuid::Uuid::new_v4().to_string()[..SHORT_ID_LEN].to_string();
                    match conn.execute(
                        "INSERT INTO proposals (short_id, run_id, task_id, title, description, \
                         category, priority, affected_files, estimated_effort, status, \
                         created_at, updated_at) \
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                        params![
                            short_id,
                            run_id,
                            task_id,
                            title,
                            description,
                            category.as_str(),
                            priority.as_str(),
                            affected_files,
                            estimated_effort.as_str(),
                            ProposalStatus::Pending.as_str(),
                            now,
                            now,
                        ],
                    ) {
                        Ok(_) => return Ok(conn.last_insert_rowid()),
                        Err(rusqlite::Error::SqliteFailure(err, _))
                            if err.code == rusqlite::ErrorCode::ConstraintViolation
                                && attempt < SHORT_ID_RETRIES - 1 =>
                        {
                            continue; // Retry with new UUID
                        }
                        Err(e) => return Err(StoreError::Sqlite(e)),
                    }
                }
                Err(StoreError::NotFound(
                    "failed to generate unique proposal short_id after retries".to_string(),
                ))
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    pub async fn list_proposals(
        &self,
        status_filter: Option<ProposalStatus>,
    ) -> Result<Vec<Proposal>, StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let (sql, filter_val) = if let Some(status) = status_filter {
                    (
                        "SELECT id, short_id, run_id, task_id, title, description, category, \
                         priority, affected_files, estimated_effort, status, reviewer_notes, \
                         created_at, updated_at FROM proposals WHERE status = ?1 \
                         ORDER BY \
                           CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 \
                           WHEN 'medium' THEN 2 WHEN 'low' THEN 3 END, id",
                        Some(status.as_str().to_string()),
                    )
                } else {
                    (
                        "SELECT id, short_id, run_id, task_id, title, description, category, \
                         priority, affected_files, estimated_effort, status, reviewer_notes, \
                         created_at, updated_at FROM proposals \
                         ORDER BY \
                           CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 \
                           WHEN 'medium' THEN 2 WHEN 'low' THEN 3 END, id",
                        None,
                    )
                };

                let mut stmt = conn.prepare(sql)?;
                let mut rows = if let Some(ref val) = filter_val {
                    stmt.query(params![val])?
                } else {
                    stmt.query([])?
                };

                let mut proposals = Vec::new();
                while let Some(row) = rows.next()? {
                    proposals.push(row_to_proposal(row)?);
                }
                Ok(proposals)
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    pub async fn get_proposal_by_short_id(&self, short_id: &str) -> Result<Proposal, StoreError> {
        let store = self.clone();
        let short_id = short_id.to_string();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                conn.query_row(
                    "SELECT id, short_id, run_id, task_id, title, description, category, \
                     priority, affected_files, estimated_effort, status, reviewer_notes, \
                     created_at, updated_at FROM proposals WHERE short_id = ?1",
                    params![short_id],
                    |row| row_to_proposal(row),
                )
                .map_err(|e| match e {
                    rusqlite::Error::QueryReturnedNoRows => {
                        StoreError::NotFound(format!("proposal {short_id}"))
                    }
                    other => StoreError::Sqlite(other),
                })
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    pub async fn update_proposal_status(
        &self,
        short_id: &str,
        status: ProposalStatus,
        notes: Option<&str>,
    ) -> Result<(), StoreError> {
        let store = self.clone();
        let short_id = short_id.to_string();
        let notes = notes.map(String::from);
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let rows = conn.execute(
                    "UPDATE proposals SET status = ?1, reviewer_notes = ?2, updated_at = ?3 \
                     WHERE short_id = ?4",
                    params![status.as_str(), notes, now_epoch(), short_id],
                )?;
                if rows == 0 {
                    return Err(StoreError::NotFound(format!("proposal {short_id}")));
                }
                Ok(())
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    // -------------------------------------------------------------------------
    // Model Rankings
    // -------------------------------------------------------------------------

    pub async fn insert_model_ranking(
        &self,
        run_id: i64,
        task_id: &str,
        model: &str,
        rank: i64,
        utility: f64,
    ) -> Result<i64, StoreError> {
        let store = self.clone();
        let task_id = task_id.to_string();
        let model = model.to_string();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let now = now_epoch();
                conn.execute(
                    "INSERT INTO model_rankings (run_id, task_id, model, rank, utility, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![run_id, task_id, model, rank, utility, now],
                )?;
                Ok(conn.last_insert_rowid())
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    pub async fn get_model_rankings(&self) -> Result<Vec<ModelRanking>, StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, run_id, task_id, model, rank, utility, created_at \
                     FROM model_rankings ORDER BY run_id, task_id, rank",
                )?;
                let mut rows = stmt.query([])?;
                let mut rankings = Vec::new();
                while let Some(row) = rows.next()? {
                    rankings.push(row_to_ranking(row)?);
                }
                Ok(rankings)
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    // -------------------------------------------------------------------------
    // Aggregate queries
    // -------------------------------------------------------------------------

    pub async fn total_spend(&self) -> Result<i64, StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let total: i64 = conn.query_row(
                    "SELECT COALESCE(SUM(total_cost_nanodollars), 0) FROM runs",
                    [],
                    |row| row.get(0),
                )?;
                Ok(total)
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    pub async fn total_proposals(&self) -> Result<i64, StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let total: i64 =
                    conn.query_row("SELECT COUNT(*) FROM proposals", [], |row| row.get(0))?;
                Ok(total)
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    /// Update officer-specific cost columns on a run.
    pub async fn update_run_officer_costs(
        &self,
        run_id: i64,
        briefing_cost: i64,
        reflection_cost: i64,
        output_dir: Option<&str>,
    ) -> Result<(), StoreError> {
        let store = self.clone();
        let output_dir = output_dir.map(String::from);
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                conn.execute(
                    "UPDATE runs SET briefing_cost_nanodollars = ?1, \
                     reflection_cost_nanodollars = ?2, output_dir = ?3, \
                     updated_at = ?4 WHERE id = ?5",
                    params![
                        briefing_cost,
                        reflection_cost,
                        output_dir,
                        now_epoch(),
                        run_id
                    ],
                )?;
                Ok(())
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    // -------------------------------------------------------------------------
    // Cross-run intelligence queries (officer upgrade)
    // -------------------------------------------------------------------------

    /// Acceptance rate across last N runs (or all runs).
    /// Returns fraction of accepted proposals out of all non-pending proposals.
    pub async fn acceptance_rate(&self, last_n_runs: Option<i64>) -> Result<f64, StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let (reviewed, accepted) = if let Some(n) = last_n_runs {
                    let reviewed: i64 = conn.query_row(
                        "SELECT COUNT(*) FROM proposals WHERE status != 'pending' \
                         AND run_id IN (SELECT id FROM runs ORDER BY id DESC LIMIT ?1)",
                        params![n],
                        |row| row.get(0),
                    )?;
                    let accepted: i64 = conn.query_row(
                        "SELECT COUNT(*) FROM proposals WHERE status IN ('accepted', 'implemented') \
                         AND run_id IN (SELECT id FROM runs ORDER BY id DESC LIMIT ?1)",
                        params![n],
                        |row| row.get(0),
                    )?;
                    (reviewed, accepted)
                } else {
                    let reviewed: i64 = conn.query_row(
                        "SELECT COUNT(*) FROM proposals WHERE status != 'pending'",
                        [],
                        |row| row.get(0),
                    )?;
                    let accepted: i64 = conn.query_row(
                        "SELECT COUNT(*) FROM proposals WHERE status IN ('accepted', 'implemented')",
                        [],
                        |row| row.get(0),
                    )?;
                    (reviewed, accepted)
                };
                if reviewed == 0 {
                    Ok(0.0)
                } else {
                    Ok(accepted as f64 / reviewed as f64)
                }
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    /// Cost per accepted proposal (total spend / accepted count).
    /// Returns None if no proposals have been accepted.
    pub async fn cost_per_accepted(&self) -> Result<Option<f64>, StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let accepted: i64 = conn.query_row(
                    "SELECT COUNT(*) FROM proposals WHERE status IN ('accepted', 'implemented')",
                    [],
                    |row| row.get(0),
                )?;
                if accepted == 0 {
                    return Ok(None);
                }
                let total_spend: i64 = conn.query_row(
                    "SELECT COALESCE(SUM(total_cost_nanodollars), 0) FROM runs",
                    [],
                    |row| row.get(0),
                )?;
                Ok(Some(total_spend as f64 / accepted as f64))
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    /// Recent proposal titles + status for dedup detection.
    pub async fn recent_proposal_titles(
        &self,
        limit: i64,
    ) -> Result<Vec<(String, ProposalStatus)>, StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let mut stmt =
                    conn.prepare("SELECT title, status FROM proposals ORDER BY id DESC LIMIT ?1")?;
                let mut rows = stmt.query(params![limit])?;
                let mut results = Vec::new();
                while let Some(row) = rows.next()? {
                    let title: String = row.get(0)?;
                    let status = ProposalStatus::from_str(&row.get::<_, String>(1)?);
                    results.push((title, status));
                }
                Ok(results)
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    /// Model performance summary: model → (avg_rank, win_count, total_appearances).
    pub async fn model_performance_summary(
        &self,
    ) -> Result<Vec<ModelPerformanceSummary>, StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT model, AVG(rank) as avg_rank, \
                     SUM(CASE WHEN rank = 1 THEN 1 ELSE 0 END) as wins, \
                     COUNT(*) as appearances \
                     FROM model_rankings GROUP BY model ORDER BY avg_rank ASC",
                )?;
                let mut rows = stmt.query([])?;
                let mut results = Vec::new();
                while let Some(row) = rows.next()? {
                    results.push(ModelPerformanceSummary {
                        model: row.get(0)?,
                        avg_rank: row.get(1)?,
                        win_count: row.get(2)?,
                        total_appearances: row.get(3)?,
                    });
                }
                Ok(results)
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    /// Recent run summaries for briefing context.
    pub async fn recent_run_summaries(&self, limit: i64) -> Result<Vec<RunBrief>, StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT r.id, r.directive, r.status, r.tasks_completed, r.tasks_failed, \
                     r.total_cost_nanodollars, \
                     (SELECT COUNT(*) FROM proposals WHERE run_id = r.id) as proposals_count, \
                     r.created_at \
                     FROM runs r ORDER BY r.id DESC LIMIT ?1",
                )?;
                let mut rows = stmt.query(params![limit])?;
                let mut results = Vec::new();
                while let Some(row) = rows.next()? {
                    results.push(RunBrief {
                        id: row.get(0)?,
                        directive: row.get(1)?,
                        status: RunStatus::from_str(&row.get::<_, String>(2)?),
                        tasks_completed: row.get(3)?,
                        tasks_failed: row.get(4)?,
                        total_cost_nanodollars: row.get(5)?,
                        proposals_count: row.get(6)?,
                        created_at: row.get(7)?,
                    });
                }
                Ok(results)
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    // -------------------------------------------------------------------------
    // LLM Traces
    // -------------------------------------------------------------------------

    /// Store an LLM trace record.
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_llm_trace(
        &self,
        run_id: i64,
        phase: &str,
        task_id: Option<&str>,
        model: &str,
        input_messages: &str,
        raw_output: &str,
        parsed_output: Option<&str>,
        cost_nanodollars: i64,
        input_tokens: i64,
        output_tokens: i64,
        latency_ms: i64,
    ) -> Result<i64, StoreError> {
        let store = self.clone();
        let phase = phase.to_string();
        let task_id = task_id.map(String::from);
        let model = model.to_string();
        let input_messages = input_messages.to_string();
        let raw_output = raw_output.to_string();
        let parsed_output = parsed_output.map(String::from);
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let now = now_epoch();
                conn.execute(
                    "INSERT INTO llm_traces (run_id, phase, task_id, model, input_messages, \
                     raw_output, parsed_output, cost_nanodollars, input_tokens, output_tokens, \
                     latency_ms, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                    params![
                        run_id,
                        phase,
                        task_id,
                        model,
                        input_messages,
                        raw_output,
                        parsed_output,
                        cost_nanodollars,
                        input_tokens,
                        output_tokens,
                        latency_ms,
                        now,
                    ],
                )?;
                Ok(conn.last_insert_rowid())
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    /// Get all LLM traces for a run, optionally filtered by phase.
    pub async fn get_traces_for_run(
        &self,
        run_id: i64,
        phase_filter: Option<&str>,
    ) -> Result<Vec<LlmTrace>, StoreError> {
        let store = self.clone();
        let phase_filter = phase_filter.map(String::from);
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let mut traces = Vec::new();
                if let Some(ref phase) = phase_filter {
                    let mut stmt = conn.prepare(
                        "SELECT id, run_id, phase, task_id, model, input_messages, raw_output, \
                         parsed_output, cost_nanodollars, input_tokens, output_tokens, \
                         latency_ms, created_at \
                         FROM llm_traces WHERE run_id = ?1 AND phase = ?2 ORDER BY id",
                    )?;
                    let mut rows = stmt.query(params![run_id, phase])?;
                    while let Some(row) = rows.next()? {
                        traces.push(row_to_llm_trace(row)?);
                    }
                } else {
                    let mut stmt = conn.prepare(
                        "SELECT id, run_id, phase, task_id, model, input_messages, raw_output, \
                         parsed_output, cost_nanodollars, input_tokens, output_tokens, \
                         latency_ms, created_at \
                         FROM llm_traces WHERE run_id = ?1 ORDER BY id",
                    )?;
                    let mut rows = stmt.query(params![run_id])?;
                    while let Some(row) = rows.next()? {
                        traces.push(row_to_llm_trace(row)?);
                    }
                }
                Ok(traces)
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    // -------------------------------------------------------------------------
    // Reflections
    // -------------------------------------------------------------------------

    /// Store a reflection record for a run.
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_reflection(
        &self,
        run_id: i64,
        quality_score: Option<f64>,
        summary: &str,
        efficiency_analysis: &str,
        recommendations: &str,
        model_insights: Option<&str>,
        cost_nanodollars: i64,
    ) -> Result<i64, StoreError> {
        let store = self.clone();
        let summary = summary.to_string();
        let efficiency_analysis = efficiency_analysis.to_string();
        let recommendations = recommendations.to_string();
        let model_insights = model_insights.map(String::from);
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let now = now_epoch();
                conn.execute(
                    "INSERT INTO reflections (run_id, quality_score, summary, efficiency_analysis, \
                     recommendations, model_insights, cost_nanodollars, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        run_id, quality_score, summary, efficiency_analysis,
                        recommendations, model_insights, cost_nanodollars, now,
                    ],
                )?;
                Ok(conn.last_insert_rowid())
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    /// Get the reflection for a run, if one exists.
    pub async fn get_reflection(&self, run_id: i64) -> Result<Option<Reflection>, StoreError> {
        let store = self.clone();
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        tokio::task::spawn_blocking(move || {
            store.with_conn(|conn| {
                let result = conn.query_row(
                    "SELECT id, run_id, quality_score, summary, efficiency_analysis, \
                     recommendations, model_insights, cost_nanodollars, created_at \
                     FROM reflections WHERE run_id = ?1",
                    params![run_id],
                    |row| row_to_reflection(row),
                );
                match result {
                    Ok(r) => Ok(Some(r)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(e) => Err(StoreError::Sqlite(e)),
                }
            })
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }
}

// =============================================================================
// Row converters — fallible (H2: propagate decode errors instead of defaults)
// =============================================================================

fn row_to_run(row: &rusqlite::Row<'_>) -> rusqlite::Result<Run> {
    Ok(Run {
        id: row.get(0)?,
        directive: row.get(1)?,
        commander_model: row.get(2)?,
        preset: row.get(3)?,
        budget_nanodollars: row.get(4)?,
        status: RunStatus::from_str(&row.get::<_, String>(5)?),
        decompose_cost_nanodollars: row.get(6)?,
        flywheel_cost_nanodollars: row.get(7)?,
        extract_cost_nanodollars: row.get(8)?,
        total_cost_nanodollars: row.get(9)?,
        tasks_completed: row.get(10)?,
        tasks_failed: row.get(11)?,
        created_at: row.get(12)?,
        updated_at: row.get(13)?,
    })
}

fn row_to_task(row: &rusqlite::Row<'_>) -> rusqlite::Result<Task> {
    Ok(Task {
        id: row.get(0)?,
        run_id: row.get(1)?,
        task_index: row.get(2)?,
        task_id: row.get(3)?,
        prompt: row.get(4)?,
        system_prompt: row.get(5)?,
        context_globs: row.get::<_, String>(6).unwrap_or_else(|_| "[]".into()),
        rationale: row.get::<_, String>(7).unwrap_or_default(),
        success: row.get::<_, Option<i64>>(8)?.map(|v| v != 0),
        cost_nanodollars: row.get(9)?,
        top_model: row.get(10)?,
        synthesis_content: row.get(11)?,
        created_at: row.get(12)?,
        updated_at: row.get(13)?,
    })
}

fn row_to_proposal(row: &rusqlite::Row<'_>) -> rusqlite::Result<Proposal> {
    Ok(Proposal {
        id: row.get(0)?,
        short_id: row.get(1)?,
        run_id: row.get(2)?,
        task_id: row.get(3)?,
        title: row.get(4)?,
        description: row.get(5)?,
        category: ProposalCategory::from_str(&row.get::<_, String>(6)?),
        priority: ProposalPriority::from_str(&row.get::<_, String>(7)?),
        affected_files: row.get::<_, String>(8).unwrap_or_else(|_| "[]".into()),
        estimated_effort: EstimatedEffort::from_str(
            &row.get::<_, String>(9).unwrap_or_else(|_| "medium".into()),
        ),
        status: ProposalStatus::from_str(&row.get::<_, String>(10)?),
        reviewer_notes: row.get(11)?,
        created_at: row.get(12)?,
        updated_at: row.get(13)?,
    })
}

fn row_to_ranking(row: &rusqlite::Row<'_>) -> rusqlite::Result<ModelRanking> {
    Ok(ModelRanking {
        id: row.get(0)?,
        run_id: row.get(1)?,
        task_id: row.get(2)?,
        model: row.get(3)?,
        rank: row.get(4)?,
        utility: row.get(5)?,
        created_at: row.get(6)?,
    })
}

fn row_to_llm_trace(row: &rusqlite::Row<'_>) -> rusqlite::Result<LlmTrace> {
    Ok(LlmTrace {
        id: row.get(0)?,
        run_id: row.get(1)?,
        phase: row.get(2)?,
        task_id: row.get(3)?,
        model: row.get(4)?,
        input_messages: row.get(5)?,
        raw_output: row.get(6)?,
        parsed_output: row.get(7)?,
        cost_nanodollars: row.get(8)?,
        input_tokens: row.get(9)?,
        output_tokens: row.get(10)?,
        latency_ms: row.get(11)?,
        created_at: row.get(12)?,
    })
}

fn row_to_reflection(row: &rusqlite::Row<'_>) -> rusqlite::Result<Reflection> {
    Ok(Reflection {
        id: row.get(0)?,
        run_id: row.get(1)?,
        quality_score: row.get(2)?,
        summary: row.get(3)?,
        efficiency_analysis: row.get(4)?,
        recommendations: row.get(5)?,
        model_insights: row.get(6)?,
        cost_nanodollars: row.get(7)?,
        created_at: row.get(8)?,
    })
}

fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> CommanderStore {
        let dir = tempfile::tempdir().expect("failed to create tempdir");
        let path = dir.path().join("test_commander.sqlite");
        // Leak the TempDir so it persists for the test
        std::mem::forget(dir);
        CommanderStore::new(path).expect("create store")
    }

    #[tokio::test]
    async fn test_run_lifecycle() {
        let store = temp_store();

        let run_id = store
            .create_run("harden durability", "opus-4-6", "frontier", 5_000_000_000)
            .await
            .expect("create run");

        let run = store
            .get_run(run_id)
            .await
            .expect("fetching run should succeed");
        assert_eq!(run.directive, "harden durability");
        assert_eq!(run.status, RunStatus::Running);

        store
            .update_run_status(run_id, RunStatus::Completed)
            .await
            .expect("update status");

        let run = store.get_run(run_id).await.expect("get run after update");
        assert_eq!(run.status, RunStatus::Completed);
    }

    #[tokio::test]
    async fn test_task_insert_and_query() {
        let store = temp_store();
        let run_id = store
            .create_run("test", "opus-4-6", "fast", 1_000_000_000)
            .await
            .expect("create run");

        store
            .insert_task(
                run_id,
                0,
                "analyze-concurrency",
                "Find concurrency bugs",
                None,
                "[\"crates/slate/src/**/*.rs\"]",
                "Slate has known DashMap issues",
            )
            .await
            .expect("insert task");

        let tasks = store
            .get_tasks_for_run(run_id)
            .await
            .expect("fetching tasks should succeed");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].task_id, "analyze-concurrency");
    }

    #[tokio::test]
    async fn test_proposal_lifecycle() {
        let store = temp_store();
        let run_id = store
            .create_run("test", "opus-4-6", "fast", 1_000_000_000)
            .await
            .expect("create run");

        store
            .insert_proposal(
                run_id,
                "analyze-concurrency",
                "Replace DashMap with SlateMap in shoal::registry",
                "The DashMap is read without synchronization...",
                ProposalCategory::Safety,
                ProposalPriority::Critical,
                "[\"crates/shoal/src/registry.rs\"]",
                EstimatedEffort::Small,
            )
            .await
            .expect("insert proposal");

        let proposals = store
            .list_proposals(Some(ProposalStatus::Pending))
            .await
            .expect("list proposals");
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].priority, ProposalPriority::Critical);

        let short_id = proposals[0].short_id.clone();
        store
            .update_proposal_status(&short_id, ProposalStatus::Accepted, Some("ship it"))
            .await
            .expect("update proposal");

        let updated = store
            .get_proposal_by_short_id(&short_id)
            .await
            .expect("get updated proposal");
        assert_eq!(updated.status, ProposalStatus::Accepted);
        assert_eq!(updated.reviewer_notes.as_deref(), Some("ship it"));
    }

    #[tokio::test]
    async fn test_model_rankings() {
        let store = temp_store();
        let run_id = store
            .create_run("test", "opus-4-6", "fast", 1_000_000_000)
            .await
            .expect("create run");

        store
            .insert_model_ranking(run_id, "task-1", "anthropic/claude-opus-4-6", 1, 0.95)
            .await
            .expect("insert ranking");

        store
            .insert_model_ranking(run_id, "task-1", "google/gemini-3.1-pro", 2, 0.72)
            .await
            .expect("insert ranking 2");

        let rankings = store.get_model_rankings().await.expect("get rankings");
        assert_eq!(rankings.len(), 2);
        assert_eq!(rankings[0].rank, 1);
    }

    #[tokio::test]
    async fn test_update_nonexistent_run_returns_not_found() {
        let store = temp_store();
        let result = store.update_run_status(99999, RunStatus::Completed).await;
        assert!(matches!(result, Err(StoreError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_update_nonexistent_proposal_returns_not_found() {
        let store = temp_store();
        let result = store
            .update_proposal_status("nonexist", ProposalStatus::Accepted, None)
            .await;
        assert!(matches!(result, Err(StoreError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_short_id_is_12_chars() {
        let store = temp_store();
        let run_id = store
            .create_run("test", "opus", "fast", 1_000_000_000)
            .await
            .unwrap();
        store
            .insert_proposal(
                run_id,
                "t",
                "title",
                "desc",
                ProposalCategory::Improvement,
                ProposalPriority::Medium,
                "[]",
                EstimatedEffort::Small,
            )
            .await
            .unwrap();
        let proposals = store.list_proposals(None).await.unwrap();
        assert_eq!(proposals[0].short_id.len(), SHORT_ID_LEN);
    }

    #[tokio::test]
    async fn test_schema_migration_idempotent() {
        // Opening the same store twice should not fail (migration is additive)
        let dir = tempfile::tempdir().expect("failed to create tempdir");
        let path = dir.path().join("migration_test.sqlite");
        let _store1 = CommanderStore::new(&path).expect("first open");
        let _store2 = CommanderStore::new(&path).expect("second open (migration idempotent)");
        std::mem::forget(dir);
    }

    #[tokio::test]
    async fn test_acceptance_rate_no_reviews() {
        let store = temp_store();
        let rate = store.acceptance_rate(None).await.unwrap();
        assert_eq!(rate, 0.0);
    }

    #[tokio::test]
    async fn test_acceptance_rate_with_reviews() {
        let store = temp_store();
        let run_id = store
            .create_run("test", "opus", "fast", 1_000_000_000)
            .await
            .unwrap();

        // Insert 4 proposals
        for i in 0..4 {
            store
                .insert_proposal(
                    run_id,
                    "t",
                    &format!("Proposal {i}"),
                    "desc",
                    ProposalCategory::Improvement,
                    ProposalPriority::Medium,
                    "[]",
                    EstimatedEffort::Small,
                )
                .await
                .unwrap();
        }

        let proposals = store.list_proposals(None).await.unwrap();
        // Accept 2, reject 1, leave 1 pending
        store
            .update_proposal_status(&proposals[0].short_id, ProposalStatus::Accepted, None)
            .await
            .unwrap();
        store
            .update_proposal_status(&proposals[1].short_id, ProposalStatus::Accepted, None)
            .await
            .unwrap();
        store
            .update_proposal_status(&proposals[2].short_id, ProposalStatus::Rejected, None)
            .await
            .unwrap();

        // 2 accepted out of 3 reviewed (pending doesn't count)
        let rate = store.acceptance_rate(None).await.unwrap();
        assert!((rate - 2.0 / 3.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_cost_per_accepted() {
        let store = temp_store();
        // No accepted proposals
        assert_eq!(store.cost_per_accepted().await.unwrap(), None);

        let run_id = store
            .create_run("test", "opus", "fast", 1_000_000_000)
            .await
            .unwrap();
        store
            .update_run_costs(run_id, 100_000_000, 200_000_000, 50_000_000, 1, 0)
            .await
            .unwrap();

        store
            .insert_proposal(
                run_id,
                "t",
                "Good proposal",
                "desc",
                ProposalCategory::Improvement,
                ProposalPriority::High,
                "[]",
                EstimatedEffort::Small,
            )
            .await
            .unwrap();

        let proposals = store.list_proposals(None).await.unwrap();
        store
            .update_proposal_status(&proposals[0].short_id, ProposalStatus::Accepted, None)
            .await
            .unwrap();

        let cpa = store.cost_per_accepted().await.unwrap().unwrap();
        // Total spend: 350M, 1 accepted → 350M per accepted
        assert_eq!(cpa as i64, 350_000_000);
    }

    #[tokio::test]
    async fn test_recent_proposal_titles() {
        let store = temp_store();
        let run_id = store
            .create_run("test", "opus", "fast", 1_000_000_000)
            .await
            .unwrap();

        for i in 0..5 {
            store
                .insert_proposal(
                    run_id,
                    "t",
                    &format!("Proposal {i}"),
                    "desc",
                    ProposalCategory::Improvement,
                    ProposalPriority::Medium,
                    "[]",
                    EstimatedEffort::Small,
                )
                .await
                .unwrap();
        }

        let titles = store.recent_proposal_titles(3).await.unwrap();
        assert_eq!(titles.len(), 3);
        // Most recent first (DESC)
        assert_eq!(titles[0].0, "Proposal 4");
        assert_eq!(titles[0].1, ProposalStatus::Pending);
    }

    #[tokio::test]
    async fn test_model_performance_summary() {
        let store = temp_store();
        let run_id = store
            .create_run("test", "opus", "fast", 1_000_000_000)
            .await
            .unwrap();

        store
            .insert_model_ranking(run_id, "t1", "model-a", 1, 0.9)
            .await
            .unwrap();
        store
            .insert_model_ranking(run_id, "t1", "model-b", 2, 0.7)
            .await
            .unwrap();
        store
            .insert_model_ranking(run_id, "t2", "model-a", 2, 0.8)
            .await
            .unwrap();
        store
            .insert_model_ranking(run_id, "t2", "model-b", 1, 0.85)
            .await
            .unwrap();

        let summary = store.model_performance_summary().await.unwrap();
        assert_eq!(summary.len(), 2);
        // Both models have avg_rank 1.5 (one win each)
        for s in &summary {
            assert_eq!(s.win_count, 1);
            assert_eq!(s.total_appearances, 2);
            assert!((s.avg_rank - 1.5).abs() < 1e-6);
        }
    }

    #[tokio::test]
    async fn test_recent_run_summaries() {
        let store = temp_store();
        for i in 0..3 {
            let run_id = store
                .create_run(&format!("directive {i}"), "opus", "fast", 1_000_000_000)
                .await
                .unwrap();
            store
                .update_run_status(run_id, RunStatus::Completed)
                .await
                .unwrap();
        }

        let summaries = store.recent_run_summaries(2).await.unwrap();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].directive, "directive 2"); // most recent first
    }

    #[tokio::test]
    async fn test_llm_trace_lifecycle() {
        let store = temp_store();
        let run_id = store
            .create_run("test", "opus", "fast", 1_000_000_000)
            .await
            .unwrap();

        let trace_id = store
            .insert_llm_trace(
                run_id,
                "decompose",
                Some("task-1"),
                "anthropic/claude-opus-4-6",
                r#"[{"role":"user","content":"test"}]"#,
                "raw output here",
                Some("parsed json"),
                50_000_000,
                1000,
                500,
                2345,
            )
            .await
            .unwrap();
        assert!(trace_id > 0);

        store
            .insert_llm_trace(
                run_id,
                "extract",
                Some("task-1"),
                "anthropic/claude-opus-4-6",
                "input2",
                "output2",
                None,
                30_000_000,
                800,
                400,
                1500,
            )
            .await
            .unwrap();

        // Get all traces
        let all = store.get_traces_for_run(run_id, None).await.unwrap();
        assert_eq!(all.len(), 2);

        // Filter by phase
        let decompose_only = store
            .get_traces_for_run(run_id, Some("decompose"))
            .await
            .unwrap();
        assert_eq!(decompose_only.len(), 1);
        assert_eq!(decompose_only[0].phase, "decompose");
        assert_eq!(decompose_only[0].cost_nanodollars, 50_000_000);
    }

    #[tokio::test]
    async fn test_reflection_lifecycle() {
        let store = temp_store();
        let run_id = store
            .create_run("test", "opus", "fast", 1_000_000_000)
            .await
            .unwrap();

        // No reflection yet
        assert!(store.get_reflection(run_id).await.unwrap().is_none());

        store
            .insert_reflection(
                run_id,
                Some(0.75),
                "Good run overall",
                "Cost was reasonable",
                r#"["use more models"]"#,
                Some(r#"{"opus": "strong"}"#),
                40_000_000,
            )
            .await
            .unwrap();

        let refl = store.get_reflection(run_id).await.unwrap().unwrap();
        assert_eq!(refl.run_id, run_id);
        assert!((refl.quality_score.unwrap() - 0.75).abs() < 1e-6);
        assert_eq!(refl.summary, "Good run overall");
        assert_eq!(refl.cost_nanodollars, 40_000_000);
    }

    #[tokio::test]
    async fn test_update_run_officer_costs() {
        let store = temp_store();
        let run_id = store
            .create_run("test", "opus", "fast", 1_000_000_000)
            .await
            .unwrap();

        store
            .update_run_officer_costs(run_id, 50_000_000, 30_000_000, Some("/tmp/run_1"))
            .await
            .unwrap();

        // Verify via raw SQL (officer columns not in Run struct yet)
        let store2 = store.clone();
        let result = store2
            .with_conn(|conn| {
                let (bc, rc, od): (i64, i64, Option<String>) = conn.query_row(
                    "SELECT briefing_cost_nanodollars, reflection_cost_nanodollars, output_dir \
                 FROM runs WHERE id = ?1",
                    params![run_id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )?;
                Ok((bc, rc, od))
            })
            .unwrap();
        assert_eq!(result.0, 50_000_000);
        assert_eq!(result.1, 30_000_000);
        assert_eq!(result.2.as_deref(), Some("/tmp/run_1"));
    }
}
