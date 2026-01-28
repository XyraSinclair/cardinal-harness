//! SQLite-backed cache for pairwise LLM judgements.

use async_trait::async_trait;
use blake3;
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct PairwiseCacheKey {
    pub model: String,
    pub prompt_template_slug: String,
    pub attribute_id: String,
    pub attribute_prompt_hash: String,
    pub entity_a_id: String,
    pub entity_b_id: String,
    pub entity_a_hash: String,
    pub entity_b_hash: String,
    pub key_hash: String,
}

impl PairwiseCacheKey {
    pub fn new(
        model: &str,
        prompt_template_slug: &str,
        attribute_id: &str,
        attribute_prompt: &str,
        entity_a_id: &str,
        entity_a_text: &str,
        entity_b_id: &str,
        entity_b_text: &str,
    ) -> Self {
        let attribute_prompt_hash = hash_text(attribute_prompt);
        let entity_a_hash = hash_text(entity_a_text);
        let entity_b_hash = hash_text(entity_b_text);
        let key_hash = hash_fields(&[
            model,
            prompt_template_slug,
            attribute_id,
            &attribute_prompt_hash,
            entity_a_id,
            &entity_a_hash,
            entity_b_id,
            &entity_b_hash,
        ]);

        Self {
            model: model.to_string(),
            prompt_template_slug: prompt_template_slug.to_string(),
            attribute_id: attribute_id.to_string(),
            attribute_prompt_hash,
            entity_a_id: entity_a_id.to_string(),
            entity_b_id: entity_b_id.to_string(),
            entity_a_hash,
            entity_b_hash,
            key_hash,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CachedJudgement {
    pub higher_ranked: Option<String>,
    pub ratio: Option<f64>,
    pub confidence: Option<f64>,
    pub refused: bool,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub provider_cost_nanodollars: Option<i64>,
}

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("cache lock poisoned")]
    Poisoned,
    #[error("task join error: {0}")]
    Join(String),
}

#[async_trait]
pub trait PairwiseCache: Send + Sync {
    async fn get(&self, key: &PairwiseCacheKey) -> Result<Option<CachedJudgement>, CacheError>;
    async fn put(&self, key: &PairwiseCacheKey, value: &CachedJudgement)
        -> Result<(), CacheError>;
}

#[derive(Clone)]
pub struct SqlitePairwiseCache {
    path: PathBuf,
    conn: Arc<Mutex<Connection>>,
}

impl SqlitePairwiseCache {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, CacheError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(&path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;\
             PRAGMA synchronous=NORMAL;\
             CREATE TABLE IF NOT EXISTS pairwise_cache (\
               key_hash TEXT PRIMARY KEY,\
               model TEXT NOT NULL,\
               prompt_template_slug TEXT NOT NULL,\
               attribute_id TEXT NOT NULL,\
               attribute_prompt_hash TEXT NOT NULL,\
               entity_a_id TEXT NOT NULL,\
               entity_b_id TEXT NOT NULL,\
               entity_a_hash TEXT NOT NULL,\
               entity_b_hash TEXT NOT NULL,\
               higher_ranked TEXT,\
               ratio REAL,\
               confidence REAL,\
               refused INTEGER NOT NULL,\
               input_tokens INTEGER,\
               output_tokens INTEGER,\
               provider_cost_nanodollars INTEGER,\
               created_at INTEGER NOT NULL,\
               updated_at INTEGER NOT NULL,\
               hit_count INTEGER NOT NULL DEFAULT 0\
             );",
        )?;

        Ok(Self {
            path,
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn default_path() -> PathBuf {
        if let Ok(path) = std::env::var("CARDINAL_CACHE_PATH") {
            return PathBuf::from(path);
        }
        PathBuf::from(".cardinal_pairwise_cache.sqlite")
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn with_conn<F, R>(&self, f: F) -> Result<R, CacheError>
    where
        F: FnOnce(&Connection) -> Result<R, CacheError>,
    {
        let guard = self.conn.lock().map_err(|_| CacheError::Poisoned)?;
        f(&guard)
    }
}

#[async_trait]
impl PairwiseCache for SqlitePairwiseCache {
    async fn get(&self, key: &PairwiseCacheKey) -> Result<Option<CachedJudgement>, CacheError> {
        let key_hash = key.key_hash.clone();
        let conn = self.clone();
        tokio::task::spawn_blocking(move || {
            conn.with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT higher_ranked, ratio, confidence, refused, input_tokens, output_tokens,\
                            provider_cost_nanodollars\
                     FROM pairwise_cache WHERE key_hash = ?1",
                )?;
                let mut rows = stmt.query(params![key_hash])?;
                if let Some(row) = rows.next()? {
                    let entry = CachedJudgement {
                        higher_ranked: row.get::<_, Option<String>>(0)?,
                        ratio: row.get::<_, Option<f64>>(1)?,
                        confidence: row.get::<_, Option<f64>>(2)?,
                        refused: row.get::<_, i64>(3)? != 0,
                        input_tokens: row.get::<_, Option<i64>>(4)?.map(|v| v as u32),
                        output_tokens: row.get::<_, Option<i64>>(5)?.map(|v| v as u32),
                        provider_cost_nanodollars: row.get::<_, Option<i64>>(6)?,
                    };
                    conn.execute(
                        "UPDATE pairwise_cache\
                         SET hit_count = hit_count + 1, updated_at = ?1\
                         WHERE key_hash = ?2",
                        params![now_epoch(), key_hash],
                    )?;
                    Ok(Some(entry))
                } else {
                    Ok(None)
                }
            })
        })
        .await
        .map_err(|e| CacheError::Join(e.to_string()))?
    }

    async fn put(
        &self,
        key: &PairwiseCacheKey,
        value: &CachedJudgement,
    ) -> Result<(), CacheError> {
        let key = key.clone();
        let value = value.clone();
        let conn = self.clone();
        tokio::task::spawn_blocking(move || {
            conn.with_conn(|conn| {
                let now = now_epoch();
                conn.execute(
                    "INSERT INTO pairwise_cache (\
                        key_hash, model, prompt_template_slug, attribute_id, attribute_prompt_hash,\
                        entity_a_id, entity_b_id, entity_a_hash, entity_b_hash,\
                        higher_ranked, ratio, confidence, refused,\
                        input_tokens, output_tokens, provider_cost_nanodollars,\
                        created_at, updated_at\
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)\
                     ON CONFLICT(key_hash) DO UPDATE SET\
                        higher_ranked = excluded.higher_ranked,\
                        ratio = excluded.ratio,\
                        confidence = excluded.confidence,\
                        refused = excluded.refused,\
                        input_tokens = excluded.input_tokens,\
                        output_tokens = excluded.output_tokens,\
                        provider_cost_nanodollars = excluded.provider_cost_nanodollars,\
                        updated_at = excluded.updated_at",
                    params![
                        key.key_hash,
                        key.model,
                        key.prompt_template_slug,
                        key.attribute_id,
                        key.attribute_prompt_hash,
                        key.entity_a_id,
                        key.entity_b_id,
                        key.entity_a_hash,
                        key.entity_b_hash,
                        value.higher_ranked,
                        value.ratio,
                        value.confidence,
                        if value.refused { 1 } else { 0 },
                        value.input_tokens.map(|v| v as i64),
                        value.output_tokens.map(|v| v as i64),
                        value.provider_cost_nanodollars,
                        now,
                        now,
                    ],
                )?;
                Ok(())
            })
        })
        .await
        .map_err(|e| CacheError::Join(e.to_string()))?
    }
}

fn hash_text(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex().to_string()
}

fn hash_fields(fields: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new();
    for (idx, field) in fields.iter().enumerate() {
        if idx > 0 {
            hasher.update(b"|");
        }
        hasher.update(field.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
