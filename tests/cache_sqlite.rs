use cardinal_harness::cache::{CachedJudgement, PairwiseCacheKey, SqlitePairwiseCache};
use cardinal_harness::PairwiseCache;
use tempfile::tempdir;

#[derive(Debug, serde::Deserialize)]
struct ExportRow {
    key_hash: String,
    hit_count: i64,
    higher_ranked: Option<String>,
    ratio: Option<f64>,
    confidence: Option<f64>,
    refused: bool,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    provider_cost_nanodollars: Option<i64>,
}

#[tokio::test]
async fn sqlite_cache_put_get_and_export_increments_hit_count() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("cache.sqlite");
    let cache = SqlitePairwiseCache::new(&db_path).unwrap();

    let key = PairwiseCacheKey::new(
        "openai/gpt-5-mini",
        "canonical_v2",
        "template_hash",
        "clarity",
        "clarity of explanation",
        "a",
        "Entity A text",
        "b",
        "Entity B text",
    );

    let value = CachedJudgement {
        higher_ranked: Some("A".to_string()),
        ratio: Some(2.0),
        confidence: Some(0.9),
        refused: false,
        input_tokens: Some(10),
        output_tokens: Some(5),
        provider_cost_nanodollars: Some(123),
    };

    cache.put(&key, &value).await.unwrap();

    let hit1 = cache.get(&key).await.unwrap().unwrap();
    assert_eq!(hit1.higher_ranked.as_deref(), Some("A"));
    assert_eq!(hit1.ratio, Some(2.0));
    assert_eq!(hit1.confidence, Some(0.9));
    assert!(!hit1.refused);
    assert_eq!(hit1.input_tokens, Some(10));
    assert_eq!(hit1.output_tokens, Some(5));
    assert_eq!(hit1.provider_cost_nanodollars, Some(123));

    let _ = cache.get(&key).await.unwrap().unwrap();

    let export_path = dir.path().join("export.jsonl");
    cache.export_jsonl(&export_path).await.unwrap();

    let raw = std::fs::read_to_string(&export_path).unwrap();
    let rows: Vec<ExportRow> = raw
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();

    let row = rows
        .into_iter()
        .find(|r| r.key_hash == key.key_hash)
        .unwrap();

    assert_eq!(row.hit_count, 2);
    assert_eq!(row.higher_ranked.as_deref(), Some("A"));
    assert_eq!(row.ratio, Some(2.0));
    assert_eq!(row.confidence, Some(0.9));
    assert!(!row.refused);
    assert_eq!(row.input_tokens, Some(10));
    assert_eq!(row.output_tokens, Some(5));
    assert_eq!(row.provider_cost_nanodollars, Some(123));
}

#[tokio::test]
async fn sqlite_cache_prune_max_rows_keeps_most_recent() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("cache.sqlite");
    let cache = SqlitePairwiseCache::new(&db_path).unwrap();

    let old_key = PairwiseCacheKey::new(
        "openai/gpt-5-mini",
        "canonical_v2",
        "template_hash",
        "clarity",
        "clarity of explanation",
        "a",
        "Entity A text",
        "b",
        "Entity B text",
    );
    let new_key = PairwiseCacheKey::new(
        "openai/gpt-5-mini",
        "canonical_v2",
        "template_hash",
        "clarity",
        "clarity of explanation",
        "c",
        "Entity C text",
        "d",
        "Entity D text",
    );
    let value = CachedJudgement {
        higher_ranked: Some("A".to_string()),
        ratio: Some(2.0),
        confidence: Some(0.9),
        refused: false,
        input_tokens: None,
        output_tokens: None,
        provider_cost_nanodollars: None,
    };

    cache.put(&old_key, &value).await.unwrap();
    cache.put(&new_key, &value).await.unwrap();

    // Make the "old" row deterministically older so prune ordering is stable.
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute(
        "UPDATE pairwise_cache SET updated_at = 0 WHERE key_hash = ?1",
        rusqlite::params![old_key.key_hash],
    )
    .unwrap();

    let stats = cache.prune(None, Some(1)).await.unwrap();
    assert_eq!(stats.remaining, 1);
    assert_eq!(stats.deleted, 1);

    let export_path = dir.path().join("export.jsonl");
    cache.export_jsonl(&export_path).await.unwrap();

    let raw = std::fs::read_to_string(&export_path).unwrap();
    let rows: Vec<ExportRow> = raw
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].key_hash, new_key.key_hash);
}

#[test]
fn sqlite_cache_lock_does_not_truncate_lockfile() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("cache.sqlite");
    let cache = SqlitePairwiseCache::new(&db_path).unwrap();

    let mut lock_path = db_path.clone();
    lock_path.set_extension("lock");
    std::fs::write(&lock_path, "keep").unwrap();

    let lock = cache.lock_exclusive().unwrap();
    drop(lock);

    let contents = std::fs::read_to_string(&lock_path).unwrap();
    assert_eq!(contents, "keep");
}
