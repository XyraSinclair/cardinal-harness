//! End-to-end integration test for the Commander module.
//!
//! Seeds a realistic store, then exercises every code path: dashboard rendering,
//! review operations, proposal lifecycle, model rankings, aggregation queries,
//! and edge cases.

use cardinal_harness::commander::dashboard;
use cardinal_harness::commander::extract::extract_json;
use cardinal_harness::commander::store::*;

fn temp_store() -> CommanderStore {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("e2e_commander.sqlite");
    std::mem::forget(dir); // keep alive for test duration
    CommanderStore::new(path).expect("create store")
}

/// Seed a store with realistic multi-run data.
async fn seed_store(store: &CommanderStore) -> (i64, i64) {
    // Run 1: completed successfully
    let run1 = store
        .create_run(
            "harden the durability layer",
            "anthropic/claude-opus-4-6",
            "frontier",
            10_000_000_000,
        )
        .await
        .unwrap();

    // Tasks for run 1
    store
        .insert_task(
            run1,
            0,
            "audit-spiel-fsync",
            "Audit Spiel batch commit for missing fsync fences",
            Some("You are a storage durability expert."),
            "[\"crates/spiel/src/**/*.rs\"]",
            "Spiel is the durability layer — missing fsync means data loss on crash",
        )
        .await
        .unwrap();
    store
        .update_task_result(
            run1,
            "audit-spiel-fsync",
            true,
            1_500_000_000,
            Some("anthropic/claude-opus-4-6"),
            Some("The analysis found that Spiel's batch_commit() calls fdatasync but not after WAL header writes..."),
        )
        .await
        .unwrap();

    store
        .insert_task(
            run1,
            1,
            "trace-slate-compaction",
            "Trace Slate compaction for data integrity during concurrent writes",
            None,
            "[\"crates/slate/src/core/**/*.rs\"]",
            "Compaction must preserve referential integrity under concurrent mutation",
        )
        .await
        .unwrap();
    store
        .update_task_result(
            run1,
            "trace-slate-compaction",
            true,
            1_200_000_000,
            Some("google/gemini-3.1-pro"),
            Some("Compaction is mostly safe but the entity GC phase has a TOCTOU window..."),
        )
        .await
        .unwrap();

    store
        .insert_task(
            run1,
            2,
            "failure-mode-analysis",
            "Enumerate failure modes in the durability stack",
            None,
            "[\"crates/spiel/src/**/*.rs\", \"crates/slate/src/**/*.rs\"]",
            "Must understand what breaks before we can harden it",
        )
        .await
        .unwrap();
    store
        .update_task_result(run1, "failure-mode-analysis", false, 0, None, None)
        .await
        .unwrap();

    // Model rankings for run 1
    store
        .insert_model_ranking(
            run1,
            "audit-spiel-fsync",
            "anthropic/claude-opus-4-6",
            1,
            0.95,
        )
        .await
        .unwrap();
    store
        .insert_model_ranking(run1, "audit-spiel-fsync", "google/gemini-3.1-pro", 2, 0.78)
        .await
        .unwrap();
    store
        .insert_model_ranking(run1, "audit-spiel-fsync", "x-ai/grok-4.1-fast", 3, 0.65)
        .await
        .unwrap();
    store
        .insert_model_ranking(
            run1,
            "trace-slate-compaction",
            "google/gemini-3.1-pro",
            1,
            0.88,
        )
        .await
        .unwrap();
    store
        .insert_model_ranking(
            run1,
            "trace-slate-compaction",
            "anthropic/claude-opus-4-6",
            2,
            0.82,
        )
        .await
        .unwrap();

    // Proposals — spanning all priorities and categories
    store
        .insert_proposal(
            run1,
            "audit-spiel-fsync",
            "Add fsync fence after Spiel WAL header write",
            "The WAL header is written before batch data but no fdatasync() follows. \
             On power loss, the header could reference data that was never persisted. \
             Add fdatasync() after header write, before batch payload.",
            ProposalCategory::Safety,
            ProposalPriority::Critical,
            "[\"crates/spiel/src/batch.rs\", \"crates/spiel/src/wal.rs\"]",
            EstimatedEffort::Small,
        )
        .await
        .unwrap();

    store
        .insert_proposal(
            run1,
            "audit-spiel-fsync",
            "Replace DashMap in shoal::registry with SlateMap",
            "The DashMap is read without a fence after Spiel commit, creating a window \
             where stale state is visible to concurrent readers.",
            ProposalCategory::Refactor,
            ProposalPriority::High,
            "[\"crates/shoal/src/registry.rs\"]",
            EstimatedEffort::Medium,
        )
        .await
        .unwrap();

    store
        .insert_proposal(
            run1,
            "trace-slate-compaction",
            "Add epoch fence to Slate entity GC",
            "The entity garbage collector has a TOCTOU window where a concurrent write \
             can reference an entity that GC is about to delete. Use epoch-based \
             reclamation to close the window.",
            ProposalCategory::BugFix,
            ProposalPriority::High,
            "[\"crates/slate/src/core/gc.rs\"]",
            EstimatedEffort::Medium,
        )
        .await
        .unwrap();

    store
        .insert_proposal(
            run1,
            "trace-slate-compaction",
            "Batch Slate compaction file renames for atomicity",
            "Individual renames during compaction create intermediate states where some \
             files are new and some are old. Batch them via rename-to-temp + rename-all.",
            ProposalCategory::Performance,
            ProposalPriority::Medium,
            "[\"crates/slate/src/core/compaction.rs\"]",
            EstimatedEffort::Small,
        )
        .await
        .unwrap();

    store
        .insert_proposal(
            run1,
            "audit-spiel-fsync",
            "Document durability guarantees in DESIGN.md",
            "The current design doc doesn't specify fsync policy. Add a section.",
            ProposalCategory::Improvement,
            ProposalPriority::Low,
            "[\"docs/design/CONSISTENCY_MODEL.md\"]",
            EstimatedEffort::Trivial,
        )
        .await
        .unwrap();

    // Update run 1 costs
    store
        .update_run_costs(run1, 140_000_000, 6_210_000_000, 320_000_000, 2, 1)
        .await
        .unwrap();
    store
        .update_run_status(run1, RunStatus::Completed)
        .await
        .unwrap();

    // Run 2: budget exceeded
    let run2 = store
        .create_run(
            "optimize hot-path allocations in Steel",
            "anthropic/claude-opus-4-6",
            "balanced",
            500_000_000,
        )
        .await
        .unwrap();
    store
        .update_run_status(run2, RunStatus::BudgetExceeded)
        .await
        .unwrap();

    (run1, run2)
}

// =============================================================================
// Tests
// =============================================================================

#[tokio::test]
async fn test_full_store_lifecycle() {
    let store = temp_store();
    let (run1, run2) = seed_store(&store).await;

    // --- Verify runs ---
    let runs = store.list_runs().await.unwrap();
    assert_eq!(runs.len(), 2);
    // list_runs is DESC, so run2 first
    assert_eq!(runs[0].id, run2);
    assert_eq!(runs[0].status, RunStatus::BudgetExceeded);
    assert_eq!(runs[1].id, run1);
    assert_eq!(runs[1].status, RunStatus::Completed);
    assert_eq!(runs[1].tasks_completed, 2);
    assert_eq!(runs[1].tasks_failed, 1);

    // --- Verify tasks ---
    let tasks = store.get_tasks_for_run(run1).await.unwrap();
    assert_eq!(tasks.len(), 3);
    assert_eq!(tasks[0].task_id, "audit-spiel-fsync");
    assert_eq!(tasks[0].success, Some(true));
    assert_eq!(
        tasks[0].top_model.as_deref(),
        Some("anthropic/claude-opus-4-6")
    );
    assert_eq!(tasks[2].task_id, "failure-mode-analysis");
    assert_eq!(tasks[2].success, Some(false));

    // --- Verify proposals ---
    let all_proposals = store.list_proposals(None).await.unwrap();
    assert_eq!(all_proposals.len(), 5);

    // Priority ordering: critical first, then high, medium, low
    assert_eq!(all_proposals[0].priority, ProposalPriority::Critical);
    assert_eq!(all_proposals[1].priority, ProposalPriority::High);
    assert_eq!(all_proposals[2].priority, ProposalPriority::High);
    assert_eq!(all_proposals[3].priority, ProposalPriority::Medium);
    assert_eq!(all_proposals[4].priority, ProposalPriority::Low);

    // Filter by status
    let pending = store
        .list_proposals(Some(ProposalStatus::Pending))
        .await
        .unwrap();
    assert_eq!(pending.len(), 5);

    let accepted = store
        .list_proposals(Some(ProposalStatus::Accepted))
        .await
        .unwrap();
    assert_eq!(accepted.len(), 0);

    // --- Verify aggregates ---
    let total_spend = store.total_spend().await.unwrap();
    let run1_data = store.get_run(run1).await.unwrap();
    assert_eq!(total_spend, run1_data.total_cost_nanodollars);

    let total_proposals = store.total_proposals().await.unwrap();
    assert_eq!(total_proposals, 5);

    // --- Verify model rankings ---
    let rankings = store.get_model_rankings().await.unwrap();
    assert_eq!(rankings.len(), 5); // 3 for task1, 2 for task2
}

#[tokio::test]
async fn test_proposal_review_workflow() {
    let store = temp_store();
    let (_run1, _) = seed_store(&store).await;

    let proposals = store.list_proposals(None).await.unwrap();
    assert!(!proposals.is_empty());

    // Accept the critical one
    let critical = &proposals[0];
    assert_eq!(critical.priority, ProposalPriority::Critical);
    store
        .update_proposal_status(
            &critical.short_id,
            ProposalStatus::Accepted,
            Some("ship it"),
        )
        .await
        .unwrap();

    // Reject the low one
    let low = &proposals[4];
    assert_eq!(low.priority, ProposalPriority::Low);
    store
        .update_proposal_status(
            &low.short_id,
            ProposalStatus::Rejected,
            Some("not worth it"),
        )
        .await
        .unwrap();

    // Defer one of the highs
    let high = &proposals[1];
    assert_eq!(high.priority, ProposalPriority::High);
    store
        .update_proposal_status(&high.short_id, ProposalStatus::Deferred, None)
        .await
        .unwrap();

    // Verify filtered lists
    let pending = store
        .list_proposals(Some(ProposalStatus::Pending))
        .await
        .unwrap();
    assert_eq!(pending.len(), 2); // 1 high + 1 medium remaining

    let accepted = store
        .list_proposals(Some(ProposalStatus::Accepted))
        .await
        .unwrap();
    assert_eq!(accepted.len(), 1);
    assert_eq!(accepted[0].reviewer_notes.as_deref(), Some("ship it"));

    let rejected = store
        .list_proposals(Some(ProposalStatus::Rejected))
        .await
        .unwrap();
    assert_eq!(rejected.len(), 1);

    let deferred = store
        .list_proposals(Some(ProposalStatus::Deferred))
        .await
        .unwrap();
    assert_eq!(deferred.len(), 1);
    assert_eq!(deferred[0].reviewer_notes, None);

    // Verify get_proposal_by_short_id
    let fetched = store
        .get_proposal_by_short_id(&critical.short_id)
        .await
        .unwrap();
    assert_eq!(fetched.status, ProposalStatus::Accepted);
    assert_eq!(fetched.title, critical.title);

    // Verify not-found
    let result = store.get_proposal_by_short_id("nonexist").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_dashboard_renders_without_panic() {
    let store = temp_store();
    let _ = seed_store(&store).await;

    // This should not panic — exercises all dashboard code paths
    dashboard::render_dashboard(&store).await.unwrap();
}

#[tokio::test]
async fn test_dashboard_empty_store() {
    let store = temp_store();
    // Empty store should render gracefully
    dashboard::render_dashboard(&store).await.unwrap();
}

#[tokio::test]
async fn test_proposal_list_rendering() {
    let store = temp_store();
    let _ = seed_store(&store).await;

    let proposals = store.list_proposals(None).await.unwrap();
    // Should not panic
    dashboard::render_proposal_list(&proposals);

    // Empty list
    dashboard::render_proposal_list(&[]);
}

#[tokio::test]
async fn test_proposal_detail_rendering() {
    let store = temp_store();
    let _ = seed_store(&store).await;

    let proposals = store.list_proposals(None).await.unwrap();
    for p in &proposals {
        dashboard::render_proposal(&p);
    }
}

#[tokio::test]
async fn test_run_cost_breakdown() {
    let store = temp_store();
    let (run1, _) = seed_store(&store).await;

    let run = store.get_run(run1).await.unwrap();
    assert_eq!(run.decompose_cost_nanodollars, 140_000_000);
    assert_eq!(run.flywheel_cost_nanodollars, 6_210_000_000);
    assert_eq!(run.extract_cost_nanodollars, 320_000_000);
    let expected_total = 140_000_000i64 + 6_210_000_000 + 320_000_000;
    assert_eq!(run.total_cost_nanodollars, expected_total);
}

#[tokio::test]
async fn test_multiple_runs_accumulate_spend() {
    let store = temp_store();
    let run1 = store
        .create_run("run-a", "opus-4-6", "fast", 1_000_000_000)
        .await
        .unwrap();
    store
        .update_run_costs(run1, 100_000_000, 200_000_000, 50_000_000, 1, 0)
        .await
        .unwrap();

    let run2 = store
        .create_run("run-b", "opus-4-6", "fast", 1_000_000_000)
        .await
        .unwrap();
    store
        .update_run_costs(run2, 80_000_000, 150_000_000, 30_000_000, 2, 0)
        .await
        .unwrap();

    let total = store.total_spend().await.unwrap();
    // run1: 350M, run2: 260M = 610M
    assert_eq!(total, 610_000_000);
}

#[tokio::test]
async fn test_extract_json_edge_cases() {
    // Empty input
    assert_eq!(extract_json(""), "");

    // Whitespace only
    assert_eq!(extract_json("   "), "");

    // No JSON at all
    assert_eq!(extract_json("just some text"), "just some text");

    // JSON in markdown code fence
    let fenced = "```json\n{\"key\": \"value\"}\n```";
    let result = extract_json(fenced);
    assert!(result.starts_with('{'));
    assert!(result.contains("key"));

    // Multiple JSON objects — should return first complete one
    let multi = "{\"a\":1} {\"b\":2}";
    let result = extract_json(multi);
    assert_eq!(result, "{\"a\":1}");

    // Unclosed brace — fallback to full string
    let unclosed = "{\"a\": 1";
    let result = extract_json(unclosed);
    assert_eq!(result, unclosed);

    // Deeply nested
    let deep = r#"{"a": {"b": {"c": [1,2,3]}}}"#;
    assert_eq!(extract_json(deep), deep);
}

#[tokio::test]
async fn test_proposal_status_filter_all_variants() {
    let store = temp_store();
    let run_id = store
        .create_run("test", "opus", "fast", 1_000_000_000)
        .await
        .unwrap();

    // Insert proposals in each status
    let statuses = [
        ("pending-one", ProposalPriority::Medium),
        ("accepted-one", ProposalPriority::High),
        ("rejected-one", ProposalPriority::Low),
        ("deferred-one", ProposalPriority::Critical),
    ];

    for (title, priority) in &statuses {
        store
            .insert_proposal(
                run_id,
                "task-x",
                title,
                "desc",
                ProposalCategory::Improvement,
                *priority,
                "[]",
                EstimatedEffort::Small,
            )
            .await
            .unwrap();
    }

    let all = store.list_proposals(None).await.unwrap();
    assert_eq!(all.len(), 4);

    // Transition some
    store
        .update_proposal_status(&all[1].short_id, ProposalStatus::Accepted, None)
        .await
        .unwrap();
    store
        .update_proposal_status(&all[2].short_id, ProposalStatus::Rejected, Some("no"))
        .await
        .unwrap();
    store
        .update_proposal_status(&all[3].short_id, ProposalStatus::Deferred, None)
        .await
        .unwrap();

    // Verify each filter
    assert_eq!(
        store
            .list_proposals(Some(ProposalStatus::Pending))
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        store
            .list_proposals(Some(ProposalStatus::Accepted))
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        store
            .list_proposals(Some(ProposalStatus::Rejected))
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        store
            .list_proposals(Some(ProposalStatus::Deferred))
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        store
            .list_proposals(Some(ProposalStatus::Implemented))
            .await
            .unwrap()
            .len(),
        0
    );
}

#[tokio::test]
async fn test_unicode_in_proposals() {
    let store = temp_store();
    let run_id = store
        .create_run("日本語のディレクティブ", "opus", "fast", 1_000_000_000)
        .await
        .unwrap();

    store
        .insert_proposal(
            run_id,
            "タスク",
            "修正: UTF-8境界でのパニックを防ぐ",
            "マルチバイト文字を含む文字列の切り詰め処理を修正",
            ProposalCategory::BugFix,
            ProposalPriority::High,
            "[\"src/commander/dashboard.rs\"]",
            EstimatedEffort::Trivial,
        )
        .await
        .unwrap();

    let proposals = store.list_proposals(None).await.unwrap();
    assert_eq!(proposals.len(), 1);
    assert!(proposals[0].title.contains("UTF-8"));

    // Verify dashboard doesn't panic on unicode truncation
    dashboard::render_proposal_list(&proposals);
    dashboard::render_proposal(&proposals[0]);

    let run = store.get_run(run_id).await.unwrap();
    assert!(run.directive.contains("日本語"));
    dashboard::render_dashboard(&store).await.unwrap();
}

#[tokio::test]
async fn test_concurrent_store_access() {
    let store = temp_store();
    let run_id = store
        .create_run("concurrent test", "opus", "fast", 1_000_000_000)
        .await
        .unwrap();

    // Spawn multiple concurrent inserts
    let mut handles = Vec::new();
    for i in 0..10 {
        let s = store.clone();
        let handle = tokio::spawn(async move {
            s.insert_proposal(
                run_id,
                &format!("task-{i}"),
                &format!("Proposal {i}"),
                &format!("Description {i}"),
                ProposalCategory::Improvement,
                ProposalPriority::Medium,
                "[]",
                EstimatedEffort::Small,
            )
            .await
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    let proposals = store.list_proposals(None).await.unwrap();
    assert_eq!(proposals.len(), 10);
}

#[tokio::test]
async fn test_nonexistent_run() {
    let store = temp_store();
    let result = store.get_run(99999).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_tasks_for_nonexistent_run() {
    let store = temp_store();
    let tasks = store.get_tasks_for_run(99999).await.unwrap();
    assert!(tasks.is_empty());
}
