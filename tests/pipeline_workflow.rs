use std::sync::{Arc, Mutex};
use std::time::Duration;

use cardinal_harness::gateway::{
    ChatGateway, ChatRequest, ChatResponse, FinishReason, ProviderError, Role,
};
use cardinal_harness::pipeline::{
    run_flywheel, run_pipeline, FlywheelManifest, FlywheelRunConfig, FlywheelTask,
    PipelineAttribute, PipelineRankConfig, PipelineRequest,
};
use tempfile::tempdir;

#[derive(Default)]
struct ScriptedGateway {
    callers: Mutex<Vec<String>>,
}

impl ScriptedGateway {
    fn callers(&self) -> Vec<String> {
        self.callers.lock().unwrap().clone()
    }

    fn record_call(&self, caller: &str) {
        self.callers.lock().unwrap().push(caller.to_string());
    }
}

#[async_trait::async_trait]
impl ChatGateway for ScriptedGateway {
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, ProviderError> {
        self.record_call(req.attribution.caller);

        match req.attribution.caller {
            "pipeline::generate" => scripted_generation(&req),
            "pipeline::rank" => scripted_pairwise_rank(&req),
            "pipeline::synthesize" => Ok(chat_response(
                "Merged answer grounded in BEST response",
                18,
                12,
                700,
            )),
            other => Err(ProviderError::provider(
                "scripted",
                format!("unexpected caller: {other}"),
                false,
            )),
        }
    }
}

fn scripted_generation(req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
    let prompt = user_message(req);
    if prompt.contains("FAIL_ALL") {
        return Err(ProviderError::provider(
            "scripted",
            "generation failed for requested task",
            false,
        ));
    }

    let content = match req.model.model_id() {
        "test/best" => "BEST: isolate the bug, add the guard, and ship the focused fix",
        "test/mid" => "MID: investigate the issue and add a reasonable patch",
        "test/worst" => "WORST: rewrite the whole subsystem without evidence",
        other => {
            return Err(ProviderError::provider(
                "scripted",
                format!("unexpected generation model: {other}"),
                false,
            ))
        }
    };

    Ok(chat_response(content, 10, 6, 100))
}

fn scripted_pairwise_rank(req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
    let user_content = user_message(req);
    let a_ctx = extract_between(user_content, "<entity_A_context>", "</entity_A_context>")
        .unwrap_or("")
        .trim();
    let b_ctx = extract_between(user_content, "<entity_B_context>", "</entity_B_context>")
        .unwrap_or("")
        .trim();

    let a_score = score_for_context(a_ctx);
    let b_score = score_for_context(b_ctx);

    let (higher, ratio) = if a_score >= b_score {
        let diff = (a_score - b_score).abs();
        ("A", if diff >= 2 { 4.0 } else { 1.7 })
    } else {
        let diff = (b_score - a_score).abs();
        ("B", if diff >= 2 { 4.0 } else { 1.7 })
    };

    Ok(chat_response(
        format!(r#"{{"higher_ranked":"{higher}","ratio":{ratio},"confidence":0.9}}"#),
        12,
        8,
        50,
    ))
}

fn user_message(req: &ChatRequest) -> &str {
    req.messages
        .iter()
        .find(|message| matches!(message.role, Role::User))
        .map(|message| message.content.as_str())
        .unwrap_or("")
}

fn extract_between<'a>(s: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let start_idx = s.find(start)? + start.len();
    let rest = &s[start_idx..];
    let end_idx = rest.find(end)?;
    Some(&rest[..end_idx])
}

fn score_for_context(ctx: &str) -> i32 {
    if ctx.contains("BEST") {
        3
    } else if ctx.contains("MID") {
        2
    } else if ctx.contains("WORST") {
        1
    } else {
        0
    }
}

fn chat_response(
    content: impl Into<String>,
    input_tokens: u32,
    output_tokens: u32,
    cost_nanodollars: i64,
) -> ChatResponse {
    ChatResponse {
        content: content.into(),
        reasoning: None,
        reasoning_tokens: None,
        input_tokens,
        output_tokens,
        cost_nanodollars,
        upstream_cost_nanodollars: None,
        latency: Duration::from_millis(1),
        finish_reason: FinishReason::Stop,
        output_logprobs: None,
        cache_read_tokens: None,
        cache_write_tokens: None,
    }
}

fn test_attribute() -> PipelineAttribute {
    PipelineAttribute {
        id: "quality".to_string(),
        prompt: "quality".to_string(),
        weight: 1.0,
    }
}

fn test_rank_config() -> PipelineRankConfig {
    PipelineRankConfig {
        k: 1,
        tolerated_error: 0.2,
        comparison_budget: Some(4),
        judge_model: Some("test/judge".to_string()),
    }
}

fn test_models() -> Vec<String> {
    vec![
        "test/best".to_string(),
        "test/mid".to_string(),
        "test/worst".to_string(),
    ]
}

#[tokio::test]
async fn run_pipeline_executes_generate_rank_and_synthesize() {
    let gateway = Arc::new(ScriptedGateway::default());
    let req = PipelineRequest {
        prompt: "Diagnose the issue and propose the best fix".to_string(),
        system_prompt: Some("Be concrete.".to_string()),
        models: test_models(),
        preset: None,
        context_files: vec![],
        max_context_tokens: None,
        attributes: vec![test_attribute()],
        synthesis_model: "test/synth".to_string(),
        synthesis_system_prompt: None,
        generation_temperature: 0.0,
        synthesis_temperature: 0.0,
        max_generation_tokens: 256,
        max_synthesis_tokens: 256,
        rank_config: test_rank_config(),
    };

    let session = run_pipeline(gateway.clone(), None, None, None, req, Vec::new())
        .await
        .expect("pipeline should succeed");

    assert_eq!(session.generations.len(), 3);
    assert_eq!(
        session.synthesis.content,
        "Merged answer grounded in BEST response"
    );
    assert_eq!(
        session
            .ranking
            .entities
            .iter()
            .find(|entity| entity.rank == Some(1))
            .map(|entity| entity.id.as_str()),
        Some("test/best")
    );
    assert!(session.ranking.meta.comparisons_used > 0);
    assert!(session.cost.total_cost_nanodollars > 0);

    let callers = gateway.callers();
    assert_eq!(
        callers
            .iter()
            .filter(|caller| caller.as_str() == "pipeline::generate")
            .count(),
        3
    );
    assert!(
        callers
            .iter()
            .filter(|caller| caller.as_str() == "pipeline::rank")
            .count()
            > 0
    );
    assert_eq!(
        callers
            .iter()
            .filter(|caller| caller.as_str() == "pipeline::synthesize")
            .count(),
        1
    );
}

#[tokio::test]
async fn run_flywheel_writes_artifacts_and_reports_failures() {
    let gateway = Arc::new(ScriptedGateway::default());
    let temp = tempdir().expect("tempdir");
    let out_dir = temp.path().join("sessions");
    let synth_dir = temp.path().join("synthesis");
    let trace_dir = temp.path().join("traces");
    std::fs::create_dir_all(&out_dir).expect("create session dir");
    std::fs::create_dir_all(&synth_dir).expect("create synthesis dir");
    std::fs::create_dir_all(&trace_dir).expect("create trace dir");

    let manifest = FlywheelManifest {
        tasks: vec![
            FlywheelTask {
                id: "task-ok".to_string(),
                prompt: "Ship the best fix".to_string(),
                system_prompt: None,
                extra_context_files: vec![],
                models: Some(test_models()),
                synthesis_model: Some("test/synth".to_string()),
            },
            FlywheelTask {
                id: "task-fail".to_string(),
                prompt: "FAIL_ALL".to_string(),
                system_prompt: None,
                extra_context_files: vec![],
                models: Some(test_models()),
                synthesis_model: Some("test/synth".to_string()),
            },
        ],
        preset: None,
        context_files: vec![],
        attributes: Some(vec![test_attribute()]),
        synthesis_model: Some("test/synth".to_string()),
        rank_config: Some(test_rank_config()),
        max_context_tokens: None,
    };

    let summary = run_flywheel(
        gateway,
        manifest,
        FlywheelRunConfig {
            cache: None,
            model_policy: None,
            out_dir: &out_dir,
            synthesis_out_dir: Some(&synth_dir),
            trace_dir: Some(&trace_dir),
            preset_override: None,
            parallel: 2,
            gates: Vec::new(),
        },
    )
    .await;

    assert_eq!(summary.tasks_completed, 1);
    assert_eq!(summary.tasks_failed, 1);
    assert!(summary.total_cost_nanodollars > 0);

    let ok = summary
        .task_summaries
        .iter()
        .find(|task| task.task_id == "task-ok")
        .expect("success summary");
    assert!(ok.success);
    assert_eq!(ok.top_model.as_deref(), Some("test/best"));

    let failed = summary
        .task_summaries
        .iter()
        .find(|task| task.task_id == "task-fail")
        .expect("failure summary");
    assert!(!failed.success);
    assert!(failed
        .error
        .as_deref()
        .is_some_and(|err| err.contains("All generations failed")));

    assert!(out_dir.join("task-ok.json").exists());
    assert!(!out_dir.join("task-fail.json").exists());
    assert!(synth_dir.join("task-ok.md").exists());
    assert!(trace_dir.join("task-ok.trace.jsonl").exists());
}

#[tokio::test]
async fn run_flywheel_marks_artifact_write_failures_as_task_failures() {
    let gateway = Arc::new(ScriptedGateway::default());
    let temp = tempdir().expect("tempdir");
    let blocked_out = temp.path().join("not-a-directory");
    let synth_dir = temp.path().join("synthesis");
    let trace_dir = temp.path().join("traces");
    std::fs::write(&blocked_out, "occupied").expect("create blocking file");
    std::fs::create_dir_all(&synth_dir).expect("create synthesis dir");
    std::fs::create_dir_all(&trace_dir).expect("create trace dir");

    let manifest = FlywheelManifest {
        tasks: vec![FlywheelTask {
            id: "task-ok".to_string(),
            prompt: "Ship the best fix".to_string(),
            system_prompt: None,
            extra_context_files: vec![],
            models: Some(test_models()),
            synthesis_model: Some("test/synth".to_string()),
        }],
        preset: None,
        context_files: vec![],
        attributes: Some(vec![test_attribute()]),
        synthesis_model: Some("test/synth".to_string()),
        rank_config: Some(test_rank_config()),
        max_context_tokens: None,
    };

    let summary = run_flywheel(
        gateway,
        manifest,
        FlywheelRunConfig {
            cache: None,
            model_policy: None,
            out_dir: &blocked_out,
            synthesis_out_dir: Some(&synth_dir),
            trace_dir: Some(&trace_dir),
            preset_override: None,
            parallel: 1,
            gates: Vec::new(),
        },
    )
    .await;

    assert_eq!(summary.tasks_completed, 0);
    assert_eq!(summary.tasks_failed, 1);
    assert_eq!(summary.task_summaries.len(), 1);
    assert!(summary.total_cost_nanodollars > 0);
    assert!(!blocked_out.join("task-ok.json").exists());

    let task = &summary.task_summaries[0];
    assert!(!task.success);
    assert!(task.cost_nanodollars > 0);
    assert!(task
        .error
        .as_deref()
        .is_some_and(|err| err.contains("failed to write session artifact")));
}
