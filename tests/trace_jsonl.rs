use cardinal_harness::{ComparisonTrace, JsonlTraceSink, TraceSink};
use tempfile::tempdir;

#[derive(Debug, serde::Deserialize)]
struct TraceRow {
    comparison_index: usize,
}

fn make_trace(comparison_index: usize) -> ComparisonTrace {
    ComparisonTrace {
        timestamp_ms: 0,
        comparison_index,
        attribute_id: "clarity".to_string(),
        attribute_index: 0,
        attribute_prompt_hash: "attr_hash".to_string(),
        prompt_template_slug: "canonical_v2".to_string(),
        template_hash: "template_hash".to_string(),
        entity_a_id: "a".to_string(),
        entity_b_id: "b".to_string(),
        entity_a_index: 0,
        entity_b_index: 1,
        entity_a_hash: "a_hash".to_string(),
        entity_b_hash: "b_hash".to_string(),
        cache_key_hash: "key_hash".to_string(),
        model: "openai/gpt-5-mini".to_string(),
        higher_ranked: Some("A".to_string()),
        ratio: Some(2.0),
        confidence: Some(0.9),
        refused: false,
        cached: true,
        input_tokens: 0,
        output_tokens: 0,
        provider_cost_nanodollars: 0,
        error: None,
    }
}

#[test]
fn jsonl_trace_sink_writes_events_and_flushes_on_join() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("trace.jsonl");

    let (sink, worker) = JsonlTraceSink::new(&path).unwrap();
    sink.record(make_trace(1)).unwrap();
    sink.record(make_trace(2)).unwrap();

    drop(sink);
    worker.join().unwrap();

    let raw = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = raw.lines().collect();
    assert_eq!(lines.len(), 2);

    let first: TraceRow = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(first.comparison_index, 1);
}
