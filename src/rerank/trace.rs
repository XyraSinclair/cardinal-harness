//! Comparison trace capture for rerank runs.

use serde::Serialize;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonTrace {
    pub timestamp_ms: i64,
    pub comparison_index: usize,
    pub attribute_id: String,
    pub attribute_index: usize,
    pub attribute_prompt_hash: String,
    pub prompt_template_slug: String,
    pub template_hash: String,
    pub entity_a_id: String,
    pub entity_b_id: String,
    pub entity_a_index: usize,
    pub entity_b_index: usize,
    pub entity_a_hash: String,
    pub entity_b_hash: String,
    pub cache_key_hash: String,
    pub model: String,
    pub higher_ranked: Option<String>,
    pub ratio: Option<f64>,
    pub confidence: Option<f64>,
    pub refused: bool,
    pub cached: bool,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub provider_cost_nanodollars: i64,
    pub error: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum TraceError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(String),
    #[error("trace channel closed")]
    Closed,
    #[error("trace worker failed: {0}")]
    Join(String),
}

pub trait TraceSink: Send + Sync {
    fn record(&self, event: ComparisonTrace) -> Result<(), TraceError>;
}

#[derive(Clone)]
pub struct JsonlTraceSink {
    sender: mpsc::Sender<ComparisonTrace>,
}

pub struct TraceWorker {
    handle: Option<std::thread::JoinHandle<Result<(), TraceError>>>,
}

impl TraceWorker {
    pub fn join(mut self) -> Result<(), TraceError> {
        let handle = self.handle.take();
        match handle {
            Some(handle) => match handle.join() {
                Ok(result) => result,
                Err(_) => Err(TraceError::Join("trace worker panicked".to_string())),
            },
            None => Ok(()),
        }
    }
}

impl JsonlTraceSink {
    pub fn new(path: impl AsRef<Path>) -> Result<(Self, TraceWorker), TraceError> {
        let file = std::fs::File::create(path)?;
        let (sender, receiver) = mpsc::channel::<ComparisonTrace>();
        let handle = std::thread::spawn(move || write_trace_loop(file, receiver));
        Ok((
            Self { sender },
            TraceWorker {
                handle: Some(handle),
            },
        ))
    }
}

impl TraceSink for JsonlTraceSink {
    fn record(&self, event: ComparisonTrace) -> Result<(), TraceError> {
        self.sender.send(event).map_err(|_| TraceError::Closed)
    }
}

fn write_trace_loop(
    file: std::fs::File,
    receiver: mpsc::Receiver<ComparisonTrace>,
) -> Result<(), TraceError> {
    let mut writer = BufWriter::new(file);
    for event in receiver {
        let line = serde_json::to_string(&event).map_err(|e| TraceError::Serde(e.to_string()))?;
        writeln!(writer, "{line}")?;
    }
    writer.flush()?;
    Ok(())
}

pub fn now_epoch_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
