//! Minimal live chat completion through the Claude Code subscription adapter.
//!
//! Run with `cargo run --example claude_code_chat`.

use std::sync::Arc;

use cardinal_harness::gateway::claude_code::{ClaudeCodeAdapter, ClaudeCodeConfig};
use cardinal_harness::gateway::{
    Attribution, ChatModel, ChatRequest, Message, NoopUsageSink, ProviderGateway,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = ClaudeCodeAdapter::new(ClaudeCodeConfig {
        effort: Some("low".to_string()),
        ..ClaudeCodeConfig::default()
    });
    let gateway = ProviderGateway::claude_code(adapter, Arc::new(NoopUsageSink));
    let request = ChatRequest::new(
        ChatModel::claude_code("fable"),
        vec![
            Message::system("Answer the user exactly and without explanation."),
            Message::user("Reply with exactly: cardinal"),
        ],
        Attribution::new("example::claude_code_chat"),
    );

    let response = gateway.chat(request).await?;
    println!("content: {}", response.content);
    println!(
        "served_model: {}",
        response.served_model.as_deref().unwrap_or("unknown")
    );
    println!(
        "provider_request_id: {}",
        response.provider_request_id.as_deref().unwrap_or("unknown")
    );
    println!("input_tokens: {}", response.input_tokens);
    println!("output_tokens: {}", response.output_tokens);
    println!("cache_read_tokens: {:?}", response.cache_read_tokens);
    println!("cache_write_tokens: {:?}", response.cache_write_tokens);
    println!("cost_nanodollars: {}", response.cost_nanodollars);
    println!("cost_is_estimate: {}", response.cost_is_estimate);
    println!("latency_ms: {}", response.latency.as_millis());
    println!("finish_reason: {:?}", response.finish_reason);

    Ok(())
}
