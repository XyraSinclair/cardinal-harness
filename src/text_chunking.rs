//! Semantic-aware text chunking for embedding generation.
//!
//! This module provides token-aware, structure-aware chunking that respects
//! paragraph and sentence boundaries. The canonical implementation is now in
//! `continual_scraping/src/chunking.rs`.
//!
//! # Example
//!
//! ```rust,ignore
//! use exopriors_api::text_chunking::{chunk_text_by_tokens, ChunkingParams};
//!
//! let params = ChunkingParams::default(); // target=300, min=180, max=480, overlap=60
//! let chunks = chunk_text_by_tokens("Your long text here...", &params);
//!
//! for chunk in chunks {
//!     println!("Text: {} chars, {} tokens", chunk.text.len(), chunk.tokens);
//!     println!("Position: {}..{}", chunk.char_start, chunk.char_end);
//! }
//! ```
//!
//! # Algorithm
//!
//! 1. **Paragraph splitting**: First split on double newlines (`\n\s*\n+`)
//! 2. **Sentence splitting**: For paragraphs exceeding `max_tokens`, split on sentence boundaries
//! 3. **Token windowing**: For very long sentences, fall back to token-boundary windows
//! 4. **Greedy packing**: Pack units into chunks up to `max_tokens`
//! 5. **Overlap**: Maintain `overlap_tokens` between consecutive chunks
//! 6. **Tail merge**: Avoid tiny final chunks by merging with previous

use fancy_regex::Regex as FancyRegex;
use once_cell::sync::Lazy;
use regex::Regex;
use tiktoken_rs::cl100k_base;

/// Default configuration matching Python: target=300, 20% overlap
pub const DEFAULT_CHUNK_TOKENS: usize = 300;
pub const DEFAULT_OVERLAP_RATIO: f64 = 0.20;
pub const MIN_PAYLOAD_LENGTH: usize = 100;

// Lazy-initialized regex patterns
static PARAGRAPH_SPLIT: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\n\s*\n+").expect("Invalid paragraph split regex"));

// Use fancy-regex for lookbehind support
static SENTENCE_SPLIT: Lazy<FancyRegex> = Lazy::new(|| {
    // Split after . ! ? followed by whitespace
    FancyRegex::new(r"(?<=[.!?])\s+").expect("Invalid sentence split regex")
});

/// Resolved chunking parameters.
#[derive(Debug, Clone, Copy)]
pub struct ChunkingParams {
    pub target_tokens: usize,
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub overlap_tokens: usize,
}

impl Default for ChunkingParams {
    fn default() -> Self {
        resolve_chunking_params(
            DEFAULT_CHUNK_TOKENS,
            None,
            None,
            DEFAULT_OVERLAP_RATIO,
            None,
        )
        .expect("Default params should be valid")
    }
}

impl ChunkingParams {
    /// Create params with explicit values.
    pub fn new(
        target_tokens: usize,
        min_tokens: usize,
        max_tokens: usize,
        overlap_tokens: usize,
    ) -> Result<Self, ChunkingError> {
        if target_tokens == 0 {
            return Err(ChunkingError::InvalidParams(
                "target_tokens must be positive".into(),
            ));
        }
        if min_tokens > target_tokens {
            return Err(ChunkingError::InvalidParams(
                "min_tokens cannot exceed target_tokens".into(),
            ));
        }
        if target_tokens > max_tokens {
            return Err(ChunkingError::InvalidParams(
                "target_tokens cannot exceed max_tokens".into(),
            ));
        }
        Ok(Self {
            target_tokens,
            min_tokens,
            max_tokens,
            overlap_tokens,
        })
    }
}

/// Errors that can occur during chunking.
#[derive(Debug, thiserror::Error)]
pub enum ChunkingError {
    #[error("Invalid chunking parameters: {0}")]
    InvalidParams(String),
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
}

/// Minimal unit of text with precomputed token count and position.
#[derive(Debug, Clone)]
struct TextUnit {
    text: String,
    tokens: usize,
    char_start: usize,
    char_end: usize,
}

/// A chunk of text with its token count and source offsets.
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub text: String,
    pub tokens: usize,
    pub char_start: usize,
    pub char_end: usize,
}

/// Resolve user-specified chunking parameters into a consistent config.
///
/// # Arguments
/// * `target_tokens` - Target tokens per chunk
/// * `min_tokens` - Minimum tokens (default: 0.6 * target)
/// * `max_tokens` - Maximum tokens (default: min(1.6 * target, 650))
/// * `overlap_ratio` - Overlap as fraction of target (0.0-1.0)
/// * `overlap_tokens` - Fixed overlap in tokens (overrides ratio)
pub fn resolve_chunking_params(
    target_tokens: usize,
    min_tokens: Option<usize>,
    max_tokens: Option<usize>,
    overlap_ratio: f64,
    overlap_tokens: Option<usize>,
) -> Result<ChunkingParams, ChunkingError> {
    if target_tokens == 0 {
        return Err(ChunkingError::InvalidParams(
            "target_tokens must be positive".into(),
        ));
    }

    let min_tokens = min_tokens.unwrap_or_else(|| (target_tokens as f64 * 0.6).max(1.0) as usize);
    let max_tokens = max_tokens.unwrap_or_else(|| ((target_tokens as f64 * 1.6) as usize).min(650));

    if min_tokens == 0 || max_tokens == 0 {
        return Err(ChunkingError::InvalidParams(
            "min_tokens and max_tokens must be positive".into(),
        ));
    }
    if min_tokens > target_tokens {
        return Err(ChunkingError::InvalidParams(
            "min_tokens cannot exceed target_tokens".into(),
        ));
    }
    if target_tokens > max_tokens {
        return Err(ChunkingError::InvalidParams(
            "target_tokens cannot exceed max_tokens".into(),
        ));
    }

    let overlap_tokens = if let Some(ot) = overlap_tokens {
        ot.min(target_tokens / 2) // Clamp to at most half target
    } else {
        if !(0.0..1.0).contains(&overlap_ratio) {
            return Err(ChunkingError::InvalidParams(
                "overlap_ratio must be in [0.0, 1.0)".into(),
            ));
        }
        ((target_tokens as f64 * overlap_ratio) as usize).min(target_tokens / 2)
    };

    Ok(ChunkingParams {
        target_tokens,
        min_tokens,
        max_tokens,
        overlap_tokens,
    })
}

/// Count tokens in text using cl100k_base tokenizer.
pub fn count_tokens(text: &str) -> usize {
    let bpe = cl100k_base().expect("Failed to load cl100k_base tokenizer");
    bpe.encode_with_special_tokens(text).len()
}

/// Encode text to tokens using cl100k_base.
fn encode(text: &str) -> Vec<u32> {
    let bpe = cl100k_base().expect("Failed to load cl100k_base tokenizer");
    bpe.encode_with_special_tokens(text)
}

/// Decode tokens back to text.
fn decode(tokens: &[u32]) -> String {
    let bpe = cl100k_base().expect("Failed to load cl100k_base tokenizer");
    bpe.decode(tokens.to_vec()).unwrap_or_default()
}

/// Split text into paragraphs (double newline or multiple newlines).
fn split_into_paragraphs(text: &str) -> Vec<&str> {
    PARAGRAPH_SPLIT
        .split(text)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Split text into sentences using fancy-regex lookbehind.
fn split_into_sentences(text: &str) -> Vec<String> {
    // fancy-regex doesn't have a simple split method, so we find all matches
    // and split manually
    let mut result = Vec::new();
    let mut last_end = 0;

    // Find all sentence-ending positions (after . ! ?)
    let re = &*SENTENCE_SPLIT;
    let mut finder = re.find_iter(text);

    while let Some(Ok(m)) = finder.next() {
        if m.start() > last_end {
            let sentence = &text[last_end..m.start()];
            if !sentence.is_empty() {
                result.push(sentence.to_string());
            }
        }
        last_end = m.end();
    }

    // Add remaining text
    if last_end < text.len() {
        let remaining = &text[last_end..];
        if !remaining.is_empty() {
            result.push(remaining.to_string());
        }
    }

    result
}

/// Fallback: break a long span into token windows.
fn window_text(text: &str, max_unit_tokens: usize, base_offset: usize) -> Vec<TextUnit> {
    let tokens = encode(text);
    let mut units = Vec::new();
    let mut char_offset = 0;
    let mut token_start = 0;

    while token_start < tokens.len() {
        let token_end = (token_start + max_unit_tokens).min(tokens.len());
        let chunk_tokens = &tokens[token_start..token_end];
        let chunk_text = decode(chunk_tokens);

        // Find this chunk in the original text
        let char_start = text[char_offset..]
            .find(&chunk_text)
            .map(|pos| char_offset + pos)
            .unwrap_or(char_offset);
        let char_end = char_start + chunk_text.len();

        units.push(TextUnit {
            text: chunk_text,
            tokens: chunk_tokens.len(),
            char_start: base_offset + char_start,
            char_end: base_offset + char_end,
        });

        char_offset = char_end;
        token_start = token_end;
    }

    units
}

/// Convert text into units (paragraphs or sentences), each <= max_unit_tokens.
///
/// Strategy:
/// 1. Start with paragraphs
/// 2. If paragraph is too large, split into sentences
/// 3. If a sentence is still too large, fall back to token windows
fn to_units(text: &str, max_unit_tokens: usize) -> Vec<TextUnit> {
    let mut units = Vec::new();
    let mut search_start = 0;

    for para in split_into_paragraphs(text) {
        if para.is_empty() {
            continue;
        }

        // Find paragraph position in original text
        let para_start = text[search_start..]
            .find(para)
            .map(|pos| search_start + pos)
            .unwrap_or(search_start);
        let para_end = para_start + para.len();
        search_start = para_end;

        let para_tokens = count_tokens(para);

        if para_tokens <= max_unit_tokens {
            units.push(TextUnit {
                text: para.to_string(),
                tokens: para_tokens,
                char_start: para_start,
                char_end: para_end,
            });
            continue;
        }

        // Paragraph too big: split into sentences
        let sentences = split_into_sentences(para);
        if sentences.is_empty() {
            // Fallback: directly window the paragraph
            units.extend(window_text(para, max_unit_tokens, para_start));
            continue;
        }

        // Track position within paragraph
        let mut sent_search_start = 0;
        for sent in sentences {
            let sent_tokens = count_tokens(&sent);

            // Find sentence position within paragraph, then adjust to absolute
            let sent_rel_start = para[sent_search_start..]
                .find(&sent)
                .map(|pos| sent_search_start + pos)
                .unwrap_or(sent_search_start);
            let sent_rel_end = sent_rel_start + sent.len();
            sent_search_start = sent_rel_end;

            let sent_start = para_start + sent_rel_start;
            let sent_end = para_start + sent_rel_end;

            if sent_tokens <= max_unit_tokens {
                units.push(TextUnit {
                    text: sent,
                    tokens: sent_tokens,
                    char_start: sent_start,
                    char_end: sent_end,
                });
            } else {
                // Rare: single long sentence, fall back to token windows
                units.extend(window_text(&sent, max_unit_tokens, sent_start));
            }
        }
    }

    units
}

/// Split text into overlapping chunks, biased toward paragraphs/sentences.
///
/// Behavior:
/// - Only chunk when total_tokens > target_tokens
/// - Each chunk aims for [min_tokens, max_tokens], with some flexibility
/// - Overlap is specified in tokens and aligned to whole units
pub fn chunk_text_by_tokens(text: &str, params: &ChunkingParams) -> Vec<TextChunk> {
    let total_tokens = count_tokens(text);

    // Empty text
    if total_tokens == 0 {
        return Vec::new();
    }

    // Short text: return as single chunk (no overlap)
    if total_tokens <= params.target_tokens {
        return vec![TextChunk {
            text: text.to_string(),
            tokens: total_tokens,
            char_start: 0,
            char_end: text.len(),
        }];
    }

    // Build units first
    let units = to_units(text, params.max_tokens);
    if units.is_empty() {
        // Fallback: treat entire text as one chunk
        return vec![TextChunk {
            text: text.to_string(),
            tokens: total_tokens,
            char_start: 0,
            char_end: text.len(),
        }];
    }

    let mut chunks_units: Vec<Vec<&TextUnit>> = Vec::new();
    let n = units.len();
    let mut start = 0;

    while start < n {
        let mut token_count = 0;
        let mut end = start;

        // Greedily add units up to max_tokens
        while end < n && token_count + units[end].tokens <= params.max_tokens {
            token_count += units[end].tokens;
            end += 1;
        }

        // Ensure we always make progress, even if a unit somehow exceeds max_tokens
        if end == start {
            token_count = units[end].tokens;
            end += 1;
        }

        // If chunk is still below min_tokens and we can extend, allow one more unit
        if token_count < params.min_tokens && end < n {
            end += 1;
        }

        let chunk_units: Vec<&TextUnit> = units[start..end].iter().collect();
        chunks_units.push(chunk_units);

        if end >= n {
            break;
        }

        // Determine next start index with overlap
        if params.overlap_tokens == 0 {
            start = end;
            continue;
        }

        let mut overlap_acc = 0;
        let mut idx = end - 1;
        // Walk backwards until we reach desired overlap or the start of this chunk
        while idx > start && overlap_acc < params.overlap_tokens {
            overlap_acc += units[idx].tokens;
            idx -= 1;
        }

        // Next chunk starts at the first unit whose suffix gives us the overlap
        start = idx + 1;
    }

    // Optional tail merge: avoid a very small final chunk when possible
    if chunks_units.len() >= 2 {
        let last_tokens: usize = chunks_units.last().unwrap().iter().map(|u| u.tokens).sum();
        if last_tokens < params.min_tokens {
            let merged_tokens: usize = chunks_units[chunks_units.len() - 2]
                .iter()
                .chain(chunks_units.last().unwrap().iter())
                .map(|u| u.tokens)
                .sum();
            // Allow merging if it doesn't create an unreasonable final chunk
            if merged_tokens <= params.max_tokens || chunks_units.len() == 2 {
                let last = chunks_units.pop().unwrap();
                let prev = chunks_units.pop().unwrap();
                let merged: Vec<&TextUnit> = prev.into_iter().chain(last.into_iter()).collect();
                chunks_units.push(merged);
            }
        }
    }

    // Convert units back to text chunks with precomputed token counts and offsets
    chunks_units
        .into_iter()
        .map(|cu| {
            let chunk_text = cu
                .iter()
                .map(|u| u.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            let chunk_tokens: usize = cu.iter().map(|u| u.tokens).sum();
            let char_start = cu.first().map(|u| u.char_start).unwrap_or(0);
            let char_end = cu.last().map(|u| u.char_end).unwrap_or(0);
            TextChunk {
                text: chunk_text,
                tokens: chunk_tokens,
                char_start,
                char_end,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = ChunkingParams::default();
        assert_eq!(params.target_tokens, 300);
        assert_eq!(params.min_tokens, 180);
        assert_eq!(params.max_tokens, 480);
        assert_eq!(params.overlap_tokens, 60);
    }

    #[test]
    fn test_count_tokens() {
        let count = count_tokens("Hello, world!");
        assert!(count > 0);
        assert!(count < 10);
    }

    #[test]
    fn test_short_text_single_chunk() {
        let text = "This is a short text that should be a single chunk.";
        let params = ChunkingParams::default();
        let chunks = chunk_text_by_tokens(text, &params);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
        assert_eq!(chunks[0].char_start, 0);
        assert_eq!(chunks[0].char_end, text.len());
    }

    #[test]
    fn test_empty_text() {
        let params = ChunkingParams::default();
        let chunks = chunk_text_by_tokens("", &params);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_paragraph_splitting() {
        let text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph.";
        let paragraphs = split_into_paragraphs(text);
        assert_eq!(paragraphs.len(), 3);
        assert_eq!(paragraphs[0], "First paragraph here.");
        assert_eq!(paragraphs[1], "Second paragraph here.");
        assert_eq!(paragraphs[2], "Third paragraph.");
    }

    #[test]
    fn test_sentence_splitting() {
        let text = "First sentence. Second sentence! Third sentence? Fourth.";
        let sentences = split_into_sentences(text);
        assert_eq!(sentences.len(), 4);
    }

    #[test]
    fn test_long_text_chunking() {
        // Create a text that's definitely longer than 300 tokens
        let long_para = "This is a test sentence. ".repeat(50);
        let text = format!("{}\n\n{}", long_para, long_para);

        let params = ChunkingParams::default();
        let chunks = chunk_text_by_tokens(&text, &params);

        // Should produce multiple chunks
        assert!(chunks.len() > 1);

        // Each chunk should respect token limits
        for chunk in &chunks {
            // Allow some flexibility for merging behavior
            assert!(chunk.tokens <= params.max_tokens + 100);
        }

        // Verify char_start/char_end are reasonable
        for chunk in &chunks {
            assert!(chunk.char_end >= chunk.char_start);
        }
    }

    #[test]
    fn test_resolve_params_invalid() {
        // Zero target should fail
        assert!(resolve_chunking_params(0, None, None, 0.2, None).is_err());

        // Invalid overlap ratio should fail
        assert!(resolve_chunking_params(300, None, None, 1.5, None).is_err());
    }
}
