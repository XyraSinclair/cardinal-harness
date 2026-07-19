//! Tolerant extraction of proposal JSON from model completions.
//!
//! json_mode output varies by model family: a bare array, an object
//! wrapping one, an object whose VALUES are the strings, or a valid
//! object under a pathological key (gpt-5.4-mini shipped `{"[]": [...]}`
//! during the Manifund P1 run — the old first-`[`-to-last-`]` slice
//! turned that VALID document into a parse error). The rule here:
//! parse the whole completion first; only when that fails, scan for the
//! first balanced JSON span (string- and escape-aware) that parses.

use serde_json::Value;

/// Parse a completion into JSON, whole-string first, then the first
/// balanced `[...]`/`{...}` span that parses. `None` only when no
/// parseable JSON exists anywhere in the content.
pub(crate) fn lenient_value(content: &str) -> Option<Value> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Some(v);
    }
    let bytes = trimmed.as_bytes();
    for start in 0..bytes.len() {
        let open = bytes[start];
        if open != b'[' && open != b'{' {
            continue;
        }
        if let Some(end) = balanced_end(bytes, start) {
            if let Ok(v) = serde_json::from_str::<Value>(&trimmed[start..=end]) {
                return Some(v);
            }
        }
    }
    None
}

/// Index of the byte closing the bracket opened at `start`, respecting
/// strings and escapes. `None` when the span never closes.
fn balanced_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'[' | b'{' => depth += 1,
            b']' | b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(idx);
                }
            }
            _ => {}
        }
    }
    None
}

/// Pull a list of strings out of whatever JSON shape the model chose:
/// a bare array, an object wrapping an array (any key), an object whose
/// values are the strings, or array elements that are single-string
/// objects.
pub(crate) fn lenient_string_array(content: &str) -> Option<Vec<String>> {
    let value = lenient_value(content)?;
    let strings = value_string_array(&value)?;
    (!strings.is_empty()).then_some(strings)
}

pub(crate) fn value_string_array(value: &Value) -> Option<Vec<String>> {
    let element = |v: &Value| -> Option<String> {
        v.as_str().map(str::to_string).or_else(|| {
            v.as_object()
                .and_then(|o| o.values().find_map(Value::as_str))
                .map(str::to_string)
        })
    };
    if let Some(array) = value.as_array().or_else(|| {
        value
            .as_object()
            .and_then(|o| o.values().find_map(Value::as_array))
    }) {
        return array.iter().map(element).collect();
    }
    if let Some(object) = value.as_object() {
        let strings: Vec<String> = object.values().filter_map(element).collect();
        if !strings.is_empty() {
            return Some(strings);
        }
    }
    None
}
