//! Pins for committed HTML artifact pages under `artifacts/live/`.
//!
//! Two bug classes, both live-caught 2026-07-08:
//!
//! 1. **Charset**: `python3 -m http.server` (the working-norm server for
//!    these pages) sends no charset header, so a UTF-8 page without
//!    `<meta charset="utf-8">` renders as windows-1252 mojibake — every
//!    em-dash, ρ and χ garbled. The evidence viewer shipped mojibake in its
//!    first render, and the sweep then found map.html and leaderboard.html
//!    already carrying the same latent bug (fixed 2abcb43). Served bytes
//!    are physics (PRINCIPLES §11).
//!
//! 2. **Page/evidence drift**: the evidence viewer inlines every measured
//!    number verbatim from the committed JSON sources. Nothing but this test
//!    enforces that the page and the sources stay in agreement—an artifact
//!    page that drifts from its evidence is precisely the “faking numbers for
//!    a better demo” failure the differentiation doc names as the thing that
//!    would torch the stack's credibility.

use std::fs;
use std::path::{Path, PathBuf};

fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn collect_html(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("readable dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_html(&path, out);
        } else if path.extension().is_some_and(|e| e == "html") {
            out.push(path);
        }
    }
}

/// Every committed HTML page must declare UTF-8 near the top of the file —
/// the static server will not do it for them.
#[test]
fn committed_html_declares_charset() {
    let root = repo_path("artifacts/live");
    let mut pages = Vec::new();
    collect_html(&root, &mut pages);
    assert!(
        !pages.is_empty(),
        "expected at least one committed HTML page under artifacts/live"
    );
    let mut missing = Vec::new();
    for page in &pages {
        let text = fs::read_to_string(page).expect("readable page");
        let head: String = text.chars().take(512).collect::<String>().to_lowercase();
        if !head.contains(r#"<meta charset="utf-8">"#) {
            missing.push(page.display().to_string());
        }
    }
    assert!(
        missing.is_empty(),
        "committed pages missing <meta charset=\"utf-8\"> in the first 512 bytes \
         (they will render as windows-1252 mojibake when served): {missing:?}"
    );
}

/// Extract the raw numeric literal tokens following `"key":` occurrences in
/// JSON text. Raw-text extraction (not serde -> f64 -> Display) on purpose:
/// the page inlined the source bytes verbatim, and the pin is on those
/// exact byte sequences (e.g. `0.0` must stay `0.0`, not `0`).
fn raw_number_tokens(json_text: &str, key: &str) -> Vec<String> {
    let needle = format!("\"{key}\":");
    let mut tokens = Vec::new();
    let mut rest = json_text;
    while let Some(at) = rest.find(&needle) {
        let after = &rest[at + needle.len()..];
        let token: String = after
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E'))
            .collect();
        if !token.is_empty() {
            tokens.push(token);
        }
        rest = &rest[at + needle.len()..];
    }
    tokens
}

/// Every measured number in the committed spin evidence must appear verbatim
/// in the evidence viewer page. If a source is corrected or the page is edited,
/// the two must move together.
#[test]
fn evidence_viewer_numbers_match_committed_sources() {
    let page = fs::read_to_string(repo_path(
        "artifacts/live/evidence-viewer-2026-07-08/index.html",
    ))
    .expect("evidence viewer page");
    let sources = [
        "artifacts/live/spin-sweep-2026-07-05/contested_gpt-5.4-mini.json",
        "artifacts/live/spin-sweep-2026-07-05/contested_claude-sonnet-4.6.json",
        "artifacts/live/spin-probe-2026-07-05/contested_pair_gpt-5.4-mini.json",
        "artifacts/live/spin-probe-2026-07-05/contested_pair_gemini-2.5-flash.json",
        "artifacts/live/spin-probe-2026-07-05/clear_pair_shininess.json",
    ];
    let keys = [
        "mean_log_ratio",
        "chi_slope",
        "linearity_r2",
        "susceptibility_nats",
        "cost_nanodollars",
    ];
    let mut checked = 0usize;
    let mut missing = Vec::new();
    for rel in sources {
        let text = fs::read_to_string(repo_path(rel)).expect("readable evidence source");
        for key in keys {
            for token in raw_number_tokens(&text, key) {
                checked += 1;
                if !page.contains(&token) {
                    missing.push(format!("{rel} {key}={token}"));
                }
            }
        }
    }
    // 2 sweeps x (7 readings + chi + r2 + cost) + 3 probes x (3 readings + chi + cost).
    assert_eq!(
        checked, 35,
        "evidence shape changed: expected 35 pinned numbers, found {checked} — update the pin \
         and the page together"
    );
    assert!(
        missing.is_empty(),
        "evidence numbers absent from the viewer page (page and sources have drifted): {missing:?}"
    );
}
