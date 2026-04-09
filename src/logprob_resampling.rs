//! Logprob-based ratio resampling for pairwise comparisons.
//!
//! Part of the logprob extraction research thread (see
//! [`docs/RESEARCH_THREADS.md`](../docs/RESEARCH_THREADS.md#logprob-extraction)).
//! This module provides a spec/prototype for reconstructing the LLM's latent
//! confidence distribution over ratio ladder values from token-level
//! log-probabilities. It is kept as a research prototype for future integration
//! into the main comparison pipeline.

use serde::{Deserialize, Serialize};

use crate::gateway::TokenLogprob;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RatioResamplingPlan {
    pub selected_ratio: f64,
    pub selected_ratio_text: String,
    pub reconstructed_output: String,
    pub ratio_token_span: Option<(usize, usize)>,
    pub steps: Vec<RatioResamplingStep>,
    pub branches: Vec<RatioResamplingBranch>,
    pub blocked_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RatioResamplingStep {
    pub token_index: usize,
    pub prefix_before: String,
    pub selected_token: String,
    pub selected_probability: f64,
    pub prefix_after: String,
    pub candidate_ratios_after: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RatioResamplingBranch {
    pub token_index: usize,
    pub alternative_token: String,
    pub alternative_probability: f64,
    pub branch_prefix: String,
    pub candidate_ratios: Vec<f64>,
}

#[derive(Debug, Clone)]
struct AlignedRatioToken {
    token_index: usize,
    prefix_affix: String,
    core_segment: String,
    suffix_affix: String,
}

pub fn canonical_ratio_string(ratio: f64) -> String {
    let mut s = format!("{ratio:.2}");
    while s.ends_with('0') && s.contains('.') {
        s.pop();
    }
    if s.ends_with('.') {
        s.push('0');
    }
    s
}

pub fn plan_ratio_resampling(
    output_logprobs: &[TokenLogprob],
    selected_ratio: f64,
    ratio_ladder: &[f64],
    min_branch_probability: f64,
    max_branches_per_step: usize,
) -> RatioResamplingPlan {
    let selected_ratio_text = canonical_ratio_string(selected_ratio);
    let reconstructed_output = output_logprobs
        .iter()
        .map(|entry| entry.token.as_str())
        .collect::<String>();
    let ladder_strings: Vec<(f64, String)> = ratio_ladder
        .iter()
        .copied()
        .map(|ratio| (ratio, canonical_ratio_string(ratio)))
        .collect();

    let Some(aligned_tokens) = align_ratio_tokens(output_logprobs, &selected_ratio_text) else {
        return RatioResamplingPlan {
            selected_ratio,
            selected_ratio_text,
            reconstructed_output,
            ratio_token_span: None,
            steps: Vec::new(),
            branches: Vec::new(),
            blocked_reason: Some(
                "selected ratio text does not align cleanly to the returned token stream"
                    .to_string(),
            ),
        };
    };

    let span = aligned_tokens
        .first()
        .zip(aligned_tokens.last())
        .map(|(first, last)| (first.token_index, last.token_index + 1));

    let mut prefix = String::new();
    let mut steps = Vec::new();
    let mut branches = Vec::new();

    for aligned in &aligned_tokens {
        let entry = &output_logprobs[aligned.token_index];
        let prefix_before = prefix.clone();
        prefix.push_str(&aligned.core_segment);
        let candidate_ratios_after = ladder_candidates_with_prefix(&ladder_strings, &prefix);

        steps.push(RatioResamplingStep {
            token_index: aligned.token_index,
            prefix_before: prefix_before.clone(),
            selected_token: aligned.core_segment.clone(),
            selected_probability: entry.logprob.exp(),
            prefix_after: prefix.clone(),
            candidate_ratios_after,
        });

        let mut local_branches = entry
            .top_alternatives
            .iter()
            .filter_map(|alternative| {
                let probability = alternative.logprob.exp();
                if probability < min_branch_probability {
                    return None;
                }
                let core = strip_affixes(
                    &alternative.token,
                    &aligned.prefix_affix,
                    &aligned.suffix_affix,
                )?;
                let branch_prefix = format!("{prefix_before}{core}");
                let candidate_ratios =
                    ladder_candidates_with_prefix(&ladder_strings, &branch_prefix);
                if candidate_ratios.is_empty() {
                    return None;
                }
                Some(RatioResamplingBranch {
                    token_index: aligned.token_index,
                    alternative_token: core,
                    alternative_probability: probability,
                    branch_prefix,
                    candidate_ratios,
                })
            })
            .collect::<Vec<_>>();

        local_branches.sort_by(|left, right| {
            right
                .alternative_probability
                .partial_cmp(&left.alternative_probability)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.branch_prefix.cmp(&right.branch_prefix))
        });
        local_branches.dedup_by(|left, right| left.branch_prefix == right.branch_prefix);
        local_branches.truncate(max_branches_per_step);
        branches.extend(local_branches);
    }

    RatioResamplingPlan {
        selected_ratio,
        selected_ratio_text,
        reconstructed_output,
        ratio_token_span: span,
        steps,
        branches,
        blocked_reason: None,
    }
}

fn ladder_candidates_with_prefix(ladder_strings: &[(f64, String)], prefix: &str) -> Vec<f64> {
    ladder_strings
        .iter()
        .filter_map(|(ratio, text)| text.starts_with(prefix).then_some(*ratio))
        .collect()
}

fn strip_affixes(token: &str, prefix: &str, suffix: &str) -> Option<String> {
    if !token.starts_with(prefix) || !token.ends_with(suffix) {
        return None;
    }

    let start = prefix.len();
    let end = token.len().checked_sub(suffix.len())?;
    if end < start || !token.is_char_boundary(start) || !token.is_char_boundary(end) {
        return None;
    }

    let core = &token[start..end];
    (!core.is_empty()).then(|| core.to_string())
}

fn align_ratio_tokens(
    output_logprobs: &[TokenLogprob],
    selected_ratio_text: &str,
) -> Option<Vec<AlignedRatioToken>> {
    let reconstructed = output_logprobs
        .iter()
        .map(|entry| entry.token.as_str())
        .collect::<String>();
    let token_spans = token_byte_spans(output_logprobs);

    let mut best: Option<(usize, Vec<AlignedRatioToken>)> = None;

    for (match_start, _) in reconstructed.match_indices(selected_ratio_text) {
        let match_end = match_start + selected_ratio_text.len();
        let mut aligned = Vec::new();

        for (token_index, (token_start, token_end)) in token_spans.iter().copied().enumerate() {
            if token_end <= match_start || token_start >= match_end {
                continue;
            }

            let local_start = match_start.max(token_start) - token_start;
            let local_end = match_end.min(token_end) - token_start;
            let token = &output_logprobs[token_index].token;
            if local_start == local_end
                || !token.is_char_boundary(local_start)
                || !token.is_char_boundary(local_end)
            {
                aligned.clear();
                break;
            }

            aligned.push(AlignedRatioToken {
                token_index,
                prefix_affix: token[..local_start].to_string(),
                core_segment: token[local_start..local_end].to_string(),
                suffix_affix: token[local_end..].to_string(),
            });
        }

        if aligned.is_empty() {
            continue;
        }
        let combined_core = aligned
            .iter()
            .map(|token| token.core_segment.as_str())
            .collect::<String>();
        if combined_core != selected_ratio_text {
            continue;
        }

        let extra_affix_len = aligned
            .iter()
            .map(|token| token.prefix_affix.len() + token.suffix_affix.len())
            .sum();
        match &best {
            Some((best_extra, _)) if *best_extra <= extra_affix_len => {}
            _ => best = Some((extra_affix_len, aligned)),
        }
    }

    best.map(|(_, aligned)| aligned)
}

fn token_byte_spans(output_logprobs: &[TokenLogprob]) -> Vec<(usize, usize)> {
    let mut cursor = 0usize;
    output_logprobs
        .iter()
        .map(|entry| {
            let start = cursor;
            cursor += entry.token.len();
            (start, cursor)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{canonical_ratio_string, plan_ratio_resampling};
    use crate::gateway::{TokenAlternative, TokenLogprob};
    use crate::prompts::RATIO_LADDER;

    #[test]
    fn canonical_ratio_string_preserves_ladder_format() {
        assert_eq!(canonical_ratio_string(1.0), "1.0");
        assert_eq!(canonical_ratio_string(1.05), "1.05");
        assert_eq!(canonical_ratio_string(1.1), "1.1");
        assert_eq!(canonical_ratio_string(18.0), "18.0");
    }

    #[test]
    fn plan_ratio_resampling_tracks_branch_prefixes_for_split_decimal_tokens() {
        let logprobs = vec![
            TokenLogprob {
                token: "2.".to_string(),
                logprob: 0.72f64.ln(),
                top_alternatives: vec![
                    TokenAlternative {
                        token: "1.".to_string(),
                        logprob: 0.19f64.ln(),
                    },
                    TokenAlternative {
                        token: "3.".to_string(),
                        logprob: 0.06f64.ln(),
                    },
                ],
            },
            TokenLogprob {
                token: "1".to_string(),
                logprob: 0.81f64.ln(),
                top_alternatives: vec![TokenAlternative {
                    token: "5".to_string(),
                    logprob: 0.12f64.ln(),
                }],
            },
        ];

        let plan = plan_ratio_resampling(&logprobs, 2.1, RATIO_LADDER, 0.05, 4);
        assert_eq!(plan.ratio_token_span, Some((0, 2)));
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].prefix_after, "2.");
        assert_eq!(plan.steps[1].prefix_after, "2.1");

        let prefixes = plan
            .branches
            .iter()
            .map(|branch| branch.branch_prefix.as_str())
            .collect::<Vec<_>>();
        assert!(prefixes.contains(&"1."));
        assert!(prefixes.contains(&"3."));
        assert!(prefixes.contains(&"2.5"));

        let one_dot = plan
            .branches
            .iter()
            .find(|branch| branch.branch_prefix == "1.")
            .expect("1. branch");
        assert_eq!(
            one_dot.candidate_ratios,
            vec![1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75]
        );
    }

    #[test]
    fn plan_ratio_resampling_handles_shared_whitespace_affixes() {
        let logprobs = vec![
            TokenLogprob {
                token: " 2".to_string(),
                logprob: 0.62f64.ln(),
                top_alternatives: vec![TokenAlternative {
                    token: " 1".to_string(),
                    logprob: 0.21f64.ln(),
                }],
            },
            TokenLogprob {
                token: ".".to_string(),
                logprob: 0.95f64.ln(),
                top_alternatives: vec![],
            },
            TokenLogprob {
                token: "1".to_string(),
                logprob: 0.88f64.ln(),
                top_alternatives: vec![],
            },
        ];

        let plan = plan_ratio_resampling(&logprobs, 2.1, RATIO_LADDER, 0.05, 4);
        assert_eq!(plan.ratio_token_span, Some((0, 3)));
        assert_eq!(plan.steps[0].selected_token, "2");
        assert!(plan
            .branches
            .iter()
            .any(|branch| branch.branch_prefix == "1"));
    }

    #[test]
    fn plan_ratio_resampling_reports_blocked_alignment() {
        let logprobs = vec![TokenLogprob {
            token: "twenty one".to_string(),
            logprob: -0.1,
            top_alternatives: vec![],
        }];

        let plan = plan_ratio_resampling(&logprobs, 2.1, RATIO_LADDER, 0.05, 4);
        assert!(plan.ratio_token_span.is_none());
        assert!(plan.blocked_reason.is_some());
    }
}
