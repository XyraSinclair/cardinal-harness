use super::*;

/// Raw JSON structure from LLM response.
#[derive(Debug, Deserialize)]
pub(super) struct PairwiseEvalJson {
    #[serde(default)]
    higher_ranked: Option<String>,
    #[serde(default)]
    lower_ranked: Option<String>,
    #[serde(default)]
    ratio: Option<f64>,
    #[serde(default)]
    fraction: Option<f64>,
    #[serde(default)]
    ratio_bucket: Option<usize>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    refused: Option<bool>,
}

/// Parse LLM response JSON into a PairwiseJudgement.
pub fn parse_pairwise_response(
    raw: &str,
    prompt_template_slug: &str,
    _output_logprobs: Option<&[TokenLogprob]>,
) -> Result<PairwiseJudgement, ComparisonError> {
    // Try to extract JSON from the response (may have surrounding text)
    let json_str = extract_json(raw);

    let parsed: PairwiseEvalJson =
        serde_json::from_str(json_str).map_err(|e| ComparisonError::Parse(e.to_string()))?;

    if parsed.refused.unwrap_or(false) {
        return Ok(PairwiseJudgement::Refused);
    }

    // Slug-aware answer recovery: every template lowers to (winner slot,
    // ratio >= 1 by which the winner has more). The inverse-question
    // templates exist so wording invariance is measurable — a coherent
    // judge asked "times more", "what fraction", and "which has less" must
    // yield the same signed log-ratio.
    let (higher, ratio) = match prompt_template_slug {
        // Ordinal judgements carry only direction plus confidence, so we map
        // them onto a shared modest fixed ratio before passing them to the
        // solver.
        "ordinal_v1" => (
            parsed
                .higher_ranked
                .ok_or_else(|| ComparisonError::Parse("missing 'higher_ranked'".into()))?,
            ORDINAL_OBSERVATION_RATIO,
        ),
        "canonical_v2" => (
            parsed
                .higher_ranked
                .ok_or_else(|| ComparisonError::Parse("missing 'higher_ranked'".into()))?,
            parsed
                .ratio
                .ok_or_else(|| ComparisonError::Parse("missing 'ratio'".into()))?,
        ),
        "canonical_bucket_v1" => {
            let bucket = parsed
                .ratio_bucket
                .ok_or_else(|| ComparisonError::Parse("missing 'ratio_bucket'".into()))?;
            (
                parsed
                    .higher_ranked
                    .ok_or_else(|| ComparisonError::Parse("missing 'higher_ranked'".into()))?,
                *RATIO_LADDER.get(bucket).ok_or_else(|| {
                    ComparisonError::Parse(format!(
                        "ratio_bucket out of allowed range [0,16]: {bucket}"
                    ))
                })?,
            )
        }
        // "Which has LESS, and how many times less": the winner is the
        // OTHER slot.
        "less_v1" => {
            let lower = parsed
                .lower_ranked
                .ok_or_else(|| ComparisonError::Parse("missing 'lower_ranked'".into()))?;
            let winner = match lower.to_uppercase().as_str() {
                "A" => "B".to_string(),
                "B" => "A".to_string(),
                other => {
                    return Err(ComparisonError::Parse(format!(
                        "invalid 'lower_ranked': {other}"
                    )))
                }
            };
            (
                winner,
                parsed
                    .ratio
                    .ok_or_else(|| ComparisonError::Parse("missing 'ratio'".into()))?,
            )
        }
        // "What fraction of the greater one's level does the lesser reach":
        // ratio = 1/fraction, capped at the ladder maximum.
        "fraction_v1" => {
            let fraction = parsed
                .fraction
                .ok_or_else(|| ComparisonError::Parse("missing 'fraction'".into()))?;
            if !(fraction > 0.0 && fraction <= 1.0) {
                return Err(ComparisonError::Parse(format!(
                    "fraction out of allowed range (0,1]: {fraction}"
                )));
            }
            (
                parsed
                    .higher_ranked
                    .ok_or_else(|| ComparisonError::Parse("missing 'higher_ranked'".into()))?,
                (1.0 / fraction).min(26.0),
            )
        }
        other => {
            return Err(ComparisonError::Parse(format!(
                "unknown prompt template slug: {other}"
            )))
        }
    };

    if !(1.0..=26.0).contains(&ratio) {
        return Err(ComparisonError::Parse(format!(
            "ratio out of allowed range [1,26]: {ratio}"
        )));
    }

    let higher_ranked = match higher.to_uppercase().as_str() {
        "A" => HigherRanked::A,
        "B" => HigherRanked::B,
        other => {
            return Err(ComparisonError::Parse(format!(
                "invalid higher_ranked: {other}"
            )))
        }
    };
    let confidence = parsed
        .confidence
        .ok_or_else(|| ComparisonError::Parse("missing 'confidence'".into()))?;

    Ok(PairwiseJudgement::Observation {
        higher_ranked,
        ratio,
        confidence: confidence.clamp(0.0, 1.0),
    })
}

pub(super) fn token_preferred_side(token: &str) -> Option<PairwisePreferredSide> {
    let letters: String = token
        .chars()
        .filter(|ch| ch.is_ascii_alphabetic())
        .collect();
    match letters.to_ascii_uppercase().as_str() {
        "A" => Some(PairwisePreferredSide::A),
        "B" => Some(PairwisePreferredSide::B),
        _ => None,
    }
}

pub(super) fn token_bucket_index(token: &str) -> Option<usize> {
    let digits: String = token.chars().filter(|ch| ch.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let idx = digits.parse::<usize>().ok()?;
    (idx < RATIO_LADDER.len()).then_some(idx)
}

pub(super) fn token_bucket_index_at(logprobs: &[TokenLogprob], position: usize) -> Option<usize> {
    let token = &logprobs[position].token;
    let previous_token = position
        .checked_sub(1)
        .and_then(|idx| logprobs.get(idx))
        .map(|entry| entry.token.as_str());
    token_bucket_index_with_previous(token, previous_token)
}

pub(super) fn token_bucket_index_with_previous(
    token: &str,
    previous_token: Option<&str>,
) -> Option<usize> {
    let token_digits: String = token.chars().filter(|ch| ch.is_ascii_digit()).collect();
    if token_digits.len() == 1 {
        if let Some(previous) = previous_token {
            let previous_digits: String =
                previous.chars().filter(|ch| ch.is_ascii_digit()).collect();
            if previous_digits == "1" {
                let second_digit = token_digits.parse::<usize>().ok()?;
                let idx = 10 + second_digit;
                if idx < RATIO_LADDER.len() {
                    return Some(idx);
                }
            }
        }
    }
    token_bucket_index(token)
}

pub(super) fn collect_distribution<T: Copy>(
    position: &TokenLogprob,
    support: &[T],
    mut token_index: impl FnMut(&str) -> Option<usize>,
) -> DiscreteDistribution<T> {
    let mut probabilities = vec![0.0; support.len()];

    if let Some(idx) = token_index(&position.token) {
        probabilities[idx] = position.logprob.exp();
    }

    for alternative in &position.top_alternatives {
        if let Some(idx) = token_index(&alternative.token) {
            probabilities[idx] = probabilities[idx].max(alternative.logprob.exp());
        }
    }

    let covered: f64 = probabilities.iter().sum();
    DiscreteDistribution::new(
        support
            .iter()
            .copied()
            .zip(probabilities)
            .map(|(value, probability)| WeightedValue { value, probability })
            .collect(),
        (1.0 - covered).max(0.0),
    )
}

pub(super) fn previous_tokens_name_field(
    logprobs: &[TokenLogprob],
    position: usize,
    field: &str,
) -> bool {
    let start = position.saturating_sub(8);
    let context = logprobs[start..position]
        .iter()
        .map(|entry| entry.token.as_str())
        .collect::<String>()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
        .collect::<String>()
        .to_ascii_lowercase();
    context.contains(field)
}

pub(super) fn pairwise_bucket_logprob_posterior(
    logprobs: &[TokenLogprob],
    selected_higher_ranked: PairwisePreferredSide,
    selected_ratio: f64,
) -> Option<PairwiseLogprobPosterior> {
    let selected_ratio_bucket = RatioBucket::from_ratio(selected_ratio)?;
    let selected_answer =
        PairwiseAnswer::observation(selected_higher_ranked, selected_ratio_bucket);

    let side_position = logprobs.iter().enumerate().find_map(|(idx, entry)| {
        (token_preferred_side(&entry.token) == Some(selected_higher_ranked)
            && previous_tokens_name_field(logprobs, idx, "higher_ranked"))
        .then_some(entry)
    })?;

    let selected_bucket_idx = selected_ratio_bucket.index();
    let bucket_position_idx = logprobs.iter().enumerate().find_map(|(idx, _entry)| {
        (token_bucket_index_at(logprobs, idx) == Some(selected_bucket_idx)
            && previous_tokens_name_field(logprobs, idx, "ratio_bucket"))
        .then_some(idx)
    })?;
    let bucket_position = &logprobs[bucket_position_idx];
    let bucket_previous_token = bucket_position_idx
        .checked_sub(1)
        .and_then(|idx| logprobs.get(idx))
        .map(|entry| entry.token.as_str());

    let higher_ranked_distribution = collect_distribution(
        side_position,
        &[PairwisePreferredSide::A, PairwisePreferredSide::B],
        |token| match token_preferred_side(token)? {
            PairwisePreferredSide::A => Some(0),
            PairwisePreferredSide::B => Some(1),
        },
    );
    let ratio_distribution = collect_distribution(bucket_position, RatioBucket::all(), |token| {
        token_bucket_index_with_previous(token, bucket_previous_token)
            .map(|idx| RatioBucket::ALL[idx].index())
    });

    let selected_idx = selected_ratio_bucket.index();
    let neighbor_indices: Vec<usize> = (0..RATIO_LADDER.len())
        .filter(|&idx| idx.abs_diff(selected_idx) <= 1)
        .collect();
    let answer_distribution = higher_ranked_distribution
        .product(&ratio_distribution, |higher_ranked, ratio| {
            PairwiseAnswer::observation(*higher_ranked, *ratio)
        });
    let signed_ln_ratio_distribution =
        SignedLogRatioDistribution::from_answer_distribution(&answer_distribution);
    let top_prob = answer_distribution.probability_of(|answer| *answer == selected_answer);
    let neighborhood_prob = answer_distribution.probability_of(|answer| {
        answer.preferred_side() == Some(selected_higher_ranked)
            && answer
                .ratio_bucket()
                .map(|bucket| neighbor_indices.contains(&bucket.index()))
                .unwrap_or(false)
    });
    let confidence = ConfidenceSource::Logprob {
        entropy: answer_distribution.entropy(),
        top_prob,
        neighborhood_prob: neighborhood_prob.clamp(0.0, 1.0),
    };

    Some(PairwiseLogprobPosterior {
        selected_answer,
        selected_higher_ranked,
        selected_ratio,
        selected_ratio_bucket,
        higher_ranked_distribution,
        ratio_distribution,
        answer_distribution,
        signed_ln_ratio_distribution,
        confidence,
    })
}

pub(super) fn compact_bucket_output_logprobs(
    logprobs: &[TokenLogprob],
    selected_higher_ranked: PairwisePreferredSide,
    selected_ratio: f64,
) -> Option<Vec<TokenLogprob>> {
    let selected_ratio_bucket = RatioBucket::from_ratio(selected_ratio)?;
    let side_position = logprobs.iter().enumerate().find_map(|(idx, entry)| {
        (token_preferred_side(&entry.token) == Some(selected_higher_ranked)
            && previous_tokens_name_field(logprobs, idx, "higher_ranked"))
        .then_some(idx)
    })?;
    let bucket_position = logprobs.iter().enumerate().find_map(|(idx, _entry)| {
        (token_bucket_index_at(logprobs, idx) == Some(selected_ratio_bucket.index())
            && previous_tokens_name_field(logprobs, idx, "ratio_bucket"))
        .then_some(idx)
    })?;

    let mut compact = Vec::with_capacity(2);
    compact.push(logprobs[side_position].clone());
    if bucket_position != side_position {
        compact.push(logprobs[bucket_position].clone());
    }
    Some(compact)
}

pub(super) fn fallback_stored_logprobs(
    response_logprobs: Option<&[TokenLogprob]>,
) -> Option<Vec<TokenLogprob>> {
    response_logprobs.map(|logprobs| truncate_output_logprobs(logprobs, 50))
}

/// Extract JSON object from response (handles models that add surrounding text).
pub(super) fn extract_json(raw: &str) -> &str {
    let trimmed = raw.trim();

    // If it starts with {, assume it's already JSON
    if trimmed.starts_with('{') {
        // Find matching closing brace
        let mut depth = 0;
        let mut end_idx = 0;
        for (i, c) in trimmed.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end_idx = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if end_idx > 0 {
            return &trimmed[..end_idx];
        }
    }

    // Try to find JSON anywhere in the response
    if let Some(start) = trimmed.find('{') {
        let remainder = &trimmed[start..];
        let mut depth = 0;
        for (i, c) in remainder.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &remainder[..=i];
                    }
                }
                _ => {}
            }
        }
    }

    trimmed
}
