use super::*;
use crate::discrete::{DiscreteDistribution, WeightedValue};
use crate::gateway::types::{Attribution, ChatModel, ChatRequest, Message, TokenAlternative};
use uuid::Uuid;

#[test]
fn test_confidence_from_logprobs_basic() {
    let ladder = vec![1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.1, 2.5];

    let logprobs = vec![
        TokenLogprob {
            token: "\"A\"".to_string(),
            logprob: -0.1,
            top_alternatives: vec![TokenAlternative {
                token: "\"B\"".to_string(),
                logprob: -2.4,
            }],
        },
        TokenLogprob {
            token: "2.5".to_string(),
            logprob: -0.22_f64, // ~0.80 probability
            top_alternatives: vec![
                TokenAlternative {
                    token: "2.1".to_string(),
                    logprob: -1.61, // ~0.20
                },
                TokenAlternative {
                    token: "3.1".to_string(),
                    logprob: -4.61, // ~0.01
                },
            ],
        },
    ];

    let result = confidence_from_logprobs(&logprobs, PairwisePreferredSide::A, 2.5, &ladder);
    assert!(result.is_some(), "should extract confidence");

    let cs = result.unwrap();
    match cs {
        ConfidenceSource::Logprob {
            entropy,
            top_prob,
            neighborhood_prob,
        } => {
            assert!(top_prob > 0.7 && top_prob < 0.9, "top_prob={top_prob}");
            assert!(
                neighborhood_prob >= top_prob,
                "neighborhood should include top"
            );
            assert!(entropy > 0.0, "entropy should be positive");
        }
        _ => panic!("expected Logprob variant"),
    }
}

#[test]
fn test_confidence_from_logprobs_missing_ratio() {
    let ladder = vec![1.0, 1.5, 2.0];
    let logprobs = vec![
        TokenLogprob {
            token: "\"A\"".to_string(),
            logprob: -0.1,
            top_alternatives: vec![],
        },
        TokenLogprob {
            token: "hello".to_string(),
            logprob: -0.1,
            top_alternatives: vec![],
        },
    ];

    // Ratio 2.0 not in any logprob token.
    let result = confidence_from_logprobs(&logprobs, PairwisePreferredSide::A, 2.0, &ladder);
    assert!(result.is_none());
}

#[test]
fn test_confidence_from_logprobs_handles_ratio_one_without_prefix_collision() {
    let ladder = vec![1.0, 1.05, 12.7];
    let logprobs = vec![
        TokenLogprob {
            token: "\"A\"".to_string(),
            logprob: -0.1,
            top_alternatives: vec![TokenAlternative {
                token: "\"B\"".to_string(),
                logprob: -2.3,
            }],
        },
        TokenLogprob {
            token: "\"1.0\"".to_string(),
            logprob: -0.1,
            top_alternatives: vec![TokenAlternative {
                token: "12.7".to_string(),
                logprob: -2.0,
            }],
        },
    ];

    let result = confidence_from_logprobs(&logprobs, PairwisePreferredSide::A, 1.0, &ladder);
    assert!(result.is_some(), "should match the 1.0 token exactly");

    let cs = result.unwrap();
    match cs {
        ConfidenceSource::Logprob { top_prob, .. } => {
            assert!(top_prob > 0.8, "top_prob={top_prob}");
        }
        _ => panic!("expected Logprob variant"),
    }
}

#[test]
fn test_pairwise_logprob_posterior_captures_winner_uncertainty() {
    let ladder = vec![1.0, 1.5, 2.5];
    let confident_ratio = TokenLogprob {
        token: "2.5".to_string(),
        logprob: -0.05,
        top_alternatives: vec![TokenAlternative {
            token: "1.5".to_string(),
            logprob: -3.5,
        }],
    };
    let confident_winner = TokenLogprob {
        token: "\"A\"".to_string(),
        logprob: -0.05,
        top_alternatives: vec![TokenAlternative {
            token: "\"B\"".to_string(),
            logprob: -3.5,
        }],
    };
    let ambiguous_winner = TokenLogprob {
        token: "\"A\"".to_string(),
        logprob: -0.7,
        top_alternatives: vec![TokenAlternative {
            token: "\"B\"".to_string(),
            logprob: -0.75,
        }],
    };

    let high_confidence = pairwise_logprob_posterior(
        &[confident_winner.clone(), confident_ratio.clone()],
        PairwisePreferredSide::A,
        2.5,
        &ladder,
    )
    .expect("posterior");
    let low_confidence = pairwise_logprob_posterior(
        &[ambiguous_winner, confident_ratio],
        PairwisePreferredSide::A,
        2.5,
        &ladder,
    )
    .expect("posterior");

    let high_b = high_confidence
        .higher_ranked_distribution
        .probability_of(|side| *side == PairwisePreferredSide::B);
    let low_b = low_confidence
        .higher_ranked_distribution
        .probability_of(|side| *side == PairwisePreferredSide::B);
    assert!(high_b < 0.1);
    assert!(low_b > 0.4);
}

#[test]
fn test_ratio_bucket_roundtrips_and_pairwise_answer_maps_to_latent() {
    let bucket = RatioBucket::from_ratio(2.5).expect("bucket");
    assert_eq!(bucket, RatioBucket::R08);
    assert!((bucket.ratio() - 2.5).abs() < 1e-9);

    let answer = PairwiseAnswer::observation(PairwisePreferredSide::B, bucket);
    assert_eq!(answer.preferred_side(), Some(PairwisePreferredSide::B));
    assert_eq!(answer.ratio_bucket(), Some(RatioBucket::R08));
    assert_eq!(answer.ratio(), Some(2.5));
    assert!(answer.signed_ln_ratio().expect("latent") < 0.0);
}

#[test]
fn test_signed_log_ratio_distribution_operator_overloads() {
    let left = SignedLogRatioDistribution::from_answer_distribution(&DiscreteDistribution::new(
        vec![
            WeightedValue {
                value: PairwiseAnswer::A(RatioBucket::R05),
                probability: 0.5,
            },
            WeightedValue {
                value: PairwiseAnswer::B(RatioBucket::R00),
                probability: 0.5,
            },
        ],
        0.0,
    ));
    let right = SignedLogRatioDistribution::from_answer_distribution(&DiscreteDistribution::new(
        vec![WeightedValue {
            value: PairwiseAnswer::A(RatioBucket::R05),
            probability: 1.0,
        }],
        0.0,
    ));

    let added = left.clone() + right.clone();
    let negated = -right.clone();
    let subtracted = added.clone() - right.clone();
    let scaled = right.clone() * 2.0;

    assert!(added.mean().expect("mean") > right.mean().expect("mean"));
    assert!(negated.mean().expect("mean") < 0.0);
    assert!(subtracted.mean().expect("mean") < added.mean().expect("mean"));
    assert!(scaled.mean().expect("mean") > right.mean().expect("mean"));
}

#[test]
fn test_token_parsers_handle_common_structured_output_noise() {
    assert_eq!(token_numeric_value(".5"), Some(0.5));
    assert_eq!(token_numeric_value("Ratio: 1.05"), Some(1.05));
    assert_eq!(token_numeric_value("1.0.5"), None);

    assert_eq!(
        pairwise_preferred_side_for_token("**A**"),
        Some(PairwisePreferredSide::A)
    );
    assert_eq!(
        pairwise_preferred_side_for_token("B."),
        Some(PairwisePreferredSide::B)
    );
}

#[test]
fn test_push_merged_float_probability_updates_centroid() {
    let mut support = Vec::new();
    push_merged_float_probability(&mut support, 1.0, 0.5, 1e-12);
    push_merged_float_probability(&mut support, 1.0 + 0.9e-12, 0.5, 1e-12);

    assert_eq!(support.len(), 1, "values within tolerance should merge");
    assert!((support[0].probability - 1.0).abs() < 1e-12);
    assert!((support[0].value - (1.0 + 0.45e-12)).abs() < 1e-13);
}

#[test]
fn test_signed_log_ratio_distribution_bimodality_survives_latent_projection() {
    let bimodal = SignedLogRatioDistribution::from_answer_distribution(&DiscreteDistribution::new(
        vec![
            WeightedValue {
                value: PairwiseAnswer::A(RatioBucket::R00),
                probability: 0.5,
            },
            WeightedValue {
                value: PairwiseAnswer::A(RatioBucket::R16),
                probability: 0.5,
            },
        ],
        0.0,
    ));
    let local_blur =
        SignedLogRatioDistribution::from_answer_distribution(&DiscreteDistribution::new(
            vec![
                WeightedValue {
                    value: PairwiseAnswer::A(RatioBucket::R07),
                    probability: 0.5,
                },
                WeightedValue {
                    value: PairwiseAnswer::A(RatioBucket::R08),
                    probability: 0.5,
                },
            ],
            0.0,
        ));

    assert!((bimodal.probability_positive() - 0.5).abs() < 1e-12);
    assert!((local_blur.probability_positive() - 1.0).abs() < 1e-12);
    assert!(
        bimodal.variance().expect("variance") > local_blur.variance().expect("variance"),
        "far-apart magnitude uncertainty should inflate latent variance more than local blur"
    );
}

#[test]
fn test_signed_log_ratio_distribution_long_run_convolution_preserves_mass_and_mean() {
    let base = SignedLogRatioDistribution::new(
        DiscreteDistribution::new(
            vec![
                WeightedValue {
                    value: -0.4,
                    probability: 0.4,
                },
                WeightedValue {
                    value: 1.2,
                    probability: 0.6,
                },
            ],
            0.0,
        ),
        0.0,
    );

    let analytical_mean = base.mean().expect("mean");
    let analytical_variance = base.variance().expect("variance");
    let mut current = base.clone();
    let steps = 24;

    for _ in 0..steps {
        current = (current + base.clone()).compress(32);
        assert!(
            (current.total_probability() - 1.0).abs() < 1e-9,
            "probability mass should stay normalized after repeated convolution/compression"
        );
    }

    let expected_mean = analytical_mean * (steps + 1) as f64;
    let expected_variance = analytical_variance * (steps + 1) as f64;
    assert!((current.mean().expect("mean") - expected_mean).abs() < 1e-9);
    assert!(
        current.variance().expect("variance") <= expected_variance + 1e-9,
        "compression should not hallucinate extra variance"
    );
}

#[test]
fn test_pairwise_logprob_posterior_tracks_residual_mass_from_unmodeled_alternatives() {
    let ladder = vec![1.0, 1.5];
    let posterior = pairwise_logprob_posterior(
        &[
            TokenLogprob {
                token: "\"A\"".to_string(),
                logprob: 0.4f64.ln(),
                top_alternatives: vec![TokenAlternative {
                    token: "\"garbage\"".to_string(),
                    logprob: 0.6f64.ln(),
                }],
            },
            TokenLogprob {
                token: "1.5".to_string(),
                logprob: 0.5f64.ln(),
                top_alternatives: vec![],
            },
        ],
        PairwisePreferredSide::A,
        1.5,
        &ladder,
    )
    .expect("posterior");

    assert!((posterior.higher_ranked_distribution.support_probability() - 0.4).abs() < 1e-9);
    assert!((posterior.higher_ranked_distribution.residual_probability - 0.6).abs() < 1e-9);
    assert!((posterior.ratio_distribution.support_probability() - 0.5).abs() < 1e-9);
    assert!((posterior.ratio_distribution.residual_probability - 0.5).abs() < 1e-9);
    assert!((posterior.answer_distribution.support_probability() - 0.2).abs() < 1e-9);
    assert!((posterior.answer_distribution.residual_probability - 0.8).abs() < 1e-9);
    assert!((posterior.answer_distribution.total_probability() - 1.0).abs() < 1e-9);
}

#[test]
fn test_pairwise_logprob_posterior_exposes_answer_and_latent_distributions() {
    let ladder = vec![1.0, 1.5, 2.5];
    let posterior = pairwise_logprob_posterior(
        &[
            TokenLogprob {
                token: "\"A\"".to_string(),
                logprob: -0.1,
                top_alternatives: vec![TokenAlternative {
                    token: "\"B\"".to_string(),
                    logprob: -2.3,
                }],
            },
            TokenLogprob {
                token: "2.5".to_string(),
                logprob: -0.22,
                top_alternatives: vec![TokenAlternative {
                    token: "1.5".to_string(),
                    logprob: -1.61,
                }],
            },
        ],
        PairwisePreferredSide::A,
        2.5,
        &ladder,
    )
    .expect("posterior");

    assert_eq!(posterior.selected_ratio_bucket, RatioBucket::R08);
    assert_eq!(
        posterior.selected_answer,
        PairwiseAnswer::A(RatioBucket::R08)
    );
    assert!(posterior.answer_distribution.top_probability() > 0.6);
    assert!(posterior.mean_signed_ln_ratio().expect("mean") > 0.0);
    assert!(posterior.variance_signed_ln_ratio().expect("variance") >= 0.0);
}

#[test]
fn test_chat_request_logprobs_builder() {
    let req = ChatRequest::new(
        ChatModel::openrouter("test/model"),
        vec![Message::user("hi")],
        Attribution::new("test"),
    )
    .with_logprobs(5);

    assert!(req.logprobs);
    assert_eq!(req.top_logprobs, Some(5));
}

#[test]
fn test_chat_request_default_no_logprobs() {
    let req = ChatRequest::new(
        ChatModel::openrouter("test/model"),
        vec![Message::user("hi")],
        Attribution::new("test"),
    );

    assert!(!req.logprobs);
    assert!(req.top_logprobs.is_none());
}

#[test]
fn test_attribution_with_api_key_builder() {
    let api_key_id = Uuid::new_v4();
    let attribution = Attribution::new("test").with_api_key(api_key_id);

    assert_eq!(attribution.api_key_id, Some(api_key_id));
    assert_eq!(attribution.caller, "test");
}
