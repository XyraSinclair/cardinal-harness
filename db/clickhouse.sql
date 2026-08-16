-- Ratiometer durable judgment records on scry ClickHouse (colo2).
-- Every pairwise judgment a judge model ever makes lands here, denormalized
-- and content-addressed: cache_key_hash covers (pair, attribute, model,
-- template), so replays and cache hits dedupe under ReplacingMergeTree.
-- Apply: clickhouse-client --multiquery < db/clickhouse.sql

CREATE DATABASE IF NOT EXISTS ratiometer;

CREATE TABLE IF NOT EXISTS ratiometer.judgments
(
    ts                    DateTime64(3),
    run_tag               LowCardinality(String),
    corpus                LowCardinality(String),
    model                 LowCardinality(String),
    served_model          LowCardinality(String),
    template              LowCardinality(String),
    attribute             String,
    attribute_prompt_hash FixedString(64),
    seed                  UInt32,
    entity_a              String,
    entity_b              String,
    entity_a_hash         FixedString(64),
    entity_b_hash         FixedString(64),
    cache_key_hash        FixedString(64),
    higher_ranked         LowCardinality(String),
    ratio                 Float64,
    confidence            Float64,
    -- logprob posterior (canonical_bucket_v1 / ratio_letter_v1 templates):
    -- dir_prob = P(chosen side) from the direction distribution; entropy,
    -- top_prob, neighborhood_prob from the answer distribution; posterior =
    -- full serialized PairwiseLogprobPosterior (incl. signed-ln-ratio PMF).
    dir_prob              Float64 DEFAULT 0,
    entropy               Float64 DEFAULT 0,
    top_prob              Float64 DEFAULT 0,
    neighborhood_prob     Float64 DEFAULT 0,
    posterior             String  DEFAULT '',
    swapped               Bool,
    cached                Bool,
    refused               Bool,
    input_tokens          UInt32,
    output_tokens         UInt32,
    error                 String DEFAULT ''
)
ENGINE = ReplacingMergeTree(ts)
ORDER BY (model, attribute_prompt_hash, cache_key_hash);
