# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic
Versioning once it reaches `1.0.0`.

## [Unreleased]

## [0.5.0] - 2026-07-02

### Added
- Adversarial test battery: six new suites, 74 tests (266 total across 27
  suites) attacking solver recovery (planted truth, Huber influence bounds,
  gauge invariance, confidence weighting, ladder monotonicity), metamorphic
  invariances of the sort path, uncertainty calibration coverage, a
  pathological-judge taxonomy (position-biased, intransitive, compressed,
  refusing, gaslighting, format-vandal), method head-to-heads vs Likert and
  ordinal baselines, and planner/pruning/stopping efficiency. Authored and
  adversarially reviewed by independent agents; see docs/TESTING.md.

### Fixed
- `solve_irls_huber`: MAD outlier-scale estimate collapsed when residuals
  were tied up to floating-point noise (absolute 1e-18 zero-guard), clipping
  every edge and crushing the fit by 3–4 orders of magnitude. Now falls back
  to the max-abs scale when MAD is below 1e-8 of the max-abs residual.
  Found by the battery's adversarial review; regression test pinned to the
  hand-solved normal equations.
- Synthetic evaluation gate-prewarm loop could overrun `comparison_budget`
  before the main loop's budget check ever ran; prewarm now spends from and
  stops at the same budget.

## [0.4.0] - 2026-07-02

### Added
- `cardinal judge`: single fully-transparent pairwise judgement (`--show-prompt`
  prints the rendered system+user prompt; `--json` for structured output;
  ratio, ordinal, and bucket templates).
- `cardinal elaborate` and `sort --elaborate`: one LLM call expands a terse
  criterion into a precise judging rubric (definition, what counts, what must
  not be rewarded), printed and used verbatim as the attribute prompt.
- `cardinal explain`: reverse-engineer an existing ranking — measure candidate
  attributes (user-supplied and/or `--propose`d by an LLM) against a believed
  order, report per-attribute Spearman and fitted non-negative weights
  (`explain_ranking` / `propose_candidates` in the library).
- Top-k exploration pruning: `prune_p_topk_below` on top-k specs (and
  `--prune-below` on `sort`) stops spending forced-exploration comparisons on
  items whose posterior chance of reaching the top-k is negligible;
  `entities_pruned` receipt in response meta.
- Live taste-tooling receipt pack under
  `artifacts/live/taste-tools-demo-2026-07-02/` showing attribute recovery:
  explain identifies the criterion that actually generated a ranking (ρ=+0.98,
  weight 0.85) against three LLM-proposed decoys.

## [0.3.0] - 2026-07-02

### Added
- Counterbalanced comparisons: `counterbalance_pairs` on rerank requests asks
  every planned pair in both presentation orders, cancelling position bias
  per-pair; `pairs_counterbalanced` / `position_flips` receipts in response
  meta. Default ON for the `sort` surface (`--no-counterbalance` to opt out).
- Attribute health probes on `sort`: `--two-sided` judges the opposite of the
  criterion ("lack of X", weight −1) and `--also-by` judges paraphrases; both
  report sign-adjusted Spearman rank-consistency receipts (`probes` in JSON
  output, verdict lines on stderr).
- Natural ordinal prompt template `ordinal_v1` (direction + confidence only),
  entering the solver as a fixed modest log-ratio shared with the synthetic
  ordinal mode (`ORDINAL_OBSERVATION_RATIO`).
- Live healthy-elicitation receipt pack under
  `artifacts/live/healthy-sort-demo-2026-07-02/`: a real Sonnet 4.6 run
  measuring 11/51 order flips, +0.81 opposite-side consistency, and a +0.35
  (shaky) paraphrase.

## [0.2.0] - 2026-07-02

### Added
- `cardinal sort`: sort newline-delimited items (or a JSON array) from a file
  or stdin by a natural-language criterion, with `--scores`, `--reverse`,
  `--format text|json|jsonl|csv`, `--top-k`, `--budget`, `--trace`,
  `--cache-only` (keyless offline replay), and a one-line cost/stop receipt
  on stderr. Refuses to print output when every comparison failed.
- Library conveniences `sort_texts` / `sort_documents` (`rerank::sort`) over
  the single-attribute rerank path, including a middle-boundary default for
  whole-list sorts (a `top_k = n` degenerate case would stop before the first
  comparison).
- Tag-triggered release workflow building `cardinal` binaries for six targets
  with sha256 checksums.
- Tight crates.io packaging (explicit `include`, ~50 files), docs.rs metadata,
  and `CITATION.cff`.
- Live `cardinal sort` demo receipt pack under
  `artifacts/live/sort-demo-2026-07-02/`.
- Fixed CI checks under current stable toolchain (rustfmt/clippy/rustdoc).
- Updated transitive dependency `bytes` to address RUSTSEC-2026-0007.
- Added a Likert baseline synthetic eval runner (`cardinal eval-likert`) for comparisons.

### Removed
- Retired prompt template `canonical_v2_attr_first`. Empirically tested in a
  comprehensive prompt layout sweep (4 variants × 7 models × 8 attributes) and
  found to offer no advantage over `canonical_v2`. The slug still resolves to
  `canonical_v2` for backward compatibility but is no longer a distinct template.

## [0.1.0] - 2026-01-31

- Initial public release.
