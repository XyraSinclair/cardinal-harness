# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic
Versioning once it reaches `1.0.0`.

## [Unreleased]

### Added
- `cardinal judge --consortium m1,m2,...`: the consortium verdict primitive.
  Each judge measures the full Z₂³ orbit; complete orbits become judgment
  packets (`--packets-out`) and the belief is computed by fusing them —
  composition of the orbit transform, the judgment packet, and the robust
  solver into one operation with an explicit error budget (within-judge
  orbit-bias rms, cross-judge spread, direction unanimity, shared-bias
  residual correlation). Live smoke on a Manifund ACX pair: 3 judges,
  24 comparisons, $0.021, unanimous direction with per-judge coherence
  0.049–0.572.
- An experimental ordered-probit module for ladder-valued judgements, with
  symmetric cut construction, interval-censored likelihood fitting, a declared
  weak prior, gauge-projected covariance, and zero-spend synthetic comparison
  against the former point-center model. It remains off the production path
  until contaminated-channel and calibration gates pass.

### Changed
- `cardinal canonize --budget` is now the TOTAL comparison budget across
  every sort the protocol runs (accepted + candidates × judges), divided
  evenly, with the projected sort count printed before any spend and a loud
  error when the budget cannot cover the sorts. The old per-(candidate,
  judge) reading was a measured footgun: the Manifund P1 run turned
  `--budget 240` into ~1,900 comparisons and a 20-minute silent run.
- Proposal-JSON parsing (`slate`, `weigh --propose`, `canonize --propose`,
  `explain --propose`, `distinguish --propose`) is now lenient — whole
  completion parsed first, then the first balanced JSON span — and an
  empty or unparseable completion earns exactly one retry. Both failure
  modes were measured on the Manifund P1 run (deepseek intermittent empty
  completions; gpt-5.4-mini's valid-but-decorated `{"[]": [...]}` envelope,
  which the old first-bracket slice turned into a parse error).
- Point observations now use explicit measured `precision` when present and
  unit precision otherwise. Removed the anti-calibrated
  `eps_confidence`/`gamma_confidence` transform and planner
  `default_confidence`; model-stated confidence remains trace metadata.
  The deterministic method suite moves from ratio 0.648 versus ordinal 0.726
  under the old transform to ratio 0.808 versus ordinal 0.726, and three named
  cases now match full-budget Likert tau at half the comparison budget.
- Renamed spectral, leave-one-out, and multi-attribute diagnostic APIs to say
  what they contain rather than using a generic audit-artifact label.
- Corrected install and release documentation: source installs track `main`,
  tagged binaries come from GitHub Releases, and the crate is not currently
  published to crates.io.

## [0.8.0] - 2026-07-04

### Added
- `cardinal calibrate`: null-pair artifact measurement — identical text in
  both slots; directional mass = pure position+letter prior. Live study:
  four models measured clean (parity 1.000, bias 0.0000 nats) at the null
  point.
- Multi-attribute diagnostics on every multi-attribute response: the Pareto
  front (non-dominated on weight-oriented posterior means) and the
  attribute correlation matrix (planted trade-off test pins a negative
  off-diagonal). Cross-attribute information SHARING remains open (#44).
- Fixed-budget planner accuracy benchmark alongside first-hit-time, after
  catching the flicker artifact in exact-set first-hit metrics.

### Changed
- Exploration anchor diversity (issue #43): quantile-rotating anchors
  (chain fallback) replace the hub-and-spoke single-anchor geometry.
  Measured: global-tau regret flipped to a planner WIN (ratio 0.92);
  scarce-budget accuracy now favors the planner (budget 60: tau 0.894 vs
  0.871, top-5 12/16 vs 10/16).
- The synthetic ratio-vs-ordinal suite relationship FLIPPED under the new
  geometry (ordinal 0.726 vs ratio 0.648) — re-pinned with measurement
  history preserved; live logprob-PMF evidence is unaffected.

## [0.7.0] - 2026-07-04

### Added
- `ordinal_letter_v1`: the seriate three-token direction instrument
  (A / B / =) as a second evidence template — the cheapest logprob-native
  path; direction PMFs enter the solver at fixed modest magnitude with
  measured uncertainty.
- Order-residual diagnostic: for pairs asked in both orders in evidence mode,
  the mean |sum of presented-coordinate log-ratio means| — position bias in
  nats, per run (`evidence_order_residual_mean_abs`; ~0 for an unbiased
  judge, large under pure position bias; strictly richer than binary flip
  counts).
- `cardinal sort --estimate`: worst-case comparisons, per-call tokens, and
  provider dollars before any network or cache touch — with per-template
  honesty (single-letter evidence calls cap at 16 output tokens, ~100x
  cheaper worst case than the JSON path).
- Planner regret benchmark (`tests/planner_regret.rs`): comparisons-to-
  answer for the active planner vs uniform random pair selection.

### Findings (measured, pinned two-sided)
- HONEST NEGATIVE: the current planner LOSES to uniform random pair
  selection at n=20 under a noisy simulated judge — on top-5
  identification (~134.7 vs ~86.7 comparisons) and global tau (~51.3 vs
  ~47.3); the gap widens with noise. README claims tempered; fix cycle
  tracked in #43 with the benchmark as the instrument.

## [0.6.0] - 2026-07-04

### Added
- The seriate evidence path (`--template ratio_letter_v1`): single-token
  ratio-letter elicitation whose answer-position top-k logprobs form the
  judgement PMF; rendering/parsing delegated to the `seriate` crate (no
  prompt duplication, cache identity derived from seriate's content-
  addressed template hash).
- Explicit-precision observations: `Observation::from_log_ratio_moments`
  feeds PMF mean/variance into the IRLS solver directly, replacing the
  `g(c)` stated-confidence mapping for evidence-mode judgements.
- Evidence health diagnostics in response meta and the sort summary line:
  `evidence_judgements`, `logprob_mode_judgements`,
  `evidence_visible_mass_mean`.
- Loud degradation: providers that reject the logprobs parameter
  (reasoning-class models) or silently omit logprobs fall back to sampled
  mode, visibly in run metadata.
- Cache schema: nullable `log_ratio_mean` / `log_ratio_var` /
  `visible_mass` columns; evidence moments survive cache replay.
- Live study: at equal budget and cost on gpt-5.4-mini the PMF path
  yields ~3x the top-to-bottom separation per dollar (4.0 sigma vs 1.4
  sigma); instruments agree at Spearman 0.74 — documented honestly.

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
  `entities_pruned` count in response meta.
- Live taste-tooling study pack under
  `artifacts/live/taste-tools-demo-2026-07-02/` showing attribute recovery:
  explain identifies the criterion that actually generated a ranking (ρ=+0.98,
  weight 0.85) against three LLM-proposed decoys.

## [0.3.0] - 2026-07-02

### Added
- Counterbalanced comparisons: `counterbalance_pairs` on rerank requests asks
  every planned pair in both presentation orders, cancelling position bias
  per-pair; `pairs_counterbalanced` / `position_flips` diagnostics in response
  meta. Default ON for the `sort` surface (`--no-counterbalance` to opt out).
- Attribute health probes on `sort`: `--two-sided` judges the opposite of the
  criterion ("lack of X", weight −1) and `--also-by` judges paraphrases; both
  report sign-adjusted Spearman rank-consistency diagnostics (`probes` in JSON
  output, verdict lines on stderr).
- Natural ordinal prompt template `ordinal_v1` (direction + confidence only),
  entering the solver as a fixed modest log-ratio shared with the synthetic
  ordinal mode (`ORDINAL_OBSERVATION_RATIO`).
- Live healthy-elicitation study pack under
  `artifacts/live/healthy-sort-demo-2026-07-02/`: a real Sonnet 4.6 run
  measuring 11/51 order flips, +0.81 opposite-side consistency, and a +0.35
  (shaky) paraphrase.

## [0.2.0] - 2026-07-02

### Added
- `cardinal sort`: sort newline-delimited items (or a JSON array) from a file
  or stdin by a natural-language criterion, with `--scores`, `--reverse`,
  `--format text|json|jsonl|csv`, `--top-k`, `--budget`, `--trace`,
  `--cache-only` (keyless offline replay), and one-line cost/stop accounting
  on stderr. Refuses to print output when every comparison failed.
- Library conveniences `sort_texts` / `sort_documents` (`rerank::sort`) over
  the single-attribute rerank path, including a middle-boundary default for
  whole-list sorts (a `top_k = n` degenerate case would stop before the first
  comparison).
- Tag-triggered release workflow building `cardinal` binaries for six targets
  with sha256 checksums.
- Tight crates.io packaging (explicit `include`, ~50 files), docs.rs metadata,
  and `CITATION.cff`.
- Live `cardinal sort` demo study pack under
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
