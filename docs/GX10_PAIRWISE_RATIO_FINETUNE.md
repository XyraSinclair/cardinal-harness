# GX10 Pairwise Ratio Fine-Tune

Concrete training spec for a `gx10` fine-tuned pairwise ratio judge that drops
into cardinal-harness as a better local measurement instrument.

This document is intentionally narrow. The fine-tuned model is not responsible
for global ranking, uncertainty propagation, or stopping. Those remain in the
solver and rerank loop:

- prompt contract and ratio ladder: `src/prompts.rs`
- pairwise parse contract: `src/rerank/comparison.rs`
- confidence weighting and robust IRLS: `src/rating_engine.rs`
- planner and stopping: `src/rating_engine.rs`, `src/rerank/multi.rs`
- synthetic evaluation harness: `src/rerank/evaluation.rs`

## 1. Goal

Train a model that is better than the current general-purpose comparison models
at exactly one task:

1. Read `(attribute, entity_A_text, entity_B_text)`.
2. Emit a valid structured judgement over the existing ratio ladder.
3. Be more calibrated, more antisymmetric, and less noisy.
4. Improve end-to-end top-k quality per unit cost when embedded in the existing
   rerank loop.

The target is not "maximize pairwise agreement in the abstract." The target is:

- fewer bad local measurements
- better uncertainty honesty
- fewer redundant comparisons before certified stop

## 1.1 First Experiment Narrowing

The full roadmap below is broader than the first run should be.

Default v0:

- supervised fine-tuning or distillation only
- exact ladder-valued outputs only
- no RL
- held-out session replay as the ship/no-ship gate

If that supervised baseline does not win in replay, do not add RL yet.

## 2. Non-goals

Do not train the model to:

- produce chain-of-thought or visible rationales
- replace the IRLS solver
- replace the planner
- learn direct listwise ranking over a whole candidate set
- invent new ratio scales at runtime

Prompt experiments in `docs/PROMPTS.md` already suggest that reasoning-heavy
judge prompts inflate ratios and reduce compatibility. Keep the model narrow.

## 3. Runtime contract

### 3.1 Compatibility target

The runtime output must remain compatible with the current parser:

```json
{"higher_ranked":"A","ratio":2.1,"confidence":0.74}
```

or:

```json
{"refused":true}
```

This preserves drop-in compatibility with `parse_pairwise_response()` in
`src/rerank/comparison.rs`.

For training, be stricter than runtime:

- snap targets to the exact configured ratio ladder
- treat off-ladder floats as data-quality noise, not desired behavior
- canonicalize `ratio == 1.0` to one winner convention in the training labels
  so the model does not waste probability mass on a degenerate distinction

### 3.2 Preferred structured decoding target

If `gx10` supports JSON schema or function-call style decoding, prefer the
following canonical answer shape during training and inference:

```json
{
  "refused": false,
  "higher_ranked": "A",
  "ratio_bucket": "2.1",
  "confidence_mode": "posterior",
  "confidence": 0.74
}
```

Notes:

- `ratio_bucket` is an enum over the fixed ladder.
- `confidence_mode` is optional metadata. It lets us distinguish:
  `self_report`, `posterior`, or `calibrated_projection`.
- A thin adapter can map `ratio_bucket` back to numeric `ratio`.

If schema-constrained decoding is unavailable, fall back to the current plain
JSON contract and compute the latent class loss by parsing the generated JSON.

## 4. Prediction space

Treat the task as discrete structured classification, not scalar regression.

Primary answer state space:

- 17 ratio buckets
- 2 winner choices
- 1 refusal state

Total semantic states:

- `2 x 17 + 1 = 35`

Represent each non-refusal answer as:

- `winner in {A, B}`
- `ratio_bucket in RATIO_LADDER`

Convert to the solver's latent space with:

- signed log-ratio `y = +ln(r)` if `A`
- signed log-ratio `y = -ln(r)` if `B`

This matches the current algebra in `compute_ln_ratio()`.

## 5. Structured input format

The model should be trained on normalized structured records, even if they are
rendered as chat messages at serving time.

Canonical training example:

```json
{
  "schema_version": "gx10_pairwise_ratio_v1",
  "pair_id": "clarity::doc_17::doc_42",
  "attribute": {
    "id": "clarity",
    "prompt": "Which response is clearer, more direct, and easier to trust?"
  },
  "entity_a": {
    "id": "doc_17",
    "text": "..."
  },
  "entity_b": {
    "id": "doc_42",
    "text": "..."
  },
  "presentation": {
    "orientation": "A_first",
    "is_swap_augmentation": false
  },
  "targets": {
    "teacher_distribution": {
      "A@1.3": 0.10,
      "A@1.5": 0.24,
      "A@1.75": 0.36,
      "A@2.1": 0.20,
      "B@1.0": 0.00,
      "refused": 0.10
    },
    "winner_probability_a": 0.90,
    "signed_ln_ratio_mean": 0.53,
    "signed_ln_ratio_variance": 0.18,
    "calibrated_confidence": 0.71
  },
  "metadata": {
    "source_split": "panel_distill",
    "attribute_family": "quality",
    "difficulty": "frontier_near_tie",
    "near_topk_boundary": true
  }
}
```

The model does not need the `targets` or `metadata` fields at inference time.
Those are for training and evaluation only.

## 6. Data sources

Use a mixture of four data tiers.

### 6.1 Tier A: synthetic supervision

Use the synthetic rerank harness in `src/rerank/evaluation.rs` to generate
triples with known latent truth:

- true winner
- true ratio bucket
- true noise level
- true uncertainty

Synthetic data is useful for:

- early bootstrapping
- exact calibration checks
- reward estimation for session-level rollouts

Synthetic data is not enough by itself because the text side of the task is
still natural language and open-ended.

### 6.2 Tier B: panel distillation on real pairs

For each real pair:

1. Query a diverse teacher panel.
2. Query both presentation orders.
3. Canonicalize all answers into unordered-pair space.
4. Build a soft teacher posterior over the 35 states.

Recommended teacher pool:

- one strong OpenAI model
- one strong Anthropic model
- one strong Gemini/Kimi/DeepSeek class model
- optionally one smaller but fast model for diversity

The point is not majority vote by brand prestige. The point is to avoid baking
one provider's local biases into the student.

### 6.3 Tier C: hard cases mined from the rerank loop

Mine examples that are disproportionately valuable:

- near-ties on the ratio ladder
- pairs near the top-k boundary
- pairs with high repeat instability
- pairs that triggered Huber downweighting downstream
- pairs with high teacher disagreement
- pairs that frequently produced refusals or malformed JSON

This is the highest-leverage subset for fine-tuning.

### 6.4 Tier D: audited gold data

Keep a smaller, high-quality set of human-audited or deeply adjudicated cases.

Use this for:

- final calibration checks
- refusal policy checks
- regression gating

Do not train primarily on this if it is small. Use it as a hard benchmark.

## 7. Target construction

### 7.1 Canonicalization

Every teacher answer should be mapped into canonical unordered-pair space.

If a teacher sees `(B, A)` and answers:

```json
{"higher_ranked":"B","ratio":2.1,"confidence":0.8}
```

then the canonical orientation `(A, B)` target becomes:

- `higher_ranked = A`
- `ratio = 2.1`

This is mandatory. Otherwise the student will learn position artifacts.

### 7.2 Soft answer posterior

Build a soft distribution over the 35 semantic states rather than a hard label.

Suggested construction:

1. Parse every valid teacher answer into one of 35 states.
2. Weight each teacher by a calibrated reliability weight.
3. Average in canonical orientation.
4. Smooth locally across neighboring ratio buckets.

Local smoothing is important because adjacent ladder values are semantically
close in log-space. For example, if teachers split between `1.5` and `1.75`,
the target should not treat those as maximally different classes.

Initial smoothing rule:

- 70% mass on selected bucket
- 15% on each immediate neighbor
- renormalize at ladder boundaries

Only apply smoothing when the teacher target started as a hard one-hot label.
If the teacher already exposes a posterior over ratio buckets, use that instead.

### 7.3 Confidence target

Do not use raw self-reported confidence as the only target.

Construct a calibrated confidence target from:

- teacher agreement entropy
- swap consistency
- repeat stability
- optional logprob posterior entropy when available

Recommended target:

```text
confidence_target = 0.5 * teacher_top_prob
                  + 0.3 * swap_consistency
                  + 0.2 * repeat_stability
```

Then calibrate that scalar against actual correctness on a held-out set.

### 7.4 Refusal target

Refusal should be rare and explicit.

Mark `refused` as positive only when one of the following is true:

- policy block is real
- entity text is missing/corrupt/truncated beyond interpretation
- attribute prompt is malformed or undefined
- the task is genuinely impossible from the supplied text

Teacher hesitation is not enough. Near-uncertainty should become low confidence,
not refusal.

## 8. Training objective

Start with supervised fine-tuning. Use RL only after SFT is stable.

### 8.1 Base loss

Let:

- `p_theta(s | x)` be the model distribution over 35 semantic states
- `q(s | x)` be the teacher posterior

Primary answer loss:

```text
L_answer = CE(q, p_theta)
```

### 8.2 Swap consistency loss

For each pair, include both `(A, B)` and `(B, A)` views.

Transform the swapped prediction back into canonical space and penalize
disagreement:

```text
L_swap = KL(p_theta(s | A,B) || T[p_theta(s | B,A)])
```

where `T` flips winner and inverts ratio bucket.

This is the single most important structural regularizer.

### 8.3 Triangle consistency loss

For triplets `(a, b, c)`, define the expected signed log-ratio:

```text
mu_ab = E_theta[y_ab]
```

Penalize transitivity violation:

```text
L_tri = mean((mu_ab + mu_bc - mu_ac)^2)
```

This should be sampled sparsely. Full triplet enumeration is too expensive.

### 8.4 Calibration loss

Use a calibration term on winner correctness and local ratio neighborhood:

```text
L_cal = Brier(winner_event) + Brier(neighborhood_event)
```

If the model exposes a posterior directly, use ECE only for monitoring and keep
the training loss differentiable with Brier or log loss.

### 8.5 Refusal loss

False refusals are expensive because they waste comparisons and shrink the
usable graph. Penalize them asymmetrically:

```text
L_refuse = 3.0 * FP_refusal + 1.0 * FN_refusal
```

where the coefficients are starter values, not fixed truths.

### 8.6 Format validity loss

If the model is not schema-constrained, add a small auxiliary penalty for:

- invalid JSON
- missing required fields
- out-of-ladder ratio values

### 8.7 Initial combined loss

Good starting mixture:

```text
L_total = 1.0 * L_answer
        + 0.30 * L_swap
        + 0.15 * L_tri
        + 0.20 * L_cal
        + 0.35 * L_refuse
        + 0.05 * L_format
```

Tune these coefficients empirically. `L_answer` should dominate; the others are
regularizers and calibration constraints.

## 9. Optional RL / reward optimization stage

Only do this after SFT produces:

- near-perfect schema validity
- low false refusal rate
- clear gain on swap consistency

If `gx10` supports rollout-based optimization, optimize short rerank episodes
through the existing multi-rerank loop.

This is explicitly not part of v0. The first real run should be supervised.

### 9.1 Episode definition

One episode is:

1. sample a rerank task
2. run `multi_rerank` or an equivalent offline simulator
3. use the fine-tuned model as the comparison oracle
4. compute reward from final ranking metrics and cost

Use synthetic tasks first because they have clean ground truth.

### 9.2 Reward

Use a cost-normalized top-k reward, not a pure agreement reward.

Starter reward:

```text
R = 3.0 * topk_recall
  + 2.0 * topk_precision
  + 1.0 * curl_harmonic
  + 1.0 * coverage_95ci
  + 0.5 * kendall_tau_b
  - 0.002 * comparisons_attempted
  - 0.001 * provider_input_tokens
  - 2.0 * false_refusals
  - 1.0 * parse_failures
  - 1.5 * overconfidence_penalty
```

Use normalized units in practice. The constants above are only seed values.

### 9.3 Baseline-relative advantage

To reduce noise, optimize relative to a baseline judge:

```text
A = R(student) - R(baseline)
```

The baseline should be the best current untuned model/prompt combination.

### 9.4 Guardrails

Reject any candidate checkpoint that worsens either:

- refusal rate
- calibration

even if top-k reward improves a little. A brittle judge will cause trouble in
real deployments.

## 10. Benchmark protocol

Benchmark at three levels.

### 10.1 Pair-level benchmark

Measure on held-out pairs:

- valid JSON rate
- exact ladder validity rate
- winner accuracy vs gold
- bucketed ratio accuracy
- expected signed log-ratio error
- false refusal rate
- swap consistency error
- repeat stability
- winner ECE
- ratio-neighborhood calibration

### 10.2 Session-level benchmark

Run the fine-tuned model through the existing rerank harness and report:

- Kendall tau-b
- Spearman rho
- top-k precision
- top-k recall
- coverage @ 95% CI
- nDCG@K
- CURL
- weighted rank reversals
- Bayesian regret
- comparisons attempted
- comparisons used
- comparisons refused
- tokens and cost
- stop reason distribution

These metrics already exist in `src/rerank/evaluation.rs` and should remain the
primary decision criteria.

### 10.3 Cost-quality frontier

Plot both:

1. quality at fixed comparison budget
2. comparisons or tokens required to hit a fixed target quality

The second plot is often more important. If the student reaches the same top-k
quality with fewer comparisons, that is a real product win.

### 10.4 Replay protocol

Use full-cache replay, not trace-only replay.

Reason:

- once the student changes pairwise judgements, the planner can choose
  different future pairs
- a cache seeded only from historical traces can therefore fail with misses on
  exactly the cases where the student is meaningfully different

Recommended offline replay loop:

1. export labeled training data from held-in traces with `cardinal dataset-export`
2. export the full attribute x unordered-pair x orientation prompt grid for each
   held-out request with `cardinal prompt-grid-export`
3. run the fine-tuned student over that prompt grid and seed a SQLite cache
4. run `cardinal rerank --cache-only` against the held-out request using that
   cache
5. compare replayed response vs baseline response on:
   comparisons attempted, refusals, top-k overlap, and tolerated-error outcome

This is the cheapest faithful ship/no-ship gate for v0.

## 11. Data splits

Use leakage-resistant splits.

### 11.1 By entity cluster

Do not randomly split pairs. Split by entity clusters so the model cannot see
the same entities in both training and validation through different pairings.

### 11.2 By attribute family

Reserve some attribute families entirely for evaluation:

- quality
- truthfulness
- feasibility
- taste
- safety

This checks transfer, not memorization.

### 11.3 By source epoch

Keep a temporal holdout if the source data comes from live traces.

## 12. Recommended ablations

Run these before any rollout optimization:

1. hard labels vs soft teacher posterior
2. no swap loss vs swap loss
3. self-reported confidence target vs calibrated confidence target
4. plain JSON decoding vs schema-constrained decoding
5. panel distillation alone vs panel + hard-case mining
6. answer-only loss vs answer + triangle consistency
7. self-report confidence runtime vs posterior-derived runtime confidence

If a simple ablation wins, keep it. Do not overcomplicate the stack for style.

## 13. Deployment strategy

Do not replace all comparison traffic at once.

### 13.1 Ladder deployment

Use the fine-tuned judge first as:

- low-cost model for easy or mid-confidence pairs
- baseline model for synthetic and replay benchmarks

Escalate to a stronger general model when:

- posterior entropy is high
- refusal probability is high
- pair lies near the top-k frontier and uncertainty is still large

This fits the existing model ladder idea in `src/rerank/model_policy.rs`.

### 13.2 Confidence source policy

Preferred runtime order:

1. logprob posterior if available
2. fine-tuned calibrated confidence head
3. self-reported scalar confidence as fallback only

The repo already has a strong direction toward posterior-derived confidence in
`src/gateway/types.rs`.

## 14. Concrete implementation milestones

### M0: data plumbing

- export canonical pairwise traces from rerank runs
- add swap augmentation
- export full prompt grids for held-out replay
- add teacher panel aggregation job
- build train/val/test splits by entity cluster

### M1: benchmark harness

- pair-level held-out benchmark
- session-level synthetic benchmark
- cache-seeded held-out replay benchmark
- cost-quality frontier plots

### M2: SFT v0

- train on hard labels or soft teacher posterior
- enforce strict output validity
- compare against current best prompt baseline

### M3: structural regularization

- add swap loss
- add calibration loss
- add hard-case oversampling

### M4: posterior integration

- if provider support exists, train/evaluate with ratio-bucket posterior targets
- compare self-report confidence vs posterior-derived confidence

### M5: short-horizon reward tuning

- optimize baseline-relative session reward on synthetic tasks
- gate on calibration and refusal regressions

### M6: production ladder rollout

- deploy behind a model policy
- compare cost to certified stop against the current baseline

## 15. Decision rule for success

The fine-tune is worth shipping only if it clears all of:

1. no regression in schema validity
2. no regression in false refusal rate
3. improved swap consistency
4. improved calibration
5. improved session-level top-k quality at fixed cost, or same quality at lower
   cost

Inter-model agreement is useful, but only as a secondary metric. Agreement
without calibration and session wins is not enough.

## 16. Default recommendation

If we start this now, the best path is:

1. SFT first
2. soft teacher posterior targets
3. mandatory swap augmentation and swap loss
4. hard-case mining near the top-k frontier
5. posterior-derived confidence when the serving stack supports it
6. RL only after the SFT model is already clearly better in offline replay

That keeps the project aligned with the current cardinal-harness architecture:
the model produces better local measurements, and the existing solver does the
global work.
