# First principles: the type system of structured LLM judgement

Permute the smallest words and see what the space actually is; then check,
cell by cell, whether the repo occupies it. ✓ = implemented with receipts,
◐ = partial, ✗ = absent (and honestly so).

## 1. The primitives

Seven nouns, everything else is composition:

| Primitive | What it is | Where |
|---|---|---|
| **Entity** | a thing that can be judged (text shown to the judge) | seriate `Entity`, cardinal `MultiRerankEntity` |
| **Attribute** | one way an entity set wants to be ordered | `Attribute` (content-addressed: reworded = different) |
| **Presentation** | which entity in which slot, this call | `Presentation`, reflection algebra |
| **Judgement** | one elicited answer, typed by instrument | `PairwiseJudgement`, seriate `JudgementRecord` |
| **Evidence** | the judgement as a distribution, mass accounted | `AnswerEvidence` + `PmfCompleteness` |
| **Weight** | how much one judgement moves the fit | `g(c)` or measured PMF precision |
| **Posterior** | latents ± uncertainty over the set, per attribute | `CompiledPosterior`, rating engine solve |

## 2. The instrument grid

An instrument is a point in **arity × scale × output-form**:

- arity: 1 (pointwise) · 2 (pairwise) · k (setwise) · n (listwise)
- scale (Stevens): nominal (pick) · ordinal (order) · interval (differences
  meaningful) · ratio (ratios meaningful)
- output form: point (one answer) · sample-set (repeat draws) · **PMF**
  (answer-token logprobs — the model's whole prior in one call)

| Cell | Status | Notes |
|---|---|---|
| pairwise · ratio · point | ✓ `canonical_v2` | the founding instrument |
| pairwise · ratio · PMF | ✓ `ratio_letter_v1` | 52-letter alphabet; live receipt: ~3× separation/dollar vs point |
| pairwise · ordinal · point | ✓ `ordinal_v1` | direction + confidence JSON |
| pairwise · ordinal · PMF | ✓ `ordinal_letter_v1` | 3-token alphabet; cheapest instrument |
| pairwise · interval · any | ✗ | "how much more, additively" — meaningful only for bounded attributes; low priority, ratio subsumes on log scale |
| pointwise · ordinal/ratio (rubric) | ◐ scalar control | seriate `scalar` digit-PMF; a baseline, not a path — anchor drift is the documented disease |
| pointwise · ratio **to fixed anchors** | ✗ | classic magnitude estimation with pinned anchor entities; fixes anchor drift; cheap O(n); worth building |
| k-wise · nominal ("which is most") | ◐ | seriate `kwise` + lowering; not exposed in cardinal |
| k-wise · **best–worst** (MaxDiff) | ✗ | pick best AND worst of k → 2k−3 implied pairs per call; the classic efficiency instrument; highest-value missing cell |
| k-wise · full order of k | ✗ | k! answer space; PMF impossible past tiny k; likely not worth it |
| listwise · any | ✗ by design | context limits, position bias, silent drops — documented rejection |

**The permutation that matters most and is missing: best–worst scaling.**
One call yields the top and bottom of a k-subset — O(k) pairwise
implications for one call's price. It composes with everything here
(lowering to weighted pairwise evidence, exactly like k-wise).

## 3. What each scale admits (so we don't compute nonsense)

- ordinal: medians, ranks, concordance (tau). Means of ranks are already a
  liberty. ✓ we report tau/rank metrics for ordinal paths.
- interval: differences, means, Pearson. Ratios meaningless.
- ratio (ours, in log space): products/ratios meaningful; geometric means;
  the log map makes ratio evidence additive — the entire reason the solver
  is least-squares on log-ratios. ✓
- Aggregation discipline: PMF moments (mean, var of signed log-ratio) are
  the sufficient statistics we consume; escape mass never averaged in. ✓

## 4. Efficiency: bits, calls, dollars

Theory: a noiseless binary comparison yields ≤1 bit; full order of n needs
Θ(n log n) bits ⇒ Θ(n log n) point-comparisons; top-k identification needs
Θ(n) for fixed k. A PMF answer yields the model's whole conditional prior —
up to log₂(52) ≈ 5.7 bits per call at the same price as a point.

| Property | Status |
|---|---|
| per-call cost receipts (nanodollars, tokens) | ✓ everywhere |
| pre-run worst-case pricing | ✓ `--estimate` (per-template honest: $0.011 vs $1.19) |
| budget defaults O(n), not O(n²) | ✓ 4·n |
| planner vs baseline measured | ✓ regret benchmark; wins scarce-budget, pinned two-sided |
| **bits-per-dollar as a formal receipt** | ✗ — we showed 3× separation/dollar informally; entropy-of-posterior-reduction per nanodollar is computable from what we already store |
| best–worst call efficiency | ✗ (see grid) |

## 5. Stability: the invariance group of a belief

Her question, formalized: a judgement deserves the name *belief* only if it
is a fixed point of the transformations that shouldn't matter. The group:

| Axis | Instrumented? |
|---|---|
| presentation order | ✓ counterbalance default + flip receipts + order-residual (nats) |
| polarity (attribute ↔ its opposite) | ✓ `--two-sided`, consistency receipt |
| paraphrase (same attribute, other words) | ✓ `--also-by`, consistency receipt |
| null content (pure prior) | ✓ `cardinal calibrate` (4 models measured clean) |
| **framing spin** (persuasive preamble pushing a side) | ✓ `cardinal judge --spin`: the same pair under neutral, pro-first, and pro-second requester framings × both orders; reports susceptibility χ in nats and whether the belief survives — "if you spin it, do they still agree" made a receipt |
| **temperature** | ✗ never swept — one JSD data point (t=0 logprobs vs t=1 samples, 0.128 on gpt-5.4-mini) is a hint, not a map |
| **reasoning effort / thinking params** | ✗ never swept; we only know reasoning models refuse logprobs |
| judge model | ◐ seriate probe compares models; no standing cross-model receipt in cardinal runs |
| time (same judge, days apart) | ✗ |

The stability axes we cover, we cover with receipts; the remaining ✗ rows
(parameter sweeps, time drift) are the cheapest untouched science in the repo.

The invariance group is now also an INSTRUMENT: `cardinal bench` (the Judge
Coherence Benchmark, docs/BENCHMARK.md) runs order swap, reciprocal
antisymmetry, frustration, spin, polarity, paraphrase, and null calibration
as a standardized 114-call battery per model, × a signal axis, → one
leaderboard number labs can hill-climb without ground-truth labels. The
benchmark validates itself against five scripted pathological judges in
`tests/judge_bench.rs`.

## 5½. The physics of a judge (receipts, not metaphors)

Prior elicitation behaves like a physical system, and each analogy lands on
a computable receipt:

| Physics | Judgement meaning | Receipt |
|---|---|---|
| **Frustration** (spin glass) | cyclic preference structure no scores can satisfy (A>B>C>A) | ✓ `judgement_frustration_mean` — the Hodge curl fraction Σλr²/Σλμ² of the log-ratio edge field, shipped; transitive judge ≈ quantization floor, planted rock-paper-scissors > 0.3 |
| **Hysteresis** | path dependence: judging A→B vs B→A | ✓ order-residual in nats + flip counts |
| **Susceptibility** | response to a small applied field: framing spin | ✓ `judge --spin`: χ = (m₊ − m₋)/2 nats per unit spin, plus a survives/echoes verdict; a scripted sycophant judge is caught in tests, a framing-blind judge measures χ = 0 |
| **Temperature/entropy** | PMF spread per judgement; annealing across sampling temperature | ◐ entropy computable from stored PMFs; sweep unmeasured |
| **Ground state** | the solved scores: minimum-energy potential for the field | ✓ the solver itself |
| **Relaxation** | drift of the same judgement re-asked over time | ✗ |

**Finding from shipping susceptibility** (2026-07-05, live,
`artifacts/live/spin-probe-2026-07-05/`): χ is state-dependent, not a model
constant. gpt-5.4-mini on a clear pair: χ = −0.18 nats (mild reactance,
belief survives). The same model on a contested pair: zero-field belief
EXACTLY 0.000 and χ = +0.64 — a **paramagnetic judgement**, moving with
whoever asks. gemini-2.5-flash holds a direction on that same pair and
yields only +0.12. The sycophancy question decomposes into: does this
judgement have a spontaneous direction, and how much field moves it?

Honesty note (adversarial self-review, notes/ideation-2026-07-05/): the
shipped χ is a **two-point secant at one field intensity**, not a measured
slope — the neutral reading is reported but does not enter χ, which
silently assumes the pro/con framings perturb symmetrically around the
true zero-field point. A 3–5-intensity preamble ladder fitting log-ratio
against field strength (slope + linearity residual) would distinguish a
genuinely low-χ judge from a step-function sycophant that ignores mild
pressure and folds completely past a threshold. Named, not yet built.

**Finding from shipping frustration** (2026-07-05): a directionally
transitive judge still shows ~0.13 curl — quantization frustration. First
hypothesis blamed the ladder's non-constant log step; the controlled test
(`tests/ladder_curl.rs`, same planted-transitive judge, full ladder)
REFUTED the emphasis: repo ladder floor 0.00198 vs constant-log-step
0.00155 — the uneven step is real but third-order. The ~0.13 floor comes
from **rung usage coarseness**: a judge that expresses everything as two
rungs (1.5-or-3.9) injects two orders of magnitude more curl than the
ladder geometry does. Design consequence: to lower quantization curl,
elicit finer *distinctions* (PMF instruments already do), not finer rungs.
Corrected 2026-07-05, same day — receipts over vibes.

## 6. Parameters, honestly

Current posture: temperature pinned 0.0 for evidence calls, 1.0 for
agreement samples; `top_logprobs` 20 (providers silently cap at 5);
max_tokens 16 (provider floor). **Nothing has been swept.** The open
questions with direct receipts waiting: does temperature move the PMF or
only the sample? (provider-dependent logit scaling — measurable via JSD
curves); does reasoning effort change judgement stability on models that
allow it in sampled mode? Both are afternoon-sized experiments on the
existing probe machinery.

## 7. AHP: attributes are entities too

The Analytic Hierarchy Process is already expressible in our primitives:
attributes compared pairwise ("how many times more important is *clarity*
than *depth* **for goal G**?") are just entities whose bodies are attribute
descriptions, judged under a meta-attribute ("importance for G"). The
solver's log-latents ARE the log priority vector; softmax of latents gives
normalized ratio-scale weights — the least-squares analog of Saaty's
eigenvector. `cardinal weigh` ships this (see §9): goal in, attribute
weights out, consumable directly by multi-attribute rerank. `weigh
--propose N` automates the decomposition too: the model proposes the
considerations, the weighing measures them — AHP end to end from a single
goal sentence. The full AHP hierarchy (goal → criteria → alternatives) is
two chained applications of the same primitive.

## 8. The canonical-attribute loop (the "ultimate loop")

Goal: discover the small set of attributes — the latents — that actually
span a multi-objective space. All pieces exist; the loop is composition:

1. **Propose** candidates (`explain --propose`, elaborate) — over-generate.
2. **Measure** all candidates on a probe entity set (multi-rerank).
3. **Correlate** (`attribute_correlations`, shipped): near-duplicates
   (|ρ| high) merge; keep the better-worded one (probe consistency decides).
4. **Complement**: ask for an attribute that *distinguishes items the
   current basis ties* — the residual direction, elicited in words.
5. Repeat until new candidates stop adding rank information (correlation
   with span ≈ 1) — the surviving set is the canonical basis; AHP-weight it
   against the goal (§7) and the multi-objective problem is mapped:
   Pareto front + named latents + weights.

Status: steps 1–3 shipped with receipts; 4–5 are orchestration, not new
math. This loop is the road from "sort my list" to "map my space" — and
custom feeds are exactly a mapped space plus a standing AHP weighting.

The focal-item inverse of this loop ships as `cardinal distinguish`
(`differentiation_profile` in the library): given a set and one item,
propose the attributes under which the item plausibly stands out, then
MEASURE all of them over the whole set and report the item's percentile
and z-score per attribute, best direction first. That is the propagation
question made judgeable — "under which attribute does this deserve to
travel far?" — answered by the counterbalanced pairwise machinery, never
by the proposer's say-so.

## 9. The judgement ledger, counted (2026-07-05)

Are we storing the judgements? Yes — in-repo, replayable:

- **~900 live provider judgements** across 9 receipt packs under
  `artifacts/live/`: 1,483 trace/cache JSONL lines (sort demos 32+120,
  taste tools 164, evidence path 110, policy benchmark 459 across three
  policies, smoke 2) plus 838 per-call raw receipt files
  (request/response/parsed/usage) in the method-comparison packs.
- Synthetic/eval receipts under `artifacts/eval/` (regenerable
  deterministically; stored as summaries by design).
- Test-time judgements: thousands per `cargo test` run across 282 tests —
  simulated, seeded, regenerated rather than stored, deliberately.
- Gap found while counting: the evidence-path pack had unexported caches
  (sqlite gitignored); repaired in the same commit as this document.
