# Cardinal Bench: the public ratio-consistency benchmark

Design for taking the Judge Coherence Benchmark (JCB, `docs/BENCHMARK.md`)
public as a lab-facing, hill-climbable benchmark with meaningful real-world
entities, a website, and an honest monetization story. Drafted 2026-07-18.

## Thesis

Every LLM-judge eval in circulation measures agreement with a reference.
Cardinal Bench measures whether a model's judgments deserve the name
*beliefs*: a genuine belief is a fixed point of the transformations that
shouldn't matter. Swap the entities — the direction must hold and the ratio
must invert. Ask "how many times less" — same magnitude. Negate the
attribute — correlation flips to −1. Compose around a triangle — ratios must
multiply. No ground truth is needed; the model is scored against itself.

This matters because LLM-as-judge is now production infrastructure (RLAIF,
eval pipelines, ranking, moderation, grant screening) and none of it works
if the judge contradicts itself under permutation. That is the one-sentence
pitch to labs and press.

## What exists vs. what the public benchmark adds

The engine is done: `cardinal bench` runs the invariance table (order,
reciprocity, frustration, spin, polarity, paraphrase, null calibration,
signal gate) over a fixed 8-text corpus; `orbit.rs` has the full Z₂³
estimator; `transitivity.rs` has WST/MST/SST; logprob PMFs flow through
`comparison.rs`. What the public version adds:

1. **Meaningful entity pools** (below) replacing the 8-text demo corpus.
2. **Public dev set / private held-out set** split, with procedural
   rotation of entities and wordings per version (anti-memorization).
3. **An externally-runnable harness**: one command, open source, $1–20 and
   under an hour per model.
4. **The website**: leaderboard + explanation of the pairwise-ratio format.
5. **A logprob column** where providers expose it (secondary, never
   required — Anthropic/OpenAI reasoning models don't expose logprobs via
   OpenRouter; Kimi K3 doesn't either).

## Entity pools

Constraints: models must know the entities without retrieval; attributes
must be ratio-feelable (cardinal, not just ordinal); the subject matter
should carry real stakes; pools must refresh to resist memorization.

| Tier | Entities | Attributes (each with elaborated rubric) | Why |
|---|---|---|---|
| **Funders** | Open Philanthropy, Gates Foundation, ACX Grants, EA Funds, Emergent Ventures, DARPA, NIH, Wellcome, Sloan, HHMI (~20) | expected impact per marginal dollar, epistemic transparency, risk appetite, decision speed | Provocative, press-legible, squarely OpenPriors' domain |
| **Interventions** | bednets, unconditional cash, deworming, vitamin A, measles campaigns, psychotherapy, lead abatement, TB screening (~16) | lives saved per $1M, DALYs averted per $1M, evidence strength | GiveWell CEAs give a semi-ground-truth *sidebar* (external validity), never the score |
| **Manifund live** | live projects from `data/manifund/items/` with existing rubrics (`impact_per_dollar`, `theory_of_change`, `team_evidence`, `epistemic_integrity`) | the four canonized rubrics | Real allocation problem; refreshes naturally every quarter — built-in anti-memorization |
| **Anchors** | countries, cities, rivers (population, GDP, length) | known true ratios | The only tier with ground truth: adds an *accuracy/magnitude-calibration* axis so the leaderboard is legible to non-experts. Clearly labeled; memorizable by design; never the headline |

Headline score stays the JCB composite (signal × coherence) over the
consistency tiers. Anchors and GiveWell agreement are sidebars.

## Scoring

Unchanged from JCB: per-axis consistency in [−1, 1] or [0, 1], composite =
signal × mean coherence, so a constant judge scores zero and a decisive
incoherent judge scores low. Additions:

- **Transitivity axis promoted** into the composite (WST/MST/SST violation
  rates beyond 2 SE, from `transitivity.rs`) — cyclic ratio inconsistency
  is the most shareable failure mode ("A is 3× B, B is 2× C, C is 1.5× A").
- **PMF mode column** where logprobs exist: counterbalance residual and
  entropy calibration computed on the bucket PMF (`canonical_bucket_v1`).
  Sampled mode with repeat draws elsewhere; noise floors quoted per
  PRINCIPLES (every number carries its denominator and noise class).
- **Refusal accounting** stays explicit (no refusal-laundering; caught once
  already per PRINCIPLES).

## How labs come to hill-climb a benchmark (honest mechanics)

Labs climb what is **visible, runnable, discriminating, and narratable**:

1. **Visible**: it appears in model cards and launch-post comparison tables.
   That happens when third parties keep a leaderboard current within days
   of every frontier release, so the number exists before the lab writes
   the card. We commit to same-week runs on every frontier release.
2. **Runnable**: internal eval teams ingest external benchmarks wholesale
   when they're one command, cheap, and automated. Ship the harness open
   source and contribute adapters to `inspect_ai` and
   `lm-evaluation-harness` — those are the distribution channels into lab
   CI, which is where hill-climbing physically happens.
3. **Discriminating**: if all frontier models cluster, it's dead. The JCB
   already separates models; the public version must publish the spread
   and the test-retest floor so gaps are legibly real.
4. **Narratable**: one screenshot-able contradiction per weak model (the
   "caught red-handed" card: the actual prompt pair and the actual
   contradictory answers). Benchmarks spread as artifacts of failure, not
   tables of success.

**Goodhart analysis — why climbing this one is aligned.** For most
benchmarks, gaming ≠ capability (memorize answers, overfit style). Here the
score is invariance of the model's own output under permutations of the
prompt. The only *general* way to score well is to compute an
order-independent internal judgment before verbalizing — which is exactly
the competence every LLM-as-judge deployment needs. Even the "cheat"
(canonicalize the pair internally, judge once, re-sign) *is the desired
behavior*. Residual gaming vectors and their mitigations:

- *Corpus memorization* → procedural rotation of entities and wording
  paraphrases per version; private held-out pool scored quarterly; public
  dev pool for free iteration.
- *Degenerate flatness* (answer 1.0 always) → signal gate, already in the
  composite.
- *Template special-casing* → templates rotate procedurally (prompt bytes
  are physics; wordings are generated, not fixed strings), and the format
  is the same one used in real deployments, so overfitting it still
  transfers.

## Monetization (honest)

Direct benchmark revenue is rare; there are three working models in the
wild and one aligned default:

1. **Private pre-release evals** (the Scale SEAL / Epoch FrontierMath
   model): labs pay for held-out-set runs on unreleased checkpoints plus
   the axis-level diagnostic report (orbit character decomposition, failure
   exemplars, per-axis noise floors). This is the realistic paid product —
   the public leaderboard is the free tier that creates the demand.
2. **Judge certification**: companies deploying LLM-as-judge (eval vendors,
   RLAIF pipelines, marketplaces that rank with LLMs) pay for a coherence
   certificate on *their* judge+prompt+model configuration. Recurring,
   because configurations churn. This is SOC2-shaped, and nobody occupies
   the niche yet.
3. **The harness as product**: cardinal-harness itself (counterbalancing,
   orbit elicitation, IRLS fusion) sold as the *fix* for whatever the
   benchmark exposes. Benchmark = top of funnel for OpenPriors.
4. **Zero-revenue default is still a win**: if labs internalize the metric
   and climb it for free, frontier models become consistent cardinal
   judges — which is upstream infrastructure for everything OpenPriors
   wants to exist. Publish regardless of whether 1–3 land.

Do not paywall the leaderboard or the harness; credibility is the asset.

## Website

Committed HTML under `artifacts/live/` (claude.ai Artifacts are banned),
local static server for review, then a public deploy (e.g.
`bench.openpriors.org`). Pages:

1. **Leaderboard** — composite + per-axis columns, PMF-mode badge,
   test-retest error bars, cost-per-run column (everything denominated).
2. **The format** — the pairwise-ratio explainer: why cardinal beats
   ordinal, the 17-point ratio ladder, confidence, the invariance table as
   the centerpiece diagram.
3. **Model cards** — per-model axis radar, the "caught red-handed" panel
   (real prompt bytes, real contradictory outputs), orbit character
   decomposition for the initiated.
4. **Run it yourself** — one command, expected cost, how to submit a PR
   with results for a new model.

## Operations

- **Versioning**: pools and wordings rotate per minor version; scores are
  only comparable within a version; the changelog is public.
- **Cost**: JCB today is 194 comparisons ≈ $0.10–4/model depending on
  model. Public version with tiers lands ~600–1,000 comparisons: $1–20 per
  frontier model, test-retest included. Same-week coverage of a release is
  a <$50 event.
- **Held-out custody**: private pool lives off-GitHub (owned infra, per
  repo law); published hashes of the held-out packets let anyone verify
  after-the-fact that the pool didn't move.

## Open questions

- Public name: "Cardinal Bench" (working title) vs. keeping "JCB".
- Whether the Funders tier is too spicy for lab partnerships (it names
  real institutions; the failure cards would show models' implied funder
  rankings). Could swap headline tier to Interventions.
- inspect_ai vs. lm-evaluation-harness first (probably inspect_ai; UK AISI
  gravity and native logprob support).
