# Prompts

The sole prompt template is `canonical_v2` in `src/prompts.rs`, selected via
`prompt_template_slug` in `MultiRerankAttributeSpec`.

All legacy slugs (`canonical_v1`, `canonical_v3`, `*_repeat_full`,
`*_literal_double`, `experimental_*`) resolve to `canonical_v2` at runtime
for backward compatibility. They were empirically tested and retired.

## Entity Context Placement

Pairwise comparisons pass the *entity text* via `EntityRef.context` (not the label).

Two ways to place context in the rendered user message:

1. Implicit prefix injection (most templates)
   - If the template user text does **not** include `{entity_A_context_block}` /
     `{entity_B_context_block}`, then context blocks are inserted before the template body.
2. Inline placement (opt-in per template)
   - If the template includes `{entity_A_context_block}` and/or `{entity_B_context_block}`,
     those placeholders are replaced in-place and no extra prefix is injected.

This is intentional so you can experiment with prompt ordering without changing code.

## Why Prompt Ordering Matters

Ordering affects two things:

1. **Model quality**
   - Some models do better when the scoring attribute is stated before reading long entities.
   - Others do better when they read the entities first and see the attribute prompt immediately
     before answering.
2. **Provider read-cache efficiency**
   - Some providers discount cached input tokens for repeated prefixes.
   - If a large prefix is repeated across calls (same attribute prompt, same entity text in the
     same position, etc.), total cost and latency can drop substantially.

The library intentionally keeps prompt templates as simple string templates so you can benchmark
these tradeoffs per provider/model/attribute.

## Empirical Results: Prompt Layout Sweep (March 2026)

### Experiment design

Comprehensive sweep over the four base layouts (`v1`, `v2`, `v3`,
`v2_literal_double`) across **7 model families** and **8 attributes**, with
200 comparisons per run (5,600 total comparisons, $6 total cost).

**Models**: gpt-5.4-nano, gemini-2.5-flash, kimi-k2.5, qwen3-32b,
llama-4-maverick, deepseek-v3.1, mistral-small-3.1.

**Attributes**: civilizational leverage, epistemic rigor, execution capability,
alignment seriousness, public credibility, talent density, counterfactual
replaceability, strategic clarity.

**Entities**: 10 frontier AI organizations (Anthropic, OpenAI, DeepMind, xAI,
ARC, METR, Redwood, Epoch, Apollo, CAIS).

Study pack: `openpriors-research/inputs/prompt_layout_sweep_v2.study.json`.
Analysis script: `openpriors-research/scripts/analyze_prompt_layout_sweep.py`.

### Inter-model agreement

Mean per-attribute Kendall tau across all 21 model pairs:

| Layout | Mean tau | Relative cost |
|--------|---------|---------------|
| **canonical_v2** | **0.433** | 1.0x |
| canonical_v3 | 0.422 | 0.58x |
| canonical_v2_literal_double | 0.421 | 1.33x |
| canonical_v1 | 0.388 | 0.99x |

v2 narrowly wins inter-model agreement. The differences are genuine but modest
(0.388--0.433 across the full range). v3 achieves near-identical agreement at
roughly half the token cost, making it the efficiency pick.

### Intra-model stability

How much does switching prompt layout change a given model's rankings? (Mean tau
of each layout vs all other layouts, averaged across models.)

| Layout | Mean stability |
|--------|---------------|
| canonical_v1 | 0.744 |
| canonical_v2 | 0.723 |
| canonical_v3 | 0.723 |
| canonical_v2_literal_double | **0.655** |

The literal double is the *least* stable layout: it moves rankings furthest from
what the other three produce. On kimi-k2.5 specifically, v2 vs literal_double
had a tau of only 0.422, meaning the doubling substantially rearranged
that model's judgements.

### Trace-level effects

| Layout | Mean ratio | Mean confidence | Refusal rate |
|--------|-----------|-----------------|-------------|
| canonical_v1 | 2.06 | 0.78 | 0.1% |
| canonical_v2 | 2.01 | 0.73 | 0.1% |
| canonical_v2_literal_double | 2.43 | 0.71 | 0.2% |
| canonical_v3 | 2.43 | 0.73 | 1.1% |

The literal double pushes ratios higher (models become more decisive) and
slightly depresses confidence. v3's compressed format drives a notably higher
refusal rate (1.1%), likely because the terse system prompt gives less
scaffolding for models that would otherwise express uncertainty as a low
confidence score rather than a refusal.

### Conclusions

1. **Layout choice matters less than expected.** The inter-model tau spread
   across all four layouts is only 0.045. Model selection, entity description
   quality, and attribute prompt specificity are higher-leverage interventions.

2. **v2 remains the right default.** Best inter-model agreement, good
   stability, negligible refusal rate. The refuse clause and confidence
   semantics pull their weight.

3. **v3 is the budget pick.** Nearly identical agreement at ~58% of the token
   cost. Good choice for high-volume sweeps where cost matters.

4. **Literal doubling does not improve measurement quality.** It costs 1.33x
   the tokens, produces the least stable rankings across layout switches, and
   does not improve inter-model agreement over plain v2. The doubling makes
   models more extreme in their ratios without making them more consistent
   with each other.

5. **v1 is the worst choice.** Lowest inter-model agreement despite being the
   most stable within a model. The refuse clause in v2/v3 helps models handle
   genuinely uncertain comparisons rather than forcing a noisy answer.

## Empirical Results: Creative Prompt Structure Experiments (March 2026)

### Experiment design

Six creative prompt structures tested against the v2 baseline across **7 model
families**, **8 attributes**, 200 comparisons per run (49 runs, ~9,800
comparisons attempted).

**Experimental prompts**:

| Slug | Idea | Mechanism |
|------|------|-----------|
| `experimental_reasoning` | Chain-of-thought before JSON | "Write 2-3 sentences of reasoning, then output JSON." |
| `experimental_decompose` | Facet decomposition | "Identify 2-3 facets of the attribute, assess each, then synthesize." |
| `experimental_anchored` | Calibration anchors on the ratio ladder | Each ratio range gets a plain-language anchor ("1.5-1.75: Meaningful gap. A thoughtful observer would consistently pick the winner.") |
| `experimental_adversarial` | Steel-man the loser | "Before your final answer, state the strongest case for the entity you think ranks LOWER." |
| `experimental_logspace` | Think in doublings | Frame the scale as powers of 2 ("1 doubling = ratio 2.1, 2 doublings = ratio 3.9"), then map to ratio ladder. |
| `experimental_protocol` | Measurement instrument framing | Reframe the model as a measurement instrument reading specimens, not an agent evaluating entities. |

Study pack: `openpriors-research/inputs/prompt_creative_sweep.study.json`.

### Data quality: refusal and parse rates

The creative prompts revealed sharp model-specific failure modes:

| Prompt | Models with >=100 used | Notable failures |
|--------|----------------------|------------------|
| canonical_v2 | 7/7 | None |
| experimental_anchored | 7/7 | None (best compatibility) |
| experimental_protocol | 7/7 | None |
| experimental_logspace | 7/7 | None |
| experimental_adversarial | 5/7 | qwen3 114 refusals, llama 176 refusals |
| experimental_reasoning | 4/7 | qwen3 155 refusals, llama 145 refusals, deepseek 110 refusals |
| experimental_decompose | 3/7 | gemini 122 refusals, qwen3 135 refusals, llama all refused, kimi 167 errors |

Decompose and reasoning prompts trigger heavy refusal rates on models that
interpret the multi-step instructions as requesting analysis they aren't
comfortable providing. The adversarial prompt's "steel-man the loser" framing
similarly triggers refusal in models with strong RLHF guardrails.

### Inter-model agreement

Mean per-attribute Kendall tau across model pairs (only counting pairs where
both models had >=100 successful comparisons):

| Prompt | Mean tau | Model pairs | Notes |
|--------|---------|-------------|-------|
| **canonical_v2** | **0.433** | 21 (7 models) | **Baseline, still best** |
| experimental_anchored | 0.420 | 21 (7 models) | Near-baseline, full compatibility |
| experimental_decompose | 0.415 | 3 (3 models) | Promising but too few valid runs |
| experimental_reasoning | 0.397 | 6 (4 models) | Below baseline, limited data |
| experimental_protocol | 0.390 | 21 (7 models) | Below baseline despite full compat |
| experimental_logspace | 0.379 | 21 (7 models) | Below baseline |
| experimental_adversarial | 0.358 | 10 (5 models) | Worst agreement |

No experimental prompt beat v2 on inter-model agreement. The anchored prompt
came closest (0.420 vs 0.433) while maintaining perfect model compatibility.

### Intra-model stability vs v2

How closely does each experimental prompt's rankings match the same model's v2
rankings? (Mean Kendall tau across models with valid data for both.)

| Prompt | Mean tau vs v2 | Interpretation |
|--------|---------------|----------------|
| experimental_anchored | **0.829** | Most conservative -- barely changes rankings |
| experimental_decompose | 0.778 | Moderate perturbation |
| experimental_logspace | 0.727 | Moderate perturbation |
| experimental_reasoning | 0.722 | Moderate perturbation |
| experimental_adversarial | 0.707 | Meaningful shift |
| experimental_protocol | 0.663 | Largest departure from v2 |

The anchored prompt is the most stable extension of v2: it preserves rankings
while adding calibration information. The protocol framing causes the largest
departure, suggesting that reframing the model's role genuinely changes its
judgement patterns -- but not in a direction that improves cross-model agreement.

### Trace-level effects

| Prompt | Mean ratio | Mean confidence | Key observation |
|--------|-----------|-----------------|-----------------|
| canonical_v2 | 2.01 | 0.73 | Baseline |
| experimental_anchored | 1.88 | 0.74 | Slightly compressed ratios, anchors tighten the scale |
| experimental_adversarial | 2.21 | 0.73 | Steel-manning doesn't change much where it works |
| experimental_reasoning | 2.48 | 0.70 | Reasoning inflates ratios, drops confidence |
| experimental_logspace | 2.68 | 0.68 | Log-space framing inflates ratios substantially |
| experimental_protocol | 2.62 | 0.72 | Instrument framing also inflates ratios |
| experimental_decompose | 2.16 | 0.77 | Decomposition slightly increases confidence |

Prompts that ask for more thinking (reasoning, logspace, protocol) consistently
produce higher ratios and lower confidence. The anchored prompt has the opposite
effect: its calibration descriptions ("barely distinguishable," "different
leagues") compress the scale and slightly raise confidence.

### Conclusions

1. **v2 remains undefeated.** None of the six creative structures beat v2 on
   inter-model agreement. The simplest prompt that includes a refuse clause and
   confidence semantics is still the best default.

2. **Anchored is the best extension.** It maintains full model compatibility
   (7/7), produces the closest inter-model agreement to v2 (0.420 vs 0.433),
   and is the most stable perturbation (tau=0.829 vs v2). If we iterate on the
   prompt, calibration anchors are the most promising direction -- they need
   tuning, not abandoning.

3. **Reasoning-before-JSON is a trap for judge prompts.** It triggers massive
   refusal rates on 3/7 models, inflates ratios, and doesn't improve agreement.
   The reasoning tokens are not improving judgement quality -- they're adding
   noise. This is likely because the models generate performative reasoning that
   post-hoc justifies a snap judgment rather than genuinely deliberating.

4. **Decomposition is promising but fragile.** Where it works (gpt-5.4-nano,
   deepseek, mistral), it produces slightly higher confidence and reasonable
   agreement. But it triggers refusals or errors on 4/7 models, making it
   unusable as a default.

5. **Log-space and instrument framing backfire.** Both inflate ratios
   substantially (mean 2.6+ vs 2.0 baseline) and reduce inter-model agreement.
   The doublings framing may be more natural for human reasoning but confuses
   models that are calibrated to the linear ratio ladder. The instrument
   framing changes the model's self-conception in ways that hurt consistency.

6. **The adversarial prompt has the worst inter-model agreement** (0.358). The
   steel-man instruction adds variance without convergence -- each model
   steel-mans differently, amplifying rather than reducing disagreement.

7. **Model compatibility is a first-order constraint.** Any prompt that
   triggers >10% refusals on multiple model families is disqualified as a
   default, regardless of how well it works on the models that tolerate it.
   This eliminates reasoning, decompose, and adversarial from default
   consideration.
