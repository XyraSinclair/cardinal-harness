# experimental/

Scratch space for research ideas that are not yet part of the supported API surface.

Principles:
- Prefer *measuring* before optimizing (tokens, cached tokens, ranking quality, cost).
- Keep experiments reproducible (fixed seeds, frozen datasets, trace artifacts).
- Promote work into `docs/` + `src/` only once the objective + metrics are crisp.

Current threads:

## Prompt Layout + Read Cache (Concluded)

Goal: understand how prompt ordering affects:
- quality of judgements (consistency, refusal rate, calibration)
- provider read-cache efficiency (cached input tokens, latency, cost)

**Result:** A comprehensive sweep (4 layout variants × 7 models × 8 attributes)
found no significant advantage to attribute-first ordering. `canonical_v2`
(entity-first) won on inter-model agreement (tau=0.433) with zero refusals.
The `canonical_v2_attr_first` variant was retired; its slug now silently aliases
to `canonical_v2`. See `docs/PROMPTS.md` for the full experimental record.

Relevant repo hooks:
- Prompt templates live in `src/prompts.rs`.
- `canonical_v2` is the sole active template, placing entity context blocks before the attribute text.

## Baselines (Likert vs Ratio)

Goal: quantify when ratio judgements + robust solving beat direct Likert-style scoring,
as a function of budget and model quality.

Relevant repo hooks:
- Pairwise synthetic eval: `cardinal eval`
- Likert synthetic baseline: `cardinal eval-likert`

Next: extend the synthetic suite to emit budget-quality curves that fairly account for
token cost (pairwise prompts include two entities; Likert prompts include one).

## ANP Typed Contexts

Goal: measure failure modes when pairwise-only prompts are incorrectly forced into global composition.

Relevant repo hooks:
- Typed ANP module: `src/anp.rs`
- Full demo pipeline: `cardinal anp-demo`
- Synthetic typed-vs-forced benchmark: `cardinal eval-anp`

Recommended experiment:
1. Start from a real decomposition and mark one ambiguous axis as `pairwise_only_ratio`.
2. Run `anp-demo`, capture priorities + `next_query`.
3. Flip the same axis to `composable_ratio`, rerun, and compare rank shifts / Kendall tau.
4. Track which contexts most often trigger high inconsistency and prompt rewrites.

## Planner Research (Differential / Optimal Design)

Goal: go beyond heuristic blending by treating pair selection as an optimal design / value-of-information problem:
- define an explicit utility loss for top-k mistakes (regret)
- compute expected marginal improvement of candidate comparisons
- explore differentiable surrogates to top-k error for gradient-based planning

This work should start as design notes + offline experiments before touching the core planner.
