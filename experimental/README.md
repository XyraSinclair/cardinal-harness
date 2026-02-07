# experimental/

Scratch space for research ideas that are not yet part of the supported API surface.

Principles:
- Prefer *measuring* before optimizing (tokens, cached tokens, ranking quality, cost).
- Keep experiments reproducible (fixed seeds, frozen datasets, trace artifacts).
- Promote work into `docs/` + `src/` only once the objective + metrics are crisp.

Current threads:

## Prompt Layout + Read Cache

Goal: understand how prompt ordering affects:
- quality of judgements (consistency, refusal rate, calibration)
- provider read-cache efficiency (cached input tokens, latency, cost)

Relevant repo hooks:
- Prompt templates live in `src/prompts.rs`.
- `canonical_v2` currently places entity context blocks before the attribute text.
- `canonical_v2_attr_first` inverts that ordering by embedding context blocks after the attribute.

Recommended experiment:
1. Pick a realistic request JSON and run two reranks that differ only in `prompt_template_slug`.
2. Capture traces (`--trace`) and usage logs.
3. Compare: cost, refusal rates, stop reasons, and stability of top-k.

## Baselines (Likert vs Ratio)

Goal: quantify when ratio judgements + robust solving beat direct Likert-style scoring,
as a function of budget and model quality.

Relevant repo hooks:
- Pairwise synthetic eval: `cardinal eval`
- Likert synthetic baseline: `cardinal eval-likert`

Next: extend the synthetic suite to emit budget-quality curves that fairly account for
token cost (pairwise prompts include two entities; Likert prompts include one).

## Planner Research (Differential / Optimal Design)

Goal: go beyond heuristic blending by treating pair selection as an optimal design / value-of-information problem:
- define an explicit utility loss for top-k mistakes (regret)
- compute expected marginal improvement of candidate comparisons
- explore differentiable surrogates to top-k error for gradient-based planning

This work should start as design notes + offline experiments before touching the core planner.

