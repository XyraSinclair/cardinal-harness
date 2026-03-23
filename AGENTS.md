# AGENTS.md

`cardinal-harness` is the pairwise-ratio elicitation engine behind the rest of
the OpenPriors stack.

It is the algorithmic core for:

- pairwise ratio prompts and prompt templates
- robust IRLS solving over noisy pairwise judgments
- uncertainty-aware reranking and top-k stopping
- OpenRouter-backed gateway calls, pricing, usage tracking, and cacheing
- higher-level `pipeline`, `flywheel`, and `commander` workflows

## Collaboration Mode

This repo is in the fast direct-to-main collaboration set.

Default git behavior:

1. Before starting work, if the current checkout is clean:

```bash
git checkout main && git pull --rebase origin main
```

2. After a minimum good chunk of work:

```bash
git add <intentional-paths>
git diff --cached --quiet || git commit -m "<clear message>"
git pull --rebase origin main
git push origin HEAD:main
```

Interpretation:

- default to `main`; do not create branches unless there is a strong reason
- commit small, coherent changes frequently
- push soon after useful progress
- pull with rebase, not merge
- stage only the files you intentionally changed
- do not use `git add -A` unless the entire worktree is intentionally part of
  the task
- never force-push `main`
- if the checkout is already dirty with unrelated work, or another agent is
  active, prefer a separate worktree or clean checkout rather than disturbing
  existing state
- if a rebase conflict is trivial, resolve it carefully; otherwise stop and
  leave a clear note

Background sync automation, if any, should default to `git fetch`, not blind
`pull --rebase` against an active working tree.

## Key Areas

- `src/rating_engine.rs`: IRLS solver, planning, diagnostics
- `src/rerank/`: orchestration loop, stopping logic, trace/report output
- `src/rerank/comparison.rs`: pairwise LLM comparison logic
- `src/prompts.rs`: prompt templates and ratio ladder
- `src/gateway/`: OpenRouter adapter, pricing, usage, logprobs
- `src/pipeline.rs` and `src/pipeline/`: generate -> rank -> synthesize flows
- `src/commander/`: strategic codebase evaluation workflow
- `tests/`: gateway, rerank, trace, pipeline, and cache coverage

## Working Norms

- Preserve the core contract: pairwise ratio judgments aggregate into globally
  consistent cardinal scores with uncertainty.
- Prefer extending existing rerank/gateway seams over adding parallel
  implementations.
- Keep cost, uncertainty, and trace semantics explicit. These are part of the
  product, not incidental metadata.
- Avoid broad prompt churn unless it is deliberate and benchmark-motivated.
- When changing public request/response shapes or CLI behavior, update examples,
  tests, and docs in the same change.
