# What this is good for, why, and how

The one-page version, for sharing. Every claim below has evidence in this
repository—a test, a checked-in artifact from a live run, or both.

## What it's good for

Sorting and ranking **short lists of things by judgement calls** — the work
you'd give a careful human with taste: prompts, research ideas, candidate
plans, tweets, abstracts, backlog items. Anywhere "which is better, and by
how much?" is the real question and the list is dozens-to-hundreds, not
millions.

It is NOT for: query→document retrieval at scale (use a reranker API),
deterministic metrics (just compute them), or attributes too incoherent for
two judges to agree on (the tool will tell you when that's happening —
see probes — but it can't fix it).

## Why it's good

1. **It asks a stronger question.** Not "rate this 1–10" (miscalibrated,
   evidence in `docs/EVALUATION.md`) and not "which is better" (throws away
   magnitude) — but "how many times more?", whose answers compose additively
   in log space and over-determine a global fit.
2. **It reads the model's whole prior, where possible.** With
   `--template ratio_letter_v1`, one completion token's top-k logprobs are
   the full judgement distribution. Measured effect: ~3× the resolving
   power per dollar versus sampled JSON at equal budget
   (`artifacts/live/evidence-path-2026-07-04/`). Where providers hide
   logprobs, it degrades to sampling and says so in the run summary.
3. **It treats every claim about a judge as an empirical question.**
   Position bias: measured per run (order flips + order-residual in nats).
   Pure elicitation artifacts: measured per model (`cardinal calibrate`,
   null pairs — four models measured clean). Attribute coherence:
   measured per attribute (`--two-sided`, `--also-by` probes; a live probe
   caught a shaky paraphrase at +0.35). Logprob trustworthiness: measured
   per provider (seriate's reality map: DeepSeek's logprobs disagree with
   its own sampling at JSD 0.81).
4. **It applies the same skepticism to itself.** The planner's efficiency
   claim was benchmarked against uniform random pair selection, FAILED,
   got fixed (anchor-diverse exploration), and the fix cycle is pinned
   two-sided in `tests/planner_regret.rs` with history. The test suite is
   adversarial; honest negatives are kept, dated, and load-bearing.
5. **Everything is accounted for.** Comparisons, tokens, dollars, stop reasons,
   evidence health, and per-judgement traces that bind the exact solver input
   to a content-addressed engine configuration. SQLite supports zero-provider-call
   reruns when the cache is present; portable keyless bundles remain the explicit
   #52 gap. Sibling project
   [seriate](https://github.com/XyraSinclair/seriate) anchors judgements to
   raw provider bytes when full provenance matters.

## How to use it

```bash
cargo install --git https://github.com/XyraSinclair/cardinal-harness --locked
export OPENROUTER_API_KEY=...

# What will this cost? (no network)
cardinal sort ideas.txt --by "expected impact" --estimate

# Is this judge clean? (pennies)
cardinal calibrate --models openai/gpt-5.4-mini

# Sharpen the criterion, then sort with full-PMF evidence:
cardinal sort ideas.txt \
  --by "$(cardinal elaborate --by 'expected impact')" \
  --template ratio_letter_v1 --two-sided --scores

# Reverse-engineer a ranking you already believe:
cardinal explain my-ranking.txt --propose 3

# Multiple objectives? The response carries the Pareto front and the
# attribute correlation matrix — see docs and the multi_rerank API.
```

The source-install command builds current `main`. Tagged binaries are available
from GitHub Releases; the crate is not currently published to crates.io.

Every run prints its evidence summary: comparisons, cost, order flips,
evidence health, stop reason. When something would be uninformative, it refuses
loudly instead of printing garbage.

## Boundaries, stated plainly

- Evidence quality varies by provider; the reality map tells you where
  logprobs are real. Reasoning-class models refuse them outright.
- The synthetic evaluation suite shows regimes where simpler instruments
  (ordinal, even Likert) win — kept, pinned, documented. Ratio elicitation
  is a bet whose payoff depends on the judge and the attribute; the probes
  exist so you can check YOUR case instead of trusting ours.
- Dense solver: comfortable at hundreds of items, not tens of thousands.
