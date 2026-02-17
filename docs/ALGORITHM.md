# Algorithm and design decisions

This document explains not just *what* cardinal-harness does, but *why* each design choice was made.

## Why pairwise comparisons instead of direct scoring?

When you ask an LLM to "rate this essay 1–10", the scores are:
- **Miscalibrated**: a 7 for one essay might mean something different than a 7 for another.
- **Anchor-dependent**: the first item scored sets an arbitrary reference point.
- **Inconsistent**: the same item can get different absolute scores in different contexts.

Pairwise comparisons ("which is better, and by how much?") avoid all of these problems. Relative judgments are more stable because the LLM has a concrete reference (the other item) rather than an abstract scale. This is well-established in psychometrics and preference learning — it's the same reason tournament systems work better than asking each player to self-report a skill number.

## Why ratios instead of just "which is better?"

Pure ordinal comparisons (A > B) waste information. If A is *barely* better than B, that's very different from A being *vastly* better, but a binary comparison throws away the magnitude.

Ratio judgments ("A is 2.5× clearer than B") capture magnitude on a natural scale. In log-space, these become additive: ln(A/B) + ln(B/C) = ln(A/C), which means observations compose cleanly and the system of equations is well-behaved.

## The ratio ladder: 1.0 .. 26.0

The LLM chooses from a fixed set of ratio values: `[1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.1, 2.5, 3.1, 3.9, 5.1, 6.8, 9.2, 12.7, 18.0, 26.0]`.

This ladder is approximately geometric (roughly evenly spaced in log-space), which means each step represents a similar perceptual increment. The range caps at 26× because extreme ratios are unreliable — if two items are that different, the comparison is already decisive and precision beyond 26× doesn't help. The fine gradations near 1.0 (1.05, 1.1) allow the model to express near-ties with nuance.

## The solver: why IRLS with Huber loss?

Given a set of noisy pairwise log-ratio observations, we need to recover a consistent global set of scores. This is a weighted least-squares problem on a graph, where each comparison is an edge with a measured difference and a weight (from the confidence score).

**Why robust regression?** LLMs are noisy judges. Some comparisons are outliers — the model misunderstands, hallucinates, or produces an inconsistent judgment. Ordinary least squares would let one bad comparison distort all the scores. Huber loss provides a smooth transition between L2 (for small residuals) and L1 (for large residuals), automatically downweighting outliers without discarding them entirely.

**Why IRLS specifically?** Iteratively Reweighted Least Squares turns the robust regression into a sequence of ordinary weighted least-squares problems, each solvable via Cholesky decomposition. This is numerically stable, well-understood, and efficient for the problem sizes we handle (up to ~5,000 items with dense matrices).

**Why Huber over other robust losses?** Huber loss (with k=1.5) is the standard choice in robust statistics. It's less aggressive than Tukey's biweight (which can zero-out observations entirely) but more robust than plain L2. The k=1.5 threshold means observations with residuals beyond 1.5 standard deviations get downweighted but never fully discarded.

## Confidence mapping

Each LLM comparison includes a confidence score (0 to 1). This maps to observation variance via:

```
g(c) = eps + (1 - eps) * c^gamma
```

With `eps=0.001` and `gamma=2.0`, this means: low-confidence observations get high variance (low weight in the solver), and high-confidence observations get low variance (high weight). The squaring (`gamma=2`) makes the system more sensitive to confidence differences — the LLM has to be quite confident before an observation gets substantial weight.

## Gauge pinning

Latent scores are relative — you can shift all scores by a constant without changing any pairwise difference. The solver "pins" one node per connected component to zero to break this symmetry. This is standard in pairwise comparison models (analogous to setting a reference voltage in an electrical circuit).

## Uncertainty estimation

After solving, the system computes posterior variances from the inverse of the Hessian (the information matrix). For small problems (≤256 items), this is computed exactly via Cholesky; for larger problems, the diagonal is estimated via Hutchinson's stochastic trace estimator.

These variances tell you how uncertain each item's score is — items with few comparisons have wide uncertainty, items with many consistent comparisons have tight uncertainty.

## Dynamic query planning

Given current scores and uncertainties, the planner proposes the next pair to compare. It targets pairs that would maximally reduce uncertainty about the **top-K ranking** specifically — not overall score uncertainty.

The planner uses **effective resistance** from spectral graph theory: the expected variance reduction from observing edge (i,j) is proportional to the effective resistance between nodes i and j in the comparison graph. Pairs with high effective resistance (poorly connected in the comparison graph) and high relevance to the top-K boundary are prioritized.

## Top-K uncertainty and stopping

The system doesn't just track individual score uncertainties — it estimates the probability that the current top-K ranking is wrong. Specifically, for each item near the K-boundary, it computes the **frontier inversion probability**: the probability that an item currently ranked just below K actually belongs above K (or vice versa), using a Gaussian approximation of the score difference.

The global top-K error is the sum of these boundary flip probabilities. The system stops when this error falls below `tolerated_error`, or when it achieves a **certified separation bound** — meaning the gap between item K and item K+1 is large enough relative to their joint uncertainty that no plausible re-estimation would flip them.

The `stop_sigma_inflate` parameter (default 1.25) inflates the uncertainty estimate for the stopping check to be conservative — the system would rather ask one extra question than stop prematurely with an incorrect ranking.

## Multi-attribute composition

When scoring on multiple attributes (e.g., clarity *and* depth *and* originality), each attribute gets its own independent rating engine. Attributes are then combined into a single utility score:

1. **Normalize** each attribute with robust scales (MAD — median absolute deviation, which is outlier-resistant unlike standard deviation).
2. **Weight** according to user-specified attribute weights.
3. **Apply gates** — optional hard filters that exclude items below a threshold on any attribute (e.g., "must be above the 25th percentile on safety").

The planner operates on the combined utility, targeting pairs that reduce uncertainty about the top-K of the *combined* ranking — so it naturally allocates more comparison budget to attributes where the top-K items are poorly separated.

## ANP typed contexts (optional layer)

For open-ended prioritization tasks with dependency graphs, use the ANP layer in `src/anp.rs`.

- Keep contexts explicitly typed as `composable_ratio` or `pairwise_only_ratio`.
- Fit local priorities per context from confidence-weighted log-ratio judgments.
- Use context/pair query helpers to target highest-value next judgments.
- Propagate only composable contexts through a weighted, damped supermatrix.

See `docs/ANP.md` for data structures and usage.

## Caching

Pairwise judgments are cached in SQLite, keyed on (model, prompt template, attribute, entity text hashes). This means:
- Re-running the same request reuses all prior LLM calls.
- Adding a new entity to the set only requires comparisons involving that entity.
- Changing the model or prompt invalidates the relevant cache entries (correctly).
- Cache hits are treated as observations with the same weight as fresh LLM calls.
