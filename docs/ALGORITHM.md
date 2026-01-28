# Algorithm sketch

This harness treats pairwise ratio judgements as noisy observations of latent scores.

## Pairwise ratio model

Each comparison produces:
- higher_ranked: A or B
- ratio: how many times stronger the attribute is (1.0 .. 26.0)
- confidence: 0..1

We map this to a signed log-ratio and variance:
- ln_ratio = ln(ratio) or -ln(ratio) depending on winner
- variance = f(confidence) with a bounded sigma range

These observations feed the solver as weighted edges on a graph.

## Rating engine (per attribute)

For each attribute:
- build an IRLS system (Huber robust)
- solve for latent scores
- compute diagnostics (variance, rank risk, stability)
- propose high-value next pairs

The solver uses a dense linear algebra backend, so practical size is ~5k items per run.

## Trait search (multi-attribute)

Across attributes:
- normalize each attribute with robust scales (MAD)
- combine as weighted utility
- apply gates (latent/z/percentile/min_norm)
- focus uncertainty on top-k boundary

Top-k uncertainty is estimated by frontier inversion probability. The planner targets
pairs that maximally reduce global error.

## Dynamic query selection

The rerank loop:
1. Solve per-attribute engines and build global utility.
2. Estimate top-k error.
3. If error > tolerated, propose next pairs via planner.
4. Ask the LLM for those pairs.
5. Add observations and repeat.

Stop conditions include:
- tolerated_error met
- certified separation bound
- comparison budget exhausted
- latency budget exceeded

## Caching

Pairwise judgements are cached by model, prompt, attribute, and entity text hashes.
Cache hits skip the LLM call and are treated as observations.

