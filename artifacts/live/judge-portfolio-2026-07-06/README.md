# The judge portfolio, live (2026-07-06)

`cargo run --example judge_portfolio -- <fresh reports.jsonl>`: ensemble
geometry of the six benchmark models computed from the RETEST pack's
primary latent vectors and fresh (uncached) costs — zero marginal spend.
One-factor model, Spearman-triad loadings, full error covariance
Ψ = R − llᵀ projected to the PSD cone, minimum-variance weights Ψ⁻¹l,
marginal information ΔI = I − I₋ᵢ per judge. n = 8 entities, one
attribute, one run: instrument demonstration; loadings near 1 are
small-sample-inflated and say "indistinguishable from consensus at this
n", not "perfect".

| metric | value |
|---|---|
| consensus share | 0.920 |
| **effective error sources** | **2.89** (six models, fewer than three independent error channels — the labs share failure modes) |

| by ΔI/$ | loading | weight | ΔI | cost | ΔI/$ |
|---|---|---|---|---|---|
| deepseek-v4-flash | 0.965 | −0.110 | 477 | $0.016 | **29,838** |
| gemini-2.5-flash | 0.979 | +0.087 | 422 | $0.026 | 16,448 |
| claude-haiku-4.5 | 0.999 | +0.758 | 1162 | $0.198 | 5,870 |
| claude-sonnet-4.6 | 0.979 | +0.255 | 740 | $0.258 | 2,869 |
| gpt-5.4-mini | 0.916 | +0.022 | 85 | $0.054 | 1,572 |
| gpt-5.4-nano | 0.879 | −0.012 | 12 | $0.015 | 819 |

## Readings

1. **Information per dollar inverts the prestige order.** deepseek and
   gemini dominate the budgeted portfolio; sonnet — the most coherent
   individual judge on the JCB — buys a tenth of deepseek's marginal
   information per dollar for consensus estimation. Being the best judge
   and being the best BUY are different theorems.
2. **Negative weights are real portfolio structure**, not bugs: GLS
   hedges shared error the way Markowitz shorts a correlated asset.
   deepseek's −0.11 says: given the rest of this roster, subtract a
   little of its view to cancel a common error component.
3. **The OpenAI smalls contribute ≈ nothing to this consensus** (ΔI 85
   and 12 vs 400–1200): consistent with their JCB coherence, measured
   through an entirely different formalism — two instruments, one
   conclusion.
4. Estimator provenance: three designs failed before this one — raw-PR
   (counts opinion dimensions), eigen loadings (inconsistent across J),
   un-projected Ψ (live total information −1.94, a negative precision).
   Each failure is documented in the module and pinned where testable.
