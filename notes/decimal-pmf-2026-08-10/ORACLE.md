# Oracle consult: decimal-PMF kernel design (GPT-5.6 Sol Pro, browser, 22m15s, 2026-08-10)

Session slug: `decimal-pmf-kernel-design-three`. Model verified from the picker evidence
(`resolvedLabel=Pro; verified=yes`). Prompt: the PROBLEM.md/DESIGN.md framing plus seven
design questions. Raw answer below the reconciliation.

## Reconciliation (my accept/reject/modify, checked against the same-night census)

**Adopted — these upgrade the design:**

1. **Bounded domain is an identifiability REQUIREMENT, not a convenience.** With
   unbounded decimals, residual mass q>0 makes E[log R] certificates unboundedly wrong.
   Every certified request carries [R_min, R_max], max significant digits, termination
   rule, refusal outcome. Matches our grammar-bounds instinct; hardens it into law.
2. **The "Frankenstein distribution" threat.** Public endpoints are hidden mixtures over
   deployments; a fresh forced-prefix call returns ∫P_θ(t|x)dπ, not the
   likelihood-weighted conditional. A tree glued from independent calls can be a
   distribution NO deployment has. Our measured 5.4-mini mass drift (±10% between
   calls) is direct evidence this is live. → adopt guarantee tiers (TokenExact /
   TextValidated / SampleOnly / Heuristic) with runtime prefix-law tests
   (martingale residuals of indicator − p).
3. **Selection-bias correction in the estimator.** Folding sample-discovered atoms into
   the exact head and estimating the residual with the SAME draws biases the residual
   conditional. Fix: exploration batch vs estimation batch, K-fold cross-fitting, or
   the atom-inclusion Horvitz–Thompson estimator (π_y = 1−(1−p_y)^N) + optional GREG
   correction using known q. Default: **exact head + stratified residual rejection
   sampling, Neyman allocation M_b ∝ q_b·osc_b(h)/√c_b**, empirical-Bernstein/DKW
   certificates, confidence sequences under adaptivity.
4. **My expansion-policy claim was wrong for linear targets.** For E[Z], a cell's
   priority is q_b·osc_b(Z), full stop — distance from the current mean does NOT add
   priority (that intuition applies only to variance/nonlinear functionals, via
   influence functions). Use q_b·osc/c_b as optimistic bound + one-step expected
   contraction; no universal greedy guarantee exists (ancestor-unlock counterexamples);
   depth-two rollout occasionally; stop on the global certificate.
5. **`SignedLogRatioDistribution` is the product wedge.** Z = signed log R ∈ [−B,B]
   with a tie atom collapses (direction, ratio) into one bounded variable, kills
   inconsistent pairs, is linear in what IRLS consumes, and CDF-band output is
   representation-invariant. Build this one object excellently first.
6. **Radix-4 interval code beats decimal digit-grid** as the encoded instrument:
   partition [−B,B] into ≤4 children + STOP within top-5, zero per-node truncation
   residual, uniform multiplicative resolution. Still a DIFFERENT instrument
   (never merge with the natural distribution) — our enum-jump measurement (RESULTS
   §1b) is direct evidence for that doctrine.
7. **Measure non-canonical mass, don't assume it small**: canonicality check
   y == encode(decode(y)) on full traces; Clopper–Pearson upper bound (≈3/N at zero
   hits, 95%); error budget q_nc·osc(h) explicitly in the ledger. Also: termination
   must be a SCORED delimiter — our JSON `"}` token qualifies (measured).
8. **Joint fields: quotient token traces to values only after the last dependent field
   terminates** — equal decoded text ≠ equal transformer state. Immaterial for the
   shallow single-number instrument, load-bearing for multi-field.

**Confirmed (independently derived both sides):** samples from P need no importance
weights; chosen-token logprobs turn sampled tail atoms exact (our resampling-is-peeling
= their "samples split omitted-cylinder cells"); credal random-set object is
authoritative with any Bayesian layer as labeled overlay; residual cells carry known
mass + unknown location, never invented children; empirical digit priors rejected as
defaults.

**Where our census is ahead of Oracle:** it says masking semantics "must be
runtime-discovered" — we already measured masked+renormalized on Azure/5.x, plus the
sharper finding that the schema is context-visible (enum-jump). It flags Anthropic
prefill as no-longer-supported on current models (updates DESIGN's L3 row: Anthropic is
sample-only, and without logprobs entirely).

**Judgment on the product claim:** adopt Oracle's phrasing — not "full PMFs from five
logprobs" but **"target-aware reconstruction of numeric model distributions with
auditable mass accounting and anytime error certificates."** The conservation ledger we
designed is exactly the "auditable mass accounting"; the envelope machinery is the
"anytime certificate"; the census is the "oracle-law audit" tier.

**Experiment sequence:** Oracle's 5-probe plan subsumes ours; its #1 (oracle-law
martingale tests) and #3 (exact-ground-truth estimator shootout on a local model with
full logits, 10^3–10^4-leaf language) are the two we hadn't specified and both are
right. #3 is the one that turns estimator choice from taste into measurement.

---

# Raw answer (GPT-5.6 Sol Pro)

## Challenge to thesis

The right kernel is **not “recover the full PMF over arbitrary decimals.”** It is:

> **Construct an anytime-certified partial measure over a bounded numeric language, then estimate only the unresolved conditional measures.**

Several parts of the current framing need tightening.

### “Free-form, arbitrary resolution” is not identifiable without semantic bounds

Suppose a finite run leaves residual mass (q>0). If decimal length or exponent is unbounded, two distributions can agree with every observed top-(k) probability while placing (q) on either (10) or (10^{10^{6}}). The resulting difference in (E[\log R]) can be arbitrarily large.

Therefore every certified request needs:

[
R\in[R_{\min},R_{\max}],\qquad
\text{maximum characters/significant digits/exponent},
]

plus an explicit termination rule and an `invalid/refusal` outcome. “Arbitrary resolution” can mean adaptively fine resolution **inside a bounded domain**. It cannot mean an unbounded decimal language with finite expectation certificates.

For pairwise ratios, the clean domain is usually

[
Z=\operatorname{signed}\log R\in[-B,B].
]

### A text continuation is not necessarily a conditional oracle

For a fixed model state and exact token-prefix continuation, chain rule applies. A prompt-echo workaround generally does not: it changes the chat template, role boundaries, or decoder constraints. It is an intervention that asks the model to continue text resembling a prefix, not necessarily sampling from (P(,\cdot\mid Y_{1:j}=x)).

There is another failure mode if a public endpoint is a hidden mixture over deployments (\theta). The natural conditional is

[
P(t\mid x)
==========

\frac{\int P_\theta(x)P_\theta(t\mid x),d\pi(\theta)}
{\int P_\theta(x),d\pi(\theta)},
]

whereas a fresh forced-prefix request may return approximately

[
\int P_\theta(t\mid x),d\pi(\theta).
]

These are not equal because the first likelihood-weights deployments by how likely they were to emit (x). A tree assembled from independent calls can otherwise become a “Frankenstein distribution.”

The kernel consequently needs guarantee tiers:

1. **Token-exact, pinned oracle:** exact token IDs/KV state; strongest guarantees.
2. **Text-prefix oracle validated as equivalent:** certified only after empirical prefix-law tests.
3. **Sample-only oracle:** samples define the operational distribution; no chain-rule claims.
4. **Prompt-echo heuristic:** useful for exploration, never counted as exact probability mass.

As of August 10, 2026, this must be runtime-discovered rather than encoded from provider names. OpenAI’s official example describes chosen-token and limited top-token conditional logprobs and currently documents `top_logprobs` from 0 to 5; Structured Outputs guarantees schema adherence, but that alone does not establish whether reported probabilities are before or after decoder masking. Anthropic’s current documentation says assistant prefilling is no longer supported on Sonnet 4.6 and later/current 5-series models, while OpenRouter’s protocol accepts values through 20 without guaranteeing that every routed provider implements them. ([OpenAI Developers][1])

### The core object is a weighted token automaton, not a digit trie

A token may contain digits, punctuation, structural JSON, or several numeric characters. Use a deterministic parser/transducer over token bytes:

[
(\text{parser state},\text{token bytes})
\longrightarrow
(\text{new parser state},\text{reachable value set}).
]

Represent complete decimals exactly as canonical integer coefficients plus decimal exponents, not `f64`. Strip trailing zeroes only after the complete scalar field terminates.

### Do not let unresolved mass “compound” into vague path intervals

At each probed prefix (x), keep the omitted alternatives as one disjoint stopping cell. If (m_x) is the exact prefix mass, (K_x) is every token whose probability has been observed, and

[
\rho_x=1-\sum_{t\in K_x}P(t\mid x),
]

then the omitted-token cylinder has exact aggregate mass (m_x\rho_x). Do not invent child masses beneath it. A later sample that reveals an omitted token and its chosen-token logprob simply splits that cell.

This gives a clean measure representation:

[
\mu
===

\sum_{e\in E}p_e\delta_{v_e}
+
\sum_{b\in\mathcal B}q_b\nu_b,
\qquad
\operatorname{supp}\nu_b\subseteq S_b.
]

Here:

* (E) is the set of fully resolved token traces;
* (p_e) and (v_e) are exact trace mass and parsed value;
* (b) is an unresolved cylinder;
* (q_b) is its known mass;
* (S_b) is its reachable value set;
* (\nu_b) is its unknown conditional value distribution.

For bounded (h),

[
L_h
===

\sum_e p_e h(v_e)
+
\sum_b q_b\inf_{v\in S_b}h(v),
]

[
U_h
===

\sum_e p_e h(v_e)
+
\sum_b q_b\sup_{v\in S_b}h(v).
]

That is the fundamental certificate.

### Decoder reconstruction is not epistemic calibration

The pushforward measure is the distribution of outputs under a particular prompt, temperature, schema, reasoning mode, and decoder. It is not automatically the model’s posterior over the true quantity. Temperature entropy, lexical ambiguity, and prompt-format behavior all enter.

The product should distinguish:

* **reconstruction accuracy:** how faithfully the kernel recovers the decoder-induced measure;
* **elicitation calibration:** whether that measure predicts real uncertainty or outcomes.

The second requires held-out proper-scoring evaluation. It cannot be certified from logprobs alone.

---

## 1. Estimator design

**Concrete recommendation: use a resolved-head plus stratified residual Monte Carlo estimator as the default. Add an atom-inclusion Horvitz–Thompson estimator only for recycling exploration samples.**

### Samples drawn from (P) do not require importance weighting

For (Y_i\sim P), the ordinary sample estimator already has weight one. Multiplying each observation by (1/P(Y_i)) would estimate something closer to counting measure over sequences and usually produces catastrophic variance.

Chosen-token logprobs are valuable because they:

* reveal exact probabilities of tail edges encountered by samples;
* give exact complete-trace probabilities when all generated/terminating tokens are scored;
* enable inclusion-probability estimators for distinct sampled atoms;
* validate whether probes and samples belong to the same distribution.

They are not necessary to make same-(P) sampling unbiased.

### Default estimator: exact head plus residual rejection sampling

Freeze a resolved region (C\subseteq\Omega), selected using probes and an **independent exploration sample**. Let

[
q=P(Y\notin C),
]

and let (\mu_C) be the exact submeasure on (C).

Now sample until obtaining (M) traces outside (C):

[
Y^{R}_1,\ldots,Y^{R}_M
\overset{\text{iid}}{\sim}
P(\cdot\mid Y\notin C).
]

Then output the coherent random measure

[
\widehat\mu
===========

\mu_C
+
\frac{q}{M}\sum_{j=1}^M\delta_{f(Y^R_j)}.
]

It has exactly unit mass and, for every measurable set (A),

[
E[\widehat\mu(A)]=\mu(A).
]

For a scalar functional (\mu_h=E[h(f(Y))]),

[
\widehat\mu_h
=============

\mu_{C,h}
+
q\overline h_R,
]

[
\operatorname{Var}(\widehat\mu_h)
=================================

\frac{q^2}{M}\operatorname{Var}(h(f(Y))\mid Y\notin C).
]

This is the cleanest probe/sample combination. Heavy mass is integrated exactly; sampling is spent only on the unresolved conditional distribution.

The expected number of unconditional calls needed for (M) residual hits is (M/q). When (q) is tiny, its maximum target impact (q,\operatorname{osc}(h)) is also tiny, so the certificate will often stop before residual sampling becomes expensive.

### Stratify unresolved cells where exact conditioning is available

For disjoint unresolved cells (b) with known masses (q_b), obtain fixed numbers (M_b) of exact conditional samples:

[
Y_{bj}\sim P(\cdot\mid b).
]

Then

[
\widehat\mu_h
=============

\mu_{E,h}
+
\sum_b q_b\overline h_b,
]

[
\operatorname{Var}(\widehat\mu_h)
=================================

\sum_b\frac{q_b^2\sigma_b^2}{M_b}.
]

With per-sample costs (c_b), the variance-minimizing Neyman allocation is

[
M_b
\propto
\frac{q_b\sigma_b}{\sqrt{c_b}}.
]

Before estimating (\sigma_b), use the robust proxy

[
\sigma_b\leq\frac{\operatorname{osc}_{S_b}(h)}{2},
]

giving

[
M_b\propto
\frac{q_b,\operatorname{osc}_{S_b}(h)}{\sqrt{c_b}}.
]

Aggregate omitted-token buckets usually cannot be directly forced. They must be sampled by unconditional rejection until particular omitted tokens are discovered and split out.

### Simple unbiased subtraction estimator

For a resolved region (C) frozen independently of an estimation batch,

[
\widehat\mu_h^{\mathrm{diff}}
=============================

\mu_{C,h}
+
\frac1N\sum_{i=1}^N
h(f(Y_i)),\mathbf 1{Y_i\notin C}
]

is unbiased, with

[
\operatorname{Var}
==================

\frac1N
\operatorname{Var}!\left(
h(f(Y))\mathbf 1{Y\notin C}
\right).
]

For distributions,

[
\widehat\mu(A)
==============

\mu_C(A)
+
\frac1N\sum_i
\mathbf 1{f(Y_i)\in A,;Y_i\notin C}.
]

Its total mass is random, although its expected mass is one. The residual rejection estimator is usually preferable for a user-facing coherent measure.

### A real Horvitz–Thompson construction

There is a principled HT estimator using the inclusion probability of each **distinct token trace**.

Let (p_y=P(Y=y)), and let (D_N) be the set of distinct traces appearing in (N) iid samples. The inclusion probability of trace (y) is

[
\pi_y
=====

# P(y\in D_N)

1-(1-p_y)^N.
]

For a fixed probe-covered region (C),

[
\widehat\mu_h^{\mathrm{HT}}
===========================

\mu_{C,h}
+
\sum_{y\in D_N\setminus C}
\frac{p_y h(f(y))}
{1-(1-p_y)^N}.
]

It is unbiased because

[
E!\left[
\mathbf 1{y\in D_N}
\frac{p_yh_y}{\pi_y}
\right]
=======

p_yh_y.
]

This operates at token-trace level, so alternate tokenizations that parse to the same value are correctly summed after estimation.

Under Poissonized sampling with expected sample count (n),

[
\pi_y=1-e^{-np_y},
]

and distinct-atom inclusions are independent, yielding

[
\operatorname{Var}(\widehat\mu_h^{\mathrm{HT}})
===============================================

\sum_{y\notin C}
(p_yh_y)^2
\frac{1-\pi_y}{\pi_y}.
]

For (p_y\ll1/N), (p_y/\pi_y\approx1/N), so it behaves like ordinary Monte Carlo. For heavy atoms, repeated observations collapse into one exact weighted atom, which can reduce variance greatly.

Its disadvantages are material:

* estimated total mass is random;
* diffuse tails receive little benefit over ordinary sampling;
* it is sensitive to errors in (p_y);
* it requires the complete trace probability, including termination;
* it is invalid if the reported per-trace likelihood is conditional on a varying hidden deployment rather than the operational mixture probability.

A generalized-regression correction can use the known residual mass (q):

[
\widehat\mu_h^{\mathrm{GREG}}
=============================

\widehat\mu_h^{\mathrm{HT}}
+
\beta\left(q-\widehat q^{\mathrm{HT}}\right),
]

with (\beta) fixed independently or estimated by cross-fitting. This remains unbiased and often removes the undesirable total-mass fluctuation for scalar targets.

A Hájek normalization,

[
q,
\frac{\widehat\mu_h^{\mathrm{HT}}}
{\widehat q^{\mathrm{HT}}},
]

is coherent and often lower-MSE, but only asymptotically unbiased.

### Adaptivity rule

Do not discover exact atoms and then subtract them using the same sample without correction. That produces selection bias.

Use either:

* an exploration batch followed by an independent residual-estimation batch;
* (K)-fold cross-fitting, where each fold is evaluated against a resolved set built from other folds;
* the atom-inclusion HT correction.

### Confidence intervals

For cell (b), suppose (h\in[\ell_b,u_b]) and (n_b) conditional samples produce mean (\bar h_b) and variance (s_b^2). A fixed-sample empirical-Bernstein radius is of the form

[
r_b
===

\sqrt{\frac{2s_b^2\log(3/\alpha_b)}{n_b}}
+
\frac{3(u_b-\ell_b)\log(3/\alpha_b)}{n_b}.
]

Then a simultaneous interval is

[
\left[
\mu_{E,h}
+\sum_bq_b\max(\ell_b,\bar h_b-r_b),
\quad
\mu_{E,h}
+\sum_bq_b\min(u_b,\bar h_b+r_b)
\right],
]

with (\sum_b\alpha_b\leq\alpha). Cells with no samples retain their full support interval. Because your sampler and planner are adaptive, use confidence sequences rather than fixed-(n) intervals in production.

For a residual CDF estimated from (M) fixed residual samples, DKW gives

[
\sup_t\left|
\widehat F(t)-F(t)
\right|
\leq
q\sqrt{\frac{\log(2/\alpha)}{2M}},
]

after adding the exact resolved CDF. This is a particularly clean distribution-level certificate.

---

## 2. Expansion policy

**Concrete recommendation: replace reachable mass with target oscillation mass, then use one-step value of information. Do not claim a universal near-optimality result for the greedy policy.**

For a linear target

[
F(\mu)=E_\mu[h(V)],
]

the current deterministic envelope width is

[
W_h(T)
======

\sum_{b\in\mathcal B}
q_b\omega_b,
\qquad
\omega_b
========

## \sup_{v\in S_b}h(v)

\inf_{v\in S_b}h(v).
]

Reachable-mass priority (q_b) is appropriate only when all cells have similar target oscillation and similar expected contraction after expansion.

The admissible maximum value of expanding cell (b) is

[
A_b=q_b\omega_b.
]

If an expansion would produce children (bj), including a new aggregate residual child, its realized gain is

[
\Delta_b
========

## q_b\omega_b

\sum_jq_{bj}\omega_{bj}.
]

The practical index should be

[
I_b
===

\frac{E[\Delta_b\mid\mathcal D]}{c_b},
]

where the expectation is estimated using exploration samples, sibling statistics, and grammar geometry. Use (q_b\omega_b/c_b) as an optimistic upper bound for candidate pruning.

### For (E[\log R]), distance from the current mean does not matter

Let (Z=\log R). For the linear target (E[Z]), the contribution to envelope width is

[
q_b
\left(
\sup_{S_b}Z-\inf_{S_b}Z
\right).
]

A narrow cell far above the current mean is not intrinsically more important than an equally narrow cell near it. Distance matters for variance, squared loss, tail risk, or a nonlinear statistic—not for the worst-case interval width of an ordinary mean.

Distance can also appear if you represent prefix mass itself as an interval. The disjoint-cylinder construction avoids that: each unresolved aggregate cell has known mass and unknown location.

### General nonlinear functionals

For a differentiable functional (F), use the influence function as a first-order planner heuristic:

[
A_b^F
\approx
q_b,
\operatorname{osc}_{v\in S_b}
\operatorname{IF}_F(v;\widehat\mu).
]

Examples:

* Mean of (h): (\operatorname{IF}=h-F), whose oscillation is just that of (h).
* Variance: (\operatorname{IF}(v)) contains ((v-\mu)^2), so distant cells matter.
* Ratio of expectations:
  [
  F=\frac{E[a(V)]}{E[b(V)]},
  \qquad
  \operatorname{IF}(v)
  ====================

  \frac{a(v)-F b(v)}{E[b(V)]}.
  ]
* Quantiles: prioritize cells that can move mass across the current CDF/quantile bracket. Cells wholly below or above the bracket often have zero immediate value.

Always recompute the exact functional envelope after expansion; use influence functions only for scheduling.

### Exact offline formulation

If the entire finite child tree and edge masses were already known, the optimal budgeted expansion is a tree-knapsack dynamic program. For cell (b), let (J_b(k)) be the minimum remaining width using at most (k) expansions:

[
J_b(0)=q_b\omega_b,
]

[
J_b(k)
======

\min\left{
q_b\omega_b,;
\min_{1+\sum_jk_j\leq k}
\sum_jJ_{bj}(k_j)
\right}.
]

That is exact, pseudo-polynomial in the call budget.

In your actual setting, child probabilities and supports are observed only after paying for the expansion. This is a Bayesian adaptive tree-search/POMDP problem. A local index has no general additive-(\epsilon) guarantee: an ancestor can have almost no immediate gain while unlocking a highly valuable descendant, creating arbitrary failures for one-step greedy policies.

A good production planner is:

1. Rank cells by (q_b\omega_b/c_b).
2. For the top few, estimate one-step contraction from sample-derived token distributions.
3. Occasionally run a depth-two rollout to detect unlock effects.
4. Compare the best probe action against the best additional-sampling action.
5. Stop from the global certificate, not a fixed node count.

For additional conditional sampling in cell (b), the approximate reduction in confidence half-width from one more sample is

[
\frac{q_b\widehat\sigma_b}{\sqrt{c_b}}
\left(
\frac1{\sqrt{n_b}}
------------------

\frac1{\sqrt{n_b+1}}
\right),
]

up to the confidence multiplier. This puts probing and sampling into approximately common “target-width reduction per dollar” units.

---

## 3. Residual mass semantics

**Concrete recommendation: make the credal/random-set object authoritative; make the point estimate a sample-derived overlay. Do not use an empirical digit prior as the default source of single-number estimates.**

The residual representation

[
\mu
===

\sum_ep_e\delta_{v_e}
+
\sum_bq_b\nu_b,
\qquad
\operatorname{supp}\nu_b\subseteq S_b
]

is already a precise random-set or belief-function object. It has three useful properties:

* it never invents probability structure inside an omitted bucket;
* it composes naturally into target-specific lower and upper expectations;
* it remains valid under arbitrary adversarial placement of residual mass.

A prior such as “omitted mass follows empirical digit frequencies” is not innocuous. Numeric-prefix distributions are strongly context-, scale-, punctuation-, and tokenizer-dependent. A global empirical prior can confidently put residual mass in exactly the wrong semantic region.

The product-grade output should contain all of:

```text
point_measure:
    exact atoms + weighted empirical conditional measures

certificate:
    deterministic probe-only envelope
    statistical confidence envelope
    combined envelope

mass_ledger:
    exact_complete
    known_frontier
    top_k_aggregate_residual
    inaccessible_tokenization
    invalid_or_refusal

oracle_uncertainty:
    repeated-prefix probability variation
    prefix-law test results
    tokenizer/decoder fingerprint

elicitation_uncertainty:
    natural-vs-encoded format shift
    external calibration results
```

For a target (h), return:

```text
estimate
probe_only_interval
confidence_interval_95
resolved_mass
largest_remaining_cells
stopping_reason
```

### Where a Bayesian layer belongs

A Bayesian residual model is useful as:

* a search-policy prior;
* smoothing for a UI density;
* an optional posterior when sampling is sparse;
* a learned model across many prompts from the exact same model/prompt family.

It should have a versioned prior identifier and sensitivity output. It must not silently replace the certified interval.

When labs demand one number, return the coherent stratified-sample estimate. When no residual samples exist, the minimax absolute-error scalar estimate is the midpoint of the credal interval, but label it as a minimax imputation—not a calibrated posterior mean.

### Calibration needs two evaluations

Run separate tests for:

1. **Reconstruction:** compare the kernel against exhaustive/full-logit ground truth.
2. **Truth calibration:** compare predicted CDFs or probabilities against outcomes using coverage, PIT, CRPS, Brier score, or log score.

A perfectly reconstructed decoder distribution can still be badly calibrated to reality.

---

## 4. Tokenization off-manifold

**Concrete recommendation: do not assume canonical-only mass is tiny. Measure it directly from natural output traces, and convert the empirical upper bound into a target-error bound.**

For a sampled full token trace (y):

1. Decode its exact bytes to text.
2. Re-encode the entire output using the exact tokenizer/version.
3. Mark
   [
   C(y)=\mathbf 1{y=\operatorname{encode}(\operatorname{decode}(y))}.
   ]

Do this on the full output, not the scalar substring alone, because canonical tokenization can merge across the opening/closing delimiter.

Let (X) of (N) samples be noncanonical. A one-sided Clopper–Pearson upper bound is

[
q_{\mathrm{nc}}^{U}
===================

\operatorname{BetaQuantile}
(1-\alpha;X+1,N-X).
]

With zero observations,

[
q_{\mathrm{nc}}^{U}
===================

1-\alpha^{1/N}
\approx
\frac{-\log\alpha}{N}.
]

At 95% confidence this is approximately (3/N).

If the canonical-conditioned approximation is (P_C=P(\cdot\mid C)), then for (h\in[a,b]),

[
\left|E_P[h]-E_{P_C}[h]\right|
\leq
q_{\mathrm{nc}}(b-a),
]

and for CDFs,

[
\sup_t|F_P(t)-F_{P_C}(t)|
\leq q_{\mathrm{nc}}.
]

This turns the tokenization audit into a direct go/no-go criterion:

[
q_{\mathrm{nc}}^{U}\operatorname{osc}(h)
\leq
\text{allocated tokenization-error budget}.
]

### Canonical-only traversal is conditionally sound

It is sound when all three hold:

* the semantic output domain is bounded;
* natural samples establish a sufficiently small (q_{\mathrm{nc}});
* the omitted mass is retained in the certificate rather than silently discarded.

It is not sound merely because noncanonical prefixes “feel off-manifold.”

For a fixed model and exact token-ID prefix, an unusual sequence still has a well-defined chain-rule conditional. Its downstream behavior may be strange, but its aggregate importance is already attenuated by its path mass. A target-aware planner will naturally avoid spending calls there.

### Public APIs often cannot force the relevant object

Supplying the text `"2."` typically asks the provider to tokenize `"2."` canonically. It does not force the exact output token sequence `["2", "."]`. Therefore:

* token-ID prefix support permits complete token-tree traversal;
* text prefix support permits only the provider’s chosen input tokenization;
* prompt echo is not exact traversal at all.

For inaccessible tokenizations, natural sampling is the correct measurement channel.

### Cheap refinement beyond a global bound

For every heavy decoded value:

* collect all naturally observed tokenizations;
* sum their exact trace probabilities when stable trace likelihoods are available;
* maintain a value-level alias ledger;
* estimate unseen alias mass from the global/per-family noncanonical bound.

Exact enumeration of all tokenizations of a string can be exponentially large because different tokenizations produce different transformer states. Do not make that the default algorithm. The empirical mass bound is usually the right abstraction.

Also ensure termination is an observable generated delimiter. If the API strips an EOS token and does not report its probability, complete sequence probabilities are unavailable and atom-HT must be disabled. A JSON closing quote/delimiter inside the scored output is preferable.

---

## 5. Digit-grid reparameterization

**Concrete recommendation: treat digit alignment as a different elicitation instrument, not as an exact shortcut to the original free-form distribution. Prefer a low-radix interval code over decimal digits when top-(k) is small.**

The digit grid solves real problems:

* finite semantic support;
* explicit precision;
* reduced punctuation/termination ambiguity;
* more predictable tokenizer behavior;
* easier reachable-set calculations.

But it does **not** eliminate the autoregressive tree.

Suppose the output has digits (D_1,\ldots,D_m). One sampled call returns

[
P(D_1),
\quad
P(D_2\mid D_1=d_1),
\quad\ldots,\quad
P(D_m\mid D_{1:m-1}=d_{1:m-1})
]

only along the sampled prefix. It does not return the marginal (P(D_j)), nor the conditionals under counterfactual earlier digits.

“Nested probes to decorrelate later digits” should be understood as **counterfactual marginalization**, not decorrelation.

A distribution over (m) decimal digits has up to (10^m) leaves. A single path exposes (m) internal nodes. No representation can recover an arbitrary (10^m)-leaf distribution from (O(mk)) reported numbers without additional structural assumptions. Your answer-key method works because all categories are siblings at depth one.

With `top_logprobs=5`, decimal digits are also poorly matched to the oracle: only five of ten alternatives can be returned at each visited position, leaving an aggregate residual.

### Better encoded instrument: bounded radix tree

For an elicited distribution head, define

[
Z=\operatorname{signed}\log R\in[-B,B],
]

then recursively partition the interval into (b\leq k) children. With (k=5), use a four-way code and reserve one symbol for `STOP`, `TIE`, or overflow.

At depth (d), the instrument has (4^d) cells. Every expanded node can expose all four child probabilities when the decoder truly limits the next-token alphabet to those symbols.

Worst-case exact traversal still requires

[
1+4+\cdots+4^{d-1}
==================

\frac{4^d-1}{3}
]

internal-node queries. The gain is not elimination of the tree; it is:

* zero top-(k) residual at each valid node;
* no decimal prefix ambiguity;
* uniform multiplicative resolution;
* clean target-aware pruning.

Text-level JSON schema does not by itself guarantee one-token symbols: a token can span a quote, comma, or multiple code symbols. Audit actual token traces, or use token-level constrained decoding in the internal-lab regime.

### Natural free-form versus encoded distribution

Maintain two explicitly different products:

1. **Faithful reconstruction**

   * Target: the model’s natural/schema-constrained free-form decoder distribution.
   * Method: partial token automaton plus samples.
   * Strength: behavioral fidelity.
   * Weakness: tokenization and traversal cost.

2. **Calibrated distribution elicitation**

   * Target: the model’s responses under a purpose-built bounded radix instrument.
   * Method: low-branching prefix tree.
   * Strength: much cleaner measurement.
   * Weakness: format-induced judgment shift.

A natural mode followed by a radix distribution can be a strong practical instrument, but it estimates a new conditional distribution. If the second prompt includes a sampled natural mode (M=m), the result is a distribution conditional on that anchor and on the second prompt—not the unconditional distribution from the first prompt.

Safer versions use:

* a deterministic task-defined center;
* an independently estimated center frozen before evaluation;
* integration over multiple anchor samples;
* explicit calibration of the two-stage protocol as one instrument.

Do not merge the natural and grid distributions and call the result the original model pushforward.

---

## 6. Multi-dimensional composition

**Concrete recommendation: preserve a joint cell measure until all semantically dependent fields terminate. For direction and ratio, replace the pair with one signed log-ratio variable.**

Naively multiplying independently reconstructed field marginals is wrong for a single autoregressive output:

[
P(A,B)=P(A)P(B\mid A\text{ token trace}),
]

not generally (P(A)P(B)).

More subtly, two token traces that parse to the same first-field value can induce different distributions for later fields. Therefore, for a joint object,

[
P(a,b)
======

\sum_{y_A:f_A(y_A)=a}
P(y_A)
\sum_{y_B:f_B(y_B)=b}
P(y_B\mid y_A).
]

You may quotient token traces to semantic values only after the last dependent field has terminated. Equality of decoded text is not equality of transformer state.

For credal reconstruction, retain cells

[
(q_b,S_b),
\qquad
S_b\subseteq V_1\times\cdots\times V_m.
]

Then for any joint target (h),

[
L_h
===

\sum_bq_b\inf_{\mathbf v\in S_b}h(\mathbf v),
\qquad
U_h
===

\sum_bq_b\sup_{\mathbf v\in S_b}h(\mathbf v).
]

Marginalization is projection of (S_b). Multiplying marginal interval bounds destroys dependence information and can produce impossible combinations.

### Direction plus ratio

Use

[
Z=
\begin{cases}
+\log R,& A>B,\
0,& \text{tie},\
-\log R,& B>A.
\end{cases}
]

Then derive:

[
\text{direction}=\operatorname{sign}(Z),
\qquad
\text{ratio}=e^{|Z|}.
]

This:

* removes logically inconsistent direction/ratio pairs;
* gives a bounded symmetric support;
* makes the main estimator linear in the quantity used by pairwise cardinal fitting;
* reduces a two-field joint problem to a one-dimensional distribution with a possible tie atom.

For genuinely separate fields, branch exactly on low-cardinality fields first when their alternatives fit inside top-(k), then reconstruct continuous fields conditionally. If fields are obtained from separate independent API calls, multiplying the resulting measures is mathematically valid for the **independent-call instrument**, but it deliberately removes any shared latent uncertainty.

---

## 7. Product/kernel framing

**Concrete recommendation: make the core a capability-parametric certified-measure engine. The most valuable first capability is an anytime-certified CDF over bounded signed log-ratios, not an arbitrary dense PMF.**

A suitable Rust-facing architecture is:

```rust
trait TraceOracle {
    fn capabilities(&self) -> OracleCapabilities;

    async fn sample(
        &self,
        request: &FrozenInference,
        n: usize,
    ) -> Result<Vec<Trace>>;

    async fn probe_prefix(
        &self,
        request: &FrozenInference,
        prefix: &ExactPrefix,
    ) -> Result<NodeObservation>;
}
```

```rust
struct OracleCapabilities {
    guarantee: PrefixGuarantee, // TokenExact, TextValidated, SampleOnly, Heuristic
    dense_logits: bool,
    max_top_logprobs: Option<usize>,
    exact_token_ids: bool,
    exact_token_bytes: bool,
    exact_prefix_forcing: bool,
    scored_termination: bool,
    batch_prefixes: bool,
    kv_cache_reuse: bool,
}
```

```rust
struct FrozenInference {
    provider: String,
    deployment_or_snapshot: String,
    model_fingerprint: Option<String>,
    tokenizer_hash: Option<String>,
    prompt_hash: Hash,
    schema_hash: Hash,
    temperature: f64,
    top_p: f64,
    penalties: Penalties,
    reasoning_mode: ReasoningMode,
    routing_policy: RoutingPolicy,
}
```

The semantic language is separate:

```rust
trait SemanticLanguage {
    type State;
    type Value;
    type ValueSet;

    fn initial_state(&self) -> Self::State;
    fn advance(
        &self,
        state: &Self::State,
        token_bytes: &[u8],
    ) -> Transition<Self::State>;

    fn reachable_values(&self, state: &Self::State) -> Self::ValueSet;
    fn parse_complete(&self, state: &Self::State) -> Option<Self::Value>;
}
```

The reconstruction layer stores:

```rust
struct CertifiedMeasure<V, S> {
    exact_atoms: Vec<ExactAtom<V>>,
    unresolved_cells: Vec<ResidualCell<S>>,
    empirical_conditionals: Vec<EmpiricalCellMeasure<V>>,
    mass_accounting: MassLedger,
    oracle_audit: OracleAudit,
}
```

The objective layer should accept functionals, not only bins:

```rust
enum Functional<V> {
    Mean(Box<dyn Fn(&V) -> f64>),
    CdfAt(V),
    Quantile(f64),
    TailProbability { threshold: V },
    Custom(Box<dyn CertifiedFunctional<V>>),
}
```

And the planner receives:

```rust
struct ReconstructionGoal<V> {
    targets: Vec<Functional<V>>,
    tolerances: Vec<f64>,
    confidence: f64,
    budget: Budget,
}
```

The output certificate should be resumable:

```rust
struct ReconstructionCertificate<V> {
    point_measure: WeightedMeasure<V>,
    exact_atoms: Vec<ExactAtom<V>>,
    residual_cells: Vec<ResidualCellSummary>,
    functionals: Vec<FunctionalCertificate>,
    mass_ledger: MassLedger,
    guarantee_level: GuaranteeLevel,
    oracle_fingerprint: OracleFingerprint,
    continuation_state: ReconstructionState,
}
```

### Full-access and constrained-access regimes

The same core works in both:

* **Full logits/token-prefix access:** every expanded node has residual zero; batch frontier states and reuse KV caches.
* **Top-(k) exact prefix:** node residuals remain explicit; samples split tail edges.
* **Sample-only:** exact atom/cylinder fields may remain empty; output is empirical with statistical bands.
* **Heuristic text continuation:** probes may guide the planner but cannot tighten certified mass bounds.

No planner should be allowed to promote a heuristic observation into exact mass.

### The first wedge

Build one excellent object:

> `SignedLogRatioDistribution`: a bounded one-dimensional measure returning an exact-atom ledger, an anytime CDF band, (E[Z]), quantile intervals, and a complete uncertainty decomposition.

Why this first:

* it directly serves pairwise ratio fitting;
* direction and magnitude collapse into one variable;
* (E[Z]) is linear and has additive envelope width;
* a CDF is representation-invariant and supports arbitrary later binning;
* quantiles and tail probabilities follow by inversion;
* the same object cleanly benchmarks full-logit, top-(k), and sample-only oracles.

The defensible product claim is not “full arbitrary-precision PMFs from five logprobs.” It is:

> **Target-aware reconstruction of numeric model distributions with auditable mass accounting and anytime error certificates.**

---

## Minimal experiment sequence

### 1. August 10, 2026 — Oracle-law and decoder-semantics probe

Use roughly 24 deliberately difficult numeric prefixes covering:

* token-spanning numerals such as `"2.5"`;
* canonical and noncanonical segmentations;
* decimal point, sign, exponent, and termination;
* strict schema versus unconstrained generation;
* top-(k) and sampled-token-outside-top-(k) cases.

For each prefix/configuration, collect repeated continuations and test the martingale residual

[
\sum_i
\left(
\mathbf 1{T_i=t}
----------------

p_i(t)
\right)
]

for heavy tokens and

[
\mathbf 1{T_i\notin K_i}
------------------------

\left(1-\sum_{t\in K_i}p_i(t)\right)
]

for aggregate residuals.

Also compare:

* natural conditional frequencies after prefix (x);
* exact assistant/token prefill where available;
* schema-forced prefix;
* prompt-echo continuation.

**Decision:** classify every model/route into `TokenExact`, `TextValidated`, `SampleOnly`, or `Heuristic`. Test whether constrained-output logprobs reflect the actual post-mask sampling probabilities.

### 2. August 11, 2026 — Tokenization and termination mass audit

Collect approximately 20,000 natural numeric traces across a representative prompt set.

For each trace:

* preserve exact token bytes/IDs;
* compare against canonical re-encoding;
* identify the first noncanonical divergence;
* verify whether the terminating delimiter is scored;
* group alternate traces by canonical decimal value.

Compute (q_{\mathrm{nc}}^U) globally and by prompt family. The pass condition for a target (h) is

[
q_{\mathrm{nc}}^U\operatorname{osc}(h)
<
\frac14\epsilon_h.
]

If this fails, canonical-only traversal cannot receive the strongest guarantee tier.

### 3. August 12–13, 2026 — Exact-ground-truth estimator shootout

Use a local or lab-accessible model with full logits and exact token-prefix control. Define a finite bounded numeric language with roughly (10^3)–(10^4) leaves and exhaustively enumerate its true measure.

Then emulate:

* top-(5);
* top-(20);
* chosen-token logprobs only;
* inaccessible noncanonical prefixes;
* mild oracle probability noise.

Compare at equal call/token budgets:

1. plain Monte Carlo;
2. mass-first partial tree;
3. exact-head plus residual rejection sampling;
4. post-stratified conditional sampling;
5. atom-inclusion HT;
6. HT plus GREG correction.

Measure:

* bias;
* RMSE for (E[Z]);
* CDF sup error;
* 95% interval coverage;
* cost to reach fixed target width;
* sensitivity to trace-probability error.

The expected winner is exact-head plus stratified residual sampling, with HT useful mainly for recycling exploration data.

### 4. August 14, 2026 — Instrument-shift experiment

Randomize prompts among:

* natural free-form decimal;
* fixed-width spaced decimal;
* semantic digit fields;
* four-way signed-log-ratio radix code.

Measure separately:

* natural answer quality;
* tokenization alias mass;
* reconstruction cost;
* external calibration/proper scores;
* Wasserstein/CDF differences between instruments;
* dependence on prompt ordering and anchor choice.

This determines whether the radix head is a superior calibrated elicitation instrument despite not reconstructing the natural free-form distribution.

### 5. August 17, 2026 — Target-aware planner benchmark

On at least 200 held-out prompts, compare:

* reachable-mass-first;
* (q_b\operatorname{osc}_b(h))-first;
* one-step expected envelope contraction;
* one-step contraction plus depth-two rollout;
* unified probe-versus-Neyman-sampling scheduling.

Run targets:

[
E[Z],\qquad
P(Z>0),\qquad
q_{0.1},q_{0.5},q_{0.9},
]

and uniform CDF width.

Primary metric:

[
\text{dollars or latency to a valid target certificate},
]

not resolved mass or number of expanded nodes. The result should decide the first production planner without requiring a general optimal-search theorem.

[1]: https://developers.openai.com/cookbook/examples/using_logprobs "https://developers.openai.com/cookbook/examples/using_logprobs"