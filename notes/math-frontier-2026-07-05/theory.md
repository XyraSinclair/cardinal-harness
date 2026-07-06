# Math frontier, 2026-07-05

Grounded against the shipped code, not the docs' prose about it:
`src/rating_engine.rs` (weighted-Laplacian IRLS solve, `compute_hcr`, `cycle_dim`,
effective resistance, the planner's `0.5·ln(1+λr)` information-gain rule),
`src/gain_calibration.rs` (bilinear template-gain fit), `src/rerank/anp.rs`
(Cesàro-limit supermatrix), `src/rerank/spin.rs` (odd/even response-function
sweep), and the `seriate` crate's primitive types (`Entity`, `Attribute`,
`Presentation`, `JudgementRecord`, `AnswerEvidence`/`PmfCompleteness`,
`InstrumentKind`/`AcquisitionMode`, the reflection algebra). Each section
below states a precise claim, says why it changes what the engine computes
or trusts, and gives one buildable receipt. Two of the operator's seven
"wilder" suggestions are rejected below with reasons — rejection is a finding,
not a cop-out.

---

## 1. The full Hodge decomposition: curl splits into local and harmonic

**The gap.** `compute_hcr` (`rating_engine.rs:816-854`) computes
`Σλr²/Σλμ²` and its own doc comment already says the honest thing: this is
"the field's curl + harmonic component," not curl alone. The FIRST_PRINCIPLES
table's claim that we compute "only the curl fraction" undersells what's
already true and overclaims what's missing — the two pieces are conflated,
not separately measured. Separately, `cycle_dim = m − n + components`
(`rating_engine.rs:1619`) is already the exact dimension of the residual's
home: the cyclomatic number (first Betti number) of the comparison graph as a
1-complex. That number is sitting right next to the un-split energy.

**The statement.** Fix the weighted inner product `⟨X,Y⟩_w = Σ_e w_e X_e Y_e`
on edge flows, `w_e = λ_eff(e)`. The combinatorial gradient
`(grad φ)_{ij} = φ_i − φ_j` has `w`-adjoint `div = grad*`, and
`L = grad* W grad` is exactly the reduced Laplacian `l_red` the solver builds
in `build_system`. The normal-equations solve computes the orthogonal
projection of `μ` onto `im(grad)`; the residual `r = μ − grad φ*` is,
by construction, `w`-orthogonal to every gradient (`div(Wr) = 0`) — it lives
in `ker(div)`, the **cycle space**, dimension `cycle_dim`. This much is
already implicit in the code.

The new step: given the *filled triangles* `T` — 3-cliques `(i,j,k)` where all
three fused edges `ij, jk, ki` were actually judged (already the natural
object; the code's doc comment notes repeat judgements of a pair fuse into
one edge before this stage) — define `curl: R^E → R^T`,
`(curl x)_{ijk} = x_{ij} + x_{jk} + x_{ki}` under a fixed cyclic orientation,
with `w`-adjoint `curl*`. Then

```
R^E = im(grad) ⊕_w im(curl*) ⊕_w H,   H = ker(div) ∩ ker(curl)
```

an exact three-way orthogonal (Pythagorean) split — this is combinatorial
Hodge theory (Jiang–Lim–Yao–Ye 2011) applied to the flag complex our own
judged triads generate. `im(curl*)` is the **locally cyclic** part: curl
supported entirely on triangles we actually filled in, i.e. disagreement a
per-triad audit (check A>B>C>A) *can* catch. `H`, the harmonic space, is
disagreement that is divergence-free AND curl-free on every filled triangle,
yet still nonzero globally — a cycle whose chords were never judged. Its
dimension is exactly `cycle_dim − rank(curl restricted to T)`: **harmonic
disagreement requires a cycle longer than a triangle, with the closing
chords never elicited.** No amount of "spot-check triads" ever detects it,
by construction — it's invisible to any test that only ever looks at
3-cliques.

**Why load-bearing, not decoration.** The planner (`plan_edges_for_rater`)
optimizes for information-per-dollar and ships budgets at `O(n)`, not
`O(n²)` — i.e. it deliberately produces *sparse* comparison graphs. Sparse
graphs are exactly the graphs whose cycles are mostly long, unfilled ones:
efficiency and triad-auditability are in tension by construction. A team
that only ever audits "does the judge close its triangles" can watch HCR
climb from run to run and have zero visibility into whether the growth is
the harmless, catchable kind (curl) or the invisible kind (harmonic) that no
spot-check will ever surface. This is the mathematically precise version of
"is the frustration receipt actually telling us something we can act on."

**Receipt.** `compute_hodge_split(edges, residuals, lam_eff, cfg) ->
HodgeSplit { local_curl_frac, harmonic_frac, filled_triangles, harmonic_dim }`
— build `C` (triangle × edge incidence, ±1 per filled triangle), solve
`(C W⁻¹ Cᵀ) z = Cr` for the triangle potential (an "upper Laplacian" solve,
the same Cholesky machinery already used for the ordinary Laplacian), then
`local = W⁻¹Cᵀz`, `harmonic = r − local`, energies from `⟨·,·⟩_w`. Test in a
new `tests/hodge_split.rs`: (a) a single fully-judged triangle A/B/C with a
planted quantization inconsistency (reuse `ladder_curl.rs`'s judge) — expect
`harmonic_frac ≈ 0`, all curl local, since the only cycle present *is* the
filled triangle; (b) a 4-cycle A→B→C→D→A with the two diagonals AC, BD
*never* queried, planted so the log-ratios don't close around the loop —
`cycle_dim = 1`, zero filled triangles by construction, so
`harmonic_frac ≈ 1` necessarily. The two planted cases should sit at
opposite ends of the same statistic while `hcr` (the un-split total) looks
similar in both — the split, not the total, is the new information.

---

## 2. Information geometry: Fisher–Rao is the right denominator for PMF drift

**The gap.** The PMF instrument (`ratio_letter_v1`, `AnswerEvidence`) gives a
full distribution over a finite alphabet per call — the richest signal we
elicit — and the one standing cross-run comparison anyone has computed
(FIRST_PRINCIPLES §6: "one JSD data point... 0.128") is an ad hoc
Jensen–Shannon number with no stated metric properties and no error bar.

**The statement.** A PMF over a `K`-letter alphabet lives on the simplex
`Δ_{K-1}`. Its Fisher information metric is, under the classical square-root
embedding `θ_i = 2√p_i`, exactly the round metric on the positive orthant of
the radius-2 sphere in `R^K` (Kass 1989; Amari–Nagaoka). The induced
geodesic distance has closed form — the **Fisher–Rao / Bhattacharyya-arccos
distance**:

```
d_FR(p, q) = 2 · arccos( Σ_i √(p_i q_i) )
```

a genuine Riemannian metric (exact triangle inequality, not an
approximation). JSD is a different, non-Riemannian object: it's an
`f`-divergence, bounded in `[0, ln 2]`, and only its square root is a proven
metric (Endres–Schindelin 2003); JSD itself agrees with `(1/8)·d_FR²` only
to *second order* near `p = q` — every "nice" divergence shares the same
local (Fisher) geometry and differs away from it. So a JSD number reported
as "how far the PMF moved" is a fine LOCAL proxy but an unjustified GLOBAL
one, and it isn't the quantity whose triangle inequality lets you chain
claims ("moved this much under wording, that much under temperature, hence
at most this much under both").

This connects directly to §5: pooling multiple judges' evidence is choosing
a point on the same statistical manifold, and *which* divergence you
minimize to find that point is an α-divergence choice (Amari), not a free
parameter — see below.

**Why load-bearing.** Every open row in the invariance table that measures
PMF drift (temperature sweep, reasoning-effort sweep, time drift in §7) needs
a *denominated, composable* distance, per Principle #12 ("everything is
denominated") and Principle #3 ("no claim without its noise class"). Right
now the only tool on hand doesn't compose.

**Receipt.** `fisher_rao_distance(p: &DiscreteDistribution<T>, q: &..) -> f64`
in `discrete.rs`, computing the arccos-Bhattacharyya formula over the shared
support (fold refusal/escape mass into one extra "abstain" bin using the
existing `total_probability`/`renormalized_support` so `Complete` and
`Truncated` evidence stay comparable). Property tests: triangle inequality
holds on 1,000 random Dirichlet-sampled triples; `d_FR(p,p) = 0`; invariant
under any common relabeling of the alphabet (mirrors the reflection-algebra
discipline already in `seriate`); `d_FR` between two disjoint point masses is
exactly `π` (the sphere's geodesic diameter) — a clean closed-form check with
no simulation needed.

---

## 3. Elicitation as effect typing: the free structure is a commutative monoid of sufficient statistics

The operator's explicit ask. The type product space, grounded in what
actually exists (`seriate::record::InstrumentKind`, `AcquisitionMode`):

| Axis | Values | Where |
|---|---|---|
| arity | 1 · 2 · k · n | not a field anywhere; implicit in template choice |
| scale | nominal · ordinal · interval · ratio | implicit in `InstrumentKind` variant |
| output form | point · sample-set · PMF | implicit in `AcquisitionMode` |
| acquisition | `Logprob` · `Sampled` · `Fused` | `seriate::record::AcquisitionMode` (real, exists) |

`InstrumentKind` today is a flat enum naming *concrete cells*
(`OrdinalPairwise`, `RatioLetterPairwise`, ...), not the product type. That's
the shape a refactor should target:

```rust
struct InstrumentType { arity: Arity, scale: Scale, output: OutputForm }
enum Purity { Pure, SeededPure }
fn purity(mode: AcquisitionMode) -> Purity {
    match mode { Logprob => Pure, Sampled | Fused => SeededPure }
}
```

**Static cacheability, precisely.** A call is a pure function of
`(entity-set, attribute, instrument, model, decode-params)` — the exact
five-tuple `seriate`'s content-addressing already hashes — **iff**
`AcquisitionMode = Logprob`. Every `Sampled`/`Fused` call becomes equally
referentially transparent once the RNG seed is lifted into the input tuple
(`SeededPure`). Consequence worth stating because it's reassuring, not
because it's surprising: **there is no impure primitive in the current
design.** Nothing depends on wall-clock time or unaudited external state.
The type system's job is to make `SeededPure` the default requirement and
`Pure` (seed-free) a refinement that `Logprob` instruments earn for free —
exactly the ordering the code already respects, just not typed as such.

**Program equivalence — the real theorem, and its exact boundary.**
"When do two elicitation programs provably compute the same posterior" has a
sharp two-part answer, both readable straight out of
`solve_weighted_least_squares` / `solve_irls_huber`:

1. **Order-of-arrival invariance (always true).** The reduced Laplacian and
   RHS are literally `L = Σ_e w_e(e_i−e_j)(e_i−e_j)ᵀ`,
   `rhs = Σ_e w_e μ_e(e_i−e_j)` — a sum over a *multiset*. Every later stage
   (MAD, `max_abs_residual`, the Huber weights, the converged fixed point)
   is a function of the *set* of accumulated `(i,j,μ,λ_raw)` triples, never
   of the order calls were issued in. So: **two elicitation programs that
   accumulate the same multiset of raw observations produce byte-identical
   posteriors**, regardless of interleaving, batching, or which agent asked
   first. This is the free structure the operator asked for: the free
   *commutative monoid* on evidence contributions, with the accumulation-into-
   `(L, rhs)` map as the monoid homomorphism into the sufficient-statistic
   vector space. Composition of elicitation programs is exactly monoid
   concatenation; two programs are equivalent iff they hit the same
   coequalizer.
2. **Split/merge invariance (true only pre-Huber-clip).** Splitting one
   weight-`2w` observation into two weight-`w` observations with the same
   total `(μ, λ)` is invariant *only* in the pure-L2 regime — no residual
   anywhere exceeds `huber_k · scale`. Once Huber clipping is active,
   splitting changes which sub-observations get clipped (clipping is
   nonlinear in the per-edge weight), so the converged fit can differ. The
   monoid structure survives arrival-order permutation unconditionally; it
   does **not** survive re-partitioning the weight mass, in general. This is
   the self-refutation Principle #1 asks for: the naive "elicitation forms a
   free monoid, full stop" claim is false, and the true statement is the
   one with the Huber-threshold caveat attached.

**Receipt.** Two property tests in `tests/property_solver.rs`: (a) shuffle
ingestion order / split a single `engine.ingest(&obs)` call into several
interleaved batches with the same total observation multiset — assert
identical `scores`/`hcr`/`pcr` to `1e-9`; (b) construct a case with one
active outlier edge (residual past the Huber threshold at convergence),
split it into two half-weight edges with identical total `(μ,λ)`, and show
the solve *changes* — a deliberate, documented counterexample proving claim
2's boundary is real, not a hedge.

---

## 4. Spectral theory: Foster's theorem as a free correctness receipt, and the sharp identifiability threshold

**What's already there.** `effective_resistance_with_chol` computes
`R_eff(i,j)` via the reduced Cholesky; the planner's information gain is
literally `0.5·ln(1 + λ·R_eff)` — the Gaussian-channel capacity formula
(Shannon), i.e. the planner is already doing D-optimal experimental design
on a resistor network without naming it that. `Var(s_i − s_j) = R_eff(i,j)`
(the electrical-network fact, `rating_engine.rs:1286`) is also already used.
What's missing is a *headline* spectral number and the theorem that
connects "is m comparisons on n items enough" to graph structure alone.

**Foster's theorem, free.** For any connected weighted graph,
`Σ_{(i,j)∈E} w_ij · R_eff(i,j) = n − 1` (Foster 1949); the version with
`c` components generalizes to `= n − c`. This holds for *any* choice of edge
weights — a pure graph-topology conservation law, provable via
`trace(L⁺L) = rank(L)`. That means: after any solve, summing
`λ_eff(e) · R_eff(e)` over every edge must equal `kdim`
(`keep_idx.len()`), exactly, independent of what the actual λ values are.
This is a **zero-new-data** correctness check on existing quantities
(`diag_cov`, `chol`, `keep_idx`) — if it fails beyond floating-point slack,
the effective-resistance or reduced-Laplacian code has a bug, full stop.

**The sharp threshold.** Two theorems, not one, and they answer different
halves of "does a budget of m comparisons identify top-k":
1. **Connectivity is a hard prerequisite, sharp and classical.** For a
   uniformly random comparison graph `G(n,p)`, connectivity holds w.h.p. iff
   `p > ln(n)/n` (Erdős–Rényi 1959) — below that threshold, no amount of
   measurement precision recovers a global ranking, because the graph is
   disconnected and `compute_hcr`'s own `components` field already reports
   the failure mode. This lower-bounds any planner: `m* ~ (n ln n)/2` edges
   is necessary before identifiability is even *possible*, before noise is
   discussed at all.
2. **Given connectivity, the noise-conditional bound is already computable
   per-pair.** From `Var(s_i−s_j) = R_eff(i,j)`, pair `(i,j)` is correctly
   ordered at confidence `1−δ` once `Δ_ij² / R_eff(i,j) ≥ 2 z_{1−δ}²` — the
   exact Gaussian tail bound `normal_cdf`/`pair_probability` already compute
   pointwise. `expected_rank_reversals`, `max_pair_reversal_prob`, and
   `rank_risk` are this threshold theorem, already operationalized per-run.
   What's missing is collapsing it to ONE scalar: the **Fiedler value**
   (algebraic connectivity, `λ_2` of the reduced weighted Laplacian) as the
   single number saying "how well-conditioned is this graph for
   identification in general," independent of any specific pair — and it's
   nearly free to report, since `SymmetricEigen` is already computed on the
   small-graph fallback path.

**Receipt.** `foster_check(engine) -> f64` returning
`Σ_e λ_eff(e)·R_eff(e) − kdim` (should be `~0`), wired as an assertion inside
the existing `planted_recovery_tau_n8/16/32` tests in
`tests/property_solver.rs` rather than a new file — it's a free invariant on
data those tests already generate. Separately, add `fiedler_value: f64` to
`SolveSummary` (smallest nonzero eigenvalue of `l_red`), reported alongside
`hcr`/`pcr` as the standing per-run spectral health number.

---

## 5. Opinion pooling: log-pooling is forced, not chosen, by the ratio-scale symmetry

**The statement.** Two classical pooling rules exist for combining several
experts' distributions: the **linear pool** `p̄ = Σ w_i p_i` (minimizes
`Σ w_i KL(p_i ‖ p̄)`) and the **logarithmic pool**
`p̄ ∝ Π p_i^{w_i}` (minimizes `Σ w_i KL(p̄ ‖ p_i)`). For an exponential
family in natural parameters `η`, the log pool is *exactly*
`η̄ = Σ w_i η_i` — for a Gaussian, natural parameters are
`(μ/σ², −1/(2σ²))`, so log-pooling reduces to precision-weighted averaging
of means and summed precisions. **This is exactly what the engine already
does** (the reduced-Laplacian solve pools multiple raters'/templates'
log-ratio evidence by precision-weighted least squares; `gain_calibration`'s
bilinear fit is the same rule applied per-template). The engine picked log
pooling implicitly; here is why it had no other choice:

Our belief lives on the ratio scale in log-space, and the group that
"shouldn't matter" (per Principles §5 / the invariance table) includes the
reflection `x → 1/x`, i.e. `log x → −log x`. The **arithmetic** mean does
not commute with reciprocation (`mean(1/x_i) ≠ 1/mean(x_i)` in general); the
**geometric** mean does, exactly (`gmean(1/x_i) = 1/gmean(x_i)`). So linear
pooling of raw ratios is *not equivariant* under the very symmetry the
codebase already tests for via `AnswerEvidence::reflected()`; log/geometric
pooling is the unique linear-in-log-space rule that is. Separately, and for
an independent reason, log pooling is the unique pooling rule that is
**externally Bayesian** (Genest 1984): pool-then-condition-on-shared-new-
evidence equals condition-each-expert-then-pool. Linear pooling fails this.
Two independent arguments — symmetry and Bayesian consistency — land on the
same rule the code already uses. That's worth stating as a theorem, not an
implementation detail: **it rules out ever "trying" arithmetic pooling as an
alternative aggregation scheme without first breaking one of these two
properties.**

**Cooke-style calibration weighting, made concrete.** Cooke's classical
model weights each expert by (calibration score) × (informativeness),
calibration measured against seed/calibration questions with known answers.
The exact analogue already exists as a first-class parameter:
`RaterParams::beta`, currently set by hand. Our "calibration questions" are
free: null pairs (identical entity vs. itself — the correct answer is
ratio = 1, a hard zero) and any planted/known-order pairs (the ladder,
`ladder_curl.rs`'s scripted judges). **Fit `beta_r` from a proper scoring
rule evaluated only on those pairs**, rather than eyeballing it — this
replaces a hand-tuned knob with a measured one, using data the repo already
collects for a different purpose (calibration receipts).

**Receipt.** `fit_rater_beta(calibration_observations, known_truth) -> f64`:
score each rater's residual on null/known-order pairs via log score
(equivalently, the IRLS scale estimate already computed — `mad(&residuals)`
— restricted to the calibration subset instead of pooled across all pairs),
invert to a precision multiplier. Test: a synthetic rater with inflated
variance on calibration pairs gets a `beta_r < 1`; a rater who's *quieter*
than claimed on calibration pairs (already well-calibrated or
over-cautious) gets `beta_r ≈ 1` unchanged — the correction should only
ever discount, never manufacture confidence beyond what the raw calibration
data supports.

---

## 6. Exchangeability and repeat draws: the estimator, before it ships

**The gap.** `AcquisitionMode::Sampled`/`Fused` already exist in `seriate`;
repeat-sampling the same pair is future work per the docs. The naive plan
— pool `k` repeat draws into one edge with precision `k/σ_draw²` — is wrong,
and provably so given a finding this repo has *already shipped*: the Hodge
curl/frustration receipt proves that persistent, non-gradient-explainable
disagreement on specific pairs is real (§1). That means a pair's repeat
draws are not i.i.d. around the globally-fit score difference `s_i − s_j`;
there's a genuine pair-specific bias term.

**The right hierarchical model.** By de Finetti (finite version:
Diaconis–Freedman), if repeat draws of the same pair are judged exchangeable
(no information distinguishes draw order — the null hypothesis worth
stating explicitly, and testable against the martingale framing in §7),
they must be conditionally i.i.d. given a latent per-pair parameter. The
correct two-level model:

```
draw_t = (s_i − s_j) + b_{ij} + ε_t,   b_{ij} ~ N(0, σ_b²),  ε_t ~ N(0, σ_w²)
```

`σ_b²` (between-pair, persistent) is exactly the local-curl/harmonic energy
of §1 attributable to that specific pair — a real, already-partially-measured
quantity, not a free nuisance parameter invented for this section. Standard
random-effects meta-analysis (DerSimonian–Laird) gives the correct pooled
variance for `k` repeat draws of one pair: `σ_b² + σ_w²/k`, **not**
`σ_w²/k`. As `k → ∞` this asymptotes to `σ_b²`, not zero: **there is a hard
floor on how much repeat-sampling one pair can ever buy, set by the
frustration energy already measured in §1.** Deciding a repeat-sampling
budget without this floor will silently overspend on pairs whose
disagreement is structural, not noise.

**Receipt.** `variance_components(repeat_draws_by_pair) ->
(within_var, between_pair_var)`, a one-way random-effects ANOVA /
DerSimonian–Laird moment estimator. Test: simulate synthetic repeat draws
with known `σ_w², σ_b²`, verify the estimator recovers both to within
simulation noise, and verify that the naive `k/σ_w²` precision formula
diverges from `1/(σ_b² + σ_w²/k)` by an amount that grows with `k` — i.e.
show the naive estimator gets *more* wrong the more you resample, which is
the concrete, unignorable argument for building the right one before
repeat-sampling ships.

---

## 7. Wilder directions: one accepted, two rejected with reasons

**Accepted — martingale drift, because it fills an explicit `✗` row.**
FIRST_PRINCIPLES §5 lists "time (same judge, days apart)" as `✗`, unmeasured.
The right null hypothesis for "did this judge's belief drift" is that the
sequence of repeated readings over calendar time is a **martingale
difference sequence** around a constant: `E[m_t | F_{t-1}] = m^*`. That's a
sharper frame than "check for a linear trend" (which `spin_sweep`'s
machinery could cheaply be repointed at `days_since_first` for a first cut)
because provider-side model updates are realistically **step changes** on an
unknown date, not smooth drifts — a linear-slope test has poor power against
a step, and the mathematically appropriate test is a **CUSUM /
change-point** statistic on the partial sums of `(m_t − m^*)`, not a
regression slope. Receipt: `time_drift_probe`, structurally identical to
`SpinSweepReport` but keyed on elapsed days instead of field intensity,
reporting both the cheap linear-slope fit (reusing `spin.rs`'s existing
least-squares code almost verbatim) and a CUSUM statistic with its
change-point location — the two should usually disagree in exactly the way
that's diagnostic (slope ≈ 0, CUSUM large ⇒ step, not drift).

**Rejected — optimal transport between rankings/score distributions.**
Our score posteriors are already compared via rank-based metrics (`tau`,
`rank_risk`, `expected_rank_reversals`) and PMFs already live on a *shared,
known* discrete alphabet (§2). OT's actual selling point over KL/JSD-type
divergences is comparing distributions with **non-overlapping or mismatched
support** — moving mass across supports that don't coincide. That's not our
situation: entity sets and letter alphabets are shared across the
comparisons we actually run. Fisher–Rao (§2) already gives a true metric on
the shared simplex at lower conceptual and computational cost. Revisit only
if cross-model comparison ever needs to compare PMFs over *different*
alphabets (e.g. a 3-token ordinal instrument vs. a 52-letter ratio ladder)
where support genuinely doesn't align — not the case today.

**Rejected — tropical geometry of the ratio ladder.** The idea has a real
kernel: a discretized log scale is a tropicalization (a lattice
approximation) of the continuous multiplicative group, and the ladder's
quantization curl (already measured: floor 0.00198 vs. 0.00155 for a
constant-log-step ladder, per FIRST_PRINCIPLES §5½) is exactly "lattice
rounding error of a cocycle." But right now that's a relabeling of an
already-shipped, already-measured finding — it produces no new receipt,
just a fancier vocabulary for the same 0.00198. It earns its place only if
it produces a **closed-form predictive bound** on the curl floor from ladder
geometry *before* running an experiment (so a ladder could be sized
analytically instead of measured after the fact) — worth returning to, not
worth building this week.

**Minor note, not a section.** The conal/lattice structure of consistent
preference sets (the polytope of score vectors satisfying observed
inequalities) is real and clean, but it's the natural tool for
*ordinal-only* data with no continuous posterior — and the repo has mostly
moved past ordinal-only elicitation to the richer log-Gaussian ratio
posterior everywhere it matters. Worth a paragraph if `k-wise best–worst`
(the FIRST_PRINCIPLES "highest-value missing cell") ships and needs a
confidence region cheaper than the Gaussian ellipsoid; not worth a section
today.

---

## Top 3, buildable this week

1. **Hodge harmonic split** (§1). `compute_hodge_split` in
   `rating_engine.rs`, triangle-incidence matrix built from already-fused
   edges, one upper-Laplacian Cholesky solve reusing existing machinery.
   `tests/hodge_split.rs`: filled-triangle case (harmonic ≈ 0) vs. unfilled
   4-cycle case (harmonic ≈ hcr), same `cycle_dim`, opposite split — the
   sharpest, most novel, most "Book"-worthy result of the seven.
2. **Program-equivalence property tests** (§3). Two additions to
   `tests/property_solver.rs`: shuffled/batched-ingestion invariance
   (positive theorem, should pass); split-vs-merge of an active-Huber-clip
   edge (negative result, should visibly fail identical-output, proving the
   boundary is real). Zero new elicitation calls, pure refactor of existing
   test infrastructure plus one new adversarial case.
3. **Foster's theorem receipt + Fiedler value** (§4). `foster_check`
   wired into the existing `planted_recovery_tau_n*` tests (a free
   correctness invariant, no new data); `fiedler_value` added to
   `SolveSummary` next to `hcr`/`pcr` as the standing spectral-health number
   (already-computed `SymmetricEigen` on the small-graph path, negligible
   marginal cost).

Runner-up for next week: the DerSimonian–Laird variance-components
estimator (§6) — it's the one piece of math that must exist *before*
`Sampled`/`Fused` repeat-sampling ships, or the naive `k/σ²` pooling will
quietly overspend on structurally-frustrated pairs from day one.
