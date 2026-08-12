# Rust kernel ground-truth property battery (2026-08-11)

> **Errata / evolution**: numbers in the first two sections are the
> PRE-review-round measurements (kernel as first ported). The
> "Adversarial review round" section below records the three kernel
> changes that came out of the coherence + falsifier reviews and the
> post-fix numbers (±2σ and matrix coverage now 1.000 everywhere,
> recovery RMSE roughly halved). The historical numbers are kept because
> they document the mechanisms the fixes address.

The Python prototype was validated against exact ground truth (SHOOTOUT.md).
The shipped Rust port (`src/rerank/decimal_ledger.rs`) had four adversarial
reviews and two live smokes — but "reviewed" is not "measured". This battery
closes that gap: it runs the **production Rust code path**
(`extract_trajectory` / `analyze`) against the same exact gemma-4-E2B
pushforward, via `examples/decimal_ledger_groundtruth.rs`.

Method: load `groundtruth_tree.json` (22,200 exact leaves, true
E[Z] = −2.568622 presented-ln), marginalize the digit-level PMF into the
o200k-shaped three-node instrument (direction, integer token, fraction
digit), emulate the census-verified access surface (temp-1 draws, exact
chosen-token probability + top-5 sidebands per node, optional ±2%
multiplicative provider jitter), feed the draws to `analyze`. 400
replications per (K, access) cell; deterministic xorshift seeds.

Run: `cargo run --release --example decimal_ledger_groundtruth`
(asserts P1/P3/P4/P6/P7/P8/P9 — a violation aborts the run).

## Results (all asserted properties held, first run, commit-pinned)

| property | exact access | 2% jitter | denominator |
|---|---|---|---|
| P1 envelope covers true E[Z] | **1.000** at every K | **1.000** at every K | 400 reps × K∈{2,4,8,16,32} × 2 = 4,000 outcomes |
| P2 mean±2σ covers truth | 0.92 @K=8, 0.97–0.98 @K≥16 | 0.95 @K=8, 0.98–0.99 @K≥16 | 400/cell |
| P3 antisymmetry (mirrored presentation) | \|mean+mean′\| = 0.0 exactly, \|var−var′\| ≤ 1e−15 | — | 100 mirrored reps |
| P4 conservation | gap ≤ 4.4e−16 (pure fp) | gap ≈ 1e−3 (∝ injected jitter — the detector works) | max over 2,000/mode |
| P5 production K=8 point | bias +0.043, RMSE 0.570 ln (0.25 log10) | bias +0.032, RMSE 0.551 | 400/cell |
| P6 median envelope width monotone in K | 2.42 → 0.98 | 2.45 → 1.04 | 5 K levels |
| P7 determinism / envelope order-invariance | identical outcomes; envelope order-invariant | — | — |
| P8 p_dir_a vs truth marginal (8e−8) | within 0.05 | — | max over all reps |
| P9 extract_trajectory discipline | JSON stream accepted; prose ` 12`, split `1`,`2`, two-digit frac `53` all rejected | — | 4 constructed streams |

## Honest characteristics (not failures)

1. **±2σ undercovers at K≤4** (0.57–0.81): with 2–4 observations the jitter
   bootstrap can't see spread and width²/12 treats unresolved cell mass as
   uniform. The credal envelope still covers at 1.000 — the envelope is the
   sound object; σ is the working-precision summary. At the production K=8
   it reads 0.92–0.95 against a ~0.95 nominal, and IRLS's robust reweighting
   plus EVIDENCE_VAR_FLOOR absorb this class of mild overconfidence.
2. **Cross-fit point is draw-order sensitive** (Δmean ~0.16 on a reversed
   K=8 batch) — inherent to the half-split; halves are exchangeable so it is
   not a bias, and the envelope/certificate are exactly order-invariant.
3. **This pair at K=8 lands in the cross-fit regime** (max enum mass 0.73 <
   0.9), so the battery genuinely exercises the below-threshold path — the
   same regime the live gpt-4.1-mini smokes showed (visible 0.59).
4. Small positive (toward-zero) bias at low K from minimax midpoint
   imputation of unresolved cells; −0.003 by K=32.

Together with SHOOTOUT.md (Python estimator theory), the four review passes
(port fidelity), and the live smokes (wire integration), this completes the
validation chain: theory → port → wire, each layer measured, not assumed.

## Matrix-level gauntlet (P10–P16, added 2026-08-11 evening)

The single-pair battery validates the instrument in isolation. The PMF-shaped
outcome only earns its keep if it *composes*: a matrix of per-pair credal
ledgers must fuse — through the real sign algebra (multi.rs presented-
coordinate flip) and the real solver (`Observation::from_log_ratio_moments`
→ IRLS) — into a coherent cardinal scale. Same example binary, second phase.

Setup: 8 items on a latent nats scale spanning 1.4x–200x gaps; each pair's
judge is an **exact quantizer PMF** (latent signed log10 ratio ~
Normal(gap, σ=0.25), pushed through the decimal grid — quantization bias of
the grid itself measured at 0.0004 nats, negligible); 2 counterbalanced
observations per pair, K=8 draws each, through `analyze` → `RatingEngine`.

| property | result | denominator |
|---|---|---|
| P10 end-to-end recovery | Kendall tau **1.000** every rep; gap RMSE 0.216 nats, bias +0.094 (toward zero) | 40 reps × 28 pairs |
| P16 calibration decomposition | kernel-level \|z\|≤2 coverage **0.932**; matrix-level 0.821 | 1,120 pair-reps |
| P16b anytime (K=32) | bias +0.030, RMSE 0.087, tau 1.000 — shrinkage converges out with draws | 10 reps |
| P11 matrix presentation invariance | max \|Δgap\| = 6e−15 when EVERY pair is presented mirrored (sign algebra exact end-to-end) | 28 pairs |
| P12 triangle composition of raw ledger means | max \|m_ij+m_jk−m_ik\|/σ = **1.40** — raw means compose additively within uncertainty | 56 triangles |
| P13 precision flow | flat judge (σ=0.8) carries 2.1× ledger var; adding it *improves* recovery (0.202→0.192 RMSE) — precision weighting works, more evidence never poisons | 28 pairs, 10 reps |
| P15 round-number attractor | 50% of fraction mass snapped to .0: tau stays **1.000**, bias +0.27 reported honestly — the instrument faithfully measures the pathological judge's actual PMF | 10 reps |

**The one deep finding — fusion undercoverage (0.93 kernel → 0.82 matrix)**:
the kernel's small toward-zero magnitude shrinkage at K=8 is *correlated*
across the two counterbalanced observations (mirror symmetry preserves it),
so fusing them halves the variance without shrinking the bias — z inflates
by ~√2. This is a structural property of any moment-collapse fusion, not a
bug in the ledger: kernel calibration is 0.93–0.94 at every K, and the bias
term decays with draws (+0.094 @K=8 → +0.030 @K=32). The engine's
temperature/beta calibration layer (`CalibrationEvidence`) is the designed
absorber for exactly this residual-scatter-vs-claimed-precision mismatch.
Asserted floor: kernel ≥ 0.85, matrix ≥ 0.75, and K=32 must not degrade
either calibration or RMSE. **Resolved same night** — see below: folding the
estimator-disagreement term into var lifted kernel AND matrix coverage to
1.000 and halved recovery RMSE.

## Adversarial review round (2026-08-11 late): coherence + falsifier

Two independent agents ran after the matrix gauntlet: a deep-coherence
review (are the PMF structures mathematically consistent credal objects;
is the moment collapse the right matrix interface) and an empirical
falsifier (attack the kernel with concrete inputs in an isolated worktree).

**Coherence verdicts**: the trie/ledger is a coherent "conserved measure +
slack" credal object (the conservation gap doubles as a jitter detector);
the collapse to (E[Z], var) is the **correct** matrix interface — interval
matrices hit the dependency problem, separated direction×magnitude cannot
enter a linear-Gaussian likelihood — keep it; the sign algebra is trap-free
end-to-end (presentation is baked into the cache key, so cross-presentation
replay is structurally impossible). Sub-additivity is approximate, not
strict (top-k selection censoring can oversubscribe node masses; the gap
channel catches aggregate oversubscription) — measured negligible at 1–2%
jitter, documented rather than "fixed".

**Kernel changes landed from the round** (each verified by full battery
re-run):

1. **Estimator-disagreement variance** (coherence F6): var gains
   `(e_mid − crossfit)²` whenever both estimators are computable — exactly
   the estimator-choice variance the 0.9 threshold switch left unreported.
   Effect: ±2σ coverage 1.000 at every K and access mode (was 0.57–0.98);
   matrix-level coverage 0.82 → **1.000**; recovery RMSE 0.216 → **0.114**
   nats; K=32 RMSE 0.087 → 0.048. The var is now deliberately conservative
   (coverage above nominal) — right side to err on for a credal instrument.
2. **Sub-1.0 convention unified** (coherence F3): integer-token-"0" atoms
   and cells now contribute exactly z = 0 (the validated clamp convention)
   instead of paying full ±zmax envelope slack for a value the convention
   itself makes exact.
3. **Crossfit clamped into its own envelope** (falsifier BUG-2): the
   cross-fit residual mean could exit [e_lo, e_hi] under tail-heavy
   est-half draws (constructed case: mean 3.95 vs envelope [−0.68, 1.29]),
   handing IRLS a certificate-incompatible observation. The certificate is
   the sound object; the point is now clamped into it.
4. **Panic on untrusted output fixed** (falsifier BUG-1, HIGH):
   `parse_decimal_ledger_text` sliced `content[start..=end]` with a `}`
   preceding the first `{` — reachable panic on live model prose. Now
   Unparseable.
5. **Probability sanitation at intake** (falsifier BUG-3): non-finite
   chosen-token probabilities reject the draw; all masses clamp to [0, 1]
   (providers do emit logprob > 0 in the wild); non-finite sideband
   alternatives are dropped.
6. **Direction anchor tightened** (falsifier FRAGILE-1): dir binds only
   after the full key substring `higher_ranked` (concatenation-robust to
   any tokenizer split), so a "reason" field containing prose like "B is
   higher than A" can no longer mint a sign-flipped direction atom.

**Known non-fixes** (documented, deliberate): key-order-swapped JSON
(`ratio` before `higher_ranked`) degrades to frequency-MC rather than
parsing (prompt mandates order; degradation not corruption);
`enumerated_mass` can exceed 1 under direction-contradictory certainty
(the gap channel widens the envelope to full domain — conservative).

Coherence F5 — persisting `e_lo`/`e_hi`/`conservation_gap` through
`EvidenceMoments` + the sqlite cache for post-solve interval audits —
LANDED 2026-08-12 (commit 2a35cab): the certificate now survives the
(mean, var) collapse and round-trips through cache replay; ratio-letter
and MC-fallback evidence carries `None` (no certificate to fake).

Post-fix battery: single-pair P1–P9 and matrix P10–P16 all held; final
numbers: P10 tau 1.000, bias +0.002, RMSE 0.114; P12 max cycle 0.65σ;
P13 var ratio 2.4, RMSE 0.107 vs 0.114 (within tolerance); P15 tau 1.000,
bias +0.115; full test suite + clippy clean.
