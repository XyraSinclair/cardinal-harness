# Rust kernel ground-truth property battery (2026-08-11)

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
