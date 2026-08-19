# harvest.py — first full run of the descend/resample kernel (2026-08-10 morning)

`harvest.py` is the working prototype of the reconciled design: joint stochastic trie
(direction × int-token × frac-token) under a byte-identical grammar, exact-atom ledger
from chosen-token logprobs, top-k sidebands per visited node, credal envelope on
Z = log10(B/A) over the declared domain r ∈ [1.0, 999.9], anytime per-call trajectory,
and a three-layer uncertainty report:

1. **credal envelope** (`width`): adversarial placement of unresolved cell mass +
   truncation residual + conservation gap within grammar-bounded Z-ranges;
2. **conservation gap** (`gap`): mass the drift-averaged ledger cannot attribute
   (node masses need not sum to 1 under provider jitter) — widened into the
   envelope as full-domain slack, never silently dropped;
3. **provider-noise band** (`noise±`): bootstrap over per-token mass observations
   (resample each token's observed masses, rebuild ledger, recompute E[Z]).

Grid: 2 pairs × 3 models, sequential prompt-cached draws, ~$0.15, ~80s wall total.

## Results (fixed grammar `^[0-9]{1,3}\.[0-9]$`, temp 1, effort none on 5.x)

| pair | model | draws | enum mass | gap | envelope width | E[Z] mid | noise± | median jitter |
|---|---|---|---|---|---|---|---|---|
| egg vs bowling ball | gpt-5.4-mini (top5) | 40 | 0.490 | 0.000 | 1.085 | 1.713 | 0.010 | ~0 |
| egg vs bowling ball | gpt-4.1-mini (top20) | 40 | 0.765 | 0.000 | 0.071 | 2.082 | 0.047 | 0.009 |
| egg vs bowling ball | gpt-5.6-sol (top5) | 25 | 0.952 | 0.029 | 0.173 | 2.163 | 0.321 | ~0 |
| cat vs raccoon | gpt-5.4-mini | 40 | 1.041 | 0.042 | 0.261 | 0.316 | 0.086 | 0.069 |
| cat vs raccoon | gpt-4.1-mini | 40 | 1.016 | 0.017 | 0.102 | 0.540 | 0.065 | 0.169 |
| cat vs raccoon | gpt-5.6-sol | 25 | 0.988 | 0.000 | 0.003 | 0.355 | 0.011 | 0.168 |

Reference: a 14lb bowling ball / 53g egg ≈ 120x → Z ≈ 2.08. Both capable models land
exactly there (2.08, 2.16) with tight envelopes. Adult raccoon/house-cat ≈ 1.6x →
Z ≈ 0.20; all three models overestimate (2.1x/3.5x/2.3x) with unanimous direction —
a real, now-MEASURABLE inter-model disagreement, not sampling noise (the noise bands
don't overlap 0.20 for 4.1-mini).

## Findings

1. **The kernel works end-to-end.** 25–40 prompt-cached draws produce a calibrated
   joint PMF with honest uncertainty for ~$0.01–0.03/pair-model. On peaked models one
   sitting closes the envelope to ~0.003–0.1 log10 units.
2. **Flat models + top-5 is the documented hard case** (5.4-mini egg: enum 0.49 after
   40 draws, width 1.09). The envelope says so honestly instead of pretending. Its
   *point* estimate (1.71) is still usable — and its flatness is itself signal that
   this judge carries little information for this pair.
3. **Provider noise is continuous logit jitter, not backend bimodality** (12-call
   probe: direction P(B)=1.0000 twelve times; int-node '3' ∈ [0.772, 0.804], '4' ∈
   [0.085, 0.152]). Absolute jitter ~1–2% per heavy token. The measurement model
   "mass = mean of observations ± spread, bootstrap into an E[Z] band" is adequate;
   nothing exact should be claimed below jitter scale — matching the Oracle's
   `oracle_uncertainty` output field and DESIGN.md's provenance doctrine.
4. **Direction certainty is essentially free** — the direction node is the stablest
   measurement in the instrument (useful: the IRLS layer can consume sign confidence
   separately from magnitude).
5. **Cross-run reproducibility**: repeated cell runs shift E[Z] mid within the
   reported noise bands (e.g. sol egg 2.092 → 2.163 across two 25-draw runs while
   its band is ±0.32). The bands are doing their job.

## v1 limitations (deliberate, documented)

- No prefill → allocation across subtrees is proportional-to-mass, not Neyman;
  minority-direction cells resolve slowly (immaterial for decisive pairs).
- Midpoint imputation for unresolved cells is minimax, not a posterior mean.
- Discovery and estimation share draws; because all discovered atom masses are exact
  measurements (not empirical frequencies), the ledger is unbiased, but a residual-
  conditional *statistical* estimator (Oracle's exact-head + stratified rejection)
  would tighten flat-model cells beyond the credal envelope — that is the next
  estimator upgrade, gated on Oracle probe #3 (ground-truth shootout).

## Next steps (in value order)

1. ~~Rust seam~~ **SHIPPED 2026-08-11** — `src/rerank/decimal_ledger.rs` +
   slug `decimal_ledger_v1` (K=8 temp-1 redraws → exact-atom ledger →
   EvidenceMoments → IRLS precision via the existing evidence seam; cross-fit
   below 0.9 enum mass, never same-batch subtract). Live-smoked on
   gpt-4.1-mini ($0.0008/judgement, 12/12 logprob-mode); four review passes
   (port-fidelity, integration, mom, dad) reconciled in the two follow-up
   commits after the initial one. Ground-truth property battery over the
   production Rust code path: `RUST-BATTERY.md` (envelope coverage 1.000
   across 4,000 outcomes incl. 2% provider jitter, exact antisymmetry,
   conservation detector verified, extract discipline verified).
2. ~~Oracle probe #3: exact-ground-truth estimator shootout~~ **SHIPPED** — see
   `SHOOTOUT.md`: envelope coverage 1.00 across 4,500 replications, ~47x call
   efficiency over MC in the high-enum regime, selection bias measured (−0.60 →
   ≤0.01 via cross-fit), chosen-token-only access confirmed viable.
3. Prefill census (unlocks Neyman allocation + deep grammars).
4. Radix-4 signed-log instrument A/B against free-form (instrument-shift experiment).
