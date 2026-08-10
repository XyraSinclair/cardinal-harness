# Oracle probe #3 — exact-ground-truth estimator shootout (2026-08-10 afternoon)

`groundtruth_shootout.py` computes the EXACT grammar-masked pushforward of
google/gemma-4-E2B-it (full logits, MPS, 36s) over the ratio instrument
`{"higher_ranked": A|B, "ratio": [0-9]{1,3}.[0-9]}` — masked+renormalized
conditionals at every stochastic position, matching the measured API semantics
(RESULTS.md census finding 1). gemma-4 tokenizes digits singly, so the trie is
natively digit-level: 2,443 node forwards, 22,200 leaves, leaf mass total
1 − 8.8e−8. Truth: `groundtruth_tree.json`.

The truth is a realistic mid-entropy judge: TRUE E[h] = 1.1155
(h = signed log10 ratio clamped to [1.0, 999.9]), direction P(B) ≈ 1,
50% of mass in 4 leaves (B:100.0 at 0.195, B:15.0 at 0.163, B:12.5 at 0.112,
B:3.5 at 0.070), 90% in 33 leaves — neither the peaked-easy nor flat-hard
extreme from HARVEST.md.

Against this known truth we emulate the constrained access tiers offline
(top5 / top20 / chosen-token-only / sample-only; a "call" = one temp-1 draw
returning chosen tokens with exact conditionals + top-k sidebands per node,
exactly the census-verified API surface) and race five estimators of E[h] at
equal call budgets, R = 300 replications each: `shootout_results.json`.

## Results (bias / RMSE vs exact truth; hv1 also envelope coverage + median width)

| tier | n | mc | hv1 mid (cover, width) | head_same | head_split | atom_ht |
|---|---|---|---|---|---|---|
| top5 | 5 | .00/.256 | +.27/.282 (1.00, .91) | −.60/.630 | .00/.342 | .00/.285 |
| top5 | 25 | −.01/.118 | +.07/.085 (1.00, .30) | −.22/.220 | −.01/.098 | −.01/.111 |
| top5 | 100 | .00/.059 | +.006/**.009** (1.00, .08) | −.10/.103 | .00/.029 | +.01/.048 |
| top20 | 100 | −.01/.060 | +.02/.024 (1.00, .05) | −.09/.092 | .00/.029 | .00/.047 |
| chosen | 100 | .00/.057 | .000/**.012** (1.00, .13) | −.17/.172 | .00/.045 | .00/.051 |
| sample | 100 | .00/.059 | — | — | — | — |

(Full 15-cell × 5-estimator grid in `shootout_results.json`.)

## Findings

1. **The credal envelope is sound: coverage 1.00 in all 15 cells, 4,500
   replications.** The hv1 envelope never once excluded truth. Anytime
   soundness — the property the whole design leans on — is now measured, not
   argued.

2. **Discover-then-subtract selection bias is real and catastrophic — Oracle
   probe #3 answered.** `head_same` (same draws pick the exact-mass head AND
   estimate the residual) degenerates: every drawn leaf enters the head, so
   the residual sample is empty by construction and the estimator zero-fills
   the undiscovered mass → bias −0.60 at n=5, still −0.10 at n=100. Cross-fit
   (`head_split`) kills it dead: |bias| ≤ 0.01 in every cell. The Oracle's
   warning was correct and is now quantified.

3. **Exact-mass harvesting beats frequency counting by ~47x calls in its
   regime.** At n=100/top5 the hv1 ledger hits RMSE 0.0086; plain MC has
   σ ≈ 0.59, so matching that needs ≈ 4,700 samples. Once enumerated mass is
   high the ledger is nearly deterministic — atoms carry exact masses, and
   sampling noise only enters through which cells remain.

4. **Regime doctrine, now measured** (this replaces intuition in DESIGN.md's
   L-ladder): plain MC is unbeatable at n ≤ 10 (hv1 mid is minimax-biased
   +0.27 there, though always covered by its honest 0.9-wide envelope);
   cross-fit head+residual takes over at n ≈ 25–50 (2x MC); hv1 mid wins
   outright at high enum mass (n ≈ 100 here). Deployable rule: always report
   the hv1 envelope; use hv1 mid as the point estimate when residual mass
   < ~0.1, else cross-fit.

5. **Chosen-token-only access is nearly as good as top-5.** hv1 at
   chosen/n=100: RMSE 0.012 vs top5's 0.009 (envelope 0.13 vs 0.08 — the
   sidebands' real contribution is faster cell resolution, not accuracy).
   Resampling-is-peeling confirmed from the truth side: drawing IS discovery,
   because each draw's exact chosen-token conditionals mint atoms. Providers
   exposing only chosen-token logprobs still support the full kernel.

6. **atom-HT (π_y = 1−(1−p_y)^N) is honest but dominated**: unbiased
   everywhere (|bias| ≤ 0.02) and better than MC from n ≈ 25, but cross-fit
   beats it at n ≥ 50 and hv1 crushes both at n = 100. Its niche: unbiased
   point estimates with no ledger machinery, chosen-only access.

7. Curiosity, one sentence: top20's hv1 mid is slightly WORSE than top5's at
   n=100 (0.024 vs 0.009) — midpoint-imputation errors across many small
   resolved-tail cells cancel differently; the envelope (which top20 halves)
   is the honest comparison, and it behaves monotonically.

## Caveats

- No provider jitter in the simulator (deliberate: measures structural bias,
  not noise; HARVEST.md finding 3 characterizes the jitter separately, and
  the bootstrap band in harvest.py carries it).
- One truth distribution (one prompt/pair/model). The regime boundaries
  (n≈25, residual<0.1) are indicative, not universal constants; the ordering
  and the coverage result are structural.
- h clamps sub-1.0 ratios to Z=0 (grammar-legal but instrument-incoherent
  region; same convention as harvest.py's domain).

## Consequences for the kernel

- harvest.py v1 ships the right core: exact-atom ledger + credal envelope is
  both sound (finding 1) and, in its regime, the most call-efficient point
  estimator we tested (finding 3).
- The one upgrade worth making before the Rust seam: add cross-fit
  head+residual as the point estimate when enumerated mass is low
  (finding 4), and NEVER same-batch subtract (finding 2).
- Chosen-only providers are in-scope for the sellable kernel (finding 5).

Reproduce: `/tmp/dpmf-venv/bin/python groundtruth_shootout.py enumerate`
(needs torch-MPS + transformers 5.x, ~40s), then `... shootout` (~2s).
