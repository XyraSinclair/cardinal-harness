# Stochastic transitivity, live (2026-07-08)

`stochastic_transitivity` over nonce-draw choice probabilities: the
WST/MST/SST hierarchy on four contested corpus items, deepseek-v4-flash,
12 draws/pair (refusals reduced some pairs to 4 usable — the honest
min_draws denominator), $0.0135.

Result: 4 testable triads, **zero WST violations** (no cyclic
majorities), one raw SST flag at **0.2 SE depth** — which the binomial
margin machinery correctly declines to call (0 violations beyond 2 SE).
The raw/margin split doing its job on first live contact: flag
everything, certify only what noise cannot explain.

Instrument provenance (two corrections forced by its own tests):
1. The "unorientable triad" branch was dead code — every 3-tournament
   has a Hamiltonian path, so an orientation always exists and a WST
   violation is EXACTLY a cyclic majority. The equivalence is now the
   definition.
2. The margin test initially asserted a 0.25-deep violation at k=20 was
   "far beyond noise" — it is 1.49 combined SE across three binomial
   estimates, and the instrument refused. The test lost the argument.

The keystone pin (tests/transitivity.rs): a judge whose mean log-ratios
telescope EXACTLY (hcr < 1e-9 — invisible to every Hodge receipt) while
its choice probabilities violate SST — the pathology class this
instrument alone can see, closing the WST/MST/SST gap in the invariance
program.
