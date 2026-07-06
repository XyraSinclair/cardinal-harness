# Live orbit transforms (2026-07-05)

`cardinal judge --orbit`: the judgment measured under the full Z₂³
elicitation group (order swap × polarity negation × wording inversion,
8 comparisons), pulled back through the known equivariances, decomposed
by characters. Belief = the invariant coefficient; every bias a named
orthogonal coefficient; Parseval exact (residual 0 in all three runs).
Pair: "The obstacle is the way." vs "Live, laugh, love.", criterion
"depth of insight about living well". n = 1 pair per model — instrument
demonstrations, not model properties.

| coefficient | sonnet-4.6 | gemini-2.5-flash | gpt-5.4-mini |
|---|---|---|---|
| belief (nats) | +1.162 (98.7%) | +1.365 (92.6%) | +0.852 (**53.1%**) |
| order | −0.027 | +0.217 | +0.370 |
| polarity | −0.027 | +0.091 | +0.098 |
| **order·polarity** | −0.084 | −0.034 | **−0.552 (22.3%)** |
| wording | −0.031 | +0.145 | +0.370 |
| order·wording | +0.027 | −0.087 | +0.083 |
| polarity·wording | +0.027 | −0.105 | −0.189 |
| triple | +0.084 | −0.230 | +0.098 |
| **coherence** | **0.987** | 0.926 | **0.531** |

## Findings (stated as measured functional properties)

1. **gpt-5.4-mini's largest bias component is an interaction**: the
   order·polarity coefficient (−0.552 nats, 22.3% of judgment energy) —
   its slot preference reverses sign under criterion negation. This
   coefficient is invisible to counterbalancing, to two-sided probes,
   and to every one-axis-at-a-time bias measurement: it exists only in
   the joint decomposition. Marginal probes would report mini's order
   bias as +0.370 and miss that a component half again larger lives in
   the coupling.
2. **claude-sonnet-4.6 is close to G-invariant on this pair**: 98.7% of
   its judgment energy is in the trivial character; no bias coefficient
   exceeds 0.084 nats.
3. Full spectrum cost: $0.0015–0.013 per pair (8 calls). The transform
   subsumes, in one instrument, what counterbalance + two-sided +
   wording probes measure separately — and adds the four interaction
   coefficients they cannot.

Algebra receipts pinned in tests (`tests/judge_explain_cli.rs`): a
perfect judge concentrates all energy in the trivial character
(coefficient exactly ln 3.9); the two content-blind position
pathologies — always-favor-slot-A vs always-name-token-A,
indistinguishable to counterbalancing — land in order·polarity and the
triple character respectively, exactly ln 2 each, all other coefficients
vanishing. The first derivation conflated them; the test separated them.
