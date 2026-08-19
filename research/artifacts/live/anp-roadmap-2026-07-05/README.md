# ANP, live, on our own roadmap (2026-07-05)

`cardinal anp`: goal + 4 model-proposed criteria + 5 roadmap alternatives,
every supermatrix edge a solved pairwise measurement, judged by
google/gemini-2.5-flash (selected because it currently tops the Judge
Coherence Benchmark — judge choice justified by measured coherence, not
habit). α = 0.4, 184 comparisons, $0.023, Cesàro limit converged.

Goal: "make cardinal-harness the definitive, trusted infrastructure for
measuring and using LLM judgment."

## Criteria: direct (hierarchy) vs limiting (network)

| criterion | direct | limiting | Δ (network correction) |
|---|---|---|---|
| Accuracy of judgment | 0.462 | 0.264 | **−0.198** |
| Ease of integration | 0.191 | 0.260 | +0.069 |
| Scalability of infrastructure | 0.173 | 0.262 | +0.089 |
| Community adoption | 0.174 | 0.214 | +0.040 |

The mathematical content of the correction: the hierarchy concentrates
0.46 of the mass on accuracy; the measured inner-dependence and
alternative-feedback edges redistribute it until the criteria cluster
sits near its entropy maximum (0.26/0.26/0.26/0.21). Interdependence is
strong enough that in the limit no criterion retains a dominant share —
a property of the measured edge structure, checkable from the
supermatrix in `report.json`.

## Alternatives: limiting priorities

| priority | alternative |
|---|---|
| 0.231 | best-worst scaling instrument (+ IIA receipts) |
| 0.223 | longitudinal drift tracking |
| 0.208 | benchmark corpus scale-up |
| 0.179 | temperature / reasoning-effort sweeps |
| 0.159 | wording-gain calibration in the production solver |

Spread is modest (0.16–0.23): under this goal and this judge, the
roadmap items are near-substitutes, with best-worst scaling first. The
full supermatrix, per-criterion z-scores, and iteration counts are in
`report.json`; alternatives were provided (not proposed) and criteria
were proposed by the judge (`--propose 4`).

n = 1 judge, 1 run: a demonstration of the pipeline producing
column-stochastic, convergent, fully-receipted network priorities — not
a settled prioritization.
