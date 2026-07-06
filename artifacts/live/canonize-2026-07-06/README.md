# Canonize, live: three minds converge on a wording (2026-07-06)

`cardinal canonize`: the seed attribute plus three LLM-proposed
refinements, each measured over the 8-aphorism corpus by THREE judge
models (gemini-2.5-flash, claude-haiku-4.5, gpt-5.4-mini), ranked by
transmissibility — mean cross-judge Spearman of the induced cardinal
latents. 384 comparisons, $0.27. n = 8 entities, 1 run: instrument
demonstration.

| transmissibility | signal (nats) | wording |
|---|---|---|
| **0.857** | 0.491 | illuminates principles for a flourishing life |
| 0.849 | **0.832** | depth of insight about living well (seed) |
| 0.778 | 0.591 | reveals profound truths about human well-being |
| 0.738 | 0.550 | offers actionable wisdom for ethical living |

## Reading

1. **The protocol discriminates the way it should.** The wording that
   drifted dimension ("actionable wisdom for ETHICAL living" — 
   actionability is not depth) pays for the drift in cross-mind
   agreement: three different judges recover each other's ordering least
   under it. Dimension drift is measured as transmission loss.
2. **Transmissibility alone is not the whole criterion.** The top two
   wordings tie within noise (0.857 vs 0.849) while the seed carries
   1.7× the signal — three minds agree on it AND it separates the
   entities more. A composite (transmissibility × signal, or a Pareto
   view) is the right v2 ranking; v1 ranks by transmissibility and
   prints both, letting the operator see the trade.
3. All four wordings transmit at ρ ≥ 0.74 across three different labs'
   models on this corpus: "depth of insight" is already close to a
   canonical dimension here — the protocol's job on such attributes is
   wording optimization at the margin, and its job on murkier attributes
   (where candidate transmissibility will spread widely) is triage.

The communication framing this operationalizes: an attribute prompt is a
protocol between minds. Its quality as a protocol = do different minds,
given only the wording and the entities, recover the same cardinal
latent. That is now a number with receipts (per-judge latent vectors in
report.json), not a vibe.
