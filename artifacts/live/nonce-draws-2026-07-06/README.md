# Nonce draws, live: cache-priced repeat elicitation (2026-07-06)

`cardinal judge A B --by X --draws k`: the same judgement drawn k times,
varying only a suffix nonce — the long prefix (prelude + attribute +
both entities, ~1300 tokens here) stays byte-identical (pinned in tests)
so provider prompt caching bills it at the cached rate. At temperature 0
the spread over nonces is the model's pure CONTEXT-SENSITIVITY noise —
the within-pair σ_w the DerSimonian–Laird floor consumes, measured with
the actual contaminant (irrelevant context) rather than sampling
temperature. Pair: long-form obstacle-is-the-way vs measured-gets-managed
passages, "depth of insight about living well". n = 1 pair per model.

| run | model | σ_w (nats) | sign flips | cached / input tokens | cost (8 draws) |
|---|---|---|---|---|---|
| 1 | gpt-5.4-mini | 0.114 | 1/8 | 0 / 10,483 | $0.0088 |
| 2 | gpt-5.4-mini | 0.100 | 0/8 | 0 / 10,483 | $0.0088 |
| 3 | deepseek-v4-flash | 0.537 | 1/8 | **9,984 / 10,507 (95%)** | **$0.0010** |

## Findings

1. **Context-sensitivity is real, large, and model-dependent.** A single
   irrelevant suffix token moves mini's judgement by σ_w ≈ 0.11 nats
   (one draw flips sign) and deepseek's by 0.54 — on the same pair,
   deepseek is ~5× more nonce-sensitive. This discounts deepseek's
   portfolio ΔI/$ exactly the way the DL floor prescribes: a cheap judge
   whose draws scatter needs more of them, and σ_b (structural) vs σ_w
   (contextual) are now separately measurable.
2. **The cache economics work as designed where the router confirms
   them**: deepseek billed 95% of input as cached — 8 judgements over a
   1300-token prompt for $0.0010. OpenRouter did not report cache reads
   for gpt-5.4-mini (0/10,483, cost flat across warm reruns): the
   receipt surface is ready, the router isn't confirming for that
   provider — recorded as a provider fact, not assumed away.
3. **Between-run mean shift** (mini: 0.071 → 0.198 across two runs,
   comparable to σ_w) is the two-level DL structure appearing unprompted:
   within-run nonce noise + between-run drift. Repeat instruments must
   model both; ours now measures both.
4. Truncation caught live: the first deepseek run lost 4/8 draws to a
   96-token output cap (looked like refusals); raised to 256 → 8/8.
   A parse failure that presents as a refusal is a receipts bug — fixed
   same run.
