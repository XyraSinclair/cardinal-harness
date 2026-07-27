# P2 second-judge gate — results (2026-07-27)

Verdicts applied mechanically from the rules in `REGISTRATION.md`
(committed `e304a5d`, before either run). Evidence:
`secondjudge-kimi-explain.json`, `polishcontrol-deepseek-explain.json`.

## Run A — kimi-k2.6 direction replication: **REPLICATED**

| attribute | deepseek w (P2) | kimi w | deepseek ρ | kimi ρ |
|---|---|---|---|---|
| theory_of_change | 0.012 | 0.113 | +0.190 | +0.171 |
| impact_per_dollar | **0.000** | **0.000** | +0.049 | +0.065 |
| team_evidence | 0.194 | 0.000 | +0.072 | +0.157 |
| epistemic_integrity | **0.794** | **0.887** | +0.360 | +0.374 |

Rule check: w(EI) is the maximum ✓; w(I/$) = 0.000 ≤ 0.1 ✓; ρ(EI) >
ρ(I/$) ✓ → REPLICATED. A judge from a different family and training
lineage, with zero cached judgments (646 fresh comparisons, 7 refusals
quoted), reconstructs the juried dollar order BETTER than the original
judge (combined Spearman 0.398 vs 0.339) and lands the same inversion:
zero weight on the community's professed top value, dominant weight on
epistemic integrity. Kimi's position-flip rate is 16.7% (53/317) vs
deepseek's 25% — under the registered loser condition, no third judge is
required.

## Run B — polish control: **CONFOUND EXCLUDED**

With `overall writing quality and polish of the prose` present as a
fifth candidate, the fit gives it weight 0.000 and epistemic_integrity
weight 1.000. Polish alone is nearly uninformative about the dollar
order (ρ +0.034). The mundane alternative — "juried readers reward
careful prose, and the EI rubric proxies it" — is excluded: the order
tracks the specific epistemic-integrity content (honest failure modes,
quantified claims, falsifiable milestones), not general writing quality.

Noise notes, stated: run B's per-attribute alone-ρs are noisier than the
original (165 comparisons/attribute, different pair schedule under seed
7 — only 118 cache hits of 824; flip rate 36% this run). The weight
verdict (1.000 vs 0.000) does not lean on the noisy per-attribute ρs.

## Publication gate

REPLICATED ∧ CONFOUND EXCLUDED → per the registration, **the post draft
advances**. Publication itself remains operator queue Q4: venue and
framing are the operator's decision alone.

## Spend — one honest overrun

Run A $6.78 (kimi produced 1.64M output tokens of reasoning — 4.5× the
original judge's verbosity; the registered estimate $1.3–3.2 assumed
deepseek-like output volume). Run B $0.29. **Total $7.07, above the $5
abort threshold registered for this pack** — the threshold was written
as an in-flight abort, but cardinal surfaces cost only at completion, so
the overrun was discovered when the run returned. Within the guardian
panel's <$10 bound for the whole move. Lesson recorded: registrations
should cap kimi-family runs in comparisons-per-dollar terms, or the CLI
needs an interim cost stream before a dollar-denominated abort line is
enforceable.

## Standing

The twice-measured P2 headline now stands on two judge families for the
revealed leg, with the polish confound excluded. Remaining known
weaknesses, honestly: the "professed" leg is an LLM rendering of
community discourse (Feynman seat, 2026-07-26 — the defensible headline
is the ACX↔EACC mechanism flip, which needs no professed leg); and the
mechanism↔cohort confound remains (two mechanisms measured on two
cohorts) — the within-cohort deconfound is the pre-registered P3
follow-up, quoted as an open limitation in any public write-up.
