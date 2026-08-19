# P0 — comfort pass results (2026-07-14)

Total spend: **~$0.10** (ceiling was $2). Cache at
`data/manifund/probes/cache.sqlite`; rubrics at `data/manifund/rubrics/`.

## judge — three probe pairs, all agree with the human funding direction

| pair | attribute | model | verdict | note |
|---|---|---|---|---|
| charge-selective (funded $80k) vs growing-human-bl (unfunded) | impact/$ | deepseek-v4-flash | A wins 2.5× (conf .70) | agrees with funders |
| african-school (funded $100k) vs write-messaging (unfunded) | theory of change | deepseek-v4-flash | A wins **1.5×** (conf .70) | agrees, but *narrowly* — not rubber-stamping |
| charge-selective (funded) vs conduct-research-biomarkers (unfunded) | impact/$ | deepseek-v4-flash | A wins 2.5× (conf .70) | agrees |
| " same pair " | impact/$ | claude-opus-4.6 | A wins 5.1× (conf .62) | agrees on direction, more decisive, honest lower confidence |

Reads: the judge tracks human decisions in direction on all three, but the
*magnitudes* carry signal — the "obvious" charter-city-vs-YIMBY gap is only
1.5×, and opus is more decisive yet less confident than deepseek on the hard
biomedical tie. Direction-agreement on hand-picked pairs is necessary, not
sufficient; P2 retrodiction over a full cohort is the real test.

Cost spread is the ladder's whole argument: deepseek $0.0005/judgment vs opus
$0.0202 — 40×. The frontier model is worth it only where the cheap one is
uncertain, which is exactly what `frontier_ladder` policy routes.

## `--show-prompt` — the microscope holds up

Full system prompt renders: the ratio ladder
`[1.0 … 26.0]`, strict JSON contract `{higher_ranked, ratio, confidence}`, and
the principled refusal clause. Both project bodies render with the `[truncated]`
marker landing exactly at the 8,000-char cap, so the judge sees its own
denominator. No naked numbers — every verdict traces to a visible prompt.

## `elaborate` — rubrics name what NOT to reward

All four campaign attributes expanded (deepseek, ~$0.001 total,
`data/manifund/rubrics/*.md`). Each rubric's discriminating move is the
negative clause: theory_of_change "do not reward the importance or novelty of
the impact itself"; impact_per_dollar "do not reward total impact or average
impact per dollar"; team_evidence "do not reward reputation, enthusiasm, or
general experience"; epistemic_integrity "do not reward mere confidence,
polish, or comprehensiveness". This is the anti-gaming spine of the campaign.

## `distinguish` — the carving primitive

Focus: `synapse-student-research-incubator` vs 15 live peers, 5 proposed axes,
198 comparisons, $0.079, frustration 0.080 (low — coherent judgments).

```
p97  z+11.28  focuses on life and earth sciences
p97  z+4.01   targets underserved high school students
p97  z+4.01   is a member of LA STEM Collective
p50  z 0.00   requires minimum $500 donation
p3   z 0.00   has completed a pilot program
```

The proposals were hypotheses; the measured profile is the evidence. The model
guessed "completed a pilot program" would distinguish it — measurement put that
at p3 (it does not). This is the loop an agent runs when lost in the sauce:
propose cheap, measure honestly, keep only what the profile earns.

## Gate → P1

Machinery verified end-to-end on real Manifund data at negligible cost. Nothing
blocks P1 (attribute layer: `weigh` professed weights, `slate` stakeholder
proposals, `canonize` transmissibility across judges). Proceed.
