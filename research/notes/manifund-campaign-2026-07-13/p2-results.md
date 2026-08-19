# P2 — retrodiction gauntlet results (2026-07-19)

Evidence pack: `artifacts/live/manifund-p2-2026-07-19/` (requests, responses,
traces, run logs, pre-registered analysis scripts, outputs). Ceiling was $15;
spend **$1.43** (ACX rerank $0.49 + explain $0.33 + EA CC rerank $0.61).

## Registration

Predictions committed (`ca1928c`) before any comparison to ground truth was
computed; `unblind.py` fixed the analysis in the same commit. One
post-registration fix, quoted honestly: the ground-truth join needed the
`slug` column, not `id` — no statistic definition changed. EA CC unblinding
was pre-registered (`a8569a2`) before its response existed.

## ACX Grants 2024 — predict the funded set (n = 83, 41 funded)

Four canonized attributes, deepseek-v4-flash judge (P1 model policy), 987
comparisons, $0.49, zero refusals, stop = no_new_pairs. The top-41 boundary
was NOT resolved to the requested 0.15 tolerance (global top-k error 11.4 at
n≈12 comparisons/item) — quoted, not hidden; the AUC below is what the
composite achieves anyway.

| statistic | value |
|---|---|
| AUC, combined score vs funded | **0.635** |
| Spearman, combined vs dollars raised | +0.288 |
| Spearman among funded only (vs dollars) | +0.286 (n = 41) |
| AUC theory_of_change alone | 0.633 |
| AUC impact_per_dollar alone | 0.557 |
| AUC team_evidence alone | 0.605 |
| AUC epistemic_integrity alone | **0.655** |

Read: a $0.49, one-judge, 4-attribute composite recovers Scott's funding
decisions meaningfully above chance, and the *epistemic-integrity* rubric —
the attribute canonize tightened in P1 — is the strongest single predictor.

Disagreement shortlists (both directions, per campaign discipline) are in
`unblind-output.txt`: e.g. `run-a-self-help-` ranked #4 by the model but
unfunded; `conduct-field-re` funded at $13k but ranked #79 of 83.

## The headline: professed vs revealed weights

`explain` fit the four attributes against the REALIZED funding order
(funded-only, dollars descending, n = 41; 660 comparisons, $0.33, 8 cached).
Combined reconstruction Spearman 0.339.

| attribute | professed (P1 weigh / campaign wt) | revealed (fitted) | alone ρ |
|---|---|---|---|
| impact_per_dollar | **top** (0.24 weigh · 0.30 campaign) | **0.000** | +0.049 |
| theory_of_change | 0.30 campaign | 0.012 | +0.190 |
| team_evidence | 0.25 campaign | 0.194 | +0.072 |
| epistemic_integrity | **lowest** (0.15 campaign) | **0.794** | +0.360 |

The community's *professed* top value — expected impact per marginal dollar —
carries **zero** fitted weight in reconstructing what actually got funded.
What the realized order rewards most is the write-up's epistemic integrity
(honest failure modes, quantified claims, falsifiable milestones). This is
the gap no Likert rubric can exhibit: both vectors are measured on the same
ratio-scale machinery, on the same corpus, with the same judge.

Noise class, stated: one judge model, one cohort, funded-only order (n = 41),
25% position-flip rate on this judge (83/326 counterbalanced pairs flipped —
consistent with the consortium smoke that measured deepseek coherence 0.049
on one Manifund pair). The inversion's *direction* is large enough to survive
this noise class; its magnitudes are wording- and judge-conditional until a
second judge replicates (cheap: the cache makes replication ~$0.35).

## EA Community Choice replication (n = 78, 57 funded)

Same request shape, crowd quadratic-matching truth signal. 1,296
comparisons, $0.61, zero refusals, stop = no_new_pairs (top-57 boundary
error 11.1, same honesty as ACX).

| statistic | value |
|---|---|
| AUC, combined score vs funded | **0.682** |
| Spearman, combined vs dollars raised | **+0.420** |
| AUC theory_of_change alone | 0.666 |
| AUC impact_per_dollar alone | **0.682** |
| AUC team_evidence alone | 0.625 |
| AUC epistemic_integrity alone | 0.657 |

Read: the replication is STRONGER than ACX (0.682 vs 0.635, Spearman +0.420
vs +0.288) — and the attribute profile flips. Impact-per-dollar, the weakest
predictor of Scott's decisions (0.557), is the strongest single predictor of
the crowd's quadratic-matching outcomes (0.682). Two funding mechanisms on
the same platform reveal different value functions under the same judge,
same wordings, same machinery — the crowd behaves closer to its professed
EV/$ values than the juried process does. That is target-ontology row 7
(cohort standards) measured, and it sharpens the headline: the
professed-vs-revealed gap is a property of the *mechanism*, not of Manifund.

## Gate → P3

Closed inside budget: total P2 spend **$1.43** of the $15 ceiling
(ACX $0.49 + explain $0.33 + EA CC $0.61). Headline measured twice over:
professed values (EV/$ first) vs revealed values (epistemic integrity first
for the juried order; EV/$ genuinely predictive for the crowd order).
P3 (live slate, 109 proposals, top-20) inherits the four canonized
wordings, the deepseek judge with a frontier replication judge for the top
of the board (motivated by the measured 25% position-flip rate and the
consortium smoke's deepseek coherence 0.049), and the P2 evidence that
which attribute is load-bearing depends on the funding mechanism being
served.
