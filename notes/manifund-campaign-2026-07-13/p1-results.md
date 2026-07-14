# P1 — attribute layer results (2026-07-14)

Total spend: **~$0.09** (weigh $0.011 + slate $0.007 + canonize runs ~$0.07,
most of it the two abandoned over-budgeted attempts). Ceiling was $5.

## `weigh` — professed attribute importance (AHP)

Goal: "allocate Manifund's marginal $10,000 grant for maximal expected impact."
8 model-proposed considerations, weighed pairwise (120 comparisons, $0.011).

| weight | consideration |
|---|---|
| 0.236 | Expected value per dollar |
| 0.139 | Evidence base for the intervention |
| 0.136 | Counterfactual impact |
| 0.115 | Tractability of the problem |
| 0.109 | Quality of the grantee organization |
| 0.094 | Scalability potential |
| 0.092 | Neglectedness of the cause area |
| **0.080** | **Risk of negative side effects** |

Reads: EV/dollar dominates professed importance (~2.4× the mean). The load-
bearing design point — **risk of negative side effects ranks *last* in
professed weight**. That is not "risk doesn't matter"; it means the community
treats downside as a *disqualifier* (a gate), not a term you average against
upside. This directly justifies the campaign's architecture: downside risk is a
gate, never a weighted attribute. The professed-weights vector also becomes the
prior we compare against `explain`'s revealed weights in P2.

## `slate` — stakeholder-blind attribute proposals

Item: `charge-selective` (implantable artificial kidney). Four stakeholders
propose attributes blind to each other; breadth = how many unlike perspectives
back an attribute (the "alive in social reality" proxy). 5 calls, $0.007.

| breadth | attribute | backers |
|---|---|---|
| **2** | technical feasibility | regrantor, skeptical donor |
| 1 | clear path to prototype / to impact | regrantor, skeptical donor (2 phrasings) |
| 1 | membrane albumin selectivity | domain scientist |
| 1 | pore size without clogging | domain scientist |
| 1 | filtration rate / reduces dialysis dependence | domain scientist, beneficiary |
| 1 | track record in field | skeptical donor |

Reads: "technical feasibility" is the only attribute two *unlike* stakeholders
(a funder and a skeptic) independently reach for — the strongest candidate for
a shared decision axis. The domain scientist supplies the specialist sub-
criteria (albumin selectivity, pore size) that a generalist rubric would miss;
the beneficiary reframes everything as "reduces dialysis dependence". The slate
is cheap hypotheses; breadth ranks which to promote to measurement.

## `canonize` — transmissibility across judge families

Which *wording* of "epistemic integrity" makes two different judge models agree
on the ranking? 8 EA-Community-Choice projects, judges = gpt-5.4-mini +
gemini-2.5-flash-lite (deliberately different families), 160 comparisons, $0.063.

| transmissibility | signal | wording |
|---|---|---|
| **+0.833** | 0.535 | Commitment to falsifiable, quantified claims *(proposed)* |
| +0.810 | 0.629 | epistemic integrity: honest failure modes, quantified claims… *(seed)* |

Reads: the tightened proposal transmits *better* (+0.833 vs +0.810 mean cross-
judge Spearman) — two independent judges agree more on how to rank projects
under it. Transmissibility is the property that makes an attribute canonical:
it induces the same latent in different minds, measured not asserted. Note the
tradeoff — the seed has higher per-judge *signal* (0.629, it spreads projects
more) but lower cross-judge agreement; canonize surfaces exactly this
signal-vs-transmissibility tension instead of hiding it.

## Findings for the harness (robustness, not blockers)

Two real rough edges hit during P1 — worth fixing in cardinal, flagged not
patched (no code changes without approval):

1. **`--budget` on canonize is per-(candidate, judge) sort, not total.** The
   help says so, but it's a footgun: `--propose 3 --budget 240` with 2 judges =
   up to ~1,900 comparisons and a 20-min silent run. A total-budget flag or a
   startup log of the projected comparison count would prevent this.
2. **The canonize proposal-JSON parser is brittle across model families.**
   deepseek-v4-flash intermittently returns an empty completion
   (`ProposalParse EOF`); gpt-5.4-mini returned valid wordings inside a
   malformed envelope `{"[]":[...]}` (`trailing characters`). Same empty-JSON
   fragility broke `slate` on deepseek until I moved it to gpt-5.4-mini. The
   proposal/structured-output path needs a tolerant extractor (pull the first
   JSON array/object, or retry-with-repair) rather than strict whole-string
   parsing. **Campaign implication: use gpt-5.4-mini, not deepseek, for any
   structured-proposal step (slate/weigh/canonize propose); deepseek is fine as
   a pure pairwise judge** (it never failed a `judge` call).

## Measured rubric handed to P2

- Weights (professed prior): EV/$ 0.24, evidence 0.14, counterfactual 0.14,
  tractability 0.12, org quality 0.11, scalability/neglectedness ~0.09, risk =
  **gate**.
- Canonical wordings so far: impact/$ and epistemic-integrity tightened;
  "technical feasibility" promoted from the slate as a shared decision axis.
- Model policy for the campaign: **deepseek judge + gpt-5.4-mini for
  structured/proposal steps**, frontier only where the ladder flags uncertainty.

## Gate → P2

Attribute layer delivered a measured (not asserted) rubric and a professed-
weights prior to test revealed weights against. Proceed to the retrodiction
gauntlet: ACX Grants 2024 (83 projects, 41 funded), predict the funded set
*before* unblinding, report AUC + Spearman vs dollars, then `explain` the
realized order for revealed weights and the both-directions disagreement list.
