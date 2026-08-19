# Manifund P2 — retrodiction gauntlet (2026-07-19)

Campaign phase P2 per `notes/manifund-campaign-2026-07-13/plan.md`: judge the
ACX Grants 2024 cohort (83 projects, 41 funded) on the four canonized
attributes, register the predictions, then unblind against the realized
funding decisions.

## Registration protocol

Ground truth (`data/manifund/ground_truth.csv`) sits in this same repository,
so blinding is procedural, not cryptographic — stated plainly: the prediction
run was executed and **committed before any comparison against ground truth
was computed**, and `unblind.py` (also committed in the prediction commit)
fixes the exact analysis — AUC on funded-vs-not, Spearman vs dollars raised,
per-attribute AUCs, both-directions disagreement shortlists — before the
first look. The commit history is the registration.

- Prediction commit: see the commit that adds `acx2024-response.json`.
- Unblinding commit: the following commit, which adds `unblind-output.txt`
  and the results note. The AUC is reported whatever it is (plan discipline).

## Contents

- `acx2024-response.json` — full rerank response: 83 entities ranked,
  per-attribute latents ± std, Pareto front, attribute correlations, meta
  (comparisons, cost, order residual, frustration, stop reason).
- `acx2024-report.md` — human-readable run report.
- `acx2024-trace.jsonl` — per-judgment trace bound to the engine spec.
- `acx2024-run.log` — run stdout/stderr.
- `unblind.py` — the pre-registered analysis (committed with predictions).
- `unblind-output.txt` — the analysis output (committed at unblinding).

Request: `data/manifund/requests/p2-acx2024-4attr.json` — judge
`deepseek/deepseek-v4-flash` (P1 model policy), 4 attributes
(theory_of_change .30, impact_per_dollar .30, team_evidence .25,
epistemic_integrity .15), top-k 41 @ tolerated_error 0.15, budget 1,400
comparisons, seed 7.
