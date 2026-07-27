# P2 second-judge gate — registration (2026-07-27)

Committed BEFORE either run executes. The commit history is the registration
(same protocol as `artifacts/live/manifund-p2-2026-07-19/README.md`).

## Why

Guardian panel 2026-07-26 (notes/guardians-2026-07-26/SYNTHESIS.md, conflict
C1): the P2 professed-vs-revealed inversion publishes only after (a) a
second-judge direction replication and (b) a polish-control test excluding
the mundane alternative "juried readers reward careful prose." Feynman seat
Move 2, adopted verbatim; Hamming's bound on total spend <$10, synthesis
working cap ~$2, hard abort $5.

## Input (frozen, unchanged from P2)

`artifacts/live/manifund-p2-2026-07-19/acx2024-funded-order.json` — the 41
funded ACX 2024 proposals in realized dollars-descending order (believed
order, best first).

The four canonized attribute wordings, byte-identical to the P2 explain run:

1. `plausibility of the causal path from the proposed activities to the claimed impact`
2. `expected impact per marginal dollar at the stated minimum funding ask`
3. `strength of verifiable track-record evidence that this team can execute this plan`
4. `epistemic integrity of the write-up: honest failure modes, quantified claims, falsifiable milestones`

Polish-control wording (frozen here, first use anywhere):

5. `overall writing quality and polish of the prose`

## Runs

**Run A — second-judge direction replication.** Judge
`moonshotai/kimi-k2.6` (JCB co-#1, different family and training lineage
from deepseek). Candidates 1–4 only. `--budget 660 --seed 7`. Outputs
`secondjudge-kimi-explain.json` + `.log` in this directory.

**Run B — polish control.** Judge `deepseek/deepseek-v4-flash` (the original
P2 judge; the pairwise cache should cover most of candidates 1–4's
comparisons, so marginal spend is mostly candidate 5). Candidates 1–5.
`--budget 825 --seed 7`. Outputs `polishcontrol-deepseek-explain.json` +
`.log`.

Exact command shape (key fetched from vault into process env, never logged):

```
target/release/cardinal explain artifacts/live/manifund-p2-2026-07-19/acx2024-funded-order.json \
  --candidate "<wording>" [x4 or x5] \
  --model <judge> --budget <n> --seed 7 --format-json
```

Note stated honestly: the original P2 explain run's `--seed` was not recorded
in its pack; seed affects pair scheduling only, not the fit definition. Both
new runs pin `--seed 7`.

## Pre-committed verdict rules

Let `w(x)` = fitted weight, `ρ(x)` = spearman_alone, EI = attribute 4,
I/$ = attribute 2, POL = attribute 5.

**Run A (direction replication):**
- REPLICATED iff `w(EI)` is the maximum of the four AND `w(I/$) ≤ 0.1`
  AND `ρ(EI) > ρ(I/$)`.
- KILLED iff `w(I/$) ≥ w(EI)`.
- Otherwise AMBIGUOUS — reported as such; per Feynman's loser condition, an
  ambiguous result means consider one third judge, stop if total spend
  passes the cap, and report the ambiguity rather than adjudicating two
  noisy votes.

**Run B (polish control):**
- CONFOUND EXCLUDED iff `w(EI) > w(POL)`.
- CONFOUND WINS iff `w(POL) > w(EI)` AND `ρ(POL) ≥ ρ(EI)` — then the
  publishable claim is not "juried order rewards epistemic integrity" but
  "juried order rewards writing quality, which the EI rubric partially
  proxies"; the post must not run in its current framing.
- Otherwise AMBIGUOUS, reported.

**Publication gate:** the post draft advances iff Run A = REPLICATED and
Run B = CONFOUND EXCLUDED. Any other combination: results are written up
honestly in the pack and the operator decides the next step.

## Spend

Estimates: Run A ≈ $1.3–3.2 (kimi-k2.6 at $0.95/M in, $4/M out; output-token
volume is the uncertainty). Run B ≈ $0.1–0.5 (cache-dominated). Hard abort
threshold $5 total; actuals reported in the results note whatever they are.
