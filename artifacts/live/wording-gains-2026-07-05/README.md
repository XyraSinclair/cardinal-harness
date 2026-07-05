# Per-model wording gains, live (2026-07-05)

`examples/wording_gains.rs`: the JCB corpus (20 pairs) judged through all
three wordings of the ratio question, solved jointly by
`solve_with_template_gains` — scores and per-template multiplicative
gains fitted together, canonical_v2 pinned at 1. This is the
estimator-level answer to the wording-invariance finding: treat each
wording as an instrument channel with unknown gain and calibrate on
shared events, instead of forbidding mixed elicitation.

| Model | fraction_v1 gain | less_v1 gain | rms | rms naive | payoff |
|---|---|---|---|---|---|
| gemini-2.5-flash | 1.365 | 1.204 | 0.651 | 0.670 | 3% |
| claude-sonnet-4.6 | 1.427 | **1.009** | 0.232 | 0.317 | **27%** |
| gpt-5.4-mini | **0.564** | 1.524 | 0.497 | 0.545 | 9% |

## Findings

1. **Gains are per-model instrument constants, not a universal bias.**
   sonnet and gemini's fraction channels run hot (1.43×, 1.37×); mini's
   runs COLD (0.56×) while its less-channel runs hot (1.52×). The
   single-pair receipt (wording-invariance pack) showed mini's fraction
   reading higher on one pair — the corpus-wide regression, which
   calibrates against the model's own fitted scores, reverses that
   impression. One pair is an anecdote; twenty pairs against a joint fit
   is a gain.
2. **sonnet's "times less" channel is perfectly calibrated** (1.009): it
   mirrors its own scale essentially exactly through the group inverse.
   Its fraction channel is its one miscalibrated instrument.
3. **The payoff column is a lie detector for the gain model itself.**
   Calibration cuts sonnet's residual 27% — its wording disagreement
   really is a scale factor. gemini improves only 3%: its wording
   disagreement is mostly NOT a pure gain (pair-specific/nonlinear), so a
   scalar per channel cannot absorb it — honest evidence the linear gain
   model has limits, printed rather than hidden.
4. Solver guarantees (pinned in `tests/gain_calibration.rs`): planted
   gains recovered ±0.1; uniform-gain data yields no phantom gains and no
   manufactured improvement; a sign-incoherent channel COLLAPSES toward
   zero instead of being "calibrated" — brokenness is not a scale factor.

Cost: $0.14 across three models (60 calls each, no cache). Raw console
output in `gains.txt`.
