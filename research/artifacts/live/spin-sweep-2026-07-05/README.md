# Live susceptibility SWEEPS — χ as a slope, not a secant (2026-07-05)

## ERRATUM (2026-07-09, red team vs the pack's own receipts)

The shape narrative below misstates the even components; the raw numbers
stand, the reading dies. Recomputing the instrument's own statistic
(`even_response_mean`, mean over f = 1..3 of (m(+f)+m(−f))/2 − m(0)) from
the committed JSON:

- **gpt-5.4-mini even mean = +0.239 nats** — NOT the "near-zero even
  component" narrated below; it is comparable in size to the celebrated
  odd slope (+0.200/step). "Monotone in f" is also false of the data
  table (only the fitted line is monotone), and the "linear" verdict at
  R² = 0.81 sits below this instrument's own step-exposure pin
  (tests expose steps at R² < 0.9).
- **claude-sonnet-4.6 even mean = +0.035 nats** — NOT "the structure is
  in the EVEN part"; the quoted +0.31 is the single f = 3 edge value of
  three even readings with disagreeing signs (−0.148, −0.056, +0.310).
  The honest statement is: even component sign-indefinite at this n.

The two models' narrated even components are the REVERSE of what the
defined statistic gives. Root cause: the stored receipts carry
`even_response_mean: null` and the narration was written from hand-picked
edge values — the 2se raw-vs-certified discipline (used in the
transitivity pack) was not applied here. Downstream surfaces corrected
same-day: FIRST_PRINCIPLES sweep paragraph, receipt-viewer copy. n = 1
pair throughout; both directions of the correction are also n = 1.

`cardinal judge --sweep`: framing intensity swept from −3 (insistent
pro-second) through 0 (neutral) to +3 (insistent pro-first), each field
point measured in both presentation orders (14 comparisons). Fitting
log-ratio against field strength gives susceptibility as a genuine
linear-response coefficient plus a linearity R² — built because the
two-point secant cannot distinguish a flat response from a step response (adversarial self-review, notes/ideation-2026-07-05/).

Pair: "The obstacle is the way." vs "What gets measured gets managed.",
criterion "depth of insight about living well" — the contested pair from
the spin-probe pack.

## gpt-5.4-mini — m(f) ≈ 0.20·f (odd-dominant, linear)

| field | −3 | −2 | −1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| m (nats) | −0.77 | −0.41 | +0.04 | −0.23 | +0.22 | +0.65 | +0.33 |

**χ slope = +0.200 nats/step, R² = 0.81, sign(m) not constant over the
sweep.** The response is monotone in f with no detectable threshold and a
near-zero even component; m crosses zero inside the swept range. The
earlier two-point secant (+0.64) was sampling a real slope, not an
artifact.

## claude-sonnet-4.6 — flat odd part, positive even part

| field | −3 | −2 | −1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| m (nats) | +1.25 | +0.92 | +0.92 | +0.94 | +0.66 | +0.85 | +1.25 |

**χ slope = −0.014 nats/step, R² = 0.02, sign constant over the sweep.**
The odd part of the response is indistinguishable from zero; the
structure is in the EVEN part: (m(+3)+m(−3))/2 − m(0) = +0.31 nats, i.e.
the response depends on |f| rather than f at the sweep's edge. A secant
through ±2 would report χ ≈ 0.035 and represent none of this. n = 1 pair:
the even component is a measured quantity here, not a model property.

## Why this instrument earns its 14 calls

The two-point probe yields one secant. The sweep yields the response
FUNCTION, decomposable into odd and even parts about zero field — flat,
linear-odd, step-odd (low R²), and even-dominant shapes are all
distinguishable, and the first three are pinned by scripted judges in
`tests/judge_explain_cli.rs` (slope 0.300 recovered exactly; step exposed
at R² < 0.9).

Wording note with teeth: building this caught that prompt templates
HTML-escape apostrophes (`'` → `&apos;`) — the first shipped spin framing
reached every model slightly garbled ("I&apos;ve already looked...").
All framings are now apostrophe-free; earlier receipts remain valid as
measurements of the wording they actually sent.

Cost: $0.09 for both sweeps. Raw JSON per model in this directory.
