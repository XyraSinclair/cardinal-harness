# Council synthesis — the essence, distilled (2026-07-09)

Question put to a multi-model council (identical curated bundle:
PRINCIPLES, differentiation, red-team synthesis, FIRST_PRINCIPLES opening,
README pitch — `prompt-bundle.txt`, 778 lines): what is the ONE object,
the ONE tool, what to delete, the 60-second stranger experience, and the
cheapest decisive test that could fail.

Seats: **gpt-5.5 (xhigh)**, **gemini-3.1-pro-preview**, **claude-fable-5**
(answers committed verbatim as `seat-*.md`). Two seats failed on
plumbing, honestly noted: gpt-5.4-pro (dead OPENAI_API_KEY + npx/oracle
CLI breakage) and kimi-k2.5 (dead OPENROUTER_API_KEY; macOS lacks
`timeout` for the wrapper retry). The gpt-family perspective is covered
by the 5.5 seat.

## Consensus (3/3 seats, independently)

1. **The one object.** The judgment measurements form a flow (a 1-form)
   on the entity graph; its gradient part is the belief; the invariance
   group *defines* "belief" (a fixed point of transformations that
   shouldn't matter); curl, framing response, and variance are the
   measured ways a judgment fails to be one. Every mechanism in the repo
   is this object from a different side — IRLS solves the potential,
   probes are the group action, the JCB turns the object on the judge,
   packets make it mergeable, portfolios take it over correlated judges.
   *There is no second idea. That is a strength.* Stranger's sentence:
   **it measures whether the model actually believes what it just told
   you, or was only echoing how you asked — and gives you the number.**

2. **The one verb.** Fold the probe battery behind a single front-door
   verb — one file + one criterion in, one verdict + one self-contained
   HTML receipt out. The page leads with the verdict (Codex's triad:
   **Stable / Contested / Manipulable**; Claude's: BELIEF / WEAK BELIEF /
   ECHO), not with a ranking — *sorting is the byproduct; believing is
   the essence*. The hero interaction is the framing slider (the
   shipped receipt-viewer generalized from a hand-built page into a
   compiler output). Names offered: `audit` (5.5), `believe` (fable),
   `sort --audit` + a separate `render` compiler (gemini).

3. **The deletions.** (a) Flag archaeology: ~20 probe flags collapse
   behind the verb; a stranger must not discover the product through
   flags. (b) Physics vocabulary comes OFF the outer surface — belief,
   echo, survives, breaks, "moves with the asker"; χ/Hodge/paramagnetic
   stay where the math is literal (code, fine print). (c) stderr math
   dies; the math lives in the rendered artifact. (d) Sheaf vocabulary:
   finish the already-adjudicated deletion. (e) The README headline
   "LLM sorting" undersells the object.

4. **The von Neumann test — and the gate.** All three seats converge on
   the same falsifiable core: **the verdict must PREDICT judgment
   stability out-of-group.** Sharpest composite (pre-register before
   running):
   - Classify ~20 pairs as BELIEF/ECHO with battery A (order, polarity,
     spin).
   - Test with a disjoint battery B the classifier never saw: fresh
     procedurally-rotated paraphrases, wording inversion, and **re-ask
     after 3+ days** — the ✗ time row becomes the validation set (the
     cheapest untouched science in the repo).
   - Pre-registered bars: BELIEF pairs hold direction on battery B
     ≥ 90%; ECHO pairs near coin-flip; and (5.5's condition) the verdict
     must beat the neutral margin |Δ̂| alone as a predictor of retest
     survival. Gemini's χ state-dependence contrast (clear pair χ≈0 vs
     contested χ>0, certified beyond noise, currently held only at n=1)
     rides along as a sub-claim.
   - Cost: ~$1–2 at receipted prices + 3 days of clock.
   Three visible failure modes, each fatal to a specific pretension: if
   BELIEF pairs don't survive held-out transformations, the invariance
   group is decoration; if ECHO pairs do survive, "echo" is slander; if
   indistinguishable, the one object is a story.

## The binding rule the council handed back

**Run the test before shipping the verb.** A `believe`/`audit` command
whose verdict has never been validated out-of-group is exactly the
un-red-teamed countermeasure PRINCIPLES §7 forbids. The build order is
therefore: (1) pre-register the protocol with bars in the repo,
(2) run battery A + the immediate half of battery B, start the 3-day
clock for the time axis, (3) ship the verb only on a pass — and if it
fails, the failure is a bigger finding than the tool.

## Divergences (real, small)

- Verb name: `audit` vs `believe` vs `sort --audit`+`render`. (`render`
  as a separate trace→HTML compiler is infrastructure the atlas and
  leaderboard want regardless — it can be built without waiting on the
  validation gate, since it renders receipts that already exist.)
- Page lead: gemini wants the conflict-colored evidence graph first;
  5.5/fable want the verdict block first with the graph behind "Why?".
  Verdict-first won the synthesis: the graph is second-click.

## Relation to the red team (yesterday's panel)

The council's deletions are a superset-consistent restatement of
rt-parsimony #1 (orbit absorption) and rt-doctrine #4 (physics vocabulary
binds hardest on our own instruments); the held-out validation demand is
rt-strategy's "every probe can lose" applied to the product itself. No
contradictions between the two consultations; the council adds the
outer-surface/inner-register split and the ship-gate.

## Disposition

- This synthesis + seat receipts: committed (this pack).
- Pre-registration protocol doc: next concrete deliverable; the bars
  above are the draft to freeze.
- `render` compiler (trace/receipts → self-contained HTML): buildable
  now, validation-independent.
- Verb naming + README reframing: operator's call, blocked behind the
  validation result anyway.
