# Operator Queue — decisions only Xyra can close

The one control loop this repo lacked (guardian panel 2026-07-26, Leveson
M1, unopposed): agent-executable findings close at machine speed; decisions
owned by the operator had zero measured throughput and no visibility. This
file is the queue. **Cap: 5 open items.** Any process (panel, red-team,
campaign) wanting to add a sixth must first get one closed or explicitly
dropped. Sessions: read this at start when work touches strategy, doctrine,
or publication; update states in the same commit as the work that changes
them.

## OPEN (5 of 5 — queue is at cap)

**Q1. Adjudicate the six 07-09 doctrine word-ledger edits.** Written and
waiting since 2026-07-09 (notes/red-team-2026-07-09/SYNTHESIS.md § "Proposed
to operator"; full text in doctrine.md). Six one-line accept/rejects: §8
external-reference clause · delete-or-enforce "Distribution > capability"
(PRINCIPLES.md:84, still verbatim) · delete the "self-refutation count"
metric sentence · §4 retitle + "binds hardest on our own instruments" ·
§10 retirement criterion (stopping rule) · sub-$0.10 cost sig-figs.
Verified 2026-07-26: zero of six landed.

**Q2. Stake reorder + external-contact stake #0.** 07-09 strategy findings:
external-contact event as stake #0, habit loop above the map, 14K scale-up
shrunk pending the 20-entity probe, JCB published as the one asset with a
living external consumer. Partially overtaken: pairwiseratio.org is live;
the contact *event* (a post a stranger reads) has still never happened.
Q4 is its vehicle.

**Q3. Adopt the instrument admission/retirement constraint.** Guardian
panel Leveson M3: no new instrument or axis without a pre-registered
admission rubric and a named retirement condition at birth (the pattern
that already worked: axis wave 2 admitted 6/69 via rubric + decoys;
eps/gamma knobs died by pre-named gate). Subsumes Q1's §10 line but is a
standing constraint, not a doctrine edit.

**Q4. Publish P2?** The professed-vs-revealed mechanism inversion
(notes/manifund-campaign-2026-07-13/p2-results.md). **GATE PASSED
2026-07-27**: kimi-k2.6 REPLICATED the inversion (EI weight 0.887, EV/$
0.000, better reconstruction than the original judge) and the polish
confound is EXCLUDED (EI 1.000 vs polish 0.000) —
artifacts/live/manifund-p2-secondjudge-2026-07-27/RESULTS.md, verdicts
mechanical from pre-committed rules; one honest spend overrun quoted
($7.07 vs $5 registered abort line). The remaining decision is venue +
framing — the post publicly characterizes a named community's funding
decisions; mechanism-property framing (neither mechanism the villain) is
the proposed mitigation. Draft in progress; SENDING is yours alone.

**Q5. Axis wave 3 arrest gate.** Guardian panel Hamming M3 + Feynman M3:
wave 3 does not run until (a) one smart stranger sees one admitted-axis
top-10 and the arrest (or non-arrest) is recorded, and (b) the small-tier
replication + test-retest baseline runs on the frozen wave-2 probe sets.
**(b) DONE 2026-07-27** (RESULTS-REPLICATION.md: verdict UPGRADED TO
FINDING per frozen rules — 3/6 axes tier-general, retest median +0.965;
corrected headline: haiku-4.5 breaks the price-tier story, the divergence
is a capability-class property). (a) still needs a stranger you pick;
until then wave 3 stays gated. The author-family confound is the honest
next kill before any public tier claim.

## DUE DATES

- **2026-08-08** — the 07-09 doctrine pack's 30-day tests fell due. Run
  2026-08-10, two days late (the miss is itself a data point on the
  queue's throughput). Results:
  - **External-reference pack (F1): PASS** — closed early as C1 below.
  - **Instrument-vs-artifact commit ratio (F2): moved, without the edit.**
    Window 2026-07-09..08-08, 78 commits: 21 instrument-only (src/tests),
    36 artifact/notes/site-only, 6 mixed, 15 other — 0.58:1 vs the ~9:1
    baseline the finding measured. The proposed doctrine edit (Q1) never
    landed, so the wording was epiphenomenal: the ratio inverted anyway.
    F2's premise that the slogan drives the ratio is refuted in the
    favorable direction.
  - **Errata placement (F5a): PASS** — every standing erratum found by
    grep over pack READMEs/RESULTS and notes (judge-bench ×2, spin-sweep,
    kimi-k3-bench, axis-research ×2, best-library ×2, decimal-pmf) sits in
    the top ~10 lines of its file; none buried.
  - **Retirement/dormant marking (F6b): FAIL — Finding 6 stands
    confirmed.** Zero retirements or dormant markings in the window
    (grep retir/dormant/removed over the commit log). First
    retirement-shaped event is 839c7a7 (2026-08-10, dead embed/batch
    subsystem excision) — outside the window, and code-level, not
    instrument-level. Nothing in the doctrine can yet kill an instrument;
    Q3 remains the live fix.
  - The >$10-map twin test (F6a) never triggered: no >$10 map shipped in
    the window (max single-run spend on record: $7.07, P2).

## CLOSED

- **C1 (2026-07-26). P2 passed doctrine Finding 1's external-reference
  test — 12 days early.** The 07-09 test asked: "does any pack exist by
  2026-08-08 whose reference column was produced by neither the operator
  nor the stack?" (base rate then: 0/28). ACX 2024 funding decisions and
  EA CC crowd outcomes are exactly that, with predictions registered
  before unblinding (ca1928c → 892e0c0 → c0f5cd9). Noticed by the
  guardian panel's Leveson seat; recorded here so a passed test no longer
  looks identical to an ignored one.
