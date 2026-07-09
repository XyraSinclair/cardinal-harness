# Red-team synthesis — adjudication of 30 findings (2026-07-09)

Five adversarial lenses over the repo's IDEAS (not code quality): parsimony,
mathematical register, from-The-Book, strategy, doctrine. Full findings in
this directory. **Tier receipt**: all five agents served `claude-fable-5`
on every turn (30–58 turns each), verified from transcript `message.model`
fields; pre-flight probe also fable-5. Main-loop spot-verification was
performed on every finding acted on (the sweep even-component arithmetic
was recomputed by hand; fuse() non-idempotency and AGENTS.md staleness
were verified in source before any edit).

Scoreboard: 30 findings → **7 executed same-day** (commits 7c3aa13, +1
follow-up), **6 filed as issue/comments** (#51 new; comments on #45, #46,
#49, #50), **10 proposed to operator** (doctrine text, strategy reorder,
publishing — not the panel's or this session's to decide), **5 deferred
with named gates**, **2 amended** (accepted in part, with the weak half
identified). Zero findings rejected outright — and 8 explicit SURVIVALS
were recorded where the panel attacked and the existing work held
(Hodge, orbit/Parseval, transitivity 2se discipline, packet composition,
scope fences, §2/§7/§11, fused-beats-both which *gained* a Steiger receipt
z≈2.2–2.5 it never had).

## Convergence map (independent lenses, same verdict — the strongest evidence class here)

| Verdict | Lenses |
|---|---|
| fuse() non-idempotent, CRDT claim false as shipped | math #4 ∧ parsimony #3 |
| #46 sheaf layer is vocabulary without new receipts; wire format is the real content | book #4 ∧ math #6 ∧ strategy #5 |
| No validation loop exits the operator (dogfooding closed / zero demand-side data) | doctrine #1 ∧ strategy #1 |
| "Distribution > capability" lost silently to the roadmap | doctrine #2 ∧ strategy #5-context |
| Monoid sentence: the theorem is the multiset, not compressed statistics | book #4-bonus ∧ math #6 |
| Narration outruns instrument exactly in the proudest packs | math meta ∧ doctrine #4 |

## Executed same-day (receipts: commit 7c3aa13 and follow-up)

1. **Spin-sweep even-component narrative reversed** (math #1, hand-verified:
   gpt even mean +0.239 not "near-zero", sonnet +0.035 not "+0.31") →
   erratum ON TOP of the pack; FIRST_PRINCIPLES sweep paragraph corrected
   (including the ordinal-ladder and excerpt-asymmetry caveats from math #2,
   downgrading "CLOSED" to "PARTIALLY closed"); receipt-viewer copy corrected
   same day. This conviction includes work shipped by this session yesterday.
2. **fuse() made genuinely idempotent** (math #4 ∧ parsimony #3) —
   packet-id dedup, redelivery pin byte-identical, validated against the
   planted pre-fix behavior (first plant was secretly a compile error;
   caught, redone honestly). CRDT sentence now stated as a theorem with its
   precondition (evidence-disjointness of distinct packets = provenance
   layer's contract).
3. **"Sufficient statistics" phrase sharpened** to the free-commutative-
   monoid statement (no mergeable compression under Huber) in packet.rs.
4. **AGENTS.md stale Key Areas deleted** (parsimony #4): pipeline/commander
   entries gone (deleted by 91af6a8, never removed from the doc).
5. **Two prior errata hoisted to top** of judge-bench and v1.1 packs
   (doctrine #5's checkable half).

## Filed (issue/comments)

6. **#51 (new): interval-censored (ordered-probit) likelihood** — book #1,
   the deepest single finding of the panel: the ratio ladder is a quantizer
   treated as a ruler; the censored likelihood dissolves ≥5 mechanisms
   (ORDINAL_OBSERVATION_RATIO hack, quantization-curl floor, ladder-geometry
   debate, template trichotomy, possibly the ordinal-beats-ratio negative).
   Zero-spend confirmations specified (ladder_curl refit; heavy-noise regime).
7. **#45 comment**: orbit absorption becomes a done-criterion (parsimony #1
   ∧ book #6) — spin stays outside the group (linear response), the rest
   restrict to subgroups or #45 ships probe N+1.
8. **#46 comment**: descope to wire format + cross-context transport; sheaf
   demoted unless a cellular sheaf with real restriction maps (template
   gains) is instantiated; add cross-target determinism (libm powf/ln
   unpinned — byte-identity currently per-binary, marketed per-party).
9. **#49 comment**: JCB v2 scored as energy fractions of the orbit
   decomposition (book #6); JCB = diagonal block of the portfolio Gram
   matrix — cross-judge blocks at fixed character name WHICH bias channels
   labs share, the refinement behind effective-error-sources 2.89 (book #3).
10. **#50 comment**: the Φ formula prices an unspecified estimand (math #3)
    — persistent b_p belongs in the conditional mean, the trigger is the
    certified crossing not the raw crossing, Gaussian tails vs Huber; the
    on-disk backtest must gate any pricing language.

## Proposed to operator (not executed — doctrine text, strategy, publishing are yours)

- **PRINCIPLES.md word-ledger edits** (doctrine #1–#4, #6; net +14 words,
  every rhetorical target a deletion): §8 external-reference clause; delete
  "Distribution > capability" slogan OR enforce it (note: it lives in
  differentiation.md, and the panel showed it lost 9:1 in 3 days); delete
  the "self-refutation count" metric sentence (keep the absence-framing
  line) — accepting this would also retire this session's own reporting
  habit; §4 retitle + "binds hardest on our own instruments"; §10 retirement
  criterion (stopping rule). All are edits to an operator-authored file.
- **Stake reorder** (strategy #1–#4): external-contact event as stake #0;
  habit loop above the map; 14K scale-up shrunk pending the 20-entity
  "did this change anything I'll do" probe; JCB published as the one asset
  with a living external consumer. Six probes, <$10 total, each can lose.
  The GitHub-Pages move specifically: the panel's claim that the Artifacts
  ban "doesn't cover" it is an interpretation of an operator ban — that
  scope call, and any publishing, is yours alone.
- **Cost sig-figs** (doctrine #5's other half): round sub-$0.10 costs to
  1 sig fig across receipts to end ritual-digit skimming.

## Deferred with gates

- **Seven primitives → six** (parsimony #2): the weight-as-functional half
  is verified (PacketObservation fuses them); the entity|attribute-as-roles
  merge cites §7's duality theorem — verify that theorem's statement before
  restructuring the doc.
- **canonical_bucket_v1 retirement** (parsimony #6): gate = confirm
  ratio_letter_v1 receipt covers bucket's grid cell; freeze-for-replay is
  the right mechanism if so.
- **transmissibility → judge_geometry statistic** (parsimony #5): gate =
  one refactor PR when canonize is next touched; same input object confirmed.
- **8-block Pythagoras master identity** (book #2): $0 synthetic-array pin;
  enters when pulled as a dependency (parsimony rule). Honest boundary
  recorded: stochastic transitivity does NOT reduce — it is the GoF test of
  the whole generative model.
- **Confidence-map knobs (eps, γ)** (book #5): $0 ML fit on committed
  receipts; if flat, delete the knob. Runs with the next receipts session.

## Amendments (accepted in part)

- math #2 (χ not a susceptibility): the reparameterization critique is
  accepted and now in FIRST_PRINCIPLES; the probe itself keeps its value as
  a *shape detector* — the kill is of the "linear-response coefficient with
  units" language, not the instrument.
- strategy #6 (#50 gate): accepted with the gate re-pointed at "first
  external counterparty", but the backtest stays scheduled regardless — it
  tests the math, not the market.

## What this cost and what it caught

Five fable-5 agents, ~2.1M transcript tokens. It caught: one live
receipts-vs-narrative contradiction in a shipped pack (with same-day
downstream contamination in a page shipped yesterday), one
theorem-level false claim in the protocol core (now a passing test), one
sign-flipped doctrine violation (errata placement), stale authority docs,
and one genuinely new research direction (#51) that no document in the
repo had conceived. The panel also *strengthened* one claim the repo was
right about but had never tested (fused-beats-both, Steiger z≈2.2–2.5).
The session health claim is left as the receipts above rather than a
count — one of the panel's own findings is that counting them is the
wrong frame, and that finding is pending operator adjudication.
