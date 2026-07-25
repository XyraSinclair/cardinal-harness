# Lens C generator output — epistemic & structural depth (2026-07-24)

## collapse_radius
- **def:** A has more collapse_radius than B when falsifying A's central claim would invalidate a larger volume of downstream conclusions, systems, and decisions that currently rest on it.
- **smell:** Requires holding the dependency graph of human knowledge and tracing what actually *relies on* the claim versus merely mentions it; small models collapse to citation count or topical importance.
- **top:** A foundational method paper whose correctness thousands of results silently assume — Diffie-Hellman key exchange, or the original randomized-trial methodology.
- **confounder:** Fame / citation count.

## incompressibility
- **def:** A has more incompressibility than B when a faithful summary of A must retain nearly all of A's length to preserve its claims, while B collapses to a tweet without loss.
- **smell:** Judging it means actually attempting the compression internally and measuring what breaks; small models proxy to lexical density and jargon rate.
- **top:** A proof-sketch essay where every paragraph is a distinct load-bearing move — Shannon 1948.
- **confounder:** Rare-vocabulary density / aphoristic style.

## retrospective_inevitability
- **def:** A has more than B when A's conclusion feels more forced-once-seen — competent contemporaries could have derived it in a page — yet demonstrably wasn't anticipated.
- **smell:** Needs a dated model of what was known at the time *plus* the felt shortness of the derivation path; small models read confidence of tone.
- **top:** The Bitcoin whitepaper; Wegener on continental drift.
- **confounder:** Assertive, definitive prose style.

## antimemetic_payload
- **def:** A has more than B when A carries more claims that are both true and structurally resistant to transmission — unflattering to every faction, unquotable out of context, costly to the person who repeats them.
- **smell:** The judge must evaluate truth and simultaneously model the memetic selection landscape filtering what spreads; small models collapse into edginess detection.
- **top:** A postmortem naming the boring, blame-diffuse, statistically correct cause of a disaster everyone wants a villain for.
- **confounder:** Contrarian tone / taboo-topic word count.

## prescience_margin
- **def:** A has more than B when A stated, earlier and against stronger contemporary opposition, positions that later hardened into consensus.
- **smell:** Requires dating the consensus *trajectory* of specific claims — not just what's true now but when it became respectable; small models reward hedged vagueness that "predicted everything."
- **top:** Licklider's "Man-Computer Symbiosis" (1960); a 2014 essay asserting scaling laws would eat structured approaches.
- **confounder:** Prophecy vagueness (Nostradamus effect) and mere oldness.

## seedbank_fertility
- **def:** A has more than B when a competent reader can extract more distinct, workable research programs, tools, or companies from A that are not mere applications of its headline claim.
- **smell:** Judging requires actually *generating* the downstream ideas and checking they're live — latent search only a strong generator can perform; small models count explicit "future work" bullets.
- **top:** Von Neumann's self-reproducing automata notes — a technical artifact whose asides each contain an unbuilt field.
- **confounder:** Number of explicit open questions / breadth of topics touched.

## scar_tissue_density
- **def:** A has more than B when more of A's specifics are the kind only contact with the territory produces — failure modes, off-by-one costs, boring parameter values — that plausible extrapolation would not generate.
- **smell:** Distinguishing unfakeable detail from confabulated texture requires knowing what reality does at that resolution; small models count anecdotes and first-person pronouns.
- **top:** A production postmortem or field-notes document whose every specific number is faintly surprising yet checks out — Dan Luu-grade measurement essays.
- **confounder:** Anecdote count / first-person narrative style.

## book_proximity
- **def:** A has more than B when fewer of A's structural choices could be altered without loss — the proof or design approaches the unique form Erdős's Book would record.
- **smell:** Requires searching the neighborhood of alternative designs and confirming each perturbation is worse — an internal optimization audit; small models reward shortness and minimalist aesthetics.
- **top:** Euclid's infinitude of primes; the Unix pipe.
- **confounder:** Brevity / minimalism-signaling.

## latent_theoremhood
- **def:** A has more than B when more of A's prose argument would survive translation into a formal system with conclusions intact — the essay is secretly a theorem.
- **smell:** The judge must attempt the formalization internally and detect exactly where hand-waving would fail to compile; small models read notation as formality.
- **top:** An informal essay on mechanism design or scheduling that a Lean formalization would vindicate nearly verbatim.
- **confounder:** Presence of equations and formal-sounding jargon.

## negative_space_discipline
- **def:** A has more than B when A's claims stop more precisely at the boundary of its evidence — it visibly knows what it doesn't know, without hedging what it does.
- **smell:** Requires independently mapping the true validity region of each claim and comparing it to the claimed region; small models count hedge words.
- **top:** An experimental paper whose limitations section predicts exactly the later failed replications — and nothing more.
- **confounder:** Hedge frequency / "epistemic status" boilerplate.

## conceptual_arbitrage_depth
- **def:** A has more than B when A imports a structure from a domain where it is well-understood into one where it is expensive, and the mapping preserves more relations rather than surface metaphor.
- **smell:** Verifying relation-preservation demands genuine simultaneous fluency in both domains; small models score interdisciplinary name-dropping.
- **top:** Thermodynamic formalism carried into information (Shannon's entropy); auction theory carried into spectrum allocation.
- **confounder:** Analogy count / cross-domain vocabulary.

## deconfusion_yield
- **def:** A has more than B when reading A permanently dissolves more confusions — questions that felt live before become visibly ill-posed after.
- **smell:** The judge must model the reader's prior confusion and verify the dissolution is real rather than rhetorical anesthesia; small models detect definitional pedantry.
- **top:** The paper after which "is it a wave or a particle" stops being a question anyone competent asks.
- **confounder:** Definition density / taxonomy-building.

## joint_carving
- **def:** A has more than B when A's central distinction re-partitions more existing cases in a way subsequent evidence keeps respecting — the categories keep paying rent.
- **smell:** Requires projecting the distinction across many held-out cases and checking it still cuts cleanly; small models score coinage catchiness.
- **top:** The artifact that introduced a now-load-bearing dichotomy — exploration/exploitation, Type I/Type II error.
- **confounder:** Memorable naming flair / neologism count.

## worlds_forbidden
- **def:** A has more than B when A's claims rule out more otherwise-plausible future observations — higher Popperian content, more neck exposed.
- **smell:** Computing what a claim *forbids* requires holding the space of plausible worlds and intersecting it with the claim's logical shadow; small models count numeric predictions and bold adverbs.
- **top:** A theory paper predicting a precise, surprising, later-confirmed value — general relativity on light bending.
- **confounder:** Bold tone / explicit forecasts regardless of how little they constrain.

## hostile_paraphrase_invariance
- **def:** A has more than B when a maximally uncharitable but honest restatement of A loses less of its force — the insight survives being stripped of its rhetoric.
- **smell:** The judge must actually construct the hostile paraphrase and re-evaluate the residue; small models conflate plain style with robustness.
- **top:** An impossibility result opponents state accurately in their own words and still cannot defeat — Arrow's theorem.
- **confounder:** Dry / plain prose style.

## regeneration_potential
- **def:** A has more than B when, if all other knowledge in its domain were erased, competent successors could regrow more of the field from A alone.
- **smell:** Requires simulating the rederivation tree from the artifact's actual content — what it *enables*, not what it mentions; small models reward encyclopedic coverage.
- **top:** Feynman's "everything is made of atoms" sentence at scale — a compact axiomatization like Maxwell's equations or Peano's postulates.
- **confounder:** Comprehensiveness / textbook length.

## premature_arrival
- **def:** A has more than B when the gap between A and the prerequisites its milieu actually offered is larger — the artifact had to invent more of its own scaffolding to exist at all.
- **smell:** Needs a dated map of the tools and concepts available to the author versus those the artifact uses; small models proxy to sheer age or lone-genius mythology.
- **top:** Mendel's genetics; Babbage's analytical engine; Ramanujan's notebooks.
- **confounder:** Artifact age / romantic isolation narrative.

## bootstrap_depth
- **def:** A has more than B when A rests less on authority and rebuilds more of its conclusions from primitives the reader can verify directly.
- **smell:** Auditing each inferential step for hidden appeals to authority requires re-deriving the chain, not pattern-matching the bibliography; small models reward "first principles" phrasing and citation sparseness.
- **top:** A systems paper that re-measures every folklore number it depends on instead of citing it.
- **confounder:** Low citation count / "from first principles" framing.
