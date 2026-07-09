codex
**1. The One Object**

A **judgement field**: noisy pairwise ratio measurements form a 1-form on an entity graph; the exact part is belief, and the residual curl, framing response, and variance are the measured ways it fails to be belief.

Stranger translation: it tells whether a model, person, or panel actually has a stable preference, or is just echoing the way the question was asked.

**2. The One Tool**

Verb: `audit`

Input: one file containing items and one criterion.

Output: one static receipt page: `belief-audit.html`, plus JSONL receipts.

Example:

```bash
cardinal audit ideas.txt --by "most worth building next"
```

The page does not lead with “ranking.” It leads with: **Stable / Contested / Manipulable**.

That is the essence. Sorting is downstream. The primary question is: “is there a belief here?”

**3. What To Delete Or Fold**

Delete or demote:

- `sort` as the flagship verb. Keep it, but make it a subcase of `audit`.
- “LLM sorting” as the README headline. It undersells the object.
- Separate public emphasis on `weigh`, `distinguish`, `calibrate`, `judge --spin`, `--two-sided`, `--also-by`, `bench`. Fold them into audit modes or receipt sections.
- Physics vocabulary that is not immediately attached to a number on the page.
- “Susceptibility χ” as a front-door word. Show “framing moved this judgement by 0.64 nats.”
- “Taste vector” as the first concept. It is useful after the object lands.
- CLI flag archaeology. A stranger should not discover invariance probes through flags.

Keep the primitives internally. Externally, expose one question:

> Does this judgement survive transformations that should not matter?

**4. The Inevitable Interface**

Second 0:

```bash
cardinal audit ideas.txt --by "most likely to change how software is built"
```

Second 10:

Terminal prints:

```text
Belief audit complete.
Stable ordering: weak
Most contested pair: 0.03±0.41 nats
Framing moved it: +0.64 nats
Receipt page: artifacts/live/audit-2026-07-09/index.html
```

Second 30:

The page opens or is served locally. First screen:

```text
Is there a belief here?

Result: CONTESTED AND FRAME-SENSITIVE

Neutral judgement: A ≈ B
Pro-A framing: A > B by 0.67 nats
Pro-B framing: B > A by 0.61 nats
```

A slider moves between framings. The preference needle crosses zero.

Second 60:

They click “Why?” and see the evidence graph:

- the pairwise comparisons
- the order-flip residuals
- the spin response
- the solved score with uncertainty
- the raw receipt path and replay command

The feeling should be: “I just watched the model’s preference stop being a preference.”

**5. The Von Neumann Test**

Cheapest decisive demo:

Take 20 item pairs across 2 criteria. Before running spin/order/paraphrase probes, ask humans to predict which neutral judgements are stable and which are manipulable.

Then run:

```bash
cardinal audit pairs.json --by criteria.json --probe order,paraphrase,spin
```

It can fail visibly if:

- allegedly “stable” judgements flip under harmless transformations
- framing movement is indistinguishable from noise
- humans find the receipt page unsurprising or unactionable
- the audit does not predict retest stability better than neutral margin alone

The pass condition is not beauty. It is: the audit identifies which judgements deserve to be reused as beliefs, and which must be quarantined as presentation artifacts.
tokens used
