# Council answer — Claude (fable-5)

## 1. THE ONE OBJECT

A preference is a flow on the graph of alternatives; it deserves the name **belief** exactly when that flow is the gradient of a single scalar potential that is invariant under the group of transformations that shouldn't matter (order, wording, polarity, framing) — and this stack measures the flow, solves the potential, and prices everything the potential cannot explain (curl, cycle, susceptibility) in nats with error bars.

Physicist's version: it is a gauge-fixed potential reconstruction from noisy edge measurements, with the Hodge residual and the response to an applied field reported as first-class observables. Stranger's version: **it measures whether the model actually believes what it just told you, or was only echoing how you asked — and gives you the number.**

Everything in the repo is this one object seen from a different side: IRLS is the potential solve; the invariance probes are the group action; frustration is the non-gradient part; χ is dV under an applied field; the bench is the object turned on the judge; packets are the object made mergeable; portfolios are the object over correlated judges. There is no second idea here. That is a strength.

## 2. THE ONE TOOL

**`cardinal believe`** — one verb, one question, one verdict.

Input: a question with two alternatives (or a pair from a list + criterion) and a judge model. Output: a verdict line — **BELIEF / WEAK BELIEF / ECHO** — with the potential (direction and magnitude with an interval), which transformations it survived, which one broke it and by how many nats, the dollar cost, and a single **self-contained HTML receipt file** (raw provider bytes embedded, replayable offline from the cache). No server, no auth, no DB: the receipt is a file you can `open`, commit, or send. Git/file-native by construction; every number recomputable — this is the existing engine (`judge --spin --sweep --wordings`, counterbalance, two-sided, calibrate) fused into one battery behind one word, plus the receipt-viewer from differentiation §3 as the output format instead of stderr.

Why this and not `sort`: sorting is the utility; *believing* is the essence. Nothing else on the internet answers "does the model mean it?" with a receipted number, and the question is one every stranger already has.

## 3. WHAT TO DELETE

- **The probe flags on `sort`.** `--two-sided`, `--also-by`, `--spin`, `--no-counterbalance` as opt-in flags say "the probes are optional garnish." They are the product. Fold the battery into `believe`; `sort` keeps counterbalance-by-default and one `--audit` that runs the rest. ~20 flags → ~6.
- **The subcommand sprawl.** `elaborate`, `explain`, `policy`, `experiment-expand`, `report`, `distinguish` — fold or kill. `eval`/`eval-likert`/`eval-compare` → `bench`. `calibrate` → a battery stage inside `believe`/`bench`, not a citizen. Target surface: `sort`, `believe`, `bench`, `weigh`, `cache`. Five verbs.
- **The physics nouns as public vocabulary.** Keep Hodge/curl/χ where the math is literal (per §4 of PRINCIPLES) — in code, internal docs, and the receipt's fine print. But "susceptibility," "paramagnetic judgement," "frustration," "hysteresis" on the outer surface make a stranger decode a metaphor before feeling the finding. Outer nouns: **belief, echo, survives, breaks, moves with the asker.** The nats stay; the spin-glass costume goes.
- **FIRST_PRINCIPLES.md as the landing page it has become.** It is a permutation audit for maintainers — correct, and the wrong front door. The arXiv-grade findings buried in it (χ=+0.64 on a zero-field pair; wording-calibrated magnitudes) move to the front; the grid stays as the maintainer appendix it is.
- **The sheaf vocabulary and the "taste vector / seven ideas" branding fan-out** — already adjudicated (#46 descope); finish the deletion. One object, one verb, one noun ("belief"). Seven primitives → six per the deferred parsimony finding, gated as stated.
- **Do not delete:** receipts culture, denominators, the ✗ rows. The honest absences are load-bearing.

## 4. THE INEVITABLE INTERFACE

```console
$ cardinal believe --judge gpt-5.4-mini \
    "Better advice for a new engineer: 'move fast and break things' vs 'measure twice, cut once'?"
```

**Second 10** — the neutral question, both orders, streaming as it lands:

```
neutral · A first : B 1.6×
neutral · B first : B 1.7×      order survives (residual 0.04 nats)
```

**Second 30** — the attack rounds, visibly adversarial:

```
opposite polarity : consistent ✓
paraphrase        : consistent ✓
pro-A framing     : B 1.1×   ← moved
pro-B framing     : B 2.8×   ← moved      answer follows the asker: 0.48 nats
```

**Second 60** — the verdict block and the artifact:

```
VERDICT: WEAK BELIEF
  direction : "measure twice" by 1.7× [1.2, 2.4]
  survives  : order ✓  polarity ✓  paraphrase ✓
  breaks    : framing — asker moves the answer ~1.6×
  14 calls · $0.006 · receipt: believe-8f3a2c.html (replayable offline)
```

Opening the HTML: the two items, the neutral judgement, and **a framing slider** — drag from "pro-A asker" to "pro-B asker" and watch the stated preference move in real time, every point clickable down to raw provider bytes. On a contested pair it shows the already-receipted killer frame: zero-field preference **exactly 0.000**, sliding to ±0.64 with the asker. The stranger doesn't read a nats table; they *watch the model's opinion get pushed around*, then watch a real belief refuse to move. That contrast — one judgement that bends, one that doesn't, same model, sixty seconds, six tenths of a cent — is the whole thesis felt in the body.

## 5. THE VON NEUMANN TEST

**Pre-registered held-out invariance: the verdict must predict transformations it never measured.** Take 20 pairs. Classify each as BELIEF or ECHO using battery A (order, polarity, spin). Then test with a *disjoint* battery B the classifier never saw: fresh procedurally-rotated paraphrases, the fraction/times-less wording inversion, and re-ask after 3+ days (the ✗ time row becomes the validation set — the cheapest untouched science in the repo, per your own grid). Pre-register the bar: BELIEF pairs hold direction on battery B ≥ 90%; ECHO pairs land near coin-flip. Cost: roughly $1–2 at receipted per-call prices, plus three days of clock time.

This can fail three visible ways, each fatal to a specific pretension: if BELIEF pairs don't survive held-out transformations, the invariance group is decoration — the battery measures its own questions, not a property of the judgement; if ECHO pairs *do* survive, the battery is oversensitive and "echo" is slander; if the two classes are indistinguishable on battery B, the one object in §1 is a story, not a structure. If it passes, the verdict noun is earned: "belief" would then mean *measured on axes 1–3, predictive on axes 4–6*, which is what a physical property is.

Run it before shipping the verb. A `believe` command whose verdict has never been tested out-of-group is exactly the un-red-teamed countermeasure PRINCIPLES §7 forbids.
