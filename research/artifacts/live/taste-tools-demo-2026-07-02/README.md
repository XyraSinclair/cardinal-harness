# Live taste-tooling demo receipt — 2026-07-02

Real, unedited runs of the three taste-building commands against
`anthropic/claude-sonnet-4.6`, forming one loop: **elaborate** a terse
criterion into a rubric → **judge** one pair under it, fully transparent →
**explain** an existing ranking by testing candidate attributes against it.

## 1. `cardinal elaborate` — $0.0034

```bash
cardinal elaborate --by "usefulness as advice for a software engineer" \
  --model anthropic/claude-sonnet-4.6
```

Produced `elaborated-rubric.txt`: a one-paragraph judging rubric with a crisp
definition, what counts as more (specificity, applicability, accuracy), and
what must NOT be rewarded (motivational content, mere informativeness).

## 2. `cardinal judge` — $0.0057

```bash
cardinal judge "measure twice, cut once" \
  "premature optimization is the root of all evil" \
  --by "$(cat elaborated-rubric.txt)" \
  --model anthropic/claude-sonnet-4.6 --no-cache --json
```

`judge-output.json`: B wins, ratio 1.75, confidence 0.72 — consistent with
the full sort's ordering of the same two items.

## 3. `cardinal explain` — $0.4464, 200 comparisons, 11/82 order flips

`believed-ranking.txt` is the output order of the preserved live sort
(`artifacts/live/sort-demo-2026-07-02/`), i.e. a ranking whose true
generating attribute is known. We asked explain to test the true attribute
plus 3 LLM-proposed candidates:

```text
attribute                                    | alone ρ | weight
---------------------------------------------|---------|-------
usefulness as advice for a software engineer |   +0.98 | 0.85
relevance to software engineering principles |   +0.79 | 0.01
encourages proactive careful planning        |   +0.21 | 0.00
wisdom applicable to technical decision-maki |   +0.81 | 0.15

weighted combination reconstructs your ranking at ρ = +0.98
```

**The receipt shows attribute recovery**: the criterion that actually
generated the ranking is identified — highest standalone correlation AND
dominant fitted weight — while three plausible-sounding decoys are correctly
down-weighted. Caveats, honestly: this is one run, on one list, where the
reference ranking was itself produced by the same judge family; treat it as a
demonstration of the mechanism, not a benchmark.

## Files

| File | Contents |
|------|----------|
| `elaborated-rubric.txt` | The generated judging rubric, verbatim |
| `elaborate-stderr.txt` | Elaboration usage/cost line |
| `judge-output.json` | Structured single-judgement receipt |
| `believed-ranking.txt` | The reference ranking given to explain (best first) |
| `explain-output.txt` | The attribute table exactly as printed |
| `explain-stderr.txt` | Proposal list + run summary (comparisons, cost, flips) |
| `cache-export.jsonl` | Cached pairwise judgements from the explain run |
