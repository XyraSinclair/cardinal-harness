# Live evidence-path receipt: ratio_letter_v1 vs canonical_v2 — 2026-07-04

Same model (`openai/gpt-5.4-mini`), same list (`examples/sort-demo.txt`),
same criterion, same seed, same budget (64), near-identical cost
($0.0228 vs $0.0209). The only difference: the elicitation instrument.

| | `ratio_letter_v1` (seriate PMF path) | `canonical_v2` (JSON point path) |
|---|---|---|
| Judgement unit | one answer token; top-k logprobs = full PMF | one sampled JSON `(ratio, confidence)` |
| Solver weight | measured PMF variance (explicit precision) | stated confidence via `g(c)` |
| Evidence receipts | 63/63 logprob-mode, visible mass 0.99 | — |
| Posterior std (per item) | ±0.013–0.014 | ±0.41–0.45 |
| Top-vs-bottom separation | **≈ 4.0σ** | ≈ 1.4σ |
| Order flips (counterbalance) | 13/27 | 10/27 |
| Stop | budget_exhausted | budget_exhausted |

## Reading it honestly

- **The headline is separation per dollar, not raw std.** The two latent
  scales differ (the PMF path's expected log-ratios are small because the
  model spreads mass over near-parity buckets; the JSON path samples single
  larger ratios), so compare gaps in sigma units: ≈4.0σ vs ≈1.4σ top-to-
  bottom at the same budget — roughly 3× the resolving power per dollar.
  This is the predicted effect of consuming the model's full prior per call
  instead of one draw from it, on a provider whose logprobs are real
  (gpt-5.4-mini: JSD 0.128 in seriate's reality map).
- **The instruments do not induce identical orderings** (Spearman 0.738):
  e.g. "premature optimization…" ranks #2 under the letter instrument and
  #5 under JSON. Different elicitations tap different priors; neither is
  ground truth. This is a finding, not a defect — and exactly the kind of
  thing only receipts surface.
- One refusal in letter mode (`!` token), counted, not hidden.
- Both runs still stop at `budget_exhausted`; the tighter posterior means
  the letter path is far closer to certifiable stops at equal spend.

## Reproduce

```bash
cardinal sort examples/sort-demo.txt \
  --by "usefulness as advice for a software engineer" \
  --model openai/gpt-5.4-mini --template ratio_letter_v1 \
  --seed 7 --budget 64 --scores
```

Files: `letter-sorted.txt`, `json-sorted.txt` (stdout with scores),
`letter-stderr.txt`, `json-stderr.txt` (receipt lines), caches gitignored.
