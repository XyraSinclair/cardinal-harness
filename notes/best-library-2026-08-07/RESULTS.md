# Results: weak-judge amplification + cardinal recovery (2026-08-07)

## Errata / verdicts up top

- **H1 (weak-judge amplification): REFUTED on this domain.** Both judges'
  direct estimates are perfect (Spearman 1.000, 253/253 concordant-ish rank
  pairs) — country populations are memorized, so there is no weak-judge
  regime here for pairwise aggregation to rescue. M4 cardinal scored 0.947
  (mini) / 0.892 (nano), below M1 and M3 in all four comparisons.
- **H2 (parity with best baseline under mini): REFUTED by 0.003.** Gap to
  best baseline is 0.053 Spearman; the registered bound was 0.05.
- **H3 (cardinal recovery r ≥ 0.90, mini): technically supported**
  (r = 0.936) **but hollow here** — M1's ln(estimate) achieves r = 1.0000,
  slope 0.995. On memorized quantities, direct estimation IS the superior
  cardinal instrument, at ~1/4 the calls and ~1/3 the cost.

One pre-run amendment (gap floor 1.34) and one mid-run erratum (corpus
wrapper mis-parse, $0.14 discarded run) are recorded in DESIGN.md.

## Numbers (N=23 items, 253 rank pairs per cell)

| Cell | Spearman | Kendall | calib r (vs ln truth) | slope |
|---|---|---|---|---|
| mini M1 direct estimate | **1.000** | 1.000 | **1.0000** | 0.995 |
| mini M2 score 0-100 | 0.936 | 0.798 | 0.887 | (8/23 at scale edges) |
| mini M3 listwise one-shot | 0.998 | 0.984 | — | — |
| mini M4 cardinal (92 cmp, $0.034) | 0.947 | 0.850 | 0.936 | 0.520 |
| nano M1 direct estimate | **1.000** | 1.000 | 0.9999 | 0.987 |
| nano M2 score 0-100 | 0.791 | 0.648 | 0.814 | (6/23 at edges) |
| nano M3 listwise one-shot | 0.999 | 0.992 | — | — |
| nano M4 cardinal (89 cmp, 3 refused, $0.009) | 0.892 | 0.763 | 0.908 | 0.269 |

## What the refutation teaches (the actual findings)

1. **Ceiling domains cannot discriminate sorting methods.** Famous-entity
   numeric attributes are parametric recall for every method; the pairwise
   machinery can only add noise there. Any "best library" benchmark must
   use domains where pointwise judgment is unreliable — composite or
   subjective attributes, non-memorized quantities, cross-document
   tradeoffs (the product's actual arXiv use case).
2. **Ratio-ladder magnitude compression is real and measured.** The latent
   scale recovered 0.52 (mini) / 0.27 (nano) of the true log-span. The
   prompt ladder's bounded ratio vocabulary cannot express a 145,000×
   spread; per-comparison saturation compresses the tails. Engineering
   handle: ladder extension or log-anchored prompts for wide-span
   attributes; document the span limit in the sort contract.
3. **Position bias on this prompt/domain is large.** Order flips 12/43
   (mini) and 12/42 (nano); systematic order energy 1.319 nats/pair (mini)
   — an order of magnitude above yesterday's arXiv run (0.192). The
   diagnostic caught it, which is the instrument working, but the
   comparison prompt is measurably position-sensitive for
   short-famous-entity comparisons.
4. **M2 (0-100 scoring), the common-practice baseline, is the worst method
   in both cells** — the compression everyone accepts by default is worse
   than either listwise or pairwise. Cardinal beats it by +0.011/+0.101
   Spearman. That is the honest floor we clear tonight, not "best in the
   world".

## What survives for the best-library claim

- The uncertainty/diagnostic surface: no baseline reports order flips,
  frustration, systematic-order energy, posterior rank risk. Tonight those
  diagnostics correctly flagged the weakest cell before truth was consulted.
- The scaling regime (lists ≫ context, attributes without memorized
  answers) was not exercised here — registered as out of scope, still open.

## Costs

Baselines $0.014, cardinal cells $0.043, discarded mis-parse run $0.145.
Total $0.202 of the $5.00 cap.

## Reproduce

```
cd notes/best-library-2026-08-07
vrun python3 baselines.py openai/gpt-5.4-mini
vrun python3 baselines.py openai/gpt-5.4-nano
vrun ../../target/debug/cardinal sort corpus.json --by "current national population" \
  --model openai/gpt-5.4-mini --seed 20260807 --no-cache --format json \
  --trace cardinal-mini-trace.jsonl > cardinal-mini.json
# same with openai/gpt-5.4-nano -> cardinal-nano.json
python3 analyze.py
```
