# Evaluation Evidence

This page is the checked-in receipt for the synthetic evaluation harness. It records what the current code and deterministic simulator produce. It is not a benchmark claim about live LLM traffic, and it does not show a universal win over scalar ratings.

## Reproduce the receipt

Run from the repository root:

```bash
mkdir -p artifacts/eval

cargo run --bin cardinal -- eval \
  --out artifacts/eval/synthetic_eval.jsonl \
  --curve-csv artifacts/eval/synthetic_curves.csv

cargo run --bin cardinal -- eval-likert \
  --out artifacts/eval/likert_eval.jsonl \
  --curve-csv artifacts/eval/likert_curves.csv

cargo run --bin cardinal -- eval-compare \
  --mode ratio \
  --out artifacts/eval/comparison_summary.json
```

Checked-in outputs:

- `artifacts/eval/synthetic_eval.jsonl`: raw cardinal pairwise synthetic results, one JSON object per case.
- `artifacts/eval/likert_eval.jsonl`: raw Likert/scalar baseline results, one JSON object per case.
- `artifacts/eval/comparison_summary.json`: compact cardinal-vs-Likert deltas and win/loss/tie counts.
- Optional control: `cargo run --bin cardinal -- eval-compare --mode ordinal --out artifacts/eval/comparison_summary_ordinal.json` compares the same active-comparison loop using ordinal "which item is higher?" judgements instead of ratio magnitudes.
- `artifacts/eval/synthetic_curves.csv`: cardinal trajectory receipt.
- `artifacts/eval/likert_curves.csv`: Likert trajectory receipt.

Do not compare the two curve CSV `error` columns directly. `synthetic_curves.csv` records the cardinal model's estimated top-k boundary error and can exceed 1. `likert_curves.csv` records observed `1 - topk_precision`. They are trajectory receipts with different semantics, not a shared y-axis.

## Method

The suite runs the same synthetic cases through two deterministic evaluators:

1. `eval` uses the cardinal pairwise-ratio path. Synthetic pairwise judgements feed the reranker, which fits latent scores, estimates uncertainty, and records the final rank metrics.
2. `eval-likert` uses a scalar baseline. It spends the same number of model-call slots as the cardinal run, samples per-item ratings on a 10-point scale, infers utility scores from those ratings, and records the same headline rank metrics where possible.
3. `eval-compare` runs both suites with `levels = 10` and `budget_multiplier = 1.0`, then emits mechanical deltas in `comparison_summary.json`.

The current comparison gives equal call counts to cardinal comparisons and Likert ratings. That is a simple reproducible regime, not a claim about equal token cost: pairwise prompts contain two items, while scalar prompts rate one item.

## Metrics

`comparison_summary.json` compares five metrics for each case:

| Metric | Meaning | Better direction |
|---|---|---|
| `topk_precision` | Fraction of predicted top-k items that are truly in the top-k set. | Higher |
| `topk_recall` | Fraction of true top-k items recovered by the predicted top-k set. | Higher |
| `kendall_tau_b` | Pairwise rank-order agreement with ground truth, tie-aware. | Higher |
| `coverage_95ci` | Fraction of true scores covered by the model's reported 95% intervals. | Higher |
| `comparisons_used` | Cardinal comparisons used versus Likert ratings used. | Lower |

For quality metrics, `delta` means cardinal minus Likert. For `comparisons_used`, `delta` is still cardinal minus Likert, but `higher_is_better` is `false` and the `outcome` field applies the lower-is-better direction.

## Current cardinal-vs-Likert receipt

Source: `artifacts/eval/comparison_summary.json`.

| Case | Cardinal top-k P/R | Likert top-k P/R | Cardinal tau-b | Likert tau-b | Cardinal 95% cov. | Likert 95% cov. | Calls | Case W/L/T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_ordering_10` | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 40 / 40 | 0 / 0 / 5 |
| `noisy_ordering_50` | 0.600 / 0.600 | 0.600 / 0.600 | 0.641 | 0.735 | 1.000 | 0.800 | 200 / 200 | 1 / 1 / 3 |
| `multi_attr_weighted_20` | 0.800 / 0.800 | 1.000 / 1.000 | 0.895 | 0.905 | 1.000 | 0.750 | 240 / 240 | 1 / 3 / 1 |
| `clustered_scores_30` | 1.000 / 1.000 | 1.000 / 1.000 | 0.830 | 0.868 | 1.000 | 0.033 | 120 / 120 | 1 / 1 / 3 |
| `scale_compression_40` | 1.000 / 1.000 | 0.200 / 0.200 | 0.821 | 0.224 | 0.975 | 0.000 | 160 / 160 | 4 / 0 / 1 |
| `outlier_robustness_25` | 0.400 / 0.400 | 0.800 / 0.800 | 0.567 | 0.830 | 1.000 | 0.360 | 100 / 100 | 1 / 3 / 1 |
| `gated_feasibility_30` | 0.800 / 0.800 | 1.000 / 1.000 | 0.667 | 0.831 | 1.000 | 0.250 | 120 / 120 | 1 / 3 / 1 |
| `inconsistent_cycle_12` | 0.800 / 0.800 | 0.800 / 0.800 | 0.636 | 0.848 | 0.250 | 0.167 | 200 / 200 | 1 / 1 / 3 |

Aggregate over 8 cases and 5 compared metrics per case: cardinal wins 10, Likert wins 12, and 18 are ties.

## What the receipt says

- Cardinal clearly wins `scale_compression_40`: an extreme outlier collapses 10-level Likert ratings for the non-outlier frontier, while ratio comparisons recover the true top-k and a much stronger Kendall tau-b at the same call count.
- Cardinal ties Likert on top-k precision/recall for `clean_ordering_10`, `noisy_ordering_50`, `clustered_scores_30`, and `inconsistent_cycle_12`; it beats Likert on coverage in the latter three under this simulator.
- Likert wins important ranking metrics. It beats cardinal on top-k precision/recall and tau-b for `multi_attr_weighted_20`, `outlier_robustness_25`, and `gated_feasibility_30`, and it beats tau-b for `noisy_ordering_50`, `clustered_scores_30`, and `inconsistent_cycle_12`.
- No method wins on resource use in this receipt. Every case uses the same number of cardinal comparisons and Likert ratings, so `comparisons_used` is a tie in all eight cases. Equal call count is not equal token cost.
- Gated top-k precision and recall are gate-aware: predicted top-k over predicted-feasible items is compared with true top-k over true-feasible items. Gate false negatives do not shrink the target set and make a run look better than it is.
- All current cardinal synthetic runs stop at `budget_exhausted`. Do not describe the current artifact as proof of early stopping, lower cost, or strict accuracy dominance.

## What this proves

The repo has a deterministic local evaluation surface with checked-in raw artifacts, a compact comparison summary, and reproducible commands. It proves that claims can be checked and falsified locally. It does not prove that cardinal reranking is generally superior to Likert or scalar rating baselines.
