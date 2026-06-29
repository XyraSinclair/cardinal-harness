# Evaluation Evidence

This page is a checked-in receipt for the synthetic evaluation harness. It records what the current code does; it is not a benchmark claim about real LLM traffic.

Generated locally with:

```bash
cargo run --bin cardinal -- eval --out artifacts/eval/synthetic_eval.jsonl --curve-csv artifacts/eval/synthetic_curves.csv
cargo run --bin cardinal -- eval-likert --out artifacts/eval/likert_eval.jsonl --curve-csv artifacts/eval/likert_curves.csv
```

Raw outputs:

- `artifacts/eval/synthetic_eval.jsonl`
- `artifacts/eval/synthetic_curves.csv`
- `artifacts/eval/likert_eval.jsonl`
- `artifacts/eval/likert_curves.csv`

## Current results

| Case | Cardinal precision | Likert precision | Cardinal recall | Likert recall | Cardinal 95% coverage | Likert 95% coverage | Cardinal tau-b | Likert tau-b | Cardinal comparisons | Likert ratings | Cardinal stop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clean_ordering_10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 40 | 40 | budget_exhausted |
| noisy_ordering_50 | 0.800 | 0.600 | 0.800 | 0.600 | 1.000 | 0.800 | 0.799 | 0.735 | 200 | 200 | budget_exhausted |
| multi_attr_weighted_20 | 0.800 | 1.000 | 0.800 | 1.000 | 1.000 | 0.750 | 0.874 | 0.905 | 240 | 240 | budget_exhausted |
| clustered_scores_30 | 1.000 | 1.000 | 1.000 | 1.000 | 0.933 | 0.033 | 0.830 | 0.868 | 120 | 120 | budget_exhausted |
| outlier_robustness_25 | 0.200 | 0.800 | 0.200 | 0.800 | 0.880 | 0.360 | 0.540 | 0.830 | 100 | 100 | budget_exhausted |
| gated_feasibility_30 | 0.600 | 1.000 | 0.600 | 1.000 | 1.000 | 0.250 | 0.545 | 0.831 | 120 | 120 | budget_exhausted |
| inconsistent_cycle_12 | 0.800 | 0.800 | 0.800 | 0.800 | 1.000 | 0.167 | 0.545 | 0.848 | 200 | 200 | budget_exhausted |

## Read this table honestly

- Cardinal is perfect on `clean_ordering_10`, beats the Likert baseline on `noisy_ordering_50`, and keeps 95% coverage high in most synthetic cases.
- The Likert baseline still wins several synthetic top-k cases, especially `outlier_robustness_25`, `gated_feasibility_30`, and `inconsistent_cycle_12`. That is useful signal, not a failure to hide.
- Gated top-k precision/recall are gate-aware: predicted top-k over predicted-feasible items is compared against true top-k over true-feasible items. Gate false-negatives no longer shrink the target set and make the run look better than it is.
- All current cardinal synthetic runs stopped at `budget_exhausted`; the public claim should be framed as an uncertainty-aware active-comparison engine, not as a universally cheaper or strictly more accurate method.
- `outlier_robustness_25` is the current embarrassment case. Treat it as a regression target before making broad robustness claims.
- The harness records error trajectories. The next proof upgrade is a larger matrix varying item count, graph density, outlier rate, gate rate, and comparison budget.

## What this proves

The repo has a runnable, deterministic evaluation surface with raw artifacts. It does not yet prove superiority over scalar ratings. It does prove that claims can be falsified locally and that regressions can be caught with checked-in outputs.
