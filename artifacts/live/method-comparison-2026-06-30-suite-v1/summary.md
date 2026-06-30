# Live Structured-Judgment Method Comparison

This receipt is generated from real OpenRouter chat-completion calls.
It compares structured judgment regimes on the same frozen attribute-weighted item lists.
The reference is another live LLM regime, not human ground truth.

Suite: `live_method_suite_v1`
Suite path: `examples/live-method-suite.json`
Candidate model: `openai/gpt-5.4-mini`
Reference model: `anthropic/claude-sonnet-4.6`
Cases: 6
OpenRouter calls: 552
Prompt tokens: 111282
Completion tokens: 50809
Total tokens: 162091
Provider cost: $0.806604
Any estimated cost rows: False

| Case | Method | Model | Kendall tau vs reference | Top-k Jaccard | Cost USD | Ranking |
|---|---|---|---:|---:|---:|---|
| `public_artifact_work` | `scalar_matrix` | `openai/gpt-5.4-mini` | 0.600 | 0.500 | $0.001604 | live_baseline_suite, large_frozen_benchmark, portable_receipts, first_user_path, dependency_audit_cleanup |
| `public_artifact_work` | `list_sort` | `openai/gpt-5.4-mini` | 0.800 | 0.500 | $0.000523 | large_frozen_benchmark, live_baseline_suite, portable_receipts, first_user_path, dependency_audit_cleanup |
| `public_artifact_work` | `ordinal_pairwise` | `openai/gpt-5.4-mini` | 0.400 | 0.500 | $0.007697 | portable_receipts, large_frozen_benchmark, live_baseline_suite, first_user_path, dependency_audit_cleanup |
| `public_artifact_work` | `cardinal_pairwise_ratio` | `openai/gpt-5.4-mini` | 0.800 | 0.500 | $0.009078 | large_frozen_benchmark, live_baseline_suite, portable_receipts, first_user_path, dependency_audit_cleanup |
| `judgment_method_properties` | `scalar_matrix` | `openai/gpt-5.4-mini` | 0.400 | 0.500 | $0.001500 | single_scalar_rating, bandit_policy_cardinal, ordinal_pairwise, cardinal_pairwise_ratio, whole_list_sort |
| `judgment_method_properties` | `list_sort` | `openai/gpt-5.4-mini` | 0.800 | 1.000 | $0.000487 | bandit_policy_cardinal, cardinal_pairwise_ratio, ordinal_pairwise, single_scalar_rating, whole_list_sort |
| `judgment_method_properties` | `ordinal_pairwise` | `openai/gpt-5.4-mini` | 0.600 | 1.000 | $0.007200 | bandit_policy_cardinal, cardinal_pairwise_ratio, ordinal_pairwise, whole_list_sort, single_scalar_rating |
| `judgment_method_properties` | `cardinal_pairwise_ratio` | `openai/gpt-5.4-mini` | 0.600 | 1.000 | $0.008559 | bandit_policy_cardinal, cardinal_pairwise_ratio, ordinal_pairwise, whole_list_sort, single_scalar_rating |
| `model_policy_options` | `scalar_matrix` | `openai/gpt-5.4-mini` | 0.000 | 0.500 | $0.001698 | sonnet_reference_then_fast, glm_5_2_logprob_candidate, kimi_k2_thinking_diverse, deepseek_v4_flash_fast, quality_only_opus_46 |
| `model_policy_options` | `list_sort` | `openai/gpt-5.4-mini` | 0.800 | 1.000 | $0.000572 | sonnet_reference_then_fast, quality_only_opus_46, kimi_k2_thinking_diverse, glm_5_2_logprob_candidate, deepseek_v4_flash_fast |
| `model_policy_options` | `ordinal_pairwise` | `openai/gpt-5.4-mini` | 0.000 | 0.500 | $0.007664 | sonnet_reference_then_fast, glm_5_2_logprob_candidate, kimi_k2_thinking_diverse, deepseek_v4_flash_fast, quality_only_opus_46 |
| `model_policy_options` | `cardinal_pairwise_ratio` | `openai/gpt-5.4-mini` | -0.200 | 0.500 | $0.009090 | kimi_k2_thinking_diverse, glm_5_2_logprob_candidate, sonnet_reference_then_fast, deepseek_v4_flash_fast, quality_only_opus_46 |
| `first_user_path` | `scalar_matrix` | `openai/gpt-5.4-mini` | 0.600 | 0.500 | $0.001579 | copy_paste_quickstart, public_api_recipe, when_not_to_use_cardinal, worked_artifact_walkthrough, receipt_debugging_guide |
| `first_user_path` | `list_sort` | `openai/gpt-5.4-mini` | 0.800 | 0.500 | $0.000521 | copy_paste_quickstart, when_not_to_use_cardinal, public_api_recipe, worked_artifact_walkthrough, receipt_debugging_guide |
| `first_user_path` | `ordinal_pairwise` | `openai/gpt-5.4-mini` | -0.600 | 0.200 | $0.007088 | public_api_recipe, receipt_debugging_guide, when_not_to_use_cardinal, worked_artifact_walkthrough, copy_paste_quickstart |
| `first_user_path` | `cardinal_pairwise_ratio` | `openai/gpt-5.4-mini` | 0.000 | 0.200 | $0.008726 | public_api_recipe, copy_paste_quickstart, receipt_debugging_guide, when_not_to_use_cardinal, worked_artifact_walkthrough |
| `benchmark_design_rigor` | `scalar_matrix` | `openai/gpt-5.4-mini` | 0.000 | 0.500 | $0.001571 | pre_registered_cases, equal_dollar_budget, repeated_model_swaps, failure_mode_taxonomy, human_adjudicated_slice |
| `benchmark_design_rigor` | `list_sort` | `openai/gpt-5.4-mini` | 0.600 | 0.500 | $0.000490 | pre_registered_cases, failure_mode_taxonomy, human_adjudicated_slice, repeated_model_swaps, equal_dollar_budget |
| `benchmark_design_rigor` | `ordinal_pairwise` | `openai/gpt-5.4-mini` | 0.400 | 0.500 | $0.007070 | pre_registered_cases, failure_mode_taxonomy, human_adjudicated_slice, equal_dollar_budget, repeated_model_swaps |
| `benchmark_design_rigor` | `cardinal_pairwise_ratio` | `openai/gpt-5.4-mini` | 0.600 | 0.500 | $0.008563 | pre_registered_cases, repeated_model_swaps, failure_mode_taxonomy, human_adjudicated_slice, equal_dollar_budget |
| `public_release_risks` | `scalar_matrix` | `openai/gpt-5.4-mini` | 1.000 | 1.000 | $0.001470 | unportable_artifacts, overclaiming_evidence, opaque_failure_cases, first_run_complexity, security_advisory_noise |
| `public_release_risks` | `list_sort` | `openai/gpt-5.4-mini` | 0.800 | 1.000 | $0.000479 | overclaiming_evidence, unportable_artifacts, opaque_failure_cases, first_run_complexity, security_advisory_noise |
| `public_release_risks` | `ordinal_pairwise` | `openai/gpt-5.4-mini` | 0.200 | 0.500 | $0.006901 | opaque_failure_cases, overclaiming_evidence, first_run_complexity, unportable_artifacts, security_advisory_noise |
| `public_release_risks` | `cardinal_pairwise_ratio` | `openai/gpt-5.4-mini` | 0.800 | 1.000 | $0.008364 | overclaiming_evidence, unportable_artifacts, opaque_failure_cases, first_run_complexity, security_advisory_noise |

## Budget-normalized aggregate

`agreement_score` averages normalized Kendall tau `((tau + 1) / 2)` and top-k Jaccard.
The per-call/token/dollar columns are descriptive normalizations of this run, not a substitute for a separately equalized-budget experiment.

| Method | Mean tau | Mean top-k | Mean agreement | Calls | Total tokens | Cost USD | Agreement/call | Agreement/1k tokens | Agreement/USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reference_pairwise_ratio` | 1.000 | 1.000 | 1.000 | 180 | 77600 | $0.698112 | 0.0333 | 0.0773 | 8.59 |
| `list_sort` | 0.767 | 0.750 | 0.817 | 6 | 2511 | $0.003072 | 0.8167 | 1.9514 | 1595.05 |
| `cardinal_pairwise_ratio` | 0.433 | 0.617 | 0.667 | 180 | 41695 | $0.052380 | 0.0222 | 0.0959 | 76.37 |
| `scalar_matrix` | 0.433 | 0.583 | 0.650 | 6 | 3987 | $0.009421 | 0.6500 | 0.9782 | 413.95 |
| `ordinal_pairwise` | 0.167 | 0.533 | 0.558 | 180 | 36298 | $0.043618 | 0.0186 | 0.0923 | 76.80 |

## Interpretation guardrails

- These are real provider calls, but the reference is still an LLM reference, not an external oracle.
- High agreement with the reference means a regime recovered this live reference ordering on this task family and budget.
- Low agreement is useful evidence of prompt/regime brittleness, not proof that the model is incapable.
- Budget-normalized columns are useful for comparing this run's efficiency; they do not prove equal-token or equal-dollar dominance.
- Cost comparisons are only headline-safe when `cost_is_estimate` is false for every row being compared.
