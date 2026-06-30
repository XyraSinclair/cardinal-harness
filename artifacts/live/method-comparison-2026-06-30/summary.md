# Live Structured-Judgment Method Comparison

This receipt is generated from real OpenRouter chat-completion calls.
It compares structured judgment regimes on the same attribute-weighted item lists.
The reference is another live LLM regime, not human ground truth.

Candidate model: `openai/gpt-4.1-mini`
Reference model: `anthropic/claude-sonnet-4.6`
Cases: 3
OpenRouter calls: 276
Prompt tokens: 57797
Completion tokens: 25061
Provider cost: $0.395864
Any estimated cost rows: False

| Case | Method | Model | Kendall tau vs reference | Top-k Jaccard | Cost USD | Ranking |
|---|---|---|---:|---:|---:|---|
| `public_artifact_work` | `scalar_matrix` | `openai/gpt-4.1-mini` | 0.600 | 0.500 | $0.000749 | live_baseline_suite, large_frozen_benchmark, portable_receipts, first_user_path, dependency_audit_cleanup |
| `public_artifact_work` | `list_sort` | `openai/gpt-4.1-mini` | 0.600 | 0.500 | $0.000230 | live_baseline_suite, large_frozen_benchmark, portable_receipts, first_user_path, dependency_audit_cleanup |
| `public_artifact_work` | `ordinal_pairwise` | `openai/gpt-4.1-mini` | 0.600 | 0.500 | $0.002941 | live_baseline_suite, large_frozen_benchmark, portable_receipts, first_user_path, dependency_audit_cleanup |
| `public_artifact_work` | `cardinal_pairwise_ratio` | `openai/gpt-4.1-mini` | 0.600 | 0.500 | $0.003530 | live_baseline_suite, large_frozen_benchmark, portable_receipts, first_user_path, dependency_audit_cleanup |
| `judgment_method_properties` | `scalar_matrix` | `openai/gpt-4.1-mini` | 0.600 | 1.000 | $0.000706 | bandit_policy_cardinal, cardinal_pairwise_ratio, ordinal_pairwise, whole_list_sort, single_scalar_rating |
| `judgment_method_properties` | `list_sort` | `openai/gpt-4.1-mini` | 0.600 | 1.000 | $0.000211 | bandit_policy_cardinal, cardinal_pairwise_ratio, ordinal_pairwise, whole_list_sort, single_scalar_rating |
| `judgment_method_properties` | `ordinal_pairwise` | `openai/gpt-4.1-mini` | 0.400 | 1.000 | $0.002700 | bandit_policy_cardinal, cardinal_pairwise_ratio, ordinal_pairwise, single_scalar_rating, whole_list_sort |
| `judgment_method_properties` | `cardinal_pairwise_ratio` | `openai/gpt-4.1-mini` | 0.600 | 1.000 | $0.003261 | bandit_policy_cardinal, cardinal_pairwise_ratio, ordinal_pairwise, whole_list_sort, single_scalar_rating |
| `model_policy_options` | `scalar_matrix` | `openai/gpt-4.1-mini` | 0.200 | 0.500 | $0.000781 | sonnet_reference_then_fast, glm_5_2_logprob_candidate, kimi_k2_thinking_diverse, quality_only_opus_46, deepseek_v4_flash_fast |
| `model_policy_options` | `list_sort` | `openai/gpt-4.1-mini` | 0.600 | 1.000 | $0.000246 | sonnet_reference_then_fast, kimi_k2_thinking_diverse, quality_only_opus_46, glm_5_2_logprob_candidate, deepseek_v4_flash_fast |
| `model_policy_options` | `ordinal_pairwise` | `openai/gpt-4.1-mini` | -0.200 | 0.500 | $0.002887 | deepseek_v4_flash_fast, kimi_k2_thinking_diverse, quality_only_opus_46, sonnet_reference_then_fast, glm_5_2_logprob_candidate |
| `model_policy_options` | `cardinal_pairwise_ratio` | `openai/gpt-4.1-mini` | -0.800 | 0.200 | $0.003451 | glm_5_2_logprob_candidate, kimi_k2_thinking_diverse, deepseek_v4_flash_fast, quality_only_opus_46, sonnet_reference_then_fast |

## Interpretation guardrails

- These are real provider calls, but the reference is still an LLM reference, not an external oracle.
- High agreement with the reference means a regime recovered this live reference ordering on this task family and budget.
- Low agreement is useful evidence of prompt/regime brittleness, not proof that the model is incapable.
- Cost comparisons are only headline-safe when `cost_is_estimate` is false for every row being compared.
