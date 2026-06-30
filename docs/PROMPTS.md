# Prompt Contract

`cardinal-harness` supports two prompt templates: `canonical_v2` and `canonical_bucket_v1`.

## Slugs

| Slug | Model output | Use when |
|------|--------------|----------|
| `canonical_v2` | Decimal `ratio` on the canonical ladder range | General pairwise-ratio judgement. This is the default when no `prompt_template_slug` is set. |
| `canonical_bucket_v1` | Integer `ratio_bucket` in `0..16` | Runs that need output-token logprobs mapped directly to the ratio ladder. The bucket index avoids reconstructing multi-token decimal probabilities. |

Unknown slugs are rejected. Omit `prompt_template_slug` only when you want the default `canonical_v2`.

## Ratio ladder

```text
[1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.1, 2.5, 3.1, 3.9, 5.1, 6.8, 9.2, 12.7, 18.0, 26.0]
```

The ladder is approximately geometric in log-space, with extra density near 1.0 for near-ties.

## Output shapes

`canonical_v2` successful judgement:

```json
{"higher_ranked":"A","ratio":2.1,"confidence":0.74}
```

`canonical_bucket_v1` successful judgement:

```json
{"higher_ranked":"A","ratio_bucket":7,"confidence":0.74}
```

In bucket mode, `ratio_bucket` is the zero-based index into the ratio ladder above. For example, bucket `7` means ratio `2.1`.

Refusal for either template:

```json
{"refused":true}
```

## Semantics

- `higher_ranked`: which side has more of the attribute
- `ratio`: how much more, constrained to the canonical ladder range; used by `canonical_v2`
- `ratio_bucket`: zero-based ratio ladder index; used by `canonical_bucket_v1`
- `confidence`: self-reported confidence in `[0, 1]`
- `refused`: explicit refusal channel for genuinely blocked cases

## Request examples

- Multi-attribute CLI request: [`../examples/multi-rerank-request.json`](../examples/multi-rerank-request.json)
- Simple single-attribute request shape for library/API callers: [`../examples/simple-rerank-request.json`](../examples/simple-rerank-request.json)
- Prompt/attribute variant specs for request expansion: [`../examples/prompt-experiment-variants.json`](../examples/prompt-experiment-variants.json)
- Model policy recipes: [`../examples/model-policy-quality-only.json`](../examples/model-policy-quality-only.json), [`../examples/model-policy-cost-aware-fast.json`](../examples/model-policy-cost-aware-fast.json), [`../examples/model-policy-frontier-ladder.json`](../examples/model-policy-frontier-ladder.json)

Run the multi-rerank example with:

```bash
export OPENROUTER_API_KEY=your_key_here
cargo run --bin cardinal -- rerank \
  --request examples/multi-rerank-request.json \
  --out output.json \
  --trace trace.jsonl \
  --report report.md
```

Use an explicit current model policy when you want reproducible routing:

```bash
# Quality-only frontier run.
cargo run --bin cardinal -- rerank \
  --request examples/multi-rerank-request.json \
  --policy-config examples/model-policy-quality-only.json \
  --out output.json \
  --trace trace.jsonl \
  --report report.md

# Cost-aware/fast run.
cargo run --bin cardinal -- rerank \
  --request examples/multi-rerank-request.json \
  --policy-config examples/model-policy-cost-aware-fast.json \
  --out output.json \
  --trace trace.jsonl \
  --report report.md

# Frontier ladder: start with Opus 4.6, step through Gemini 3.1 Pro preview,
# then use GPT-5.4 Mini for low-uncertainty near-tie checks.
cargo run --bin cardinal -- rerank \
  --request examples/multi-rerank-request.json \
  --policy-config examples/model-policy-frontier-ladder.json \
  --out output.json \
  --trace trace.jsonl \
  --report report.md
```

The checked-in policy files use live OpenRouter model IDs from the 2026-06 refresh: `anthropic/claude-opus-4.6`, `google/gemini-3.1-pro-preview`, `openai/gpt-5.4-mini`, and `deepseek/deepseek-v4-flash`.
If a model is newer than the local pricing table, reports use OpenRouter's provider-reported upstream cost when available; otherwise they label the local fallback cost as an estimate instead of pretending it is exact.


Generate a local prompt-surface experiment request without touching the network:

```bash
cargo run --bin cardinal -- experiment-expand \
  --request examples/multi-rerank-request.json \
  --prompt-template canonical_v2 \
  --prompt-template canonical_bucket_v1 \
  --include-negative \
  --variant-json examples/prompt-experiment-variants.json \
  --out expanded-request.json
```

The current CLI accepts the multi-rerank request shape. The simple request shape is converted through the library API.

## Notes

- Keep large prompt experiments and archived comparisons in `openpriors-research`; keep small, reproducible request expansion examples here.
