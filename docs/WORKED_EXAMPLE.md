# Worked Example: Three Proposals, Two Attributes, One Gate

This example is deliberately small. The point is not to demonstrate a heroic benchmark; it is to show the moving parts of a `cardinal-harness` rerank run without pretending that a synthetic transcript is a real model result.

We will rank three proposal drafts for review priority. The utility is a weighted combination of two attributes:

- `clarity`: how clearly the proposal explains its claim and trade-offs
- `evidence`: how much concrete evidence, reproduction detail, or worked example material it contains

A proposal must also pass a gate on `evidence` before it can enter the top-k set. That gate is intentionally modest: it filters out items that are high-level but not yet reviewable.

## 1. Request shape

Save this as `worked-example-request.json` if you want to run it locally.

```json
{
  "entities": [
    {
      "id": "proposal_a",
      "text": "Argues that the retry layer should be simplified. Names the affected module, but gives no failing trace or concrete before/after behavior."
    },
    {
      "id": "proposal_b",
      "text": "Shows a customer-visible timeout, includes a reproduction command, explains why the current retry policy amplifies load, and proposes a bounded retry budget."
    },
    {
      "id": "proposal_c",
      "text": "Suggests replacing an ad hoc scoring script with a checked-in report. The motivation is clear, but the implementation risks and expected output are only sketched."
    }
  ],
  "attributes": [
    {
      "id": "clarity",
      "prompt": "clarity of explanation and trade-offs for an engineering reviewer",
      "prompt_template_slug": "canonical_v2",
      "weight": 0.55
    },
    {
      "id": "evidence",
      "prompt": "strength of concrete evidence, reproduction detail, and worked examples",
      "prompt_template_slug": "canonical_bucket_v1",
      "weight": 0.45
    }
  ],
  "topk": {
    "k": 1,
    "tolerated_error": 0.1,
    "band_size": 3,
    "stop_min_consecutive": 2
  },
  "gates": [
    {
      "attribute_id": "evidence",
      "unit": "percentile",
      "op": ">=",
      "threshold": 0.34
    }
  ],
  "comparison_budget": 10,
  "rater_id": "worked-example-doc",
  "comparison_concurrency": 2,
  "max_pair_repeats": 1,
  "randomize_presentation_order": true
}
```

A few details matter:

- `canonical_v2` asks the model for a direct ratio on the canonical ratio ladder.
- `canonical_bucket_v1` asks for a bucket index into the same ladder, which is useful when a run wants output-token logprob accounting for the ratio choice.
- The `evidence` gate uses `percentile`, not an absolute latent score. That makes the example less sensitive to the arbitrary location and scale of the latent solver.
- `rater_id` is part of the run provenance. Use a stable value when you want repeated runs to be attributable and auditable.
- `model` is omitted here so the CLI default or an explicit policy can choose the current model. Pin a model only when you want that model choice to become part of the receipt.
- `randomize_presentation_order` should normally stay true; the trace records whether a comparison was presented swapped.

## 2. What one comparison asks for

For a single `evidence` comparison between `proposal_b` and `proposal_c`, the prompt contract reduces to this question:

> Which side has more strength of concrete evidence, reproduction detail, and worked examples, and by what ratio on the fixed ladder?

The successful response shape is one of the prompt contracts in `docs/PROMPTS.md`.

Illustrative shape only, not a recorded model output:

```json
{"higher_ranked":"A","ratio_bucket":7,"confidence":0.74}
```

Bucket `7` means ratio `2.1` on the canonical ladder:

```text
[1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.1, 2.5, 3.1, 3.9, 5.1, 6.8, 9.2, 12.7, 18.0, 26.0]
```

For `canonical_v2`, the analogous illustrative shape would be:

```json
{"higher_ranked":"A","ratio":2.1,"confidence":0.74}
```

Do not read these snippets as evidence that `proposal_b` will win on a live provider. They show only the JSON shape the parser accepts.

## 3. Run the rerank

Validate the request first if you want schema and invariant checks without an API key, cache, or network call:

```bash
cargo run --bin cardinal -- validate \
  --request worked-example-request.json
```

The CLI command writes three useful artifacts: the machine-readable response, the per-comparison trace, and a markdown report.

```bash
export OPENROUTER_API_KEY=your_key_here

cargo run --bin cardinal -- rerank \
  --request worked-example-request.json \
  --out worked-example-response.json \
  --trace worked-example-trace.jsonl \
  --report worked-example-report.md
```

If you need reproducibility for debugging, add an RNG seed:

```bash
cargo run --bin cardinal -- rerank \
  --request worked-example-request.json \
  --out worked-example-response.json \
  --trace worked-example-trace.jsonl \
  --report worked-example-report.md \
  --rng-seed 20260629
```

The seed controls harness-side randomization, including presentation-order randomization. It does not make a remote LLM deterministic unless the provider and model settings are also deterministic.

## 4. Generate or regenerate the report

A report can be regenerated later from the saved request and response. This is useful when report rendering changes but the run itself should not be repeated.

```bash
cargo run --bin cardinal -- report \
  --request worked-example-request.json \
  --response worked-example-response.json \
  --out worked-example-report.md
```

For automation, emit the report data as JSON instead of markdown:

```bash
cargo run --bin cardinal -- report \
  --request worked-example-request.json \
  --response worked-example-response.json \
  --format json \
  --out worked-example-report.json
```

## 5. Read the result without overclaiming

The response has two important parts:

- `entities`: ranked outputs with `u_mean`, `u_std`, `p_flip`, feasibility, and per-attribute scores
- `meta`: run-level evidence such as comparisons used, cached comparisons, token counts, provider cost, and `stop_reason`

Selected fields, illustrative shape only:

```json
{
  "entities": [
    {
      "id": "proposal_b",
      "rank": 1,
      "feasible": true,
      "u_mean": 0.812,
      "u_std": 0.184,
      "p_flip": 0.041,
      "attribute_scores": {
        "clarity": {
          "latent_mean": 0.62,
          "latent_std": 0.21,
          "z_score": 0.88,
          "min_normalized": 1.0,
          "percentile": 1.0
        },
        "evidence": {
          "latent_mean": 0.77,
          "latent_std": 0.19,
          "z_score": 1.04,
          "min_normalized": 1.0,
          "percentile": 1.0
        }
      }
    }
  ],
  "meta": {
    "global_topk_error": 0.074,
    "tolerated_error": 0.1,
    "k": 1,
    "comparisons_attempted": 8,
    "comparisons_used": 8,
    "comparisons_refused": 0,
    "comparisons_cached": 0,
    "comparison_budget": 10,
    "rater_id_used": "worked-example-doc",
    "stop_reason": "tolerated_error_met"
  }
}
```

Those numbers are not a receipt. They are included only to make the fields legible, and the token/cost fields are intentionally omitted from the illustration. Treat your actual `worked-example-response.json`, trace, and report as the receipt.

### Stop reason

`stop_reason` tells you why the loop ended; it does not by itself prove the answer is correct. Response JSON, JSON reports, and markdown reports use the snake-case values below.

| JSON stop reason | How to read it |
|------------------|----------------|
| `tolerated_error_met` | The estimated top-k error is at or below `topk.tolerated_error`. This is the ordinary good stop. |
| `certified_stop` | The stricter separation check found the top-k boundary stable. This is also a good stop. |
| `budget_exhausted` | The run spent the comparison budget before meeting the requested tolerance. Read `global_topk_error` before trusting the frontier. |
| `latency_budget_exceeded` | The latency budget stopped the run. Treat it like an incomplete run unless the error is already acceptable. |
| `no_proposals` | The planner found no useful next comparison under the current constraints. Check gates, budgets, and whether all useful pairs are already known. |
| `no_new_pairs` | Candidate comparisons existed, but eligible pairs were already known or blocked. Cache state and `max_pair_repeats` are relevant. |
| `cancelled` | The run did not converge normally. Do not cite it as evidence. |

### Uncertainty

`global_topk_error` is the main run-level uncertainty number. For a top-1 run with `tolerated_error = 0.1`, a final `global_topk_error = 0.074` means the harness estimates less than a 10% chance that the selected top-1 boundary is wrong under its approximation.

Per entity:

- `u_mean` is the combined utility estimate after attribute weighting and gates.
- `u_std` is the uncertainty of that combined utility.
- `p_flip` is the estimated probability of crossing the k-boundary.
- `attribute_scores.*.latent_mean` and `latent_std` are solver-scale values; compare them within the same run, not across unrelated runs.
- `percentile` and `min_normalized` are often easier to inspect than raw latent values when explaining gates.

A narrow-looking rank with high `global_topk_error` is not settled. A low `global_topk_error` with an incoherent attribute prompt is still only a precise answer to a bad question.

## 6. Cache and reproducibility receipts

With SQLite cache enabled by the default CLI path, repeated comparisons can be served from `.cardinal_pairwise_cache.sqlite`. Cache hits matter for both cost and reproducibility:

- In the response, `meta.comparisons_cached` counts comparisons served from cache.
- In the trace, each row includes `cache_key_hash`, `cached`, token counts, provider cost, and the model/prompt/entity hashes used to identify the comparison.
- In the markdown report, the request hash binds the report back to the parsed request content used to build it.
- If `--rng-seed` was supplied, the report includes the seed.

Export the cache when you need an auditable bundle:

```bash
cargo run --bin cardinal -- cache-export --out worked-example-cache.jsonl
```

A minimal receipt bundle for a run is:

```text
worked-example-request.json
worked-example-response.json
worked-example-trace.jsonl
worked-example-report.md
worked-example-cache.jsonl   # optional, but useful when cached comparisons influenced the run
```

For a no-network replay, use cache-only mode and keep the same request, model, prompt template slug, entity text, and attribute prompts. A cache-only run should fail on a missing comparison rather than silently calling the provider.

```bash
cargo run --bin cardinal -- rerank \
  --request worked-example-request.json \
  --out replay-response.json \
  --trace replay-trace.jsonl \
  --report replay-report.md \
  --cache-only
```

Do not compare cache-only latency or provider-cost fields to a live run as if they measure model performance. Cache hits report zero provider tokens and zero provider cost for the cached comparison path because no provider call was made.

## 7. What would make this example untrustworthy

This harness is useful only when the attribute is meaningful and the receipts are preserved. Be skeptical if:

- the prompt asks for a vague virtue such as "best" without saying best for what;
- the winner changed after `budget_exhausted`, but the report is presented as settled;
- a gate uses raw `latent` thresholds copied from another run;
- cached and live runs are mixed without preserving the trace;
- provider pricing is treated as a proof of billing rather than a local estimate plus provider-reported usage where available;
- illustrative snippets like the ones above are quoted as model outputs.

The honest claim from a good run is narrow: for this request, under these prompts, with these model calls or cache hits, the fitted pairwise-ratio model selected this top-k set with this estimated uncertainty.
