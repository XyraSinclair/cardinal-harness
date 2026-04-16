# Prompt Contract

`cardinal-harness` supports one prompt template: `canonical_v2`.

## Slug

Use `canonical_v2` when you want to pin a request to the canonical pairwise-ratio prompt.

Unknown slugs are rejected.

## Ratio ladder

```text
[1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.1, 2.5, 3.1, 3.9, 5.1, 6.8, 9.2, 12.7, 18.0, 26.0]
```

The ladder is approximately geometric in log-space, with extra density near 1.0 for near-ties.

## Output shape

Successful judgement:

```json
{"higher_ranked":"A","ratio":2.1,"confidence":0.74}
```

Refusal:

```json
{"refused":true}
```

## Semantics

- `higher_ranked`: which side has more of the attribute
- `ratio`: how much more, constrained to the canonical ladder range
- `confidence`: self-reported confidence in `[0, 1]`
- `refused`: explicit refusal channel for genuinely blocked cases

## Notes

- `canonical_v2` is the only supported prompt surface in this repo.
- Prompt experimentation and archived comparisons belong in `openpriors-research`.
