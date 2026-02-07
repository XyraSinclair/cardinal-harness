# Prompts

Prompt templates live in `src/prompts.rs` and are selected per-attribute via
`prompt_template_slug` in `MultiRerankAttributeSpec`.

## Available Template Slugs

- `canonical_v1`
- `canonical_v2` (default)
- `canonical_v2_attr_first` (attribute text appears before entity context)
- `canonical_v3`

## Entity Context Placement

Pairwise comparisons pass the *entity text* via `EntityRef.context` (not the label).

Two ways to place context in the rendered user message:

1. Implicit prefix injection (most templates)
   - If the template user text does **not** include `{entity_A_context_block}` /
     `{entity_B_context_block}`, then context blocks are inserted before the template body.
2. Inline placement (opt-in per template)
   - If the template includes `{entity_A_context_block}` and/or `{entity_B_context_block}`,
     those placeholders are replaced in-place and no extra prefix is injected.

This is intentional so you can experiment with prompt ordering without changing code.

## Why Prompt Ordering Matters

Ordering affects two things:

1. **Model quality**
   - Some models do better when the scoring attribute is stated before reading long entities.
   - Others do better when they read the entities first and see the attribute prompt immediately
     before answering.
2. **Provider read-cache efficiency**
   - Some providers discount cached input tokens for repeated prefixes.
   - If a large prefix is repeated across calls (same attribute prompt, same entity text in the
     same position, etc.), total cost and latency can drop substantially.

The library intentionally keeps prompt templates as simple string templates so you can benchmark
these tradeoffs per provider/model/attribute.

