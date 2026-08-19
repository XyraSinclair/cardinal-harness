# Multi-objective rerank optimization recon (2026-08-06)

Read-only recon (Explore agent) over multi.rs, comparison.rs, trait_search.rs,
cache.rs, prompts.rs. Line references are as of commit 74da9cd.

## Findings

1. **Objectives are fully independent.** One RatingEngine per attribute
   (multi.rs:564-573); the planner emits a proposal per (attribute, pair)
   (trait_search.rs:585,696,774); dedup keys on (attribute_id, i, j)
   (trait_search.rs:824-873). M objectives over a band ⇒ ~M× calls; a pair is
   never judged for two attributes in one call.
2. **Single-attribute call shape.** The seam for multi-attribute fusion is
   three coordinated points: template render (prompts.rs:92,118-132), response
   parse (comparison.rs:229), packet/cache recording + per-attribute
   add_observation (cache.rs:22-143, multi.rs:1172).
3. **Cache key includes attribute AND presented order** (cache.rs:62-90):
   cross-objective repeats of a pair are misses by design; counterbalanced
   orders are two rows.
4. **Prompt structure:** system = instructions + ladder + attribute (stable
   within an objective), user = the two entity texts + question (variable).
   Entity texts are never prefix-shared across objectives because the
   attribute sits ahead of them.

## Optimization ladder (risk-ordered)

1. **prompt_cache_key on the per-attribute system prefix** — LANDED
   2026-08-06 (comparison.rs both paths, shared helper in rerank/mod.rs).
   Routing hint only; bytes unchanged. Realizes once the system prefix
   crosses the provider floor (~1024 tokens) — i.e. long rubrics/--elaborate.
2. **Canonical-order pairwise cache key with sign reflection on read** —
   GATED: semantic (not byte-identical) reuse; must stay off during
   position-bias measurement (counterbalance/randomized-order runs). Worth a
   scripted-pathology test (sign-broken-channel class) before any live use.
3. **Multi-attribute-per-call fusion for co-proposed pairs** — GATED:
   changes template_hash and packet identity, couples M judgments to one
   call's failure; per AGENTS.md prompt churn must be deliberate and
   benchmark-motivated. Needs: benchmark showing per-attribute agreement vs
   split calls, plus trace semantics preserved (one ComparisonTrace per
   (attribute, pair)).

Ledger rule: (2) and (3) do not land without their gates; this file is the
standing record of why.
