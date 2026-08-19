# Elicitation batteries — more priors per judgment (ideation, 2026-08-11)

Operator direction (chat, 2026-08-11): push toward sophisticated batteries
that extract more evidence per LLM call, under the product framing
**LLM-sorter** — `sort_texts`/`sort_documents` is already the canonical
surface; these are competing elicitation backends behind it, free to try
alien hacks.

**Denominators (fixed up front):** instruments race on evidence bits per
dollar; sorts race on tau-per-dollar against ground truth, on the same
groundtruth-battery pattern as `decimal_ledger_groundtruth`. No battery is
believed without its denominator.

## Battery menu

1. **k-wise Plackett-Luce from rank-position logprobs.** Present k labeled
   entities, elicit the full ordering, read the logprob distribution at
   each rank position → k conditional choice PMFs from one k-token
   completion. O(k²) pair relations at O(k) prompt cost. Prior art:
   `instrument/kwise.rs` (502 ln) culled in the seriate fold 2026-08-11
   with zero users — history preserved in the tombstoned repo, but design
   fresh: it predates the credal/evidence-moments machinery. Kill
   criterion to measure first: judgment quality vs k (attention dilution,
   lost-in-the-middle); expect a small optimal k.

2. **Within-call scalars with per-call fixed effect.** Cross-call scalar
   miscalibration is why pairwise exists; scores over k entities *within
   one call* share a calibration. Discard each call's absolute scale
   (call-level fixed effect), keep within-call score gaps as ratio
   observations → scalar elicitation rehabilitated for listwise use.
   Plugs into the `evidence_moments` seam.

3. **Multi-attribute per entity-set call.** Amortizes the entity-reading
   token bulk across attributes. Tax: answers in one completion share a
   reading and sampling path — NOT independent evidence; honest variance
   must carry a within-call correlation term (same class as the ×√2
   correlated shrinkage the matrix gauntlet measured for fused redraws)
   or coverage lies. Sequential halo: attribute order is a nuisance
   variable to randomize.

4. **Adjacent-ratio chains.** Elicit ordering of k entities plus ladder
   ratios between adjacent ranks → ordering + k−1 ratio observations in
   one call. Cheap composite of (1) and the ratio ladder.

## Prompt-cache co-design

All three current templates (`canonical_v2`, `bucket_v1`, `ordinal_v1`)
put the attribute BEFORE the entities (verified in `src/prompts.rs`,
2026-08-11). Cache savings scale with shared-prefix length and entities
are the token bulk, so:

- **Entities-first, attribute-last** template: attribute swaps reuse the
  cached entity prefix — the enabling layout for battery (3).
- **Cache-aware pair scheduling:** pin a shared anchor entity in slot A so
  its document prefix caches across every comparison it appears in;
  compounds with hub/anchor comparison graphs the planner already favors.

Doctrine: reordered surface ships as a NEW template slug and re-runs the
presentation-invariance battery (position bias must be re-measured for
the new layout); no churn to `canonical_v2`. Per AGENTS.md, all of this
is research-grade surface — no new permanent public verbs by default.
