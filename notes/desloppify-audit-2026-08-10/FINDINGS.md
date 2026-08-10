# Slop audit 2026-08-10 — residual findings

Four independent audit lenses (src, docs truthfulness, repo shape,
tests+CLI surface) swept the repo on 2026-08-10. The verdict: correctness
hygiene is genuinely high (zero clippy warnings, zero non-test panics in
the daemon, real CI, real errata culture, no hype register). The slop was
surface incoherence, and the mechanical part is fixed (commits 6b9e9ba..
0f92a40: cardinald contract doc, CHANGELOG resurrection, README counts,
dead embed/batch excision, codex usage error parity, key-guard dedup,
[research] verb marking, doctrine 30-day tests recorded).

This file holds what was found and NOT fixed, so the findings survive as
handles instead of chat scrollback. Each is behavior-sensitive or needs a
design decision; none is a quick edit.

## Structural (needs a deliberate slice each)

1. **The rerank planning loop is duplicated and has drifted.**
   `src/rerank/evaluation.rs:~980-1107` vs `src/rerank/multi.rs:~717-832`:
   certified-stop → tolerated-error → budget → propose-batch → dedup is
   copy-pasted between simulator and live orchestrator, and multi
   oversamples proposals 3× while evaluation does not (comment at
   evaluation.rs:~1006). The simulator no longer simulates the
   orchestrator. Decide the correct behavior, then extract one loop.

2. **Two god-functions.** `cardinal.rs` `main()` is ~2,080 lines with no
   per-verb function boundary; `multi_rerank` (multi.rs) is ~975 lines
   doing six jobs, with near-identical feasible/infeasible entity-assembly
   loops inside it.

3. **84% of tracked files are two evidence packs** (operator decision,
   surfaced to Xyra): `method-comparison-2026-06-30-suite-v1` (2,247
   files) + `method-comparison-2026-06-30` (1,125) commit per-call
   request/response/usage/parsed quadruplets. Not orphaned, only 33MB —
   the cost is file-count legibility. Keep raw call dumps in git, or roll
   up and move raw calls to evidence storage.

## Real-debt

4. **Duplicated math kernels**: Pearson correlation hand-rolled 3×
   (multi.rs:~1567, ensemble.rs:~152, consortium.rs:~221); standard
   normal CDF 2× (rating_engine.rs:~1578, censored_likelihood.rs:~117);
   observation preprocessing pasted between `fuse_bulk` and
   `add_observations` (rating_engine.rs:~1950/~2039); `p_flip` logic
   re-inlined twice in `plan_edges_for_rater` instead of calling
   `pair_prob_and_flip`; `CompareTask` defined twice (evaluation.rs,
   multi.rs); gate precision/recall and rank-metric blocks duplicated
   within evaluation.rs (cardinal vs Likert paths).

5. **Subprocess-CLI adapter skeleton duplicated** between
   `gateway/codex.rs` and `gateway/claude_code.rs`: `map_messages`
   byte-identical, `classify_cli_error` + `SAFEGUARD_MARKERS` +`tail`
   near-identical (~120 lines). A `subprocess_cli` module parameterized
   by build_args + parse_stdout collapses it.

6. **`land_completed_run` collapses landed/preserved/failed into one
   bool** (landing.rs:~202/~275) with failures visible only via eprintln;
   with `CARDINALD_CLICKHOUSE_URL` unset the daemon preserves forever
   with no operator signal. Wants a tri-state and a startup line.
   Related: `nonnegative_cost` (landing.rs:~536) silently clamps negative
   costs to 0.

7. **`RerankMeta` vs `MultiRerankMeta`** (rerank/types.rs:~122/~490):
   near-duplicate ~25-field struct families with duplicated doc comments.

8. **Stringly-typed `unit`/`op` on public gate specs** (rerank/types.rs
   `MultiRerankGateSpec`, trait_search.rs `GateSpec`) re-parsed at runtime
   into existing `GateUnit`/`GateOp` enums.

9. **PRICING_MAP staleness** (pricing.rs): hand-maintained ~30-model
   snapshot behind `OPENROUTER_PRICING_AS_OF = "2026-06-29"`; drift
   degrades silently to `DEFAULT_CHAT_ESTIMATE` (mitigated by preferring
   live upstream cost). Refresh the snapshot or automate it.

## Test-suite shape (needs operator approval — test scope)

10. 27 of 39 integration files pin research-instrument shapes; 11 guard
    the canonical surface. `tests/live_artifact_pages.rs` asserts
    committed bytes contain committed bytes (delete-with-zero-loss);
    `ladder_curl.rs`, `ordinal_prompt_validation.rs`,
    `trait_search_gates.rs`, `trace_jsonl.rs` are low-signal.
    `judge_explain_cli.rs` is 1,646 lines guarding research verbs.
    No test changes made (operator gate on test work).

## Smaller

11. `comparison.rs` `raw_output`/`question_text` on `ComparisonUsage` are
    write-only; `PAIRWISE_MAX_OUTPUT_TOKENS_GPT5` aliases the default so
    the gpt-5 branch is a no-op; `same_component` returns `false` when
    `topology_dirty` (success-shaped default).
12. `less_v1`/`fraction_v1` unreachable from `sort --template` (rerank
    JSON only) — surface-coverage mismatch with PROMPTS.md.
13. site/index.html board data tops out 2026-07-18; newest packs are
    2026-08-07 (likely fine — 07-18 matches the newest bench pack — but
    glance before the next publish).
14. 42 raw `.stderr`/`.log` scrollback files tracked in packs; evidence
    doctrine covers caches and JSONL, arguably not scrollback.
15. TESTING.md counts 364 workspace tests; direct `#[test]` grep finds
    326 (gap plausibly doctests). The canonicality doctrine wants counts
    that carry a sync check.
