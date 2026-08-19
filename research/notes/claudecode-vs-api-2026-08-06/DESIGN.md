# Pre-registered design: Claude Code rail vs Claude API rail (2026-08-06)

AMENDMENT (2026-08-06, before any judged call): the 16-item corpus estimated
$8.09 worst case, over the registered $8 abort line. Per the registered
remedy the corpus shrank to 15 (dropped the last-fetched item, 2608.06264v1);
re-estimate $7.59 worst case. All item counts below read 15.

Question: do subscription-billed Claude Code judgments reproduce API-billed
Claude judgments on the same corpus and attribute — closely enough to trust
the $0-marginal rail for provenanced judgment runs?

## Fixed before any judged call

- Corpus: `corpus.json` — 16 most recent cs.LG arXiv abstracts (title +
  abstract), fetched 2026-08-06, ids are arXiv ids.
- Attribute: "practical real-world applicability of the paper's contribution,
  as evidenced by the abstract"
- Engine: `cardinal sort`, template default (canonical_v2), default
  counterbalancing (both orders), seed 20260806, no cache reads
  (`--no-cache` on both runs so neither rail feeds the other), full-list
  certification (no --top-k), JSONL trace per run.
- Rail A (API): OpenRouter `anthropic/claude-sonnet-4.6`, vault openpriors
  key ($142.19 remaining pre-run).
- Rail B (subscription): `claude-code/claude-sonnet-4-6`. Smoke calls
  2026-08-06 (pre-registration, "Say OK" only): the `sonnet` alias serves
  `claude-sonnet-5`; explicit `claude-sonnet-4-6` pins and serves
  `claude-sonnet-4-6` (modelUsage evidence); the dated id
  `claude-sonnet-4-6-20250929` errors. Pinning 4.6 both rails makes this a
  same-model rail comparison. Isolated scratch config dir (the
  `~/.claude-judge` pure dir), no operator CLAUDE.md, hooks, or memory in
  context.
- Served-model provenance recorded from every response on both rails; a
  family mismatch between rails is a stated caveat, not silently absorbed.

## Metrics (all with denominators)

1. Board-level: Spearman and Kendall tau between the two final rankings
   (n=16 items).
2. Pair-level: agreement rate on directed pairs judged by both rails
   (denominator = shared judged pairs), split by whether |score gap| is
   above/below the run's median gap.
3. Cost: nanodollars (rail A) vs $0 marginal + wall time (rail B), tokens
   both rails.
4. Noise context: this is n=1 run per rail — an instrument demonstration,
   not a model property (PRINCIPLES.md §3). If the boards disagree
   (Spearman < 0.8), the follow-up is a same-rail retest to measure each
   rail's own test–retest floor before narrating rail difference.

## Abort lines

- Rail A spend abort: $8 hard stop (estimate must come in under this via
  `sort --estimate` before launch; if not, shrink corpus, re-register).
- Rail B quota: if the subscription rate-limit class fires mid-run, the run
  is recorded as truncated; no partial-board claims.

## Confounds stated up front

- Claude Code injects its own system scaffolding around --system-prompt;
  prompt bytes are NOT identical across rails even with identical templates.
  This run measures the rails as deployed, not the model in isolation.
- Sampling temperature defaults may differ between rails (CLI does not
  expose temperature); recorded as systematic, not statistical, variance.
