# AGENTS.md

`llmsort` is the engine: pairwise LLM ratio judgements fitted into cardinal
scores with uncertainty, active pair selection, and explicit cost and
provenance. This repo is a two-part workspace:

- **the crate** (root package, published to crates.io) — small, promised,
  under the shape mandate below;
- **`experiments/`** (`llmsort-experiments`, never published) — the research
  side: experimental verbs (`cardinal`), the `cardinald` daemon, live
  batteries, instruments whose evidence is not yet in. An instrument
  graduates into the crate only after its evidence pack earns it.

Evidence packs, campaign definitions, python analysis, dated notes, and
structured judgements live in the cold archive,
[llmsort-lab](https://github.com/XyraSinclair/llmsort-lab), which also
carries full pre-extraction history. New experiment *code* is written here;
new experiment *evidence* is archived there.

## Shape mandate (the crate — root package only)

- ≤ 120 tracked files under `src/`, `tests/`, `examples/`, `docs/`; no source file over 800 lines;
  ≤ 16 integration test suites in `tests/`.
- The five rooms (see `src/lib.rs` docs): solve, evidence, elicit,
  gateway, run. Dependencies point one way; nothing in solve/ evidence
  knows about gateways or I/O.
- The stability-promised surface: `sort_texts`/`sort_documents`, CLI
  `sort` + `judge`, and the packet format. Everything else may move —
  do not promise it to external consumers.
- `#[doc(hidden)] pub` items are seams for `experiments/`, not public
  API — they may change without notice.

`experiments/` is exempt from the file and suite ceilings but not from
discipline: zero Python anywhere in this repo, CI green at every commit
(fmt + clippy `-D warnings` + tests + docs run workspace-wide), and the
crate must never depend on `experiments/` — the dependency points one way.

## Core invariants (the embarrass-us list)

1. Solver math: IRLS/Huber fusion and the evidence currency
   (E[log-ratio], honest variance). Property-tested against planted truth.
2. Error-bar honesty: calibration coverage pinned; drift toward
   overconfidence must fail loudly.
3. Identity stability: packets and judgement records are
   content-addressed; serialization is load-bearing (`serde_json`
   float_roundtrip). A content address must never drift across versions.
4. Cost truth: comparisons, tokens, dollars reported per run.
5. The 30-second experience: `llmsort sort ideas.txt --by "..."` works on
   a cold clone.

## Collaboration

Fast direct-to-main: commit small coherent changes, push promptly, rebase
not merge, stage only intended paths, never force-push main. Publishing to
crates.io ships only the root package (`cargo publish -p llmsort`; the
include-list excludes `experiments/` — verify with
`cargo package -p llmsort --list` when touching packaging). When changing
public request/response shapes or CLI behavior, update examples, tests,
and docs in the same change.
