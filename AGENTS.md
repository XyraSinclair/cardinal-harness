# AGENTS.md

`llmsort` is the engine: pairwise LLM ratio judgements fitted into cardinal
scores with uncertainty, active pair selection, and explicit cost and
provenance. The research program that produced it — evidence packs, dated
notes, benchmark sites, research verbs, the `cardinald` daemon — lives at
[llmsort-lab](https://github.com/XyraSinclair/llmsort-lab) with full
pre-extraction history. This repo stays small; new experiments go to the
lab, and instruments graduate here only after their evidence is in.

## Shape mandate

- ≤ 120 tracked files; no source file over 800 lines (grandfathered files
  may only shrink); ≤ 16 integration test suites; zero Python.
- The five rooms (see `src/lib.rs` docs): solve, evidence, elicit,
  gateway, run. Dependencies point one way; nothing in solve/ evidence
  knows about gateways or I/O.
- The stability-promised surface: `sort_texts`/`sort_documents`, CLI
  `sort` + `judge`, and the packet format. Everything else may move —
  do not promise it to external consumers.

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
not merge, stage only intended paths, never force-push main. CI is fmt +
clippy `-D warnings` + tests + doctests + docs; keep it green at every
commit. When changing public request/response shapes or CLI behavior,
update examples, tests, and docs in the same change.
