# AGENTS.md

This repo is **llmsort-lab: the cold archive** of the llmsort research
program. It holds evidence, not code.

- The engine + CLI: [llmsort](https://github.com/XyraSinclair/llmsort)
  (crates.io `llmsort`; extracted from this repo at `0e30e1c`).
- The living research code (the `cardinal` research CLI, `cardinald`,
  research modules, live batteries): llmsort's `experiments/` workspace
  crate (folded 2026-08-18, llmsort `c2080b9`).
- The in-tree crate was deleted 2026-08-19 after the fold; every version
  of it is in this repo's git history. Do not resurrect code here — code
  fixes belong in llmsort.
- The public sites (llmsorting.com, pairwiseratio.org) live in
  exopriors-core `sites/` (consolidated 2026-08-18).

## What lives here

| What | Where |
|---|---|
| Evidence packs (replayable, content-addressed) | `artifacts/live/` — 38+ dated packs |
| Structured judgements + `*-cache.sqlite` | inside each pack |
| Research threads | `notes/` — dated investigations, red teams, decisions |
| Program docs | `docs/` (FIRST_PRINCIPLES, MATH_FRONTIER, PRINCIPLES, …), `PROGRAM.md` |
| Campaign definitions | `campaigns/`, `batteries/`, `data/` |
| Python analysis | `scripts/`, `examples/*.py` |

New experiment *code* is written in llmsort `experiments/`; new
experiment *evidence* (packs, campaign records, analysis) is committed
here. `PROGRAM.md` remains the book of tricks — every method as a rung
with its pack — and is served at <https://llmsorting.com/PROGRAM.md>
(fetched from this repo's main at deploy time).

## Norms that survive the freeze

- Never publish claude.ai Artifacts from this repo (operator ban,
  2026-07-08). Shareable pages are committed HTML served locally.
- `docs/PRINCIPLES.md` is the anti-slop discipline and still governs any
  analysis committed here: refutability, denominators, errata-on-top.
- `notes/OPERATOR-QUEUE.md` caps operator decisions at five open items.
- Git: fast direct-to-main, small commits, stage intended paths only,
  never force-push. Python analysis scripts that invoked the old in-tree
  binary now use `cargo run --bin cardinal` from a llmsort checkout.
