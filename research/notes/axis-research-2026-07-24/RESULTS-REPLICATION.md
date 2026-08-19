# Wave-2 replication results — 48 runs, $2.35, 0 refusals (2026-07-27)

Scored by `analyze_replication.py` against the rules frozen in
`REPLICATION_SPEC.md` before any run. Evidence: `wave2/replication/*.json`.
One engine bug found and fixed mid-run, before any statistic was computed:
the logprob gate requested logprobs from the mandatory-reasoning gpt-5 base
family, 400-ing every gpt-5-mini comparison (fix `c242460`, code brought
into agreement with the docs/LOGPROBS.md census).

**Erratum (2026-07-27, Fable audit):** two lines below quoted the frontier
scar-tissue decoy rank as 9; the replication runs place it 10th for both
frontiers (wave-2 originals were 10th opus-4.6 / 9th gpt-5.6-sol).
Corrected in place; verdicts unchanged.

## Program verdict (mechanical, per frozen rule 4)

**UPGRADED TO FINDING** — power gate PASS (median retest ρ +0.965 across
24 cells) and 3 of 6 admitted axes TIER-GENERAL (scar_tissue_density 2/3,
antimemetic_payload 2/3, hostile_paraphrase_invariance 2/3).

Registered predictions, scored: `scar_tissue_density` TIER-GENERAL —
**correct**. `eschatological_seriousness` TIER-GENERAL — **wrong** (0/3;
prediction registered, miss reported).

## The two refinements the numbers force

**1. The tier law is not a price law — haiku-4.5 breaks it.** The wave-2
framing "cheap models cannot see scar tissue" is falsified as stated:
claude-haiku-4.5 ($1/M, cheaper than gpt-5.4-mini on output) behaves
frontier-like on nearly every axis (gaps −0.01 to +0.13, never a
signature; scar-tissue decoy at rank 8 beside the frontiers' 10). The
divergence separates deepseek-v4-flash / gpt-5-mini / gpt-5.4-mini from
opus-4.6 / gpt-5.6-sol / haiku-4.5. The surviving claim is about a
capability class that does not coincide with price or parameter count —
haiku appears to inherit its lineage's prior on what genuine operational
detail looks like. This is arguably more interesting than the original
claim, and it is the corrected headline: **which models can price
territory-contact is a model property the probe measures, not a tier you
can read off a price sheet.**

**2. The small-tier ranking noise floor is real and localized.** Frontier
retest is essentially perfect (opus median +1.000, sol +0.958). mini54
retests at median +0.804 with a floor of **+0.343** on
hostile_paraphrase_invariance — the mini's ranking on that axis is barely
stable run-to-run, so wave-2 T2 gaps measured against a single mini run
carry real noise on exactly the cells where the mini is weakest. The
global power gate passes, but any future per-axis claim that leans on one
small-model run should quote this floor. (Repeat runs used `--no-cache
--seed 7`; original seeds were unrecorded, so these deltas include
schedule variance — the conservative direction.)

## Per-axis outcomes (admitted six)

| axis | verdict | note |
|---|---|---|
| scar_tissue_density | **TIER-GENERAL** (2/3) | decoy rank 3 for dsflash AND gpt5mini vs 10/10 frontiers; haiku holds at 8 |
| antimemetic_payload | **TIER-GENERAL** (2/3) | hot-take decoy at 7/5 for dsflash/gpt5mini vs 10 frontier; haiku 10 |
| hostile_paraphrase_invariance | **TIER-GENERAL** (2/3) | via T2-analog: gaps +0.46/+0.35; note the mini54 retest floor caveat above |
| eschatological_seriousness | MINI-SPECIFIC (0/3) | prediction miss; dsflash gap +0.26 but axis was T3-admitted and no small model moved the decoy ≥3 ranks |
| rosetta_load | MINI-SPECIFIC (0/3) | wave-2's largest gap (+0.42) does not transfer; all three new smalls order it near-frontier |
| live_wire_prose | MINI-SPECIFIC (0/3) | decoy discrimination holds at every tier tested here |

## Standing

Wave 3 remains gated on the arrest event (operator queue Q5a) — that gate
is untouched by this result. What changes: the battery/canonize stage for
the three TIER-GENERAL axes now has a defensible tier claim to carry, and
the three MINI-SPECIFIC axes keep full value as axes (T1 frontier
coherence was never in question) while losing the tier-divergence story.
The author-family confound (items authored by a frontier-family agent)
remains open and is the honest next kill to attempt before any public
claim stronger than "measured on these six models."

Cumulative program cost: $4.67 across 81 runs.
