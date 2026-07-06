# JCB v1.2 — the spectral board (2026-07-06)

Same six models; the benchmark now measures three new quantities per
model (194 calls total, warm-cache marginal spend ≈ $0.25): **orbit
coherence** (mean G-invariant energy fraction over six full Z₂³
transforms), **interaction share** (energy in the |S| ≥ 2 characters —
bias invisible to every marginal probe), and **harmonic** (frustration on
a dedicated chordless-cycle block with harmonic_dim = 1 by design — the
first live measurement of cyclic inconsistency that NO triad audit can
see, since the main graph's harmonic dimension is 0, pinned).

| # | Model | JUDGE | orbit coherence | interaction share | harmonic |
|---|---|---|---|---|---|
| 1 | gemini-2.5-flash | 0.538 | 0.394 | 0.286 | 0.083 |
| 2 | claude-sonnet-4.6 | 0.522 | **0.561** | **0.160** | 0.023 |
| 3 | claude-haiku-4.5 | 0.461 | 0.213 | 0.197 | 0.009 |
| 4 | deepseek-v4-flash | 0.422 | 0.357 | 0.324 | **0.002** |
| 5 | gpt-5.4-mini | 0.365 | 0.247 | 0.333 | 0.215 |
| 6 | gpt-5.4-nano | 0.191 | 0.151 | **0.432** | 0.381 |

## Findings

1. **The harmonic axis discriminates hard and independently.**
   deepseek-v4-flash closes the chordless loop to 0.002 (essentially
   exact) and haiku to 0.009, while mini leaks 0.215 and nano 0.381 of
   the block's judgment energy into a cycle no triangle check could ever
   flag. This is not the same ordering as curl (deepseek's triangle curl
   is mid-pack): local and global consistency are different abilities,
   now separately measured.
2. **Interaction structure is large at the population level**: 16–43% of
   orbit energy sits in characters invisible to marginal probes. nano:
   43.2% — nearly half its judgment behavior is coupled bias that no
   existing one-axis instrument would even have a name for.
3. **Rank order is stable across v1 → v1.1 → v1.2** while the composite
   compresses (more, harsher axes): gemini > sonnet > haiku > deepseek >
   mini > nano, unchanged.

## Honest limitation of the coherence axis (v1.2)

Orbit coherence averages over pairs INCLUDING contested ones. On a
near-tie, the belief coefficient is ≈ 0 by honesty, so mean-square energy
is all noise/bias and coherence → 0 even for a perfect judge — the axis
currently conflates "biased" with "genuinely torn". That is why the
population values (0.15–0.56) sit far below the clear-pair demo values
(0.93–0.99, orbit-2026-07-05 pack). v2 (#49) should report coherence per
clear/contested stratum or weight by |belief|. The BETWEEN-model ordering
is still informative (all models face the same pairs), but the absolute
level understates every judge.

Receipts: reports.jsonl (all raw calls), replayable bench-cache.sqlite.
An accounting bug was caught during this run's first launch — comparisons
reported 138 while spending on 194 — fixed (b4403de) and the board rerun
from cache; the committed JSONL carries correct counts (194/model).
