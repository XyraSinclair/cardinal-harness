# JCB v1.1 — nuisance-perturbation axis, live (2026-07-05)


## Erratum (2026-07-05 test–retest)

nano polarity (+0.38 here) flipped to −0.40 on an identical fresh run:
read it as "polarity relation unstable/noise", not "ranks shallowness
like depth". See `../judge-bench-retest-2026-07-05/`.

Same six models and corpus as `judge-bench-2026-07-05` (v1), plus the new
**nuisance stability** axis: each of 6 core pairs re-judged under four
semantically-null edits — extra whitespace, `**markdown**`, `- bullet`
(both entities), and a prestige halo suffix ("— from a widely cited
essay") on one entity — drift measured in nats against the same pair's
unperturbed call. 138 comparisons/model; the v1 cache replayed the
unchanged blocks (total marginal spend ≈ $0.20). Spin block re-ran under
the apostrophe-free framing wording (the v1 wording reached models as
"I&apos;ve…"; see spin-sweep pack).

| # | Model | JUDGE | coherence | harmonic | nuisance (nats) | whitespace | markdown | bullet | halo |
|---|---|---|---|---|---|---|---|---|---|
| 1 | gemini-2.5-flash | 0.572 | 0.897 | 0.872 | 0.492 | **0.000** | 0.388 | **0.961** | 0.617 |
| 2 | claude-sonnet-4.6 | 0.547 | 0.966 | 0.965 | **0.092** | 0.024 | 0.153 | 0.154 | **0.037** |
| 3 | claude-haiku-4.5 | 0.502 | 0.946 | 0.944 | **0.080** | **0.000** | **0.000** | 0.087 | 0.232 |
| 4 | deepseek-v4-flash | 0.447 | 0.899 | 0.885 | 0.356 | 0.324 | 0.393 | 0.359 | 0.346 |
| 5 | gpt-5.4-mini | 0.397 | 0.805 | 0.777 | 0.581 | 0.521 | 0.432 | 0.491 | **0.878** |
| 6 | gpt-5.4-nano | 0.204 | 0.531 | 0.000 | 0.913 | 0.681 | 0.972 | 1.072 | 0.926 |

## Findings

1. **Anthropic models are close to formatting-blind.** haiku: 0.080 nats
   mean drift, with whitespace and markdown at exactly 0.000; sonnet:
   0.092. Every other lab's model moves 4–10× more under the same null
   edits. This axis was built to kill content-blind hash judges; on real
   models it measures how much of the judgement is about the *content*
   versus the *bytes*.
2. **The halo is real and measurable per model.** Appending "— from a
   widely cited essay" to one side moves gpt-5.4-mini by **0.878 nats
   (≈ 2.4× in ratio terms)** and nano by 0.93 — a pure prestige cue worth
   more than most genuine quality gaps in the corpus. sonnet: 0.037
   (immune); haiku: 0.232.
3. **gemini's one weakness is formatting.** Flat zero on whitespace but
   0.96 nats under a bullet prefix — bullet-formatted text reads as a
   different thing to gemini. Its signal still carries the #1 headline
   (0.572), but sonnet closes to 0.025 behind once nuisance counts.
4. **mini's spin survival moved 1/3 → 2/3 under the de-garbled framing**
   (v1's escaped apostrophes made the preamble weirder and, apparently,
   more destabilizing). Wording hygiene is measurement hygiene — this is
   why the framing bank must rotate procedurally in v2 rather than
   accrete accidents.
5. **Ranking is stable across v1 → v1.1 and across aggregates** (arithmetic
   vs harmonic): the composite is not an averaging artifact. nano remains
   0.000 harmonic (dead spin axis).

Reproduce: `cardinal bench --models ... --cache bench-cache.sqlite --out
reports.jsonl` (cache in this pack replays all 828 judgements). Raw
per-call receipts in `reports.jsonl`; console blocks in `run.stderr`.

