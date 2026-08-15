# Grok battery v1 — does a judge model reasonably grok an attribute? (2026-08-15)

Three judges, six attribute cells, two independent seeds per cell (cache
bypassed for measurement; every judgment trace-recorded). Runner + corpora +
raw traces: `~/Projects/scratch/grok-battery-2026-08-15/`.

## The metric doctrine

Attributes worth tagging by are generally transitive, so "the model groks the
attribute" is measured as:

- **flip%** — direction coherence: of counterbalanced pairs, how many changed
  verdict with presentation order (position bias);
- **cyclic%** — transitivity: share of solution energy in cyclic (A>B>C>A)
  inconsistency;
- **retest ρ** — stability: Spearman between two fully independent runs;
- **ρ_truth** — truth recovery, on cells with a known answer (mass, lifespan,
  top speed, historical antiquity).

Low flip% + low cyclic% + high retest ρ = grokked; ρ_truth confirms where
truth exists. Soft attributes (Manifund's "existential seriousness",
"technical depth") ride the first three alone.

## Scorecards

**gemma4-26b-a4b** (128-expert MoE, ~4B active, fp8 online quant, local :8023):

| cell | ρ_truth s1/s2 | retest ρ | flip% | cyclic% | wall (2 runs) |
|---|---|---|---|---|---|
| animals_mass | 0.985 / 0.976 | 0.985 | 0.0 | 9.9 | 21.0s |
| animals_lifespan | 0.965 / 0.979 | 0.991 | 10.9 | 3.0 | 7.7s |
| animals_speed | 0.888 / 0.888 | 0.974 | 31.7 | 4.9 | 7.1s |
| events_antiquity | 0.992 / 0.988 | 0.995 | 7.4 | 8.0 | 9.9s |
| manifund_serious | — | 0.926 | 24.2 | 10.5 | 29.2s |
| manifund_depth | — | 0.972 | 10.0 | 5.9 | 29.2s |

**qwen38-27b** (dense hybrid GDN, fp8, canonical_v2, non-thinking):

| cell | ρ_truth s1/s2 | retest ρ | flip% | cyclic% | wall |
|---|---|---|---|---|---|
| animals_mass | 0.812 / 0.812 | 1.000* | 3.1 | 6.0 | 27.2s |
| animals_lifespan | 0.929 / 0.929 | 1.000* | 6.7 | 3.3 | 23.5s |
| animals_speed | 0.753 / 0.732 | 0.985 | 21.0 | 3.2 | 24.4s |
| events_antiquity | 0.952 / 0.914 | 0.926 | 12.0 | 10.5 | 34.1s |
| manifund_serious | — | 0.954 | 21.4 | 10.9 | 101.4s |
| manifund_depth | — | 0.956 | 21.5 | 6.3 | 103.5s |

\* deterministic judge + coinciding pair plans at budget 64 make seed retest
trivially 1.0 on small corpora; an unseeded 08-14 run of the same cell hit
0.976 vs today's 0.812 — plan selection matters at small budgets. Open probe.

**claude-sonnet-5** (claude-code subscription rail, $0 marginal, ~1s/cmp):

| cell | ρ_truth s1/s2 | retest ρ | flip% | cyclic% | wall |
|---|---|---|---|---|---|
| animals_mass | 0.953 / 0.879 | 0.894 | 6.1 | 7.5 | 99.0s |
| events_antiquity | 0.988 / 0.982 | 0.982 | 13.8 | 7.2 | 116.9s |

## Verdict: gemma4-26b-a4b takes the judge slot

- Best or tied-best ρ_truth on every ground-truth cell — including beating
  claude-sonnet-5 on both shared cells.
- 3–4× faster wall than qwen38 (MoE 4B-active decode) and ~10× vs Sonnet.
- **Prefix caching works: 74.4% token hit rate during the battery**
  (481888/647798). qwen38's hybrid GDN arch auto-disables vLLM prefix caching
  (`enable_prefix_caching=False` forced, counters hard-zero across 1000+
  requests) — a structural strike for high-perturbation judging, where the
  shared system+attribute prefix should be nearly free.
- Weak spots to watch: animals_speed flip 31.7% (fuzzy truth: mixed
  locomotion modes), manifund_serious retest 0.926 with flip 24.2% (soft
  criterion; qwen was steadier there, 0.954).
- 2 refusals on events_antiquity (qwen: 0).

Lane state after the battery: **gemma4-26b-a4b is the serving judge** on
colo2 :8023 (`/data/models/launch_gemma4.sh`, same co-tenant budget as the
qwen recipe; qwen relaunch is one command if wanted). Qwen's weights stay on
disk.

## Production doctrine going forward

- Measurement runs bypass the pairwise cache (independence) but always
  `--trace`; production judging keeps the cache ON so every judgment is paid
  for once — the cache is content-addressed per (pair, attribute, model,
  template).
- The relentless-judging loop wants: cardinald + ClickHouse landing for
  durable judgment records, attribute battery grown corpus-first (attributes
  chosen for expected transitivity), and prefix-ordered prompts (shared
  system+attribute head) to ride the 74% cache hit rate.
